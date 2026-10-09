"""A DSL kernel as `torch.ops.hk.<name>`.

This is the boundary a framework actually calls. vLLM and SGLang do not import
pybind modules out of a cache directory; they call `torch.ops.*`, and they
expect what that implies -- the op runs on torch's stream, it is visible to
`torch.compile`, and it can be captured into a CUDA graph.

Plain `hipcc -shared -fPIC` plus `torch.ops.load_library`, **not**
`torch.utils.cpp_extension.load`. On ROCm the latter runs hipify over the
source first, and this source is already HIP: hipify rewrites identifiers that
were correct, and then the compiler's diagnostics point into a generated file
nobody has. The same lesson is written down in
`kernels/rdna3/attn/torch_ext/Makefile`; this module is that Makefile, in
Python, with the content hash cache and the spill gate already attached.

    import hk
    op = hk.torch_op(hk.ops.elementwise.KERNELS["add_bf16"])
    op(a, b, out)                 # == torch.ops.hk.add_bf16(a, b, out)

The op takes its outputs as arguments. `codegen/scaffold.torch_library` says
why; the short version is that the IR knows where a kernel writes but not what
shape the result should be, and a framework usually has the buffer already.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import compile as _compile
from .compile import CompileError

#: (namespace, op) -> the loaded callable. Loading the same .so twice is not an
#: error (torch keeps its own set of paths) but *registering* the same op twice
#: is, and a serving process that built the op once per request would hit it.
_LOADED: Dict[Tuple[str, str], Any] = {}


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        raise CompileError(
            "hk.torch_op needs torch: the generated source includes "
            "<torch/library.h> and links against libtorch. The rest of hk -- "
            "tracing, codegen, even compiling -- does not."
        ) from None
    return torch


@functools.lru_cache(maxsize=1)
def torch_flags() -> Tuple[str, ...]:
    """Include and link flags for libtorch.

    The ABI flag is load-bearing and is *asked*, not assumed: libtorch is built
    one way or the other, a mismatch links cleanly, and the first std::string
    to cross the boundary segfaults.
    """
    torch = _torch()
    lib = Path(torch.__file__).parent
    return (
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"-I{lib / 'include'}",
        f"-I{lib / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-L{lib / 'lib'}",
        f"-Wl,-rpath,{lib / 'lib'}",
        "-ltorch",
        "-ltorch_hip",
        "-lc10",
        "-lc10_hip",
        "-lamdhip64",
    )


def source(kernel, *, ns: str = "hk", name: str = "", **const_overrides) -> str:
    """The generated .hip, without compiling it. For reading and for tests."""
    from ..codegen import scaffold as sc

    ir = kernel.trace(**const_overrides)
    return sc.torch_library(ir, ns=ns, op_name=name or ir.name)


def build(kernel, *, ns: str = "hk", name: str = "", **const_overrides):
    """Compile the torch registration for `kernel`. Needs hipcc and torch, but
    no GPU -- and it still gates on spills, so a torch op cannot be the way a
    silently-wrong kernel reaches production."""
    ir = kernel.trace(**const_overrides)
    op = name or ir.name
    return _compile.build(
        source(kernel, ns=ns, name=op, **const_overrides),
        ir.arch,
        name=op,
        extra_flags=torch_flags(),
        max_vgprs=getattr(kernel, "max_vgprs", None),
        min_occupancy=getattr(kernel, "min_occupancy", None),
    )


def torch_op(kernel, *, ns: str = "hk", name: str = "", **const_overrides):
    """Build, load and return `torch.ops.<ns>.<name>`.

    Memoised on (ns, name): the second call in a process returns the same op
    rather than re-registering it, which torch refuses with "Tried to register
    an operator with the same name twice".
    """
    torch = _torch()
    ir = kernel.trace(**const_overrides)
    op = name or ir.name
    if (hit := _LOADED.get((ns, op))) is not None:
        return hit

    # Another .so in this process may already have registered it -- a wheel
    # that ships AOT-built ops, say. Take that one; building a second copy
    # would compile fine and then throw at load.
    existing = _existing(torch, ns, op)
    if existing is None:
        b = build(kernel, ns=ns, name=op, **const_overrides)
        torch.ops.load_library(str(b.so_path))
        existing = _existing(torch, ns, op)
        if existing is None:  # pragma: no cover -- a torch-side failure
            raise CompileError(
                f"{b.so_path} loaded but torch.ops.{ns}.{op} is still not "
                f"there. Check that the .so was built from the torch scaffold."
            )
    _LOADED[(ns, op)] = existing
    return existing


def _existing(torch, ns: str, op: str) -> Optional[Any]:
    try:
        return getattr(getattr(torch.ops, ns), op)
    except (AttributeError, RuntimeError):
        # RuntimeError is what torch raises for an unregistered name; it is a
        # lookup miss, not a failure.
        return None


def forget() -> None:
    """Drop the memo. For tests -- it cannot unregister an op, so a rebuild
    after this still has to produce the same schema."""
    _LOADED.clear()
