"""Your own kernel in a serving engine: register it, check it, route a function to it.

    import torch.nn.functional as F
    from hk.integration import register

    silu_mul = register.op(
        "silu_mul",                         # torch.ops.hk_user.silu_mul
        kernel=my_silu_mul_kernel,          # an @hk.kernel: inputs, then outputs
        reference=lambda x: F.silu(x[:, :x.shape[1] // 2]) * x[:, x.shape[1] // 2:],
        out=lambda x: x.new_empty(x.shape[0], x.shape[1] // 2),
        route=lambda x: x.shape[0] <= 8192, # calls that take the kernel
    )
    silu_mul.check(torch.randn(64, 2 * 4096, device="cuda", dtype=torch.bfloat16))
    register.patch("vllm.model_executor.layers.activation.SiluAndMul.forward_cuda",
                   lambda self, x: silu_mul(x.reshape(-1, x.shape[-1])).view(
                       *x.shape[:-1], x.shape[-1] // 2))

**The op.** `register.op` makes `kernel` an opaque torch custom op,
`torch.ops.hk_user.<name>`. `out` allocates the outputs from the inputs, and
on meta tensors it is the op's fake, which is what torch.compile traces. The
kernel is called with the inputs and then the outputs, so its tensor
parameters come in that order. `route` decides, per call, whether the kernel
runs or `reference` does. The decision happens inside the op, where shapes
are numbers rather than symbols torch.compile would guard on. It launches on
torch's stream, so a CUDA graph captures it, and the op counts its calls
(`op.stats`) the way hk's own patches do.

**The patch.** `register.patch(target, fn)` rebinds a dotted attribute,
`module.Class.method` or `module.function`, and remembers it for
`register.revert()`.

**In vLLM's processes.** vLLM runs the model in a process of its own, so the
module that registers and patches has to be imported there. The plugin does
that: `HK_PATCH_VLLM=1 HK_VLLM_USER=<modules or .py files, comma-separated>`.
`python -m hk.integration ab` runs the end-to-end A/B, stock against your
module, in one command.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import engine

NAMESPACE = "hk_user"

#: name -> the registered op.
OPS: Dict[str, "UserOp"] = {}
_PATCHED: List[Tuple[Any, str, Any]] = []
_TARGETS: List[str] = []


def _schema(name: str, n_inputs: int, n_outputs: int) -> str:
    args = ", ".join(f"Tensor x{i}" for i in range(n_inputs))
    ret = "Tensor" if n_outputs == 1 else "(" + ", ".join(["Tensor"] * n_outputs) + ")"
    return f"{name}({args}) -> {ret}"


def _as_tuple(v) -> tuple:
    return v if isinstance(v, tuple) else (v,)


class UserOp(engine.Op):
    """A kernel registered by `op`: call it like the reference."""

    def __init__(self, name: str, kernel, reference: Callable, out: Callable,
                 route: Optional[Callable], n_inputs: int, n_outputs: int):
        self.kernel, self.reference, self.out, self.route = kernel, reference, out, route
        self.n_outputs = n_outputs
        super().__init__(f"{NAMESPACE}::{name}", _schema(name, n_inputs, n_outputs),
                         self._fake, paths=("hk", "fallback"), graph="safe")
        self._launch = None

    def _fake(self, *inputs):
        return self.out(*inputs)

    def _impl(self, *inputs):
        if self.route is not None and not self.route(*inputs):
            self.count("fallback")
            return self.reference(*inputs)
        if self._launch is None:
            from ..runtime.torch_ext import torch_op  # noqa: PLC0415

            self._launch = torch_op(self.kernel)
        outs = _as_tuple(self.out(*inputs))
        self._launch(*inputs, *outs)
        self.count("hk")
        return outs[0] if self.n_outputs == 1 else outs

    def check(self, *inputs, rel: float = 2e-2) -> float:
        """Run the kernel and the reference on `inputs`; raise unless every
        output agrees to `rel` of the reference's magnitude. Returns the
        largest relative error."""
        import torch  # noqa: PLC0415

        got = _as_tuple(self._impl_forced(*inputs))
        want = _as_tuple(self.reference(*inputs))
        worst = 0.0
        for g, w in zip(got, want):
            scale = w.float().abs().max().item()
            err = (g.float() - w.float()).abs().max().item() / max(scale, 1e-30)
            if not err <= rel:
                raise AssertionError(f"{self.name}: kernel differs from the reference "
                                     f"by {err:.3g} (allowed {rel})")
            worst = max(worst, err)
        torch.cuda.synchronize()
        return worst

    def _impl_forced(self, *inputs):
        route, self.route = self.route, None
        try:
            return self._impl(*inputs)
        finally:
            self.route = route

    @property
    def stats_line(self) -> str:
        return self.summary()


def op(name: str, *, kernel, reference: Callable, out: Callable,
       route: Optional[Callable] = None, inputs: Optional[int] = None) -> UserOp:
    """Register `kernel` as `torch.ops.hk_user.<name>`; see the module
    docstring. `inputs` is how many of the kernel's tensor parameters are
    inputs, by default all but the outputs `out` returns, counted on the
    first call to `out` the registration makes on meta tensors."""
    if name in OPS:
        return OPS[name]
    n_tensors = len(kernel._tensors) if hasattr(kernel, "_tensors") else None
    n_out = None
    if inputs is None:
        if n_tensors is None:
            raise ValueError(f"{name}: say how many tensor parameters are inputs")
        n_out = _count_outputs(out, n_tensors)
        inputs = n_tensors - n_out
    if n_out is None:
        n_out = (n_tensors - inputs) if n_tensors is not None else 1
    o = UserOp(name, kernel, reference, out, route, inputs, n_out)
    o.define(o._impl)
    OPS[name] = o
    return o


def _count_outputs(out: Callable, n_tensors: int) -> int:
    """How many tensors `out` returns, found by calling it on meta tensors for
    each possible input count until one fits."""
    import torch  # noqa: PLC0415

    for n_in in range(n_tensors - 1, 0, -1):
        try:
            got = _as_tuple(out(*[torch.empty(64, 64, device="meta") for _ in range(n_in)]))
        except Exception:  # noqa: BLE001 -- the wrong arity, or shapes it rejects
            continue
        if n_in + len(got) == n_tensors:
            return len(got)
    raise ValueError("could not tell how many outputs `out` returns; pass inputs=")


def patch(target: str, fn) -> str:
    """Rebind the dotted attribute `target` to `fn`. Returns what it did."""
    parts = target.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:-1]:
            obj = getattr(obj, attr)
        _PATCHED.append((obj, parts[-1], getattr(obj, parts[-1])))
        setattr(obj, parts[-1], fn)
        _TARGETS.append(target)
        note = ""
        if target.startswith("vllm."):
            # vLLM's compile cache does not know what is patched into it.
            from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

            digest = hashlib.sha1(",".join(sorted(_TARGETS)).encode()).hexdigest()[:8]
            note = f" (compile cache {_own_compile_cache('hk-user-' + digest)})"
        return f"{target} -> {getattr(fn, '__name__', 'user function')}{note}"
    raise ImportError(f"no importable module in {target!r}")


def revert() -> int:
    """Undo every `patch`, newest first. Returns how many."""
    n = len(_PATCHED)
    while _PATCHED:
        obj, attr, old = _PATCHED.pop()
        setattr(obj, attr, old)
    _TARGETS.clear()
    return n


def load_user(spec: str) -> List[str]:
    """Import each comma-separated module or .py file in `spec` -- what the
    vLLM plugin does with HK_VLLM_USER. Returns their names."""
    names = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if item.endswith(".py") or os.sep in item:
            mod_name = "hk_user_" + os.path.splitext(os.path.basename(item))[0].replace("-", "_")
            s = importlib.util.spec_from_file_location(mod_name, item)
            if s is None or s.loader is None:
                raise ImportError(f"cannot load {item!r}")
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
            names.append(mod_name)
        else:
            importlib.import_module(item)
            names.append(item)
    return names


def summary() -> Sequence[str]:
    return [o.summary() for o in OPS.values()]


__all__ = ["NAMESPACE", "OPS", "UserOp", "load_user", "op", "patch", "revert", "summary"]
