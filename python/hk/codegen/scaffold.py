"""What wraps the kernel: the module boundary.

Split from cpp.py because the kernel body is the same artifact in every
deployment and only its wrapper changes. A pybind module is what `hk.compile()`
loads; a TORCH_LIBRARY registration is what vLLM and SGLang want (Phase 6); an
AOT build wants neither, just the object file. Keeping this in one place means
those three share a single emitter and cannot drift.

The module name is a macro rather than a literal because the compiled artifact
is named after its content hash, and pybind requires PYBIND11_MODULE's name to
match the .so's basename exactly -- get that wrong and the import fails with
"dynamic module does not define module export function", which says nothing
about the real cause.
"""

from __future__ import annotations

from ..ir.nodes import KernelIR
from . import cpp

MODULE_NAME_MACRO = "HK_MODULE_NAME"


def _members(ir: KernelIR) -> str:
    return ", ".join(f"&globals::{p.name}" for p in ir.tensors())


def pybind_module(ir: KernelIR) -> str:
    """A self-contained pybind11 extension exporting one function, `ir.name`,
    that takes the tensor parameters positionally in declaration order."""
    src = cpp.emit(ir, includes=('"kittens.cuh"', '"pyutils/hk_bind.cuh"'))
    lines = [
        src,
        f"#ifndef {MODULE_NAME_MACRO}",
        f"#define {MODULE_NAME_MACRO} {ir.name}",
        "#endif",
        "",
        f"PYBIND11_MODULE({MODULE_NAME_MACRO}, m) {{",
        # bind_function rather than bind_kernel: the latter re-derives the
        # launch inline, which would duplicate the dynamic-LDS attribute logic
        # that `launch` already has and that the torch path also needs.
        #
        # `py::fast` rather than `py` -- pyutils.cuh's converter spends eleven
        # Python operations and three std::strings per tensor, which is free
        # behind a millisecond kernel and is not free behind a sixty-
        # microsecond one. See include/pyutils/hk_bind.cuh for what it drops
        # and what it keeps.
        f'    kittens::py::fast::bind_function<launch>(m, "{ir.name}", '
        f'{_members(ir)});',
        "}",
    ]
    return "\n".join(lines) + "\n"


def bare(ir: KernelIR) -> str:
    """Kernel and `launch`, no module boundary. For AOT objects and for reading
    the generated code when debugging a schedule."""
    return cpp.emit(ir)


SCAFFOLDS = {"pybind": pybind_module, "bare": bare}


def render(ir: KernelIR, scaffold: str = "pybind") -> str:
    try:
        fn = SCAFFOLDS[scaffold]
    except KeyError:
        raise ValueError(
            f"unknown scaffold {scaffold!r}; have {sorted(SCAFFOLDS)}"
        ) from None
    return fn(ir)
