"""hk.templates: complete, tuned kernels with hooks for the part that is yours.

A template owns the tiling, the layout, the per-architecture plan and the
schedule; the hook is a traced function (like an `@hk.func` body) for the
math that makes the kernel yours -- an elementwise function, a row
reduction's epilogue, a GEMM's epilogue, an attention score modifier.
docs/design/dx-goal.md says what they are for and how they are judged.

Each template lives in its own module and is imported on first use, so the
modules are independent of one another.
"""

from __future__ import annotations

import importlib

# name -> module that defines it; a module added here is all a new template needs.
_TEMPLATES = {
    "elementwise": "rowwise",
    "rowwise": "rowwise",
    "gemm": "gemm",
}


def __getattr__(name):
    mod = _TEMPLATES.get(name)
    if mod is None:
        raise AttributeError(f"module 'hk.templates' has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


def __dir__():
    return sorted(set(globals()) | set(_TEMPLATES))
