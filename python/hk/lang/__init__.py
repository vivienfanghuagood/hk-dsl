"""The DSL surface: what a kernel author writes.

`__all__` deliberately lists only names a kernel author writes -- no submodules.
`hk.lang.ops` and `hk.ops` are different things (the op *set* and the shipped
*kernels*), and star-exporting the submodules would let one shadow the other
depending on import order, which is the kind of bug that only shows up in
somebody else's file.
"""

from . import collective, control, group as _group_mod, host, kernel as _kernel_mod, ops
from .collective import cross_warp, fold_rows
from .group import Group, group
from .control import range  # noqa: A001 -- shadowing builtins.range is the point
from .host import cdiv
from .kernel import GL, Kernel, const, kernel
from .ops import *  # noqa: F401,F403  -- the op set is defined by ops.__all__
from .ops import __all__ as _op_names

__all__ = ["cdiv", "cross_warp", "fold_rows", "GL", "Group", "Kernel", "const",
           "group", "kernel", "range",
           *_op_names]
