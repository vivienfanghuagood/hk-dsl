"""The hk kernel language.

    import hk

    @hk.kernel(warps=4, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1))
    def rmsnorm(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *, EPS=1e-6):
        row = hk.arch.block_idx.y * 16
        sq = hk.zeros(hk.col_vec(hk.RT(hk.fp32, 16, 64)))
        for cb in range(hk.arch.warp_id(), hk.cdiv(x.cols, 64), 4):
            v = x.at[row, cb * 64].load(hk.RT(hk.fp32, 16, 64))
            sq += (v * v).sum(axis=1)
        ...

Types are values you build (`hk.RT`, `hk.ST`, `hk.col_vec`); device values
are objects with operators and methods (`Int`, `Tile`, `Vec`, a tensor's
`.at[...]` view); the machine and shared memory have namespaces
(`hk.arch`, `hk.lds`, `hk.math`, `hk.collective`, `hk.group`). A `for` over a
device range and an `if` on a device comparison compile to scf.for / scf.if.
"""

from __future__ import annotations

import builtins as _builtins

from ..types import (  # noqa: F401
    DType, bf16, fp16, fp32, fp8e4m3, fp8e5m2, i8, i32,
    RegTileType as _RT, RegVecType as _RV, SharedTileType as _ST,
    SharedVecType as _SV,
)
from . import host as _host
from . import arch, collective, lds, math  # noqa: F401
from .control import if_, range, range_constexpr, scope  # noqa: A004
from .group import Group, group
from .kernel import Constexpr, Kernel, TracedKernel, Tensor, func, kernel
from .values import (  # noqa: F401
    Bool, Coord, Global, Granules, Int, SharedTile, SharedVec, Tile, Value,
    Vec, View, as_vec, broadcast_col, broadcast_row, coord, elem_coord, full,
    copy_mx_scales, load_fp4_frags, load_fp4_frags_lds, load_mx_scales, load_w4, mma, mma_ABt_mx, neg_infty, rand, tile_coord, transpose, transpose_sep, zeros,
)
from .values import scalar_op as _scalar_op


def RT(dtype: DType, rows: int, cols: int, layout: str = "row", frag=None) -> _RT:
    """A register tile type: kittens::rt<T, rows, cols, layout>. `frag`,
    (rows, cols), names the matrix-core fragment where it is not 16x16; left
    out, the hk-fragments pass sets it on the operand tiles that need one."""
    return _RT(dtype, rows, cols, layout, tuple(frag) if frag else None)


def RS(dtype: DType, rows: int, cols: int, run: int = 8) -> _RT:
    """A register strip type: kittens::rs<T, rows, cols, run>.

    The layout for kernels that stream rows -- a norm, a softmax -- rather
    than feed WMMA: lane l owns, of every row, the runs of `run` contiguous
    columns starting at j*wave*run + l*run (wave = 32 on gfx1100, 64 on
    gfx950; `hk.arch.target().wave`), so a bf16 run of 8 is one 16-byte
    access and nothing is held twice. `cols` is a whole number of wave*run
    chunks: 256 or 512 columns at run 8, so 512 fits both. A strip loads and stores (element coordinates only, the element
    type converting on the way), maps, reduces along rows (a butterfly that
    leaves the answer in every lane) and columns, broadcasts its col_vec and
    row_vec, and fills by column. Its row_vec has the strip's own column
    layout: a weight vector loads with the same accesses as the rows it
    scales."""
    return _RT(dtype, rows, cols, "strip", run=run)


def ST(dtype: DType, rows: int, cols: int, count: int = 1) -> _ST:
    """An LDS tile type; count > 1 is a multi-buffer."""
    return _ST(dtype, rows, cols, count)


def SV(dtype: DType, length: int, count: int = 1) -> _SV:
    """An LDS vector type."""
    return _SV(dtype, length, count)


def col_vec(tile: _RT) -> _RV:
    """One value per *row* of `tile`: what `t.sum(axis=1)` gives."""
    return _RV(tile, "col")


def row_vec(tile: _RT) -> _RV:
    """One value per *column* of `tile`."""
    return _RV(tile, "row")


def _dyn(*xs) -> bool:
    return _builtins.any(isinstance(x, Value) for x in xs)


def cdiv(a, b):
    """Ceiling division: of ints at trace time, of device Ints in the kernel,
    of tensor extents in a grid lambda."""
    if isinstance(a, _host.HostExpr) or isinstance(b, _host.HostExpr):
        return _host.cdiv(a, b)
    if _dyn(a, b):
        return _scalar_op("cdiv", a, b)
    return -(-a // b)


def min(a, b):  # noqa: A001
    """Of ints, of device Ints, or elementwise of tiles and vectors."""
    if isinstance(a, (Tile, Vec)):
        return a.minimum(b)
    if _dyn(a, b):
        return _scalar_op("min", a, b)
    return _builtins.min(a, b)


def max(a, b):  # noqa: A001
    """Of ints, of device Ints, or elementwise of tiles and vectors."""
    if isinstance(a, (Tile, Vec)):
        return a.maximum(b)
    if _dyn(a, b):
        return _scalar_op("max", a, b)
    return _builtins.max(a, b)


__all__ = [
    "kernel", "func", "Kernel", "TracedKernel", "Tensor", "Constexpr",
    "RT", "RS", "ST", "SV", "col_vec", "row_vec", "DType",
    "bf16", "fp16", "fp32", "i32", "i8", "fp8e4m3", "fp8e5m2",
    "arch", "lds", "math", "collective", "group", "Group",
    "range", "range_constexpr", "if_", "scope",
    "cdiv", "min", "max", "zeros", "full", "neg_infty", "rand", "load_w4", "load_mx_scales", "load_fp4_frags", "load_fp4_frags_lds", "copy_mx_scales", "mma_ABt_mx", "transpose",
    "transpose_sep", "broadcast_row", "broadcast_col", "as_vec", "mma",
    "coord", "tile_coord", "elem_coord",
    "Int", "Bool", "Tile", "Vec", "Global", "View", "SharedTile", "SharedVec",
    "Granules", "Coord",
]
