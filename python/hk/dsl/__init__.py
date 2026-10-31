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
from . import maps as _maps
from . import arch, collective, comm, lds, math, placement  # noqa: F401
from .control import if_, range, range_constexpr, scope, uniform  # noqa: A004
from .group import Group, group
from .kernel import Constexpr, Kernel, TracedKernel, Tensor, func, kernel
from .placement import Digit, Placement
from .values import (  # noqa: F401
    Bool, Coord, Global, Granules, Int, MxScales, RowIndex, Scaled, SharedTile, SharedVec, Tile, Value,
    Vec, View, as_vec, broadcast_col, broadcast_row, coord, elem_coord, full,
    copy_mx_scales, fp4_operand, fp8_operand, iota, mfma_strip, mx_scales_loose, load_fp4_frags, lane_coords, load_fp4_frags_lds, map_cpp, mx_quant_store, load_mx_scales, load_w4, mma, mma_ABt_mx, neg_infty, rand, tile_coord, transpose, transpose_sep, zeros,
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
    """Of ints, of device Ints, elementwise of tiles and vectors (an i32 strip: C's signed min),
    or of hk.map elements."""
    if isinstance(a, _maps.Scalar) or isinstance(b, _maps.Scalar):
        return _maps.minimum_(a, b)
    if isinstance(b, Tile) and not isinstance(a, (Tile, Vec)):
        a, b = b, a
    if isinstance(a, (Tile, Vec)):
        return a.minimum(b)
    if _dyn(a, b):
        return _scalar_op("min", a, b)
    return _builtins.min(a, b)


def max(a, b):  # noqa: A001
    """Of ints, of device Ints, elementwise of tiles and vectors (an i32 strip: C's signed max),
    or of hk.map elements."""
    if isinstance(a, _maps.Scalar) or isinstance(b, _maps.Scalar):
        return _maps.maximum_(a, b)
    if isinstance(b, Tile) and not isinstance(a, (Tile, Vec)):
        a, b = b, a
    if isinstance(a, (Tile, Vec)):
        return a.maximum(b)
    if _dyn(a, b):
        return _scalar_op("max", a, b)
    return _builtins.max(a, b)


def map(fn, *args):  # noqa: A001
    """Element (r, c) of the result is `fn` of element (r, c) of each strip argument (an
    `hk.Int` or a Python number argument is passed as itself): one `hk.map` op, `fn` traced
    once. `fn` sees elements (`hk.dsl.maps.Scalar`) with Python's operators at C's semantics
    -- `//` and `%` truncate, `>>` is arithmetic --, comparisons making masks, `hk.where`,
    `hk.min` / `hk.max`, `.to(dtype)`, `.bitcast(dtype)` and the hk.math functions. The
    result's dtype is what `fn` returns: fp32, i32 or a mask (b1). Floating-point contraction
    is off: `a * b + c` is two roundings, `hk.math.fma(a, b, c)` one.

        idx = hk.map(lambda c, w: ((c >> 4) & 63) * 4 + w * 512, col, hk.arch.warp_id())

    The operators on i32 strips and masks build these one op at a time and the hk-map-fuse
    pass merges a chain; `hk.map` is for an expression to read as one, or to keep a float
    expression's roundings as written. On gfx950, gfx1100 and gfx1201; on RDNA a mask only
    feeds another map (its strips hold no bool)."""
    return _maps.map_(fn, *args)


def where(m, a, b):
    """Elementwise `a` where the mask `m` holds, else `b` (C's `m ? a : b`): `m` a comparison of
    strips (or of hk.map elements), `a` and `b` strips of one dtype or numbers -- a Python
    number takes the other arm's dtype. One hk.map."""
    if _builtins.any(isinstance(x, _maps.Scalar) for x in (m, a, b)):
        return _maps.where_(m, a, b)
    if not (isinstance(m, Tile) and m.ty.dtype.name == "b1"):
        raise TypeError(f"hk.where(m, a, b): m is a mask -- a comparison of strips -- not {m!r}")
    return _maps.lift(_maps.where_, m, a, b, name="where")


__all__ = [
    "kernel", "func", "Kernel", "TracedKernel", "Tensor", "Constexpr",
    "RT", "RS", "ST", "SV", "col_vec", "row_vec", "DType",
    "bf16", "fp16", "fp32", "i32", "i8", "fp8e4m3", "fp8e5m2",
    "arch", "lds", "math", "collective", "comm", "placement", "Placement", "Digit", "group",
    "Group",
    "range", "range_constexpr", "if_", "scope", "uniform",
    "cdiv", "min", "max", "map", "where", "zeros", "full", "neg_infty", "rand", "load_w4", "load_mx_scales", "fp4_operand", "fp8_operand", "mx_scales_loose", "load_fp4_frags", "load_fp4_frags_lds", "mx_quant_store", "copy_mx_scales", "mma_ABt_mx", "transpose",
    "transpose_sep", "broadcast_row", "broadcast_col", "as_vec", "mma", "map_cpp", "iota", "lane_coords", "mfma_strip",
    "coord", "tile_coord", "elem_coord",
    "Int", "Bool", "Tile", "Vec", "Global", "View", "SharedTile", "SharedVec",
    "Granules", "Coord", "MxScales", "Scaled", "RowIndex",
]
