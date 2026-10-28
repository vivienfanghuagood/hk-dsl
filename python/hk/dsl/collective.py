"""hk.collective: combining a reduction across the warps of a workgroup.

Every reduction on a tile is per-warp. A workgroup that splits a row across W
warps holds W partial answers, and nothing in the register file sees across a
wave; these two route them through LDS with the barriers in the right place.
"""

from __future__ import annotations

from typing import Optional

from ..types import RegTileType, RegVecType
from . import arch, lds
from . import builder as B
from .values import SharedVec, Vec, _vec_of, as_vec, broadcast_col, binary

#: combiner -> the reduction that implements it. Associative and commutative
#: only: the warps write concurrently and the read order is not fixed.
_REDUCE = {"add": "sum", "max": "max", "min": "min", "mul": "prod"}
COMBINERS = tuple(_REDUCE)


def cross_warp(vec: Vec, op: str, warps: int, *,
               scratch: Optional[SharedVec] = None, name: str = "xw") -> Vec:
    """One register vector per warp combined, the answer in every warp."""
    if op not in COMBINERS:
        raise ValueError(f"cross_warp combiner must be one of {COMBINERS}, got {op!r}")
    vec = _vec_of(vec)
    if not isinstance(vec, Vec):
        raise TypeError(f"cross_warp takes a register vector, got {vec!r}")
    if warps == 1:
        return vec
    # A scratch allocated here, outside any loop, has never been read: LDS is
    # allocated linearly (hk-lds-alloc), so no earlier combine shares it.
    fresh = scratch is None and _top_level()
    if scratch is None:
        scratch = lds.alloc_vec(lds.shared_vec(vec.ty, warps), name=name)
    elif scratch.ty.count < warps:
        raise ValueError(f"scratch holds {scratch.ty.count} vectors, need {warps}")
    # [1] the scratch may still be being read by a previous combine -- not a
    # fresh one; [2] the stores must land before any warp reads. No third:
    # after the loads every warp holds the answer in registers. Both drain: a
    # vector store is ordinary ds_writes that nothing retires before an
    # s_barrier, and a warp on the WGP's other CU reads past one before they
    # land.
    if not fresh:
        arch.barrier(drain=True)
    scratch.store(vec, arch.warp_id())
    arch.barrier(drain=True)
    acc = scratch.load(vec.ty, 0)
    for w in range(1, warps):
        # max / min: a reduction's combine (NaN propagates, -0 < +0 on gfx950), as within a warp
        binary(op, acc, scratch.load(vec.ty, w), into=acc, reduce=op in ("max", "min"))
    return acc


def _top_level() -> bool:
    """Is the op being traced in the kernel's entry block, outside every loop
    and branch?"""
    t = B.current()
    return t.ip.block == t.entry


def fold_rows(vec: Vec, op: str, warps: int, *,
              scratch: Optional[SharedVec] = None, name: str = "fold") -> Vec:
    """Reduce a column vector's own entries, the answer in all of them -- for
    a row folded onto a tile's 16 rows. The result is uniform."""
    if op not in COMBINERS:
        raise ValueError(f"fold_rows combiner must be one of {COMBINERS}, got {op!r}")
    if not isinstance(vec, Vec) or vec.ty.kind != "col":
        raise TypeError(f"fold_rows takes a column vector, got {vec!r}")
    src = vec.ty.tile
    n = vec.ty.length
    square = RegTileType(src.dtype, n, n, src.layout)
    if scratch is None:
        scratch = lds.alloc_vec(lds.shared_vec(vec.ty, warps), name=name)
    slot = arch.warp_id() if warps > 1 else 0
    scratch.store(vec, slot)
    arch.barrier(drain=True)
    across = scratch.load(RegVecType(square, "row"), slot)
    spread = broadcast_col(square, across)
    folded = getattr(spread, _REDUCE[op])(axis=1).value()
    return as_vec(folded, src, uniform=True)


__all__ = ["COMBINERS", "cross_warp", "fold_rows"]
