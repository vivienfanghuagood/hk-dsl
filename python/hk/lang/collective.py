"""Combining a reduction across the warps of a workgroup.

Every reduction in `lang.ops` is per-warp: `row_sum` folds along the lane and
element axes of one wave and stops there. That is the right primitive -- it is
what the hardware does -- but it means a workgroup that splits one row of a
tensor across W warps ends up holding W partial answers, and nothing in the
register file can see across a wave.

Why split a row at all: a normalization has to see the whole row before it can
write any of it, so the row cannot be split across *workgroups* without a
second pass through global memory. With one warp per row block, a 4096-row
tensor at 16 rows per block is 256 waves, and this GPU has 192 SIMDs -- 1.3
waves per SIMD, nowhere near enough to cover a load's latency. Splitting the
row across 16 warps of one workgroup turns that into 4096 waves without adding
a pass. That is most of the gap to torch.compile on these kernels; the rest is
the number of passes, which `ops.norm` closes separately by holding the row in
registers.

The whole mechanism is three ops and a barrier, and the only thing that is
easy to get wrong is where the barriers go. Hence this file rather than an
idiom repeated at each call site.
"""

from __future__ import annotations

from typing import Optional

from ..ir.nodes import RegVecType, Value
from . import ops as _ops

#: combiner name -> the per-tile reduction that implements it. A combine here
#: has to be associative *and* commutative -- the warps write concurrently and
#: the read order is an implementation detail -- so `sub` and `div` are absent
#: on purpose rather than by omission.
_REDUCE = {"add": "row_sum", "max": "row_max", "min": "row_min", "mul": "row_prod"}
COMBINERS = tuple(_REDUCE)


def cross_warp(
    vec: Value,
    op: str,
    warps: int,
    *,
    scratch: Optional[Value] = None,
    name: str = "xw",
) -> Value:
    """Combine one register vector per warp, leaving the result in every warp.

    Every warp gets the full answer, not just warp 0: the caller's next step is
    to rescale its own slice of the row, and a broadcast back out would be a
    third barrier for nothing.

    `scratch` is an `alloc_shared_vec` of count >= warps, to reuse across
    several combines in one kernel. Left out, one is allocated here -- LDS for
    a vector is 64 bytes per warp, so allocating per call is not the thing to
    economize on, but a kernel that combines inside a loop must pass its own
    and knows why.
    """
    if op not in COMBINERS:
        raise ValueError(
            f"cross_warp combiner must be one of {COMBINERS}, got {op!r}. A "
            f"non-commutative combine has no defined answer here: the warps "
            f"write concurrently and nothing fixes the read order."
        )
    if not isinstance(vec.type, RegVecType):
        raise TypeError(f"cross_warp takes a register vector, got {vec.type}")
    if warps < 1:
        raise ValueError(f"warps must be positive, got {warps}")
    if warps == 1:
        # Not an error and not a special case worth a branch at the call site:
        # a one-warp workgroup has already combined everything it has.
        return vec

    vt = vec.type
    if scratch is None:
        scratch = _ops.alloc_shared_vec(_ops.shared_vec(vt, warps), name=name)
    elif scratch.type.count < warps:
        raise ValueError(
            f"scratch holds {scratch.type.count} vectors, need {warps}"
        )

    # Three barriers' worth of ordering in two, which is the minimum:
    #
    #   [1] before the store, because this scratch may still be being read by
    #       a previous combine -- the second call in softmax reuses the same
    #       LDS, and without this the sum-partials can land on top of
    #       max-partials another warp has not read yet.
    #   [2] after the store, the obvious one.
    #
    # There is no third: after the loads below, every warp holds the answer in
    # registers and nothing reads the scratch again until [1] of the next call.
    _ops.barrier()
    _ops.store_shared_vec(scratch, vec, _ops.warp_id())
    _ops.barrier()

    # Fully unrolled, and every warp reads every entry. The alternative -- a
    # tree -- would need a barrier per level; this is W broadcast LDS reads
    # with no barriers at all, and W is 16 at the most.
    acc = _ops.load_shared_vec(scratch, vt, 0)
    for w in range(1, warps):
        other = _ops.load_shared_vec(scratch, vt, w)
        getattr(_ops, op)(acc, other, out=acc)
    return acc


def fold_rows(
    vec: Value,
    op: str,
    warps: int,
    *,
    scratch: Optional[Value] = None,
    name: str = "fold",
) -> Value:
    """Reduce a column vector's own entries, leaving the answer in all of them.

    `row_sum` gives one value per tile row. Normally those are different rows
    of the tensor and must stay apart. But there is one shape where they are
    not: when the caller has reshaped an (R, C) tensor into (R*16, C/16), a
    tile's 16 rows are 16 consecutive chunks of a *single* row of the original,
    and the row's answer is the combination of all 16.

    Why anyone would do that: holding a row in registers is what turns a
    normalization from three passes over memory into one, and 16 rows of a
    16384-wide tensor do not fit in any register file. One row does. Folding
    the row onto the tile's rows is how a 16-row tile addresses a 1-row
    problem.

    The mechanism is a relayout through LDS -- write the vector indexed by
    row, read it back indexed by column -- and then a reduction of the tile
    that broadcasts it. Every lane ends up with the same value in all 16
    entries, which is exactly what the rescale pass wants: all 16 chunk-rows
    of the tile share the answer.

    Each warp uses its own slot and reads only what it wrote, so the barrier
    here orders nothing between warps; it is there because a store and a load
    to the same LDS address in one wave is the kind of dependence this library
    has already been bitten by once. It costs one instruction outside every
    loop.
    """
    if op not in COMBINERS:
        raise ValueError(f"fold_rows combiner must be one of {COMBINERS}, got {op!r}")
    if not isinstance(vec.type, RegVecType) or vec.type.kind != "col":
        raise TypeError(
            f"fold_rows takes a column vector (one entry per tile row), got "
            f"{vec.type}"
        )

    src = vec.type.tile
    n = vec.type.length
    # Square, so that the vector read back along the column axis has somewhere
    # to be: a tile whose column count is the vector's length.
    square = _ops.rt(src.dtype, n, n, src.layout)

    if scratch is None:
        scratch = _ops.alloc_shared_vec(_ops.shared_vec(vec.type, warps), name=name)
    slot = _ops.warp_id() if warps > 1 else 0

    _ops.store_shared_vec(scratch, vec, slot)
    _ops.barrier()
    # Same 16 floats, now indexed along the element axis instead of the lane
    # axis. sv is a flat array; the two register layouts differ and LDS is
    # where they meet.
    across = _ops.load_shared_vec(scratch, _ops.row_vec(square), slot)
    spread = _ops.broadcast_col(square, across)
    folded = getattr(_ops, _REDUCE[op])(spread)
    # uniform=True: `spread` has the same value in every column of every row,
    # so reducing it along the rows gives that value in every entry. This is
    # the one place in the library where that is true, and the one place that
    # says so. See lang.ops.store_scalar for what it buys.
    return _ops.as_vec(folded, src, uniform=True)


__all__ = ["COMBINERS", "cross_warp", "fold_rows"]
