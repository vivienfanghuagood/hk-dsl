"""The `uniform` refinement on a register vector, and the one op that needs it.

A folded kernel reduces a row that has been laid across a tile's sixteen rows,
so its answer is one number and the tensor it writes has one slot for it.
`hk.store_scalar` writes that slot. On any vector whose entries differ it would
write an arbitrary lane's value and be wrong on the other fifteen -- silently,
because every shape still agrees -- so it refuses anything the IR has not
proven uniform.

What is asserted here is that the proof is carried, that it is carried the
*right way* (it survives arithmetic, it is not invented by it), and that it
costs the generated C++ nothing: a uniform vector and a plain one are the same
type to hipcc, so they must share one alias and one declaration.

The negative cases are written as kernels that fail to trace rather than as
calls into a hand-built Builder, because tracing is the only way a kernel
author reaches these ops and the message they get is half of what is being
tested.
"""

from __future__ import annotations

import pytest

import hk
from hk import fp32
from hk.ir.nodes import RegVecType


@hk.kernel(arch="gfx1100", warps=4,
           grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1))
def folded_amax(x: hk.GL[fp32], s: hk.GL[fp32], *,
                ROWS=16, COLS=64, WARPS=4):
    t = hk.rt(fp32, ROWS, COLS)
    idx = hk.elem_coord(0, 0, hk.s_mul(hk.block_idx.y, ROWS),
                        hk.s_mul(hk.warp_id(), COLS))
    m = hk.row_max(hk.load(x, idx, t))
    m = hk.cross_warp(m, "max", WARPS)
    m = hk.fold_rows(m, "max", WARPS)
    hk.store_scalar(s, hk.mul(m, 0.5), hk.elem_coord(0, 0, 0, hk.block_idx.y))


def _kernel(body):
    """Trace `body` as a one-warp kernel over two fp32 globals."""
    body.__annotations__ = {"x": hk.GL[fp32], "s": hk.GL[fp32]}
    return hk.kernel(body, arch="gfx1100", grid=lambda p: (1, 1, 1))


# -- where the proof comes from ----------------------------------------------


def test_a_plain_reduction_is_not_uniform():
    """One entry per row, and under FOLD the rows are different chunks of the
    row. The default has to be the unproven one or the refinement proves
    nothing."""
    ir = folded_amax.trace()
    reds = ir.ops("row_max")
    # Two: the kernel's own, and the one inside fold_rows that reduces the
    # broadcast tile. The first is the one that must not claim uniformity.
    assert reds[0].result.type.uniform is False


def test_cross_warp_alone_does_not_make_it_uniform():
    """It combines the *warps'* partials. Each entry is still its own tile
    row."""
    loads = folded_amax.trace().ops("load_shared_vec")
    assert loads and all(v.result.type.uniform is False for v in loads)


def test_fold_rows_is_what_establishes_uniformity():
    (retype,) = folded_amax.trace().ops("retype_vec")
    assert retype.operands[0].type.uniform is False
    assert retype.result.type.uniform is True


# -- how it travels -----------------------------------------------------------


def test_scaling_a_uniform_vector_leaves_it_uniform():
    """quantize's scale is `amax / 127`, so it needs this exact step to keep
    the proof. A constant is the same in every entry."""
    (store,) = folded_amax.trace().ops("store_scalar")
    assert store.operands[1].type.uniform is True


def test_mixing_with_a_plain_vector_loses_it():
    """Elementwise ops meet the refinements. quant's `127 / amax` is written
    this way round and must not be mistaken for a proof."""
    seen = {}

    def body(x, s, *, ROWS=16, COLS=64):
        t = hk.rt(fp32, ROWS, COLS)
        plain = hk.zeros_vec(hk.col_vec(t))
        uni = hk.as_vec(plain, t, uniform=True)
        seen["scaled"] = hk.mul(uni, 2.0).type.uniform
        seen["unary"] = hk.rsqrt(uni).type.uniform
        seen["uni/plain"] = hk.div(uni, plain).type.uniform
        seen["plain/uni"] = hk.div(plain, uni).type.uniform

    _kernel(body).trace()
    assert seen == {"scaled": True, "unary": True,
                    "uni/plain": False, "plain/uni": False}


def test_out_may_drop_the_refinement_but_never_gain_it():
    """One direction loses a fact and is sound; the other launders an unproven
    one into a type store_scalar trusts."""
    def drop(x, s, *, ROWS=16, COLS=64):
        t = hk.rt(fp32, ROWS, COLS)
        plain = hk.zeros_vec(hk.col_vec(t))
        hk.mul(hk.as_vec(plain, t, uniform=True), 2.0, out=plain)

    _kernel(drop).trace()  # fine: forgets that the result was uniform

    def gain(x, s, *, ROWS=16, COLS=64):
        t = hk.rt(fp32, ROWS, COLS)
        plain = hk.zeros_vec(hk.col_vec(t))
        uni = hk.as_vec(plain, t, uniform=True)
        hk.mul(plain, 2.0, out=uni)

    with pytest.raises(TypeError, match="out is"):
        _kernel(gain).trace()


# -- what it refuses ----------------------------------------------------------


def test_store_scalar_refuses_a_vector_that_was_never_folded():
    def body(x, s, *, ROWS=16, COLS=64):
        t = hk.rt(fp32, ROWS, COLS)
        v = hk.zeros_vec(hk.col_vec(t))
        hk.store_scalar(s, v, hk.elem_coord(0, 0, 0, 0))

    with pytest.raises(TypeError) as e:
        _kernel(body).trace()
    msg = str(e.value)
    assert "uniform" in msg and "fold_rows" in msg
    # It has to say what to do instead: the shapes all agree, so the reader has
    # no other signal that the obvious call was the wrong one.
    assert "hk.store" in msg


def test_store_scalar_refuses_a_tile_unit_coordinate():
    def body(x, s, *, ROWS=16, COLS=64):
        t = hk.rt(fp32, ROWS, COLS)
        uni = hk.as_vec(hk.zeros_vec(hk.col_vec(t)), t, uniform=True)
        hk.store_scalar(s, uni, hk.tile_coord(0, 0, 0, 0))

    with pytest.raises(TypeError, match="element coordinate"):
        _kernel(body).trace()


# -- what it costs the generated code -----------------------------------------


def test_the_refinement_has_no_c_spelling():
    t = hk.rt(fp32, 16, 64)
    plain = RegVecType(t, "col")
    uni = RegVecType(t, "col", True)
    assert plain != uni
    assert plain.cpp() == uni.cpp()
    assert uni.plain == plain and plain.plain == plain


def test_the_two_share_one_alias():
    """`using rv_... = ...;` twice would be legal C++ and still wrong to emit:
    it would tell the reader there are two types where there is one. The 16x64
    column vector appears in this kernel both plain (the per-warp max) and
    uniform (the folded scale)."""
    src = folded_amax.source()
    decls = [ln.strip() for ln in src.splitlines()
             if ln.strip().startswith("using rv_")]
    assert len(decls) == len(set(decls)), decls
    assert sum("16x64_row_col" in d for d in decls) == 1, decls


def test_the_store_reaches_the_generated_source():
    ir = folded_amax.trace()
    # Not a whole-vector store, which is the bug the op exists to prevent.
    assert not ir.ops("store_global")
    assert folded_amax.source().count("kittens::store_scalar(") == 1
