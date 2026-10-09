"""Tracing: does the Python body produce the IR we expect.

Pure Python -- no compiler, no GPU.
"""

from __future__ import annotations

import pytest

import hk
from hk import bf16, fp32


@hk.kernel(
    arch="gfx1100",
    grid=lambda p: (hk.cdiv(p.o.cols, p.COLS), hk.cdiv(p.o.rows, p.ROWS), 1),
)
def add2(a: hk.GL[bf16], b: hk.GL[bf16], o: hk.GL[bf16], *, ROWS=16, COLS=64):
    t = hk.rt(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.block_idx.y, hk.block_idx.x)
    hk.store(o, hk.load(a, idx, t) + hk.load(b, idx, t), idx)


def test_params_split_into_tensors_and_constexprs():
    ir = add2.trace()
    assert [p.name for p in ir.tensors()] == ["a", "b", "o"]
    assert {p.name: p.const_value for p in ir.const_params()} == {"ROWS": 16, "COLS": 64}


def test_ops_recorded_in_order():
    ir = add2.trace()
    assert [op.opcode for op in ir.ops("load_global")] == ["load_global"] * 2
    assert len(ir.ops("add")) == 1
    assert len(ir.ops("store_global")) == 1


def test_operator_sugar_is_the_same_op_as_the_function():
    """`a + b` has to be exactly hk.add, or the two surfaces diverge silently."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def explicit(a: hk.GL[bf16], o: hk.GL[bf16]):
        t = hk.rt(bf16, 16, 16)
        i = hk.tile_coord()
        x = hk.load(a, i, t)
        hk.store(o, hk.add(x, x), i)

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def sugared(a: hk.GL[bf16], o: hk.GL[bf16]):
        t = hk.rt(bf16, 16, 16)
        i = hk.tile_coord()
        x = hk.load(a, i, t)
        hk.store(o, x + x, i)

    assert [op.opcode for op in explicit.trace().body] == [
        op.opcode for op in sugared.trace().body
    ]


def test_constexpr_specialisation_produces_a_different_ir():
    wide = add2.trace(COLS=128)
    assert wide.consts["COLS"] == 128
    (load,) = wide.ops("load_global")[:1]
    assert load.results[0].type.cols == 128
    # and the default is untouched
    assert add2.trace().consts["COLS"] == 64


def test_trace_is_cached_per_constexpr_set():
    assert add2.trace() is add2.trace()
    assert add2.trace() is not add2.trace(COLS=128)


def test_integer_literals_are_emitted_once():
    """A coordinate built from literals must not emit one const op per use."""
    ir = add2.trace()
    assert len(ir.ops("const")) == 1


def test_value_names_are_unique_and_deterministic():
    ir = add2.trace()
    names = ir.value_names()
    ids = [id(r) for op in ir.body for r in op.results]
    assert len(set(names[i] for i in ids)) == len(ids)
    assert ir.dump() == add2.trace().dump()


def test_dump_distinguishes_the_two_loads():
    dump = add2.trace().dump()
    assert "%ld1 = load_global" in dump, dump


# -- rejections ---------------------------------------------------------------


def test_tile_extents_must_be_multiples_of_the_wmma_fragment():
    with pytest.raises(ValueError, match="multiple of 16"):
        hk.rt(bf16, 16, 24)


def test_unannotated_tensor_parameter_is_rejected():
    with pytest.raises(TypeError, match="hk.GL"):

        @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
        def bad(a, o: hk.GL[bf16]):
            pass


def test_constexpr_without_a_default_is_rejected():
    with pytest.raises(TypeError, match="needs a default"):

        @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
        def bad(o: hk.GL[bf16], *, D):
            pass


def test_grid_expression_naming_an_unknown_parameter_fails_at_trace_time():
    @hk.kernel(arch="gfx1100", grid=lambda p: (p.nope.rows, 1, 1))
    def bad(o: hk.GL[bf16]):
        i = hk.tile_coord()
        hk.store(o, hk.zeros(hk.rt(bf16, 16, 16)), i)

    with pytest.raises(AttributeError, match="not a parameter"):
        bad.trace()


def test_elementwise_ops_do_not_broadcast():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def bad(a: hk.GL[bf16], o: hk.GL[bf16]):
        i = hk.tile_coord()
        x = hk.load(a, i, hk.rt(bf16, 16, 16))
        y = hk.load(a, i, hk.rt(bf16, 32, 16))
        hk.store(o, hk.add(x, y), i)

    with pytest.raises(TypeError, match="do not broadcast"):
        bad.trace()


def test_tile_and_element_coordinates_are_different_types():
    assert hk.rt(bf16, 16, 16) != hk.rt(fp32, 16, 16)
    ir_unit = hk.ir.CoordType("tile")
    assert ir_unit != hk.ir.CoordType("element")
