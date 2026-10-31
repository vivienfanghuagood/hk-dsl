"""hk.map and the integer / mask / bit operations on strips, traced: the IR each builds, the C++
it prints, and what is refused. The GPU semantics are tests/hk/gpu/test_map.py."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402

I = hk.RS(hk.i32, 1, 512, 8)
F = hk.RS(hk.fp32, 1, 512, 8)


def _kernel(body, arch="gfx950"):
    return hk.kernel(body, arch=arch, warps=1, name=body.__name__, grid=lambda p: (1, 1, 1))


def _src(body, arch="gfx950"):
    return _kernel(body, arch).source()


def _ir(body, arch="gfx950"):
    return _kernel(body, arch).trace().dump()


def _maps(src):
    return [ln.strip() for ln in src.splitlines() if "kittens::map_cpp(" in ln]


# ---------------------------------------------------------------- hk.map(fn)


def test_a_traced_map_is_one_expression_with_contraction_off():
    def rope(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(F)
        b = x.at[1, 0].load(F)
        o.at[0, 0].store(hk.map(lambda u, v: u * v - v * 0.5, a, b))
    (m,) = _maps(_src(rope))
    assert ('[&](const auto x0, const auto x1) { _Pragma("clang fp contract(off)") '
            'return (((x0 * x1) - (x1 * (float)0.5))); }, ld, ld1);') in m


def test_a_value_used_twice_is_a_local_and_an_argument_passed_twice_is_one():
    def twice(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(F)
        o.at[0, 0].store(hk.map(lambda u, w: hk.math.fma(u + 1.0, u + 1.0, w), a, a))
    (m,) = _maps(_src(twice))
    assert "[&](const auto x0) {" in m and m.endswith("}, ld);")
    assert "__builtin_fmaf((x0 + (float)1.0), (x0 + (float)1.0), x0)" in m


def test_map_takes_ints_and_python_numbers():
    def args(x: hk.Tensor[hk.i32], n: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        a = x.at[0, 0].load(I)
        k = n.at[0].load_scalar()
        o.at[0, 0].store(hk.map(lambda u, s, c: (u >> c) * s, a, k, 3))
    (m,) = _maps(_src(args))
    assert "return (((x0 >> 3) * x1)); }, ld, scalar);" in m


def test_the_result_dtype_is_what_fn_returns():
    def kinds(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32]):
        a = x.at[0, 0].load(F)
        o.at[0, 0].store(hk.map(lambda u: (u.bitcast(hk.i32) >> 23) & 0xFF, a))
    ir = _ir(kinds)
    assert "-> !hk.rt<i32, 1x512, strip<8>>" in ir
    assert "arith.bitcast" in ir


def test_the_ops_print_as_c():
    def ops(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(I)
        v = y.at[0, 0].load(F)
        o.at[0, 0].store(hk.map(
            lambda i, f: hk.where((i // 3 % 2 != 0) | ~(f < 0.0),
                                  hk.math.ceil(hk.math.log2(hk.max(f, 0.25) / 448.0)),
                                  hk.math.rcp(hk.math.exp2(-f)) + (hk.math.popcount(i) - hk.math.ctz(i)).to(hk.fp32)),
            a, v))
    (m,) = _maps(_src(ops))
    for piece in ("((x0 / 3) % 2) != 0", "(!(x1 < (float)0.0))", "ceilf(__builtin_log2f((fmaxf(x1, (float)0.25) / (float)448.0)))",
                  "__builtin_amdgcn_rcpf(__builtin_exp2f((-x1)))", "(__builtin_popcount(x0) - __builtin_ctzg((unsigned)x0, 32))"):
        assert piece in m, piece


# ---------------------------------------------------------------- operators on strips


def test_operators_on_i32_strips_and_masks_build_maps():
    def ops(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        v = y.at[0, 0].load(F)
        k = hk.arch.warp_id()
        r = hk.where((c < 3) & (v >= 0.5) | ~(c == k), (c >> 4) % 7 + 64 - c // 2 * k, -1)
        o.at[0, 0].store(hk.max(r ^ 5, 0) << 1)
    ir = _ir(ops)
    for op in ("arith.cmpi slt", "arith.cmpf oge", "arith.cmpi eq", "arith.andi", "arith.ori",
               "arith.xori", "arith.shrsi", "arith.remsi", "arith.divsi", "arith.select",
               "arith.maxsi", "arith.shli"):
        assert op in ir, op
    assert "hk.map_cpp" not in ir and "body =" not in ir


def test_an_int_on_the_left_broadcasts_into_the_strip():
    def left(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        w = hk.arch.warp_id()
        o.at[0, 0].store(w * 64 + c)
        o.at[1, 0].store(hk.where(w < c, 1, 0))
    src = _src(left)
    ms = _maps(src)
    assert "return ((x0 + x1)); }" in ms[0]       # w * 64 is the scalar's, outside the map
    assert "return (((x0 > x1) ? 1 : 0)); }" in ms[1]


def test_a_vector_beside_an_i32_strip_still_broadcasts():
    # test_k2_ops' work queue: an i32 strip += its row sums (add_row), not a map
    def bcast(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(hk.RS(hk.i32, 1, 64, 1))
        tot = x.at[1, 0].load(hk.RS(hk.i32, 1, 64, 1))
        tot += c.sum(axis=1, lanes="seq", xor=(32, 16, 8, 4, 2, 1)).value()
        o.at[0, 0].store(tot)
    ir = _ir(bcast)
    assert "hk.add_row" in ir and "hk.map" not in ir


def test_fp32_strip_arithmetic_is_unchanged():
    def floats(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        v = y.at[0, 0].load(F)
        o.at[0, 0].store(hk.max(v * 2.0 + v, 1.0))
    ir = _ir(floats)
    assert "hk.map" not in ir
    assert "hk.mul" in ir and "hk.add" in ir and "hk.max" in ir


def test_refusals():
    def mixed(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(x.at[0, 0].load(I) + y.at[0, 0].load(F))
    with pytest.raises(TypeError, match="convert one with .to"):
        _src(mixed)

    def int_div(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        o.at[0, 0].store(x.at[0, 0].load(I) / 2)
    with pytest.raises(TypeError, match="divides with //"):
        _src(int_div)

    def float_floordiv(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(y.at[0, 0].load(F) // 2)
    with pytest.raises(TypeError, match="index or bit arithmetic"):
        _src(float_floordiv)

    def float_beside_int(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        o.at[0, 0].store(x.at[0, 0].load(I) * 0.5)
    with pytest.raises(TypeError, match="Python float 0.5 beside an i32"):
        _src(float_beside_int)

    def mask_as_bool(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        o.at[0, 0].store(c if c < 3 else -c)
    with pytest.raises(TypeError, match="hk.where"):
        _src(mask_as_bool)

    def if_in_fn(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        o.at[0, 0].store(hk.map(lambda u: u if u < 3 else -u, x.at[0, 0].load(I)))
    with pytest.raises(TypeError, match="hk.where"):
        _src(if_in_fn)

    def where_not_mask(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        o.at[0, 0].store(hk.where(c, c, 0))
    with pytest.raises(TypeError, match="mask"):
        _src(where_not_mask)

    def bf16_result(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(hk.map(lambda u: u.to(hk.bf16), y.at[0, 0].load(F)))
    with pytest.raises(TypeError, match="bf16 element"):
        _src(bf16_result)

    def tile_compare(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        t = y.at[0, 0].load(hk.RT(hk.fp32, 16, 64))
        o.at[0, 0].store(t < 0.0)
    with pytest.raises(TypeError, match="compares strips"):
        _src(tile_compare)


def test_equality_of_a_wmma_tile_is_identity():
    def same(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        t = y.at[0, 0].load(hk.RT(hk.fp32, 16, 64))
        u = y.at[0, 0].load(hk.RT(hk.fp32, 16, 64))
        assert t == t and t != u and t in [u, t]
        o.at[0, 0].store(t)
    _src(same)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_maps_lower_on_rdna_through_hk_rdna_map(arch):
    def ints(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(hk.RS(hk.i32, 1, 256))
        v = y.at[0, 0].load(hk.RS(hk.fp32, 1, 256))
        o.at[0, 0].store(hk.where((c < 3) & (v >= 0.5), ((c >> 4) & 63) * 4 + c, -1))
    src = _src(ints, arch)
    assert '#include "hk/rdna/map.cuh"' in src
    (m,) = _maps(src)
    assert "return ((((x0 < 3) & (x1 >= (float)0.5)) ? ((((x0 >> 4) & 63) * 4) + x0) : -1)); }" in m


def test_an_rdna_kernel_without_a_map_includes_nothing_new():
    def plain(y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(y.at[0, 0].load(hk.RS(hk.fp32, 1, 256)) * 2.0)
    assert "hk/rdna/map.cuh" not in _src(plain, "gfx1100")


def test_a_mask_cannot_leave_its_map_on_rdna():
    def leaves(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(hk.RS(hk.i32, 1, 512))
        m = c < 3                                   # read twice: it stays a strip
        o.at[0, 0].store(hk.where(m, c, 0))
        o.at[1, 0].store(hk.where(m, 0, c))
    with pytest.raises(Exception, match="mask strip leaves its hk.map on gfx1100"):
        _src(leaves, "gfx1100")
    _src(leaves, "gfx950")                          # gfx950's strips hold bool


def test_maps_need_strips():
    def none(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        o.at[0, 0].store(x.at[0, 0].load(hk.RS(hk.i32, 1, 512)) + 1)
    with pytest.raises(TypeError, match="strips"):
        _src(none, "gfx942")


# ---------------------------------------------------------------- hk-map-fuse


def test_a_chain_of_operators_prints_as_one_expression():
    def chain(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        v = y.at[0, 0].load(F)
        k = hk.arch.warp_id()
        o.at[0, 0].store(hk.where((c < 3) & (v >= 0.5), ((c >> 4) & 63) * k + c, -1))
    (m,) = _maps(_src(chain))
    assert ('[&](const auto x0, const auto x1, const auto x2) { _Pragma("clang fp contract(off)") '
            'return ((((x0 < 3) & (x1 >= (float)0.5)) ? ((((x0 >> 4) & 63) * x2) + x0) : -1)); }') in m


def test_a_value_another_op_reads_is_computed_once():
    def shared(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        idx = c * 2 + 1
        o.store_elems(c, idx)                 # a non-map user: idx stays a strip
        o.at[0, 0].store(idx - 3)
    ms = _maps(_src(shared))
    assert len(ms) == 2
    assert "return (((x0 * 2) + 1)); }, ld);" in ms[0]
    assert "return ((x0 - 3)); }, map);" in ms[1]


def test_a_mask_lives_inside_the_map_that_reads_it():
    def masked(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        o.at[0, 0].store(hk.where(c < 0, -c, c))
    src = _src(masked)
    assert len(_maps(src)) == 1 and "rs_b1_" not in src


def test_a_c_body_is_not_fused():
    def body(x: hk.Tensor[hk.i32], o: hk.Tensor[hk.i32]):
        c = x.at[0, 0].load(I)
        o.at[0, 0].store(hk.map_cpp("x0 + 1", c + 2) * 3)
    assert len(_maps(_src(body))) == 3
