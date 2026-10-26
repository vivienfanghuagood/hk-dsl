"""Register strips: the row-contiguous layout row kernels use (hk.RS,
`!hk.rt<T, RxC, strip<RUN>>`, kittens::rs). The type, its textual form, its
translation, and what it refuses.

No GPU, no hipcc.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.compiler import mlir as M  # noqa: E402
from hk.dsl.collective import cross_warp  # noqa: E402


def _kernel(body, warps=2):
    body.__name__ = body.__name__.lstrip("_")
    return hk.kernel(body, arch="gfx1100", warps=warps, name=body.__name__,
                     grid=lambda p: (p.x.rows, 1, 1))


S = hk.RS(hk.fp32, 1, 256)


def _rms(x: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    row, c0 = hk.arch.block_idx.x, hk.arch.warp_id() * 256
    g = w.at[0, c0].load(hk.row_vec(S))
    v = x.at[row, c0].load(S)
    sq = cross_warp((v * v).sum(axis=1), "add", 2)
    sq /= x.cols
    sq += 1e-6
    v *= sq.rsqrt()
    v *= g
    o.at[row, c0].store(v, bounded=True)


# ---------------------------------------------------------------- the type


def test_a_strip_is_a_whole_number_of_chunks():
    assert S.span == 256 and S.strip and repr(S) == "rt<fp32,1x256,strip<8>>"
    assert hk.RS(hk.fp32, 3, 128, 4).cpp() == "kittens::rs<float, 3, 128, 4>"
    with pytest.raises(ValueError, match="whole number of chunks"):
        hk.RS(hk.fp32, 1, 200)
    with pytest.raises(ValueError, match="power of two"):
        hk.RS(hk.fp32, 1, 256, 3)
    with pytest.raises(TypeError, match="no i8 strip"):
        hk.RS(hk.i8, 1, 256)
    with pytest.raises(ValueError, match="strip's"):
        hk.RT(hk.fp32, 16, 64, "row").__class__(hk.fp32, 16, 64, "row", run=8)


def test_the_dialect_spells_and_checks_a_strip():
    with M.context():
        assert str(M.type_("!hk.rt<f32, 2x512, strip<4>>")) == "!hk.rt<f32, 2x512, strip<4>>"
        assert str(M.type_("!hk.rv<f32, 1x256, strip<8>, col>")) == \
            "!hk.rv<f32, 1x256, strip<8>, col>"
        with pytest.raises(Exception, match="multiple of 32\\*run"):
            M.type_("!hk.rt<f32, 1x100, strip<8>>")
        with pytest.raises(Exception, match="power of two"):
            M.type_("!hk.rt<f32, 1x256, strip<5>>")


def test_a_strips_vectors_are_its_own():
    """A strip's col_vec is rs_cv<T, rows> whatever its columns; its row_vec
    is rs<T, 1, cols, run>, so the run is part of that one's type."""
    a, b = hk.RS(hk.fp32, 2, 256), hk.RS(hk.fp32, 2, 512)
    assert hk.col_vec(a) == hk.col_vec(b)
    assert hk.row_vec(a) != hk.row_vec(hk.RS(hk.fp32, 2, 256, 4))
    assert hk.col_vec(a) != hk.col_vec(hk.RT(hk.fp32, 16, 64))


# ---------------------------------------------------------------- translation


def test_the_ir_and_the_cpp_name_the_strip():
    k = _kernel(_rms)
    text = k.trace().module.operation.get_asm()
    assert "!hk.rt<f32, 1x256, strip<8>>" in text
    assert "!hk.rv<f32, 1x256, strip<8>, row>" in text
    src = k.source()
    assert "using rs_fp32_1x256_r8 = kittens::rs<float, 1, 256, 8>;" in src
    assert "using rv_fp32_1x256_r8_col = rs_fp32_1x256_r8::col_vec;" in src
    assert "kittens::row_sum(" in src and "kittens::mul_col(" in src
    assert "kittens::store_bounded(g.o, " in src


def test_every_rdna_kernel_reads_its_arguments_at_entry():
    """kittens::touch_kernargs: a late kernarg field is a cold miss nothing
    overlaps (0.8-1.0 us on a one-row norm)."""
    src = _kernel(_rms).source()
    body = src[src.index("_kernel(const globals g) {"):]
    assert body.splitlines()[1].strip() == "kittens::touch_kernargs<sizeof(globals)>();"


def test_a_fresh_cross_warp_combine_needs_one_barrier():
    """Its scratch has never been read: LDS is allocated linearly."""
    src = _kernel(_rms).source()
    assert src.count("__builtin_amdgcn_s_barrier();") == 1


def test_the_strip_round_trips_through_text():
    t = _kernel(_rms).trace()
    before = t.cpp(['"kittens.cuh"'])
    with M.context():
        again = M.ir.Module.parse(t.module.operation.get_asm(enable_debug_info=True))
    assert M.translate(again, ['"kittens.cuh"']) == before


# ---------------------------------------------------------------- refusals


def _refuses(body, match):
    with pytest.raises((TypeError, ValueError, hk.VerifyError), match=match):
        _kernel(body).trace()


def test_a_strip_indexes_in_elements():
    def _tile_coord(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o.at[0, 0].store(x.tile[0, 0].load(S))
    _refuses(_tile_coord, "indexes in elements")


def test_a_strips_bound_is_the_tensors():
    def _row_limit(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o.at[0, 0].store(x.at[0, 0].load(S, bounded=3))
    _refuses(_row_limit, "bounded=True")


def test_a_strip_is_never_an_mma_operand():
    def _mma(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(hk.RS(hk.bf16, 16, 256))
        acc = hk.zeros(hk.RT(hk.fp32, 16, 16, "col"))
        acc += a @ a.T
    _refuses(_mma, "strip")


def test_a_strip_does_not_go_through_lds_as_a_tile():
    def _lds(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 16, 256))
        s.store(x.at[0, 0].load(hk.RS(hk.bf16, 16, 256)))
    _refuses(_lds, "strip")


def test_only_a_strips_col_vec_goes_through_an_lds_vector():
    def _row_vec(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(S)
        sv = hk.lds.alloc_vec(hk.SV(hk.fp32, 256, 1))
        sv.store(hk.zeros(hk.row_vec(S)))
    _refuses(_row_vec, "row_vec")


def test_a_strip_fills_by_column_only():
    def _triu(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(hk.RS(hk.fp32, 256, 256))
        v.triu_(0)
    _refuses(_triu, "strip")


def test_a_strip_is_refused_where_no_tree_has_one():
    k = hk.kernel(_rms, arch="gfx1201", warps=2, name="rms_gfx12",
                  grid=lambda p: (p.x.rows, 1, 1))
    with pytest.raises(TypeError, match="include/rdna3 and hk's wave64 port"):
        k.trace()


# ---------------------------------------------------------------- wave64


S64 = hk.RS(hk.fp32, 1, 512)


def _rms64(x: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    row, c0 = hk.arch.block_idx.x, hk.arch.warp_id() * 512
    g = w.at[0, c0].load(hk.row_vec(S64))
    v = x.at[row, c0].load(S64)
    sq = cross_warp((v * v).sum(axis=1), "add", 2)
    sq /= x.cols
    sq += 1e-6
    v *= sq.rsqrt()
    v *= g
    o.at[row, c0].store(v, bounded=True)


def _k950(body, warps=2):
    body.__name__ = body.__name__.lstrip("_") + "_950"
    return hk.kernel(body, arch="gfx950", warps=warps, name=body.__name__,
                     grid=lambda p: (p.x.rows, 1, 1))


def test_a_wave64_strip_is_kittens_rs_from_hks_port():
    """The same type and translation as gfx11's; the C++ comes from
    include/hk/cdna.cuh, which a gfx950 kernel includes."""
    src = _k950(_rms64).source()
    assert "using rs_fp32_1x512_r8 = kittens::rs<float, 1, 512, 8>;" in src
    assert '"hk/cdna.cuh"' in src
    assert "kittens::row_sum(" in src and "kittens::store_bounded(g.o, " in src


def test_a_wave64_strips_chunk_is_64_lanes():
    """256 columns of runs of 8 are one gfx11 chunk and half a gfx950 one."""
    with pytest.raises(TypeError, match="chunk is 64 lanes x 8"):
        _k950(_rms).trace()


def _swap(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    S2 = hk.RS(hk.fp32, 1, 128, 2)
    v = x.at[hk.arch.block_idx.x, 0].load(S2)
    v += v.swap_halves()
    o.at[hk.arch.block_idx.x, 0].store(v)


def test_swap_halves_is_one_map_in_the_ir_and_the_cpp():
    k = _k950(_swap, warps=1)
    assert "hk.swap_halves" in k.trace().module.operation.get_asm()
    assert "kittens::swap_halves(" in k.source()


def test_swap_halves_takes_a_wave64_strip_of_one_or_an_even_number_of_chunks():
    with pytest.raises(TypeError, match="wave64 strip port"):
        _kernel(_swap, warps=1).trace()

    def _tile(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o.tile[0, 0].store(x.tile[0, 0].load(hk.RT(hk.fp32, 16, 64)).swap_halves())
    with pytest.raises(TypeError, match="takes a strip"):
        _k950(_tile, warps=1).trace()

    def _three(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(hk.RS(hk.fp32, 1, 1536))
        o.at[0, 0].store(v.swap_halves())
    with pytest.raises(TypeError, match="neither one nor an even number"):
        _k950(_three, warps=1).trace()


def _rot(x: hk.Tensor[hk.bf16], cs: hk.Tensor[hk.fp32], o: hk.Tensor[hk.bf16]):
    S8 = hk.RS(hk.fp32, 1, 512, 8)
    t = hk.arch.block_idx.x
    v = x.at[t, 0].load(S8)
    p = v.rotate_half(128)
    p *= cs.at[t, 64].load(S8, period=64)
    v *= cs.at[t, 0].load(S8, period=64)
    v += p
    o.at[t, 0].store(v)


def test_rotate_half_and_a_period_load_are_ops_and_templates():
    """A 128-column head at eight a lane: rotate_half's span in the op and
    the template, and the angles a period of D/2 repeated along the chunk."""
    k = _k950(_rot, warps=1)
    asm = k.trace().module.operation.get_asm()
    assert "hk.rotate_half" in asm and "span = 128" in asm and "period = 64" in asm
    src = k.source()
    assert "kittens::rotate_half<128>(" in src
    assert src.count("kittens::load_period<64>(") == 2


def test_rotate_half_and_period_are_refused_where_they_cannot_be():
    with pytest.raises(TypeError, match="wave64 strip port"):
        _kernel(_rot, warps=1).trace()

    def _wide_span(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(hk.RS(hk.fp32, 1, 512, 8))
        o.at[0, 0].store(v.rotate_half(384))
    with pytest.raises(TypeError, match="divides the strip's columns"):
        _k950(_wide_span, warps=1).trace()

    def _bounded(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o.at[0, 0].store(x.at[0, 0].load(hk.RS(hk.fp32, 1, 512, 8), bounded=True, period=64))
    with pytest.raises(TypeError, match="unbounded strip"):
        _k950(_bounded, warps=1).trace()


def test_rotary_takes_the_16_byte_form_for_128_and_256_wide_heads(monkeypatch):
    """hk.ops.rotary's plan on gfx950: one warp of two chunks inside 64 MB,
    a token a workgroup past it, two warps of eight-byte runs a 512-column
    group where the token does not split into 4, 8 or 16 warps; preloaded
    past 32 MB moved. A 64-wide head keeps the 4-byte kernel."""
    import hk.target as target
    from hk.ops import fused, norm

    monkeypatch.setattr(target, "current_arch", lambda: "gfx950")
    monkeypatch.setattr(norm, "_PRELOAD_HERE", [norm.STRIP_PRELOAD_PAST["gfx950"]])
    assert fused._rotary16_plan(2048, 64, 128, 2) == (1, 2, 8, True)
    assert fused._rotary16_plan(2048, 32, 128, 2) == (1, 2, 8, False)
    assert fused._rotary16_plan(8192, 64, 128, 2) == (8, 2, 8, True)
    assert fused._rotary16_plan(8192, 40, 128, 2) == (2, 1, 4, True)
    assert fused._rotary16_plan(7, 8, 256, 2) == (1, 2, 8, False)
    assert fused._rotary16_plan(5, 3, 64, 2) is None
    assert fused._rotary16_plan(4, 8, 128, 4) is None
    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    assert fused._rotary16_plan(2048, 64, 128, 2) is None


def test_the_16_byte_rope_reads_its_positions_through_the_scalar_cache(monkeypatch):
    import hk.target as target
    from hk.codegen import scaffold
    from hk.ops import fused

    monkeypatch.setattr(target, "_CURRENT", ["gfx950"])
    src = scaffold.bare(fused._rotary16_kernel("rot16_t", hk.bf16, 1, True).trace())
    assert "kittens::load_scalar_invariant(g.pos" in src
    assert "kittens::rotate_half<128>(" in src


# ---------------------------------------------------------------- a stated reduction order

S64 = hk.RS(hk.fp32, 1, 512, 8)


def _ordered(lanes="seq", xor=(32, 16, 8, 4, 2, 1)):
    def ordered_sum(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        v = x.at[row, 0].load(S64)
        s = (v * v).sum(axis=1, lanes=lanes, xor=xor)
        w = hk.zeros(S64)
        w += s                                  # the row's sum in every column
        o.at[row, 0].store(w)
    return hk.kernel(ordered_sum, arch="gfx950", warps=1, name="ordered_sum",
                     grid=lambda p: (p.x.rows, 1, 1))


def test_a_stated_order_emits_the_ordered_reduction():
    src = _ordered().source()
    assert "kittens::row_sum_ordered<1, 32, 16, 8, 4, 2, 1>(" in src
    assert "kittens::row_sum_ordered<0, 1, 2, 4, 8, 16, 32>(" in _ordered("tree", (1, 2, 4, 8, 16, 32)).source()


def test_an_unstated_order_is_hks_own():
    def plain(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        v = x.at[hk.arch.block_idx.x, 0].load(S64)
        w = hk.zeros(S64)
        w += v.sum(axis=1)
        o.at[hk.arch.block_idx.x, 0].store(w)
    src = hk.kernel(plain, arch="gfx950", warps=1, name="plain_sum", grid=lambda p: (1, 1, 1)).source()
    assert "kittens::row_sum(" in src and "_ordered" not in src


@pytest.mark.parametrize("xor, lanes", [((32, 16, 8, 4, 2), "seq"), ((1, 1, 2, 4, 8, 16), "seq"),
                                        ((32, 16, 8, 4, 2, 1), "left")])
def test_a_stated_order_names_every_offset_once(xor, lanes):
    with pytest.raises((ValueError, TypeError)):
        _ordered(lanes, xor).source()


def test_a_stated_order_is_for_a_strips_rows_on_gfx950():
    def cols(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        v = x.at[0, 0].load(S64)
        o.at[0, 0].store(v.sum(axis=0, lanes="seq"))
    with pytest.raises(TypeError, match="rows of a strip"):
        hk.kernel(cols, arch="gfx950", warps=1, name="cols", grid=lambda p: (1, 1, 1)).source()

    def wave32(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        v = x.at[0, 0].load(hk.RS(hk.fp32, 1, 256))
        o.at[0, 0] = v.sum(axis=1, lanes="seq", xor=(16, 8, 4, 2, 1, 32))
    with pytest.raises(TypeError, match="gfx950"):
        hk.kernel(wave32, arch="gfx1100", warps=1, name="w32", grid=lambda p: (1, 1, 1)).source()


def _fused(lanes="seq", times_ty=S64, acc=False):
    def fused_sum(x: hk.Tensor[hk.fp32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        v = x.at[row, 0].load(S64)
        u = y.at[row, 0].load(times_ty)
        w = hk.zeros(S64)
        if acc:
            w += 1.0
        w += v.sum(axis=1, lanes=lanes, xor=(32, 16, 8, 4, 2, 1), times=u)
        o.at[row, 0].store(w)
    return hk.kernel(fused_sum, arch="gfx950", warps=1, name="fused_sum",
                     grid=lambda p: (p.x.rows, 1, 1))


def test_times_emits_the_fused_sum_of_products():
    src = _fused().source()
    assert "kittens::row_sum_times_ordered<1, 32, 16, 8, 4, 2, 1>(" in src


def test_times_needs_a_seq_order_and_the_same_strip():
    with pytest.raises(ValueError, match="seq"):
        _fused(lanes="tree").source()
    with pytest.raises(TypeError, match="same type"):
        _fused(times_ty=hk.RS(hk.fp32, 1, 256, 4)).source()


def test_the_ordered_sum_keeps_a_products_rounding():
    # the summands are laundered so that HIP's contraction cannot fuse a product into the sum
    src = _ordered().source()
    assert "kittens::row_sum_ordered<1, 32, 16, 8, 4, 2, 1>(" in src


# ---------------------------------------------------------------- hk.map_cpp

def _mapped():
    def mapped(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32]):
        v = x.at[hk.arch.block_idx.x, 0].load(S64)
        e = hk.map_cpp("(__builtin_bit_cast(int, x0) >> 23) & 0xFF", v, dtype=hk.i32)
        o.at[hk.arch.block_idx.x, 0].store(e)
    return hk.kernel(mapped, arch="gfx950", warps=1, name="mapped", grid=lambda p: (p.x.rows, 1, 1))


def test_map_cpp_emits_the_expression_as_a_lambda():
    src = _mapped().source()
    assert "kittens::map_cpp(" in src
    assert "[&](const auto x0) { _Pragma(\"clang fp contract(off)\") return ((__builtin_bit_cast(int, x0) >> 23) & 0xFF); }" in src


@pytest.mark.parametrize("expr", ["x0; x0", "{x0}", "  "])
def test_map_cpp_is_one_expression(expr):
    def bad(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        v = x.at[0, 0].load(S64)
        o.at[0, 0].store(hk.map_cpp(expr, v))
    with pytest.raises(ValueError, match="one C\\+\\+ expression"):
        hk.kernel(bad, arch="gfx950", warps=1, name="bad", grid=lambda p: (1, 1, 1)).source()


def test_map_cpp_wants_matching_fp32_or_i32_strips_on_gfx950():
    def shapes(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(S64)
        b = x.at[0, 0].load(hk.RS(hk.fp32, 1, 1024, 8))
        o.at[0, 0].store(hk.map_cpp("x0 + x1", a, b))
    with pytest.raises(TypeError, match="shape and run"):
        hk.kernel(shapes, arch="gfx950", warps=1, name="shapes", grid=lambda p: (1, 1, 1)).source()

    def bf(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(hk.RS(hk.bf16, 1, 512, 8))
        o.at[0, 0].store(hk.map_cpp("x0", a, dtype=hk.fp32))
    with pytest.raises(TypeError, match="fp32 or i32"):
        hk.kernel(bf, arch="gfx950", warps=1, name="bf", grid=lambda p: (1, 1, 1)).source()

    def w32(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(hk.RS(hk.fp32, 1, 256))
        o.at[0, 0].store(hk.map_cpp("x0", a))
    with pytest.raises(TypeError, match="gfx950"):
        hk.kernel(w32, arch="gfx1100", warps=1, name="w32", grid=lambda p: (1, 1, 1)).source()


def test_map_cpp_wants_a_dtype_over_mixed_strips_and_takes_python_ints():
    def mixed(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(S64)
        e = hk.map_cpp("x0", a, dtype=hk.i32)
        o.at[0, 0].store(hk.map_cpp("x0 * x1 + x2", a, e, 3))
    with pytest.raises(TypeError, match="dtype="):
        hk.kernel(mixed, arch="gfx950", warps=1, name="mixed", grid=lambda p: (1, 1, 1)).source()

    def said(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        a = x.at[0, 0].load(S64)
        e = hk.map_cpp("x0", a, dtype=hk.i32)
        o.at[0, 0].store(hk.map_cpp("x0 * x1 + x2", a, e, 3, dtype=hk.fp32))
    assert "kittens::map_cpp(" in hk.kernel(said, arch="gfx950", warps=1, name="said",
                                            grid=lambda p: (1, 1, 1)).source()
