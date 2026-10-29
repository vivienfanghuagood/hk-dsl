"""The DSL-surface migration's deprecation shims (docs/design/dsl-surface.md §5): each old spelling
warns and traces to exactly the IR the new spelling gives, so a kernel moved from one to the
other is the same kernel (V0)."""

from __future__ import annotations

import warnings

import pytest

import hk
from hk import fp8e4m3, fp32, i8

K = 256


def _trace(body, ann, name):
    body.__annotations__ = ann
    body.__name__ = name
    k = hk.kernel(body, arch="gfx950", warps=1, name="shim", grid=lambda p: (1, 1, 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ir = k.trace().dump()
    return ir, [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]


MX = {"a": hk.Tensor[fp8e4m3], "b": hk.Tensor[fp8e4m3], "sa": hk.Tensor[i8],
      "sb": hk.Tensor[i8], "c": hk.Tensor[fp32]}


def _mx(old: bool):
    def body(a, b, sa, sb, c):
        ta = a.at[0, 0].load(hk.RT(fp8e4m3, 16, K))
        tb = b.at[0, 0].load(hk.RT(fp8e4m3, 32, K))
        s_a = hk.load_mx_scales(sa, 16, K // 32, 0, 0)
        s_b = hk.load_mx_scales(sb, 32, K // 32, 0, 0)
        s_c = hk.load_mx_scales(sb, 32, K // 32, 32, 0)
        acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
        if old:
            hk.copy_mx_scales(s_b, s_c)
            hk.mma_ABt_mx(ta, tb, acc, s_a, s_b, b_fmt="fp4", sb_at=(0, 0), into=acc)
        else:
            s_c.copy(into=s_b)
            hk.mma("mma_ABt", ta.scaled(s_a), tb.scaled(s_b, fmt="fp4", at=(0, 0)), acc, into=acc)
        c.at[0, 0] = acc
    return _trace(body, MX, "mx_shim")


def test_mma_ABt_mx_and_copy_mx_scales_are_hk_mma_and_MxScales_copy():
    old, warned = _mx(True)
    new, none = _mx(False)
    assert old == new
    assert any("mma_ABt_mx" in m for m in warned) and any("copy_mx_scales" in m for m in warned)
    assert not none


def _fp4_lds(old: bool):
    def body(w, c):
        ws = hk.lds.alloc(hk.ST(fp8e4m3, 16, 1024), name="ws")
        hk.group(1).load(ws, w.at[0, 0])
        ty = hk.RT(fp8e4m3, 32, 128)
        t = hk.load_fp4_frags_lds(ty, ws, 0, 0) if old else hk.load_fp4_frags(ty, ws, 0, 0)
        c.at[0, 0] = t
    return _trace(body, {"w": hk.Tensor[fp8e4m3], "c": hk.Tensor[fp8e4m3]}, "fp4_shim")


def test_load_fp4_frags_lds_is_load_fp4_frags_from_lds():
    old, warned = _fp4_lds(True)
    new, none = _fp4_lds(False)
    assert old == new
    assert any("load_fp4_frags_lds" in m for m in warned) and not none


def test_a_scaled_mma_wants_both_operands_scaled_and_mma_ABt():
    def body(a, b, sa, sb, c, *, variant="mma_ABt", one=False):
        ta = a.at[0, 0].load(hk.RT(fp8e4m3, 16, K))
        tb = b.at[0, 0].load(hk.RT(fp8e4m3, 32, K))
        s_a = hk.load_mx_scales(sa, 16, K // 32, 0, 0)
        s_b = hk.load_mx_scales(sb, 32, K // 32, 0, 0)
        acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
        c.at[0, 0] = hk.mma(variant, ta.scaled(s_a), tb if one else tb.scaled(s_b), acc)
    with pytest.raises(TypeError, match="both operands"):
        _trace(lambda a, b, sa, sb, c: body(a, b, sa, sb, c, one=True), MX, "one_scaled")
    with pytest.raises(TypeError, match="is mma_ABt"):
        _trace(lambda a, b, sa, sb, c: body(a, b, sa, sb, c, variant="mma_AB"), MX, "ab")


# -- S3: rows by index, hk.RowIndex and g.stage_rows ---------------------------------------------

ROWS = {"x": hk.Tensor[fp32], "ids": hk.Tensor[hk.i32], "o": hk.Tensor[hk.bf16]}
T32 = hk.RT(fp32, 16, 64)


def _rows(old: bool):
    def body(x, ids, o):
        g = hk.group(1)
        r0 = hk.arch.block_idx.y * 64
        t = x.at[0, 0].load(T32)
        if old:
            g.gather_prep(ids, r0, 64, raw=True)
            o.scatter(t, ids, r0 + 16, col=0, sort_topk=3, staged=64)
            o.scatter(t, ids, r0, col=64, sort_topk=3)
            o.at[0, 0] = x.gather(T32, ids, r0, count=40, div=3)
        else:
            rows = g.stage_rows(hk.RowIndex(ids, r0, sort_topk=3), 64, keep_choice=True)
            o.scatter(t, rows.offset(16), col=0)
            o.scatter(t, hk.RowIndex(ids, r0, sort_topk=3), col=64)
            o.at[0, 0] = x.gather(T32, hk.RowIndex(ids, r0, count=40, div=3))
    return _trace(body, ROWS, "rows_shim")


def test_scatter_and_gather_keywords_are_a_row_index():
    old, warned = _rows(True)
    new, none = _rows(False)
    assert old == new
    for what in ("gather_prep", "sort_topk", "staged", "div"):
        assert any(what in m for m in warned), what
    assert not none


FP8 = {"a": hk.Tensor[fp8e4m3], "ids": hk.Tensor[hk.i32]}


def _gathered_copy(old: bool):
    def body(a, ids):
        g = hk.group(4)
        xs = hk.lds.alloc(hk.ST(fp8e4m3, 64, 128), count=2, name="Xs")
        if old:
            g.gather_prep(ids, 0, 64)
            g.load(xs[0], a.at[0, 0], wait=False, gather=(ids, 0), staged=64)
            g.load(xs[1], a.at[0, 128], wait=False, gather=(ids, 0))
        else:
            rows = g.stage_rows(hk.RowIndex(ids, 0), 64)
            g.load(xs[0], a.at[0, 0], wait=False, gather=rows)
            g.load(xs[1], a.at[0, 128], wait=False, gather=hk.RowIndex(ids, 0))
        g.wait_for(xs[0])
        g.wait_for(xs[1])
    return _trace(body, FP8, "copy_shim")


def test_a_gathered_copy_takes_a_row_index():
    old, warned = _gathered_copy(True)
    new, none = _gathered_copy(False)
    assert old == new and len(warned) >= 3 and not none


def test_a_staged_index_is_read_only_in_the_form_it_was_staged_in():
    def scatter_of_gather_rows(x, ids, o):
        rows = hk.group(1).stage_rows(hk.RowIndex(ids, 0, sort_topk=3), 64)
        o.scatter(x.at[0, 0].load(T32), rows, col=0)
    with pytest.raises(TypeError, match="keep_choice=True"):
        _trace(scatter_of_gather_rows, ROWS, "bad_scatter")

    def copy_of_raw_ids(a, ids):
        g = hk.group(4)
        xs = hk.lds.alloc(hk.ST(fp8e4m3, 64, 128), name="Xs")
        rows = g.stage_rows(hk.RowIndex(ids, 0), 64, keep_choice=True)
        g.load(xs, a.at[0, 0], wait=False, gather=rows)
    with pytest.raises(TypeError, match="staged for scatters"):
        _trace(copy_of_raw_ids, FP8, "bad_copy")

    def register_gather_of_staged(x, ids, o):
        rows = hk.group(1).stage_rows(hk.RowIndex(ids, 0), 64)
        o.at[0, 0] = x.gather(T32, rows)
    with pytest.raises(TypeError, match="staged index"):
        _trace(register_gather_of_staged, ROWS, "bad_gather")

    def staged_but_never_copied(x, ids, o):
        rows = hk.RowIndex(ids, 0, sort_topk=3).staged(64, keep_choice=True)
        o.scatter(x.at[0, 0].load(T32), rows, col=0)
    with pytest.raises(TypeError, match="no g.stage_rows"):
        _trace(staged_but_never_copied, ROWS, "never_copied")


def _row_limit(form: str):
    def body(a, ids):
        g = hk.group(4)
        xs = hk.lds.alloc(hk.ST(fp8e4m3, 64, 128), count=2, name="Xs")
        at = hk.arch.block_idx.x * 64
        if form == "tuple":
            g.load(xs[0], a.at[0, 0], wait=False, gather=(ids, at), rows=hk.max(ids.cols - at, 0))
        elif form == "rows":
            g.load(xs[0], a.at[0, 0], wait=False, gather=hk.RowIndex(ids, at),
                   rows=hk.max(ids.cols - at, 0))
        else:
            g.load(xs[0], a.at[0, 0], wait=False, gather=hk.RowIndex(ids, at, count=ids.cols))
        g.wait_for(xs[0])
    return _trace(body, FP8, "row_limit")[0]


def test_a_gathered_copy_takes_a_row_limit_from_rows_or_the_index_count():
    """#125's rows= on a gathered copy: as a keyword next to an hk.RowIndex, and from the index's
    own count (rows = max(count - at, 0)) -- the same IR as the deprecated tuple form."""
    assert _row_limit("tuple") == _row_limit("rows") == _row_limit("count")

# -- S5: register views, Tile.as_fragment and Tile.as_scales --------------------------------------

VIEWS = {"a": hk.Tensor[hk.i32], "o": hk.Tensor[fp32]}


def _views(old: bool):
    def body(a, o):
        w8 = a.load_elems(hk.iota(hk.RS(hk.i32, 1, 512, 4)), runs=True)
        w4 = a.load_elems(hk.iota(hk.RS(hk.i32, 1, 256, 4)), runs=True)
        sb = hk.iota(hk.RS(hk.i32, 1, 64, 1))
        f8 = hk.RT(fp8e4m3, 16, 128)
        if old:
            x, y, s = hk.fp8_operand(w8), hk.fp4_operand(w4), hk.mx_scales_loose(sb)
        else:
            x, y, s = w8.as_fragment(f8), w4.as_fragment(f8, fmt="fp4"), sb.as_scales()
        o.at[0, 0].store(hk.mma("mma_ABt", x.scaled(s), y.scaled(s, fmt="fp4"),
                                hk.zeros(hk.RT(fp32, 16, 16, "col"))))
    return _trace(body, VIEWS, "views_shim")


def test_fp8_fp4_operand_and_loose_scales_are_register_views():
    old, warned = _views(True)
    new, none = _views(False)
    assert old == new and not none
    for what in ("fp8_operand", "fp4_operand", "mx_scales_loose"):
        assert any(what in m for m in warned), what


def test_as_fragment_takes_the_block_scaled_operand_and_fp8_or_fp4():
    def wrong_fragment(a, o):
        a.load_elems(hk.iota(hk.RS(hk.i32, 1, 512, 4)), runs=True).as_fragment(
            hk.RT(fp8e4m3, 32, 128))
    with pytest.raises(TypeError, match="16x128"):
        _trace(wrong_fragment, VIEWS, "wrong_fragment")

    def wrong_fmt(a, o):
        a.load_elems(hk.iota(hk.RS(hk.i32, 1, 512, 4)), runs=True).as_fragment(
            hk.RT(fp8e4m3, 16, 128), fmt="fp6")
    with pytest.raises(TypeError, match="'fp8' or 'fp4'"):
        _trace(wrong_fmt, VIEWS, "wrong_fmt")
