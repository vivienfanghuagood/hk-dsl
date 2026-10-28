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
