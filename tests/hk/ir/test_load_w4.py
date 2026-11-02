"""hk.load_w4's forms, traced: the asymmetric one (s and zs) as it was, the symmetric one (one zero
point, no zs, s optionally in 16-row blocks) and the zero-code one (a tensor of 4-bit zero codes,
codes optionally row-major) -- the C++ each prints, the RDNA header the latter two bring, and what is
refused. The GPU checks are tests/hk/gpu/test_w4a16.py and tests/hk/gpu/test_moe_w4.py."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.ops import w4a16  # noqa: E402

TW = hk.RT(hk.fp16, 16, 32, "row")


def _src(body, arch):
    return hk.kernel(body, arch=arch, warps=1, name=body.__name__, grid=lambda p: (1, 1, 1)).source()


def test_the_asymmetric_form_prints_as_before():
    def asym(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zs: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp16]):
        o.at[0, 0].store(hk.load_w4(TW, qw, s, zs, 0, 0, group=32))
    src = _src(asym, "gfx1100")
    assert "kittens::load_w4<32>(w4, g.qw, g.s, g.zs, " in src and "hk/rdna/w4.cuh" not in src


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx950"])
def test_the_symmetric_form_reads_no_zs(arch):
    ty = hk.RT(hk.fp16, 16, 32, "row") if arch != "gfx950" else hk.RT(hk.fp16, 16, 32, "row", frag=(16, 32))

    def sym(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp16]):
        o.at[0, 0].store(hk.load_w4(ty, qw, s, None, 0, 0, group=32, zero=8, s_blocked=True))
    src = _src(sym, arch)
    assert "kittens::load_w4<32, 8, true>(w4, g.qw, g.s, " in src
    assert ('#include "hk/rdna/w4.cuh"' in src) == (arch != "gfx950")


def test_refusals():
    def neither(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, None, 0, 0, group=32)
    with pytest.raises(TypeError, match="exactly one"):
        _src(neither, "gfx1100")

    def both(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zs: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, zs, 0, 0, group=32, zero=8)
    with pytest.raises(TypeError, match="exactly one"):
        _src(both, "gfx1100")

    def big_zero(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, None, 0, 0, group=32, zero=16)
    with pytest.raises(ValueError, match="0..15"):
        _src(big_zero, "gfx1100")

    def blocked_asym(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zs: hk.Tensor[hk.fp16],
                     o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, zs, 0, 0, group=32, s_blocked=True)
    with pytest.raises(TypeError, match="symmetric"):
        _src(blocked_asym, "gfx1100")


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
@pytest.mark.parametrize("rows", [False, True])
def test_the_zero_code_form_reads_zp(arch, rows):
    def zcode(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zp: hk.Tensor[hk.i8], o: hk.Tensor[hk.fp16]):
        o.at[0, 0].store(hk.load_w4(TW, qw, s, None, 0, 0, group=32, zp=zp, rows=rows))
    src = _src(zcode, arch)
    assert f"kittens::load_w4_zp<32, {'true' if rows else 'false'}>(w4, g.qw, g.s, g.zp, " in src
    assert '#include "hk/rdna/w4.cuh"' in src


def test_zero_code_refusals():
    def with_zs(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zs: hk.Tensor[hk.fp16],
                zp: hk.Tensor[hk.i8], o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, zs, 0, 0, group=32, zp=zp)
    with pytest.raises(TypeError, match="exactly one"):
        _src(with_zs, "gfx1100")

    def rows_asym(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zs: hk.Tensor[hk.fp16],
                  o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, zs, 0, 0, group=32, rows=True)
    with pytest.raises(TypeError, match="zero-code"):
        _src(rows_asym, "gfx1100")

    def zp_fp16(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zp: hk.Tensor[hk.fp16],
                o: hk.Tensor[hk.fp16]):
        hk.load_w4(TW, qw, s, None, 0, 0, group=32, zp=zp)
    with pytest.raises(TypeError, match="int8"):
        _src(zp_fp16, "gfx1100")

    ty = hk.RT(hk.fp16, 16, 32, "row", frag=(16, 32))

    def on_cdna(qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], zp: hk.Tensor[hk.i8], o: hk.Tensor[hk.fp16]):
        hk.load_w4(ty, qw, s, None, 0, 0, group=32, zp=zp)
    with pytest.raises(TypeError, match="wave32"):
        _src(on_cdna, "gfx950")


def test_the_w4a16_kernel_names_and_signatures():
    asym = w4a16.kernel_for(1, 2, 16, 32, 1, "fp16")
    sym = w4a16.kernel_for(1, 2, 16, 32, 1, "fp16", True)
    assert asym.name == "w4a16_t1_s2_w16_g32_b1_fp16" and sym.name == asym.name + "_sym"
    assert [p.name for p in sym.trace().tensors()] == ["x", "qw", "s", "c"]
    assert [p.name for p in asym.trace().tensors()] == ["x", "qw", "s", "zs", "c"]
