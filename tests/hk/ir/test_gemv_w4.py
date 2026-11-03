"""hk.gemv_w4, traced: the C++ each form prints, the RDNA header it brings, what is refused, and
when hk.ops.w4a16 takes the small-batch form. The GPU checks are tests/hk/gpu/test_w4a16_gemv.py."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.ops import w4a16  # noqa: E402


def _src(body, arch="gfx1100"):
    return hk.kernel(body, arch=arch, warps=1, name=body.__name__, grid=lambda p: (1, 1, 1)).source()


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_the_symmetric_form(arch):
    def sym(x: hk.Tensor[hk.fp16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(hk.transpose(hk.gemv_w4(x, qw, s, None, 0, 0, 64, 4, group=32, tokens=3,
                                                 zero=8, s_blocked=True)))
    src = _src(sym, arch)
    assert "kittens::gemv_w4<32, 8, true, 3>(gemv, g.x, g.qw, g.s, " in src
    assert '#include "hk/rdna/w4.cuh"' in src


def test_the_asymmetric_form():
    def asym(x: hk.Tensor[hk.fp16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16],
             zs: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(hk.transpose(hk.gemv_w4(x, qw, s, zs, 0, 0, 64, 4, group=128, tokens=1)))
    assert "kittens::gemv_w4<128, 1>(gemv, g.x, g.qw, g.s, g.zs, " in _src(asym)


@pytest.mark.parametrize("kw,exc,match", [
    (dict(group=16, zero=8), ValueError, "multiple of 32"),
    (dict(group=32, tokens=9, zero=8), ValueError, "1..8"),
    (dict(group=32), TypeError, "exactly one"),
    (dict(group=32, zero=16), ValueError, "0..15"),
])
def test_refusals(kw, exc, match):
    def bad(x: hk.Tensor[hk.fp16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp32]):
        hk.gemv_w4(x, qw, s, None, 0, 0, 64, 1, **kw)
    with pytest.raises(exc, match=match):
        _src(bad)


def test_bf16_x_and_wave64_are_refused():
    def bf(x: hk.Tensor[hk.bf16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp32]):
        hk.gemv_w4(x, qw, s, None, 0, 0, 64, 1, group=32, zero=8)
    with pytest.raises(TypeError, match="fp16"):
        _src(bf)

    def cdna(x: hk.Tensor[hk.fp16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16], o: hk.Tensor[hk.fp32]):
        hk.gemv_w4(x, qw, s, None, 0, 0, 64, 1, group=32, zero=8)
    with pytest.raises(TypeError, match="wave32"):
        _src(cdna, "gfx950")


def test_the_small_batch_kernel_traces_both_forms():
    for sym, group in ((True, 32), (False, 128)):
        src = w4a16.w4a16_gemv_kernel(f"gemv_ir_{int(sym)}", tokens=4, warps=16, group=group,
                                      symmetric=sym).source()
        assert ("kittens::gemv_w4<32, 8, true, 4>(" if sym else "kittens::gemv_w4<128, 4>(") in src


@pytest.mark.parametrize("m,n,k,group,dtype,arch,want", [
    (1, 5120, 6144, 32, "fp16", "gfx1100", True),
    (4, 14336, 5120, 32, "fp16", "gfx1100", True),
    (5, 14336, 5120, 32, "fp16", "gfx1100", False),        # past GEMV_MAX_M: the WMMA form
    (1, 14336, 5120, 32, "bf16", "gfx1100", False),        # fp16 only
    (1, 14336, 5120, 16, "fp16", "gfx1100", False),        # group of 16
    (1, 4096, 4608, 128, "fp16", "gfx1100", False),        # K not a multiple of 64 * 16
    (1, 14336, 5120, 32, "fp16", "gfx1201", False),        # not measured there yet
])
def test_when_the_small_batch_form_takes_a_gemm(m, n, k, group, dtype, arch, want, monkeypatch):
    import hk.target as target

    monkeypatch.setattr(target, "current_arch", lambda: arch)
    assert w4a16.gemv_takes(m, n, k, group, dtype) is want
