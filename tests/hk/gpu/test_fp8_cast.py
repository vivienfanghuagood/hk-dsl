"""hk.ops.fp8_cast on gfx1201: the fp8 cast with transpose and amax, and Megatron's SwiGLU backward fused
with it, against torch (fp32 math, one bf16 rounding where Megatron's kernel writes bf16, then
torch's fp8 conversion of value * scale)."""

from __future__ import annotations

import pytest

from hk.target import current_arch, get_target

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if "fp8e4m3" not in get_target(current_arch()).mma or get_target(current_arch()).wave != 32:
    pytest.skip("an fp8 WMMA target (gfx1201)", allow_module_level=True)

from hk.ops import fp8_cast as F  # noqa: E402

FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _codes(t):
    return t.contiguous().view(torch.uint8)


def _close(got, want, frac=1e-5):
    """fp8 codes equal but for a few one-code differences (fp32 operation order)."""
    g, w = _codes(got).int(), _codes(want).int()
    diff = (g - w).abs()
    assert diff.max().item() <= 1
    assert (diff > 0).float().mean().item() <= frac


@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
@pytest.mark.parametrize("m,n", [(256, 128), (512, 2048)])
def test_cast_transpose(dtype, m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 3
    scale = torch.full((1,), 12.5, device="cuda")
    amax = torch.full((1,), 0.25, device="cuda")      # folded with a max, never overwritten
    sinv = torch.empty(1, device="cuda")
    rw, cw = F.cast_transpose(x, scale, amax, dtype=dtype, scale_inv=sinv)
    want = (x.float() * 12.5).to(FP8[dtype])
    assert rw.stride(0) == n + 64 and cw.shape == (n, m)
    assert torch.equal(_codes(rw), _codes(want))
    assert torch.equal(_codes(cw), _codes(want.t()))
    assert amax.item() == max(0.25, x.float().abs().max().item())
    assert sinv.item() == pytest.approx(1 / 12.5)


def _swiglu_back(g, y):
    y1, y2 = y.float().chunk(2, -1)
    s = torch.sigmoid(y1)
    g = g.float()
    return torch.cat((g * s * (1 + y1 * (1 - s)) * y2, g * y1 * s), -1).to(torch.bfloat16)


@pytest.mark.parametrize("pad_col", [0, 64])
@pytest.mark.parametrize("m,f", [(256, 256), (512, 1024)])
def test_swiglu_bwd_cast_transpose(m, f, pad_col):
    g = torch.randn(m, f, device="cuda", dtype=torch.bfloat16) * 0.01
    y = torch.randn(m, 2 * f, device="cuda", dtype=torch.bfloat16) * 2
    scale = torch.full((1,), 2000.0, device="cuda")
    amax = torch.zeros(1, device="cuda")
    rw, cw = F.swiglu_bwd_cast_transpose(g, y, scale, amax, pad_col=pad_col)
    dy = _swiglu_back(g, y)
    want = (dy.float() * 2000.0).to(torch.float8_e5m2)
    assert rw.stride(0) == 2 * f + 64 and cw.stride(0) == m + pad_col
    _close(rw, want)
    _close(cw, want.t())
    assert amax.item() == pytest.approx(dy.float().abs().max().item(), rel=1e-2)


def test_swiglu_bwd_into_callers_buffers():
    """TE's contract: uint8 views of padded buffers passed in, a 0-d scale_inv written."""
    m, f = 256, 512
    g = torch.randn(m, f, device="cuda", dtype=torch.bfloat16) * 0.01
    y = torch.randn(m, 2 * f, device="cuda", dtype=torch.bfloat16) * 2
    scale = torch.full((1,), 2000.0, device="cuda")
    amax = torch.zeros(1, device="cuda")
    sinv = torch.zeros((), device="cuda")
    row = torch.empty(m, 2 * f + 64, device="cuda", dtype=torch.uint8)[:, :2 * f]
    col = torch.empty(2 * f, m + 64, device="cuda", dtype=torch.uint8)[:, :m]
    rw, cw = F.swiglu_bwd_cast_transpose(g, y, scale, amax, out=(row, col), scale_inv=sinv)
    assert rw.data_ptr() == row.data_ptr() and cw.data_ptr() == col.data_ptr()
    want = (_swiglu_back(g, y).float() * 2000.0).to(torch.float8_e5m2)
    _close(row.view(torch.float8_e5m2), want)
    _close(col.view(torch.float8_e5m2), want.t())
    assert sinv.item() == pytest.approx(1 / 2000.0)
