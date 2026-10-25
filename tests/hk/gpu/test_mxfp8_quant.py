"""hk.ops.mxfp8.quant: bf16 x -> FP8 e4m3 + E8M0 scales in one launch, bit for bit hk.mx.quant
(AITER's per_1x32 rule) followed by the layout the GEMM reads -- hk.mx.pack_scales (int32) or
hk.mx.aiter_scales (uint8) -- at ragged and whole M, and K of whole and partial 256-wide pairs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's FP8 conversion", allow_module_level=True)

pytestmark = pytest.mark.gpu

from hk.ops import mxfp8  # noqa: E402


def _x(m, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # rows of very different magnitude, so the scales span many exponents
    return (torch.randn(m, k, device="cuda", generator=g)
            * torch.exp2(torch.randint(-20, 20, (m, 1), device="cuda", generator=g).float())).bfloat16()


@pytest.mark.parametrize("m,k", [(64, 128), (100, 1152), (2048, 5120), (2059, 1280), (130, 6144)])
def test_packed_quant_is_quant_then_pack_scales(m, k):
    x = _x(m, k)
    q, s = mxfp8.quant(x, packed=True)
    rq, rs = hk.mx.quant(x)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s, hk.mx.pack_scales(rs))          # rows past M at the unit scale too


@pytest.mark.parametrize("m,k", [(64, 1024), (96, 1152), (2048, 5120)])
def test_aiter_layout_quant_is_quant_then_aiter_scales(m, k):
    x = _x(m, k, seed=1)
    q, s = mxfp8.quant(x, packed=False)
    rq, rs = hk.mx.quant(x)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s, hk.mx.aiter_scales(rs))

