"""The RMSNorm a model runs -- weighted, and vLLM's fused residual add --
against torch.

Decode normalises one row per request, so the row counts below include fewer
than a tile's sixteen and counts that are not a multiple of it. The fused op
is in place, like vLLM's: the residual becomes x + residual (rounded to the
dtype, as stored) and x its norm, computed from the unrounded sum.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

SHAPES = [(1, 4096), (5, 4096), (16, 4096), (37, 2048), (300, 4096),
          (8192, 4096), (4100, 2560)]


def _ref(z, w, eps=1e-6):
    z = z.float()
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_rms_norm_weighted(rows, cols):
    g = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.rand(cols, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    assert _rel(hk.ops.rms_norm(x, w), _ref(x, w)) < 2e-2


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_fused_add_rms_norm_in_place(rows, cols):
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    r = torch.randn_like(x)
    w = torch.rand(cols, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    z = x.float() + r.float()
    want_y, want_r = _ref(z, w), z.to(torch.bfloat16)
    y, r2 = hk.ops.fused_add_rms_norm(x, r, w)
    assert y.data_ptr() == x.data_ptr() and r2.data_ptr() == r.data_ptr()
    assert torch.equal(r, want_r)
    assert _rel(x, want_y) < 2e-2
