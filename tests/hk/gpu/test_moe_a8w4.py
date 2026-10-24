"""K1 stage 1 (hk.ops.moe_a8w4): FP8 x AITER-layout MXFP4 gate|up GEMM over sorted (token, expert)
pairs, SiLU-mul with AITER's clamp, against a torch dequantized reference on the same quantized
inputs. Ragged experts (empty ones included), tokens that are not whole tiles."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled MFMA", allow_module_level=True)

from hk.ops import moe_a8w4 as M  # noqa: E402

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


@pytest.mark.parametrize("nb,mt,warps", [(4, 2, 4), (2, 1, 4), (4, 3, 2)])
def test_stage1_matches_dequant(nb, mt, warps):
    torch.manual_seed(0)
    E, I, K, T, topk, L = 6, 128, 512, 77, 2, 10.0
    dev = "cuda"
    w = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    ws = torch.randint(122, 132, (E * 2 * I, K // 32), dtype=torch.uint8)
    hidden = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).to(dev)   # expert E-1 empty
    xq, xs, tiles, pairs = M.prepare(hidden, ids, E)
    h = M.stage1(xq, xs, M.aiter_layout_w13(w).to(dev), M.aiter_layout_w13_scale(ws, E).to(dev),
                 tiles, inter=I, nb=nb, mt=mt, warps=warps, limit=L)
    b = w.long()
    wd = torch.stack([E2M1[b & 15], E2M1[b >> 4]], -1).reshape(E, 2 * I, K)
    wd = (wd * torch.pow(2.0, ws.float() - 127).view(E, 2 * I, K // 32).repeat_interleave(32, -1)).to(dev)
    xd = xq.float() * torch.pow(2.0, xs.float() - 127).repeat_interleave(32, 1)
    live = pairs >= 0
    ex = ids.reshape(-1)[pairs.clamp_min(0).long()]
    gu = torch.einsum("pk,pnk->pn", xd[live], wd[ex[live]])
    g, u = gu[:, :I].clamp(max=L), gu[:, I:].clamp(-L, L)
    ref = (g * torch.sigmoid(g) * u)
    got = h[live].float()
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    assert err < 1e-2, err
    assert torch.equal(h[live], M.stage1(xq, xs, M.aiter_layout_w13(w).to(dev),
                                         M.aiter_layout_w13_scale(ws, E).to(dev), tiles, inter=I,
                                         nb=nb, mt=mt, warps=warps, limit=L)[live])
