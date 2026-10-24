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


@pytest.mark.parametrize("nb,mt,warps,lds,pp,depth", [(4, 2, 4, False, 1, 0), (2, 1, 4, False, 1, 0),
                                                      (4, 3, 2, False, 1, 0), (4, 2, 4, True, 1, 0),
                                                      (4, 4, 4, True, 2, 0), (4, 3, 4, True, 2, 0),
                                                      (4, 3, 4, True, 2, 4), (4, 4, 4, True, 1, 2),
                                                      (4, 4, 4, "dma", 2, 4), (4, 3, 4, "dma", 2, 2),
                                                      (4, 4, 4, "direct", 2, 3), (4, 3, 4, "direct", 1, 2),
                                                      (4, 4, 4, "exact", 2, 4), (4, 3, 4, "exact", 2, 0),
                                                      (4, 3, 4, "xsplit", 2, 2), (4, 5, 4, "xsplit", 2, 4),
                                                      (4, 0, 4, "auto", 0, 0), (4, 3, 4, "xsplit", 3, 2),
                                                      (4, 5, 4, "xsplit", 3, 4), (4, 0, 4, "auto3", 0, 0)])
def test_stage1_matches_dequant(nb, mt, warps, lds, pp, depth):
    torch.manual_seed(0)
    # I 192: pp 3's workgroup (192 columns); auto3 must pick pp 3 there
    I = 192 if pp == 3 or lds == "auto3" else 128
    lds = "auto" if lds == "auto3" else lds
    E, K, T, topk, L = 6, 512, 77, 2, 10.0
    dev = "cuda"
    w = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    ws = torch.randint(122, 132, (E * 2 * I, K // 32), dtype=torch.uint8)
    hidden = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).to(dev)   # expert E-1 empty
    xq, xs, tiles, pairs = M.prepare(hidden, ids, E)
    h = M.stage1(xq, xs, M.aiter_layout_w13(w).to(dev), M.aiter_layout_w13_scale(ws, E).to(dev),
                 tiles, inter=I, nb=nb, mt=mt, warps=warps, limit=L, lds=bool(lds), pp=pp, depth=depth, dma=lds == "dma", direct=lds == "direct", exact=lds in ("exact", "xsplit"), xsplit=lds == "xsplit", auto=lds == "auto")
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
                                         nb=nb, mt=mt, warps=warps, limit=L, lds=bool(lds), pp=pp, depth=depth, dma=lds == "dma", direct=lds == "direct", exact=lds in ("exact", "xsplit"), xsplit=lds == "xsplit", auto=lds == "auto")[live])


def test_ring_round():
    # Kimi K3 (E 384, K 5120) uniform-routing row counts at M 640, 2048, 4096, 8192, 16384 (K1 c9)
    got = [M.ring_round(p, 384, 5120) for p in (6240, 15024, 27552, 52208, 101264)]
    assert got == [(2, 2), (3, 2), (5, 2), (5, 2), (6, 2)]
    assert M.ring_round(0, 384, 5120) == (2, 2)


def test_stage1_refuses_partial_grid():
    # inter 1152 (Kimi K3) is 4.5 workgroups of pp 4: the grid would leave 128 columns unwritten
    xq = torch.zeros((32, 512), device="cuda", dtype=torch.float8_e4m3fn)
    xs = torch.zeros((32, 16), device="cuda", dtype=torch.uint8)
    tiles = torch.tensor([0, 2], device="cuda", dtype=torch.int32).view(1, 1, 1, -1)
    w = torch.zeros((2304, 256), device="cuda", dtype=torch.uint8)
    ws = torch.zeros((2304, 16), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="inter=1152"):
        M.stage1(xq, xs, w, ws, tiles, inter=1152, mt=2, lds=True, pp=4, depth=2, exact=True, xsplit=True)


def test_stage1_refuses_k_remainder():
    # K 640 is 5 steps of 128: a depth-3 ring would drop / garble the remainder
    xq = torch.zeros((32, 640), device="cuda", dtype=torch.float8_e4m3fn)
    xs = torch.zeros((32, 20), device="cuda", dtype=torch.uint8)
    tiles = torch.tensor([0, 2], device="cuda", dtype=torch.int32).view(1, 1, 1, -1)
    w = torch.zeros((256, 320), device="cuda", dtype=torch.uint8)
    ws = torch.zeros((256, 20), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="K=640"):
        M.stage1(xq, xs, w, ws, tiles, inter=128, mt=2, lds=True, pp=2, depth=3, exact=True, xsplit=True)


@pytest.mark.parametrize("mt,pp,depth,I", [(4, 4, 3, 384), (2, 2, 3, 384), (5, 4, 2, 512)])
def test_stage2_and_reduce_match_dequant(mt, pp, depth, I):
    """Stage 2 (h_q . w2^T, AITER's w2 layouts) then the weighted topk sum, against torch dequant."""
    torch.manual_seed(1)
    E, D, T, topk = 6, 256, 77, 2
    dev = "cuda"
    w = torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8)
    ws = torch.randint(122, 132, (E * D, I // 32), dtype=torch.uint8)
    hidden = torch.randn(T, I, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).to(dev)
    wt = torch.rand(T, topk, device=dev)
    hq, hs, tiles, pairs = M.prepare(hidden, ids, E)      # stands in for stage 1's quantized h
    y = M.stage2(hq, hs, M.aiter_layout_w2(w).to(dev), M.aiter_layout_w2_scale(ws).to(dev), tiles,
                 mt=mt, pp=pp, depth=depth)
    out = M.reduce(y, pairs, wt, T)
    b = w.long()
    wd = torch.stack([E2M1[b & 15], E2M1[b >> 4]], -1).reshape(E, D, I)
    wd = (wd * torch.pow(2.0, ws.float() - 127).view(E, D, I // 32).repeat_interleave(32, -1)).to(dev)
    xd = hq.float() * torch.pow(2.0, hs.float() - 127).repeat_interleave(32, 1)
    live = pairs >= 0
    ex = ids.reshape(-1)[pairs.clamp_min(0).long()]
    yr = torch.einsum("pk,pnk->pn", xd[live], wd[ex[live]])
    assert ((y[live].float() - yr).abs().max() / yr.abs().max()).item() < 1e-2
    ref = torch.zeros(T, D, device=dev)
    ref.index_add_(0, pairs[live].long() // topk, yr * wt.reshape(-1)[pairs[live].long()].unsqueeze(1))
    assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
    assert torch.equal(out, M.reduce(M.stage2(hq, hs, M.aiter_layout_w2(w).to(dev),
                                              M.aiter_layout_w2_scale(ws).to(dev), tiles, mt=mt,
                                              pp=pp, depth=depth), pairs, wt, T))


@pytest.mark.parametrize("m,d,topk", [(77, 512, 2), (200, 1024, 6), (16, 512, 6)])
def test_reduce_matches_torch(m, d, topk):
    """hk reduce: each token's topk rows of y, weighted, summed in order -- as torch's fp32 sum."""
    torch.manual_seed(2)
    E = 9
    ids = torch.stack([torch.randperm(E)[:topk] for _ in range(m)]).cuda()
    hidden = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
    _, _, tiles, pairs = M.prepare(hidden, ids, E)
    y = torch.randn(pairs.numel(), d, device="cuda").to(torch.bfloat16)
    wt = torch.rand(m, topk, device="cuda")
    got = M.reduce(y, pairs, wt, m)
    ref = M.reduce_torch(y, pairs, wt, m)
    assert ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item() < 1e-2
