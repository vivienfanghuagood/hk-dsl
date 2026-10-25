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
    got = [M.ring_round(p, 384, 5120, pairs=True) for p in (6240, 15024, 27552, 52208, 101264)]
    assert got == [(2, 2), (4, 2), (6, 2), (6, 2), (6, 2)]


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
    # the weights applied in stage 2's epilogue instead, reduce only summing
    yw = M.stage2(hq, hs, M.aiter_layout_w2(w).to(dev), M.aiter_layout_w2_scale(ws).to(dev), tiles,
                  mt=mt, pp=pp, depth=depth, wrow=M.row_weights(pairs, wt))
    out_w = M.reduce(yw, pairs, wt, T, weighted=False)
    assert ((out_w.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
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


@pytest.mark.parametrize("mt,pp,I", [(2, 3, 192), (4, 3, 192), (6, 2, 128)])
def test_stage1_pairs_matches_per_step_scales(mt, pp, I):
    """pairs: both scales as two-step 32-row dwords, read once per two K steps -- bit for bit the
    per-step kernel on the same (32-row aligned) rows."""
    from hk.ops import mxfp8

    torch.manual_seed(3)
    E, K, T, topk = 6, 1024, 91, 2
    w = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    ws = torch.randint(118, 136, (E * 2 * I, K // 32), dtype=torch.uint8)
    hidden = torch.randn(T, K, device="cuda", dtype=torch.bfloat16) * 3
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).cuda()
    xq, xs, tiles, pairs = M.prepare(hidden, ids, E, align=32)
    assert xq.shape[0] % 32 == 0 and bool((tiles.view(-1) % 2 == 0).all())
    wl, wsl = M.aiter_layout_w13(w).cuda(), M.aiter_layout_w13_scale(ws, E).cuda()
    kw = dict(inter=I, mt=mt, lds=True, pp=pp, depth=2, exact=True, xsplit=True)
    ref = M.stage1(xq, xs, wl, wsl, tiles, **kw)
    got = M.stage1(xq, mxfp8.aiter_scales(xs), wl, wsl, tiles, pairs=True, **kw)
    live = pairs >= 0
    assert torch.equal(got[live], ref[live])


@pytest.mark.parametrize("mt,pp,I", [(2, 4, 1152), (4, 2, 640), (6, 4, 512)])
def test_stage2_pairs_matches_per_step_scales(mt, pp, I):
    """stage2(pairs=True), K steps not a multiple of four (1152: 9) included: bit for bit the
    per-step kernel on the same 32-row aligned rows."""
    from hk.ops import mxfp8

    torch.manual_seed(4)
    E, D, T, topk = 6, 256, 77, 2
    w = torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8)
    ws = torch.randint(118, 136, (E * D, I // 32), dtype=torch.uint8)
    hidden = torch.randn(T, I, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).cuda()
    hq, hs, tiles, pairs = M.prepare(hidden, ids, E, align=32)
    wl, wsl = M.aiter_layout_w2(w).cuda(), M.aiter_layout_w2_scale(ws).cuda()
    steps = I // 128
    ref = M.stage2(hq, hs, wl, wsl, tiles, mt=mt, pp=pp, depth=next(d for d in (2, 3, 5) if steps % d == 0))
    got = M.stage2(hq, mxfp8.aiter_scales(hs), wl, wsl, tiles, mt=mt, pp=pp, depth=2, pairs=True)
    live = pairs >= 0
    assert torch.equal(got[live], ref[live])


def _aiter_like_sort(ids, n_experts, bm):
    """AITER moe_sorting's outputs, in torch: sorted ids (token | choice << 24, padding token = M),
    per-block experts, num_valid -- each expert's pairs in token order, padded to bm rows."""
    m, topk = ids.shape
    flat = ids.reshape(-1).long()
    rows, eids = [], []
    for e in range(n_experts):
        p = torch.nonzero(flat == e).view(-1)
        if p.numel() == 0:
            continue
        v = ((p // topk) | ((p % topk) << 24)).to(torch.int32)
        pad = -v.numel() % bm
        v = torch.cat([v, torch.full((pad,), m, dtype=torch.int32, device=ids.device)])
        rows.append(v)
        eids += [e] * (v.numel() // bm)
    sid = torch.cat(rows)
    nvalid = sid.numel()
    sid = torch.cat([sid, torch.full((2 * bm,), m, dtype=torch.int32, device=ids.device)])  # slack
    se = torch.tensor(eids + [0, 0], dtype=torch.int32, device=ids.device)
    nv = torch.tensor([nvalid, m], dtype=torch.int32, device=ids.device)
    return sid, se, nv


@pytest.mark.parametrize("mt,pp1,pp2,depth,pers,I", [(2, 3, 4, 2, 0, 768), (4, 3, 2, 2, 0, 768),
                                                     (6, 3, 4, 2, 0, 768), (4, 3, 4, 3, 0, 768),
                                                     (4, 3, 4, 2, 8, 768), (2, 3, 2, 2, 16, 768),
                                                     (6, 3, 4, 2, 64, 768),
                                                     (4, 3, 4, 2, 0, 1152), (4, 3, 4, 2, 0, 2304)])
def test_sorted_stages_match_dequant(mt, pp1, pp2, depth, pers, I):
    """stage1_sorted / stage2_sorted on AITER-style sort metadata, x gathered in the kernel, y2
    scattered choice-major and summed by reduce: the dequantized MoE."""
    from hk.ops import mxfp8

    torch.manual_seed(5)
    E, K, D, T, topk, L = 6, 512, 512, 77, 2, 10.0
    dev = "cuda"
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    w1s = torch.randint(122, 132, (E * 2 * I, K // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8)
    w2s = torch.randint(122, 132, (E * D, I // 32), dtype=torch.uint8)
    hidden = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).to(dev)
    wt = torch.rand(T, topk, device=dev)
    sid, se, nv = _aiter_like_sort(ids, E, 16 * mt)
    xq, xs = M.mx_quant(hidden)
    tok = (sid & 0xFFFFFF).long()
    xs_sorted = torch.full((sid.numel(), K // 32), 127, dtype=torch.uint8, device=dev)
    live = tok < T
    xs_sorted[live] = xs[tok[live]]
    h = M.stage1_sorted(xq, mxfp8.aiter_scales(xs_sorted), M.aiter_layout_w13(w1).to(dev),
                        M.aiter_layout_w13_scale(w1s, E).to(dev), sid, se, nv, inter=I, mt=mt,
                        pp=pp1, limit=L, depth=depth)
    # reference stage 1 on the same rows
    b = w1.long()
    w1d = torch.stack([E2M1[b & 15], E2M1[b >> 4]], -1).reshape(E, 2 * I, K)
    w1d = (w1d * torch.pow(2.0, w1s.float() - 127).view(E, 2 * I, K // 32).repeat_interleave(32, -1)).to(dev)
    xd = xq.float() * torch.pow(2.0, xs.float() - 127).repeat_interleave(32, 1)
    nvalid = int(nv[0])
    blk_e = se.long().repeat_interleave(16 * mt)[:nvalid]
    lv = live[:nvalid]
    gu = torch.einsum("pk,pnk->pn", xd[tok[:nvalid][lv]], w1d[blk_e[lv]])
    g, u = gu[:, :I].clamp(max=L), gu[:, I:].clamp(-L, L)
    href = g * torch.sigmoid(g) * u
    got = h[:nvalid][lv].float()
    assert ((got - href).abs().max() / href.abs().max()).item() < 1e-2
    # the MX-quantizing epilogue: bit for bit mx_quant(h) in AITER's scale layout
    hq_e, hs_e = M.stage1_sorted(xq, mxfp8.aiter_scales(xs_sorted), M.aiter_layout_w13(w1).to(dev),
                                 M.aiter_layout_w13_scale(w1s, E).to(dev), sid, se, nv, inter=I,
                                 mt=mt, pp=pp1, limit=L, depth=depth, mxq=True)
    q_ref, s_ref = M.mx_quant(h)
    # rows past the sort's valid count belong to no block that runs: unspecified, as h's are
    assert torch.equal(hq_e[:nvalid].view(torch.uint8), q_ref[:nvalid].view(torch.uint8))
    mark = torch.zeros_like(s_ref)
    mark[:nvalid] = 1
    real = mxfp8.aiter_scales(mark) == 1                 # valid rows' bytes; pads are not written
    assert torch.equal(hs_e[real], mxfp8.aiter_scales(s_ref)[real])
    # stage 2 + reduce
    hq, hs = M.mx_quant(h)
    sw = torch.zeros(sid.numel(), device=dev)
    sw[:nvalid][lv] = wt[tok[:nvalid][lv], (sid[:nvalid][lv] >> 24).long()]
    y2 = M.stage2_sorted(hq, mxfp8.aiter_scales(hs), M.aiter_layout_w2(w2).to(dev),
                         M.aiter_layout_w2_scale(w2s).to(dev), sid, se, nv, sw, T, topk, E,
                         mt=mt, pp=pp2, depth=depth, persistent=pers)
    out = M.reduce(y2, None, wt, T, inv=M.identity(T, topk, dev), wrow=sw.view(-1, 1),
                   weighted=False)
    if not pers and I % 384 == 0 and D % (16 * pp2 * 4 * 2) == 0:
        # h resident in LDS, two column tiles a workgroup: the same rows, bit for bit
        yr = M.stage2_sorted(hq, mxfp8.aiter_scales(hs), M.aiter_layout_w2(w2).to(dev),
                             M.aiter_layout_w2_scale(w2s).to(dev), sid, se, nv, sw, T, topk, E,
                             mt=mt, pp=pp2, resident=2)
        assert torch.equal(yr, y2)
    assert torch.equal(out, M.reduce(y2, None, wt, T, dense=True))
    # the top-k sum by bf16 atomics in the scatter (AITER's way): the same up to the adds' rounding
    if pers:
        with pytest.raises(ValueError, match="counts twice"):
            M.stage2_sorted(hq, mxfp8.aiter_scales(hs), M.aiter_layout_w2(w2).to(dev),
                            M.aiter_layout_w2_scale(w2s).to(dev), sid, se, nv, sw, T, topk, E,
                            mt=mt, pp=pp2, persistent=pers, atomic=True)
        return
    ya = M.stage2_sorted(hq, mxfp8.aiter_scales(hs), M.aiter_layout_w2(w2).to(dev),
                         M.aiter_layout_w2_scale(w2s).to(dev), sid, se, nv, sw, T, topk, E,
                         mt=mt, pp=pp2, depth=depth, persistent=pers, atomic=True)
    assert ya.shape == out.shape
    assert ((ya.float() - out.float()).abs().max() / out.float().abs().max()).item() < 1e-2
    b2 = w2.long()
    w2d = torch.stack([E2M1[b2 & 15], E2M1[b2 >> 4]], -1).reshape(E, D, I)
    w2d = (w2d * torch.pow(2.0, w2s.float() - 127).view(E, D, I // 32).repeat_interleave(32, -1)).to(dev)
    hd = hq.float() * torch.pow(2.0, hs.float() - 127).repeat_interleave(32, 1)
    yr = torch.einsum("pk,pnk->pn", hd[:nvalid][lv], w2d[blk_e[lv]]) * sw[:nvalid][lv, None]
    ref = torch.zeros(T, D, device=dev)
    ref.index_add_(0, tok[:nvalid][lv], yr)
    assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2



@pytest.mark.parametrize("mt,pp,grid", [(4, 4, (2, 2)), (8, 2, (2, 4))])
def test_stage1_warp_grid_matches_ring(mt, pp, grid):
    """stage1_sorted(grid=(tr, tc)): both operands through LDS on a warp grid -- bit for bit the
    four-warp sorted kernel on the same rows."""
    from hk.ops import mxfp8

    torch.manual_seed(6)
    E, I, K, T, topk = 6, 1024, 512, 77, 2
    dev = "cuda"
    w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    w1s = torch.randint(122, 132, (E * 2 * I, K // 32), dtype=torch.uint8)
    hidden = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).to(dev)
    sid, se, nv = _aiter_like_sort(ids, E, 16 * mt)
    xq, xs = M.mx_quant(hidden)
    tok = (sid & 0xFFFFFF).long()
    xs_sorted = torch.full((sid.numel(), K // 32), 127, dtype=torch.uint8, device=dev)
    live = tok < T
    xs_sorted[live] = xs[tok[live]]
    args = (xq, mxfp8.aiter_scales(xs_sorted), M.aiter_layout_w13(w1).to(dev),
            M.aiter_layout_w13_scale(w1s, E).to(dev), sid, se, nv)
    ref = M.stage1_sorted(*args, inter=I, mt=mt, pp=2)
    got = M.stage1_sorted(*args, inter=I, mt=mt, pp=pp, grid=grid)
    n = int(nv[0])
    assert torch.equal(got[:n][live[:n]], ref[:n][live[:n]])
