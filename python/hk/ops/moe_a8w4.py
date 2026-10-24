"""MoE a8w4 for gfx950 (K1): FP8 activations x AITER's MXFP4 expert weights, as vLLM's DeepSeek-V4.1
prefill stores them (AITER `shuffle_weight` / `shuffle_scale` with is_guinterleave, w13 gate|up).

Stage 1 here: h = act(gate, up) per (token, expert) pair, act = SiLU-mul with AITER's clamp
(gate <= L, |up| <= L), on gfx950's scaled MFMA with the weights as the FP4 A operand -- the
gate and up 16-row blocks are then accumulator rows, so the activation pairs subtiles in place.

A workgroup owns one expert and `nb` 16-row gate blocks (with their up blocks: 32 * nb weight rows,
16 * nb columns of h), and streams that block of the weight once per K step for all of the
expert's tokens: each warp takes `mt` 16-token tiles at a time, so a weight fragment loaded once
feeds `mt` MFMAs, and the workgroup's warps read the same fragments while they are in L2.

Inputs are in the sorted order a MoE sort makes (`prepare`): the (token, choice) pairs grouped by
expert, each expert's run padded to whole 16-row tiles; activations already MX FP8 in that order
(AITER's own stage 1 reads them so too: its quant writes them sorted).
"""

from __future__ import annotations

import functools

from ..types import bf16, fp8e4m3, fp32, i8, i32

TILE = 16


def stage1_kernel(name: str, *, group_rows: int, nb: int = 4, mt: int = 2, warps: int = 4,
                  limit: float = 10.0):
    """h [P, I] bf16 = act(x [P, K] fp8 (MX scales xs [P, K/32]) . w13[e]^T) for every 16-row tile of
    expert e: tiles [tiles[e], tiles[e+1]) of the sorted order (int32 [1, 1, 1, E + 1])."""
    import hk  # noqa: PLC0415

    GR = group_rows
    R = 32 * nb                                            # weight rows a workgroup: nb (gate, up) pairs
    lim = float(limit)

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], w: hk.Tensor[i8], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], h: hk.Tensor[bf16]):
        blk = hk.arch.block_idx.x
        e = hk.arch.block_idx.y
        t0 = tiles[hk.elem_coord(0, 0, 0, e)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        wid = hk.arch.warp_id()
        row_w = e * GR + blk * (16 * nb)
        steps = x.cols // 128
        span = warps * mt                                  # tiles a round of the workgroup
        accs = [hk.zeros(hk.RT(fp32, R, TILE, "col")) for _ in range(mt)]
        for rnd in range(hk.cdiv(t1 - t0, span)):
            base = t0 + rnd * span + wid * mt
            for acc in accs:
                acc.zero_()
            for kc in range(steps):
                a = hk.load_fp4_frags(hk.RT(fp8e4m3, R, 128), w, row_w, kc, group_rows=GR,
                                      gate_up=True)
                sa = hk.load_mx_scales(ws, R, 4, row_w, kc * 4, layout="aiter_gate_up",
                                       group_rows=GR)
                for i in range(mt):
                    # a tile past the expert's last reads its neighbour's rows; its result is
                    # never stored
                    r0 = hk.min(base + i, t1 - 1) * TILE
                    b = x[hk.elem_coord(0, 0, r0, kc * 128)].load(hk.RT(fp8e4m3, TILE, 128))
                    sb = hk.load_mx_scales(xs, TILE, 4, r0, kc * 4)
                    hk.mma_ABt_mx(a, b, accs[i], sa, sb, a_fmt="fp4", into=accs[i])
            for i in range(mt):
                if base + i < t1:
                    for j in range(nb):
                        g = accs[i].subtile(TILE, 2 * j).minimum(lim)
                        u = accs[i].subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                        hv = g.silu() * u
                        h[hk.elem_coord(0, 0, (base + i) * TILE, (blk * nb + j) * TILE)].store(
                            hk.transpose(hv))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name,
                     grid=lambda p: (p.h.cols // (16 * nb), p.tiles.cols - 1, 1))


@functools.lru_cache(maxsize=None)
def _stage1(group_rows: int, nb: int, mt: int, warps: int, limit: float):
    return stage1_kernel(f"moe_a8w4_s1_gr{group_rows}_nb{nb}_mt{mt}_w{warps}", group_rows=group_rows,
                         nb=nb, mt=mt, warps=warps, limit=limit)


def prepare(hidden, topk_ids, n_experts: int):
    """The sort and MX FP8 quant a stage 1 reads (torch; AITER fuses this into one kernel):
    (x fp8 [P, K], xs uint8 [P, K/32], tiles int32 [1, 1, 1, E + 1], pairs int32 [P] (token * topk +
    choice, -1 padding)). P = the experts' pair counts, each padded to whole 16-row tiles."""
    import torch  # noqa: PLC0415

    m, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    ntile = (counts + TILE - 1) // TILE
    tiles = torch.zeros(n_experts + 1, dtype=torch.int64, device=hidden.device)
    tiles[1:] = torch.cumsum(ntile, 0)
    p = int(tiles[-1]) * TILE
    start = torch.zeros_like(counts)
    start[1:] = torch.cumsum(counts, 0)[:-1]
    dst = tiles[:-1][flat[order]] * TILE + (torch.arange(order.numel(), device=hidden.device)
                                            - start[flat[order]])
    pairs = torch.full((p,), -1, dtype=torch.int64, device=hidden.device)
    pairs[dst] = order
    rows = hidden.new_zeros((p, hidden.shape[1]))
    live = pairs >= 0
    rows[live] = hidden[pairs[live] // topk]
    xq, xs = mx_quant(rows)
    return xq, xs, tiles.to(torch.int32).view(1, 1, 1, -1), pairs.to(torch.int32)


def aiter_layout_w13(w):
    """[E, 2I, K/2] uint8 (gate rows, then up) -> AITER's gfx950 MoE layout (shuffle_weight,
    is_guinterleave, gate_up), as a [E * 2I, K/2] uint8 view. For tests: vLLM's weights are stored so."""
    e, n, kp = w.shape
    x = w.view(e, 2, n // 32, 16, kp // 64, 4, 16)
    return x.permute(0, 2, 1, 4, 5, 3, 6).contiguous().view(e * n, kp)


def aiter_layout_w13_scale(s, e: int):
    """[E * 2I, K/32] uint8 E8M0 -> AITER's shuffle_scale(is_guinterleave, gate_up) layout."""
    rows, kb = s.shape
    x = s.view(e, 2, rows // e // 32, 16, kb // 8, 2, 4)
    return x.permute(0, 2, 4, 6, 3, 5, 1).contiguous().view(rows, kb)


def mx_quant(x):
    """Rows of x to FP8 e4m3 with an E8M0 scale per 32 (OCP MX: scale 2^(floor(log2 amax) - 8))."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    v = x.float().view(m, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(2.0 ** -126)
    e = torch.floor(torch.log2(amax)) - 8
    q = (v / torch.pow(2.0, e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(m, k), (e + 127).clamp(0, 254).to(torch.uint8)


def stage1(xq, xs, w13, w13_scale, tiles, *, inter: int, nb: int = 4, mt: int = 2, warps: int = 4,
           limit: float = 10.0):
    """h [P, inter] bf16 for prepare()'s sorted rows. w13 / w13_scale: AITER's shuffled uint8 views
    [E * 2 * inter, K/2] and [E * 2 * inter, K/32]."""
    import torch  # noqa: PLC0415

    h = torch.empty((xq.shape[0], inter), device=xq.device, dtype=torch.bfloat16)
    _stage1(2 * inter, nb, mt, warps, float(limit))(
        xq, xs.view(torch.int8), w13.view(torch.int8), w13_scale.view(torch.int8), tiles, h)
    return h
