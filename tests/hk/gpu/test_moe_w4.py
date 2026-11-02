"""The MoE GEMM with 4-bit W in vLLM's moe_wna16 layout (hk.ops.moe.moe_gemm_w4: hk.load_w4(zp=,
rows=True)) -- AWQ's asymmetric codes, a 4-bit zero code per row and group, fp16 activations --
against float64, and every code and zero code through one-hot rows."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import moe  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx1" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="hk.load_w4(zp=) is RDNA's")

G = 32
#: (slices, warps) of the w13- and w2-shaped GEMMs: the schedules the vLLM route uses.
W13, W2 = (8, 8), (8, 2)


def _layer(experts, n, k, g, q=None, z=None, s=None):
    """vLLM's moe_wna16 tensors (codes uint8 [E, N, K/2] low nibble = even k, scales [E, N, K/G],
    zero codes uint8 [E, N/2, K/G] row 2j low nibble) and hk's views of them."""
    if q is None:
        q = torch.randint(0, 16, (experts, n, k), device="cuda", generator=g, dtype=torch.int32)
    if z is None:
        z = torch.randint(0, 16, (experts, n, k // G), device="cuda", generator=g, dtype=torch.int32)
    if s is None:
        s = (torch.rand(experts, n, k // G, device="cuda", generator=g) * 0.02 + 0.001).half()
    qb = (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)
    zb = (z[:, 0::2] | (z[:, 1::2] << 4)).to(torch.uint8)
    qw = qb.view(torch.int32).reshape(experts * n, k // 8)
    zp = zb.view(torch.int8).reshape(experts * n // 2, k // G)
    w = (q - z.repeat_interleave(G, dim=2)).double() * s.double().repeat_interleave(G, dim=2)
    return (qw, s.reshape(experts * n, k // G), zp), w


def _routing(tokens, experts, top_k, g):
    scores = torch.rand(tokens, experts, device="cuda", generator=g)
    topk_w, topk_ids = scores.topk(top_k, dim=1)
    return topk_w.softmax(dim=1).float(), topk_ids.to(torch.int32)


@pytest.mark.parametrize("tokens", [1, 3, 8, 16, 33])
@pytest.mark.parametrize("shape", ["w13", "w2"])
def test_within_two_ulps_a_row_of_float64(tokens, shape):
    """Qwen3.6-35B-A3B's expert shapes, 256 experts, top 8: each output row within two fp16 ulps of
    its largest |value| (one rounding of each weight, fp32 accumulation, one rounding of the output)."""
    g = torch.Generator(device="cuda").manual_seed(tokens * 3 + (shape == "w2"))
    experts, top_k = 256, 8
    n, k, (sl, wp), routed = (1024, 2048, W13, False) if shape == "w13" else (2048, 512, W2, True)
    (qw, s, zp), w = _layer(experts, n, k, g)
    topk_w, topk_ids = _routing(tokens, experts, top_k, g)
    sorted_ids, expert_ids, ntp = moe.moe_align(topk_ids, 16, experts)
    rows_a = tokens if not routed else tokens * top_k
    a = torch.randn(rows_a, k, device="cuda", dtype=torch.float16, generator=g)
    c = torch.full((tokens * top_k, n), float("nan"), device="cuda", dtype=torch.float16)
    assert moe.takes_w4(a, qw, s, zp, n, 16, G, wp, sl)
    moe.moe_gemm_w4(a, qw, s, zp, c, sorted_ids, expert_ids, ntp, top_k=top_k if not routed else 1,
                    block_m=16, group=G, warps=wp, slices=sl, topk_weights=topk_w if routed else None)
    want = torch.empty(tokens * top_k, n, device="cuda", dtype=torch.float64)
    for p in range(tokens * top_k):
        x = a[p // top_k] if not routed else a[p]
        want[p] = w[topk_ids.flatten()[p].long()] @ x.double()
        if routed:
            want[p] *= topk_w.flatten()[p].double()
    got = c.double()
    assert not got.isnan().any(), "a (token, choice) row was never written"
    top = want.abs().amax(dim=1, keepdim=True)
    ulp = torch.pow(2.0, torch.floor(torch.log2(top.clamp_min(2.0 ** -14)))) * 2.0 ** -10
    assert ((got - want).abs() <= 2 * ulp).all(), ((got - want).abs() / ulp).max().item()


def test_every_code_and_zero_code_dequantizes_to_one_rounding():
    """q = 0..15 against z = 0..15 and scales from subnormal to large, read back through one-hot
    token rows: each output is one weight, (q - z) * s rounded once to nearest even, 0 at q = z."""
    experts, n, k, top_k = 2, 64, 512, 1
    scales = torch.tensor([2.0 ** -24, 3 * 2.0 ** -20, 6.0e-5, 1.0e-3, 0.0371, 0.5, 7.0, 100.0, 1000.0],
                          dtype=torch.float16)
    e_ = torch.arange(experts).view(-1, 1, 1)
    n_ = torch.arange(n).view(1, -1, 1)
    q = ((torch.arange(k).view(1, 1, -1) + n_ + e_) % 16).to(torch.int32).cuda()
    z = ((torch.arange(k // G).view(1, 1, -1) * 5 + n_ * 3 + e_) % 16).to(torch.int32).cuda()
    s = scales[(torch.arange(k // G).view(1, 1, -1) * 7 + n_ + e_) % len(scales)].cuda()
    (qw, sc, zp), _ = _layer(experts, n, k, None, q=q, z=z, s=s)
    want = ((q - z.repeat_interleave(G, dim=2)).double() * s.double().repeat_interleave(G, dim=2)).half()
    for e in range(experts):
        for k0 in range(0, k, 16):
            tokens = 16
            a = torch.zeros(tokens, k, device="cuda", dtype=torch.float16)
            a[torch.arange(tokens), k0 + torch.arange(tokens)] = 1.0
            topk_ids = torch.full((tokens, top_k), e, device="cuda", dtype=torch.int32)
            sorted_ids, expert_ids, ntp = moe.moe_align(topk_ids, 16, experts)
            c = torch.empty(tokens, n, device="cuda", dtype=torch.float16)
            moe.moe_gemm_w4(a, qw, sc, zp, c, sorted_ids, expert_ids, ntp, top_k=1, block_m=16,
                            group=G, warps=2, slices=8)
            assert torch.equal(c.t().contiguous().view(torch.int16),
                               want[e, :, k0:k0 + tokens].contiguous().view(torch.int16))


def test_refuses_what_it_does_not_take():
    g = torch.Generator(device="cuda").manual_seed(1)
    (qw, s, zp), _ = _layer(4, 64, 512, g)
    a = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16, generator=g)
    assert not moe.takes_w4(a, qw, s, zp, 64, 16, G, 2, 8)          # A bf16, scales fp16
    a = a.half()
    assert not moe.takes_w4(a, qw, s, zp, 64, 16, G, 8, 8)          # K 512 not a multiple of 1024
    assert moe.takes_w4(a, qw, s, zp, 64, 16, G, 2, 8)
