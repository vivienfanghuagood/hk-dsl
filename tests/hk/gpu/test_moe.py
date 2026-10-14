"""The MoE GEMM on vLLM's sorted lists, against the same product in torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import moe  # noqa: E402


def _reference(a, w, sorted_ids, expert_ids, ntp, top_k, block_m, topk_w, rows):
    out = torch.zeros(rows, w.shape[1], device=a.device, dtype=torch.float32)
    total = int(ntp.item())
    ids = sorted_ids[:total].long()
    for b in range(total // block_m):
        e = int(expert_ids[b])
        blk = ids[b * block_m:(b + 1) * block_m]
        blk = blk[blk < rows]
        y = a[blk // top_k].float() @ w[e].float().t()
        if topk_w is not None:
            y *= topk_w.flatten()[blk].float()[:, None]
        out[blk] = y
    return out


def _routing(tokens, experts, top_k, g):
    scores = torch.rand(tokens, experts, device="cuda", generator=g)
    topk_w, topk_ids = scores.topk(top_k, dim=1)
    return topk_w.softmax(dim=1).float(), topk_ids.to(torch.int32)


@pytest.mark.parametrize("tokens", [1, 5, 33])
@pytest.mark.parametrize("routed", [False, True], ids=["plain", "routed"])
@pytest.mark.parametrize("block_m", [16, 32])
def test_moe_gemm_matches_the_sorted_product(tokens, routed, block_m):
    g = torch.Generator(device="cuda").manual_seed(tokens * 7 + block_m)
    experts, top_k, n, k = 8, 2, 256, 1024
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16,
                    generator=g) * k ** -0.5
    topk_w, topk_ids = _routing(tokens, experts, top_k, g)
    sorted_ids, expert_ids, ntp = moe.moe_align(topk_ids, block_m, experts)
    c = torch.full((tokens, top_k, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    moe.moe_gemm(a, w, c, sorted_ids, expert_ids, ntp, top_k=top_k,
                 block_m=block_m, topk_weights=topk_w if routed else None)
    want = _reference(a, w, sorted_ids, expert_ids, ntp, top_k, block_m,
                      topk_w if routed else None, tokens * top_k)
    got = c.view(-1, n).float()
    assert not got.isnan().any(), "a (token, choice) row was never written"
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err


@pytest.mark.parametrize("routed", [False, True], ids=["plain", "routed"])
def test_the_naive_assignment_is_one_pair_a_block(routed):
    # vLLM skips the sort when tokens * top_k * 4 <= experts: no sorted list,
    # the flattened top-k ids as the experts, and pairs * BLOCK_M as the count.
    g = torch.Generator(device="cuda").manual_seed(11)
    tokens, experts, top_k, n, k, block_m = 2, 64, 8, 256, 1024, 16
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16,
                    generator=g) * k ** -0.5
    topk_w, topk_ids = _routing(tokens, experts, top_k, g)
    pairs = tokens * top_k
    ntp = torch.tensor([pairs * block_m], device="cuda", dtype=torch.int32)
    c = torch.full((tokens, top_k, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    moe.moe_gemm(a, w, c, None, topk_ids.flatten(), ntp, top_k=top_k,
                 block_m=block_m, topk_weights=topk_w if routed else None)
    flat = topk_ids.flatten().long()
    want = torch.stack([a[p // top_k].float() @ w[flat[p]].float().t()
                        for p in range(pairs)])
    if routed:
        want *= topk_w.flatten()[:, None]
    got = c.view(pairs, n).float()
    assert not got.isnan().any()
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err


def test_the_second_gemm_reads_one_row_per_pair():
    # w2's A is the activated intermediate, one row per (token, choice), so
    # top_k is 1 there and the routing weight is applied on the way out.
    g = torch.Generator(device="cuda").manual_seed(3)
    tokens, experts, top_k, n, k = 9, 8, 2, 128, 512
    a = torch.randn(tokens * top_k, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16,
                    generator=g) * k ** -0.5
    topk_w, topk_ids = _routing(tokens, experts, top_k, g)
    sorted_ids, expert_ids, ntp = moe.moe_align(topk_ids, 16, experts)
    c = torch.empty(tokens, top_k, n, device="cuda", dtype=torch.bfloat16)
    moe.moe_gemm(a, w, c, sorted_ids, expert_ids, ntp, top_k=1, block_m=16,
                 topk_weights=topk_w)
    want = _reference(a, w, sorted_ids, expert_ids, ntp, 1, 16, topk_w,
                      tokens * top_k)
    err = (c.view(-1, n).float() - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err
