"""Paged prefill against a dense reference: what chunked prefill and prefix
caching need from attention.

A request's new tokens are the suffix of its sequence -- `ctx` tokens are
already in the cache -- and each attends causally to everything up to its own
position. The reference gathers the request's pages and applies that mask by
hand. The cases are the ones that break a kernel that assumes otherwise: a
cached prefix, a chunk shorter than a query tile, a tile that runs into the
next request's tokens (whose rows must come out as that request's answer,
not this one's), and rows that belong to no prefill (left untouched).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import _attn_ref  # noqa: E402

from hk.ops import paged_prefill as pp  # noqa: E402

pytestmark = pytest.mark.gpu

PAGE = pp.KV_BLOCK


def _case(reqs, head_dim=128, h_kv=2, group=4, seed=0):
    """reqs: [(ctx, q_len)]. Every request's whole sequence is in the cache;
    its last q_len tokens are the queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(reqs)
    seq = [c + n for c, n in reqs]
    max_pages = max(-(-s // PAGE) for s in seq)
    n_blocks = r * max_pages + 5
    k_cache = torch.randn(n_blocks, h_kv, PAGE, head_dim, device="cuda",
                          dtype=torch.bfloat16, generator=g)
    v_cache = torch.randn(k_cache.shape, device=k_cache.device, dtype=k_cache.dtype, generator=g)
    perm = torch.randperm(n_blocks, generator=g, device="cuda")[: r * max_pages]
    table = perm.reshape(r, max_pages).to(torch.int32).contiguous()
    n_tok = sum(n for _, n in reqs)
    q = torch.randn(n_tok, h_kv * group, head_dim, device="cuda",
                    dtype=torch.bfloat16, generator=g)
    starts = [0]
    for _, n in reqs:
        starts.append(starts[-1] + n)
    q_start = torch.tensor(starts, device="cuda", dtype=torch.int32)
    lens = torch.tensor(seq, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens, q_start


def _reference(q, k_cache, v_cache, table, lens, q_start, group):
    n_tok, h, d = q.shape
    out = torch.zeros(n_tok, h, d, device="cuda", dtype=torch.float32)
    for i in range(len(lens)):
        q0, q1, s = int(q_start[i]), int(q_start[i + 1]), int(lens[i])
        n, ctx = q1 - q0, s - (q1 - q0)
        pages = table[i, : -(-s // PAGE)].tolist()
        pos = ctx + torch.arange(n, device="cuda")
        keep = torch.arange(s, device="cuda")[None, :] <= pos[:, None]
        for hh in range(h):
            kk = torch.cat([k_cache[p, hh // group] for p in pages])[:s]
            vv = torch.cat([v_cache[p, hh // group] for p in pages])[:s]
            out[q0:q1, hh] = _attn_ref.softmax_v64(q[q0:q1, hh], kk, vv, d ** -0.5, keep)
    return out


def _run(reqs, prefill=None, **kw):
    group = kw.get("group", 4)
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, **kw)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    prefill = range(len(reqs)) if prefill is None else prefill
    tiles = pp.work_list([(r, reqs[r][1]) for r in prefill], q.device)
    pp.kernel_for(d, group)(q.view(n_tok, h * d), k_cache, v_cache, table,
                            lens, q_start, tiles, out.view(n_tok, h * d))
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    return out, want, q_start


def test_no_prefix_two_tiles_the_second_short():
    got, want, _ = _run([(0, 300)])
    _attn_ref.check(got, want)


def test_a_cached_prefix_and_a_chunk_shorter_than_a_tile():
    got, want, _ = _run([(1000, 77)])
    _attn_ref.check(got, want)


def test_a_tile_that_runs_into_the_next_request():
    # Request 0's only tile covers rows 0..191; rows 100..149 are request 1's
    # and must come out as request 1's answer.
    got, want, _ = _run([(64, 100), (300, 50), (0, 200)])
    _attn_ref.check(got, want)


def test_ragged_batch_with_a_one_token_request():
    got, want, _ = _run([(0, 192), (500, 5), (40, 600), (33, 1)])
    _attn_ref.check(got, want)


def test_rows_of_requests_not_in_the_work_list_are_untouched():
    reqs = [(0, 40), (128, 70), (10, 30)]
    got, want, q_start = _run(reqs, prefill=[1])
    lo, hi = int(q_start[1]), int(q_start[2])
    assert torch.all(got[:lo] == 7.0) and torch.all(got[hi:] == 7.0)
    _attn_ref.check(got[lo:hi], want[lo:hi])


@pytest.mark.parametrize("group", [1, 4, 8])
def test_every_gqa_group(group):
    got, want, _ = _run([(96, 130), (0, 64)], group=group)
    _attn_ref.check(got, want)


def test_head_dim_64():
    got, want, _ = _run([(70, 250)], head_dim=64)
    _attn_ref.check(got, want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's numbering")
@pytest.mark.parametrize("h_kv,group", [(40, 1), (8, 4), (3, 2)])
def test_the_spatial_numbering_is_bitwise_the_default_s(h_kv, group):
    """attn_cdna's mapping="spatial" runs the same tiles in another order: every output
    bit the same as the default numbering's, on a ragged batch with cached prefixes."""
    from hk.ops import attn_cdna

    reqs = [(64, 100), (300, 50), (0, 600), (33, 1)]
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, h_kv=h_kv, group=group)
    n_tok, h, d = q.shape
    tiles = pp.work_list([(r, reqs[r][1]) for r in range(len(reqs))], q.device)
    outs = []
    for mapping in attn_cdna.MAPPINGS:
        out = torch.full_like(q, 7.0)
        attn_cdna.paged_kernel_for(d, mapping=mapping)(
            q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
            out.view(n_tok, h * d))
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    _attn_ref.check(outs[1], want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's KV-split prefill")
@pytest.mark.parametrize("reqs,target", [
    ([(4096, 512)], 256), ([(4100, 300), (0, 600), (33, 1), (9000, 70)], 512),
    ([(2048, 1024), (2048, 1024)], 1024), ([(5000, 40)], 64)])
def test_the_kv_split_prefill(reqs, target):
    """gfx950's prefill with its long walks cut over the cached prefix
    (paged_prefill.split_plan): cut tiles merged from their partials, uncut ones
    written straight, a tile whose rows run past its request, a one-token chunk;
    right against fp32 and the same answer twice."""
    from hk.ops import attn_cdna

    group = 4
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, group=group)
    n_tok, h, d = q.shape
    starts = q_start.tolist()
    tiles, mlist, n_merge, n_slots = pp.split_plan(
        [(i, c, n, starts[i]) for i, (c, n) in enumerate(reqs)], h, q.device, target=target,
        min_blocks=4)
    o_part, ml_part = pp.split_scratch(n_slots, h, d, q.device)
    walk, merge = attn_cdna.split_kernels_for(d)
    outs = []
    for _ in range(2):
        out = torch.full_like(q, 7.0)
        walk(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
             out.view(n_tok, h * d), o_part, ml_part)
        if n_merge:
            merge(mlist, o_part, ml_part, out.view(n_tok, h * d))
        outs.append(out)
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    _attn_ref.check(outs[0], want)
    assert torch.equal(outs[0], outs[1])
    if target >= 256:
        assert n_merge > 0


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's KV-split prefill")
def test_an_uncut_split_plan_is_bitwise_the_work_list():
    """With nothing cut, the KV-split kernel over split_plan's entries (in its
    longest-first order) writes every bit the single-tile kernel does."""
    from hk.ops import attn_cdna

    reqs = [(64, 600), (300, 50), (0, 1024), (33, 1)]
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, h_kv=1, group=8)
    n_tok, h, d = q.shape
    starts = q_start.tolist()
    tiles, _, n_merge, n_slots = pp.split_plan(
        [(i, c, n, starts[i]) for i, (c, n) in enumerate(reqs)], h, q.device, target=1)
    assert n_merge == 0
    o_part, ml_part = pp.split_scratch(1, h, d, q.device)
    walk, _ = attn_cdna.split_kernels_for(d)
    a = torch.full_like(q, 7.0)
    walk(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
         a.view(n_tok, h * d), o_part, ml_part)
    b = torch.full_like(q, 7.0)
    pp.kernel_for(d, 8)(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start,
                        pp.work_list([(i, n) for i, (_, n) in enumerate(reqs)], q.device),
                        b.view(n_tok, h * d))
    assert torch.equal(a, b)
