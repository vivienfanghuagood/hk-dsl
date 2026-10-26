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
    v_cache = torch.randn_like(k_cache)
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
            kk = torch.cat([k_cache[p, hh // group] for p in pages])[:s].float()
            vv = torch.cat([v_cache[p, hh // group] for p in pages])[:s].float()
            sc = (q[q0:q1, hh].float() @ kk.T) * d ** -0.5
            sc = sc.masked_fill(~keep, float("-inf"))
            out[q0:q1, hh] = torch.softmax(sc, dim=-1) @ vv
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


def _rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


def test_no_prefix_two_tiles_the_second_short():
    got, want, _ = _run([(0, 300)])
    assert _rel(got, want) < 2e-2


def test_a_cached_prefix_and_a_chunk_shorter_than_a_tile():
    got, want, _ = _run([(1000, 77)])
    assert _rel(got, want) < 2e-2


def test_a_tile_that_runs_into_the_next_request():
    # Request 0's only tile covers rows 0..191; rows 100..149 are request 1's
    # and must come out as request 1's answer.
    got, want, _ = _run([(64, 100), (300, 50), (0, 200)])
    assert _rel(got, want) < 2e-2


def test_ragged_batch_with_a_one_token_request():
    got, want, _ = _run([(0, 192), (500, 5), (40, 600), (33, 1)])
    assert _rel(got, want) < 2e-2


def test_rows_of_requests_not_in_the_work_list_are_untouched():
    reqs = [(0, 40), (128, 70), (10, 30)]
    got, want, q_start = _run(reqs, prefill=[1])
    lo, hi = int(q_start[1]), int(q_start[2])
    assert torch.all(got[:lo] == 7.0) and torch.all(got[hi:] == 7.0)
    assert _rel(got[lo:hi], want[lo:hi]) < 2e-2


@pytest.mark.parametrize("group", [1, 4, 8])
def test_every_gqa_group(group):
    got, want, _ = _run([(96, 130), (0, 64)], group=group)
    assert _rel(got, want) < 2e-2


def test_head_dim_64():
    got, want, _ = _run([(70, 250)], head_dim=64)
    assert _rel(got, want) < 2e-2


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
    assert _rel(outs[1], want) < 2e-2
