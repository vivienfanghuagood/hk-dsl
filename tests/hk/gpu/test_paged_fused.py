"""The decode the vLLM backend runs, against a dense reference.

The walk and its merge (hk.ops.paged.decode) read the engine's own
(tokens, kv_heads, group, head_dim) query and write its output, through a
scrambled page table: ragged lengths, lengths that do not fill their last
page, the zero a CUDA graph pads a batch with, more splits than pages, and
one captured graph replayed over lengths it was not captured with -- which is
the contract the backend's UNIFORM_SINGLE_TOKEN_DECODE claim rests on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import _attn_ref  # noqa: E402

from hk.ops import paged  # noqa: E402

pytestmark = pytest.mark.gpu

PAGE = paged.KV_BLOCK


def _case(seq_lens, head_dim=128, h_kv=2, group=4, max_len=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(seq_lens)
    max_pages = max(1, -(-(max_len or max(seq_lens)) // PAGE))
    n_blocks = r * max_pages + 3
    k_cache = torch.randn(n_blocks, h_kv, PAGE, head_dim, device="cuda",
                          dtype=torch.bfloat16, generator=g)
    v_cache = torch.randn(k_cache.shape, device=k_cache.device, dtype=k_cache.dtype, generator=g)
    perm = torch.randperm(n_blocks, generator=g, device="cuda")[: r * max_pages]
    table = perm.reshape(r, max_pages).to(torch.int32).contiguous()
    q = torch.randn(r, h_kv, group, head_dim, device="cuda",
                    dtype=torch.bfloat16, generator=g)
    lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens


def _reference(q, k_cache, v_cache, table, lens):
    """Exact decode attention in float64 (_attn_ref)."""
    r, h_kv, group, d = q.shape
    out = torch.zeros(r, h_kv, group, d, device="cuda", dtype=torch.float32)
    for i in range(r):
        n = int(lens[i])
        if n == 0:
            continue
        pages = table[i, : -(-n // PAGE)].tolist()
        for h in range(h_kv):
            kk = torch.cat([k_cache[p, h] for p in pages])[:n]
            vv = torch.cat([v_cache[p, h] for p in pages])[:n]
            out[i, h] = _attn_ref.softmax_v64(q[i, h], kk, vv, d ** -0.5)
    return out


def _decode(q, k_cache, v_cache, table, lens, splits, walk=None, merge=None, ppstep=1):
    r, h_kv, group, d = q.shape
    out = torch.empty_like(q)
    scratch = paged.fused_scratch(r * splits, h_kv, d, q.device, group=group)
    paged.decode(walk or paged.fused_kernel(d, splits, ppstep=ppstep),
                 merge or paged.merge_kernel(d, splits), q, k_cache, v_cache,
                 table, lens, out, scratch, splits)
    return out


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("splits", [1, 2, 8, 32])
def test_the_decode_is_right_at_every_split(splits, head_dim):
    case = _case([1, 31, 32, 33, 200, 1000], head_dim=head_dim)
    got = _decode(*case, splits)
    _attn_ref.check(got, _reference(*case))


@pytest.mark.parametrize("splits", [64, 256])
def test_many_splits_fold_on_several_warps(splits):
    """gfx950's plan reaches 256 splits and more at one request on few kv heads
    (paged_cdna.target_workgroups); its merge folds them on up to 8 warps
    (paged_cdna.merge_warps). Long and short sequences, a padded one, a
    sequence with fewer pages than warps' rounds, short ones that take the
    lower page floor (paged_cdna.floor_pages) over two to six rounds, and the
    deterministic answer twice."""
    case = _case([0, 4100, 33, 20000, 1100, 2000, 3000], h_kv=1, group=8)
    got = _decode(*case, splits)
    assert torch.count_nonzero(got[0]) == 0
    _attn_ref.check(got, _reference(*case))
    assert torch.equal(got, _decode(*case, splits))


@pytest.mark.parametrize("splits", [1, 2, 8, 64])
@pytest.mark.parametrize("lens,h_kv,group", [
    ([0, 1, 33, 95, 1000, 4100], 2, 4),
    ([128], 1, 4),
    ([64, 96, 128, 160, 192, 4096, 8191], 1, 8),
])
def test_two_pages_a_step(splits, lens, h_kv, group):
    """gfx950's walk at two pages a step (paged_cdna.pages_per_step): odd and even
    page counts in a split, a last page the sequence does not fill, a one-token
    sequence, a padded request, and the deterministic answer twice. A build
    without hk-mma-fence computed some scores wrong: 22-53 ulps off in its
    worst row, which _attn_ref.check catches, while 2e-2 of the largest value,
    the bound these tests used to ask, did not."""
    if not paged._native_cdna():
        pytest.skip("two pages a step is gfx950's walk")
    case = _case(lens, h_kv=h_kv, group=group)
    want = _reference(*case)
    got = _decode(*case, splits, ppstep=2)
    if lens[0] == 0:
        assert torch.count_nonzero(got[0]) == 0
    _attn_ref.check(got, want)
    assert torch.equal(got, _decode(*case, splits, ppstep=2))


def test_a_padded_request_reads_nothing_and_answers_zero():
    case = _case([0, 100, 0, 37])
    got = _decode(*case, 8)
    assert torch.count_nonzero(got[0]) == 0 and torch.count_nonzero(got[2]) == 0
    _attn_ref.check(got, _reference(*case))


@pytest.mark.parametrize("group", [1, 4, 8])
def test_groups_of_every_width_the_tile_holds(group):
    case = _case([77, 513], group=group)
    _attn_ref.check(_decode(*case, 8), _reference(*case))


def test_more_splits_than_pages():
    case = _case([5, 40])
    _attn_ref.check(_decode(*case, 32), _reference(*case))


def test_a_captured_decode_replays_over_new_lengths():
    from hk.runtime.torch_ext import torch_op

    splits = 8
    walk = torch_op(paged.fused_kernel(128, splits))
    merge = torch_op(paged.merge_kernel(128, splits))
    q, k_cache, v_cache, table, lens = _case([300, 1000, 17], max_len=1000)
    r, h_kv, group, d = q.shape
    out = torch.empty_like(q)
    scratch = paged.fused_scratch(r * splits, h_kv, d, q.device, group=group)

    def step():
        paged.decode(walk, merge, q, k_cache, v_cache, table, lens, out,
                     scratch, splits)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    graph.replay()
    torch.cuda.synchronize()
    _attn_ref.check(out, _reference(q, k_cache, v_cache, table, lens))

    # New lengths and a new page order, written in place: a replay runs no
    # Python, so only what the kernels read on the device may change.
    lens.copy_(torch.tensor([1000, 5, 600], device="cuda", dtype=torch.int32))
    table.copy_(table.flip(0))
    graph.replay()
    torch.cuda.synchronize()
    _attn_ref.check(out, _reference(q, k_cache, v_cache, table, lens))
