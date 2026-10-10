"""Paged decode attention against a dense reference.

The kernel reads K and V through a page table, so the reference has to do the
same thing by hand: gather the request's pages into a contiguous tensor and
run SDPA on it. If the two agree for a scrambled page table, a sequence length
that does not fill its last page, and a GQA group narrower than the tile, then
the addressing is right -- which is the only thing this kernel does that the
dense one does not.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402
from hk.ops import paged  # noqa: E402

pytestmark = pytest.mark.gpu

PAGE = paged.KV_BLOCK
TILE = paged.Q_TILE


def _case(seq_lens, head_dim=128, h_kv=2, group=4, n_blocks=64, seed=0):
    """Build a paged cache with a deliberately scrambled page table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(seq_lens)
    k_cache = torch.randn(n_blocks, h_kv, PAGE, head_dim, device="cuda",
                          dtype=torch.bfloat16, generator=g)
    v_cache = torch.randn_like(k_cache)

    max_pages = max((n + PAGE - 1) // PAGE for n in seq_lens)
    # A permutation, so a kernel that quietly used `i` instead of
    # `block_table[req][i]` would be wrong rather than lucky.
    perm = torch.randperm(n_blocks, generator=g, device="cuda")[: r * max_pages]
    table = perm.reshape(r, max_pages).to(torch.int32).contiguous()

    # Rows past the GQA group are padding the kernel computes and nobody
    # reads; zeros keep them finite.
    q = torch.zeros(r, h_kv, TILE, head_dim, device="cuda",
                    dtype=torch.bfloat16)
    q[:, :, :group] = torch.randn(r, h_kv, group, head_dim, device="cuda",
                                  dtype=torch.bfloat16, generator=g)
    lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens, group


def _reference(q, k_cache, v_cache, table, lens, group, scale=None):
    r, h_kv, _, d = q.shape
    scale = scale if scale is not None else d ** -0.5
    out = torch.zeros(r, h_kv, TILE, d, device="cuda", dtype=torch.float32)
    for i in range(r):
        n = int(lens[i])
        pages = table[i, : (n + PAGE - 1) // PAGE].tolist()
        for h in range(h_kv):
            kk = torch.cat([k_cache[p, h] for p in pages])[:n].float()
            vv = torch.cat([v_cache[p, h] for p in pages])[:n].float()
            qq = q[i, h, :group].float()
            out[i, h, :group] = F.scaled_dot_product_attention(
                qq[None, None], kk[None, None], vv[None, None],
                scale=scale)[0, 0]
    return out


def _run(seq_lens, **kw):
    q, k_cache, v_cache, table, lens, group = _case(seq_lens, **kw)
    o = torch.empty_like(q)
    paged.KERNELS[q.shape[-1]](q, k_cache, v_cache, table, lens, o)
    want = _reference(q, k_cache, v_cache, table, lens, group)
    got = o.float()[:, :, :group]
    want = want[:, :, :group]
    rel = (got - want).abs().max().item() / want.abs().max().item()
    return rel


@pytest.mark.parametrize("head_dim", [64, 128])
def test_pages_that_divide(head_dim):
    assert _run([PAGE * 4, PAGE * 2], head_dim=head_dim) < 5e-2


@pytest.mark.parametrize("n", [1, PAGE - 1, PAGE + 1, PAGE * 3 + 7])
def test_a_sequence_that_does_not_fill_its_last_page(n):
    # The one thing a paged kernel can get wrong that a dense one cannot: the
    # slots past the end of the sequence hold whatever the allocator left, and
    # counting them corrupts the softmax sum rather than perturbing it.
    assert _run([n]) < 5e-2


def test_requests_of_different_lengths_in_one_launch():
    assert _run([PAGE * 5, 7, PAGE * 2 + 1, PAGE]) < 5e-2


def test_a_group_narrower_than_the_tile():
    assert _run([PAGE * 3], group=1) < 5e-2
    assert _run([PAGE * 3], group=8) < 5e-2


def test_garbage_past_the_sequence_end_is_not_read():
    """Poison every slot the sequence does not own. The answer must not move.

    This is the sharp version of the tail test: if the mask is off by one, or
    applied after the running max instead of before it, the poisoned values
    dominate the softmax and the error is total rather than subtle.
    """
    q, k_cache, v_cache, table, lens, group = _case([PAGE * 2 + 5])
    want = _reference(q, k_cache, v_cache, table, lens, group)
    o = torch.empty_like(q)
    paged.KERNELS[q.shape[-1]](q, k_cache, v_cache, table, lens, o)
    clean = o.float()[:, :, :group].clone()

    n = int(lens[0])
    last = table[0, (n - 1) // PAGE].item()
    k_cache[last, :, n % PAGE:] = 1e4
    v_cache[last, :, n % PAGE:] = 1e4
    paged.KERNELS[q.shape[-1]](q, k_cache, v_cache, table, lens, o)
    poisoned = o.float()[:, :, :group]

    assert torch.equal(clean, poisoned), "the kernel read past the sequence end"
    rel = (clean - want[:, :, :group]).abs().max().item() / \
        want[:, :, :group].abs().max().item()
    assert rel < 5e-2


# ------------------------------------------------------------ split along KV
#
# The same answer, computed by `splits` workgroups that each see part of the
# sequence and then combine. The combination is the part that can be wrong in
# a way the unsplit kernel cannot: each split's softmax is normalised against
# its own maximum, so merging means rescaling by the difference of maxima, and
# getting that backwards is a plausible-looking answer rather than a crash.


def _run_split(seq_lens, splits, **kw):
    q, k_cache, v_cache, table, lens, group = _case(seq_lens, **kw)
    d = q.shape[-1]
    r, hkv = q.shape[0], q.shape[1]
    o_part = torch.empty(r * splits, hkv, TILE, d, device="cuda",
                         dtype=torch.float32)
    ml = torch.empty(r * splits, hkv, 2, TILE, device="cuda",
                     dtype=torch.float32)
    paged.split_kernel(d, splits)(q, k_cache, v_cache, table, lens,
                                  o_part, ml)
    out = torch.empty(r, hkv, TILE, d, device="cuda", dtype=torch.float32)
    paged.merge_splits(o_part, ml, splits, out)
    want = _reference(q, k_cache, v_cache, table, lens, group)
    got = out[:, :, :group]
    want = want[:, :, :group]
    return (got - want).abs().max().item() / want.abs().max().item()


@pytest.mark.parametrize("splits", [1, 2, 4, 8])
def test_splitting_the_kv_axis_gives_the_same_answer(splits):
    assert _run_split([PAGE * 8], splits) < 5e-2


@pytest.mark.parametrize("splits", [2, 4])
def test_more_splits_than_pages(splits):
    # The planner caps splits at the page count, but the kernel must not rely
    # on that: a split with no pages writes m = -inf and l = 0 and has to drop
    # out of the merge rather than poison it with a NaN.
    assert _run_split([PAGE], splits) < 5e-2


@pytest.mark.parametrize("splits", [2, 4])
def test_split_with_a_ragged_tail(splits):
    assert _run_split([PAGE * 5 + 11], splits) < 5e-2


def test_splits_agree_with_the_unsplit_kernel():
    # Same inputs, two schedules. They are not required to be bit-identical --
    # the splits sum in a different order -- but a disagreement bigger than
    # bf16 rounding means the merge is wrong, not the arithmetic.
    seq = [PAGE * 6 + 3]
    a = _run_split(seq, 1)
    b = _run_split(seq, 4)
    assert abs(a - b) < 1e-2, f"unsplit {a:.4f} vs split {b:.4f}"


def test_mixed_lengths_with_splits():
    assert _run_split([PAGE * 7, PAGE, PAGE * 3 + 5], 4) < 5e-2
