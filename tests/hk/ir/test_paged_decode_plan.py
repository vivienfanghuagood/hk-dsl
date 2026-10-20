"""The decode the vLLM backend captures: what it plans on, and that it traces.

The split count is the one host decision in a captured decode step, so what
it may depend on is the contract: the batch size and max_model_len, rounded
to a power of two so a batch ramping from 1 to 256 requests compiles a
handful of kernels rather than nineteen.
"""

from __future__ import annotations

import pytest

from hk.integration import vllm_backend
from hk.ops import paged

#: Qwen3-8B: 8 kv heads, max_model_len 40960 in 32-token pages.
HKV, PAGES = 8, 40960 // paged.KV_BLOCK


def test_split_counts_are_powers_of_two_and_few():
    counts = {vllm_backend.decode_splits(n, HKV, PAGES) for n in range(1, 257)}
    assert all(c & (c - 1) == 0 for c in counts), counts
    assert len(counts) <= 6, counts


def test_split_count_falls_as_the_batch_fills_the_machine():
    s = [vllm_backend.decode_splits(n, HKV, PAGES) for n in (1, 4, 8, 16, 32)]
    assert s == sorted(s, reverse=True)
    assert s[0] == paged.MAX_SPLITS


def test_a_short_page_cap_only_caps():
    assert vllm_backend.decode_splits(1, HKV, 4) == 4


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("splits", [1, 8, 32])
def test_the_walk_traces(head_dim, splits):
    paged.fused_kernel(head_dim, splits).trace()


@pytest.mark.parametrize("splits", [2, 8, 32])
def test_the_merge_traces_a_warp_per_fragment(splits):
    k = paged.merge_kernel(128, splits)
    k.trace()
    assert k.warps == 128 // 16


def test_an_unknown_ablation_is_refused():
    with pytest.raises(ValueError, match="unknown ablation"):
        paged.paged_decode_fused_kernel("x", ablate=("nope",))


def test_triton_takes_up_to_eight_and_the_ranges_it_is_given(monkeypatch):
    assert [vllm_backend.triton_takes(n) for n in (1, 8, 9, 16, 24, 64)] == \
        [True, True, False, False, False, False]
    monkeypatch.setattr(vllm_backend, "TRITON_DECODE_ALSO", ((17, 24), (56, 56)))
    assert [vllm_backend.triton_takes(n) for n in (16, 17, 24, 25, 55, 56, 64)] == \
        [False, True, True, False, False, True, False]


def test_the_split_plan_aims_for_256_workgroups():
    # Qwen3-8B's 8 kv heads: 2 splits at 16 requests, 1 from 32 up.
    assert [vllm_backend.decode_splits(n, HKV, PAGES) for n in (4, 8, 16, 24, 32, 64)] == \
        [8, 4, 2, 2, 1, 1]
