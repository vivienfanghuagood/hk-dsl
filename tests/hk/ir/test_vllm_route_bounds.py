"""Where hk's DeepSeek-V4.1 vLLM routes start (no GPU, no vLLM): the M each takes from, per
(N, K) for K2, as measured on an MI355X (vllm_mxfp8.MIN_M_NK, vllm_moe_a8w4.MIN_M)."""

from __future__ import annotations

import os

import pytest

from hk.integration import vllm_mxfp8 as K2


@pytest.mark.parametrize("n,k,first", [
    # the TP2 projections from MIN_M (2048)
    (1792, 5120, 2048), (2304, 5120, 2048), (5120, 1152, 2048), (5120, 4096, 2048),
    (16384, 1280, 2048),
    # their own: the indexer's wq_b from 2825, engram's wkv from 1024
    (4096, 1280, 2825), (12800, 6144, 1024)])
def test_each_measured_shape_starts_where_it_was_measured_faster(n, k, first):
    # below MIN_M only the measured ranges (vllm_mxfp8.RANGES; test_vllm_mxfp8_ranges.py)
    ranged = [(kind, sched) for lo, hi, kind, sched in K2.RANGES.get((n, k), ())
              if lo <= first - 1 <= hi]
    assert K2.schedule(first - 1, n, k) == (ranged[0] if ranged else None)
    assert K2.schedule(first, n, k) is not None
    assert K2.schedule(first + 1, n, k) is not None            # ragged M too
    assert K2.schedule(16384, n, k) is not None


@pytest.mark.parametrize("m,n,k", [(2825, 4096, 1280), (3073, 4096, 1280),
                                   (1024, 12800, 6144), (2047, 12800, 6144)])
def test_the_new_shapes_take_the_schedule_they_were_measured_with(m, n, k):
    """The eight-warp kernel on whole 256 x 256 blocks: what the scoreboard timed."""
    assert K2.schedule(m, n, k) == ("8w", {})


def test_an_unmeasured_shape_needs_any(monkeypatch):
    assert K2.schedule(4096, 4096, 4096) is None
    monkeypatch.setattr(K2, "ANY", True)
    assert K2.schedule(2048, 4096, 4096) == ("8w", {})
    assert K2.schedule(2047, 4096, 4096) is None


def test_the_route_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr(K2, "ENABLED", False)
    assert K2.schedule(16384, 12800, 6144) is None


@pytest.mark.skipif("HK_MOE_A8W4_MIN_M" in os.environ, reason="MIN_M set by the environment")
def test_k1_starts_at_2560():
    from hk.integration import vllm_moe_a8w4 as K1

    assert K1.MIN_M == 2560
