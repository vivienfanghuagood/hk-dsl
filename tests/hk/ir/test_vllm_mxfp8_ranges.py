"""K2's measured M ranges below MIN_M (hk.integration.vllm_mxfp8.RANGES): well formed, and what
`schedule` answers inside, between and around them. No GPU, no torch."""

from __future__ import annotations

import pytest

from hk.integration import vllm_mxfp8 as P


@pytest.mark.parametrize("nk", sorted(P.RANGES))
def test_ranges_are_sorted_disjoint_and_below_the_shapes_min_m(nk):
    rs = P.RANGES[nk]
    assert nk in P.MEASURED
    for lo, hi, kind, sched in rs:
        assert 1 <= lo <= hi < P.MIN_M_NK.get(nk, P.MIN_M) or kind == "8w"
        assert kind in ("4w", "8w") and isinstance(sched, dict)
    for (_, hi, _, _), (lo2, _, _, _) in zip(rs, rs[1:]):
        assert hi < lo2


@pytest.mark.parametrize("nk", sorted(P.RANGES))
def test_schedule_takes_a_range_and_leaves_its_gaps_to_vllm(nk):
    n, k = nk
    floor = P.MIN_M_NK.get(nk, P.MIN_M)
    covered = set()
    for lo, hi, kind, sched in P.RANGES[nk]:
        for m in (lo, (lo + hi) // 2, hi):
            assert P.schedule(m, n, k) == (kind, sched)
        covered.update(range(lo, hi + 1))
    for m in range(1, floor):
        if m not in covered:
            assert P.schedule(m, n, k) is None, m
    assert P.schedule(floor, n, k) is not None


def test_off_means_off_inside_a_range(monkeypatch):
    nk, (lo, _, _, _) = next((nk, rs[0]) for nk, rs in P.RANGES.items())
    monkeypatch.setattr(P, "ENABLED", False)
    assert P.schedule(lo, *nk) is None
