"""warm-routes builds every kernel K2 (hk.integration.vllm_mxfp8) can launch at serving M: for each routed
(N, K), every (kind, schedule, tail recut) that `schedule` + `mxfp8.tail_split` give at some M up to
K2_MAX_M has a warm-up call that takes exactly it. No GPU, no torch.

Regression: warm-routes used to call M = 2048 / 5120 / 8192 / 16384 only, so the four-warp stages=2
schedule (RANGES, M 832-1376 of (5120, 4096)) was never built and every server JIT-compiled it on
its first request in that range."""

from __future__ import annotations

import pytest

from hk.integration import vllm_mxfp8 as P
from hk.integration import warm_routes as W
from hk.ops import mxfp8


def _plan(m, n, k):
    p = P.schedule(m, n, k)
    if p is None:
        return None
    kind, sched = p
    n1 = mxfp8.tail_split(m, n) if kind == "8w" else None
    return kind, tuple(sorted(sched.items())), bool(n1) and n1 < n


@pytest.mark.parametrize("nk", W.k2_shapes(2))
def test_every_plan_at_serving_m_is_warmed(nk):
    n, k = nk
    plans = W.k2_plans(n, k)
    seen = set()
    for m in range(1, W.K2_MAX_M + 1, 7):                 # an independent scan, without the mb cache
        p = _plan(m, n, k)
        if p is not None:
            seen.add(p)
    assert seen <= set(plans), sorted(seen - set(plans))
    for p, m in plans.items():                           # each warm-up call takes the plan it stands for
        assert _plan(m, n, k) == p, (p, m)


@pytest.mark.parametrize("nk", sorted(P.RANGES))
def test_every_measured_range_schedule_is_warmed(nk):
    warmed = {(kind, sched) for kind, sched, _ in W.k2_plans(*nk)}
    for lo, hi, kind, sched in P.RANGES[nk]:
        assert (kind, tuple(sorted(sched.items()))) in warmed, (lo, hi, kind, sched)


def test_the_stages2_schedule_is_warmed():
    warmed = {dict(s).get("stages") for kind, s, _ in W.k2_plans(5120, 4096) if kind == "4w"}
    assert 2 in warmed and 3 in warmed


def test_other_tp_rescales_the_shapes():
    assert W.k2_shapes(4) == sorted({(n // 2, k) for n, k in P.MEASURED})
