"""Which of a scoreboard arm's rounds to believe (tools/hk-bench/_rounds.py),
on made-up rounds: a round that lost kernel records, one faster than the card
can go, and one no other round comes near are rejected; the rounds of a
noisy kernel, or a quiet one's ordinary best, are not."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def R():
    path = REPO / "tools" / "hk-bench" / "_rounds.py"
    spec = importlib.util.spec_from_file_location("rounds_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: RMSNorm at 8192 and 2048 rows of 4096: 134 MB, past the 96 MB cache; 33.5 MB, inside.
RMS_8192 = 2 * 8192 * 4096 * 2 + 2 * 4096
RMS_2048 = 2 * 2048 * 4096 * 2 + 2 * 4096
STEADY = [0.1734, 0.1736, 0.1735, 0.1737, 0.1734, 0.1736]


def test_chapter_31s_round_is_past_the_ceiling(R):
    floor = R.floor_ms(RMS_8192, "byte", RMS_8192)
    kept, rejected = R.screen([0.0974] + STEADY, [50] * 7, floor)
    assert 0 not in kept and rejected[0].startswith("ceiling")
    assert kept == list(range(1, 7))


def test_and_a_lone_round_without_a_ceiling(R):
    kept, rejected = R.screen([0.0974] + STEADY)
    assert rejected[0].startswith("lone") and len(kept) == 6


def test_a_cache_resident_call_may_beat_hbm(R):
    # 1.82 TB/s: what hk's 2048-row RMSNorm does every run.
    floor = R.floor_ms(RMS_2048, "byte", RMS_2048)
    t = RMS_2048 / 1.82e12 * 1e3
    kept, rejected = R.screen([t * f for f in (1.0, 1.001, 1.002, 1.0, 1.003)], None, floor)
    assert not rejected and len(kept) == 5


def test_the_ceiling_is_the_footprints(R):
    # The in-place fused add moves twice what it touches: 67 MB through a
    # 33.5 MB footprint stays inside the cache.
    assert R.ceiling_rate("byte", 33.5e6) > R.ceiling_rate("byte", 134e6)
    assert R.ceiling_rate("byte", 134e6) == pytest.approx(R.HBM_BYTES_S * R.MARGIN)
    assert R.ceiling_rate("flop", None) == pytest.approx(R.PEAK_FLOPS * R.MARGIN)
    assert R.footprint_of({"case": "fused_add_rms_norm 2048x4096", "work": 67e6}) == 33.5e6


def test_a_round_that_lost_records_is_rejected(R):
    kept, rejected = R.screen([0.10, 0.17, 0.17, 0.171, 0.17], [28, 50, 50, 50, 50])
    assert rejected == {0: "records: 28 kernel records, the arm's other rounds 50"}


def test_records_without_a_majority_say_nothing(R):
    kept, rejected = R.screen([1.0, 1.01, 1.02, 1.0], [50, 51, 52, 53])
    assert not rejected


def test_a_noisy_kernels_rounds_are_kept(R):
    # hk's one-row RMSNorm in one process: 2.7 to 4.6 us.
    ts = [3.66, 3.11, 3.86, 4.65, 4.28, 2.73, 2.90]
    kept, rejected = R.screen(ts, [50] * 7)
    assert not rejected


def test_a_quiet_kernels_ordinary_best_is_kept(R):
    ts = [1.000, 1.025, 1.026, 1.025, 1.027]       # 2.5% ahead, under GAP_FLOOR
    assert not R.screen(ts)[1]


def test_lone_rejects_at_most_until_half_are_left(R):
    ts = [0.5, 0.6, 0.7, 1.0, 1.0, 1.0, 1.0]
    kept, rejected = R.screen(ts)
    assert len(kept) * 2 > len(ts) and len(rejected) <= 3


def test_median_and_best(R):
    ts = [1.0, 1.2, 1.1, 1.05, 0.0]
    kept, _ = R.screen(ts)
    assert 4 not in kept
    med, spread, best = R.summarize(ts, kept, "median")
    assert med == pytest.approx(1.075) and best == 1.0
    assert spread == pytest.approx(0.0465, abs=1e-3)
    b, s, _ = R.summarize(ts, kept, "best")
    assert b == 1.0 and s == pytest.approx(0.05)
    assert R.summarize(ts, [], "median") == (None, 0.0, None)


def test_audit_reads_both_kinds_of_report(R):
    old = {"cases": [{"case": "rms_norm 8192x4096", "work": RMS_8192, "unit": "byte",
                      "arms": {"hk": {"ms": 0.0974, "spread": 0.78},
                               "torch": {"ms": 0.177, "spread": 0.01}}}]}
    lines = R.audit(old)
    assert len(lines) == 1 and "rms_norm 8192x4096 / hk" in lines[0] and "ceiling" in lines[0]
    new = {"cases": [{"case": "c", "work": 1, "unit": "byte", "arms": {"hk": {
        "ms": 0.17, "rounds": [0.0974, 0.17, 0.17], "rejected": {"0": "lone: ..."}}}}]}
    assert R.audit(new) == ["c / hk: round 1 of 3, 0.0974 ms -- lone: ..."]
