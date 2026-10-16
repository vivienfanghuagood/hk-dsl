"""The performance gate's arithmetic, on made-up scoreboard reports: a
baseline's tolerance comes from the noise of its runs, a node that is slower
across the board passes, and hk slower against its neighbours fails."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cr():
    path = REPO / "tools" / "hk-bench" / "check_regression.py"
    spec = importlib.util.spec_from_file_location("check_regression_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(tmp_path, name, times, spread=0.002, arch="gfx1100"):
    """times: {case: {arm: ms}}."""
    cases = [{"case": c, "arms": {a: {"ms": ms, "spread": spread} for a, ms in arms.items()}}
             for c, arms in times.items()]
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"arch": arch, "torch": "t", "triton": "tr",
                                      "flydsl": "f", "hk": "h", "device": "d"},
                             "cases": cases}))
    return p


def runs(tmp_path):
    """Three runs: most ratios within 0.4%, Triton's norm within 4%."""
    quiet, loud = (1.0, 1.004, 0.997), (1.0, 1.04, 0.98)
    return [report(tmp_path, f"r{i}.json", {
        "gemm 4096": {"hk": 1.0, "torch": 1.25 * q, "flydsl": 0.98 * q},
        "rms 1x4096": {"hk": 0.005, "torch": 0.004 * q, "triton": 0.0026 * t},
    }) for i, (q, t) in enumerate(zip(quiet, loud))]


@pytest.fixture
def baseline(cr, tmp_path):
    b = tmp_path / "base.json"
    assert cr.refresh(runs(tmp_path), b) == 0
    return b


def test_a_baseline_records_ratio_noise_and_tolerance(baseline, cr):
    base = json.loads(baseline.read_text())
    e = base["cases"]["gemm 4096"]["torch"]
    assert e["ratio"] == pytest.approx(1.25, rel=1e-3)
    assert e["across"] == pytest.approx(0.004, abs=1e-3)
    assert e["within"] == pytest.approx(0.004)
    assert e["tol"] == max(cr.FLOOR, round(cr.MULTIPLIER * max(e["across"], e["within"]), 4))
    assert base["args"] == cr.ARGS and "_tolerance" in base


def test_a_noisier_pair_gets_a_wider_tolerance(baseline, cr):
    base = json.loads(baseline.read_text())
    assert base["cases"]["gemm 4096"]["torch"]["tol"] == cr.FLOOR
    assert base["cases"]["rms 1x4096"]["triton"]["tol"] == pytest.approx(0.04 * cr.MULTIPLIER)


def test_noise_past_the_cap_is_not_gated(cr, tmp_path):
    rs = [report(tmp_path, f"n{i}.json", {"c": {"hk": 1.0, "torch": t}})
          for i, t in enumerate((1.0, 1.1, 0.9))]
    b = tmp_path / "b.json"
    assert cr.refresh(rs, b) == 0
    base = json.loads(b.read_text())
    assert "c" not in base["cases"] and "c / torch" in base["not_gated"]


def test_the_same_numbers_pass(baseline, cr, tmp_path):
    assert cr.check(runs(tmp_path)[1], baseline) == 0


def test_a_node_slower_across_the_board_passes(baseline, cr, tmp_path):
    slow = report(tmp_path, "slow.json", {
        "gemm 4096": {"hk": 1.05, "torch": 1.25 * 1.05, "flydsl": 0.98 * 1.05},
        "rms 1x4096": {"hk": 0.005 * 1.05, "torch": 0.004 * 1.05, "triton": 0.0026 * 1.05}})
    assert cr.check(slow, baseline) == 0


def test_hk_slower_against_its_neighbours_fails(baseline, cr, tmp_path, capsys):
    bad = report(tmp_path, "bad.json", {
        "gemm 4096": {"hk": 1.10, "torch": 1.25, "flydsl": 0.98},
        "rms 1x4096": {"hk": 0.005, "torch": 0.004, "triton": 0.0026}})
    assert cr.check(bad, baseline) == 1
    out = capsys.readouterr().out
    assert "REGRESSION" in out and "gemm 4096 / torch" in out


def test_a_gated_arm_missing_from_the_run_is_not_a_pass(baseline, cr, tmp_path):
    part = report(tmp_path, "part.json", {
        "gemm 4096": {"hk": 1.0, "torch": 1.25},
        "rms 1x4096": {"hk": 0.005, "torch": 0.004, "triton": 0.0026}})
    assert cr.check(part, baseline) == 2


def test_another_arch_is_refused(baseline, cr, tmp_path):
    other = report(tmp_path, "o.json", {"gemm 4096": {"hk": 1.0, "torch": 1.25}}, arch="gfx1201")
    assert cr.check(other, baseline) == 2


def test_fewer_than_three_runs_make_no_baseline(cr, tmp_path):
    assert cr.refresh(runs(tmp_path)[:2], tmp_path / "b.json") == 2
