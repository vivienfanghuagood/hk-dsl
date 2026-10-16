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


META = {"torch": "t", "triton": "tr", "flydsl": "f", "hk": "h", "device": "d",
        "kernels": "k1", "hipcc": "hc", "toolchain": "tc", "dialect": "hk-dialect-1"}


def report(tmp_path, name, times, spread=0.002, arch="gfx1100", work=None, **meta):
    """times: {case: {arm: ms}}; work: {case: (work, unit)}."""
    cases = [{"case": c, "arms": {a: {"ms": ms, "spread": spread} for a, ms in arms.items()},
              **(dict(zip(("work", "unit"), work[c])) if work and c in work else {})}
             for c, arms in times.items()]
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"arch": arch, **META, **meta}, "cases": cases}))
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


def _same(tmp_path, name, **meta):
    return report(tmp_path, name, {
        "gemm 4096": {"hk": 1.0, "torch": 1.25, "flydsl": 0.98},
        "rms 1x4096": {"hk": 0.005, "torch": 0.004, "triton": 0.0026}}, **meta)


def test_a_new_hk_version_with_the_same_kernels_is_no_reason_to_refresh(
        baseline, cr, tmp_path, capsys):
    # 0.3.1 -> 0.4.0 changed the dialect's version and no kernel.
    run = _same(tmp_path, "rel.json", hk="0.4.0", dialect="hk-dialect-2")
    assert cr.check(run, baseline) == 0
    assert "note:" not in capsys.readouterr().out


@pytest.mark.parametrize("key, what", [
    ("kernels", "hk's kernels"), ("hipcc", "hipcc's version"),
    ("torch", "torch"), ("triton", "triton"), ("flydsl", "flydsl")])
def test_what_changes_a_measured_kernel_or_opponent_asks_for_a_refresh(
        baseline, cr, tmp_path, capsys, key, what):
    run = _same(tmp_path, "moved.json", **{key: "other"})
    assert cr.check(run, baseline) == 0
    out = capsys.readouterr().out
    assert "note:" in out and what in out and "refresh" in out


def test_a_header_for_another_arch_is_no_reason_to_refresh(baseline, cr, tmp_path, capsys):
    # The include tree's hash moves with any header, CDNA's shim included.
    run = _same(tmp_path, "hdr.json", toolchain="other")
    assert cr.check(run, baseline) == 0
    assert "note:" not in capsys.readouterr().out


def test_a_key_one_side_lacks_is_not_compared(cr, tmp_path, capsys):
    # A baseline from before the scoreboard recorded its kernels.
    old = [report(tmp_path, f"o{i}.json", {"c": {"hk": 1.0, "torch": 1.2}}, kernels=None)
           for i in range(3)]
    b = tmp_path / "old.json"
    assert cr.refresh(old, b) == 0
    assert "kernels" not in json.loads(b.read_text())["meta"]
    assert cr.check(report(tmp_path, "r.json", {"c": {"hk": 1.0, "torch": 1.2}}), b) == 0
    assert "note:" not in capsys.readouterr().out


def test_the_baseline_records_what_its_runs_measured(baseline):
    meta = json.loads(baseline.read_text())["meta"]
    assert meta["kernels"] == "k1" and meta["hipcc"] == "hc" and meta["hk"] == "h"
    assert meta["toolchain"] == "tc" and meta["dialect"] == "hk-dialect-1"


RMS_8192 = (2 * 8192 * 4096 * 2 + 2 * 4096, "byte")     # 134 MB, past the cache


def test_a_time_faster_than_the_card_is_left_out_of_a_refresh(cr, tmp_path, capsys):
    # Chapter 31's seventh run: hk's 8192-row RMSNorm at 97.4 us, 1.38 TB/s.
    rs = [report(tmp_path, f"r{i}.json", {"rms 8192": {"hk": hk, "torch": 0.180}},
                 work={"rms 8192": RMS_8192})
          for i, hk in enumerate((0.0974, 0.1734, 0.1735, 0.1736))]
    b = tmp_path / "b.json"
    assert cr.refresh(rs, b) == 0
    base = json.loads(b.read_text())
    e = base["cases"]["rms 8192"]["torch"]
    assert len(e["runs"]) == 3 and e["ratio"] == pytest.approx(0.180 / 0.1735, rel=1e-3)
    assert base["left_out"] == [f"{rs[0]}: rms 8192 / hk"]
    assert "faster than the card can" in capsys.readouterr().out


def test_a_run_with_a_time_faster_than_the_card_cannot_pass(cr, tmp_path, capsys):
    rs = [report(tmp_path, f"r{i}.json", {"rms 8192": {"hk": 0.1734, "torch": 0.180}},
                 work={"rms 8192": RMS_8192}) for i in range(3)]
    b = tmp_path / "b.json"
    assert cr.refresh(rs, b) == 0
    fast = report(tmp_path, "fast.json", {"rms 8192": {"hk": 0.0974, "torch": 0.180}},
                  work={"rms 8192": RMS_8192})
    assert cr.check(fast, b) == 2
    assert "IMPLAUSIBLE" in capsys.readouterr().out


def test_runs_of_another_statistic_are_refused(baseline, cr, tmp_path, capsys):
    p = _same(tmp_path, "median.json")
    rep = json.loads(p.read_text())
    rep["meta"]["screen"] = {"stat": "median"}
    p.write_text(json.dumps(rep))
    assert cr.check(p, baseline) == 2
    assert "every ratio would move" in capsys.readouterr().out
