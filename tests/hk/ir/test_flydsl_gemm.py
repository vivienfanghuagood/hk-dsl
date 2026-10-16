"""FlyDSL as an optional path, without FlyDSL: when the path exists, what it
says when it does not, and how route_tune turns timings into its overlay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hk.integration import flydsl_gemm

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(flydsl_gemm, "_STATE", [])
    monkeypatch.setattr(flydsl_gemm, "_VARIANT", {})
    return flydsl_gemm


def test_hk_flydsl_0_turns_the_path_off(fresh, monkeypatch):
    monkeypatch.setenv("HK_FLYDSL", "0")
    assert not fresh.available()
    assert "HK_FLYDSL is off" in fresh.status()
    assert fresh.variant(1024, 4096, 4096) is None


def test_the_path_is_gfx11_only(fresh, monkeypatch):
    import hk.target

    monkeypatch.delenv("HK_FLYDSL", raising=False)
    monkeypatch.setattr(hk.target, "current_arch", lambda: "gfx950")
    assert not fresh.available()
    assert "gfx950" in fresh.status()


def test_a_kernels_package_that_is_not_flydsls_is_named(fresh, monkeypatch, tmp_path):
    # HuggingFace's `kernels` is installed beside SGLang, and would shadow
    # FlyDSL's repository package of the same name.
    (tmp_path / "flydsl").mkdir()
    (tmp_path / "flydsl" / "__init__.py").write_text("")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "__init__.py").write_text("")
    import hk.target

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(hk.target, "current_arch", lambda: "gfx1100")
    monkeypatch.delenv("HK_FLYDSL", raising=False)
    monkeypatch.delenv("FLYDSL_ROOT", raising=False)
    for mod in ("kernels", "flydsl"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    try:
        assert not fresh.available()
        assert "is not FlyDSL's" in fresh.status()
    finally:
        for mod in [m for m in sys.modules if m == "kernels" or m.startswith("kernels.")]:
            del sys.modules[mod]


def test_tile_names():
    assert flydsl_gemm.tile_name((4, 4, 2, 2, 2)) == "128x128x32w4"
    assert flydsl_gemm.tile_name((4, 4, 2, 4, 4)) == "256x256x32w16"
    assert flydsl_gemm.tile_name((2, 2, 4, 1, 1)) == "32x32x64w1"


@pytest.fixture(scope="module")
def rt():
    path = REPO / "tools" / "hk-bench" / "route_tune.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("route_tune_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _pt(m, engine, hk=None, fly=None, tile=None):
    t = {"engine": engine}
    if hk is not None:
        t["hk"] = hk
    if fly is None:
        return [m, t]
    t["flydsl"] = fly
    return [m, t, {"flydsl": tile}]


A, B = "128x128x32w4", "32x32x64w1"


def test_flydsl_never_enters_the_runs(rt):
    pts = [_pt(1024, 1.0, hk=0.9, fly=0.5, tile=A), _pt(2048, 2.0, hk=1.8, fly=1.0, tile=A)]
    rs, above = rt.runs(pts, 0.03)
    assert rs == [[1024, 2048, "hk"]] and above == "hk"


def test_the_overlay_needs_both_margins(rt):
    pts = [
        _pt(1024, 1.00, hk=0.90, fly=0.87, tile=A),   # beats hk by 3.4%: win
        _pt(1088, 1.00, hk=0.90, fly=0.89, tile=A),   # beats hk by 1.1%: no
        _pt(1152, 1.00, hk=0.99, fly=0.975, tile=A),  # runs say engine; beats it by 2.6%: no
        _pt(1216, 1.00, hk=0.99, fly=0.95, tile=A),   # beats the engine by 5%: win
    ]
    assert rt.prefer(pts, "flydsl", 0.03, 0.02) == [[1024, 1024, A], [1216, None, A]]


def test_the_overlay_merges_per_tile_across_other_tiles_and_gaps(rt):
    pts = [
        _pt(96, 1.0, hk=0.9, fly=0.8, tile=B),
        _pt(112, 1.0, hk=0.9),                         # FlyDSL takes no 112
        _pt(128, 1.0, hk=0.9, fly=0.8, tile=A),
        _pt(160, 1.0, hk=0.9, fly=0.8, tile=B),
        _pt(192, 1.0, hk=0.9, fly=0.95, tile=B),       # B loses here
        _pt(256, 1.0, hk=0.9, fly=0.8, tile=A),
    ]
    assert rt.prefer(pts, "flydsl", 0.03, 0.02) == [
        [96, 160, B], [128, None, A]]


def test_no_flydsl_points_no_overlay(rt):
    assert rt.prefer([_pt(1024, 1.0, hk=0.9)], "flydsl", 0.03, 0.02) == []


def test_a_points_time_is_the_median_set_and_its_noise_the_wider_spread(rt):
    sets = [{"engine": [1.00, 1.01, 1.02], "hk": [0.90, 0.95, 0.91]},
            {"engine": [1.02, 1.02, 1.03], "hk": [0.92, 0.92, 0.93]},
            {"engine": [1.01, 1.03, 1.01], "hk": [0.91, 0.94, 0.92]}]
    t, noise = rt.estimate(sets)
    assert t == {"engine": 1.01, "hk": 0.91}
    assert noise["engine"] == pytest.approx(0.0099, abs=1e-4)          # sets: (1.02-1.00)/2/1.01
    assert noise["hk"] == pytest.approx(0.0110, abs=1e-4)              # rounds: 0.92/0.91 - 1


def test_a_margin_inside_its_noise_is_a_close_call(rt):
    t = {"engine": 1.0, "hk": 1 / 1.035}              # 3.5% ahead of the engine
    assert rt.close_call(t, {"engine": 0.002, "hk": 0.002}, 0.03, 0.02)
    assert not rt.close_call(t, {"engine": 0.0005, "hk": 0.0005}, 0.03, 0.02)
    t = {"engine": 1.0, "hk": 0.7, "hk_decode": 0.705}   # hk's two paths within 0.7%
    assert rt.close_call(t, {"engine": 0.002, "hk": 0.002, "hk_decode": 0.002}, 0.03, 0.02)


def test_the_noise_bar_is_off_by_default_and_raises_the_gain_when_on(rt):
    pt = [1024, {"engine": 1.0, "hk": 0.9, "flydsl": 0.87}, {"flydsl": A},
          {"engine": 0.01, "hk": 0.01, "flydsl": 0.01}, 3]
    assert rt.prefer([pt], "flydsl", 0.03, 0.02) == [[1024, None, A]]           # 3.4% > 2%
    assert rt.prefer([pt], "flydsl", 0.03, 0.02, k=2.0) == []                   # < 2 x 2%
    assert rt.runs([pt], 0.03, k=2.0)[0] == [[1024, 1024, "hk"]]                # 11% > 4%


def test_a_confirmed_overlay_keeps_only_what_both_runs_took(rt):
    pts = [_pt(1024, 1.0, hk=0.9, fly=0.8, tile=A), _pt(1088, 1.0, hk=0.9, fly=0.8, tile=A)]
    assert rt.prefer(pts, "flydsl", 0.03, 0.02) == [[1024, None, A]]
    assert rt.prefer(pts, "flydsl", 0.03, 0.02, confirm={1088}) == [[1088, None, A]]
    assert rt.prefer(pts, "flydsl", 0.03, 0.02, confirm=set()) == []


def test_two_tables_decisions_compare_point_by_point(rt, tmp_path, capsys):
    def table(fly):
        pts = [_pt(1024, 1.0, hk=0.9, fly=fly, tile=A), _pt(2048, 2.0, hk=1.8)]
        rs, above = rt.runs(pts, 0.03)
        return {"min_gain": 0.03, "shapes": {"64x32": {
            "name": "o", "runs": rs, "above": above, "points": pts,
            "prefer": {"flydsl": rt.prefer(pts, "flydsl", 0.03, 0.02)}}}}

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(rt.dump(table(0.80)))
    b.write_text(rt.dump(table(0.89)))
    assert rt.decisions(json.loads(a.read_text()))[("64x32", 1024)] == ("hk", True)
    assert rt.compare(a, b) == 0
    assert "1 of 2 points decided differently (0 in the runs, 1 in FlyDSL's overlay)" in \
        capsys.readouterr().out


def test_the_dump_is_a_table_the_engine_reads(rt, tmp_path):
    from hk.integration import engine

    pts = [_pt(1024, 1.0, hk=0.9, fly=0.8, tile=A)]
    rs, above = rt.runs(pts, 0.03)
    table = {"arch": "gfx1100", "shapes": {"64x32": {
        "name": "o", "runs": rs, "above": above, "points": pts,
        "prefer": {"flydsl": rt.prefer(pts, "flydsl", 0.03, 0.02)}}}}
    p = tmp_path / "t.json"
    p.write_text(rt.dump(table))
    t = engine.RouteTable.from_json(p)
    assert t.lookup(1024, 64, 32) == "hk"
    assert t.preferred(4096, 64, 32) == [("flydsl", A)]
