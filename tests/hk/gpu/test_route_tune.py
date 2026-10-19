"""`hk.autotune.routes` on the card: a shape measured, written as a record,
and routed by the next table load. Small shapes and a short grid, so this is
the measuring's plumbing, not its numbers (those are the tuner's own runs,
docs/history/35)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.autotune import routes as R  # noqa: E402
from hk.integration import engine, vllm_gemm  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a Radeon"),
    pytest.mark.skipif(current_arch() != "gfx1100",
                       reason="hk's GEMM and decode GEMV are the candidates on gfx1100"),
]

FAST = dict(rounds=1, warmup=0, sets=1, remeasure=0, rotate_bytes=8 << 20)


def test_a_shape_measured_is_an_entry_the_table_reads():
    ent = R.tune_shape(512, 1024, engine=torch.nn.functional.linear,
                       ms=[1, 16, 64, 80, 300], name="t", **FAST)
    assert [p[0] for p in ent["points"]] == [1, 16, 64, 80, 300]
    by_m = {p[0]: p[1] for p in ent["points"]}
    assert "hk_decode" in by_m[16] and "hk_decode" not in by_m[80]
    assert "hk" in by_m[300] and all("engine" in t for t in by_m.values())
    t = engine.RouteTable({(512, 1024): ent})
    for m in (1, 16, 64, 80, 300):
        assert t.lookup(m, 512, 1024) in ("fallback", "hk", "hk_decode")
    assert not ent.get("partial")


def test_a_deadline_cuts_the_grid_and_says_so():
    ent = R.tune_shape(512, 1024, engine=torch.nn.functional.linear,
                       ms=[1, 2, 3], deadline=0.0, **FAST)
    assert ent["points"] == [] and ent["partial"] and ent["above"] == "fallback"


def test_measured_shapes_reach_the_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.delenv("HK_VLLM_ROUTES", raising=False)
    shapes = [("o", 512, 1024, 1), ("down", 1024, 512, 1)]
    rep = R.tune_missing("vllm_linear", "gfx1100", shapes, engine=torch.nn.functional.linear,
                         engine_name="F.linear", budget=None,
                         grid=[1, 8, 16, 64, 80, 256], **FAST)
    assert sorted(s[0] for s in rep.tuned) == ["down", "o"] and not rep.left
    try:
        vllm_gemm.set_routes(None)
        t = vllm_gemm.routes()
        assert {(512, 1024), (1024, 512)} <= t.complete()
        assert t.origin[(512, 1024)] == str(R.user_path("vllm_linear", "gfx1100"))
    finally:
        vllm_gemm.set_routes(None)
