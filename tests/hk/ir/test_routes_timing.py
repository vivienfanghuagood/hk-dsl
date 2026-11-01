"""The routes tuner's timing: context on RDNA, queued on gfx950, an unknown one refused; a record from
before timing was recorded counts as an unqueued burst on RDNA (measured again) and as queued on gfx950;
the context a shape is timed in; the TunableOp state a record carries."""

import pytest

from hk.autotune import routes


def test_the_default_timing_per_arch():
    assert routes.default_timing("gfx1100") == "context"
    assert routes.default_timing("gfx1201") == "context"
    assert routes.default_timing("gfx950") == "queued"


def test_an_unknown_timing_is_refused():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="timing"):
        routes.measure_points(64, 2048, [1], engine=lambda *a: None, timing="wall")


def test_old_records_are_stale_on_rdna_only():
    old = {"meta": {"engine": "vllm"}}
    assert routes._record_timing(old, "gfx1100") == "burst"
    assert routes._record_timing(old, "gfx950") == "queued"
    assert routes._record_timing({"meta": {"timing": "graph"}}, "gfx1100") == "graph"
    assert routes._record_timing({"runs": []}, "gfx1100") is None      # no meta: as with its engine


def test_a_context_is_the_other_layers_from_the_next_one_round():
    order = [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144), (2048, 2048)]
    at = lambda n, k: f"{n}x{k}"                                         # noqa: E731
    assert [(n, k) for n, k, _ in routes.context_of(order, 12288, 2048, at)] == [
        (2048, 6144), (2560, 2048), (2048, 2048)]
    assert [f for _, _, f in routes.context_of(order, 2560, 2048, at)] == [
        "2048x2048", "12288x2048", "2048x6144"]
    # a shape outside the order (the LM head) runs among all of them
    assert len(routes.context_of(order, 73472, 2048, at)) == 4


def test_a_contexts_route_is_the_runs_rule_at_that_m():
    pts = [[1, {"engine": 1.0, "hk_decode": 1.1}], [8, {"engine": 1.0, "hk_decode": 0.8}],
           [9, {"engine": 1.0, "hk_decode": 0.99}]]
    assert routes.route_at(pts, 1) == "engine"
    assert routes.route_at(pts, 8) == "hk_decode"
    assert routes.route_at(pts, 9) == "engine"                          # under the 3% bar
    assert routes.route_at(pts, 2) == "engine"                          # not measured


def test_the_tunableop_state_names_the_results_it_runs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    t = torch.cuda.tunable
    monkeypatch.setattr(t, "is_enabled", lambda: False)
    assert routes.tunableop_state() == "off"
    (tmp_path / "res0.csv").write_text("Validator,PT_VERSION,2.13.0\n")
    monkeypatch.setattr(t, "is_enabled", lambda: True)
    monkeypatch.setattr(t, "tuning_is_enabled", lambda: False)
    monkeypatch.setattr(t, "get_filename", lambda: str(tmp_path / "res%d.csv"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    a = routes.tunableop_state()
    assert a["file"] == "res%d.csv" and not a["tuning"] and len(a["sha"]) == 16
    (tmp_path / "res0.csv").write_text("Validator,PT_VERSION,2.13.0\nGemmTunableOp_BFloat16_TN,x\n")
    assert routes.tunableop_state()["sha"] != a["sha"]
