"""The routes tuner's timing: graph on RDNA, queued on gfx950, an unknown one refused; a record from
before timing was recorded counts as an unqueued burst on RDNA (measured again) and as queued on gfx950."""

import pytest

from hk.autotune import routes


def test_the_default_timing_per_arch():
    assert routes.default_timing("gfx1100") == "graph"
    assert routes.default_timing("gfx1201") == "graph"
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
