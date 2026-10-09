"""Schedule search: what it rejects for free, and what order it measures in.

No GPU and no hipcc here. The build stage is `runtime.warm.warm_cache`, which
`build_all` imports at call time, so these tests replace it with a function
that answers from a table -- which is also the point of the design: the half of
autotune that decides what is worth measuring runs on a laptop.
"""

from __future__ import annotations

import json

import pytest

import hk
from hk.autotune import Row, Tuner, label_of, space
from hk.runtime.resources import ResourceError
from hk.runtime.warm import WarmResult


# A kernel stand-in. The tuner never looks inside one -- it hands it to
# warm_cache and to the benchmark -- so a schedule is all it needs to be.
class FakeKernel:
    def __init__(self, **sched):
        self.sched = sched

    def __repr__(self):
        return f"<FakeKernel {label_of(self.sched)}>"


class FakeResources:
    def __init__(self, vgpr=128, occ=6, scratch=0):
        self.vgpr, self.occ_by_reg, self.occupancy, self.scratch = \
            vgpr, occ, occ, scratch


class FakeBuild:
    def __init__(self, **kw):
        self.kernels = [FakeResources(**kw)]


DEFAULT = dict(a=1, b=1)


def _tuner(monkeypatch, tmp_path, schedules, *, make=None, outcomes=None,
           name="t"):
    """A registered-nowhere tuner whose builds come from `outcomes`.

    `outcomes` maps label -> exception to raise instead of building. Everything
    else builds clean.
    """
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    hk.autotune.forget_records()

    def default_make(**s):
        return FakeKernel(**s)

    def fake_warm(jobs, workers=None, verbose=False):
        res = WarmResult()
        for label, kernel, _ in jobs:
            e = (outcomes or {}).get(label)
            if e is not None:
                res.errors[label] = e
            else:
                res.built[label] = FakeBuild()
        return res

    monkeypatch.setattr(hk.runtime.warm, "warm_cache", fake_warm)
    return Tuner(name, make or default_make, schedules, DEFAULT, register=False)


# ---------------------------------------------------------------- the space


def test_space_is_a_product_in_the_order_written():
    s = space(x=[1, 2], y=["a", "b"])
    assert s == [{"x": 1, "y": "a"}, {"x": 1, "y": "b"},
                 {"x": 2, "y": "a"}, {"x": 2, "y": "b"}]


def test_space_takes_a_constraint():
    s = space(lambda d: d["x"] != d["y"], x=[1, 2], y=[1, 2])
    assert s == [{"x": 1, "y": 2}, {"x": 2, "y": 1}]


def test_a_label_is_stable_and_filename_safe():
    assert label_of({"block_m": 128, "wgm": 8}) == "block_m128_wgm8"


def test_an_empty_space_is_refused():
    with pytest.raises(ValueError, match="empty space"):
        Tuner("x", FakeKernel, [], DEFAULT, register=False)


def test_an_axis_the_default_does_not_have_is_refused():
    # Otherwise a record could name a knob the factory has no parameter for and
    # the failure would land inside a serving process.
    with pytest.raises(ValueError, match="not in the default schedule"):
        Tuner("x", FakeKernel, [{"c": 3}], DEFAULT, register=False)


# ---------------------------------------------------------------- stage 1


def test_the_factory_refusing_a_schedule_costs_no_compile(monkeypatch, tmp_path):
    built = []

    def make(**s):
        if s["a"] == 2:
            raise ValueError("LDS over 64 KB")
        built.append(s)
        return FakeKernel(**s)

    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2, 3]), make=make)
    _, rows = t.build_all()
    by = {r.label: r for r in rows}
    assert by["a2"].status == "invalid"
    assert "64 KB" in by["a2"].reason
    assert [s["a"] for s in built] == [1, 3]


def test_a_spilling_candidate_is_rejected_and_says_so(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]),
               outcomes={"a2": ResourceError("ScratchSize=48; the kernel spills")})
    built, rows = t.build_all()
    by = {r.label: r for r in rows}
    assert by["a2"].status == "rejected"
    assert "spills" in by["a2"].reason
    assert "a2" not in built and "a1" in built


def test_a_broken_candidate_is_an_error_not_a_rejection(monkeypatch, tmp_path):
    # The difference matters to whoever reads the table: `rejected` is the gate
    # doing its job, `error` is a bug that should not be in a space.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]),
               outcomes={"a2": RuntimeError("hipcc segfaulted")})
    _, rows = t.build_all()
    assert {r.label: r.status for r in rows} == {"a1": "ok", "a2": "error"}


def test_a_surviving_row_carries_its_resources(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1]))
    _, rows = t.build_all()
    assert (rows[0].vgpr, rows[0].occupancy, rows[0].scratch) == (128, 6, 0)


def test_the_factory_sees_the_whole_schedule_not_just_the_axes(monkeypatch, tmp_path):
    # A space names the axes that vary; the factory is called with every knob,
    # so a kernel whose signature has no default for one still builds.
    seen = []
    t = _tuner(monkeypatch, tmp_path, space(a=[7]),
               make=lambda **s: (seen.append(s), FakeKernel(**s))[1])
    t.build_all()
    assert seen == [{"a": 7, "b": 1}]


# ---------------------------------------------------------------- stage 2


def test_the_winner_is_the_fastest_and_is_recorded(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2, 3]))
    res = t.tune(lambda k: {1: 3.0, 2: 1.0, 3: 2.0}[k.sched["a"]], key="4096")
    assert res.best.label == "a2"
    rec = json.loads((tmp_path / "t.json").read_text())["4096"]
    assert rec["schedule"] == {"a": 2} and rec["ms"] == pytest.approx(1.0)
    assert rec["candidates"] == 3 and rec["measured"] == 3
    assert "toolchain" in rec


def test_the_order_reverses_every_round(monkeypatch, tmp_path):
    # Cross-process A/B is invalid on this chip, so every candidate is timed in
    # one process -- and a fixed order inside that process hands the clock
    # drift to whoever ran last.
    seen = []
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2, 3]))
    t.tune(lambda k: seen.append(k.sched["a"]) or 1.0, rounds=3)
    assert seen == [1, 2, 3, 3, 2, 1, 1, 2, 3]


def test_a_candidate_is_scored_by_its_minimum(monkeypatch, tmp_path):
    # Every source of noise on a shared GPU adds time; none of it subtracts.
    times = {1: [5.0, 2.0], 2: [3.0, 3.0]}
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    res = t.tune(lambda k: times[k.sched["a"]].pop(0), rounds=2)
    assert res.best.label == "a1"
    assert {r.label: r.ms for r in res.rows} == {"a1": 2.0, "a2": 3.0}


def test_one_candidate_blowing_up_does_not_lose_the_table(monkeypatch, tmp_path):
    def bench(k):
        if k.sched["a"] == 2:
            raise RuntimeError("out of memory")
        return 1.0

    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2, 3]))
    res = t.tune(bench, rounds=2)
    by = {r.label: r.status for r in res.rows}
    assert by == {"a1": "ok", "a2": "error", "a3": "ok"}
    assert res.best is not None


def test_nothing_surviving_the_gate_writes_no_record(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1]),
               outcomes={"a1": ResourceError("spills")})
    res = t.tune(lambda k: 1.0, key="k")
    assert res.best is None
    assert not (tmp_path / "t.json").exists()
    assert "nothing survived" in res.report()


def test_tune_without_a_benchmark_says_so(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1]))
    with pytest.raises(ValueError, match="no benchmark"):
        t.tune()


# ---------------------------------------------------------------- records


def test_with_no_record_a_key_gets_the_default(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    assert t.schedule_for("anything") == DEFAULT


def test_a_record_is_merged_onto_the_default(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    t.tune(lambda k: 10.0 / k.sched["a"], key="big")   # a=2 is 2x faster
    assert t.schedule_for("big") == {"a": 2, "b": 1}
    assert t.schedule_for("small") == DEFAULT


def test_a_record_naming_a_knob_that_no_longer_exists_degrades(monkeypatch, tmp_path):
    # An old record should fall back toward the default, not raise TypeError
    # inside a serving process.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2, "gone": 9}})
    assert t.schedule_for("k") == {"a": 2, "b": 1}


def test_a_corrupt_record_is_not_a_failed_launch(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    (tmp_path / "t.json").write_text("{not json")
    hk.autotune.forget_records()
    assert t.schedule_for("k") == DEFAULT


def test_the_env_switch_pins_the_default(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    t.tune(lambda k: 10.0 / k.sched["a"], key="big")   # a=2 is 2x faster
    monkeypatch.setenv("HK_AUTOTUNE", "0")
    hk.autotune.forget_records()
    assert t.schedule_for("big") == DEFAULT


def test_a_user_record_beats_a_shipped_one(monkeypatch, tmp_path):
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "t.json").write_text(json.dumps(
        {"big": {"schedule": {"a": 1}}, "small": {"schedule": {"a": 1}}}))
    monkeypatch.setattr(hk.autotune, "shipped_dir", lambda: shipped)
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "big", {"schedule": {"a": 2}})
    assert t.schedule_for("big")["a"] == 2     # the user measured this machine
    assert t.schedule_for("small")["a"] == 1   # and said nothing about this one


def test_writing_a_record_is_visible_without_a_reload(monkeypatch, tmp_path):
    # schedule_for is on the launch path and caches; the generation counter is
    # what keeps the cache from outliving the record it was built from.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    assert t.schedule_for("k") == DEFAULT
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}})
    assert t.schedule_for("k") == {"a": 2, "b": 1}


def test_the_kernel_for_a_schedule_is_built_once(monkeypatch, tmp_path):
    n = []
    t = _tuner(monkeypatch, tmp_path, space(a=[1]),
               make=lambda **s: (n.append(s), FakeKernel(**s))[1])
    assert t.kernel("k") is t.kernel("k") is t.kernel("other")
    assert len(n) == 1


# ---------------------------------------------------------------- AOT


def test_aot_covers_the_default_as_well_as_the_records(monkeypatch, tmp_path):
    # The default is what a key with no record gets, so leaving it out would
    # move the JIT cost onto exactly the shapes nobody measured.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "big", {"schedule": {"a": 2}})
    scheds = [k.sched for _, k, _ in t.aot_jobs()]
    assert scheds == [{"a": 1, "b": 1}, {"a": 2, "b": 1}]


def test_aot_does_not_build_the_same_schedule_twice(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "x", {"schedule": {"a": 1}})
    hk.autotune.save_record("t", "y", {"schedule": {"a": 1}})
    assert len(t.aot_jobs()) == 1


# ---------------------------------------------------------------- the ops


def test_gemm_ships_a_space_and_a_default_that_agree():
    from hk.ops import gemm

    assert gemm.TUNER.default == gemm.DEFAULT
    assert (gemm.BLOCK_M, gemm.BLOCK_N, gemm.K_STEP) == (
        gemm.DEFAULT["block_m"], gemm.DEFAULT["block_n"], gemm.DEFAULT["k_step"])
    assert len(gemm.SPACE) == 48


def test_gemm_will_not_let_a_record_break_a_shape_that_worked(monkeypatch, tmp_path):
    from hk.ops import gemm

    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    hk.autotune.forget_records()
    try:
        # K=4160 is a multiple of 64 and not of 128, so the default tiles it.
        # A record asking for a deeper K-tile -- a schedule from a later space,
        # or one measured against a bucket whose other members all had K a
        # multiple of 128 -- must not turn that shape into an exception.
        hk.autotune.save_record("gemm_bf16", "4096x4096x4160",
                                {"schedule": {"k_step": 128}})
        assert gemm.schedule_for(4096, 4096, 4160) == gemm.DEFAULT
        # The control: a record the shape *can* take is taken.
        hk.autotune.save_record("gemm_bf16", "4096x4096x4160",
                                {"schedule": {"k_step": 32}})
        assert gemm.schedule_for(4096, 4096, 4160)["k_step"] == 32
    finally:
        hk.autotune.forget_records()


def test_gemm_buckets_m_so_the_record_can_hit():
    from hk.ops import gemm

    assert gemm._bucket(1) == 1 and gemm._bucket(4096) == 4096
    assert gemm._bucket(4097) == 8192


def test_attention_buckets_n_the_same_way():
    from hk.ops import attn

    assert attn._tune_key(1024) == "n4096"      # below the floor
    assert attn._tune_key(49920) == "n65536"


def test_attention_keeps_the_default_kernel_names():
    from hk.ops import attn

    # These names are in the compile cache, in the README's resource table and
    # in every report this kernel is judged by.
    assert sorted(attn.KERNELS) == [
        "attn_fwd_d128", "attn_fwd_d128_causal",
        "attn_fwd_d64", "attn_fwd_d64_causal"]


def test_attention_does_not_tune_anything_that_changes_what_it_accepts():
    from hk.ops import attn

    # q_block/kv_block/warps decide Q_TILE, which is part of the set of shapes
    # `attention` will take. Tuning them would make that set depend on a file
    # in a cache directory.
    axes = {k for s in attn.SPACE for k in s}
    assert axes == {"vt_d_chunk", "qk_tiles", "pv_tiles"}


def test_every_shipped_tuner_is_registered_under_its_own_name():
    for name, t in hk.autotune.REGISTRY.items():
        assert t.name == name
        assert t.schedules and t.default


def test_a_record_from_another_arch_is_ignored(monkeypatch, tmp_path):
    # A shipped record travels in a wheel to whatever machine installs it, and
    # a schedule measured on gfx1100 says nothing about gfx1201. The key is a
    # problem shape and means the same thing on both, so nothing else in the
    # lookup would have caught this.
    class Arched(FakeKernel):
        arch = "gfx1100"

    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]),
               make=lambda **s: Arched(**s))
    assert t.arch == "gfx1100"
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}, "arch": "gfx1201"})
    assert t.schedule_for("k") == DEFAULT
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}, "arch": "gfx1100"})
    assert t.schedule_for("k") == {"a": 2, "b": 1}


def test_an_untagged_record_is_still_honoured(monkeypatch, tmp_path):
    # Records written before the tag existed, and tuners whose kernels have no
    # arch at all. Dropping those would be a silent de-tuning.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}})
    assert t.schedule_for("k") == {"a": 2, "b": 1}


def test_tuning_tags_the_record_with_the_arch_it_measured(monkeypatch, tmp_path):
    class Arched(FakeKernel):
        arch = "gfx1100"

    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]),
               make=lambda **s: Arched(**s))
    t.tune(lambda k: 10.0 / k.sched["a"], key="big")   # a=2 is 2x faster
    rec = json.loads((tmp_path / "t.json").read_text())["big"]
    assert rec["arch"] == "gfx1100"


def test_a_record_that_measured_a_tie_does_not_change_the_launch(monkeypatch, tmp_path):
    # The attention sweep's winners beat the default by 0.10-0.21% and were
    # four different schedules across three sizes. Measured, recorded, and not
    # acted on -- a serving process stays on one kernel.
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}, "ms": 9.98,
                                       "default_ms": 10.0, "gain": 0.002})
    assert t.schedule_for("k") == DEFAULT


def test_the_lookup_bar_is_overridable_from_the_environment(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}, "gain": 0.002})
    monkeypatch.setenv("HK_AUTOTUNE_MIN_GAIN", "0.001")
    hk.autotune.forget_records()
    assert t.schedule_for("k") == {"a": 2, "b": 1}


def test_a_real_win_is_still_applied(monkeypatch, tmp_path):
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    hk.autotune.save_record("t", "k", {"schedule": {"a": 2}, "gain": 0.12})
    assert t.schedule_for("k") == {"a": 2, "b": 1}


# ---------------------------------------------------------------- --ship


def _shippable(**kw):
    rec = {"arch": "gfx1100", "schedule": {"a": 2}, "ms": 1.0,
           "default_ms": 2.0, "gain": 0.5}
    rec.update(kw)
    return rec


@pytest.fixture
def shipping(monkeypatch, tmp_path):
    """A registry of exactly one tuner, shipping into a temp dir."""
    t = _tuner(monkeypatch, tmp_path, space(a=[1, 2]))
    shipped = tmp_path / "shipped"
    monkeypatch.setattr(hk.autotune, "shipped_dir", lambda: shipped)
    monkeypatch.setattr(hk.autotune, "_load_registry", lambda: {"t": t})
    return t, shipped


def test_ship_carries_a_record_that_beat_the_default(shipping):
    t, shipped = shipping
    hk.autotune.save_record("t", "4096", _shippable())
    hk.autotune.ship()
    assert json.loads((shipped / "t.json").read_text())["4096"]["gain"] == 0.5


def test_ship_refuses_a_win_that_is_measurement_noise(shipping):
    # The attention sweep measured 0.10-0.21% wins. `min` over a column of
    # noisy numbers always names someone; that does not make it a finding.
    t, shipped = shipping
    hk.autotune.save_record("t", "4096",
                            _shippable(ms=9.98, default_ms=10.0, gain=0.002))
    hk.autotune.ship()
    assert not shipped.exists()


def test_the_bar_is_settable(shipping):
    t, shipped = shipping
    hk.autotune.save_record("t", "4096",
                            _shippable(ms=9.98, default_ms=10.0, gain=0.002))
    hk.autotune.ship(min_gain=0.001)
    assert "4096" in json.loads((shipped / "t.json").read_text())


def test_ship_recomputes_a_missing_gain_from_the_two_times(shipping):
    # `gain` is a convenience; `ms` and `default_ms` are the measurements.
    t, shipped = shipping
    rec = _shippable()
    del rec["gain"]
    hk.autotune.save_record("t", "4096", rec)
    hk.autotune.ship()
    assert "4096" in json.loads((shipped / "t.json").read_text())


def test_ship_refuses_a_record_with_nothing_to_compare_against(shipping):
    t, shipped = shipping
    rec = _shippable()
    del rec["gain"], rec["default_ms"]
    hk.autotune.save_record("t", "4096", rec)
    hk.autotune.ship()
    assert not shipped.exists()


def test_ship_refuses_an_untagged_record(shipping):
    # It is honoured locally (the machine that measured it is the machine
    # reading it) and refused in a wheel, which can reach another arch.
    t, shipped = shipping
    hk.autotune.save_record("t", "4096", _shippable(arch=None))
    hk.autotune.ship()
    assert not shipped.exists()


def test_a_dry_run_writes_nothing(shipping):
    t, shipped = shipping
    hk.autotune.save_record("t", "4096", _shippable())
    hk.autotune.ship(dry_run=True)
    assert not shipped.exists()


def test_ship_drops_a_stale_shipped_file_whose_record_no_longer_qualifies(shipping):
    t, shipped = shipping
    shipped.mkdir()
    (shipped / "t.json").write_text(json.dumps({"4096": _shippable()}))
    hk.autotune.forget_records()
    # Re-measured, and this time the win is noise.
    hk.autotune.save_record("t", "4096",
                            _shippable(ms=9.98, default_ms=10.0, gain=0.002))
    hk.autotune.ship()
    assert not (shipped / "t.json").exists()


def test_shipping_twice_is_the_same_file(shipping):
    t, shipped = shipping
    hk.autotune.save_record("t", "4096", _shippable())
    hk.autotune.ship()
    first = (shipped / "t.json").read_text()
    hk.autotune.ship()
    assert (shipped / "t.json").read_text() == first


def test_a_shipped_record_is_what_production_reads_back(shipping):
    t, shipped = shipping
    hk.autotune.save_record("t", "4096", _shippable())
    hk.autotune.ship()
    # Wipe the user half: a fresh install has only what the wheel carries.
    (hk.autotune.records_dir() / "t.json").unlink()
    hk.autotune.forget_records()
    assert t.schedule_for("4096") == {"a": 2, "b": 1}


def test_the_listing_says_when_a_record_is_not_being_applied(shipping, capsys):
    t, _ = shipping
    hk.autotune.save_record("t", "k", {"arch": "gfx1100", "schedule": {"a": 2},
                                       "ms": 9.98, "default_ms": 10.0,
                                       "gain": 0.002})
    hk.autotune.main([])
    out = capsys.readouterr().out
    assert "+0.20%" in out and "(not applied)" in out
