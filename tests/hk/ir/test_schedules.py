"""hk.schedules' protocol: check() with reasons, spaces and neighbourhoods,
records, the composite, lookup, and the route table's own candidates. No
compiler, no GPU (the builds are tests/hk/codegen/test_schedules_legality.py,
the byte-identical C++ test_schedules_roundtrip.py)."""

from __future__ import annotations

import importlib
import json

import pytest

pytest.importorskip("hk._mlir")

import hk.schedules as S  # noqa: E402
from hk.ops import gemm as rdna  # noqa: E402
from hk.ops import gemm_cdna  # noqa: E402

TG = importlib.import_module("hk.templates.gemm")

SHAPES = {"gfx950": [(1, 4096, 4096), (32, 6144, 4096), (300, 151936, 4096),
                     (2048, 16384, 4096), (8192, 8192, 8192)],
          "gfx1100": [(1, 4096, 4096), (16, 6144, 4096), (2048, 16384, 4096),
                      (4096, 4096, 4096)]}


def test_every_family_has_a_protocol_and_documented_knobs():
    for cls in (S.Rdna, S.Decode, S.PingPong, S.Dp):
        assert S.family(cls.family) is cls
        for a in cls.axes().values():
            assert a.doc, (cls.family, a.name)
            if a.values:
                assert a.default in a.values or a.gate, (cls.family, a.name)
            if a.seed:
                assert set(a.seed) <= set(a.values), (cls.family, a.name)
    assert [c.family for c in S.families_for("gfx950")] == ["decode", "pp", "dp", "cols"]
    assert [c.family for c in S.families_for("gfx1100")] == ["rdna", "decode", "cols"]


def test_check_says_why():
    why = S.Rdna(block_m=256, block_n=256, k_step=128).check(4096, 4096, 4096, "gfx1100")
    assert any("LDS" in w for w in why)
    why = S.Rdna(warp_rows=3).check(4096, 4096, 4096, "gfx1100")
    assert why == ["warp_rows=3 does not divide warps=8 into a grid"]
    why = S.Rdna().check(4096, 4096, 4000, "gfx1100")
    assert why == ["K=4000 is not a multiple of k_step=64"]
    assert S.PingPong().check(4096, 4096, 4096, "gfx1100") == [
        "pp: pp builds for gfx950, not gfx1100"]
    assert any("not 8" in w for w in S.PingPong(warps_m=2, warps_n=2).check(
        4096, 4096, 4096, "gfx950"))
    assert any("hk-lds-pipeline" in w for w in S.PingPong(
        hipkittens_waits=True, late_b=False).check(4096, 4096, 4096, "gfx950"))
    assert any("registers" in w for w in S.Rdna(block_m=256, k_step=32).check(
        4096, 4096, 4096, "gfx1100"))
    assert any("M=64 is more than" in w for w in S.Decode().check(64, 4096, 4096, "gfx1100"))
    assert any("bf16" in w for w in S.Dp().check(64, 4096, 4096, "gfx950", dtype="fp8e4m3"))
    assert any("GiB" in w for w in S.Dp(splits=8).check(65536, 65536, 4096, "gfx950"))
    assert any("splits must be 1" in w for w in S.PingPong(stream_k=1, splits=2).check(
        4096, 4096, 4096, "gfx950"))
    assert any("2 x k_step" in w for w in S.PingPong(stream_k=1).check(
        4096, 4096, 4160, "gfx950"))
    assert any("XCDs" in w for w in S.PingPong(stream_k=2, sk_grid=100).check(
        4096, 4096, 4096, "gfx950"))
    assert any("transposed_c" in w for w in S.PingPong(stream_k=1, transposed_c=True).check(
        4096, 4096, 4096, "gfx950"))
    assert not S.PingPong(stream_k=2).check(4096, 5120, 4096, "gfx950")
    # 192-wide blocks: a 96-row half is whole copies a warp, not a thread.
    assert not S.PingPong(block_n=192, warps_m=4, warps_n=2).check(1024, 3072, 1024, "gfx950")
    with pytest.raises(S.Illegal) as e:
        S.Dp(block_n=96).require(4096, 4096, 4096, "gfx950")
    assert e.value.reasons == ["N=4096 is not a multiple of block_n=96"]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_the_space_is_legal_sized_and_distinct(arch):
    for m, n, k in SHAPES[arch]:
        pts = list(S.space(m, n, k, arch))
        assert pts, (m, n, k)
        assert len(pts) == len(set(pts))
        assert len(pts) < 3000, (m, n, k, len(pts))
        for s in pts:
            assert s.check(m, n, k, arch) == [], (s, m, n, k)
            assert S.from_record(json.loads(json.dumps(s.to_record()))) == s
        fams = {s.family for s in pts}
        assert ("rdna" in fams) == (arch == "gfx1100")


def test_a_neighbourhood_moves_at_most_radius_knobs():
    s = S.lookup(2048, 16384, 4096, "gfx950")
    one = list(S.space(2048, 16384, 4096, "gfx950", around=s, radius=1))
    two = list(S.space(2048, 16384, 4096, "gfx950", around=s, radius=2))
    assert one[0] == s and set(one) <= set(two) and len(two) > len(one) > 3
    for t in two:
        assert type(t) is type(s)
        assert sum(t.knobs()[k] != v for k, v in s.knobs().items()) <= 2
    only = list(S.space(2048, 16384, 4096, "gfx950", around=s, axes={"stages": [2, 3, 4, 5]}))
    assert {t.stages for t in only} <= {2, 3, 4, 5} and all(
        t.replace(stages=s.stages) == s for t in only)


def test_axes_restrict_and_extend_the_space():
    pin = list(S.Dp.space(4096, 4096, 4096, "gfx950", axes={"block_m": 64, "splits": [1]}))
    assert pin and all(t.block_m == 64 and t.splits == 1 for t in pin)
    wide = list(S.Dp.space(4096, 4096, 4096, "gfx950", axes={"wgm": "all"}))
    assert {t.wgm for t in wide} == set(S.Dp.axes()["wgm"].values)
    with pytest.raises(ValueError, match="no knob"):
        list(S.Dp.space(4096, 4096, 4096, "gfx950", axes={"nope": [1]}))


def test_records():
    s = S.Dp(block_m=64, stages=4, splits=2)
    assert s.to_record() == {"family": "dp", "block_m": 64, "stages": 4, "splits": 2}
    assert S.from_record(["dp", {"block_m": 64, "stages": 4, "splits": 2}]) == s
    assert S.Dp.from_record({"block_m": 64, "stages": 4, "splits": 2}) == s
    assert len(s.to_record(full=True)) == len(S.Dp.axes()) + 1
    with pytest.raises(ValueError, match="no knob"):
        S.from_record({"family": "dp", "bogus": 1})
    assert S.from_record({"family": "dp", "bogus": 1}, strict=False) == S.Dp()
    with pytest.raises(ValueError, match="a pp record given to dp"):
        S.Dp.from_record({"family": "pp"})
    c = S.Columns(head=S.PingPong(), tail=S.Dp(block_m=64), cut="block")
    assert S.from_record(json.loads(json.dumps(c.to_record()))) == c
    assert hash(c) == hash(S.from_record(c.to_record()))


@pytest.mark.arch("gfx1100")  # a gfx1100 record
def test_rdna_records_carry_every_code_knob():
    """A tuned record may set any knob that changes the C++: hk.ops.gemm's
    tuner reads them all (the gates aside), and its DEFAULT -- the dict the
    launch path keys on -- is as it was."""
    code = {n for n, a in S.Rdna.axes().items() if not a.gate}
    assert rdna.TUNER.knobs == code
    assert S.Rdna.from_record(rdna.DEFAULT) == S.Rdna()
    assert S.Rdna().legacy() == rdna.DEFAULT
    assert S.Rdna(**rdna.THIN).legacy() == rdna.THIN
    from hk import autotune

    t = S.Rdna(block_m=64, setprio=True, sched=2)
    rec = t.to_record()
    rec.pop("family")
    autotune.save_record("gemm_bf16", "64x6144x4096",
                         {"arch": "gfx1100", "schedule": rec, "gain": 0.05})
    try:
        got = rdna.schedule_for(64, 6144, 4096)
        assert S.Rdna.from_record(got) == t and got == t.legacy()
    finally:
        autotune._record_path("gemm_bf16").unlink()
        autotune.forget_records()


def test_columns():
    c = S.lookup(300, 151936, 4096, "gfx950")
    assert isinstance(c, S.Columns) and c.cut == "wave"
    steps = c.steps(300, 151936, 4096)
    assert steps[0].cols[0] == 0 and steps[-1].cols[1] == 151936
    assert all(a.cols[1] == b.cols[0] for a, b in zip(steps, steps[1:]))
    tail = S.lookup(4096, 151936 - 4096 * 0, 4096, "gfx950")
    assert tail.check(4096, 151936, 4096, "gfx950") == []
    bad = S.Columns(head=S.PingPong(), tail=S.Dp(), cut=4096 * 37)
    assert any("both parts" in w for w in bad.check(300, 4096, 4096, "gfx950"))
    with pytest.raises(TypeError):
        c.kernel()


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_lookup_is_legal_where_hk_runs(arch):
    for m in (1, 16, 32, 64, 256, 1024, 4096):
        for n, k in ((4096, 4096), (12288, 4096), (4096, 14336), (151936, 4096)):
            for op in ("matmul", "template"):
                s = S.lookup(m, n, k, arch, op=op)
                assert s.check(m, n, k, arch) == [], (m, n, k, op, s)


def test_the_template_plans_an_explicit_schedule():
    s = S.Dp(block_m=64, splits=4)
    steps = TG.plan(1024, 4096, 4096, arch="gfx950", schedule=s)
    assert [(st.kernel, st.into, st.epilogue) for st in steps] == [
        ("dp", "scratch", False), ("reduce", "out", True)]
    assert steps[0].schedule == s
    assert TG.plan(1024, 4096, 4096, arch="gfx950", schedule=s.to_record()) == steps
    with pytest.raises(S.Illegal, match="block_n=96"):
        TG.plan(4096, 4096, 4096, arch="gfx950", schedule=S.Dp(block_n=96))


def test_a_route_tables_own_candidate_is_a_schedule(monkeypatch):
    """A route table may name candidates the code does not have: the table's
    "candidates" defines them as schedule records, and the route, fits and
    the template's plan use them as they are."""
    s = S.Dp(block_m=64, block_n=128, k_step=128, stages=2)
    monkeypatch.setattr(gemm_cdna, "_ROUTES", {"4096x4096": {512: "k3_a"}, "": {}})
    monkeypatch.setattr(gemm_cdna, "_EXTRA", {"k3_a": s})
    assert gemm_cdna.route(512, 4096, 4096) == "k3_a"
    assert gemm_cdna.fits("k3_a", 512, 4096, 4096)
    assert S.from_candidate("k3_a", 512, 4096, 4096) == s
    assert S.lookup(512, 4096, 4096, "gfx950") == s
    steps = TG.plan(512, 4096, 4096, arch="gfx950")
    assert [(st.kernel, st.route, st.schedule) for st in steps] == [("dp", "k3_a", s)]
