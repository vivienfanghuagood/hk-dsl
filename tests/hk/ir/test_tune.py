"""hk.tune without a GPU: the acceptance rule and the search on synthetic
timings, the records and the routes that read them, the space providers, and
the hidden-shape set's invariants.

The point of the synthetic runs is the property that matters most: a tuner
that "finds" noise is worse than none. A space whose every candidate is as
fast as the route, timed with realistic noise, must come back "tie" -- every
seed, not most of them -- and a space with a real 5% winner must find it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from hk.tune import records, stats
from hk.tune.search import Search
from hk.tune.space import Candidate

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- stats


def _sets(t, noise, n_sets=3, rounds=7, rng=None):
    rng = rng or random.Random(0)
    return [[t * (1 + rng.gauss(0, noise)) for _ in range(rounds)] for _ in range(n_sets)]


def test_decide_clear_win_tie_and_loss():
    rng = random.Random(1)
    route = _sets(1.0, 0.003, rng=rng)
    assert stats.decide(route, _sets(0.95, 0.003, rng=rng)).verdict == "win"
    assert stats.decide(route, _sets(0.996, 0.003, rng=rng)).verdict == "tie"
    assert stats.decide(route, _sets(1.06, 0.003, rng=rng)).verdict == "loss"


def test_decide_margin_grows_with_noise():
    """A 3% gain is a win on a quiet card and a tie when the rounds spread 4%."""
    rng = random.Random(2)
    quiet = stats.decide(_sets(1.0, 0.002, rng=rng), _sets(0.97, 0.002, rng=rng))
    noisy = stats.decide(_sets(1.0, 0.04, rng=rng), _sets(0.97, 0.04, rng=rng))
    assert quiet.verdict == "win" and quiet.bar == pytest.approx(stats.MIN_GAIN)
    assert noisy.verdict == "tie" and noisy.bar > 0.03


def test_decide_sets_disagreeing_is_a_tie():
    """Two sets say +4%, one says -4%: the across-set spread sets the bar."""
    route = [[1.0] * 7, [1.0] * 7, [1.0] * 7]
    cand = [[0.96] * 7, [0.96] * 7, [1.04] * 7]
    d = stats.decide(route, cand)
    assert d.verdict == "tie" and d.band[0] < 1 < d.band[1]


def test_confirm_needs_both():
    w = stats.Decision("win", 0.05, 0.002, 0.01, [1.04, 1.06], [1.05, 1.05, 1.05], 3)
    t = stats.Decision("tie", 0.004, 0.002, 0.01, [1.0, 1.01], [1.004, 1.004], 2)
    assert stats.confirm(w, w).verdict == "win"
    c = stats.confirm(w, t)
    assert c.verdict == "tie" and c.sets == 5


def test_ceiling_drops_impossible_rounds():
    floor = stats.floor_ms(8192, 8192, 8192, "gfx950")
    assert floor and 0.3 < floor < 0.5
    times = [0.8, 0.81, 0.79, floor / 2]
    assert stats.summary(times, floor)[2] == 3


def test_halve_keeps_route_and_near_ties():
    scores = {"a": [1.0] * 3, "b": [1.001] * 3, "c": [1.2] * 3, "route": [2.0] * 3}
    kept = stats.halve(scores, 1, protect=["route"])
    assert kept[0] == "a" and "route" in kept and "c" not in kept
    noisy = {"a": [1.0, 1.02, 0.98], "b": [1.01, 1.03, 0.99], "c": [1.5] * 3}
    assert set(stats.halve(noisy, 1)) == {"a", "b"}


# ---------------------------------------------------------------- the search, synthetic


class FakeSpace:
    """Candidates "c0".."cN" of kind "dp" with known true times; the route is c0."""

    arch = "gfx950"

    def __init__(self, truth, refused=(), wrong=(), model_error=0.3, seed=0):
        self.truth = truth              # {id: ms}
        self.refused, self.wrong = set(refused), set(wrong)
        self.rng = random.Random(seed)
        self.err = {i: 1 + self.rng.uniform(-model_error, model_error) for i in truth}

    def c(self, i):
        return Candidate.of("dp", {"id": i})

    def current(self, m, n, k):
        return Candidate.of("dp", {"id": 0}, "route")

    def seeds(self, m, n, k):
        return [self.c(1)] if 1 in self.truth else []

    def generate(self, m, n, k):
        return [self.c(i) for i in self.truth]

    def check(self, c, m, n, k):
        return "" if c.sched["id"] in self.truth else "not in the space"

    def estimate(self, c, m, n, k):
        i = c.sched["id"]
        return self.truth[i] * self.err[i] * 1e-3

    def structure(self, c, m, n, k):
        return ("dp", c.sched["id"])

    def preference(self, c):
        return ()

    def neighbors(self, c, m, n, k):
        i = c.sched["id"]
        return [self.c(j) for j in (i - 1, i + 1) if j in self.truth]

    def jobs(self, c, m, n, k):
        return [(c.label, None, {})]

    def runner(self, c, m, n, k):
        def fn(a=None, b=None, out=None):
            return None
        fn.cand = c
        fn.err = 1.0 if c.sched["id"] in self.wrong else 1e-3
        return fn

    def spec(self, c):
        return [c.kind, c.sched]


class FakeBench:
    copies = 4

    def __init__(self, space, noise=0.004, seed=0, drift=0.0):
        self.space, self.noise, self.drift = space, noise, drift
        self.rng = random.Random(seed)
        self.gpu_seconds = 0.0
        self.set_bias = 1.0

    def refresh(self):
        # Each set runs on its own activations and at its own clock: a
        # common factor on every arm of the set.
        self.set_bias = 1 + self.rng.gauss(0, self.drift)

    def check(self, fn):
        self.gpu_seconds += 0.01
        return fn.err

    def time(self, arms, rounds):
        out = {}
        for a in arms:
            if isinstance(a, Candidate):
                t = self.space.truth[a.sched["id"]]
            else:
                t = 1.0
            out[a] = [t * self.set_bias * (1 + self.rng.gauss(0, self.noise)) for _ in range(rounds)]
            self.gpu_seconds += rounds * (self.copies * t * 1e-3 + 1.5e-3)
        return out


class FakeCompiler:
    def __init__(self, refused=()):
        self.refused = set(refused)
        self.submitted = []

    def submit(self, c, jobs):
        self.submitted.append(c)

    def result(self, c):
        return "rejected: spills" if c.sched["id"] in self.refused else ""


def _run(truth, *, noise=0.004, seed=0, budget=60.0, refused=(), wrong=(), drift=0.0,
         model_error=0.05, **kw):
    sp = FakeSpace(truth, wrong=wrong, seed=seed, model_error=model_error)
    bench = FakeBench(sp, noise=noise, seed=seed, drift=drift)
    res = Search(sp, budget=budget, **kw).run(256, 4096, 4096, bench, FakeCompiler(refused),
                                              baselines={"torch": lambda a, b, o: None})
    return res, bench


def test_search_finds_a_real_winner():
    truth = {i: 1.0 + 0.02 * (i % 7) for i in range(40)}
    truth[23] = 0.93
    res, _ = _run(truth)
    assert res.champion == Candidate.of("dp", {"id": 23})
    assert res.verdict == "win" and 0.05 < res.decision.gain < 0.10
    assert res.accepted == res.champion


@pytest.mark.parametrize("seed", range(20))
def test_search_does_not_find_noise(seed):
    """Every candidate is the route's speed: the verdict is never a win."""
    truth = {i: 1.0 for i in range(30)}
    res, _ = _run(truth, seed=seed, noise=0.006, drift=0.004)
    assert res.verdict != "win", res.decision
    assert res.accepted == res.route


def test_search_explores_past_a_wrong_model():
    """The model puts the winner far down its ranking; the exploration
    slots measure structures it did not pick, over seeds the winner is found
    in some runs, and a run that misses it still never takes a loser."""
    found = 0
    for seed in range(10):
        truth = {i: 1.0 + 0.01 * (i % 5) for i in range(30)}
        truth[17] = 0.9
        res, _ = _run(truth, seed=seed, model_error=0.3, pool_per_kind=4, explore=4)
        found += res.champion == Candidate.of("dp", {"id": 17})
        assert res.verdict != "win" or res.champion == Candidate.of("dp", {"id": 17})
    assert found >= 3


def test_search_skips_refused_and_wrong_candidates():
    truth = {i: 1.0 for i in range(12)}
    truth[5] = 0.5                     # fastest, but it spills
    truth[6] = 0.6                     # fast, but wrong
    truth[7] = 0.9
    res, _ = _run(truth, refused={5}, wrong={6})
    assert res.champion == Candidate.of("dp", {"id": 7})
    assert res.counts["refused"] >= 1 and res.counts["wrong"] >= 1
    assert "rejected" in res.refused[Candidate.of("dp", {"id": 5}).label]


def test_search_local_moves_reach_past_the_pool():
    """The best candidate is not in the pool (its model estimate is awful)
    but next to a good one: local search walks to it."""
    truth = {i: 2.0 for i in range(60)}
    for i, t in ((40, 0.95), (41, 0.9), (42, 0.85)):
        truth[i] = t
    sp = FakeSpace(truth)
    sp.err[41] = sp.err[42] = 50.0
    res = Search(sp, budget=60, pool_per_kind=8).run(
        256, 4096, 4096, FakeBench(sp), FakeCompiler())
    assert res.champion == Candidate.of("dp", {"id": 42})


def test_search_restarts_out_of_a_basin():
    """The best pool point's neighbourhood is a dead end; the winner sits
    next to the fourth best, past the model's cut: a restart from the
    fastest structure no descent started from finds it."""
    truth = {i: 2.0 for i in range(40)}
    truth.update({9: 0.96, 10: 0.95, 11: 0.96, 20: 0.97, 21: 0.85})
    sp = FakeSpace(truth, model_error=0.0)
    sp.err[21] = 50.0
    res = Search(sp, budget=60, pool_per_kind=40, explore=0).run(
        256, 4096, 4096, FakeBench(sp), FakeCompiler())
    assert res.champion == Candidate.of("dp", {"id": 21})
    assert res.counts["restarts"] >= 1


def test_search_respects_the_budget():
    truth = {i: 1.0 + 0.01 * i for i in range(200)}
    res, bench = _run(truth, budget=2.0)
    assert res.gpu_seconds < 2.0 * 1.5
    assert res.decision is not None          # the decision always runs


def test_result_reports_baseline_ratio():
    truth = {i: 1.2 for i in range(10)}
    truth[3] = 0.9
    res, _ = _run(truth)
    r = res.baselines["torch"]["ratio"]
    assert r[0] == pytest.approx(1 / 0.9, rel=0.02) and r[1] <= r[0] <= r[2]
    assert res.baselines["torch"]["route_ratio"][0] == pytest.approx(1 / 1.2, rel=0.02)
    d = res.as_dict()
    assert json.dumps(d) and d["verdict"] == "win" and d["gpu_seconds"] > 0


# ---------------------------------------------------------------- records and routes


@pytest.fixture
def tunedir(tmp_path, monkeypatch):
    import hk.autotune

    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.delenv("HK_AUTOTUNE", raising=False)
    records.forget()
    hk.autotune.forget_records()
    yield tmp_path
    records.forget()
    hk.autotune.forget_records()


def test_records_answer_their_bucket_only(tunedir):
    spec = ["dp", {"block_m": 64, "block_n": 128, "stages": 3, "xcd_chunk": 0}]
    records.save("gfx950", 2048, 16384, 4096, "u:x", spec, {"m": 2048, "n": 16384, "k": 4096})
    assert records.user_route("gfx950", 2048, 16384, 4096)[0] == "u:x"
    assert records.user_route("gfx950", 1100, 16384, 4096)[0] == "u:x"       # bucket 2048
    assert records.user_route("gfx950", 1024, 16384, 4096) is None
    assert records.user_route("gfx950", 2048, 16384, 8192) is None
    assert records.user_route("gfx1100", 2048, 16384, 4096) is None


def test_records_replace_and_switch_off(tunedir, monkeypatch):
    records.save("gfx950", 512, 4096, 4096, "u:a", ["dp", {}], {"m": 512, "n": 4096, "k": 4096})
    records.save("gfx950", 500, 4096, 4096, "u:b", ["dp", {"splits": 2}],
                 {"m": 500, "n": 4096, "k": 4096})
    d = json.loads(records.path("gfx950").read_text())
    assert d["routes"] == {"4096x4096": {"500": "u:b"}}
    assert list(d["candidates"]) == ["u:b"] and len(d["points"]) == 1
    monkeypatch.setenv("HK_AUTOTUNE", "0")
    assert records.user_route("gfx950", 512, 4096, 4096) is None


def test_corrupt_record_is_no_record(tunedir):
    records.path("gfx950").write_text("{not json")
    records.forget()
    assert records.user_route("gfx950", 512, 4096, 4096) is None


def test_gfx950_route_and_plan_read_the_record(tunedir):
    from hk.ops import gemm_cdna
    from hk.templates import gemm as tg

    before = gemm_cdna.route(2048, 16384, 4096)
    spec = ["pp", {"block_m": 256, "block_n": 256, "wgm": 4, "xcd_chunk": 8}]
    records.save("gfx950", 2048, 16384, 4096, "u:pp256", spec, {})
    assert gemm_cdna.route(2048, 16384, 4096) == "u:pp256"
    assert gemm_cdna.route(2048, 16384, 4096, user=False) == before
    assert gemm_cdna.route(4096, 16384, 4096) == gemm_cdna.route(4096, 16384, 4096, user=False)
    steps = tg.plan(2048, 16384, 4096, arch="gfx950")
    assert steps[0].kernel == "pp" and steps[0].route == "u:pp256"
    assert steps[0].sched["wgm"] == 4 and steps[0].sched["xcd_chunk"] == 8


def test_gfx950_record_that_does_not_fit_is_ignored(tunedir):
    from hk.ops import gemm_cdna

    records.save("gfx950", 2048, 16384, 4096, "u:big",
                 ["dp", {"block_n": 96, "block_m": 64}], {})      # 16384 % 96
    assert gemm_cdna.route(2048, 16384, 4096) == gemm_cdna.route(2048, 16384, 4096, user=False)


@pytest.mark.arch("gfx1100")  # a gfx1100 record
def test_rdna_schedule_and_plan_read_the_record(tunedir):
    from hk.ops import gemm
    from hk.templates import gemm as tg

    s = dict(block_m=64, block_n=128, k_step=64, warp_rows=2, n_split=2, wgm=4)
    records.save("gfx1100", 4096, 5120, 4096, "u:r", ["rdna", s], {})
    got = gemm.schedule_for(4096, 5120, 4096)
    assert got["block_m"] == 64 and got["wgm"] == 4
    assert gemm.schedule_for(4096, 5120, 4096, user=False) == gemm.DEFAULT
    records.save("gfx1100", 8, 3840, 5120, "u:d", ["decode", {"warps": 8, "splits": 2}], {})
    steps = tg.plan(8, 3840, 5120, arch="gfx1100")
    assert steps[0].kernel == "decode" and steps[0].sched["warps"] == 8


@pytest.mark.arch("gfx1201")
def test_gfx1201_reads_its_shipped_records_behind_this_machines(tunedir):
    from hk.ops import gemm

    # The shipped table answers its M bucket (50 rows: the 64 record) ...
    got = gemm.user_route(50, 4096, 4096)
    assert got is not None and got[0] == records.shipped_route("gfx1201", 64, 4096, 4096)[0]
    assert gemm.schedule_for(50, 4096, 4096, user=False)["block_n"] == 64
    # ... and nothing past what it holds, nor for another arch.
    assert gemm.user_route(4096, 4096, 4096) is None
    assert records.shipped_route("gfx1100", 64, 4096, 4096) is None
    assert records.shipped_route("gfx950", 64, 4096, 4096) is None
    # This machine's record comes first; HK_AUTOTUNE=0 ignores both.
    records.save("gfx1201", 64, 4096, 4096, "u:mine", ["rdna", dict(block_m=64, block_n=128)], {})
    assert gemm.user_route(64, 4096, 4096)[0] == "u:mine"
    assert gemm.user_route(64, 4096, 4096, user=False)[0] == got[0]


def test_gfx1201_shipped_records_take_their_shapes():
    """Every shipped gfx1201 schedule is legal at the M it was tuned at and at
    the bottom of its bucket, where the fewest rows meet its tiles."""
    import hk.schedules as S

    d = json.loads((Path(records.__file__).resolve().parents[1] / "tuned"
                    / "routes_gemm_gfx1201.json").read_text())
    assert d["arch"] == "gfx1201" and d["routes"]
    for nk, by_m in d["routes"].items():
        n, k = map(int, nk.split("x"))
        for ms, name in by_m.items():
            s = S.from_record(d["candidates"][name])
            for m in (int(ms), int(ms) // 2 + 1):
                assert not s.check(m, n, k, "gfx1201"), (m, n, k, name)
    assert {(p["m"], p["n"], p["k"]) for p in d["points"]} == {
        (int(ms), *map(int, nk.split("x"))) for nk, by_m in d["routes"].items() for ms in by_m}


# ---------------------------------------------------------------- spaces


def test_candidate_round_trips_through_its_record():
    import hk.schedules as S
    from hk.tune.space import ScheduleSpace

    sp = ScheduleSpace("gfx950")
    c = sp.cand(S.Dp(block_m=64, block_n=128, nt="b", lds_shape="16x64"))
    back = sp.from_spec(json.loads(json.dumps(sp.spec(c))))
    assert back == c and back.obj == c.obj
    assert hash(c) == hash(sp.cand(c.obj, "a name"))
    cols = sp.cand(S.from_candidate("ppw", 2048, 12288, 4096))
    assert cols.kind == "cols" and sp.from_spec(sp.spec(cols)) == cols and "cols" in cols.label


def test_every_knob_has_its_own_short_name():
    """A record's name is its key in the routes file: two schedules that
    differ in one knob must not share it (splits=2 and stream_k=2 once did)."""
    import hk.schedules as S
    from hk.tune.space import _ABBREV, _short

    for cls in (S.PingPong, S.Dp):
        names = [_ABBREV.get(k, k) for k in cls.axes()]
        # Keys that are only ever written by position (block_n's and
        # warps_n's "x") are the one exception the format has always had.
        assert len(names) - len(set(names)) <= 1, cls
    assert _short(S.PingPong(splits=2)) != _short(S.PingPong(stream_k=2))


def test_gfx950_space_is_checked_before_compiling():
    from hk.tune.space import ScheduleSpace

    sp = ScheduleSpace("gfx950")
    m, n, k = 2048, 16384, 4096
    cur = sp.current(m, n, k)
    assert cur.name == "dp128" and cur.kind == "dp"     # the rule (tuning-goal T3)
    gen = sp.generate(m, n, k)
    assert len(gen) > 200 and all(not sp.check(c, m, n, k) for c in gen[:50])
    assert {c.kind for c in gen} >= {"pp", "dp"}
    seeds = sp.seeds(m, n, k)
    assert any(c.name == "pp" for c in seeds)
    near = sp.neighbors(cur, m, n, k)
    assert near and all(c.kind == "dp" for c in near)
    import hk.schedules as S
    bad = sp.cand(S.Dp(block_m=256, block_n=256, stages=4, k_step=128))
    assert "LDS" in sp.check(bad, m, n, k)


def test_rdna_space_current_is_matmul():
    import hk.schedules as S
    from hk.ops import gemm
    from hk.tune.space import ScheduleSpace

    sp = ScheduleSpace("gfx1100")
    assert sp.current(4096, 5120, 4096).obj == S.Rdna(**gemm.DEFAULT)
    assert sp.current(8, 5120, 4096).obj == S.Rdna(**gemm.THIN)
    assert any(c.kind == "decode" for c in sp.seeds(8, 5120, 4096))
    assert any(c.kind == "rdna" for c in sp.generate(4096, 5120, 4096))


def test_cost_model_sees_wave_quantization():
    from hk.tune import cost

    s = dict(block_m=256, block_n=256, splits=1)
    one = cost.estimate("pp", s, 4096, 4096, 4096, "gfx950")          # 256 blocks: one wave
    more = cost.estimate("pp", s, 4096 + 256, 4096, 4096, "gfx950")   # 272: two waves
    assert more > 1.6 * one


# ---------------------------------------------------------------- the hidden-shape set


@pytest.fixture(scope="module")
def shapes_mod():
    import importlib
    import sys

    sys.path.insert(0, str(REPO))
    return importlib.import_module("bench.tune.shapes")


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100"])
def test_shape_set_is_committed_as_drawn(shapes_mod, arch):
    committed = json.loads(shapes_mod.path_for(arch).read_text())
    assert committed == json.loads(shapes_mod.dump(shapes_mod.draw(arch)))


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100"])
def test_shape_set_is_hidden_and_split(shapes_mod, arch):
    d = json.loads(shapes_mod.path_for(arch).read_text())
    held = shapes_mod.shipped_nk()
    tune = {(s["n"], s["k"]) for s in d["splits"]["tuning"]}
    acc = {(s["n"], s["k"]) for s in d["splits"]["acceptance"]}
    assert not tune & acc
    assert len(d["splits"]["acceptance"]) >= 20 and len(d["splits"]["tuning"]) >= 20
    for s in d["splits"]["tuning"] + d["splits"]["acceptance"]:
        assert (s["n"], s["k"]) not in held
        assert s["n"] % 128 == 0 and s["k"] % 64 == 0
    if arch == "gfx950":
        assert (2048, 16384, 4096) in shapes_mod.load(arch, "acceptance")
