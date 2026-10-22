"""The search: from the current route, over the schedule space, to a winner
that the acceptance rule believes -- or to "the route stands".

    pool        the shipped route names that take the shape (seeds), plus the
                generated space's best by the analytic model (hk.tune.cost) --
                one per structure (kind, block, K split), a quota per kind so
                no kind is judged by the model alone -- and a few structures
                past the model's cut, at random (it is a rough model)
    compile     every pool candidate on the CPU (hk.tune.measure.Compiler),
                hipcc in parallel; refused ones (spills, LDS, a pass's
                hazard) never reach the GPU
    check       each built candidate against fp32 on the GPU
    rungs       successive halving: every survivor timed interleaved for a few
                rounds, the fastest third kept (and anything within its noise
                of the cut, and the route), timed again for more rounds on
                fresh activations, and so on
    local       one-knob moves around the best two (swizzles, priorities,
                stages, K split, the decode GEMV's knobs), compiled while the
                rungs run, timed against the incumbents; repeated while they
                improve and the budget lasts
    decide      the champion against the route, interleaved in S sets of R
                rounds on fresh activations each (hk.tune.stats.decide), then
                confirmed on fresh sets measured after the decision
                (hk.tune.stats.confirm); baselines are timed in the same sets,
                so the ratio to them comes with its band

The budget is GPU-seconds (`GemmBench.gpu_seconds`: wall time of the GPU
work); the compile is CPU time beside it. A quarter of the budget is kept for
the decision, which always runs: a tuner that stops before it can tell a win
from noise has found nothing.

The search never sees a GPU directly: it is given a bench (check / time /
refresh / gpu_seconds) and a compiler (submit / result), so
tests/hk/ir/test_tune.py runs it on synthetic timings.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import stats
from .space import Candidate

#: Defaults: candidates compiled from the generated space (besides the seeds),
#: per kind; rounds per rung; local-search moves per incumbent.
POOL_PER_KIND = 16
EXPLORE = 6
RUNG_ROUNDS = (3, 6, 10)
LOCAL_MOVES = 12
LOCAL_ITERS = 6
DECIDE_SETS, DECIDE_ROUNDS, CONFIRM_SETS = 3, 7, 2
#: Host and spin overhead a timed round of an arm costs beside its calls, s.
ROUND_OVERHEAD = 1.5e-3


@dataclass
class Result:
    """What one tuning run found and what it cost."""

    m: int
    n: int
    k: int
    arch: str
    route: Candidate
    champion: Optional[Candidate] = None
    decision: Optional[stats.Decision] = None
    route_ms: Optional[float] = None
    ms: Optional[float] = None
    baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    compile_seconds: float = 0.0
    load_seconds: float = 0.0
    phases: Dict[str, float] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    timed: Dict[str, Optional[float]] = field(default_factory=dict)
    refused: Dict[str, str] = field(default_factory=dict)
    record: str = ""

    @property
    def verdict(self) -> str:
        return self.decision.verdict if self.decision else "none"

    @property
    def accepted(self) -> Optional[Candidate]:
        """The candidate the shape should run: the champion on a win, else the route."""
        return self.champion if self.verdict == "win" else self.route

    def best_baseline(self):
        """(name, ms) of the fastest baseline timed."""
        b = [(v["ms"], name) for name, v in self.baselines.items() if v.get("ms")]
        return (min(b)[1], min(b)[0]) if b else (None, None)

    def summary(self) -> str:
        d = self.decision
        acc = self.accepted
        lines = [f"{self.m}x{self.n}x{self.k} on {self.arch}: route {self.route} "
                 f"{_ms(self.route_ms)}"]
        if self.champion is not None and d is not None:
            lines.append(f"  champion {self.champion} {_ms(self.ms)}: {d.verdict}, "
                         f"{d.gain * 100:+.2f}% vs the route (bar {d.bar * 100:.2f}%, "
                         f"band {d.band[0]:.3f}-{d.band[1]:.3f}x over {d.sets} sets)")
        for name, v in sorted(self.baselines.items()):
            r = v.get("ratio")
            if r:
                lines.append(f"  vs {name} {_ms(v.get('ms'))}: {acc} at {r[0]:.3f}x "
                             f"(band {r[1]:.3f}-{r[2]:.3f}); route {v.get('route_ratio', [0])[0]:.3f}x")
        c = self.counts
        lines.append(f"  {c.get('pool', 0)} compiled of {c.get('legal', 0)} legal "
                     f"({c.get('generated', 0)} generated), {c.get('refused', 0)} refused by the "
                     f"compile, {c.get('failed', 0)} at their first call, {c.get('wrong', 0)} "
                     f"wrong, {c.get('timed', 0)} timed; "
                     f"{self.gpu_seconds:.1f} GPU-s, {self.wall_seconds:.0f} s wall "
                     f"({self.load_seconds:.0f} s loading)")
        if self.record:
            lines.append(f"  record: {self.record}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        name, ms = self.best_baseline()
        acc = self.accepted
        return dict(
            m=self.m, n=self.n, k=self.k, arch=self.arch,
            route=str(self.route), route_spec=[self.route.kind, self.route.sched],
            route_ms=_r(self.route_ms),
            champion=str(self.champion) if self.champion else None,
            champion_spec=[self.champion.kind, self.champion.sched] if self.champion else None,
            ms=_r(self.ms), verdict=self.verdict,
            decision=self.decision.as_dict() if self.decision else None,
            accepted=str(acc), baselines=self.baselines, best_baseline=name,
            counts=self.counts, gpu_seconds=round(self.gpu_seconds, 2),
            wall_seconds=round(self.wall_seconds, 1),
            compile_seconds=round(self.compile_seconds, 1),
            load_seconds=round(self.load_seconds, 1), phases=self.phases, record=self.record,
            timed=self.timed,
            refused=dict(list(self.refused.items())[:40]))


def _ms(x):
    return f"{x * 1e3:.2f} us" if x is not None else "-"


def _r(x):
    return round(x, 6) if x is not None else None


class Search:
    """One shape's search; see the module docstring."""

    def __init__(self, space, *, budget: float = 60.0, min_gain: float = stats.MIN_GAIN,
                 k_noise: float = stats.K_NOISE, pool_per_kind: int = POOL_PER_KIND,
                 explore: int = EXPLORE, max_seeds: Optional[int] = None,
                 rung_rounds: Sequence[int] = RUNG_ROUNDS, local_moves: int = LOCAL_MOVES,
                 local_iters: int = LOCAL_ITERS, decide_sets: int = DECIDE_SETS,
                 decide_rounds: int = DECIDE_ROUNDS, confirm_sets: int = CONFIRM_SETS,
                 log: Optional[Callable[[str], None]] = None):
        self.space = space
        self.budget = budget
        self.min_gain, self.k_noise = min_gain, k_noise
        self.pool_per_kind = pool_per_kind
        self.explore = explore
        self.max_seeds = max_seeds
        self.rung_rounds = tuple(rung_rounds)
        self.local_moves, self.local_iters = local_moves, local_iters
        self.decide_sets, self.decide_rounds = decide_sets, decide_rounds
        self.confirm_sets = confirm_sets
        self._log = log or (lambda s: None)

    # ------------------------------------------------------------ helpers

    def _say(self, res: Result, s: str) -> None:
        res.log.append(s)
        self._log(s)

    def _load(self, bench, fn, res: Result) -> None:
        """A candidate's first call, outside the GPU budget: loading its
        modules is host work, and a kernel the compiler did not build (a
        reduction's variant, a composite's tail) compiles here."""
        load = getattr(bench, "load", None)
        if load is None:
            return
        dt = load(fn)
        res.counts["loaded"] = res.counts.get("loaded", 0) + 1
        if dt > 2.0:
            res.counts["compiled_late"] = res.counts.get("compiled_late", 0) + 1

    def _round_cost(self, bench, ms: float) -> float:
        return getattr(bench, "copies", 1) * ms * 1e-3 + ROUND_OVERHEAD

    def pool(self, m: int, n: int, k: int, res: Result) -> List[Candidate]:
        sp = self.space
        cur = res.route
        seeds = sp.seeds(m, n, k)
        if self.max_seeds is not None:
            seeds = seeds[:self.max_seeds]
        gen = sp.generate(m, n, k)
        res.counts["generated"] = len(gen)
        legal, est = [], {}
        seen = set()
        for c in [cur] + seeds + gen:
            if c in seen:
                continue
            seen.add(c)
            why = sp.check(c, m, n, k)
            if why:
                res.refused[c.label] = why
                continue
            legal.append(c)
            est[c] = sp.estimate(c, m, n, k)
        res.counts["legal"] = len(legal)
        chosen = [cur] + [c for c in seeds if c in est and c != cur]
        # One representative per structure (kind, block, K split): the model
        # cannot tell warp grids, stages or swizzles apart -- local search
        # does that, by measurement -- so a pool of its top N would be N
        # copies of one block. The representative is the one nearest the
        # kernel's defaults.
        groups: Dict[tuple, Candidate] = {}
        for c in legal:
            if c in chosen:
                continue
            g = sp.structure(c, m, n, k)
            if g not in groups or (sp.preference(c), c.label) < (sp.preference(groups[g]),
                                                                 groups[g].label):
                groups[g] = c
        by_kind: Dict[str, List[Candidate]] = {}
        for c in groups.values():
            by_kind.setdefault(c.kind, []).append(c)
        best = min(est.values()) if est else 0.0
        rng = random.Random(f"{m}x{n}x{k}")
        for kind, cs in sorted(by_kind.items()):
            cs.sort(key=lambda c: (est[c], c.label))
            # The model orders; it does not get to drop a kind whose best is
            # within 4x of the overall best (it is that rough).
            top = [c for c in cs[:self.pool_per_kind] if est[c] <= 4 * best + 1e-9] or cs[:2]
            # And a few structures past its cut, drawn at random among those
            # it puts within 2x of the best: what the model ranks wrong is
            # found only by measuring something it did not pick.
            rest = [c for c in cs[len(top):] if est[c] <= 2 * best + 1e-9]
            chosen += top + rng.sample(rest, min(self.explore, len(rest)))
        self._est = est
        return chosen

    # ------------------------------------------------------------ the run

    def run(self, m: int, n: int, k: int, bench, compiler, *,
            baselines: Optional[Dict[str, Callable]] = None,
            arch: Optional[str] = None) -> Result:
        sp = self.space
        t_wall = time.perf_counter()
        gpu0 = bench.gpu_seconds
        load0 = getattr(bench, "load_seconds", 0.0)
        res = Result(m, n, k, arch or sp.arch, sp.current(m, n, k))
        cur = res.route
        floor = stats.floor_ms(m, n, k, res.arch)
        budget_end = gpu0 + self.budget
        reserve = 0.25 * self.budget

        def left() -> float:
            return budget_end - bench.gpu_seconds

        pool = self.pool(m, n, k, res)
        res.counts["pool"] = len(pool)
        self._say(res, f"{m}x{n}x{k}: route {cur}; {len(pool)} candidates to compile "
                       f"({res.counts['legal']} legal of {res.counts['generated']} generated)")
        for c in pool:
            compiler.submit(c, sp.jobs(c, m, n, k))

        # Check every candidate as its build lands; the route first.
        runners: Dict[Candidate, Callable] = {}
        wrong = refused = failed = 0
        for c in pool:
            why = compiler.result(c)
            if why:
                res.refused[c.label] = why
                refused += 1
                continue
            try:
                fn = sp.runner(c, m, n, k)
                self._load(bench, fn, res)
                err = bench.check(fn)
            except Exception as e:  # noqa: BLE001 -- one bad candidate is one row
                # Refused at its first call (a factory's own check that the
                # schedule's check() let through), or it raised.
                res.refused[c.label] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                failed += 1
                continue
            if not err < 2e-2:
                res.refused[c.label] = f"wrong: relative error {err:.3g} against fp32"
                wrong += 1
                continue
            runners[c] = fn
        res.counts.update(refused=refused, wrong=wrong, failed=failed)
        if cur not in runners:
            self._say(res, f"  the route {cur} itself did not run: "
                           f"{res.refused.get(cur.label, '?')}")
            res.wall_seconds = time.perf_counter() - t_wall
            res.gpu_seconds = bench.gpu_seconds - gpu0
            return res

        res.phases["check"] = round(bench.gpu_seconds - gpu0, 2)
        # Successive halving.
        alive = list(runners)
        times: Dict[Candidate, float] = {}
        est = getattr(self, "_est", {})
        for i, rounds in enumerate(self.rung_rounds):
            ms_guess = {c: times.get(c, est.get(c, 1e-3) * 1.5e3) for c in alive}
            cost = sum(rounds * self._round_cost(bench, ms_guess[c]) for c in alive)
            if i > 0 and cost > left() - reserve:
                self._say(res, f"  rung {i}: {len(alive)} arms x {rounds} rounds would pass "
                               f"the budget; stopping the rungs")
                break
            bench.refresh()
            got = bench.time({c: runners[c] for c in alive}, rounds)
            for c, ts in got.items():
                times[c] = stats.summary(ts, floor)[0] or float("inf")
            keep = max(4, math.ceil(len(alive) / 3)) if i + 1 < len(self.rung_rounds) else 4
            alive = stats.halve(got, keep, protect=[cur], floor=floor)
            self._say(res, f"  rung {i}: {len(got)} timed x {rounds} rounds, best "
                           + ", ".join(f"{c} {_ms(times[c])}" for c in alive[:3]))
            if len(alive) <= 4 and i >= 1:
                break
        res.counts["timed"] = len(times)

        res.phases["rungs"] = round(bench.gpu_seconds - gpu0 - sum(res.phases.values()), 2)
        # Local search around the best two, while it improves.
        tried = set(runners) | set(c for c in pool)
        for it in range(self.local_iters):
            ranked = [c for c in sorted(alive, key=lambda c: times[c])]
            tops = ranked[:2]
            moves: List[Candidate] = []
            for t in tops:
                cands = [c for c in sp.neighbors(t, m, n, k) if c not in tried]
                legal = []
                for c in cands:
                    tried.add(c)
                    why = sp.check(c, m, n, k)
                    if why:
                        res.refused[c.label] = why
                    else:
                        legal.append(c)
                legal.sort(key=lambda c: (sp.estimate(c, m, n, k), c.label))
                moves += legal[:self.local_moves]
            if not moves:
                break
            rounds = self.rung_rounds[min(1, len(self.rung_rounds) - 1)]
            arms_n = len(moves) + 3
            cost = arms_n * rounds * self._round_cost(bench, times[tops[0]])
            if cost > left() - reserve:
                self._say(res, f"  local {it}: {len(moves)} moves would pass the budget; stopping")
                break
            for c in moves:
                compiler.submit(c, sp.jobs(c, m, n, k))
            ok = []
            for c in moves:
                why = compiler.result(c)
                if why:
                    res.refused[c.label] = why
                    continue
                try:
                    fn = sp.runner(c, m, n, k)
                    self._load(bench, fn, res)
                    err = bench.check(fn)
                except Exception as e:  # noqa: BLE001
                    res.refused[c.label] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    continue
                if not err < 2e-2:
                    res.refused[c.label] = f"wrong: relative error {err:.3g} against fp32"
                    continue
                runners[c] = fn
                ok.append(c)
            if not ok:
                break
            inc = list(dict.fromkeys(ranked[:3] + [cur]))
            bench.refresh()
            got = bench.time({c: runners[c] for c in inc + ok}, rounds)
            for c, ts in got.items():
                times[c] = stats.summary(ts, floor)[0] or float("inf")
            res.counts["timed"] = len(times)
            new_best = min(got, key=lambda c: times[c])
            self._say(res, f"  local {it}: {len(ok)} moves timed, best {new_best} "
                           f"{_ms(times[new_best])}")
            alive = stats.halve(got, 4, protect=[cur], floor=floor)
            if new_best not in ok:
                break

        # The champion: the fastest of what is left that is not the route.
        others = sorted((c for c in alive if c != cur), key=lambda c: times[c])
        if not others:
            others = sorted((c for c in times if c != cur), key=lambda c: times[c])
        champ = others[0] if others else None
        runner_up = others[1] if len(others) > 1 else None
        res.champion = champ

        res.phases["local"] = round(bench.gpu_seconds - gpu0 - sum(res.phases.values()), 2)
        # Decide, then confirm, with the baselines in the same sets.
        arms: Dict[Any, Callable] = {cur: runners[cur]}
        if champ is not None:
            arms[champ] = runners[champ]
        if runner_up is not None:
            arms[runner_up] = runners[runner_up]
        for name, fn in (baselines or {}).items():
            arms[f"base:{name}"] = fn
        sets: Dict[Any, List[List[float]]] = {a: [] for a in arms}

        def measure(n_sets: int) -> None:
            for _ in range(n_sets):
                bench.refresh()
                got = bench.time(arms, self.decide_rounds)
                for a, ts in got.items():
                    sets[a].append(ts)

        measure(self.decide_sets)
        if champ is not None:
            first = stats.decide(sets[cur], sets[champ], min_gain=self.min_gain,
                                 k_noise=self.k_noise, floor=floor)
            dec = first
            if first.verdict == "win" and self.confirm_sets:
                n0 = len(sets[cur])
                measure(self.confirm_sets)
                again = stats.decide(sets[cur][n0:], sets[champ][n0:], min_gain=self.min_gain,
                                     k_noise=self.k_noise, floor=floor)
                dec = stats.confirm(first, again)
            res.decision = dec
            res.ms = stats.median([stats.summary(s, floor)[0] for s in sets[champ]])
        res.route_ms = stats.median([stats.summary(s, floor)[0] for s in sets[cur]])
        acc = champ if (res.decision and res.decision.verdict == "win") else cur
        for name in baselines or {}:
            b = sets[f"base:{name}"]
            res.baselines[name] = dict(
                ms=_r(stats.median([stats.summary(s, floor)[0] for s in b])),
                ratio=[_r(x) for x in stats.ratio_band(b, sets[acc], floor)],
                route_ratio=[_r(x) for x in stats.ratio_band(b, sets[cur], floor)])
        res.timed = {str(c): _r(t) for c, t in sorted(times.items(), key=lambda x: x[1])[:12]}
        res.gpu_seconds = bench.gpu_seconds - gpu0
        res.phases["decide"] = round(res.gpu_seconds - sum(res.phases.values()), 2)
        res.wall_seconds = time.perf_counter() - t_wall
        res.compile_seconds = getattr(compiler, "cpu_seconds", 0.0)
        res.load_seconds = getattr(bench, "load_seconds", 0.0) - load0
        if res.decision:
            d = res.decision
            self._say(res, f"  decision: {champ} {d.verdict} {d.gain * 100:+.2f}% "
                           f"(bar {d.bar * 100:.2f}%, band {d.band[0]:.3f}-{d.band[1]:.3f})")
        return res
