"""What a timing means: the statistics and the acceptance rule of `hk.tune`.

Torch-free and GPU-free, so the rules are tested on synthetic timings
(tests/hk/ir/test_tune.py) exactly as they run on measured ones.

The rules are the project's (docs/history/31-32):

  * **An arm's time is the median of its rounds**, its noise their median
    absolute deviation over that median. A round faster than the card can go
    (FLOPs past the arch's peak with 15% of headroom) is dropped first, as
    tools/hk-bench/_rounds.py's `ceiling` rule drops it.
  * **Interleaved, in one process**: every set times the candidate and the
    route it would replace in the same rounds, on activations of its own.
  * **A margin from the measured spread**: a candidate replaces the route only
    if it is faster by `max(min_gain, k_noise x sigma)`, sigma the larger of
    the rounds' spread inside a set (both arms') and the sets' spread about
    their median ratio. Anything inside that is a tie -- and a tie keeps the
    route: the route is the kernel every other shape runs, and a record that
    names a schedule by a coin flip pins it for good.
  * **Confirmation**: the decision is made again on fresh sets, measured after
    it; both must say "win".
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence

#: The smallest gain that changes a route (hk.autotune.MIN_GAIN's 1%).
MIN_GAIN = 0.01
#: The margin, in units of the measured noise.
K_NOISE = 2.0
#: Headroom over the card's peak before a round is "faster than the card".
CEILING_MARGIN = 1.15
#: Dense bf16 peaks, FLOP/s (the datasheets'; tools/hk-bench/_rounds.CARDS).
PEAK_FLOPS = {"gfx950": 2.52e15, "gfx1100": 122.6e12, "gfx1201": 197e12}


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs)


def spread(xs: Sequence[float]) -> float:
    """Median absolute deviation over the median, as a fraction."""
    if len(xs) < 2:
        return 0.0
    med = statistics.median(xs)
    return statistics.median(abs(x / med - 1) for x in xs) if med > 0 else 0.0


def floor_ms(m: int, n: int, k: int, arch: str) -> Optional[float]:
    """The shortest plausible time of one call, ms."""
    peak = PEAK_FLOPS.get(arch)
    return 2.0 * m * n * k / (peak * CEILING_MARGIN) * 1e3 if peak else None


def screen(times: Sequence[float], floor: Optional[float] = None) -> List[float]:
    """The rounds that count: finite, positive, and no faster than the card."""
    ok = [t for t in times if t and t > 0 and math.isfinite(t)]
    if floor:
        kept = [t for t in ok if t >= floor]
        return kept
    return ok


def summary(times: Sequence[float], floor: Optional[float] = None):
    """(median ms, spread, rounds kept) of one arm's rounds; median None if
    no round counts."""
    ok = screen(times, floor)
    if not ok:
        return None, 0.0, 0
    return statistics.median(ok), spread(ok), len(ok)


def rank(scores: Dict[Hashable, Sequence[float]], floor: Optional[float] = None) -> List[Hashable]:
    """Arms by median time, fastest first; arms with no counted round last."""
    med = {a: summary(t, floor)[0] for a, t in scores.items()}
    return sorted(scores, key=lambda a: (med[a] is None, med[a] or 0.0, str(a)))


def halve(scores: Dict[Hashable, Sequence[float]], keep: int,
          protect: Iterable[Hashable] = (), floor: Optional[float] = None) -> List[Hashable]:
    """Successive halving's cut: the `keep` fastest by median, plus every
    protected arm (the route), in rank order. An arm within its own noise of
    the last kept one is kept too: a cut is not a decision, and cutting a
    near-tie by one noisy median throws away the arm the next rung would
    have told apart."""
    order = rank(scores, floor)
    kept = order[:keep]
    if kept:
        last = summary(scores[kept[-1]], floor)
        for a in order[keep:]:
            t, s, _ = summary(scores[a], floor)
            if t is None or last[0] is None:
                break
            if t <= last[0] * (1 + 2 * max(s, last[1])):
                kept.append(a)
            else:
                break
    for p in protect:
        if p in scores and p not in kept:
            kept.append(p)
    return kept


@dataclass
class Decision:
    """A candidate against the route it would replace, over sets."""

    verdict: str                  # "win", "tie" or "loss"
    gain: float                   # median over sets of route/candidate - 1
    sigma: float                  # the noise the margin is drawn from
    bar: float                    # the gain a win must clear
    band: List[float] = field(default_factory=list)   # [min, max] ratio over sets
    ratios: List[float] = field(default_factory=list)
    sets: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("gain", "sigma", "bar"):
            d[key] = round(d[key], 5)
        d["band"] = [round(x, 4) for x in d["band"]]
        d["ratios"] = [round(x, 4) for x in d["ratios"]]
        return d


def decide(route_sets: Sequence[Sequence[float]], cand_sets: Sequence[Sequence[float]], *,
           min_gain: float = MIN_GAIN, k_noise: float = K_NOISE,
           floor: Optional[float] = None) -> Decision:
    """Whether the candidate beats the route, from S sets of rounds timed
    interleaved (route_sets[s] and cand_sets[s] in the same rounds)."""
    ratios, within = [], []
    for r, c in zip(route_sets, cand_sets):
        tr, sr, _ = summary(r, floor)
        tc, sc, _ = summary(c, floor)
        if tr is None or tc is None:
            continue
        ratios.append(tr / tc)
        within.append(sr + sc)
    if not ratios:
        return Decision("loss", 0.0, 0.0, min_gain, [], [], 0)
    g = statistics.median(ratios) - 1
    across = (max(ratios) - min(ratios)) / 2 / statistics.median(ratios) if len(ratios) > 1 else 0.0
    sigma = max(statistics.median(within), across)
    bar = max(min_gain, k_noise * sigma)
    verdict = "win" if g >= bar else ("loss" if g <= -bar else "tie")
    return Decision(verdict, g, sigma, bar, [min(ratios), max(ratios)], ratios, len(ratios))


def confirm(first: Decision, again: Decision) -> Decision:
    """A win only if the decision and its confirmation both say so; the
    numbers are pooled over both."""
    ratios = first.ratios + again.ratios
    if not ratios:
        return first
    g = statistics.median(ratios) - 1
    sigma = max(first.sigma, again.sigma)
    bar = max(first.bar, again.bar)
    if first.verdict == "win" and again.verdict == "win":
        verdict = "win"
    elif first.verdict == "loss" and again.verdict == "loss":
        verdict = "loss"
    else:
        verdict = "tie"
    return Decision(verdict, g, sigma, bar, [min(ratios), max(ratios)], ratios,
                    first.sets + again.sets)


def ratio_band(base_sets: Sequence[Sequence[float]], arm_sets: Sequence[Sequence[float]],
               floor: Optional[float] = None):
    """(median, lo, hi) of base/arm over sets: an arm's speed against a
    baseline, >1 when the arm is faster."""
    rs = []
    for b, a in zip(base_sets, arm_sets):
        tb = summary(b, floor)[0]
        ta = summary(a, floor)[0]
        if tb and ta:
            rs.append(tb / ta)
    if not rs:
        return None, None, None
    return statistics.median(rs), min(rs), max(rs)
