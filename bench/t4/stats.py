"""What a T4 timing means, and what passes. Pure Python: the CPU tests run
these exact functions on synthetic timings.

The measurement rules are the project's (docs/history/31-32, hk.tune.stats):

  * an arm's time in a set is the **median of its rounds**, after dropping a
    round faster than the card can go (the FLOPs past the arch's dense bf16
    peak with 15% headroom); its noise, the rounds' median absolute
    deviation over that median;
  * every arm is timed **interleaved in one process**, in the same rounds,
    on the same inputs; sets repeat that on fresh activations;
  * an arm's time for a shape is the **median over sets** of its set medians;
    the **best baseline** is the correct baseline with the smallest one;
  * the solution's **ratio** is best-baseline time / solution time per set
    (> 1: the solution is faster); a shape's ratio is the median over sets,
    reported with the sets' range.

The pass rule (task.md) is `verdict()`.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence

#: Headroom over the card's peak before a round is "faster than the card".
CEILING_MARGIN = 1.15
#: Dense bf16 peaks, FLOP/s (the datasheets', as hk.tune.stats has them).
PEAK_FLOPS = {"gfx950": 2.52e15, "gfx1100": 122.6e12}

#: DRAM bandwidth, bytes/s (the datasheets', as hk.check.CARDS has them).
PEAK_BW = {"gfx950": 8.0e12, "gfx1100": 864e9}

#: The pass rule, per GPU, over the sealed shapes.
PASS_GEOMEAN = 0.95      # geometric mean of the shapes' ratios
PASS_MIN = 0.85          # no shape below this
TARGET = 0.95            # a shape "reaches" the target at this ratio (reported)


def floor_ms(flop: float, arch: str) -> Optional[float]:
    """The shortest plausible time of one call, ms."""
    peak = PEAK_FLOPS.get(arch)
    return flop / (peak * CEILING_MARGIN) * 1e3 if peak else None


def roofline_us(flop: float, nbytes: float, arch: str) -> Optional[float]:
    """The datasheet roofline of one call, us: max(FLOPs / peak, bytes / DRAM BW)."""
    if arch not in PEAK_FLOPS:
        return None
    return max(flop / PEAK_FLOPS[arch], nbytes / PEAK_BW[arch]) * 1e6


def screen(times: Sequence[float], floor: Optional[float] = None) -> List[float]:
    ok = [t for t in times if t is not None and t > 0 and math.isfinite(t)]
    return [t for t in ok if t >= floor] if floor else ok


def spread(xs: Sequence[float]) -> float:
    """Median absolute deviation over the median, as a fraction."""
    if len(xs) < 2:
        return 0.0
    med = statistics.median(xs)
    return statistics.median(abs(x / med - 1) for x in xs) if med > 0 else 0.0


def summary(times: Sequence[float], floor: Optional[float] = None):
    """(median ms, spread, rounds kept) of one arm's rounds in one set."""
    ok = screen(times, floor)
    if not ok:
        return None, 0.0, 0
    return statistics.median(ok), spread(ok), len(ok)


def arm_time(sets: Sequence[Sequence[float]], floor: Optional[float] = None):
    """(median over sets of the set medians, median spread) or (None, 0)."""
    meds = [summary(s, floor) for s in sets]
    got = [m for m, _, _ in meds if m]
    if not got:
        return None, 0.0
    return statistics.median(got), statistics.median(sp for m, sp, _ in meds if m)


def ratio_band(base_sets, arm_sets, floor: Optional[float] = None):
    """(median, lo, hi) of base/arm over sets; (None, None, None) if no set
    has both."""
    rs = []
    for b, a in zip(base_sets, arm_sets):
        tb, ta = summary(b, floor)[0], summary(a, floor)[0]
        if tb and ta:
            rs.append(tb / ta)
    if not rs:
        return None, None, None
    return statistics.median(rs), min(rs), max(rs)


def best_baseline(sets_by_arm: Dict[str, Sequence[Sequence[float]]], baselines: Sequence[str],
                  floor: Optional[float] = None) -> Optional[str]:
    """The baseline with the smallest time over sets (ties: by name)."""
    t = {b: arm_time(sets_by_arm[b], floor)[0] for b in baselines if b in sets_by_arm}
    t = {b: v for b, v in t.items() if v}
    return min(t, key=lambda b: (t[b], b)) if t else None


def score_shape(sets_by_arm: Dict[str, Sequence[Sequence[float]]], baselines: Sequence[str],
                solution: Optional[str], floor: Optional[float] = None) -> Dict:
    """One shape's numbers from its timings: every arm's time and spread,
    the best baseline, and the solution's ratio band against it."""
    ms, sp = {}, {}
    for a, s in sets_by_arm.items():
        ms[a], sp[a] = arm_time(s, floor)
    best = best_baseline(sets_by_arm, baselines, floor)
    out = dict(ms=ms, spread=sp, best=best, ratio=None, band=None)
    if solution and best and solution in sets_by_arm:
        r, lo, hi = ratio_band(sets_by_arm[best], sets_by_arm[solution], floor)
        out["ratio"], out["band"] = r, (lo, hi) if r else None
    return out


def geomean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def verdict(rows: Sequence[Dict]) -> Dict:
    """The pass rule over one GPU's shapes. Each row: `correct` (bool: every
    checked routing within tolerance), `audit_ok` (bool: the solution's GPU
    work is hk's, task.md), `ratio` (float or None).

    PASS when every shape is correct and audited clean, the geometric mean of
    the ratios is >= PASS_GEOMEAN and no shape is below PASS_MIN. A shape that
    is wrong, fails the audit or was not timed counts as ratio 0."""
    n = len(rows)
    wrong = [r.get("name") for r in rows if not r.get("correct")]
    dirty = [r.get("name") for r in rows if r.get("correct") and not r.get("audit_ok")]
    ratios = [r.get("ratio") if (r.get("correct") and r.get("audit_ok") and r.get("ratio"))
              else 0.0 for r in rows]
    gm = geomean(ratios) if all(x > 0 for x in ratios) and ratios else 0.0
    low = min(ratios) if ratios else 0.0
    reached = sum(1 for x in ratios if x >= TARGET)
    ok = bool(n) and not wrong and not dirty and gm >= PASS_GEOMEAN and low >= PASS_MIN
    why = []
    if not n:
        why.append("no shapes")
    if wrong:
        why.append(f"wrong on {len(wrong)}: {', '.join(map(str, wrong[:5]))}")
    if dirty:
        why.append(f"audit failed on {len(dirty)}: {', '.join(map(str, dirty[:5]))}")
    if n and gm < PASS_GEOMEAN:
        why.append(f"geomean {gm:.3f} < {PASS_GEOMEAN}")
    if n and low < PASS_MIN:
        why.append(f"min {low:.3f} < {PASS_MIN}")
    return dict(passed=ok, shapes=n, geomean=round(gm, 4) if gm else 0.0,
                min=round(low, 4), reached=reached, target=TARGET,
                rule=f"correct + audit clean on every shape, geomean >= {PASS_GEOMEAN}, "
                     f"min >= {PASS_MIN}",
                why="; ".join(why) if why else "pass")
