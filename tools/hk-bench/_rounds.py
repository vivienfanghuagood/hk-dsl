"""Which of an arm's timed rounds to believe, and what to report from them.

The scoreboard times each arm in R rounds of C calls, each round a profiler
session of its own whose kernel records are summed and divided by C. A round
can come out faster than the kernel is:
  * the session lost kernel records, so its sum covers fewer calls than it is
    divided by (ROCm's profiler is unreliable outside ATen, scoreboard.py's
    gpu_times says how);
  * anything else that happened once and is not the kernel.
Keeping the fastest round keeps exactly such a round. One did: an 8192x4096
RMSNorm round of 97.4 us, 1.38 TB/s for a 134 MB call on 864 GB/s of GDDR6,
when every other round of seven runs took 173 (docs/history/31).

So a round is rejected, with its reason, by any of three rules, in order:

  records   every round of an arm launches the same kernels, so its kernel
            record count is the arm's most common one. A round with another
            count is rejected (if no count is held by more than half the
            rounds, the rule says nothing).
  ceiling   no round runs faster than the card can: FLOPs at most PEAK_FLOPS;
            bytes at most IC_BYTES_S when the call's footprint (the distinct
            bytes it reads and writes) fits the 96 MB Infinity Cache, where
            back-to-back calls find it resident, and at most HBM_BYTES_S when
            it does not; each times MARGIN. The cache's rate is measured
            (`scoreboard.py --probe-ceilings`), not the datasheet's.
  lone      of the rounds left, sorted, the fastest is rejected when it is
            more than max(GAP_FLOOR, GAP_K x s) ahead of the second, where s
            is how far the other rounds' median sits above the second: a best
            round no other round comes near. Repeated while more than half
            the rounds remain.

The arm's time is then STAT of the kept rounds -- their median -- and its
`spread` the kept rounds' median absolute deviation over that median. The
best kept round is reported beside it.

Torch-free, so check_regression.py and the no-GPU tests use the same rules.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

#: The card's bf16 peak (docs/history/30: "a 123 TFLOP/s card"); hk's GEMM
#: runs at 84-88, FlyDSL's at 89-91.
PEAK_FLOPS = 122.6e12
#: GDDR6 peak, 384 bits at 18 Gbps; a plain read reaches 680-820 GB/s
#: (hbm_probe.hip), the norms at 8192 rows 770-790.
HBM_BYTES_S = 864e9
#: Infinity Cache size, and the fastest cache-resident rate measured on this
#: card by `scoreboard.py --probe-ceilings`: a two-input add at 80-96 MB,
#: 2.35-2.43 TB/s (a copy reaches 1.77; past the cache both 0.72-0.75).
IC_BYTES = 96 * 2**20
IC_BYTES_S = 2.43e12
#: Headroom over each ceiling: a ceiling is "faster than this card can", not
#: "faster than anything measured so far".
MARGIN = 1.15
GAP_FLOOR = 0.05
GAP_K = 4.0
STATS = ("median", "best")
STAT = "median"

#: The same ceilings for each card a report can come from, by arch. gfx1100's
#: are the figures above. gfx950 (MI355X): the bf16 dense peak, HBM3E's
#: 8 TB/s, and the 256 MB Infinity Cache at the fastest cache-resident rate
#: `--probe-ceilings` measured: an in-place add (a += b) at a 32 MB
#: footprint, inside the eight XCDs' 4 MB L2s, 9.1 TB/s. torch's out-of-place
#: add and neg peak at 6.9-7.2 at 160-256 MB, and past the cache all three
#: sit at 5.8-6.1; at 7.2 the rule rejected a fused add's real 8.35 TB/s.
#: gfx1201 (Radeon AI PRO R9700), measured: back-to-back
#: v_wmma_f32_16x16x16_bf16 on every CU, 197.1 TFLOP/s (above the datasheet's
#: 191 at the boost clock; tools/rdna4-probes/wmma_peaks.hip);
#: GDDR6 at 20 Gbps x 256 bits, 640 GB/s (past the cache a copy and an add
#: reach 0.62); and the 64 MB Infinity Cache at `--probe-ceilings`' fastest
#: cache-resident rate, an add at a 48 MB footprint, 1.81 TB/s.
CARDS = {
    "gfx1100": {"PEAK_FLOPS": PEAK_FLOPS, "HBM_BYTES_S": HBM_BYTES_S,
                "IC_BYTES": IC_BYTES, "IC_BYTES_S": IC_BYTES_S},
    "gfx1201": {"PEAK_FLOPS": 197e12, "HBM_BYTES_S": 640e9,
                "IC_BYTES": 64 * 2**20, "IC_BYTES_S": 1.81e12},
    "gfx950": {"PEAK_FLOPS": 2.52e15, "HBM_BYTES_S": 8.0e12,
               "IC_BYTES": 256 * 2**20, "IC_BYTES_S": 9.1e12},
}


def use(arch: str) -> None:
    """Screen against `arch`'s ceilings (CARDS); an arch with none keeps the
    current ones."""
    card = CARDS.get(arch)
    if card:
        globals().update(card)


def ceiling_rate(unit: str, footprint: Optional[float]) -> Optional[float]:
    """The fastest plausible rate (FLOP/s or bytes/s), or None if unknown."""
    if unit == "flop":
        return PEAK_FLOPS * MARGIN
    if unit == "byte" and footprint:
        return (IC_BYTES_S if footprint <= IC_BYTES else HBM_BYTES_S) * MARGIN
    return None


def floor_ms(work: float, unit: str, footprint: Optional[float] = None) -> Optional[float]:
    """The shortest plausible time of one call, in ms, or None if unknown."""
    rate = ceiling_rate(unit, footprint if unit == "byte" else None)
    return work / rate * 1e3 if rate and work else None


def screen(times: Sequence[float], records: Optional[Sequence[int]] = None,
           floor: Optional[float] = None) -> Tuple[List[int], Dict[int, str]]:
    """(indices kept, {index: reason}) for one arm's rounds, in round order."""
    n = len(times)
    rejected: Dict[int, str] = {}
    ok = [i for i in range(n) if 0 < times[i] < float("inf")]
    for i in range(n):
        if i not in ok:
            rejected[i] = "no kernel records"
    if records and len(records) == n and ok:
        (mode, held), = Counter(records[i] for i in ok).most_common(1)
        if held * 2 > len(ok):
            for i in list(ok):
                if records[i] != mode:
                    rejected[i] = (f"records: {records[i]} kernel records, the arm's "
                                   f"other rounds {mode}")
                    ok.remove(i)
    if floor:
        for i in list(ok):
            if times[i] < floor:
                rejected[i] = (f"ceiling: {times[i]:.5g} ms, faster than the card "
                               f"can ({floor:.5g} ms at the ceiling)")
                ok.remove(i)
    kept = sorted(ok, key=lambda i: times[i])
    while len(kept) >= 3 and len(kept) * 2 > n + 1:
        t1, t2 = times[kept[0]], times[kept[1]]
        rest = statistics.median(times[j] for j in kept[1:]) / t2 - 1
        gap = t2 / t1 - 1
        if gap <= max(GAP_FLOOR, GAP_K * rest):
            break
        rejected[kept[0]] = (f"lone: {gap:.1%} ahead of the next round, the others "
                             f"within {rest:.1%} of it")
        kept.pop(0)
    return sorted(kept), rejected


def summarize(times: Sequence[float], kept: Sequence[int],
              stat: str = STAT) -> Tuple[Optional[float], float, Optional[float]]:
    """(the arm's time, its spread, the best kept round) over the kept rounds."""
    ts = sorted(times[i] for i in kept)
    if not ts:
        return None, 0.0, None
    if stat == "best":
        return ts[0], (ts[1] / ts[0] - 1 if len(ts) > 1 else 0.0), ts[0]
    if stat != "median":
        raise ValueError(f"stat {stat!r}; one of {STATS}")
    med = statistics.median(ts)
    return med, statistics.median(abs(t / med - 1) for t in ts), ts[0]


def audit(report: dict) -> List[str]:
    """What in a scoreboard report would not pass these rules, one line each.

    A report written with rounds lists the rounds the scoreboard rejected. One
    written before keeps only the best round and the second's distance from it
    (`spread`), so it is checked as far as that allows: the best round against
    the ceiling, rejected; and a best round more than LONE_OLD ahead of the
    second, a candidate for `lone` (whether it is one depends on the other
    rounds, which such a report does not have)."""
    lines = []
    for c in report.get("cases", []):
        work, unit = c.get("work"), c.get("unit")
        floor = floor_ms(work, unit, footprint_of(c)) if work and unit else None
        for a, v in c.get("arms", {}).items():
            if v.get("rounds"):
                for i, why in sorted((v.get("rejected") or {}).items(), key=lambda x: int(x[0])):
                    t = v["rounds"][int(i)]
                    lines.append(f"{c['case']} / {a}: round {int(i) + 1} of "
                                 f"{len(v['rounds'])}, {'-' if t is None else f'{t:.5g}'} "
                                 f"ms -- {why}")
                continue
            ms, spread = v.get("ms"), v.get("spread") or 0.0
            if not ms:
                continue
            if floor and ms < floor:
                lines.append(f"{c['case']} / {a}: best round {ms:.5g} ms -- ceiling: faster "
                             f"than the card can ({floor:.5g} ms at the ceiling)")
            elif spread > LONE_OLD:
                lines.append(f"{c['case']} / {a}: best round {ms:.5g} ms, {spread:.1%} ahead "
                             f"of the second -- lone, if the other rounds sit within "
                             f"{spread / GAP_K:.1%} of the second")
    return lines


#: The gap past which audit() names an old report's best round as a `lone`
#: candidate.
LONE_OLD = 0.20


def footprint_of(case: dict) -> Optional[float]:
    """The distinct bytes a case's call touches; reports from before the
    scoreboard recorded it count every byte moved, which is the same for every
    case but the in-place fused add (each byte read and written once)."""
    if case.get("footprint"):
        return case["footprint"]
    work = case.get("work")
    if work and case.get("case", "").startswith("fused_add_rms_norm"):
        return work / 2
    return work
