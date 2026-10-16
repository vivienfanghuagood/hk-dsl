#!/usr/bin/env python3
"""The performance gate: hk's kernels against the arms beside them, as ratios
measured in one run, held to a baseline.

The node is shared, and absolute times drift 3-5% between sessions -- more
than most regressions worth catching. Within one scoreboard run that drift is
common to every arm, so the gate never compares a time with a time from
another run. For each case and each baseline arm A it takes r = t_A / t_hk
from the run in hand (above 1 is hk faster) and fails if r fell below the
baseline's ratio by more than that pair's tolerance.

The tolerance is measured, not chosen. A baseline is built from several
scoreboard runs, each its own process (`--refresh`). For each (case, arm) the
ratio is the median across runs, and the noise is the larger of
  * across runs: max over runs of |r_i / median - 1|, and
  * within a run: hk's and A's round-to-round spreads, summed, at their
    largest over the runs (the scoreboard's `spread`, second-best round
    against best);
and tolerance = max(FLOOR, MULTIPLIER x noise). A pair whose tolerance would
pass CAP is too noisy to gate on; it is listed as not gated, with its noise.
The baseline file carries all of it, so a tolerance can be checked against
the noise it came from rather than loosened.

    # check a run against the baseline; exit 1 on a regression
    python3 tools/hk-bench/check_regression.py run.json
    # the scoreboard arguments the baseline was measured with
    python3 tools/hk-bench/check_regression.py --print-args
    # a new baseline from three or more runs (scripts/perf_gate.sh --refresh)
    python3 tools/hk-bench/check_regression.py --refresh r1.json r2.json r3.json

scripts/perf_gate.sh runs the scoreboard with the baseline's arguments and
then this; docs/testing.md#the-performance-gate says when to refresh.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO / "tests" / "perf" / "baselines" / "gfx1100.json"

#: The scoreboard run a baseline is measured with, and the gate repeats: the
#: out-of-the-box arms only (the tuned ones search on every run), every call
#: count at 50 so the microsecond norms are measurable at all.
ARGS = ["--only", "gemm,attn,norm,softmax", "--arms", "hk,torch,inductor,triton,flydsl",
        "--rounds", "7", "--calls", "50"]
HK = "hk"
#: No tolerance below this, however quiet the baseline runs were: three runs
#: see less than the worst a fourth will.
FLOOR = 0.03
MULTIPLIER = 3.0
#: A pair noisier than this (tolerance, after the multiplier) is not gated.
CAP = 0.15

TOLERANCE_NOTE = (
    "tol = max(floor, multiplier x noise), per (case, arm). noise is the larger of the "
    "ratio's spread across the baseline runs (max |r_i / median - 1|) and its spread "
    "within a run (the two arms' second-best-round spreads, summed, worst run). "
    "The multiplier covers what a handful of runs does not see; the floor keeps it above "
    "zero. A pair whose tol would exceed cap is not gated. To change a tolerance, "
    "re-measure the noise (--refresh with more runs); do not edit tol by hand.")


def ratios(report: dict) -> dict:
    """{case: {arm: (ratio, within-run noise)}} for every arm timed beside hk."""
    out = {}
    for c in report["cases"]:
        arms = c["arms"]
        h = arms.get(HK) or {}
        if not h.get("ms"):
            out[c["case"]] = None
            continue
        row = {}
        for a, v in arms.items():
            if a == HK or a.startswith("hk") or not v.get("ms"):
                continue
            row[a] = (v["ms"] / h["ms"], (v.get("spread") or 0.0) + (h.get("spread") or 0.0))
        out[c["case"]] = row
    return out


def refresh(paths, out: Path) -> int:
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if len(reports) < 3:
        print("a baseline needs three runs or more: the noise is measured across them")
        return 2
    archs = {r["meta"]["arch"] for r in reports}
    if len(archs) != 1:
        print(f"runs from more than one arch: {sorted(archs)}")
        return 2
    per = [ratios(r) for r in reports]
    cases, skipped = {}, {}
    for case in per[0]:
        if any(p.get(case) is None for p in per):
            skipped[case] = "hk was not timed in every run"
            continue
        arms = set.intersection(*(set(p[case]) for p in per))
        row = {}
        for a in sorted(arms):
            rs = [p[case][a][0] for p in per]
            med = statistics.median(rs)
            across = max(abs(r / med - 1) for r in rs)
            within = max(p[case][a][1] for p in per)
            noise = max(across, within)
            tol = max(FLOOR, MULTIPLIER * noise)
            ent = {"ratio": round(med, 4), "runs": [round(r, 4) for r in rs],
                   "across": round(across, 4), "within": round(within, 4),
                   "tol": round(tol, 4)}
            if tol > CAP:
                skipped[f"{case} / {a}"] = (f"noise {noise:.3f} x {MULTIPLIER:g} = {tol:.3f} "
                                            f"passes the cap {CAP}")
                continue
            row[a] = ent
        if row:
            cases[case] = row
    meta = reports[0]["meta"]
    pairs = [(e["tol"], max(e["across"], e["within"]), f"{c} / {a}")
             for c, r in cases.items() for a, e in r.items()]
    noise = sorted(n for _, n, _ in pairs)
    widest = max(pairs) if pairs else None
    summary = (f"{len(reports)} runs; {len(pairs)} gated pairs, whose noise has median "
               f"{statistics.median(noise):.4f} and max {noise[-1]:.4f}; "
               f"{sum(t == FLOOR for t, _, _ in pairs)} sit at the floor; the widest "
               f"tolerance is {widest[0]:.3f} ({widest[2]}); {len(skipped)} pairs are "
               f"not gated." if pairs else f"{len(reports)} runs; nothing gated.")
    base = {
        "_about": ("hk's kernels against torch, Inductor, Triton and FlyDSL as ratios "
                   "t_arm / t_hk measured in one scoreboard run; written by "
                   "tools/hk-bench/check_regression.py --refresh, checked by "
                   "scripts/perf_gate.sh."),
        "_tolerance": TOLERANCE_NOTE,
        "_noise": summary,
        "arch": archs.pop(),
        "date": max(r["meta"].get("date", "") for r in reports)[:10] or time.strftime("%Y-%m-%d"),
        "args": ARGS, "floor": FLOOR, "multiplier": MULTIPLIER, "cap": CAP,
        "runs": len(reports),
        "meta": {k: meta.get(k) for k in ("device", "torch", "triton", "flydsl", "hk")},
        "cases": cases, "not_gated": skipped,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, indent=1) + "\n")
    tols = sorted(e["tol"] for r in cases.values() for e in r.values())
    print(f"wrote {out}: {len(tols)} gated pairs over {len(cases)} cases, {len(skipped)} "
          f"not gated" + (f"; tolerance {tols[0]:.3f}-{tols[-1]:.3f}, median "
                          f"{statistics.median(tols):.3f}" if tols else ""))
    return 0


def check(run_path: Path, base_path: Path) -> int:
    base = json.loads(base_path.read_text())
    run = json.loads(run_path.read_text())
    if run["meta"]["arch"] != base["arch"]:
        print(f"the run is {run['meta']['arch']}, the baseline {base['arch']}")
        return 2
    drift = {k: (base["meta"].get(k), run["meta"].get(k)) for k in base["meta"]
             if base["meta"].get(k) != run["meta"].get(k)}
    for k, (was, now) in drift.items():
        print(f"note: {k} was {was} when the baseline was made, is {now}; a moved "
              f"ratio may be the opponent's, and the baseline due a refresh")
    now = ratios(run)
    bad, missing, better = [], [], []
    print(f"{'case':<30}{'arm':<10}{'baseline':>9}{'now':>9}{'change':>9}{'tol':>8}  verdict")
    for case, row in base["cases"].items():
        got = now.get(case)
        for a, ent in row.items():
            if not got or a not in got:
                missing.append(f"{case} / {a}")
                print(f"{case:<30}{a:<10}{ent['ratio']:>9.3f}{'-':>9}{'':>9}"
                      f"{ent['tol']:>8.3f}  MISSING")
                continue
            r = got[a][0]
            change = r / ent["ratio"] - 1
            verdict = "ok"
            if change < -ent["tol"]:
                verdict = "REGRESSION"
                bad.append(f"{case} / {a}: {ent['ratio']:.3f} -> {r:.3f} ({change:+.1%}, "
                           f"tolerance {ent['tol']:.1%})")
            elif change > ent["tol"]:
                verdict = "better"
                better.append(case)
            print(f"{case:<30}{a:<10}{ent['ratio']:>9.3f}{r:>9.3f}{change:>+9.1%}"
                  f"{ent['tol']:>8.3f}  {verdict}")
    for what in base.get("not_gated", {}):
        print(f"not gated: {what} ({base['not_gated'][what]})")
    if better:
        print(f"\n{len(better)} pair(s) beat the baseline by more than their tolerance; "
              f"refresh it to hold the gain")
    if missing:
        print(f"\n{len(missing)} gated pair(s) missing from the run -- an arm that failed, "
              f"or one not installed here (FlyDSL needs FLYDSL_ROOT): {missing}")
    if bad:
        print(f"\n{len(bad)} REGRESSION(S):")
        for b in bad:
            print(f"  {b}")
        return 1
    if missing:
        return 2
    print(f"\nno regression: {sum(len(r) for r in base['cases'].values())} pairs within tolerance")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="scoreboard --json reports")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--refresh", action="store_true",
                    help="write --baseline from the runs given instead of checking")
    ap.add_argument("--print-args", action="store_true",
                    help="print the scoreboard arguments the baseline was measured with")
    args = ap.parse_args()
    if args.print_args:
        p = Path(args.baseline)
        print(" ".join(json.loads(p.read_text())["args"] if p.is_file() else ARGS))
        return 0
    if args.refresh:
        return refresh(args.runs, Path(args.baseline))
    if len(args.runs) != 1:
        ap.error("give one scoreboard report to check")
    return check(Path(args.runs[0]), Path(args.baseline))


if __name__ == "__main__":
    sys.exit(main())
