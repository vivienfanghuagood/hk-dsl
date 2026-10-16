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
    largest over the runs (the scoreboard's `spread`: the median absolute
    deviation of the rounds it kept, or for a report of best rounds the
    second best against the best);
and tolerance = max(FLOOR, MULTIPLIER x noise). A pair whose tolerance would
pass CAP is too noisy to gate on; it is listed as not gated, with its noise.
The baseline file carries all of it, so a tolerance can be checked against
the noise it came from rather than loosened.

Times are believed only as far as _rounds.py's rules allow. The scoreboard
drops the rounds they reject; a time in a report that is still faster than
the card can go (a report from before it did) is left out of a refresh, for
that run only, and makes a check unable to pass (exit 2). A baseline and a
run must time their arms by the same statistic (median of kept rounds, or
best round), or every ratio moves.

The note that the baseline is due a refresh is keyed on what can move a
ratio besides a slower hk: torch, Triton, FlyDSL, the device, the generated
C++ of hk's kernels (`kernels`, compile.built_fingerprint) and hipcc's
version (`hipcc`) -- not hk's version. A HipKittens header change that
leaves the generated C++ alone is not seen.

    # check a run against the baseline; exit 1 on a regression
    python3 tools/hk-bench/check_regression.py run.json
    # the scoreboard arguments the baseline was measured with
    python3 tools/hk-bench/check_regression.py --print-args
    # a new baseline from three or more runs (scripts/perf_gate.sh --refresh)
    python3 tools/hk-bench/check_regression.py --refresh r1.json r2.json r3.json
    # the rounds each report rejected, or would have
    python3 tools/hk-bench/check_regression.py --audit r1.json r2.json

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _rounds  # noqa: E402

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


#: What a moved ratio can be the fault of, besides hk's kernels getting
#: slower: the opponents, the card, and hk's kernels themselves -- their
#: generated C++ and flags (`kernels`) and the compiler (`hipcc`). Not hk's
#: version: a release that changes no kernel changes no ratio. Compared
#: where both the baseline and the run record the key.
DRIFT = ("device", "torch", "triton", "flydsl", "kernels", "hipcc")
#: Recorded beside them, and not compared. `toolchain` hashes every header
#: too, and a header for another arch moves it (chapter 33's CDNA shim did).
INFO = ("hk", "dialect", "toolchain")


def stat_of(report: dict) -> str:
    """How a report's arm times were taken: the statistic over the rounds,
    and whether the rounds' calls were queued behind a GPU spin. Reports
    from before the scoreboard screened its rounds kept the best one, of
    calls launched onto an idle GPU."""
    screen = report["meta"].get("screen") or {}
    stat = screen.get("stat", "best")
    return f"{stat}, queued" if screen.get("queued") else stat


def implausible(report: dict) -> set:
    """{(case, arm)} whose reported time is faster than the card can go. The
    scoreboard rejects such rounds itself now; a report from before it did
    kept them (_rounds.audit)."""
    out = set()
    for c in report["cases"]:
        work, unit = c.get("work"), c.get("unit")
        floor = _rounds.floor_ms(work, unit, _rounds.footprint_of(c)) if work and unit else None
        for a, v in c["arms"].items():
            if floor and v.get("ms") and v["ms"] < floor:
                out.add((c["case"], a))
    return out


def ratios(report: dict) -> dict:
    """{case: {arm: (ratio, within-run noise)}} for every arm timed beside hk,
    leaving out an implausible time; a case whose hk time is one is None."""
    out, bad = {}, implausible(report)
    for c in report["cases"]:
        arms = c["arms"]
        h = arms.get(HK) or {}
        if not h.get("ms") or (c["case"], HK) in bad:
            out[c["case"]] = None
            continue
        row = {}
        for a, v in arms.items():
            if a == HK or a.startswith("hk") or not v.get("ms") or (c["case"], a) in bad:
                continue
            row[a] = (v["ms"] / h["ms"], (v.get("spread") or 0.0) + (h.get("spread") or 0.0))
        out[c["case"]] = row
    return out


def audit(paths) -> int:
    """Print the rounds _rounds.py rejects, or would have, in each report."""
    for p in paths:
        rep = json.loads(Path(p).read_text())
        lines = _rounds.audit(rep)
        kept = any(v.get("rounds") for c in rep["cases"] for v in c["arms"].values())
        print(f"{p} ({rep['meta'].get('date', '?')}; arm times taken as '{stat_of(rep)}', "
              f"{'every round recorded' if kept else 'no other round recorded'}): "
              f"{len(lines) or 'nothing'}")
        for line in lines:
            print(f"  {line}")
    return 0


def refresh(paths, out: Path) -> int:
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if len(reports) < 3:
        print("a baseline needs three runs or more: the noise is measured across them")
        return 2
    archs = {r["meta"]["arch"] for r in reports}
    if len(archs) != 1:
        print(f"runs from more than one arch: {sorted(archs)}")
        return 2
    stats = {stat_of(r) for r in reports}
    if len(stats) != 1:
        print(f"runs whose arm times were taken differently: {sorted(stats)}")
        return 2
    bad = [implausible(r) for r in reports]
    dropped = sorted(f"{p}: {c} / {a}" for p, b in zip(paths, bad) for c, a in b)
    for d in dropped:
        print(f"left out, faster than the card can: {d}")
    per = [ratios(r) for r in reports]
    cases, skipped = {}, {}
    for case in per[0]:
        # A run whose time for the pair was left out does not count for it;
        # one where the pair is missing for any other reason ungates it.
        have = [(p, b) for p, b in zip(per, bad) if (case, HK) not in b]
        if any(p.get(case) is None for p, _ in have) or len(have) < 3:
            skipped[case] = "hk was not timed in every run"
            continue
        arms = set.intersection(*(set(p[case]) | {a for c, a in b if c == case}
                                  for p, b in have))
        row = {}
        for a in sorted(arms):
            use = [p for p, b in zip(per, bad) if p.get(case) and a in p[case]
                   and (case, a) not in b and (case, HK) not in b]
            if len(use) < 3:
                skipped[f"{case} / {a}"] = f"timed plausibly in {len(use)} runs"
                continue
            rs = [p[case][a][0] for p in use]
            med = statistics.median(rs)
            across = max(abs(r / med - 1) for r in rs)
            within = max(p[case][a][1] for p in use)
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
        "runs": len(reports), "stat": stats.pop(),
        "meta": {k: meta.get(k) for k in DRIFT + INFO if meta.get(k) is not None},
        "cases": cases, "not_gated": skipped,
    }
    if dropped:
        base["left_out"] = dropped
    kernels = {r["meta"].get("kernels") for r in reports}
    if len(kernels) > 1:
        print(f"note: the runs measured {len(kernels)} different sets of hk kernels "
              f"(meta.kernels {sorted(map(str, kernels))}); the baseline records the first")
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
    if stat_of(run) != base.get("stat", "best"):
        print(f"the run's arm times were taken as '{stat_of(run)}', the baseline's as "
              f"'{base.get('stat', 'best')}': every ratio would move. Refresh the "
              f"baseline with this scoreboard, or run the one it was made with.")
        return 2
    for k in DRIFT:
        was, now_ = base["meta"].get(k), run["meta"].get(k)
        if was is not None and now_ is not None and was != now_:
            what = ("hk's kernels (their generated C++ or flags)" if k == "kernels" else
                    "hipcc's version" if k == "hipcc" else k)
            print(f"note: {what} changed since the baseline was made ({k} {was} -> {now_}); "
                  f"a moved ratio may be that, and the baseline due a refresh")
    for line in _rounds.audit(run):
        print(f"rejected: {line}")
    now, bad_time = ratios(run), implausible(run)
    bad, missing, better = [], [], []
    print(f"{'case':<30}{'arm':<10}{'baseline':>9}{'now':>9}{'change':>9}{'tol':>8}  verdict")
    for case, row in base["cases"].items():
        got = now.get(case)
        for a, ent in row.items():
            if not got or a not in got:
                missing.append(f"{case} / {a}")
                why = ("IMPLAUSIBLE" if {(case, a), (case, HK)} & bad_time else "MISSING")
                print(f"{case:<30}{a:<10}{ent['ratio']:>9.3f}{'-':>9}{'':>9}"
                      f"{ent['tol']:>8.3f}  {why}")
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
              f"one not installed here (FlyDSL needs FLYDSL_ROOT), or a time faster "
              f"than the card can go: {missing}")
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
    ap.add_argument("--audit", action="store_true",
                    help="list the rounds each report's arms rejected, or would have")
    args = ap.parse_args()
    if args.print_args:
        p = Path(args.baseline)
        print(" ".join(json.loads(p.read_text())["args"] if p.is_file() else ARGS))
        return 0
    if args.audit:
        return audit(args.runs)
    if args.refresh:
        return refresh(args.runs, Path(args.baseline))
    if len(args.runs) != 1:
        ap.error("give one scoreboard report to check")
    return check(Path(args.runs[0]), Path(args.baseline))


if __name__ == "__main__":
    sys.exit(main())
