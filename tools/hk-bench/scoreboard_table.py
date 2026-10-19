#!/usr/bin/env python3
"""One line per case of a scoreboard --json report: hk's time, every other
arm's, the fastest other and hk's ratio to it, with the noise band the
verdict used. Torch-free.

    python3 tools/hk-bench/scoreboard_table.py report.json [--md] [--level tuned]
"""

from __future__ import annotations

import argparse
import json


def rows(report, level="tuned"):
    out = []
    for c in report["cases"]:
        arms = c["arms"]
        t = {a: v["ms"] for a, v in arms.items() if v.get("ms")}
        v = c.get("verdicts", {}).get(level)
        hk = min((t[a], a) for a in ("hk", "hk_tuned") if a in t) if any(
            a in t for a in ("hk", "hk_tuned")) else None
        others = [(ms, a) for a, ms in t.items() if not a.startswith("hk")]
        best = min(others) if others else None
        out.append({"case": c["case"], "t": t, "hk": hk, "best": best, "verdict": v,
                    "hk_note": (arms.get("hk") or {}).get("note")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report")
    ap.add_argument("--level", default="tuned", choices=("oob", "tuned"))
    ap.add_argument("--md", action="store_true", help="a markdown table, us")
    ap.add_argument("--brief", action="store_true",
                    help="markdown: hk, the fastest other arm and the verdict only")
    args = ap.parse_args()
    rep = json.load(open(args.report))
    if args.brief:
        print("| case | hk, us | fastest other, us | hk vs it |")
        print("|---|---|---|---|")
        for r in rows(rep, args.level):
            v = r["verdict"]
            hk = f"{r['hk'][0] * 1e3:.2f}" if r["hk"] else "--"
            best = f"{r['best'][0] * 1e3:.2f} {r['best'][1]}" if r["best"] else "--"
            verdict = f"{v['ratio']:.2f}x {v['verdict']}" if v else "--"
            print(f"| {r['case']} | {hk} | {best} | {verdict} |")
        return 0
    cols = ("hk", "torch", "inductor", "triton", "triton_tuned", "flydsl", "flydsl_tuned")
    if args.md:
        print("| case | " + " | ".join(cols) + " | best other | hk / best |")
        print("|---" * (len(cols) + 3) + "|")
    for r in rows(rep, args.level):
        cells = [f"{r['t'][a] * 1e3:.1f}" if a in r["t"] else "-" for a in cols]
        v = r["verdict"]
        if r["hk"] is None:
            ratio = f"hk: {str(r['hk_note'])[:60]}"
        elif v:
            ratio = f"{v['ratio']:.2f}x {v['verdict']} (band {v['band']:.3f})"
        else:
            ratio = "-"
        best = f"{r['best'][1]}" if r["best"] else "-"
        if args.md:
            print(f"| {r['case']} | " + " | ".join(cells) + f" | {best} | {ratio} |")
        else:
            print(f"{r['case']:<26}" + "".join(f"{c:>9}" for c in cells) + f"  {best:<13} {ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
