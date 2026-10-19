#!/usr/bin/env python3
"""Per op of a scoreboard --json report, the range of hk's ratio to the
fastest other arm (the tuned verdict's), split by row count, and its
verdicts counted; one or two reports side by side. Torch-free.

    python3 tools/hk-bench/scoreboard_summary.py before.json [after.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def summary(path):
    rep = json.load(open(path))
    out = defaultdict(lambda: defaultdict(list))
    verdicts = defaultdict(lambda: defaultdict(int))
    for c in rep["cases"]:
        op, shape = c["case"].split()
        rows = int(shape.split("x")[0])
        band = "1-32" if rows <= 32 else str(rows)
        v = c.get("verdicts", {}).get("tuned")
        if v is None:
            out[op][band].append(None)
            verdicts[op]["no hk"] += 1
            continue
        out[op][band].append(v["ratio"])
        verdicts[op][v["verdict"]] += 1
    return out, verdicts


def fmt(rs):
    have = [r for r in rs if r is not None]
    miss = len(rs) - len(have)
    if not have:
        return "--"
    lo, hi = min(have), max(have)
    s = f"{lo:.2f}" if abs(hi - lo) < 0.005 else f"{lo:.2f}-{hi:.2f}"
    return s + (f" ({miss} n/a)" if miss else "")


def main() -> int:
    reports = [summary(p) for p in sys.argv[1:]]
    ops = list(reports[0][0])
    for op in ops:
        cells = []
        for out, ver in reports:
            cells.append(" | ".join(fmt(out[op][b]) for b in ("1-32", "2048", "8192")))
            cells.append(", ".join(f"{n} {k}" for k, n in sorted(ver[op].items())))
        print(f"| {op} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
