#!/usr/bin/env python3
"""norm.strip_plan's choices, measured: each strip kernel at one, two and
four 16-byte runs a lane (warps to match) and at one or more rows a strip,
beside the tile kernels, across widths and row counts, one process, the
scoreboard's timing.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/strip_plan.py \\
        [--kinds rmsw,addrmsw,softmax] [--cols 1024,...] [--rows 1,...] \\
        [--targets 8,16,32] [--rows-per 1,2,4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

import hk                                          # noqa: E402
from _timing import burn_in                        # noqa: E402
from hk.ops import norm                            # noqa: E402
from scoreboard import gpu_times                   # noqa: E402

BF16 = torch.bfloat16


def plan_for(cols: int, per_lane: int, nbytes: int = 2):
    """(warps, chunks, run, exact) at `per_lane` elements a lane."""
    run = norm.strip_run(cols, nbytes)
    chunks = norm._ceil(cols, 32 * run)
    warps = min(norm.MAX_WARPS, norm._ceil(chunks, max(1, per_lane // run)))
    per = norm._ceil(chunks, warps)
    return warps, per, run, int(warps * per * 32 * run == cols)


def arms(kind: str, m: int, n: int, targets, rows_per=(1,)):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = (torch.rand(n, device="cuda") + 0.5).to(BF16)
    o = torch.empty_like(x)
    xa, ra = x.clone(), torch.randn_like(x)
    out = {}

    def tile():
        norm.LAYOUT = "tile"
        try:
            if kind == "rmsw":
                hk.ops.rms_norm(x, w, out=o)
            elif kind == "addrmsw":
                hk.ops.fused_add_rms_norm(xa, ra, w)
            else:
                hk.ops.softmax(x, out=o)
        finally:
            norm.LAYOUT = "strip"

    # The unweighted tile kernels need a whole 16-row tile.
    if m >= norm.ROWS or kind != "softmax":
        out["tile"] = tile
    for t in targets:
        for rp in rows_per:
            warps, per, run, exact = plan_for(n, t)
            if per * run * rp > norm.STRIP_MAX_PER_LANE or rp > m:
                continue
            k = norm.STRIP_KERNELS[f"strip_{kind}_bf16_w{warps}"]
            ob, xb, rb = torch.empty_like(x), x.clone(), torch.randn_like(x)
            c = dict(ROWS=rp, CHUNKS=per, RUN=run, EXACT=int(exact and m % rp == 0))
            if kind == "rmsw":
                fn = (lambda k=k, ob=ob, c=c: k(x, w.view(1, n), ob, **c))
            elif kind == "addrmsw":
                fn = (lambda k=k, xb=xb, rb=rb, c=c: k(xb, rb, w.view(1, n), xb, rb, **c))
            else:
                fn = (lambda k=k, ob=ob, c=c: k(x, ob, **c))
            out[f"{t:2d}/lane w{warps}" + (f" R{rp}" if rp > 1 else "")] = fn
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kinds", default="rmsw,addrmsw,softmax")
    ap.add_argument("--cols", default="1024,2048,4096,5120,8192")
    ap.add_argument("--rows", default="1,32,2048,8192")
    ap.add_argument("--targets", default="8,16,32")
    ap.add_argument("--rows-per", default="1", help="rows a strip holds, a list")
    args = ap.parse_args()
    targets = [int(t) for t in args.targets.split(",")]
    rows_per = [int(r) for r in args.rows_per.split(",")]
    burn_in()
    for kind in args.kinds.split(","):
        for n in (int(c) for c in args.cols.split(",")):
            for m in (int(r) for r in args.rows.split(",")):
                fns = arms(kind, m, n, targets, rows_per)
                for f in fns.values():
                    f()
                torch.cuda.synchronize()
                small = m * n < 2048 * 4096
                t = gpu_times(fns, 9 if small else 5, 50 if small else 10)
                pick = norm.strip_plan(n, kind, 2)
                row = "  ".join(f"{a} {ms * 1e3:8.2f}" for a, (ms, _) in
                                sorted(t.items(), key=lambda kv: kv[1][0]))
                print(f"{kind:8s} {m:5d}x{n:<5d} plan w{pick[0]} c{pick[1]}  | {row}",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
