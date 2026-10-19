#!/usr/bin/env python3
"""The pointwise strip plans against the other arms, in one process: add
and exp at warps x chunks a warp, beside tuned Triton, Inductor and torch,
each plan's ratio to the fastest of those. The cache-resident sizes move by
up to 20% between GPUs and runs, so a plan is chosen by its worst ratio over
several runs, not by one sweep's winner.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/strip_pointwise_ab.py \\
        [--ops add,exp] [--rows 2048,8192] [--cols 4096,5120,8192] \\
        [--plans 8x8,4x8,4x2,2x2,1x2,8x1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

import triton_ref                                  # noqa: E402
from _timing import burn_in                        # noqa: E402
from hk.ops import elementwise, norm               # noqa: E402
from scoreboard import gpu_times                   # noqa: E402

BF16 = torch.bfloat16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ops", default="add,exp")
    ap.add_argument("--rows", default="2048,8192")
    ap.add_argument("--cols", default="4096,5120,8192")
    ap.add_argument("--plans", default="8x8,4x8,4x2,2x2,1x2,8x1")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    plans = [tuple(int(v) for v in p.split("x")) for p in args.plans.split(",")]
    burn_in()
    out = []
    for op in args.ops.split(","):
        for n in (int(c) for c in args.cols.split(",")):
            for m in (int(r) for r in args.rows.split(",")):
                a = torch.randn(m, n, device="cuda", dtype=BF16)
                b = torch.randn_like(a)
                nn = m * n
                comp = torch.compile((lambda x, y: x + y) if op == "add"
                                     else (lambda x, y: torch.exp(x)), dynamic=False)
                bufs = {}
                fns = {"triton_tuned": lambda: triton_ref.elementwise_tuned(op, a, b, bufs["t"]),
                       "inductor": lambda: comp(a, b),
                       "torch": (lambda: torch.add(a, b, out=bufs["torch"])) if op == "add"
                       else (lambda: torch.exp(a, out=bufs["torch"]))}
                bufs.update(t=torch.empty_like(a), torch=torch.empty_like(a))
                run = norm.strip_run(nn, 2)
                views = (a.view(1, nn), b.view(1, nn)) if op == "add" else (a.view(1, nn),)
                for w, c in plans:
                    o = torch.empty_like(a).view(1, nn)
                    k = elementwise.STRIP_KERNELS[f"strip_{op}_bf16_w{w}"]
                    cw = c * norm._wave() * run
                    fns[f"{w}x{c}"] = (lambda k=k, o=o, cw=cw, w=w:
                                       k(*views, o, CW=cw, RUN=run, EXACT=int(nn % (w * cw) == 0)))
                for f in fns.values():
                    f()
                torch.cuda.synchronize()
                t = {k: v[0] * 1e3 for k, v in gpu_times(fns, 7, 50).items()}
                best = min(t[k] for k in ("triton_tuned", "inductor", "torch"))
                row = {"op": op, "m": m, "n": n, "best_other": best,
                       "ratio": {f"{w}x{c}": best / t[f"{w}x{c}"] for w, c in plans}, "us": t}
                out.append(row)
                print(f"{op} {m}x{n} best other {best:6.2f}  " + "  ".join(
                    f"{p} {r:.3f}" for p, r in row["ratio"].items()), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
