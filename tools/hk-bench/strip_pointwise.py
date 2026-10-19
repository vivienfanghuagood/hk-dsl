#!/usr/bin/env python3
"""The pointwise strip kernels' plan, measured: elementwise add and exp and
silu_mul at warps x chunks a warp (a chunk is one 16-byte run a lane), one
process, the scoreboard's timing, torch beside them.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/strip_pointwise.py \\
        [--ops add,exp,silu_mul] [--rows 2048,8192] [--cols 4096,8192] \\
        [--inter 14336,28672] [--warps 1,2,4,8] [--chunks 1,2,4,8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402
import torch.nn.functional as F                    # noqa: E402

import hk                                          # noqa: E402,F401
from _timing import burn_in                        # noqa: E402
from hk.ops import elementwise, fused, norm        # noqa: E402
from scoreboard import gpu_times                   # noqa: E402

BF16 = torch.bfloat16


def arms(op, m, n, warps, chunks):
    wave = norm._wave()
    if op == "silu_mul":
        x = torch.randn(m, 2 * n, device="cuda", dtype=BF16)
        a, b = x[:, :n], x[:, n:]
        o = torch.empty(m, n, device="cuda", dtype=BF16)
        out = {"torch": lambda: F.silu(a) * b}
        for run in RUNS:
            for w in warps:
                for c in chunks:
                    cw = c * wave * run
                    k = fused.STRIP_KERNELS[f"strip_silu_mul_bf16_w{w}"]
                    out[f"r{run} w{w} c{c}"] = (
                        lambda k=k, cw=cw, run=run, w=w: k(a, b, o, CW=cw, RUN=run,
                                                           EXACT=int(n % (w * cw) == 0)))
        return out
    a = torch.randn(m * n, device="cuda", dtype=BF16)
    b = torch.randn_like(a)
    o = torch.empty_like(a)
    out = {"torch": (lambda: torch.add(a, b, out=o)) if op == "add"
           else (lambda: torch.exp(a, out=o))}
    nn = a.numel()
    views = (a.view(1, nn), b.view(1, nn)) if op == "add" else (a.view(1, nn),)
    for run in RUNS:
        for w in warps:
            for c in chunks:
                cw = c * wave * run
                k = elementwise.STRIP_KERNELS[f"strip_{op}_bf16_w{w}"]
                ex = int(nn % (w * cw) == 0)
                out[f"r{run} w{w} c{c}"] = (lambda k=k, cw=cw, ex=ex, run=run:
                                            k(*views, o.view(1, nn), CW=cw, RUN=run, EXACT=ex))
    return out


#: Elements in a lane's run: 8 is a 16-byte access for bf16, 4 an 8-byte one.
RUNS = (8,)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ops", default="add,exp,silu_mul")
    ap.add_argument("--rows", default="2048,8192")
    ap.add_argument("--cols", default="4096,8192")
    ap.add_argument("--inter", default="14336,28672")
    ap.add_argument("--warps", default="1,2,4,8")
    ap.add_argument("--chunks", default="1,2,4,8")
    ap.add_argument("--runs", default="8", help="elements a lane's run, a list")
    args = ap.parse_args()
    warps = [int(v) for v in args.warps.split(",")]
    chunks = [int(v) for v in args.chunks.split(",")]
    global RUNS
    RUNS = tuple(int(v) for v in args.runs.split(","))
    burn_in()
    for op in args.ops.split(","):
        widths = args.inter if op == "silu_mul" else args.cols
        for n in (int(c) for c in widths.split(",")):
            for m in (int(r) for r in args.rows.split(",")):
                fns = arms(op, m, n, warps, chunks)
                for f in fns.values():
                    f()
                torch.cuda.synchronize()
                t = gpu_times(fns, 7, 20)
                row = "  ".join(f"{a} {ms * 1e3:7.2f}" for a, (ms, _) in
                                sorted(t.items(), key=lambda kv: kv[1][0]))
                print(f"{op:8s} {m:5d}x{n:<6d} | {row}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
