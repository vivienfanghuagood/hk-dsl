#!/usr/bin/env python3
"""Host cost of a generic kernel's launch (hk.T read from the tensors, `warps` per call) against
the same specialization called directly: the bound the dtype-parameter proposal set is <= 0.3 us
a launch (docs/language.md, "One kernel for every dtype and warp count").

Same process, the arms interleaved per round; each round times `--n` launches with no
synchronize in between (the host cost of a launch that never blocks), a synchronize after.
The kernel is fused.silu_mul_kernel on a 16x64 tensor: as small as a launch gets.

    python3 tools/hk-bench/generic_launch.py [--rounds 30] [--n 2000]
"""
import argparse
import statistics
import time

import torch

import hk
from hk.ops import fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    a = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    o = torch.empty_like(a)
    g = fused.silu_mul_kernel
    v = g.variant(T=hk.bf16, warps=1)
    arms = {
        "specialization  v(a, b, o)": lambda: v(a, b, o),
        "specialization, a kwarg  v(a, b, o, TILES=1)": lambda: v(a, b, o, TILES=1),
        "generic, inferred  g(a, b, o)": lambda: g(a, b, o),
        "generic, T given  g(a, b, o, T=bf16)": lambda: g(a, b, o, T=hk.bf16),
        "generic, warps  g(a, b, o, warps=1)": lambda: g(a, b, o, warps=1),
    }
    for f in arms.values():          # build, load, fill every cache
        for _ in range(50):
            f()
    torch.cuda.synchronize()
    per = {k: [] for k in arms}
    for _ in range(args.rounds):
        for k, f in arms.items():
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.n):
                f()
            per[k].append((time.perf_counter() - t) / args.n * 1e6)
            torch.cuda.synchronize()
    base = statistics.median(per["specialization  v(a, b, o)"])
    print(f"{torch.cuda.get_device_name(0)} ({hk.target.current_arch()}), hk {hk.__version__}, "
          f"{args.rounds} rounds x {args.n} launches, host us a launch (median, range):")
    for k, xs in per.items():
        m = statistics.median(xs)
        print(f"  {k:46s} {m:6.2f} us ({min(xs):.2f}-{max(xs):.2f})  {m - base:+.2f} us vs the specialization")
    print("  (a call with keywords pays for them in both forms: compare the generic kwarg rows with "
          "the specialization's kwarg row)")


if __name__ == "__main__":
    main()
