"""What the torch custom op costs, against the pybind path and against torch.

Phase 6 moves the serving path from a pybind module onto `torch.ops.hk.*`. The
registration buys three things a framework needs -- torch's stream, CUDA-graph
capture, `torch.compile` -- and the question this script answers is what it
charges for them. Two numbers, because they fail differently:

  * **device ms**, which should be identical: the kernel body is the same text
    (tests/hk/ir/test_torch_scaffold.py asserts that), so a difference here
    would mean the launch configuration changed, not the kernel.
  * **host microseconds**, which is where a custom op can lose. The pybind
    converter reads a tensor in five Python calls; torch's dispatcher does
    schema parsing, boxing and a dispatch-key lookup instead. At 7.9 ms of
    attention neither matters; the number is here so that nobody has to guess
    when the same path is reused for a 60 us elementwise op.

Same process, interleaved bursts (tools/hk-bench/_timing.py), aotriton in the
table as the thing actually being replaced.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/sdpa_paths.py
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import hk                                    # noqa: E402
from hk.ops import sdpa as hk_sdpa           # noqa: E402
from _timing import interleave               # noqa: E402

SHAPES = [
    # (B, H, N, D, causal) -- MiniMax H3's inference shapes plus one causal.
    (1, 56, 4096, 128, False),
    (1, 56, 8192, 128, False),
    (1, 56, 16384, 128, False),
    (1, 32, 8192, 128, True),
]


def host_us(fn, iters=200, warmup=20):
    """Wall time per call with the GPU kept behind, so what is measured is the
    CPU side of the launch and not the kernel."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def tflops(b, h, n, d, causal, ms):
    flops = 4.0 * b * h * n * n * d
    if causal:
        flops /= 2.0
    return flops / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    print(f"{'shape':>22} {'path':<12} {'ms':>9} {'TF':>7} {'host us':>9} "
          f"{'vs pybind':>10}")
    for b, h, n, d, causal in SHAPES:
        q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = torch.empty_like(q)

        op = hk_sdpa.op_for(d, causal, None, n)
        kern = hk.ops.attn._kernel_for(d, causal, None, n)

        # Bit-identity first: a timing table comparing two different answers is
        # worse than no table.
        op(q, k, v, out)
        ref = torch.empty_like(q)
        kern(q, k, v, ref)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), "torch op and pybind disagree"

        variants = {
            "pybind": lambda: kern(q, k, v, out),
            "torch.ops": lambda: op(q, k, v, out),
            "sdpa drop-in": lambda: hk_sdpa.scaled_dot_product_attention(
                q, k, v, is_causal=causal),
            "aotriton": lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=causal),
        }
        best = interleave(variants, rounds=args.rounds)
        hosts = {name: host_us(fn) for name, fn in variants.items()}
        base = best["pybind"]
        tag = f"{b}x{h}x{n}x{d}{'c' if causal else ''}"
        for name in variants:
            print(f"{tag:>22} {name:<12} {best[name]:9.3f} "
                  f"{tflops(b, h, n, d, causal, best[name]):7.1f} "
                  f"{hosts[name]:9.1f} {base / best[name]:9.2f}x")
        print()


if __name__ == "__main__":
    main()
