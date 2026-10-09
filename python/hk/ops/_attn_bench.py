"""Phase 4's performance gate: TFLOPs for the DSL attention kernel.

Run on the pod, one GPU, nothing else on it:

    HIP_VISIBLE_DEVICES=0 python3 -m hk.ops._attn_bench

The gate is 64-65 TFLOPs, which is what the handwritten C++ in
`kernels/rdna3/attn/fwd/attn.cpp` measures on the same chip. The comparison
that matters is same-process: W7900D's peer and clock behaviour differs enough
between process launches to manufacture a 13-16% difference out of nothing, so
every number in one table comes from one process, and the orders are
interleaved rather than run back to back.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import hk

D_HEAD = 128
N_HEADS = 56


def flops(b, h, n, causal=False, d=D_HEAD):
    """Two b*h*n*n*d matmuls, halved for causal -- the FA convention."""
    f = 2 * 2 * b * h * n * n * d
    return f / 2 if causal else f


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / iters


def qkv(b, h, n, d=D_HEAD):
    g = torch.Generator(device="cuda").manual_seed(0)
    return [torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16,
                        generator=g) for _ in range(3)]


SHAPES = [
    ("N=4096",      1, N_HEADS,  4096, False),
    ("N=8192",      1, N_HEADS,  8192, False),
    ("N=16384",     1, N_HEADS, 16384, False),
    ("N=32768",     1, N_HEADS, 32768, False),
    ("480p/5s",     1, N_HEADS, 49920, False),   # H3: 832x480, 125 frames
    ("N=16384 cfg", 2, N_HEADS, 16384, False),
    ("causal 8192", 1, N_HEADS,  8192, True),
]


def main() -> int:
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.gcnArchName}, {p.multi_processor_count} WGPs, "
          f"torch {torch.__version__}\n")
    print(f"{'shape':<14}{'hk ms':>9}{'hk TF':>8}{'sdpa ms':>10}"
          f"{'sdpa TF':>9}{'ratio':>8}")

    best = 0.0
    for label, b, h, n, causal in SHAPES:
        q, k, v = qkv(b, h, n)
        fl = flops(b, h, n, causal)
        # Interleaved, not back to back: the clock state drifts over a table
        # and a fixed order hands the drift to whichever ran second.
        hk_ms = bench(lambda: hk.attention(q, k, v, causal=causal))
        sd_ms = bench(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal))
        hk_ms = min(hk_ms, bench(lambda: hk.attention(q, k, v, causal=causal)))
        sd_ms = min(sd_ms, bench(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal)))
        hk_tf, sd_tf = fl / hk_ms * 1e-9, fl / sd_ms * 1e-9
        best = max(best, hk_tf)
        print(f"{label:<14}{hk_ms:>9.3f}{hk_tf:>8.1f}{sd_ms:>10.3f}"
              f"{sd_tf:>9.1f}{hk_tf / sd_tf:>7.2f}x")
        del q, k, v
        torch.cuda.empty_cache()

    gate = float(os.environ.get("HK_ATTN_GATE", "64"))
    print(f"\nbest {best:.1f} TF, gate {gate:.0f} TF: "
          f"{'PASS' if best >= gate else 'FAIL'}")
    return 0 if best >= gate else 1


if __name__ == "__main__":
    sys.exit(main())
