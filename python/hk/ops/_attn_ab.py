"""Same-process A/B: the DSL attention kernel against the handwritten C++.

The Phase 4 gate is a number -- 64-65 TFLOPs -- measured on the handwritten
kernel in `kernels/rdna3/attn/fwd/attn.cpp` on a different day. On W7900D that
is not a number you can compare across processes: the chip's clock and power
state differ enough between launches to manufacture a 13-16% difference out of
nothing, and a gate crossed or missed by 1% is exactly the size of that
artifact. So both kernels are loaded into one process and timed in
alternating order, and what the table reports is the *ratio*, which is the
only part that survives the clock drifting underneath it.

    HIP_VISIBLE_DEVICES=0 python3 -m hk.ops._attn_ab
"""

from __future__ import annotations

import os
import sys

import torch

import hk
from hk.ops._attn_bench import D_HEAD, N_HEADS, bench, flops, qkv

EXT = os.environ.get(
    "HK_ATTN_EXT",
    "/workspace/HipKittens/kernels/rdna3/attn/torch_ext/hk_attn_ext.so")

SHAPES = [
    ("N=4096",   1, N_HEADS,  4096, False),
    ("N=8192",   1, N_HEADS,  8192, False),
    ("N=16384",  1, N_HEADS, 16384, False),
    ("N=32768",  1, N_HEADS, 32768, False),
    ("480p/5s",  1, N_HEADS, 49920, False),
    ("causal 8192", 1, N_HEADS, 8192, True),
]


def main() -> int:
    torch.ops.load_library(EXT)
    cpp = torch.ops.hk_attn.fwd

    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.gcnArchName}, {p.multi_processor_count} WGPs\n")
    print(f"{'shape':<13}{'dsl ms':>9}{'dsl TF':>9}{'c++ ms':>9}"
          f"{'c++ TF':>9}{'dsl/c++':>9}")

    tot_d = tot_c = 0.0
    for label, b, h, n, causal in SHAPES:
        q, k, v = qkv(b, h, n)
        fl = flops(b, h, n, causal)
        d = lambda: hk.attention(q, k, v, causal=causal)   # noqa: E731
        c = lambda: cpp(q, k, v, D_HEAD ** -0.5, causal)   # noqa: E731
        # Interleaved and best-of-two each way. A fixed order hands whatever
        # the clock does over the span of a row to whichever ran second.
        d_ms, c_ms = bench(d), bench(c)
        d_ms, c_ms = min(d_ms, bench(d)), min(c_ms, bench(c))
        d_tf, c_tf = fl / d_ms * 1e-9, fl / c_ms * 1e-9
        tot_d, tot_c = tot_d + d_tf, tot_c + c_tf
        print(f"{label:<13}{d_ms:>9.3f}{d_tf:>9.1f}{c_ms:>9.3f}{c_tf:>9.1f}"
              f"{d_tf / c_tf:>8.3f}x")
        del q, k, v
        torch.cuda.empty_cache()

    print(f"\nmean TF: dsl {tot_d / len(SHAPES):.1f}, "
          f"c++ {tot_c / len(SHAPES):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
