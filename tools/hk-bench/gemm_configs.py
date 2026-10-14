#!/usr/bin/env python3
"""Which tiling each contender's GEMM picks at the scoreboard's shapes, and
what it reaches: Triton's autotuned config, hk's tuned schedule, and the
time of each, in one process.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/gemm_configs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import hk  # noqa: E402
import scoreboard  # noqa: E402
import triton_ref  # noqa: E402
from hk.ops import gemm  # noqa: E402

SHAPES = [(4096, 4096, 4096), (2048, 6144, 4096), (2048, 4096, 4096),
          (2048, 24576, 4096), (2048, 4096, 12288)]


def main():
    print(f"{'shape':<20}{'hk ms':>9}{'TF':>7}{'triton ms':>11}{'TF':>7}"
          f"   triton config / hk schedule")
    for m, n, k in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        oh = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ot = torch.empty_like(oh)
        hk.ops.matmul(a, b, out=oh)
        triton_ref.matmul_abt(a, b, out=ot)
        torch.cuda.synchronize()
        t = scoreboard.gpu_times({
            "hk": lambda: hk.ops.matmul(a, b, out=oh),
            "triton": lambda: triton_ref.matmul_abt(a, b, out=ot),
        }, rounds=5, calls=10)
        fl = 2 * m * n * k
        cfg = triton_ref._matmul_abt.best_config
        print(f"{f'{m}x{n}x{k}':<20}{t['hk']:>9.3f}{fl / t['hk'] / 1e9:>7.1f}"
              f"{t['triton']:>11.3f}{fl / t['triton'] / 1e9:>7.1f}   "
              f"{cfg.kwargs} w{cfg.num_warps} s{cfg.num_stages} / "
              f"{gemm.schedule_for(m, n, k)}")


if __name__ == "__main__":
    main()
