#!/usr/bin/env python3
"""GEMM schedule variants against Triton's autotuned kernel at the
scoreboard's shapes, one process, kernel-attributed GPU time.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/gemm_variants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import scoreboard  # noqa: E402
import triton_ref  # noqa: E402
from hk.ops import gemm  # noqa: E402

SHAPES = [(4096, 4096, 4096), (2048, 6144, 4096), (2048, 4096, 4096),
          (2048, 24576, 4096), (2048, 4096, 12288)]

VARIANTS = {
    "lds2 k64": dict(lds_buffers=2, k_step=64, write_pos=2),
    "lds1 k64": dict(lds_buffers=1, k_step=64, write_pos=1),
    "lds2 k32": dict(lds_buffers=2, k_step=32, write_pos=2),
    "lds1 k32": dict(lds_buffers=1, k_step=32, write_pos=1),
}


def main():
    ks = {}
    for label, v in VARIANTS.items():
        s = {**gemm.DEFAULT, **v}
        ks[label] = gemm.gemm_kernel(
            "gv_" + label.replace(" ", "_"),
            **{k: s[k] for k in ("block_m", "block_n", "k_step", "dot_slice",
                                 "warps", "warp_rows", "n_split", "wgm",
                                 "write_pos", "lds_buffers")})
    print(f"{'shape':<20}" + "".join(f"{v:>11}" for v in [*VARIANTS, "triton"])
          + "   TFLOP/s")
    for m, n, k in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        ref = (a.float() @ b.float().T)
        outs = {lab: torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                for lab in [*VARIANTS, "triton"]}
        fns = {lab: (lambda kk=kk, o=outs[lab]: kk(a, b, o))
               for lab, kk in ks.items()}
        fns["triton"] = lambda: triton_ref.matmul_abt(a, b, out=outs["triton"])
        for lab, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            err = ((outs[lab].float() - ref).abs().max() / ref.abs().max()).item()
            if err > 2e-2:
                print(f"  {lab}: wrong, rel err {err:.3e}")
        t = scoreboard.gpu_times(fns, rounds=5, calls=10)
        fl = 2 * m * n * k
        print(f"{f'{m}x{n}x{k}':<20}" + "".join(f"{t[v]:>11.3f}" for v in fns)
              + "   " + " ".join(f"{fl / t[v] / 1e9:.1f}" for v in fns),
              flush=True)


if __name__ == "__main__":
    main()
