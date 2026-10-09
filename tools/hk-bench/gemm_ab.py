"""The third of the Phase 3 gate that needs a GPU: TFLOPs, DSL against C++.

The other two thirds -- ScratchSize 0, no spill, the same occupancy -- fall out
of the build and are checked on a machine with no GPU at all. This one does
not, and it is the one with a measurement discipline attached.

Same process, interleaved bursts. On wx-ms-w7900d-0043 a cross-process A/B once
manufactured a 13-16% difference out of nothing, and the two kernels here are
close enough that such a difference would be the entire result. `_timing.py`
explains why the burst is 8 and not 1.

The baseline is `tk_kernel.dispatch_micro`, which on these shapes selects
`big_config` -- the same 128x128x64, 8-warp, N_SPLIT=4 tiling the DSL kernel
transcribes. On a shape where dispatch would pick something else the comparison
would be against a different kernel, so the shapes are checked, not assumed.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/gemm_ab.py
    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/gemm_ab.py --shapes 4096x4096x4096
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "kernels", "rdna3", "gemm", "bf16fp32"))

from _timing import interleave  # noqa: E402

import hk  # noqa: E402
from hk.ops import gemm as hk_gemm  # noqa: E402

#: The two shapes the tiling was picked on. Small shapes are bound by grid
#: quantization rather than by the inner loop and want their own sweep.
SHAPES = [(4096, 4096, 4096), (8192, 8192, 4096)]


def _dispatch_picks_big(m, n, k):
    """Mirror of dispatch_any: does the C++ select big_config for this shape?

    Transcribed rather than queried because the module exports one entry point.
    If this drifts from the C++ the comparison silently becomes DSL-against-a-
    different-kernel, so the shapes it rejects are skipped rather than run.
    """
    if k % 64:
        return False, "K does not tile 64; dispatch would pick k32_config"
    blocks = ((m + 127) // 128) * (n // 128)
    if blocks <= 64 and n % 64 == 0:
        return False, f"{blocks} blocks; dispatch would pick small_config"
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default=None,
                    help="comma-separated MxNxK (default: the two tuning shapes)")
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args(argv)

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]

    try:
        import tk_kernel
    except ImportError:
        print("tk_kernel not built: make -C kernels/rdna3/gemm/bf16fp32", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"{'shape':<22}{'variant':<14}{'ms':>9}{'TFLOPs':>10}{'vs C++':>9}")
    print("-" * 64)

    rc = 0
    for m, n, k in shapes:
        ok, why = _dispatch_picks_big(m, n, k)
        if not ok:
            print(f"{f'{m}x{n}x{k}':<22}skipped: {why}")
            continue

        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c_hand = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        c_dsl = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        # Agreement before timing. Two kernels that disagree are not two
        # measurements of the same thing, however close the milliseconds are.
        tk_kernel.dispatch_micro(a, b, c_hand)
        hk_gemm.matmul(a, b, out=c_dsl)
        torch.cuda.synchronize()
        bad = (c_dsl.float() - c_hand.float()).abs().max().item()
        if bad > 0:
            # Not a tolerance: the same schedule over the same inputs in the
            # same order should be bit-identical. A nonzero max here means the
            # transcription differs somewhere, and that is worth knowing even
            # when it is within bf16 noise.
            print(f"  note: max |DSL - C++| = {bad:g}, not bit-identical")

        flops = 2 * m * n * k
        best = interleave(
            {
                "hk (DSL)": lambda: hk_gemm.matmul(a, b, out=c_dsl),
                "C++": lambda: tk_kernel.dispatch_micro(a, b, c_hand),
                "rocBLAS": lambda: torch.matmul(a, b.t(), out=c_hand),
            },
            rounds=args.rounds,
        )
        ref = best["C++"]
        for name, ms in best.items():
            print(f"{f'{m}x{n}x{k}':<22}{name:<14}{ms:>9.3f}"
                  f"{flops / (ms * 1e9):>10.1f}{ref / ms:>8.2f}x")
        print()
        # The gate: within noise of the handwritten kernel. 3% is the
        # run-to-run spread this harness shows on a repeated identical build.
        if best["hk (DSL)"] > ref * 1.03:
            print(f"  GATE: DSL is {best['hk (DSL)'] / ref:.2f}x the C++ time, "
                  f"outside the 3% noise band")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
