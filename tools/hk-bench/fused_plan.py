#!/usr/bin/env python3
"""How silu_mul should view its tensor, measured instead of argued.

This started as a (WARPS, TILES) sweep -- `fused.plan` encoded a tradeoff
between memory-level parallelism and live registers -- and the answer was that
the knob does not matter: fifteen settings within 5%, and one warp holding one
tile won every case. The paragraph is kept in fused.plan's docstring.

What does matter turned up in those same numbers. The identical kernel ran at
716 GB/s over an 8192x4096 tensor and 675 GB/s over an 8192x11008 one, with the
same number of workgroups doing the same number of 16x64 tiles. The only
difference between the two is the row stride the tiles walk: 8192 bytes against
22016. But silu_mul is *elementwise*, so the row structure is not part of the
problem -- a contiguous tensor may be viewed at any width that divides its
element count and the answer is the same bytes in the same places. So the width
is a free parameter, and this sweeps it.

    python3 tools/hk-bench/fused_plan.py            # the Phase 2 bench shapes
    python3 tools/hk-bench/fused_plan.py 8192 11008
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

import _timing  # noqa: E402

from hk.ops import fused  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
HBM_GB_S = 864.0

CASES = [(2048, 11008), (8192, 11008), (8192, 4096)]

#: Widths to try, besides whatever the caller's own shape is. Powers of two
#: only: the point of the exercise is a row stride the memory system likes.
WIDTHS = [1 << i for i in range(6, 15)]  # 64 .. 16384

#: (warps, tiles) settings re-timed at every width, to check that the previous
#: sweep's answer does not change once the stride does.
SHAPES = [(1, 1), (4, 1), (16, 1)]


def views(rows: int, cols: int):
    """Every (rows, cols) this tensor may legally be viewed as, widest first.

    Includes the tensor's own shape, which is not necessarily a power of two
    and is the thing the alternatives have to beat.
    """
    n = rows * cols
    out = []
    for w in sorted(set(WIDTHS + [cols]), reverse=True):
        if w >= fused.COLS and n % w == 0 and n // w >= fused.ROWS:
            out.append((n // w, w))
    return out


_interleave = _timing.interleave
def run(rows: int, cols: int):
    a = torch.randn(rows, cols, device=DEV, dtype=DT)
    b = torch.randn(rows, cols, device=DEV, dtype=DT)
    out = torch.empty_like(a)
    nbytes = 3 * a.numel() * 2

    variants = {}
    for vr, vc in views(rows, cols):
        av, bv, ov = a.view(vr, vc), b.view(vr, vc), out.view(vr, vc)
        nblk = -(-vc // fused.COLS)
        for warps, tiles in SHAPES:
            if warps * tiles > nblk:
                continue
            k = fused.KERNELS[f"silu_mul_bf16_w{warps}"]
            k.build(TILES=tiles)
            own = " *" if vc == cols else "  "
            label = f"{vr:>7d}x{vc:<6d}{own} w{warps:<2d} t{tiles}"
            variants[label] = (
                lambda k=k, t=tiles, x=av, y=bv, o=ov: k(x, y, o, TILES=t))

    chosen = fused.plan(cols)
    cr, cc = fused.canonical(rows, cols)
    print(f"\nsilu_mul {rows}x{cols} bf16   picks {cr}x{cc} "
          f"w{chosen[0]} t{chosen[1]}   (* = the caller's own shape)")
    print(f"  {'variant':28s} {'ms':>8s} {'GB/s':>8s} {'%HBM':>6s}")
    best = _interleave(variants)
    want = f"{cr:>7d}x{cc:<6d}"
    for label, ms in sorted(best.items(), key=lambda kv: kv[1]):
        gbs = nbytes / (ms * 1e-3) / 1e9
        mark = "  <- picked" if label.startswith(want) and \
            label.endswith(f"w{chosen[0]:<2d} t{chosen[1]}") else ""
        print(f"  {label:28s} {ms:8.3f} {gbs:8.1f} "
              f"{100 * gbs / HBM_GB_S:5.1f}%{mark}")

    # The reference this has to beat, timed in the same process for the same
    # reason everything else is.
    ref = lambda: torch.nn.functional.silu(a) * b  # noqa: E731
    try:
        comp = torch.compile(ref, dynamic=False)
        both = _interleave({"torch.compile": comp, "eager": ref})
        for name, ms in both.items():
            print(f"  {name:28s} {ms:8.3f} {nbytes / (ms * 1e-3) / 1e9:8.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch.compile unavailable: {exc}")
    return min(best, key=best.get)


def main(argv):
    cases = CASES
    if len(argv) == 3:
        cases = [(int(argv[1]), int(argv[2]))]
    elif len(argv) != 1:
        sys.exit(__doc__)
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    for rows, cols in cases:
        w = run(rows, cols)
        print(f"  winner {rows}x{cols}: {w}")


if __name__ == "__main__":
    main(sys.argv)
