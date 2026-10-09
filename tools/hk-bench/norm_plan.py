#!/usr/bin/env python3
"""Which (WARPS, TPW, FOLD) a row width deserves, measured instead of argued.

`norm.plan(cols, kind)` answers that question with a rule: split the row
across as many warps as it has blocks (up to 16), hold it in registers if
MAX_TPW tiles per warp are enough, otherwise try the fold, otherwise stream --
except for softmax, which takes the fold first. Every clause of that rule is a
plausible-sounding guess about a tradeoff the hardware decides, and two of
them have already been wrong once:

  * holding the row saves a whole pass over memory, but four live fp32 tiles
    plus the reduction's temporaries is ~210 VGPRs, which is occupancy 7 where
    a streaming kernel gets 16;
  * folding holds the row in a quarter of the registers and multiplies the
    workgroup count by 16, but adds an LDS round trip per reduction and shrinks
    the workgroup to a quarter of the warps;
  * more warps per workgroup is more waves per SIMD, but also more LDS traffic
    and a wider barrier.

So this sweeps the candidates a given width admits and prints them sorted. It
is the `/kernel-ab-bench` discipline: one process, round-robin, and the numbers
within a run are comparable to each other and to nothing else.

    python3 tools/hk-bench/norm_plan.py                  # the shapes plan() decides
    python3 tools/hk-bench/norm_plan.py softmax 16384    # one op, one width

`kind` is passed to plan() here exactly as `norm._run` passes it, because the
three ops do not want the same answer at the same width -- which is the single
most expensive thing this sweep has found.

The kernels are invoked the way `norm._run` invokes them -- same view, same
constexprs -- so that what wins here is something plan() can actually select.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

import _timing  # noqa: E402

from hk.ops import norm  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
HBM_GB_S = 864.0

#: (kind, rows, cols). Grouped by where the working set sits relative to the
#: 96 MB Infinity Cache. plan() no longer branches on that -- the measurement
#: that deleted the branch is in its docstring -- but the grouping stays,
#: because "the held variant wins in both regimes" is a claim the sweep has to
#: keep being able to falsify, and because a case that fits in cache is not
#: measuring memory and should not be read as if it were.
#:
#: Every kind appears at 4096 columns: that is the width where rms/layer and
#: softmax disagree about FOLD, which is the whole reason plan() takes a kind.
#: 1024 and 2048 settle norm._FOLD_MIN_WARPS -- the widths where the folded
#: view exists but leaves a one- or two-warp workgroup.
CASES = [
    # Past the cache, by a lot. This is the regime a real model runs in.
    ("rms", 16384, 4096),      # 268 MB
    ("layer", 16384, 4096),    # 268 MB
    ("softmax", 8192, 4096),   # 134 MB
    ("softmax", 8192, 16384),  # 537 MB
    ("rms", 4096, 16384),      # 268 MB
    # The band MAX_TPW=5 opened: 80 blocks is five tiles a warp across 16, and
    # at four it would have fallen through to FOLD instead. 5120 is
    # Qwen2.5-14B's hidden size, which is why the band is worth a case.
    ("rms", 16384, 5120),      # 335 MB
    ("softmax", 8192, 5120),   # 168 MB
    # Narrow rows, where the folded view exists but leaves a one- or two-warp
    # workgroup. These settle norm._FOLD_MIN_WARPS.
    ("rms", 65536, 1024),      # 268 MB, folded -> 1 warp
    ("rms", 32768, 2048),      # 268 MB, folded -> 2 warps
    # Inside the cache. The deleted cache branch was fitted to the first two
    # of these, and refuted by all five.
    ("layer", 4096, 4096),     # 67 MB
    ("rms", 4096, 4096),       # 67 MB
    ("rms", 2048, 8192),       # 67 MB
    ("softmax", 2048, 4096),   # 34 MB
    ("rms", 16384, 1024),      # 67 MB, and narrow
    # Astride it. 96 MB is a capacity, not a cliff.
    ("rms", 6144, 4096),       # 101 MB
    ("rms", 8192, 4096),       # 134 MB
]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def candidates(cols: int):
    """Every (warps, tpw, fold) that is legal for this width.

    Legal means: at most one warp per column block, a persistent variant holds
    at most MAX_TPW tiles, and the folded view exists at all (the width is a
    multiple of 16 and the folded row is still a whole tile). Nothing here is
    ranked -- that is what the timing is for.
    """
    out = []
    for fold in (1, norm.ROWS):
        if fold != 1 and (cols % norm.ROWS or cols // norm.ROWS < norm.COLS):
            continue
        nblk = _ceil(cols // fold, norm.COLS)
        for warps in norm.WARP_COUNTS:
            if warps > min(norm.MAX_WARPS, nblk):
                continue
            out.append((warps, 0, fold))  # streaming: re-reads the row
            tpw = _ceil(nblk, warps)
            if tpw <= norm.MAX_TPW:
                out.append((warps, tpw, fold))  # persistent: holds it
    return sorted(set(out))


def _consts(kind, tpw, fold):
    c = {"TPW": tpw, "FOLD": fold}
    if kind != "softmax":
        c["EPS"] = 1e-6
    return c


def _thunk(kind, x, out, warps, tpw, fold):
    rows, cols = norm._shape_of(x)
    k = norm.KERNELS[f"{kind}_{norm._suffix_of(x)}_w{warps}"]
    xv, ov = norm._view(x, rows, cols, fold), norm._view(out, rows, cols, fold)
    c = _consts(kind, tpw, fold)
    return lambda: k(xv, ov, **c)


_interleave = _timing.interleave
def _label(warps, tpw, fold):
    return f"w{warps:<2d} tpw{tpw} fold{fold:<2d}"


def run(kind: str, rows: int, cols: int):
    x = torch.randn(rows, cols, device=DEV, dtype=DT)
    out = torch.empty_like(x)
    nbytes = 2 * x.numel() * 2  # one read, one write: the floor for any variant

    variants, skipped = {}, []
    for warps, tpw, fold in candidates(cols):
        label = _label(warps, tpw, fold)
        try:
            # build() before timing: a variant that spills is eliminated
            # without spending a GPU run on it, and the first timed iteration
            # is then not also the JIT.
            norm.KERNELS[f"{kind}_{norm._suffix_of(x)}_w{warps}"].build(
                **_consts(kind, tpw, fold))
            variants[label] = _thunk(kind, x, out, warps, tpw, fold)
        except Exception as exc:  # noqa: BLE001 -- report it and keep going
            skipped.append((label, str(exc).splitlines()[0][:70]))

    # Same kind the wrapper passes, so `chosen` is what a caller would
    # actually get rather than what the rms default would give.
    chosen = _label(*norm.plan(cols, kind))
    mb = 2 * rows * cols * x.element_size() / 2**20
    print(f"\n{kind} {rows}x{cols} bf16  {mb:.0f} MB "
          f"({'cache' if mb <= 96 else 'HBM'})   plan() picks: {chosen}")
    print(f"  {'variant':20s} {'ms':>8s} {'GB/s':>8s} {'%HBM':>6s}")
    best = _interleave(variants)
    for label, ms in sorted(best.items(), key=lambda kv: kv[1]):
        gbs = nbytes / (ms * 1e-3) / 1e9
        mark = "  <- plan()" if label == chosen else ""
        print(f"  {label:20s} {ms:8.3f} {gbs:8.1f} {100 * gbs / HBM_GB_S:5.1f}%{mark}")
    for label, why in skipped:
        print(f"  {label:20s} {'skipped':>8s}  {why}")
    return min(best, key=best.get)


def main(argv):
    cases = CASES
    if len(argv) == 3:
        cases = [(argv[1], 8192 if argv[1] == "softmax" else 16384, int(argv[2]))]
    elif len(argv) != 1:
        sys.exit(__doc__)

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    winners = {}
    for kind, rows, cols in cases:
        winners[(kind, rows, cols)] = run(kind, rows, cols)
    print("\nwinner per case:")
    for (kind, rows, cols), label in winners.items():
        print(f"  {kind:8s} {rows:6d}x{cols:<6d} {label}")


if __name__ == "__main__":
    main(sys.argv)
