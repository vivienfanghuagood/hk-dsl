#!/usr/bin/env python3
"""Does a non-power-of-two warp count pay at a width that splits badly?

Yes, and `norm._split` was changed because of it -- this file is the
measurement that paragraph cites, kept runnable.

5120 columns folds to 5 column blocks. `_split` used to round the warp count
down to a power of two, so 5 blocks were covered by 4 warps holding 2 tiles
each: 8 warp-slots for 5 blocks, every warp paying for a second live tile and
three eighths of them doing redundant work on the clamped last block. It was
the one width in the Phase 2 bench where hk lost to torch.compile (quantize
16384x5120: 0.407 against 0.371).

So this instantiates the warp counts `plan` could not ask for -- 3, 5, 6 -- at
that width, in one process, round robin, against compile. The answer was w5
tpw1 and w6 tpw1, both ahead of compile, and the axis that explains it is TPW
rather than the slot count: w3 and w6 hand out the same six slots and w6 wins.
`_split` now hands out one warp per block up to MAX_WARPS.

Re-run it if _split, MAX_TPW or the widening load changes; all three move
these numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/workspace/HipKittens/python")
sys.path.insert(0, "/workspace/HipKittens/tools/hk-bench")

import torch
import _timing
from hk.ops import norm, quant

DEV, DT = "cuda", torch.bfloat16
ROWS, COLS, FOLD = norm.ROWS, norm.COLS, norm.ROWS

# (warps, tpw) over the 5 blocks a folded 5120-wide row has. The first three
# are what plan() can reach today; the rest are what it cannot.
SPLITS = [(4, 2), (2, 3), (1, 5), (5, 1), (3, 2), (6, 1)]


def quantize_case(rows=16384, cols=5120):
    x = torch.randn(rows, cols, device=DEV, dtype=DT)
    q = torch.empty(rows, cols, device=DEV, dtype=torch.int8)
    s = torch.empty(1, rows, device=DEV, dtype=torch.float32)
    xv, qv = (norm._view(t, rows, cols, FOLD) for t in (x, q))
    v, skipped = {}, []
    for w, tpw in SPLITS:
        k = quant._quantize_kernel(f"quantize_bf16_w{w}", norm.bf16, w)
        c = {"TPW": tpw, "FOLD": FOLD, "EXACT": int(norm._exact(cols // FOLD, w, tpw))}
        try:
            k.build(**c)
        except Exception as exc:
            skipped.append((f"w{w} tpw{tpw}", str(exc).splitlines()[0][:60]))
            continue
        v[f"w{w:<2d} tpw{tpw}"] = lambda k=k, c=c: k(xv, qv, s, **c)

    def ref():
        amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        return (x.float() / (amax / 127.0)).round().clamp(-128, 127).to(torch.int8)
    v["compile"] = torch.compile(ref, dynamic=False)
    return f"quantize {rows}x{cols}", x.numel() * 3, v, skipped


def norm_case(kind, rows, cols):
    x = torch.randn(rows, cols, device=DEV, dtype=DT)
    o = torch.empty_like(x)
    xv, ov = (norm._view(t, rows, cols, FOLD) for t in (x, o))
    v, skipped = {}, []
    for w, tpw in SPLITS:
        k = norm._row_reduce_kernel(f"{kind}_bf16_w{w}", norm.bf16, kind, w)
        c = {"TPW": tpw, "FOLD": FOLD, "EXACT": int(norm._exact(cols // FOLD, w, tpw))}
        if kind != "softmax":
            c["EPS"] = 1e-6
        try:
            k.build(**c)
        except Exception as exc:
            skipped.append((f"w{w} tpw{tpw}", str(exc).splitlines()[0][:60]))
            continue
        v[f"w{w:<2d} tpw{tpw}"] = lambda k=k, c=c: k(xv, ov, **c)

    if kind == "rms":
        ref = lambda: x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    else:
        ref = lambda: torch.softmax(x, -1)
    v["compile"] = torch.compile(ref, dynamic=False)
    return f"{kind} {rows}x{cols}", 2 * x.numel() * 2, v, skipped


def report(label, nbytes, variants, skipped):
    print(f"\n{label}  {nbytes / 2**20:.0f} MB")
    best = _timing.interleave(variants)
    for name, ms in sorted(best.items(), key=lambda kv: kv[1]):
        print(f"  {name:12s} {ms:8.3f} {nbytes / (ms * 1e-3) / 1e9:8.1f} GB/s")
    for name, why in skipped:
        print(f"  {name:12s}  skipped  {why}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    report(*quantize_case())
    report(*norm_case("rms", 16384, 5120))
    report(*norm_case("softmax", 8192, 5120))
