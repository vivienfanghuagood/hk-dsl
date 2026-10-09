#!/usr/bin/env python3
"""Phase 2 ops: hk vs torch eager vs torch.compile, in one process.

The discipline is the one from `/kernel-ab-bench`, and it is not optional on a
shared node: absolute numbers drift between processes -- on wx-ms-w7900d-0043
that drift once manufactured a 13-16% difference out of nothing -- so every
variant is imported into *this* interpreter and the variants are interleaved
round-robin. Ratios within a process are trustworthy; the milliseconds are
not comparable to another run.

These are bandwidth-bound passes, so the number that means something is the
achieved fraction of HBM, not TFLOPs. The W7900D's ceiling is 864 GB/s;
anything reading and writing its arguments once and landing near that is done,
and a variant that is faster than the ceiling is measuring Infinity Cache, not
memory (which is why every case here is sized past 96 MB where it can be).

  python3 tools/hk-bench/phase2.py              # everything
  python3 tools/hk-bench/phase2.py rmsnorm rope
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402

import _timing  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16

#: Peak HBM on a W7900D, for the "% of HBM" column. A measured number, not the
#: spec sheet: see the memory note on Navi31 Infinity Cache.
HBM_GB_S = 864.0


#: The timing loop, and the paragraph explaining why it bursts rather than
#: alternating single calls, live in _timing.py -- norm_plan and quant_plan
#: sweep against the same reference and have to measure the same way.
_interleave = _timing.interleave
_burn_in = _timing.burn_in


def _compiled(fn):
    """torch.compile, or eager with a note if inductor is unusable here."""
    try:
        return torch.compile(fn, dynamic=False)
    except Exception:
        return None


# -- cases --------------------------------------------------------------------
#
# Each case returns (label, bytes_moved, {variant: thunk}). `bytes_moved` is
# what the kernel must read plus what it must write -- the denominator for the
# bandwidth column, and the thing a fused kernel reduces relative to eager.

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


#: (rows, cols). 4096 is the width the three plans agree on; 5120 is the one
#: norm.MAX_TPW=5 opened -- at four it fell through to FOLD -- and is
#: Qwen2.5-14B's hidden size, so it is a band a real model lands in rather
#: than a band that exists.
_NORM_SHAPES = ((4096, 4096), (16384, 4096), (16384, 5120))


@case
def rmsnorm():
    for rows, cols in _NORM_SHAPES:
        x = torch.randn(rows, cols, device=DEV, dtype=DT)
        out = torch.empty_like(x)
        ref = lambda x=x: x * torch.rsqrt(  # noqa: E731
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        c = _compiled(ref)
        v = {"hk": lambda x=x, out=out: hk.ops.rmsnorm(x, out=out), "eager": ref}
        if c:
            v["compile"] = c
        yield f"rmsnorm {rows}x{cols} bf16", 2 * x.numel() * 2, v


@case
def layernorm():
    for rows, cols in _NORM_SHAPES:
        x = torch.randn(rows, cols, device=DEV, dtype=DT)
        out = torch.empty_like(x)
        ref = lambda x=x, cols=cols: F.layer_norm(x, (cols,), eps=1e-5)  # noqa: E731
        c = _compiled(ref)
        v = {"hk": lambda x=x, out=out: hk.ops.layernorm(x, out=out), "eager": ref}
        if c:
            v["compile"] = c
        yield f"layernorm {rows}x{cols} bf16", 2 * x.numel() * 2, v


@case
def softmax():
    for cols in (4096, 16384):
        x = torch.randn(8192, cols, device=DEV, dtype=DT)
        out = torch.empty_like(x)
        ref = lambda x=x: torch.softmax(x, -1)  # noqa: E731
        c = _compiled(ref)
        v = {"hk": lambda x=x, out=out: hk.ops.softmax(x, out=out), "eager": ref}
        if c:
            v["compile"] = c
        yield f"softmax 8192x{cols} bf16", 2 * x.numel() * 2, v


@case
def silu_mul():
    for rows in (2048, 8192):
        a = torch.randn(rows, 11008, device=DEV, dtype=DT)
        b = torch.randn(rows, 11008, device=DEV, dtype=DT)
        out = torch.empty_like(a)
        ref = lambda a=a, b=b: F.silu(a) * b  # noqa: E731
        c = _compiled(ref)
        v = {"hk": lambda a=a, b=b, out=out: hk.ops.silu_mul(a, b, out=out),
             "eager": ref}
        if c:
            v["compile"] = c
        # 3 tensors for hk; eager also materializes silu(a), which is the
        # point of fusing and shows up as eager needing 4.
        yield f"silu_mul {rows}x11008 bf16", 3 * a.numel() * 2, v


@case
def rope():
    for bh, seq in ((32, 2048), (128, 1024)):
        x = torch.randn(bh, seq, 128, device=DEV, dtype=DT)
        out = torch.empty_like(x)
        ang = torch.rand(seq, 64, device=DEV) * 6.28
        cos, sin = ang.cos().contiguous(), ang.sin().contiguous()

        def ref(x=x, cos=cos, sin=sin):
            x1, x2 = x[..., :64], x[..., 64:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

        c = _compiled(ref)
        v = {"hk": lambda x=x, cos=cos, sin=sin, out=out:
             hk.ops.rope(x, cos, sin, out=out), "eager": ref}
        if c:
            v["compile"] = c
        yield f"rope {bh}x{seq}x128 bf16", 2 * x.numel() * 2, v


@case
def quantize():
    for rows, cols in _NORM_SHAPES:
        x = torch.randn(rows, cols, device=DEV, dtype=DT)
        q = torch.empty(x.shape, dtype=torch.int8, device=DEV)
        s = torch.empty(rows, dtype=torch.float32, device=DEV)

        def ref(x=x):
            amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
            return (x.float() / (amax / 127.0)).round().clamp(-128, 127).to(torch.int8)

        c = _compiled(ref)
        v = {"hk": lambda x=x, q=q, s=s: hk.ops.quantize(x, out=q, scales=s),
             "eager": ref}
        if c:
            v["compile"] = c
        yield f"quantize {rows}x{cols} bf16->int8", x.numel() * 3, v


def main(argv):
    names = argv[1:] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; have {sorted(CASES)}")

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")

    # Ramp the clocks before anything is timed. The first case in a fresh
    # process is otherwise measured on a cold GPU: in an earlier run rmsnorm,
    # which is case one, came out 38% slower here than the identical kernel did
    # in norm_plan.py, while every later case agreed. `_interleave` already
    # spreads a ramp across the variants *within* a case; it cannot do anything
    # about the ramp that happens during the first case.
    _burn_in()

    print(f"{'case':34s} {'variant':9s} {'ms':>9s} {'GB/s':>8s} {'%HBM':>6s} {'vs eager':>9s}")
    print("-" * 82)
    for name in names:
        # Consumed lazily, one yield at a time, and it has to stay that way:
        # the thunks a case yields close over its loop variables, so draining
        # the generator first would leave every case pointing at the last
        # tensors it allocated. The thunks bind them as default arguments now,
        # but the allocations are also 100s of MB apiece and there is no reason
        # to hold all of them at once.
        for label, nbytes, variants in CASES[name]():
            best = _interleave(variants)
            base = best.get("eager", min(best.values()))
            for v, ms in best.items():
                gbs = nbytes / (ms * 1e-3) / 1e9
                print(f"{label:34s} {v:9s} {ms:9.3f} {gbs:8.1f} "
                      f"{100 * gbs / HBM_GB_S:5.1f}% {base / ms:8.2f}x")
            print()


if __name__ == "__main__":
    main(sys.argv)
