#!/usr/bin/env python3
"""What the 38% disagreement between phase2.py and norm_plan.py actually was.

The same rmsnorm kernel over the same tensor read 0.355 ms in norm_plan.py and
0.491 ms in phase2.py. Two suspects, and this file convicts one of them.

**Suspect 1: host overhead.** A cuda event pair measures the gap between the
two records, and on an idle stream that gap contains whatever the host does
between them -- so a wrapper that spends 100 us in Python shows up as a 100 us
kernel. phase2 calls `hk.ops.rmsnorm()`, which does a dtype lookup, a plan(),
an _exact(), two views and a dict build per call; norm_plan calls a Kernel with
all of that precomputed. Measured below as `host us`: wall clock per call with
no synchronize at all, which for a launch that never blocks is the host cost
and nothing else. **Acquitted:** 11 us for the wrapper, 7 us for the raw
Kernel. The wrapper costs 4 us and the kernel takes 360.

**Suspect 2: the round-robin itself.** phase2 interleaves hk with torch eager
and torch.compile one call at a time; norm_plan interleaves hk with other hk
variants. Those are not the same experiment. eager rmsnorm is 2.2 ms of
low-bandwidth, launch-heavy, high-shader-clock work, and a W7900D moves its
DVFS state in milliseconds, so the bandwidth-bound kernel timed immediately
after it is not timed at the clocks it would run at in a loop of its own. Round
-robin exists to keep a clock *ramp* from being charged to whichever variant
happened to be first; it cannot keep a variant from being charged for its
neighbour's power profile.

**And then suspect 1 was convicted on a different op.** 4 us in front of a
360 us rmsnorm is nothing; the same wrapper in front of a 60 us quantize is
not. `hk.ops.quantize` read 13.9 us a call here -- it computed a working set
`plan` discarded, on top of the dtype lookup, the plan, the _exact, the two
views, the f-string and the kwargs dict -- and the three quantize rows of
phase2.py each sat about 9 us above the same kernel measured directly in
quant_plan.py, losing to torch.compile by less than that. So the wrapper is
worth measuring per op and not once: the fix was a per-shape memo in each
wrapper (`norm._PLAN_CACHE` and friends) and a launcher memo in
`Kernel.__call__`, and this file is how the before and after were read.

Three rounds of that, measured here on a W7900D:

    variant            before   wrapper memo   dtype key   fast pybind
    hk.ops.rmsnorm       11.3        8.5           6.6          6.6
    Kernel(**consts)      6.9        5.7           6.2          6.2
    specialized           6.3        5.3           5.8          5.8
    hk.ops.quantize      13.9       11.2           9.9          9.9
    hk.ops.silu_mul        --       12.9          12.7         12.7

The third column is worth a note, because two of its entries went *up*. Keying
the plan cache on the `torch.dtype` object instead of `str(dtype)` removed a
str() per call from the wrapper rows and nothing from the other two -- the 0.4
us those gained back is run-to-run spread, not a regression, and it is a useful
scale reference: anything under half a microsecond here is noise.

The fourth column is the one that did not pay. `pyutils/hk_bind.cuh` reads a
tensor in five Python calls with interned attribute names, against pybind11's
generic caster; it is plainly less work, and it moved nothing measurable. By
then the remaining 6.6 us is in hk's own Python, not in the C++ boundary, and
that is where it stops being worth chasing: the kernel under it is 345 us.

So `regimes()` times the identical thunk three ways -- alone, round-robin with
eager and compile at block 1, and round-robin with them at block 8 -- and the
spread between the last two is the size of the artifact.

    python3 tools/hk-bench/host_overhead.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

import hk  # noqa: E402
from hk.ops import norm, quant  # noqa: E402

from phase2 import _burn_in, _interleave  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16


def host_us(fn, iters=200, warmup=20):
    """Wall clock per call, never synchronizing.

    The launches queue up behind each other, so as long as the queue does not
    fill this measures the host side alone. It is checked against a full
    synchronize afterwards: if the loop took longer than the GPU needed, the
    queue did fill and the number is contaminated.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / iters * 1e6, (t2 - t1) / iters * 1e6


def regimes(x, out, hk_fn):
    """The same hk thunk, timed under three round-robin schemes.

    `block` is how many consecutive calls of one variant a round makes before
    moving on. At 1 -- what phase2 does -- every hk timing is the first
    bandwidth-bound call after 2.9 ms of somebody else's work. At 8 the first
    call of a burst still pays that, but the min over the burst comes from the
    calls after the clocks have settled, which is the number a real workload
    would see.
    """
    ref = lambda: x * torch.rsqrt(  # noqa: E731
        x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    comp = torch.compile(ref, dynamic=False)
    trio = {"hk": hk_fn, "eager": ref, "compile": comp}

    print(f"\n{'regime':28s} {'hk ms':>8s} {'eager ms':>9s} {'compile ms':>11s}")
    alone = _interleave({"hk": hk_fn})
    print(f"{'hk alone':28s} {alone['hk']:8.3f} {'':>9s} {'':>11s}")
    for block in (1, 8):
        b = _burst(trio, block=block)
        print(f"{'with eager+compile, block %d' % block:28s} "
              f"{b['hk']:8.3f} {b['eager']:9.3f} {b['compile']:11.3f}")


def _burst(variants, block=8, rounds=6, warmup=5):
    """Round-robin over bursts of `block` consecutive calls.

    Each burst is timed call by call and the minimum kept, so the reported
    number comes from whichever call in the burst ran at the best clocks --
    which after a few calls is the variant's own steady state rather than a
    transition out of its neighbour's. Rotating the bursts keeps a global drift
    from landing on one variant, which is what plain round-robin is for.
    """
    for _ in range(warmup):
        for fn in variants.values():
            fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            for _ in range(block):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                best[name] = min(best[name], s.elapsed_time(e))
    return best


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    _burn_in(2.0)

    x = torch.randn(16384, 4096, device=DEV, dtype=DT)
    out = torch.empty_like(x)
    rows, cols = norm._shape_of(x)
    w, tpw, fold = norm.plan(cols)
    k = norm.KERNELS[f"rms_{norm._suffix_of(x)}_w{w}"]
    xv, ov = norm._view(x, rows, cols, fold), norm._view(out, rows, cols, fold)
    consts = {"TPW": tpw, "FOLD": fold,
              "EXACT": int(norm._exact(cols // fold, w, tpw)), "EPS": 1e-6}
    ks = k.specialize(**consts)

    variants = {
        "hk.ops.rmsnorm": lambda: hk.ops.rmsnorm(x, out=out),
        "Kernel(**consts)": lambda: k(xv, ov, **consts),
        "specialized": lambda: ks(xv, ov),
    }

    q = torch.empty(x.shape, dtype=torch.int8, device=DEV)
    sc = torch.empty(rows, dtype=torch.float32, device=DEV)
    variants["hk.ops.quantize"] = lambda: hk.ops.quantize(x, out=q, scales=sc)

    # silu_mul is the other op whose device time is short enough for the
    # wrapper to matter: 216 us at 2048x11008, where it lost to torch.compile
    # by 1.4%.
    a = torch.randn(2048, 11008, device=DEV, dtype=DT)
    bb = torch.randn_like(a)
    ao = torch.empty_like(a)
    variants["hk.ops.silu_mul"] = lambda: hk.ops.silu_mul(a, bb, out=ao)

    print(f"\n{'variant':20s} {'host us':>9s} {'queued us':>10s} {'device ms':>10s}")
    dev = _interleave(variants)
    for name, fn in variants.items():
        h, drain = host_us(fn)
        print(f"{name:20s} {h:9.1f} {drain:10.1f} {dev[name]:10.3f}")

    del q, sc, a, bb, ao
    regimes(x, out, variants["hk.ops.rmsnorm"])


if __name__ == "__main__":
    main()
