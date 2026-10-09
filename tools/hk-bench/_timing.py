"""The timing loop every sweep in this directory uses, and why it is this one.

`/kernel-ab-bench` says: one process, interleaved, because absolute numbers
drift between processes -- on wx-ms-w7900d-0043 that drift once manufactured a
13-16% difference out of nothing. That much is settled. What was not settled
is how *finely* to interleave, and the obvious answer turned out to be wrong.

One call of each variant per round is what this directory used to do. The
reason is real: a clock ramp during a contiguous block of timings is charged
entirely to whichever variant was running. But these variants do not draw the
same power. A torch eager rmsnorm at 16384x4096 is 2.2 ms of launch-heavy,
low-bandwidth, high-shader-clock work; the hk kernel is 0.4 ms of saturated
HBM. A W7900D moves its DVFS state on a millisecond scale, so at block 1 every
hk timing is the first bandwidth-bound call after 2.9 ms of somebody else's
profile, and it is charged for the transition. The slow variants amortize their
own transition inside a single call and never pay it. The scheme was
systematically penalising whichever variant was fastest.

Measured (tools/hk-bench/host_overhead.py, rmsnorm 16384x4096):

    hk alone                       0.407
    with eager+compile, block 1    0.486  eager 2.235  compile 0.639
    with eager+compile, block 8    0.409  eager 2.246  compile 0.614

Block 8 agrees with the isolated measurement for the variant that moved and
leaves the two that did not alone. That is the test for whether a timing scheme
is measuring the kernel or the schedule, and it is why the number below is 8
and not 1. The bursts are still rotated, so a genuine global drift is still
shared out across variants rather than landing on one.
"""

from __future__ import annotations

import time

import torch

#: Consecutive calls of one variant per round.
BLOCK = 8


def interleave(variants, rounds=6, block=BLOCK, warmup=5):
    """{name: thunk} -> {name: best ms}, round-robin over bursts."""
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


def burn_in(seconds: float = 3.0, device="cuda"):
    """Hammer HBM until the clocks stop climbing.

    Not a warmup of anything in particular -- the variants warm themselves.
    This is for the shader and memory clocks, which on a W7900D that has been
    idle start low and take on the order of a second of sustained traffic to
    reach their steady state. Whatever is measured during that second is
    measuring the ramp, and `interleave` can only spread a ramp across the
    variants *within* a case -- it can do nothing about the one that happens
    during the first case of a fresh process.
    """
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
    del a, b
