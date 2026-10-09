"""Fill the compile cache in parallel.

hipcc is 4-40 s per kernel and `hk.ops` ships 291 of them, so filling a cold
cache is about twenty minutes of one core while the other 255 sit idle. Nothing
about that is inherent: the builds are independent, the cache is
content-addressed, and `runtime.compile.build` already builds into a scratch
directory and renames into place -- a concurrent builder that loses the rename
race keeps the winner's artifact, which is byte-identical by construction. So
the only thing missing was something to issue them at once.

Measured, 24 norm kernels into an empty `HK_CACHE_DIR` on a 256-core box:

    j= 1    102.9 s     (4.29 s a kernel)
    j=24      5.4 s     19.1x, i.e. 80% of linear

The 20% is the serial trace phase plus the tail -- a pool of 24 finishes when
its slowest member does, and these kernels are not all the same size.

**Tracing stays serial, and that is not an oversight.** `ir.builder` keeps the
current builder in a module-global stack, so two threads tracing at the same
time would interleave into each other's IR. Tracing is also the cheap half --
microseconds against seconds -- so the split costs nothing:

    phase 1 (serial)    trace + emit C++ for every (kernel, constexprs)
    phase 2 (threaded)  hipcc, one subprocess per job

Threads and not processes because every worker spends its life inside
`subprocess.run`, which releases the GIL, and because a process pool would have
to pickle the source text to each child for no gain.

This is also what Phase 5's autotune needs: a sweep's candidates are
independent, and the ones that spill can be rejected without ever touching the
GPU.

    python3 -m hk.runtime.warm            # everything in hk.ops
    python3 -m hk.runtime.warm -j 32
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .compile import Build, build


@dataclass
class WarmResult:
    """What came back, per job. `error` is set exactly when `build` is None."""

    built: Dict[str, Build] = field(default_factory=dict)
    errors: Dict[str, BaseException] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors


def default_workers() -> int:
    # hipcc is a single-threaded front end per invocation, so one per core is
    # the right shape -- but capped, because each one is a few hundred MB of
    # resident template instantiation and a 256-core box does not have 256
    # times that.
    return max(1, min(32, (os.cpu_count() or 4)))


def jobs_for(kernels, consts: Sequence[Dict[str, Any]] = ({},)):
    """(label, kernel, constexprs) for every kernel x every constexpr set.

    `kernels` is a name -> Kernel mapping, as `hk.ops.norm.KERNELS` is.
    """
    out = []
    for name in sorted(kernels):
        for c in consts:
            suffix = "" if not c else " " + " ".join(f"{k}={v}" for k, v in sorted(c.items()))
            out.append((name + suffix, kernels[name], dict(c)))
    return out


def warm_cache(jobs: Iterable[Tuple[str, Any, Dict[str, Any]]],
         *, workers: Optional[int] = None, verbose: bool = False,
         scaffold: str = "pybind",
         extra_flags: Sequence[str] = ()) -> WarmResult:
    """Build every job, tracing serially and compiling `workers` at a time.

    Never raises for a job: a kernel that fails to compile, or that the spill
    gate refuses, lands in `errors` so that one bad kernel does not hide the
    state of the other 290. The caller decides whether that is fatal.

    `scaffold` and `extra_flags` exist so the torch registrations can be warmed
    by the same pool as the pybind modules. They are part of the cache key, so
    they have to match what the serving path will ask for exactly -- a warm
    pass with different flags fills the cache with entries nobody will hit.
    """
    t0 = time.perf_counter()
    res = WarmResult()

    # Phase 1, serial: trace and emit. A failure here is a tracing or
    # verification failure and is recorded the same way a compile failure is.
    specs: List[Tuple[str, Any, Dict[str, Any], str]] = []
    for label, kernel, consts in jobs:
        try:
            specs.append((label, kernel, consts,
                          kernel.source(scaffold, **consts)))
        except BaseException as e:  # noqa: BLE001 -- recorded, not swallowed
            res.errors[label] = e

    def one(spec):
        label, kernel, consts, source = spec
        return label, build(
            source, kernel.arch, name=kernel.name,
            max_vgprs=kernel.max_vgprs, min_occupancy=kernel.min_occupancy,
            extra_flags=extra_flags,
        )

    n = workers or default_workers()
    with ThreadPoolExecutor(max_workers=n) as pool:
        for label, fut in [(s[0], pool.submit(one, s)) for s in specs]:
            try:
                _, b = fut.result()
                res.built[label] = b
                if verbose:
                    print(f"  ok   {label}")
            except BaseException as e:  # noqa: BLE001
                res.errors[label] = e
                if verbose:
                    print(f"  FAIL {label}: {type(e).__name__}")

    res.seconds = time.perf_counter() - t0
    return res


def _all_shipped_jobs():
    """Every kernel in hk.ops, at the constexprs the test tier asks for.

    The second constexpr set is the top of the TPW range on the sixteen-warp
    kernels, which is what `norm.plan` emits for a row of 65 to 80 blocks and
    the only place MAX_TPW is exercised. EXACT=0 on purpose: the exact variant
    has had its clamp and mask deleted and is the smaller of the two.
    """
    from ..ops import fused, norm, quant

    kernels = {
        **{f"norm.{n}": k for n, k in norm.KERNELS.items()},
        **{f"fused.{n}": k for n, k in fused.KERNELS.items()},
        **{f"quant.{n}": k for n, k in quant.KERNELS.items()},
    }
    jobs = jobs_for(kernels)
    top = {n: k for n, k in kernels.items()
           if n.endswith("_w16") and "TPW" in k.signature.parameters}
    jobs += jobs_for(top, [{"TPW": norm.MAX_TPW, "FOLD": 1, "EXACT": 0}])
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help=f"concurrent hipcc invocations (default {default_workers()})")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    jobs = _all_shipped_jobs()
    n = args.jobs or default_workers()
    print(f"warming {len(jobs)} builds, {n} at a time")
    res = warm_cache(jobs, workers=n, verbose=args.verbose)
    print(f"{len(res.built)} built, {len(res.errors)} failed, {res.seconds:.1f}s")
    for label, e in sorted(res.errors.items()):
        print(f"\n--- {label}: {type(e).__name__}\n{e}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
