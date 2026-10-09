"""`python3 -m hk.integration`.

    python3 -m hk.integration --status          # what is importable, what is patched
    python3 -m hk.integration --warm -j 32      # build the torch ops ahead of time

`--warm` is the half of "no JIT on the serving path" that `hk.autotune --aot`
does not cover. The two fill different cache entries: a torch registration is
compiled with libtorch's include and ABI flags, and the flags are part of the
cache key, so a cache full of pybind modules does not spare a server its first
20-40 second attention compile. Run both.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

from . import BACKENDS, apply, status


def warm_jobs() -> List[Tuple[str, object, dict]]:
    """Every kernel worth having a torch op for, as warm_cache jobs.

    Attention only, and deliberately: it is the one op these patches route
    through `torch.ops`, and compiling a torch registration for all 270 shipped
    kernels would spend minutes on entries nothing will look up.
    """
    from ..ops import attn  # noqa: PLC0415

    seen, jobs = set(), []
    for head_dim in attn.HEAD_DIMS:
        for causal in (False, True):
            for n in (4096, 16384, 65536):
                k = attn._kernel_for(head_dim, causal, None, n)
                if k.name in seen:
                    continue
                seen.add(k.name)
                jobs.append((k.name, k, {}))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m hk.integration",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", action="store_true",
                    help="precompile the torch.ops registrations")
    ap.add_argument("--status", action="store_true",
                    help="what is importable and what is patched")
    ap.add_argument("--patch", action="store_true",
                    help="apply every available patch and report")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="parallel compiles (default: one per core, capped)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.warm:
        from ..runtime.torch_ext import torch_flags  # noqa: PLC0415
        from ..runtime.warm import default_workers, warm_cache  # noqa: PLC0415

        jobs = warm_jobs()
        n = args.jobs or default_workers()
        print(f"warming {len(jobs)} torch registrations, {n} at a time")
        res = warm_cache(jobs, workers=n, verbose=args.verbose,
                         scaffold="torch", extra_flags=torch_flags())
        cached = sum(1 for b in res.built.values() if b.cached)
        print(f"{len(res.built)} built ({cached} already cached), "
              f"{len(res.errors)} failed, {res.seconds:.1f}s")
        for label, e in res.errors.items():
            print(f"  FAIL {label}: {type(e).__name__}: {e}")
        return 1 if res.errors else 0

    if args.patch:
        for name, what in apply().items():
            print(f"{name:10s} {what}")
        return 0

    for name, what in status().items():
        print(f"{name:10s} {what}")
    if not args.status:
        print(f"\nbackends: {sorted(BACKENDS)}; --patch to apply, --warm to "
              f"precompile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
