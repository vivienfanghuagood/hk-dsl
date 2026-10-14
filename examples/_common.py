"""What every example shares: its command line, the check against torch, and
the timing.

Every example runs four ways:

    python3 examples/01_vector_add.py              # GPU: check against torch, then time both
    python3 examples/01_vector_add.py --dump ir    # the hk MLIR module -- no GPU, no compiler
    python3 examples/01_vector_add.py --dump cpp   # the HipKittens C++ it translates to
    python3 examples/01_vector_add.py --build      # compile with hipcc, print the resources -- no GPU

`--arch gfx1201` traces and builds for a part that is not in the machine.

Timing follows the rule the kernels in this repository were measured by: the
contenders run in one process, interleaved round by round, and each reports
its best round. Absolute numbers drift between processes; ratios inside one
do not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict

# Run from a checkout: the package is next to the examples.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def parser(doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument("--dump", choices=("ir", "cpp"),
                   help="print the kernel's IR or C++ and exit (no GPU)")
    p.add_argument("--build", action="store_true",
                   help="compile with hipcc and print register/LDS usage (no GPU)")
    p.add_argument("--arch", help="build for this arch instead of the one present")
    p.add_argument("--no-bench", action="store_true", help="check correctness only")
    p.add_argument("--rounds", type=int, default=5, help="timing rounds per contender")
    return p


def parse(doc: str, extra: Callable[[argparse.ArgumentParser], None] = None):
    p = parser(doc)
    if extra:
        extra(p)
    args = p.parse_args()
    if args.arch:
        # Read on the first trace, so setting it here is early enough.
        os.environ["HK_ARCH"] = args.arch
    return args


def static(args, kernel, **consts) -> bool:
    """`--dump` / `--build`: what can be done without a GPU. True when the
    example should stop here."""
    if args.dump == "ir":
        print(kernel.trace(**consts).dump())
        return True
    if args.dump == "cpp":
        print(kernel.source(**consts))
        return True
    if args.build:
        for k in kernel.build(**consts).kernels:
            print(k.line())
        return True
    return False


def check(name: str, got, want, *, rel: float) -> float:
    """Fail unless max|got - want| <= rel * max|want|. Relative to the
    reference's magnitude, so it means the same at every shape -- and a
    reference with no magnitude fails rather than passing everything."""
    import torch

    g, w = got.float(), want.float()
    scale = w.abs().max().item()
    if scale == 0:
        raise AssertionError(f"{name}: the reference is all zeros; it checks nothing")
    err = (g - w).abs().max().item() / scale
    if not err <= rel:  # also catches NaN
        raise AssertionError(f"{name}: max error {err:.3e} of the reference's "
                             f"magnitude, allowed {rel:.1e}")
    torch.cuda.synchronize()
    print(f"  {name:<28} ok   max error {err:.2e} (allowed {rel:.0e})")
    return err


def _time_batch(fn) -> float:
    """ms per call, over a batch long enough (>= 2 ms) that the events'
    resolution and the launch gap do not dominate a short kernel."""
    import torch

    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    n = max(1, int(2.0 / max(s.elapsed_time(e), 1e-3)))
    best = float("inf")
    for _ in range(3):
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n)
    return best


def compare(fns: Dict[str, Callable[[], object]], rounds: int = 5) -> Dict[str, float]:
    """Best ms per call for each contender, interleaved round by round."""
    import torch

    for fn in fns.values():  # warm: compile, autotune lookups, allocator
        fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            best[k] = min(best[k], _time_batch(fn))
    return best


def report(title: str, best: Dict[str, float], *, baseline: str,
           work: float = 0.0, unit: str = "") -> None:
    """One row per contender: time, throughput if `work` is given (flops or
    bytes per call), and speed relative to `baseline`."""
    print(f"  {title}")
    for k, ms in best.items():
        rate = f"{work / ms * 1e-9:9.1f} {unit}" if work else ""
        print(f"    {k:<22} {ms:9.3f} ms {rate}  {best[baseline] / ms:6.2f}x vs {baseline}")


def require_gpu() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("this example needs torch on a GPU; --dump / --build work without one")
    if not torch.cuda.is_available():
        sys.exit("no GPU visible; --dump ir|cpp and --build work without one")
