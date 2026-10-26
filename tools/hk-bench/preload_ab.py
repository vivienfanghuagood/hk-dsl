"""Kernel arguments preloaded into SGPRs, per scoreboard case: hk's planner
arm as it ships, the same calls with every noalias kernel swapped for its
preloaded twin (Kernel.preloaded), and the fastest other arm, buffers
rotated over the arms, one process:

    python3 tools/hk-bench/preload_ab.py --only llm-norm,llm-quant,llm-pointwise \\
        --rows 2048,8192 --hidden 4096,5120,8192

What norm.STRIP_PRELOAD_PAST was set from (docs/history/43).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import _rounds  # noqa: E402
import scoreboard as sb  # noqa: E402
from _timing import burn_in  # noqa: E402

from hk.dsl.kernel import Kernel  # noqa: E402

OTHERS = ("torch", "inductor", "triton", "triton_tuned", "flydsl", "flydsl_tuned", "aiter")
_PRE = [None]
_call = Kernel.__call__


def _swapped_call(self, *tensors, **consts):
    if _PRE[0] is not None and self.noalias:
        self = self.preloaded() if _PRE[0] else self._plain()
    return _call(self, *tensors, **consts)


def _plain(self):
    """The twin without preload, for a planner that picked the preloaded one."""
    if not self.preload:
        return self
    k = getattr(self, "_plain_twin", None)
    if k is None:
        k = self._plain_twin = type(self)(
            self.fn, arch=self._arch, warps=self.warps, grid=self.grid_fn,
            name=self.name, max_vgprs=self.max_vgprs, min_occupancy=self.min_occupancy,
            defines=self.defines, noalias=self.noalias)
        k._defaults = dict(self._defaults)
    return k


Kernel.__call__ = _swapped_call
Kernel._plain = _plain


def forced(run, pre):
    def f():
        _PRE[0] = pre
        try:
            run()
        finally:
            _PRE[0] = None
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="llm-norm,llm-quant,llm-pointwise")
    ap.add_argument("--rows", default="2048,8192")
    ap.add_argument("--hidden", default="4096,5120,8192")
    ap.add_argument("--inter", default="14336")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args()
    for dim in ("rows", "hidden", "inter"):
        sb._SHAPES[dim] = tuple(int(v) for v in getattr(args, dim).split(","))
    sb._STATIC.append(True)
    _rounds.use(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    burn_in()
    print(f"{'case':<28}{'plain':>9}{'preload':>9}{'other':>9}  preload/plain  (other)")
    for g in args.only.split(","):
        for name, work, unit, ref, arms, _ in sb.CASES[g](False):
            fns = {}
            os.environ["FLYDSL_AUTOTUNE"] = "1"
            for a in ("hk",) + OTHERS:
                spec = arms.get(a)
                if spec is None or isinstance(spec, str):
                    continue
                try:
                    spec[0]()
                    torch.cuda.synchronize()
                    if sb.rel(spec[1](), ref) <= 3e-2:
                        fns[a] = spec[0]
                except Exception:  # noqa: BLE001 -- an arm that fails is left out
                    pass
            os.environ.pop("FLYDSL_AUTOTUNE", None)
            run = fns.pop("hk")
            fns = {"plain": forced(run, False), "preload": forced(run, True), **fns}
            for f in fns.values():
                f()
            torch.cuda.synchronize()
            raw = sb.gpu_rounds(fns, args.rounds, args.calls,
                                rotate=sb.buffer_rotation(arms, ["hk"] + list(fns)[2:]))
            t = {}
            for a, (ts, recs) in raw.items():
                kept, _ = _rounds.screen(ts, recs)
                ms, _, _ = _rounds.summarize(ts, kept, "median")
                if ms is not None:
                    t[a] = ms * 1e3
            other = min((v, a) for a, v in t.items() if a in OTHERS)
            print(f"{name:<28}{t['plain']:>9.3f}{t['preload']:>9.3f}{other[0]:>9.3f}  "
                  f"{t['plain'] / t['preload']:>13.3f}  ({other[1]})", flush=True)


if __name__ == "__main__":
    main()
