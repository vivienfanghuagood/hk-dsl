"""Does a one-row case's verdict follow the kernel or its output buffer?

Each case built as scoreboard.py builds it; hk and the named arm timed on
their own output buffers and on each other's, rounds interleaved and queued
behind a spin as the scoreboard queues them, median of the rounds:

    python3 tools/hk-bench/floor_placement.py \\
        "add 1x8192:torch" "rmsnorm 32x4096:flydsl" "exp 1x5120:torch"

At 1-32 rows every arm sits at the card's ~3.8 us floor, and which buffer a
kernel writes moved it 0.1 us either way (docs/history/43).
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import _rounds  # noqa: E402
import scoreboard as sb  # noqa: E402
from _timing import burn_in  # noqa: E402

GROUP = {"rms_norm": "llm-norm", "rmsnorm": "llm-norm", "layernorm": "llm-norm",
         "fused_add_rms_norm": "llm-norm", "softmax": "llm-norm", "quantize": "llm-quant",
         "add": "llm-pointwise", "mul": "llm-pointwise", "exp": "llm-pointwise"}


def swapped(run, dicts, a, b):
    """`run` with arms a's and b's buffers exchanged for the call."""
    def f():
        for d in dicts:
            d[a], d[b] = d[b], d[a]
        try:
            run()
        finally:
            for d in dicts:
                d[a], d[b] = d[b], d[a]
    return f


def case(name):
    op, shape = name.split(" ")
    m, n = (int(v) for v in shape.split("x"))
    sb._SHAPES.update(rows=(m,), hidden=(n,), inter=(n,))
    for c in sb.CASES[GROUP[op]](False):
        if c[0] == name:
            return c
    raise SystemExit(f"no case {name!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+", help='"<case>:<other arm>"')
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args()
    sb._STATIC.append(True)
    _rounds.use(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    burn_in()
    print(f"{'case':<26}{'other':<10}{'hk own':>9}{'other own':>11}{'hk on its':>11}"
          f"{'it on hk':>10}  us")
    for spec in args.cases:
        name, other = spec.rsplit(":", 1)
        _, _, _, _, arms, _ = case(name)
        dicts = [d for d in sb._arm_buffers(arms) if "hk" in d and other in d]
        if not dicts:
            raise SystemExit(f"{name}: hk and {other} share no buffer dict")
        hk_run, ot_run = arms["hk"][0], arms[other][0]
        fns = {"hk": hk_run, "other": ot_run,
               "hk_swapped": swapped(hk_run, dicts, "hk", other),
               "other_swapped": swapped(ot_run, dicts, "hk", other)}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        t = {k: statistics.median(v[0]) * 1e3
             for k, v in sb.gpu_rounds(fns, args.rounds, args.calls).items()}
        print(f"{name:<26}{other:<10}{t['hk']:>9.3f}{t['other']:>11.3f}"
              f"{t['hk_swapped']:>11.3f}{t['other_swapped']:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
