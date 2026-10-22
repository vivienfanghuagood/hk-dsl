"""Profile the validation cases (bench/profile/cases.py) on this GPU.

    python3 bench/profile/run.py                 # every case this arch has
    python3 bench/profile/run.py --case b        # one

Writes bench/profile/results/<arch>/<case>.{txt,json} (the summary and the
report) and <case>.record.json (the raw measurements, without the thread
trace: tests/hk/ir/test_profile.py replays the verdict logic on copies of
them). docs/profile.md#validation-three-known-cases quotes the summaries.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))

CASES = HERE / "cases.py"

#: case -> (target, inputs, vs label, vs target), per arch.
PLAN = {
    "gfx950": {
        "a": ("gemm_plain", "gate_up_inputs", "bias_gelu", "gemm_bias_gelu"),
        "b": ("matmul", "shape_b_inputs", "hipblaslt", "hipblaslt"),
        "b-pp": ("matmul", "shape_b_inputs", "pp", "matmul_pp"),
        "c": ("gelu_mul_tiles", "gelu_mul_inputs", "strips", "gelu_mul_strips"),
    },
    "gfx1100": {
        "c": ("gelu_mul_tiles", "gelu_mul_inputs", "strips", "gelu_mul_strips"),
        "gemm": ("matmul", "square_inputs", "hipblaslt", "hipblaslt"),
    },
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", help="only these cases")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    import hk.profile as hp  # noqa: PLC0415
    from hk.target import current_arch  # noqa: PLC0415

    arch = current_arch()
    plan = PLAN.get(arch)
    if not plan:
        print(f"no validation cases for {arch}", file=sys.stderr)
        return 2
    out = Path(args.out) / arch
    out.mkdir(parents=True, exist_ok=True)
    code = 0
    for case, (target, inputs, label, vs) in plan.items():
        if args.case and case not in args.case:
            continue
        t0 = time.time()
        p = hp.profile(f"{CASES}:{target}", f"{CASES}:{inputs}", name=target,
                       vs={label: f"{CASES}:{vs}"},
                       record=str(out / f"{case}.record.json"),
                       log=lambda m: print(m, file=sys.stderr))
        text = str(p)
        (out / f"{case}.txt").write_text(text + "\n")
        (out / f"{case}.json").write_text(p.dumps())
        print(f"== {case} ({time.time() - t0:.0f} s)\n{text}\n", flush=True)
        code = max(code, 0 if p.ok else 2)
    return code


if __name__ == "__main__":
    sys.exit(main())
