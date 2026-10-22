#!/usr/bin/env python3
"""One `ab` arm under two Pythons, interleaved: what a venv's packages do to it.

`python -m hk.integration ab` runs every arm under the Python that started it,
so it cannot tell two installs apart. This runs the same arm (`stock` by
default) under each venv's Python in turn, round by round with the order
rotated, each a fresh engine in a child process exactly as `ab` starts one,
and judges the second venv against the first with `ab`'s verdicts (the first
is the baseline, reported as `stock`).

    python3 tools/hk-bench/venv_ab.py \\
        --py tri371=/workspace/venv-vllm031-tri371/bin/python \\
        --py tri380=/workspace/venv-vllm031/bin/python \\
        --model Qwen/Qwen3-8B --context 1024,4096 --batch 16,32 \\
        --tokens 72 --short 32 --reps 2 --out raw.json --json verdicts.json

Environment variables pass through to every child; per-venv ones (a cache
directory, say) go in `--env NAME:VAR=VALUE`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from hk.integration import ab  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--py", action="append", required=True,
                    help="NAME=/path/to/python; the first is the baseline")
    ap.add_argument("--env", action="append", default=[],
                    help="NAME:VAR=VALUE, set for that venv's children only")
    ap.add_argument("--arm", default="stock")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--context", default="1024,4096")
    ap.add_argument("--batch", default="16,32")
    ap.add_argument("--tokens", type=int, default=72)
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    pys = [p.split("=", 1) for p in args.py]
    if len(pys) != 2:
        ap.error("two --py, the baseline first")
    extra = {}
    for e in args.env:
        name, kv = e.split(":", 1)
        k, v = kv.split("=", 1)
        extra.setdefault(name, {})[k] = v

    child = ["-m", "hk.integration.ab", "--model", args.model, "--dtype", args.dtype,
             "--context", args.context, "--batch", args.batch,
             "--tokens", str(args.tokens), "--short", str(args.short),
             "--reps", str(args.reps), "--arm", args.arm]
    rounds, toks, starts, failed = {}, {}, {}, {}
    for rnd in range(args.reps):
        order = pys[rnd % 2:] + pys[:rnd % 2]
        for name, py in order:
            label = "stock" if name == pys[0][0] else name
            f = f"/tmp/hk_venv_ab_{name}_{rnd}.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (str(ROOT / "python"), env.get("PYTHONPATH", "")) if p)
            env.update(extra.get(name, {}))
            print(f"--- {name} ({py}) round {rnd}", flush=True)
            r = subprocess.run([py] + child + ["--out", f], env=env)
            if r.returncode:
                failed.setdefault(name, []).append(r.returncode)
                print(f"--- {name} round {rnd} FAILED, exit {r.returncode}", flush=True)
                continue
            got = json.loads(Path(f).read_text())
            rounds.setdefault(label, []).append(
                {"round": rnd, **{k: {m: v[m] for m in ("min", "short", "prefill", "reps")
                                      if m in v}
                                  for k, v in got.items() if not k.startswith("_")}})
            for k, v in got.items():
                if not k.startswith("_"):
                    toks.setdefault(label, {}).setdefault(k, v["tokens"])
            starts.setdefault(label, []).append(got.get("_startup"))
    meta = {"model": args.model, "tokens": args.tokens, "short": args.short,
            "reps": args.reps, "arm": args.arm, "baseline": pys[0][0],
            "pythons": dict(pys), "failed": failed,
            "eager": os.environ.get("HK_EAGER", "1") == "1",
            "vllm_defaults": os.environ.get("HK_VLLM_DEFAULTS") == "1",
            "date": time.strftime("%Y-%m-%d %H:%M")}
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"_rounds": rounds, "_tokens": toks, "_startup": starts, "_meta": meta},
            indent=2))
    rows = ab.verdicts(rounds, args.tokens, args.short, toks)
    print(f"\nbaseline (reported as stock): {pys[0][0]}; ratio = {pys[0][0]} / {pys[1][0]}"
          f" time, below 1 the second venv is slower")
    ab.print_verdicts(rows)
    if args.json:
        ab.write_verdicts(Path(args.json), rows, meta)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
