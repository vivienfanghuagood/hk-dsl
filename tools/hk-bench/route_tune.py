#!/usr/bin/env python3
"""Which kernel each linear layer of a model should run on, at every M a step
can have -- measured, and written as the routing table vllm_gemm reads.

The measurement is `hk.autotune.routes` (`tune_shape`), which says why the
grid is a grid and how a point is timed; this is the command that writes a
whole table, the one shipped in hk/tuned. To measure a model into this
machine's records instead -- what the vLLM patch reads over the shipped
table -- run `python -m hk.integration tune --model X`, or let the patch do
it at load (`hk.integration.vllm_autotune`).

At each (N, K) of the model and each M: the engine's own path (the installed
vLLM's `rocm_unquantized_gemm`, or `F.linear` without vLLM; with `--engine
sglang`, `F.linear`, which is what SGLang's linear reaches on ROCm), the
generated GEMM, and the decode GEMV where it takes the shape. A kernel takes
an M only if it beat the engine there by `--min-gain`; a tie stays the
engine's. Two runs an hour apart decided 12 of 740 points differently, every
one in FlyDSL's overlay at Ms where it and hk's GEMM are within 2%; what is
left is drift between sessions, which no rounds within one run see.
`--confirm` an earlier run's table keeps only the overlay's Ms both runs
took: two such tables differed at 3-4 points (docs/history/32).

Where FlyDSL is installed (`hk.integration.flydsl_gemm`; FLYDSL_ROOT names
its checkout), its GEMM is timed too, through the same call the routes make,
at every M one of its tiles divides. It never enters the runs above, which
are what an install without FlyDSL routes by. It gets an overlay instead:
for each tile its dispatcher picked, the measured Ms where it beat the
engine by `--min-gain` and the runs' own answer by `--opt-gain`, merged
over consecutive Ms of that tile. `--no-flydsl` leaves it out.

    HIP_VISIBLE_DEVICES=3 FLYDSL_ROOT=/workspace/FlyDSL python3 tools/hk-bench/route_tune.py \\
        --model Qwen/Qwen3-8B --out python/hk/tuned/routes_vllm_linear_gfx1100.json
    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/route_tune.py --engine sglang \\
        --model Qwen/Qwen3-8B --out python/hk/tuned/routes_sglang_linear_gfx1100.json

`--model` takes a comma-separated list, whose shapes are measured once each.

On gfx950 hk's own GEMM and decode GEMV do not build yet; the candidate there
is HipKittens' CDNA4 GEMM (hk.ops.hipkittens), which takes M and N in
multiples of 256, so the grid is those Ms. The MI355X's Infinity Cache is
256 MB, not 96:

    python3 tools/hk-bench/route_tune.py --paths hipkittens --rotate-mb 768 \\
        --model Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B \\
        --m $(seq -s, 256 256 8192),12288,16384 \\
        --out python/hk/tuned/routes_vllm_linear_gfx950.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

from hk.autotune.routes import (  # noqa: E402,F401 -- the tests read these here
    CLOSE_K, GRID, NOISE_K, OPTIONAL, PATHS, ROTATE_BYTES, TIMINGS, base_path, close_call,
    compare, decisions, dump, engine_path, estimate, linear_shapes, margin,
    model_config, optional_wins, prefer, runs, tune_shape)

_engine = engine_path


def splice(old: dict, new: dict, **gains) -> dict:
    """`old`'s row with the points at `new`'s Ms replaced by `new`'s, its runs, above and overlay
    re-derived from the merged points (hk.autotune.routes.entry); the spliced Ms and how they were
    measured go in meta["spliced"]."""
    from hk.autotune.routes import entry

    ms = {p[0] for p in new["points"]}
    pts = sorted([p for p in old["points"] if p[0] not in ms] + new["points"], key=lambda p: p[0])
    ent = entry(old["name"], pts, partial=bool(old.get("partial")), **gains)
    meta = dict(old.get("meta", {}))
    meta["spliced"] = {"ms": sorted(ms), **new.get("meta", {})}
    ent["meta"] = meta
    return ent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF name, or a path to its config.json")
    ap.add_argument("--name", default="", help="what the table says it was "
                    "measured for (default: --model)")
    ap.add_argument("--engine", choices=("vllm", "sglang"), default="vllm",
                    help="whose linear the kernels must beat; the table is "
                    "routes_vllm_linear_* or routes_sglang_linear_*")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--shapes", default="",
                    help="comma-separated subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--m", default="", help="comma-separated Ms (default: the grid)")
    ap.add_argument("--timing", choices=TIMINGS, default=None,
                    help="how a burst is timed (hk.autotune.routes.measure_points): default "
                    "queued on gfx950, graph (captured, cold weights) on RDNA")
    ap.add_argument("--nk", default="",
                    help="name=NxK,... measured instead of --model's linear_shapes")
    ap.add_argument("--merge-into", default="",
                    help="an existing table: write it with this run's rows added, each replacing "
                    "the row of its key whole (or, with --splice, only at the measured Ms), with "
                    "this run's measurement in the row's meta")
    ap.add_argument("--splice", action="store_true",
                    help="with --merge-into: replace only the measured Ms' points of an existing "
                    "row and re-derive its runs, keeping its other points (a fix at decode Ms "
                    "without re-measuring the whole grid); a row not in the table is added whole")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--paths", default="hk,hk_decode",
                    help="the candidates, of hk (the generated GEMM), hk_decode "
                    "(the decode GEMV) and hipkittens (HipKittens' CDNA4 GEMM)")
    ap.add_argument("--rotate-mb", type=int, default=ROTATE_BYTES // 2**20,
                    help="weight bytes one timed pass streams; three times the "
                    "part's Infinity Cache (96 MB on a W7900, 256 MB on an MI355X)")
    ap.add_argument("--opt-gain", type=float, default=0.02,
                    help="what an optional path (FlyDSL) must beat the runs' own "
                    "answer by, on top of beating the engine by --min-gain")
    ap.add_argument("--no-flydsl", action="store_true",
                    help="do not time FlyDSL's GEMM even where it is installed")
    ap.add_argument("--warmup", type=int, default=1,
                    help="sets timed and thrown away first at every point: the "
                    "first set after the correctness check runs FlyDSL a median "
                    "1.5%% slower than the sets after it")
    ap.add_argument("--sets", type=int, default=1,
                    help="sets of --rounds every point is timed in, each on "
                    "activations of its own")
    ap.add_argument("--remeasure", type=int, default=2,
                    help="sets more, on fresh activations, for a point with a "
                    "decision inside --close-k x its noise of the bar")
    ap.add_argument("--close-k", type=float, default=CLOSE_K)
    ap.add_argument("--noise-k", type=float, default=NOISE_K,
                    help="how many times a pair's measured noise a margin must "
                    "clear beyond its gain (0: the bare gains)")
    ap.add_argument("--raw", action="store_true",
                    help="keep every set's rounds in the points")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="print the points two tables decide differently, and stop")
    ap.add_argument("--confirm", default="",
                    help="an earlier run's table: an optional path keeps an M only "
                    "if that run's overlay took it too")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        ap.error("--out is required")
    if args.merge_into and args.confirm:
        ap.error("--merge-into and --confirm do not combine")
    if args.splice and not args.merge_into:
        ap.error("--splice needs --merge-into")
    paths = set(args.paths.split(","))
    if not paths <= set(PATHS):
        raise SystemExit(f"--paths: unknown {sorted(paths - set(PATHS))}")

    import torch

    from hk.integration import flydsl_gemm
    from hk.runtime.compile import toolchain_fingerprint

    dev = torch.device("cuda")
    arch = (torch.cuda.get_device_properties(dev).gcnArchName or "").split(":")[0]
    shapes, seen = [], set()
    if args.nk:
        for item in args.nk.split(","):
            nm, nk = item.split("=")
            n, k = (int(v) for v in nk.lower().split("x"))
            shapes.append((nm, n, k, 0))
    else:
        for model in args.model.split(","):
            for s in linear_shapes(model_config(model), args.tp):
                if s[1:3] not in seen:
                    seen.add(s[1:3])
                    shapes.append(s)
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    grid = [int(x) for x in args.m.split(",")] if args.m else GRID

    engine, engine_name = engine_path(args.engine)
    print(f"engine path: {engine_name}", flush=True)
    hkk = None
    if "hipkittens" in paths:
        from hk.ops import hipkittens as hkk

    fly = not args.no_flydsl and flydsl_gemm.available()
    print(f"optional: {flydsl_gemm.status() if not args.no_flydsl else 'FlyDSL left out'}",
          flush=True)
    table = dict(kind=f"{args.engine}_linear", arch=arch, model=args.name or args.model,
                 engine=engine_name, tp=args.tp, device=torch.cuda.get_device_name(dev),
                 torch=torch.__version__, hip=torch.version.hip,
                 blas=str(torch.backends.cuda.preferred_blas_library()),
                 toolchain=toolchain_fingerprint(), min_gain=args.min_gain,
                 rounds=args.rounds, warmup=args.warmup, sets=args.sets,
                 remeasure=args.remeasure, close_k=args.close_k,
                 noise_k=args.noise_k, date=time.strftime("%Y-%m-%d"), shapes={})
    if hkk is not None:
        table["hipcc"] = hkk.hipcc_version()
    from hk.target import current_arch
    table["timing"] = args.timing or ("queued" if current_arch() == "gfx950" else "graph")
    if args.rotate_mb != ROTATE_BYTES // 2**20:
        table["rotate_mb"] = args.rotate_mb
    if fly:
        table.update(optional={"flydsl": flydsl_gemm.status()}, opt_gain=args.opt_gain)
    confirmed = decisions(json.loads(Path(args.confirm).read_text())) if args.confirm else None
    if args.confirm:
        table["confirmed_by"] = args.confirm
    t0, remeasured = time.time(), 0
    for sname, n, k, _ in shapes:
        conf = ({m for (key, m), (_, took) in confirmed.items() if key == f"{n}x{k}" and took}
                if confirmed is not None else None)
        ent = tune_shape(n, k, engine=engine, ms=grid, min_gain=args.min_gain, name=sname,
                         lm_head=sname == "lm_head", confirm=conf, paths=paths, flydsl=fly,
                         rounds=args.rounds, warmup=args.warmup, sets=args.sets,
                         remeasure=args.remeasure, opt_gain=args.opt_gain,
                         close_k=args.close_k, noise_k=args.noise_k,
                         rotate_bytes=args.rotate_mb * 2**20, raw=args.raw,
                         timing=args.timing, log=lambda s: print(s, flush=True))
        remeasured += ent.pop("remeasured")
        ent.pop("seconds")
        if args.merge_into:
            ent["meta"] = {k: table[k] for k in ("model", "engine", "torch", "hip", "device", "toolchain",
                                                 "min_gain", "date", "timing") if k in table}
        table["shapes"][f"{n}x{k}"] = ent
        print(f"{sname} {n}x{k}: {len(ent['points'])} Ms, {time.time() - t0:.0f}s", flush=True)
        for lo, hi, path in ent["runs"]:
            print(f"    {lo:>6}..{hi:<6} {path}", flush=True)
        for name, ranges in ent.get("prefer", {}).items():
            for lo, hi, var in ranges:
                print(f"    {lo:>6}..{'' if hi is None else hi:<6} {name} {var}", flush=True)

    table["remeasured"] = remeasured
    if args.merge_into:
        base = json.loads(Path(args.merge_into).read_text())
        for key, ent in table["shapes"].items():
            old = base["shapes"].get(key)
            if args.splice and old is not None:
                ent = splice(old, ent, min_gain=args.min_gain, noise_k=args.noise_k,
                             opt_gain=args.opt_gain)
            base["shapes"][key] = ent
        table = base
    Path(args.out).write_text(dump(table))
    print(f"wrote {args.out} ({remeasured} points re-measured as close calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
