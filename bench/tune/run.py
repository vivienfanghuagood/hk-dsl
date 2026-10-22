"""Run the tuning benchmark: tune each shape of a split with hk.tune, then
score it -- hk before (the current route), hk after (hk.ops.matmul reading the
tuner's records) and every baseline, checked against fp32, timed interleaved
in one process.

    python3 -m bench.tune.run --split acceptance --only 0:4 --out DIR
    python3 -m bench.tune.run --split acceptance --score-only --out DIR
    python3 -m bench.tune.scoreboard DIR                 # the scoreboard

Each shape appends one JSON line to DIR/<arch>-<split>.jsonl; records go to
DIR/records (HK_TUNE_DIR), never to the package. Bounded: a call does the
shapes it is given and stops, so a long run is a loop of short ones.

Scoring a shape: S sets of R rounds (bench/tune/../hk.tune.measure.GemmBench:
weights rotated past the Infinity Cache, each pass queued behind a GPU spin,
arms rotated each round), fresh activations each set. An arm's time is its
median over sets of each set's median; "best available" is the fastest
correct baseline; hk's ratio to it is best/hk per set, its median and band.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="acceptance", choices=("tuning", "acceptance"))
    ap.add_argument("--only", default="", help="index range a:b into the split, or MxNxK list")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--budget", type=float, default=60.0, help="hk.tune GPU-seconds per shape")
    ap.add_argument("--score-only", action="store_true", help="score with the records as they are")
    ap.add_argument("--baselines", default=",".join(
        ("torch", "tunableop", "aiter", "aiter_triton", "flydsl", "hipkittens", "triton")))
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("-j", "--jobs", type=int, default=32)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    os.environ["HK_TUNE_DIR"] = str((out / "records").resolve())
    os.environ.setdefault("HK_TUNE_TUNABLEOP_FILE", str((out / "tunableop_%d.csv").resolve()))

    import torch  # noqa: PLC0415

    import hk  # noqa: PLC0415,F401
    import hk.tune as T  # noqa: PLC0415
    from hk.ops import gemm as hk_gemm  # noqa: PLC0415
    from hk.target import current_arch  # noqa: PLC0415
    from hk.tune import stats  # noqa: PLC0415
    from hk.tune.measure import Compiler, GemmBench  # noqa: PLC0415
    from hk.tune.space import provider  # noqa: PLC0415

    from . import baselines as B  # noqa: PLC0415
    from . import shapes as S  # noqa: PLC0415

    arch = current_arch()
    allshapes = S.load(arch, args.split)
    if args.only and "x" in args.only:
        todo = [T.parse_shape(s) for s in args.only.split(",")]
    elif args.only:
        a, b = (int(x) if x else None for x in args.only.split(":"))
        todo = allshapes[a:b]
    else:
        todo = allshapes
    fns, missing = B.available(arch, [b for b in args.baselines.split(",") if b])
    meta = dict(T._meta(arch), versions=B.versions(), missing=missing,
                commit=os.environ.get("HK_TUNE_SHA", ""),
                sets=args.sets, rounds=args.rounds, budget=args.budget, split=args.split)
    print(json.dumps(meta), flush=True)
    sp = provider(arch)
    comp = Compiler(args.jobs)
    jl = out / f"{arch}-{args.split}.jsonl"
    for m, n, k in todo:
        t0 = time.time()
        row = dict(m=m, n=n, k=k, arch=arch, split=args.split, meta=meta)
        B.reset()
        if not args.score_only:
            res = T.tune(m, n, k, budget=args.budget, compiler=comp,
                         baselines=T.torch_baselines(), log=lambda s: print(s, flush=True))
            row["tune"] = res.as_dict()
        # Score.
        bench = GemmBench(m, n, k, arch)
        cur = sp.current(m, n, k)
        arms = {"hk_before": sp.runner(cur, m, n, k),
                "hk_after": lambda a, b, o: hk_gemm.matmul(a, b, o)}
        arms.update({f"base:{x}": f for x, f in fns.items()})
        errs, ok = {}, {}
        for name, fn in arms.items():
            try:
                e = bench.check(fn)
            except Exception as ex:  # noqa: BLE001 -- recorded
                errs[name] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:160] if str(ex) else ''}"
                continue
            errs[name] = round(e, 5)
            if e < 2e-2:
                ok[name] = fn
            else:
                print(f"  {m}x{n}x{k} {name}: wrong ({e:.3g})", flush=True)
        floor = stats.floor_ms(m, n, k, arch)
        sets = {a: [] for a in ok}
        for _ in range(args.sets):
            bench.refresh()
            got = bench.time(ok, args.rounds)
            for a, ts in got.items():
                sets[a].append(ts)
        B.reset()
        med = {a: statistics.median(stats.summary(s, floor)[0] for s in v) for a, v in sets.items()}
        spread = {a: statistics.median(stats.summary(s, floor)[1] for s in v) for a, v in sets.items()}
        base = {a: t for a, t in med.items() if a.startswith("base:")}
        best = min(base, key=base.get) if base else None
        score = {}
        for h in ("hk_before", "hk_after"):
            if h in sets and best:
                r = stats.ratio_band(sets[best], sets[h], floor)
                score[h] = [round(x, 4) for x in r]
            if h in sets and "base:torch" in sets:
                r = stats.ratio_band(sets["base:torch"], sets[h], floor)
                score[h + "_vs_torch"] = [round(x, 4) for x in r]
        if arch == "gfx950":
            from hk.ops import gemm_cdna  # noqa: PLC0415

            name = gemm_cdna.route(m, n, k)
        else:
            u = hk_gemm.user_route(m, n, k)
            name = u[0] if u else str(cur)
        row.update(route_before=str(cur), route_after=str(name), errors=errs,
                   ms={a: round(t, 6) for a, t in med.items()},
                   spread={a: round(s, 4) for a, s in spread.items()},
                   best_baseline=best[5:] if best else None, score=score,
                   triton_pick=_triton_pick(fns, m, n, k), score_gpu_seconds=round(bench.gpu_seconds, 2),
                   wall_seconds=round(time.time() - t0, 1))
        bench.close()
        with jl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        b = score.get("hk_before", [None])[0]
        a = score.get("hk_after", [None])[0]
        print(f"SCORE {m}x{n}x{k}: best {row['best_baseline']} {_us(base.get(best))}; "
              f"hk before {_us(med.get('hk_before'))} ({b}x), after {_us(med.get('hk_after'))} "
              f"({a}x) [{row['route_after']}]", flush=True)
    comp.close()
    return 0


def _us(ms):
    return f"{ms * 1e3:.1f}us" if ms else "-"


def _triton_pick(fns, m, n, k):
    if "triton" not in fns:
        return None
    try:
        from . import triton_gemm  # noqa: PLC0415

        return triton_gemm.picked(m, n, k)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
