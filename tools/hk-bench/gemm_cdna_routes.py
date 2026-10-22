#!/usr/bin/env python3
"""hk's own gfx950 GEMMs against hipBLASLt, FlyDSL and HipKittens, and the
route table hk.ops.gemm.matmul reads on gfx950.

At every (N, K, M) of the grid, each of hk's candidate kernels that takes the
shape is checked against fp32, then every arm is timed interleaved in one
process: rounds rotate which arm goes first, a round of an arm is one pass
streaming 3x the 256 MB Infinity Cache, queued behind a GPU spin so the host
has launched the whole pass before the first call runs (scoreboard.py's
queue, docs/history/32), and an arm's time is the median of its rounds. The
table records, per point, the fastest hk candidate and every arm's TFLOP/s
(median, best round and spread); the route is the fastest candidate.

    python3 tools/hk-bench/gemm_cdna_routes.py --out python/hk/tuned/routes_gemm_gfx950.json
    python3 tools/hk-bench/gemm_cdna_routes.py --only 4096x4096x4096 --no-write
    python3 tools/hk-bench/gemm_cdna_routes.py --routed --json grid.json

`--routed` times hk.ops.gemm_cdna.matmul -- the shipped table's route, one
arm -- against the same baselines instead of every candidate: the check of
a table in a process of its own, free of picking the best of many noisy
candidates.

The candidates are hk.ops.gemm_cdna.CANDIDATES. FlyDSL comes from $FLYDSL_ROOT
plus an installed flydsl wheel (skipped if it does not import).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

ROTATE_BYTES = 3 * 256 * 2**20
#: The spin a round waits behind: long enough for the host to launch a pass
#: (at most a few dozen calls of two launches each).
QUEUE_MIN_US, QUEUE_US_PER_CALL = 5000, 300
_SPIN = []


def _spin(torch, us):
    """torch's spin kernel for `us` microseconds. Its clock is measured once,
    after a warm-up call: the first call loads the spin kernel's module (1.3 ms
    in the rocm/pytorch image, ~15 ms in vLLM's), which a calibration that
    times it reads as a clock 16-100x too slow -- and every spin is then that
    much too short, queues nothing, and the rounds time the host. The clock is
    the best of three 2M-cycle spins (hk.tune.measure.spin_hz)."""
    if not _SPIN:
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            torch.cuda._sleep(2_000_000)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        _SPIN.append(2_000_000 / (best * 1e3))
    torch.cuda._sleep(int(us * _SPIN[0]))


#: (N, K) of Llama-3-8B and Qwen3-8B's linears, merged and unmerged, and the
#: two lm_heads.
NK = ((4096, 4096), (6144, 4096), (12288, 4096), (14336, 4096), (24576, 4096),
      (28672, 4096), (4096, 12288), (4096, 14336), (128256, 4096), (151936, 4096))
MS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
SQUARES = (4096, 8192)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--m", default="", help="comma-separated M (default: the grid)")
    ap.add_argument("--only", default="", help="comma-separated MxNxK instead of the grid")
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", default="", help="a table whose points the measured ones replace")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--routed", action="store_true",
                    help="time matmul's route as one arm instead of every candidate")
    ap.add_argument("--json", default="", help="write the points (with --routed)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from hk.ops import gemm_cdna, gemm_cdna_dp
    from hk.ops import hipkittens as hkk

    fly = None
    try:
        root = os.environ.get("FLYDSL_ROOT", "/home/hfang/hk-local/FlyDSL")
        if root not in sys.path:
            sys.path.append(root)
        from kernels.gemm.gemm_a16w16_gfx950 import gemm_a16w16 as fly
    except Exception as e:  # noqa: BLE001
        print(f"FlyDSL not measured: {type(e).__name__}: {str(e)[:100]}", flush=True)

    if args.only:
        pts = [tuple(int(x) for x in s.split("x")) for s in args.only.split(",")]
    else:
        ms = tuple(int(x) for x in args.m.split(",")) if args.m else MS
        pts = [(s, s, s) for s in SQUARES] + [(m, n, k) for n, k in NK for m in ms]
    dev = torch.device("cuda")
    meta = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                hip=torch.version.hip, hipcc=hkk.hipcc_version(),
                date=time.strftime("%Y-%m-%d"), rounds=args.rounds, stat="median",
                queued=True, routed=args.routed)
    print(json.dumps(meta), flush=True)
    points = []
    for (m, n, k) in pts:
        fl = 2.0 * m * n * k
        copies = max(2, -(-ROTATE_BYTES // (n * k * 2)))
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * k ** -0.5
              for _ in range(copies)]
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().t()
        arms = {}
        if args.routed:
            route = gemm_cdna.route(m, n, k)
            arms["hk:" + route] = lambda i: gemm_cdna.matmul(x, ws[i], out=out)
        else:
            for name in gemm_cdna.CANDIDATES:
                if gemm_cdna.fits(name, m, n, k):
                    arms["hk:" + name] = (lambda nm: (lambda i: gemm_cdna.run(nm, x, ws[i], out)))(name)
        arms["hipblaslt"] = lambda i: F.linear(x, ws[i])
        if fly is not None:
            arms["flydsl"] = lambda i: fly(x, ws[i].t(), out=out, layout="nt")
        if hkk.gemm_fits(m, n, k):
            arms["hipkittens"] = lambda i: hkk.gemm_bf16(x, ws[i], out=out)
        ok, errs = {}, {}
        for a, f in arms.items():
            try:
                out.zero_()
                y = f(0)
                torch.cuda.synchronize()
                y = out if y is None else y
                errs[a] = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            except Exception as e:  # noqa: BLE001
                errs[a] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
                continue
            if errs[a] < 2e-2:
                ok[a] = f
            else:
                print(f"  {m}x{n}x{k} {a}: FAILED the fp32 check ({errs[a]:.3g})", flush=True)
        for f in ok.values():
            for i in range(copies):
                f(i)
        torch.cuda.synchronize()
        times = {a: [] for a in ok}
        order = list(ok)
        for r in range(args.rounds):
            j = r % len(order)
            for a in order[j:] + order[:j]:
                ok[a](0)
                torch.cuda.synchronize()
                _spin(torch, max(QUEUE_MIN_US, QUEUE_US_PER_CALL * copies))
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for i in range(copies):
                    ok[a](i)
                e1.record()
                torch.cuda.synchronize()
                times[a].append(e0.elapsed_time(e1) / copies)
        tf = {a: fl / (statistics.median(t) * 1e-3) / 1e12 for a, t in times.items()}
        tf_best = {a: fl / (min(t) * 1e-3) / 1e12 for a, t in times.items()}
        spread = {a: statistics.median(abs(v / statistics.median(t) - 1) for v in t)
                  for a, t in times.items()}
        hk_arms = {a: v for a, v in tf.items() if a.startswith("hk:")}
        best = max(hk_arms, key=hk_arms.get) if hk_arms else None
        base = {a: v for a, v in tf.items() if not a.startswith("hk:")}
        top = max(base.values()) if base else 0.0
        rec = dict(m=m, n=n, k=k, tflops={a: round(v, 1) for a, v in tf.items()},
                   tflops_best={a: round(v, 1) for a, v in tf_best.items()},
                   spread={a: round(v, 4) for a, v in spread.items()},
                   err={a: (round(e, 5) if isinstance(e, float) else e) for a, e in errs.items()},
                   best=best[3:] if best else None,
                   ratio=round(hk_arms[best] / top, 3) if best and top else None)
        points.append(rec)
        line = "  ".join(f"{a} {v:.0f}" for a, v in tf.items())
        print(f"{m:>5}x{n:>6}x{k:>5}: {line}  || hk {best} {rec['ratio']}x best baseline",
              flush=True)
        del x, ws, out, ref
        gemm_cdna_dp._SCRATCH.clear()
        torch.cuda.empty_cache()

    if args.routed:
        if args.json:
            Path(args.json).write_text(json.dumps(dict(meta=meta, points=points), indent=1) + "\n")
            print(f"wrote {args.json}", flush=True)
        return 0
    if args.merge:
        old = json.loads(Path(args.merge).read_text())
        fresh = {(p["m"], p["n"], p["k"]) for p in points}
        points = [p for p in old["points"] if (p["m"], p["n"], p["k"]) not in fresh] + points
        meta = dict(old.get("meta", {}), merged=meta)
    routes = {}
    for p in sorted(points, key=lambda p: (p["n"], p["k"], p["m"])):
        if p["best"]:
            routes.setdefault(f"{p['n']}x{p['k']}", {})[str(p["m"])] = p["best"]
    doc = dict(arch="gfx950", meta=meta, candidates={k: list(v) for k, v in gemm_cdna.CANDIDATES.items()},
               routes=routes, points=points)
    if args.out and not args.no_write:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
