#!/usr/bin/env python3
"""Can a rule route a linear shape nobody measured? Held-out shapes, measured.

A routing table answers only the (N, K) it measured; everything else takes
the engine's path until someone tunes it. A rule for the shapes in between
-- "the decode GEMV from 6 to 64 rows" -- would give every model something
with no measuring at all, if it never sends a call to a kernel that is
slower there. This measures shapes no table has (`--models`: their linear
shapes, sharded for each `--tp`) with the tuner's own points
(`hk.autotune.routes.measure_points`), and `--check` scores rules against
any tables' points: for each, how many (shape, M) it routes to hk where hk
was slower than the engine (a misroute), where hk was faster but by less
than the table's 3% (one the table would not take), and how many of the
Ms the measurement would give hk it covers.

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/route_heldout.py \\
        --models Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen3-4B --tp 1,2 --out heldout.json
    python3 tools/hk-bench/route_heldout.py --check heldout.json \\
        python/hk/tuned/routes_vllm_linear_gfx1100.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

from hk.autotune.routes import (  # noqa: E402
    DECODE_MS, MAX_SEQS, MIN_GAIN, dump, engine_path, entry, linear_shapes, measure_points,
    model_config)

#: The rules scored. Each maps an unmeasured shape's M to the path it would
#: route it to: hk's decode GEMV over a band of Ms, and that plus hk's GEMM
#: from a row count up. "decode 6-64" is Qwen3-8B's table read as a rule;
#: on the first held-out set (53 + 18 shapes) it sent 15 points of large LM
#: heads at M 60-64 to a GEMV up to 6.2% slower, so "decode 6-48" was fixed
#: as the candidate before a second set (37 + 32 shapes) was measured. That
#: failed too: 36 points, down to -21.9%, where K leaves the GEMV 1-4 warps
#: (Qwen2.5-72B's 29568 and its tp4 7392) or a long K meets a narrow N at
#: 33-48 rows (Mistral-Small's down). No rule ships (docs/history/35).
RULES = {
    "decode 6-64": lambda m: "hk_decode" if 6 <= m <= 64 else "fallback",
    "decode 6-48": lambda m: "hk_decode" if 6 <= m <= 48 else "fallback",
    "decode 6-56": lambda m: "hk_decode" if 6 <= m <= 56 else "fallback",
    "decode 4-64": lambda m: "hk_decode" if 4 <= m <= 64 else "fallback",
    "decode 6-64, gemm >= 512": lambda m: ("hk_decode" if 6 <= m <= 64 else
                                           "hk" if m >= 512 else "fallback"),
    "decode 6-64, gemm >= 1024": lambda m: ("hk_decode" if 6 <= m <= 64 else
                                            "hk" if m >= 1024 else "fallback"),
}

PREFILL = (512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384)


def check(paths) -> int:
    pts = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for key, s in d["shapes"].items():
            gain = s.get("meta", {}).get("min_gain", d.get("min_gain", MIN_GAIN))
            for pt in s["points"]:
                pts.setdefault(key, {})[pt[0]] = (pt[1], gain, s["name"], p)
    print(f"{len(pts)} shapes, {sum(len(v) for v in pts.values())} measured points, "
          f"from {len(paths)} files")
    for rule, f in RULES.items():
        mis, tie, cover, want, worst = [], 0, 0, 0, None
        for key, by_m in pts.items():
            for m, (t, gain, name, src) in by_m.items():
                hk = {p: v for p, v in t.items() if p in ("hk", "hk_decode")}
                best = min(hk, key=hk.get) if hk else None
                deserved = best is not None and t["engine"] / hk[best] - 1 >= gain
                want += deserved
                path = f(m)
                if path == "fallback":
                    continue
                if path not in t:              # the kernel does not take it
                    continue
                r = t["engine"] / t[path] - 1
                if r < 0:
                    mis.append((key, name, m, r))
                elif r < gain:
                    tie += 1
                if deserved and path == best:
                    cover += 1
                if worst is None or r < worst[3]:
                    worst = (key, name, m, r)
        tail = (f"; worst {worst[1]} {worst[0]} M={worst[2]} hk {worst[3]:+.1%}"
                if worst else "")
        print(f"\n{rule}: {len(mis)} misroutes (hk slower), {tie} under the "
              f"{MIN_GAIN:.0%} bar, covers {cover} of {want} Ms the measurement "
              f"gives hk{tail}")
        for key, name, m, r in sorted(mis, key=lambda x: x[3])[:12]:
            print(f"    misroute: {name:8} {key:12} M={m:<6} hk {r:+.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="", help="comma-separated HF names or configs")
    ap.add_argument("--tp", default="1", help="comma-separated tensor-parallel sizes")
    ap.add_argument("--skip", default="", help="tables whose shapes are not held out")
    ap.add_argument("--m", default="", help="Ms (default: every decode M and a prefill sample)")
    ap.add_argument("--check", nargs="+", help="score the rules on these tables, and stop")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.check:
        return check(args.check)
    if not args.out:
        ap.error("--out is required")

    import torch

    skip = set()
    for p in filter(None, args.skip.split(",")):
        skip |= set(json.loads(Path(p).read_text())["shapes"])
    shapes = {}
    for model in filter(None, args.models.split(",")):
        cfg = model_config(model)
        for tp in (int(x) for x in args.tp.split(",")):
            for name, n, k, _ in linear_shapes(cfg, tp):
                key = f"{n}x{k}"
                if key not in skip and key not in shapes:
                    shapes[key] = (name, n, k, f"{model} tp{tp}")
    ms = ([int(x) for x in args.m.split(",")] if args.m else list(DECODE_MS) + list(PREFILL))
    engine, engine_name = engine_path("vllm")
    print(f"{len(shapes)} held-out shapes against {engine_name}", flush=True)
    table = dict(kind="vllm_linear", arch=(torch.cuda.get_device_properties(0).gcnArchName
                                           or "").split(":")[0],
                 engine=engine_name, min_gain=MIN_GAIN, date=time.strftime("%Y-%m-%d"),
                 heldout=True, shapes={})
    t0 = time.time()
    for key, (name, n, k, src) in shapes.items():
        grid = [m for m in ms if not (name == "lm_head" and m > MAX_SEQS)]
        pts, _, _ = measure_points(n, k, grid, engine=engine,
                                   log=lambda s: print(s, flush=True), tag=name)
        ent = entry(name, pts)
        ent["meta"] = {"from": src}
        table["shapes"][key] = ent
        print(f"{name:8} {key:12} {src}: {time.time() - t0:.0f}s  "
              + " ".join(f"{lo}-{hi}:{p}" for lo, hi, p in ent["runs"]), flush=True)
        Path(args.out).write_text(dump(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
