#!/usr/bin/env python3
"""Which kernel each linear layer of a model should run on, at every M a step
can have -- measured, and written as the routing table vllm_gemm reads.

The rule this replaces was four lines transcribed from seven Ms. The GEMM's
speed is not monotone in M: the last 128-row block is computed whole however
little of it is kept, so between multiples of 128 the kernel wastes work in a
sawtooth, and at small grids two workgroups share a WGP. A rule written as
thresholds, or bucketed by powers of two, either routes the teeth or gives up
the gaps between them. So this measures the grid itself: every M up to 64,
every 16 rows to 1024, coarser above, where the wasted rows are at most an
eighth of the work.

At each (N, K) of the model and each M: the engine's own path (the installed
vLLM's `rocm_unquantized_gemm`, or `F.linear` without vLLM; with `--engine
sglang`, `F.linear`, which is what SGLang's linear reaches on ROCm), the
generated GEMM, and the decode GEMV where it takes the shape. Each is checked against fp32 before it
is timed; they interleave in one process, min over rounds, the weights
rotated through more than three times the Infinity Cache. A kernel takes an M
only if it beat the engine there by `--min-gain`; a tie stays the engine's.

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

The table is per arch and per model: a (N, K) it has not measured takes the
engine's path, so another model needs its own run (a few minutes) before
anything of it is routed. `--model` takes a comma-separated list, whose
shapes are measured once each.

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
sys.path.insert(0, str(HERE))

from vllm_gemm import MAX_SEQS, ROTATE_BYTES, linear_shapes, model_config  # noqa: E402

GRID = (list(range(1, 65)) + list(range(80, 1025, 16)) +
        list(range(1088, 2049, 64)) + list(range(2176, 4097, 128)) +
        list(range(4608, 8193, 512)) + [12288, 16384])


def _engine(name="vllm"):
    """What the engine's unquantized linear reaches, and its name. vLLM: the
    installed vLLM's own op where it registers one -- since 0.2x it runs
    skinny GEMMs (wvSplitK, LLMM1) at M <= 5 on gfx11 -- else F.linear.
    SGLang: F.linear, which its UnquantizedLinearMethod calls on ROCm without
    aiter, whether or not a vLLM is importable beside it. A table measured
    against one is a table for that engine."""
    import torch
    import torch.nn.functional as F

    if name == "sglang":
        return F.linear, "sglang F.linear"
    try:
        import vllm
        import vllm.model_executor.layers.utils  # noqa: F401 -- registers the op
        return (torch.ops.vllm.rocm_unquantized_gemm,
                f"vllm {vllm.__version__} rocm_unquantized_gemm")
    except (ImportError, AttributeError, RuntimeError):
        return F.linear, "F.linear"


#: Candidates from libraries hk does not depend on: timed, and routed by an
#: overlay rather than the runs (engine.RouteTable).
OPTIONAL = ("flydsl",)


def base_path(t, min_gain):
    """The runs' answer at one measured M: the fastest of hk's kernels if it
    beat the engine by `min_gain`, else the engine's path."""
    cand = {p: v for p, v in t.items() if p != "engine" and p not in OPTIONAL}
    best = min(cand, key=cand.get) if cand else None
    return best if best and cand[best] * (1 + min_gain) <= t["engine"] else "fallback"


def runs(points, min_gain):
    """[[lo, hi, path], ...] over the measured Ms, consecutive equal winners
    merged, and the path at the largest M (what the table answers above it)."""
    out = []
    for m, t, *_ in points:
        path = base_path(t, min_gain)
        if out and out[-1][2] == path:
            out[-1][1] = m
        else:
            out.append([m, m, path])
    return out, (out[-1][2] if out else "fallback")


def prefer(points, name, min_gain, opt_gain):
    """[[lo, hi, variant], ...]: for each variant (tile) of optional path
    `name`, the measured Ms where it beat the engine by `min_gain` and the
    runs' answer by `opt_gain`, consecutive Ms of that variant merged. `hi`
    is None where the variant's largest measured M was a win."""
    seqs = {}
    for m, t, *rest in points:
        var = rest[0].get(name) if rest else None
        if var is None or name not in t:
            continue
        mine, base = t[name], base_path(t, min_gain)
        t_base = t["engine"] if base == "fallback" else t[base]
        win = mine * (1 + min_gain) <= t["engine"] and mine * (1 + opt_gain) <= t_base
        seqs.setdefault(var, []).append((m, win))
    out = []
    for var, seq in seqs.items():
        cur = None
        for m, win in seq:
            if not win:
                cur = None
            elif cur is None:
                cur = [m, m, var]
                out.append(cur)
            else:
                cur[1] = m
        if seq[-1][1]:
            cur[1] = None
    return sorted(out, key=lambda r: (r[0], r[2]))


def dump(table: dict) -> str:
    """JSON with one line per run and per measured M, so a diff of two tables
    reads as a diff of decisions."""
    head = {k: v for k, v in table.items() if k != "shapes"}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append(' "shapes": {')
    keys = list(table["shapes"])
    for i, key in enumerate(keys):
        s = table["shapes"][key]
        lines.append(f'  {json.dumps(key)}: {{"name": {json.dumps(s["name"])}, '
                     f'"above": {json.dumps(s["above"])},')
        pref = {name: rs for name, rs in s.get("prefer", {}).items() if rs}
        if pref:
            lines.append('   "prefer": {')
            for j, (name, ranges) in enumerate(pref.items()):
                lines.append(f"    {json.dumps(name)}: [")
                lines += [f"     {json.dumps(r)}," for r in ranges]
                lines[-1] = lines[-1].rstrip(",")
                lines.append("    ]" + ("," if j < len(pref) - 1 else ""))
            lines.append("   },")
        lines.append('   "runs": [')
        lines += [f"    {json.dumps(r)}," for r in s["runs"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append('   ], "points": [')
        lines += [f"    {json.dumps(p)}," for p in s["points"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("   ]}" + ("," if i < len(keys) - 1 else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = set(args.paths.split(","))
    if not paths <= {"hk", "hk_decode", "hipkittens"}:
        raise SystemExit(f"--paths: unknown {sorted(paths - {'hk', 'hk_decode', 'hipkittens'})}")
    rotate_bytes = args.rotate_mb * 2**20

    import torch
    import torch.nn.functional as F

    import hk
    from hk.integration import flydsl_gemm
    from hk.runtime.compile import toolchain_fingerprint

    dev = torch.device("cuda")
    arch = (torch.cuda.get_device_properties(dev).gcnArchName or "").split(":")[0]
    shapes, seen = [], set()
    for model in args.model.split(","):
        for s in linear_shapes(model_config(model), args.tp):
            if s[1:3] not in seen:
                seen.add(s[1:3])
                shapes.append(s)
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    grid = [int(x) for x in args.m.split(",")] if args.m else GRID

    ops = {}

    def gemm_op(m, n, k):
        if "hk" not in paths:
            return None
        from hk.ops import gemm as hk_gemm

        s = hk_gemm.schedule_for(m, n, k)
        if k % s["k_step"] or n % s["block_n"]:
            return None
        key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
        if key not in ops:
            ops[key] = hk.torch_op(hk_gemm.TUNER._kernel_for(s))
        return ops[key]

    def decode_op(m, n, k):
        if "hk_decode" not in paths:
            return None
        from hk.ops import decode_gemm as hk_decode

        if not hk_decode.fits(m, n, k):
            return None
        s = hk_decode.schedule_for(m, k)
        key = ("decode",) + tuple(sorted(s.items()))
        if key not in ops:
            ops[key] = hk.torch_op(hk_decode.kernel_for(**s))
        return ops[key]

    engine, engine_name = _engine(args.engine)
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
                 rounds=args.rounds, date=time.strftime("%Y-%m-%d"), shapes={})
    if hkk is not None:
        table["hipcc"] = hkk.hipcc_version()
    if args.rotate_mb != ROTATE_BYTES // 2**20:
        table["rotate_mb"] = args.rotate_mb
    if fly:
        table.update(optional={"flydsl": flydsl_gemm.status()}, opt_gain=args.opt_gain)
    t0 = time.time()
    for sname, n, k, _ in shapes:
        w_bytes = n * k * 2
        copies = max(2, -(-rotate_bytes // w_bytes))
        g = torch.Generator(device=dev).manual_seed(0)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
              * k ** -0.5 for _ in range(copies)]
        points = []
        for m in grid:
            if sname == "lm_head" and m > MAX_SEQS:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            scale = ref.abs().max().item()
            variants = {"engine": lambda w: engine(x, w, None)}
            op = gemm_op(m, n, k)
            if op is not None:
                variants["hk"] = lambda w, op=op: op(x, w, out) or out
            dop = decode_op(m, n, k)
            if dop is not None:
                variants["hk_decode"] = lambda w, dop=dop: dop(x, w, out) or out
            if hkk is not None and hkk.gemm_fits(m, n, k):
                variants["hipkittens"] = lambda w: hkk.gemm_bf16(x, w, out=out)
            tiles = {}
            if fly and flydsl_gemm.variant(m, n, k) is not None:
                tiles["flydsl"] = flydsl_gemm.variant(m, n, k)
                variants["flydsl"] = lambda w: flydsl_gemm.run(x, w, out) or out
            ok = {}
            for name, call in variants.items():
                y = call(ws[0]).float()
                err = (y - ref).abs().max().item() / max(scale, 1e-30)
                if name != "engine" and not err < 2e-2:
                    print(f"  {sname} m={m}: {name} FAILED the gate "
                          f"(rel err {err:.3g}); not a candidate", flush=True)
                    continue
                ok[name] = call
            best = {name: float("inf") for name in ok}
            order = list(ok)
            for r in range(args.rounds):
                for name in (order if r % 2 == 0 else order[::-1]):
                    call = ok[name]
                    call(ws[-1])
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for w in ws:
                        call(w)
                    e.record()
                    torch.cuda.synchronize()
                    best[name] = min(best[name], s.elapsed_time(e) / copies)
            times = {p: round(v, 5) for p, v in best.items()}
            tiles = {p: v for p, v in tiles.items() if p in times}
            points.append([m, times, tiles] if tiles else [m, times])
            del x, out, ref
        rs, above = runs(points, args.min_gain)
        ent = dict(name=sname, runs=rs, above=above, points=points)
        pref = {name: prefer(points, name, args.min_gain, args.opt_gain)
                for name in OPTIONAL if fly}
        if any(pref.values()):
            ent["prefer"] = pref
        table["shapes"][f"{n}x{k}"] = ent
        print(f"{sname} {n}x{k}: {len(points)} Ms, {time.time() - t0:.0f}s", flush=True)
        for lo, hi, path in rs:
            print(f"    {lo:>6}..{hi:<6} {path}", flush=True)
        for name, ranges in pref.items():
            for lo, hi, var in ranges:
                print(f"    {lo:>6}..{'' if hi is None else hi:<6} {name} {var}", flush=True)
        del ws
        torch.cuda.empty_cache()

    Path(args.out).write_text(dump(table))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
