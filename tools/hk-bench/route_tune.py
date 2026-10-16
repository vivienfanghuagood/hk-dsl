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
is timed; they interleave in one process, the weights rotated through more
than three times the Infinity Cache. A kernel takes an M only if it beat the
engine there by `--min-gain`; a tie stays the engine's.

How a point is timed, and why (docs/history/32). A set is `--rounds` rounds
of every path on activations of its own; a path's time in a set is its
fastest round. The first set after the correctness check is thrown away
(`--warmup`): in it FlyDSL ran a median 1.5% slower than in the sets after,
and set against set it decided 23-26 of 740 points differently where later
sets disagreed on 5-9. Then one set is timed (`--sets`), and a point with a
decision within `--close-k` x its noise of the bar gets `--remeasure` sets
more; its time is then the median of the sets. Each path's noise -- its
rounds' spread in a set, or the sets' about their median, whichever is
larger -- is written beside its time, `[M, times, tiles, noise, sets]`.
On the same two runs an hour apart, the old way (one set, no warm-up)
decided 15 of 740 points differently and this way 6; two runs of this tool
an hour apart, 12, every one in FlyDSL's overlay at Ms where it and hk's
GEMM are within 2% (the old tool's pairs: 13-17, 2-4 of them in the runs).
What is left is drift between sessions, which no rounds within one run see.
`--confirm` an earlier run's table keeps only the overlay's Ms both runs
took: two such tables differed at 3-4 points.

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
import statistics
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
#: A decision within CLOSE_K x its pair's measured noise of its bar is
#: re-measured on fresh activations before it is taken.
CLOSE_K = 2.0
#: A margin must clear its gain and NOISE_K x the pair's noise. Off: at 1 and
#: 2 it flipped as many points between runs as the bare gains, and at 2 it
#: dropped FlyDSL rows that win end to end (docs/history/32).
NOISE_K = 0.0


def _noise(point):
    """{path: relative noise} of a measured point; {} for tables from before
    route_tune measured it, whose decisions are then the bare gains'."""
    return point[3] if len(point) > 3 else {}


def _tiles(point):
    return point[2] if len(point) > 2 else {}


def margin(t, noise, fast, slow, gain, k=NOISE_K):
    """(how much faster `fast` is than `slow`, the bar it must clear, the
    pair's noise): the bar is the gain or k x noise, whichever is larger."""
    sigma = noise.get(fast, 0.0) + noise.get(slow, 0.0)
    return t[slow] / t[fast] - 1, max(gain, k * sigma), sigma


def base_path(t, min_gain, noise=None, k=NOISE_K):
    """The runs' answer at one measured M: the fastest of hk's kernels if it
    beat the engine by `min_gain` and by k x their noise, else the engine's."""
    cand = {p: v for p, v in t.items() if p != "engine" and p not in OPTIONAL}
    best = min(cand, key=cand.get) if cand else None
    if not best:
        return "fallback"
    r, bar, _ = margin(t, noise or {}, best, "engine", min_gain, k)
    return best if r >= bar else "fallback"


def optional_wins(t, noise, name, min_gain, opt_gain, k=NOISE_K):
    """Whether optional path `name` beat the engine by `min_gain` and the
    runs' answer by `opt_gain`, each by k x the pair's noise as well."""
    base = base_path(t, min_gain, noise, k)
    base = "engine" if base == "fallback" else base
    r_e, bar_e, _ = margin(t, noise, name, "engine", min_gain, k)
    r_b, bar_b, _ = margin(t, noise, name, base, opt_gain, k)
    return r_e >= bar_e and r_b >= bar_b


def close_call(t, noise, min_gain, opt_gain, k=CLOSE_K, noise_k=NOISE_K):
    """True if a decision at this point would go the other way were one of
    its margins k x its noise away: hk's best path against the engine, hk's
    two paths against each other, an optional path against both (each
    against the bar `noise_k` sets)."""
    cand = sorted((v, p) for p, v in t.items() if p != "engine" and p not in OPTIONAL)
    pairs = []
    if cand:
        pairs.append((cand[0][1], "engine", min_gain))
    if len(cand) > 1:
        pairs.append((cand[0][1], cand[1][1], None))      # which of hk's two
    base = base_path(t, min_gain, noise, noise_k)
    for name in OPTIONAL:
        if name in t:
            pairs += [(name, "engine", min_gain),
                      (name, "engine" if base == "fallback" else base, opt_gain)]
    for fast, slow, gain in pairs:
        r, bar, sigma = margin(t, noise, fast, slow, gain or 0.0, noise_k)
        if (r <= k * sigma) if gain is None else (abs(r - bar) <= k * sigma):
            return True
    return False


def estimate(sets):
    """(times, noise) of one point from its sets of rounds, each set on
    activations of its own: a path's time is the median over the sets of its
    fastest round in each, and its noise the larger of the rounds' spread
    within a set (the median set's: its median round over its fastest) and
    the sets' spread about their median (half their range)."""
    times, noise = {}, {}
    for p in sets[0]:
        mins = [min(s[p]) for s in sets]
        t = statistics.median(mins)
        within = statistics.median(statistics.median(s[p]) / min(s[p]) - 1 for s in sets)
        across = (max(mins) - min(mins)) / 2 / t if len(sets) > 1 else 0.0
        times[p], noise[p] = t, max(within, across)
    return times, noise


def decisions(table):
    """{(shape, M): (the runs' path, whether FlyDSL's overlay takes it)} for
    every measured point of a table, by the table's own rule. A point where
    FlyDSL was not timed is one it does not take."""
    k = table.get("noise_k", 0.0)
    out = {}
    for key, s in table["shapes"].items():
        pref = s.get("prefer", {}).get("flydsl", [])
        for p in s["points"]:
            m, t = p[0], p[1]
            tile = _tiles(p).get("flydsl")
            fly = tile is not None and any(
                lo <= m and (hi is None or m <= hi) and var == tile for lo, hi, var in pref)
            out[(key, m)] = (base_path(t, table["min_gain"], _noise(p), k), fly)
    return out


def compare(a_path, b_path) -> int:
    """Print the points two tables decide differently."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    da, db = decisions(a), decisions(b)
    common = sorted(set(da) & set(db), key=lambda x: (x[0], x[1]))
    names = {key: s["name"] for key, s in a["shapes"].items()}
    flips = [x for x in common if da[x] != db[x]]
    for key, m in flips:
        (pa, fa), (pb, fb) = da[(key, m)], db[(key, m)]
        print(f"  {names.get(key, key):<8} M={m:<6} {pa:>9}{' +flydsl' if fa else '':<8} -> "
              f"{pb:>9}{' +flydsl' if fb else ''}")
    print(f"{len(flips)} of {len(common)} points decided differently "
          f"({sum(da[x][0] != db[x][0] for x in common)} in the runs, "
          f"{sum(da[x][1] != db[x][1] for x in common)} in FlyDSL's overlay)")
    return 0


def runs(points, min_gain, k=NOISE_K):
    """[[lo, hi, path], ...] over the measured Ms, consecutive equal winners
    merged, and the path at the largest M (what the table answers above it)."""
    out = []
    for p in points:
        m, t = p[0], p[1]
        path = base_path(t, min_gain, _noise(p), k)
        if out and out[-1][2] == path:
            out[-1][1] = m
        else:
            out.append([m, m, path])
    return out, (out[-1][2] if out else "fallback")


def prefer(points, name, min_gain, opt_gain, k=NOISE_K, confirm=None):
    """[[lo, hi, variant], ...]: for each variant (tile) of optional path
    `name`, the measured Ms where it beat the engine by `min_gain` and the
    runs' answer by `opt_gain` (each by k x the pair's noise as well),
    consecutive Ms of that variant merged. `hi` is None where the variant's
    largest measured M was a win. With `confirm`, a set of Ms, only those
    Ms can win: the ones an earlier run's overlay took."""
    seqs = {}
    for p in points:
        m, t = p[0], p[1]
        var = _tiles(p).get(name)
        if var is None or name not in t:
            continue
        win = (optional_wins(t, _noise(p), name, min_gain, opt_gain, k)
               and (confirm is None or m in confirm))
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
                 rounds=args.rounds, warmup=args.warmup, sets=args.sets,
                 remeasure=args.remeasure, close_k=args.close_k,
                 noise_k=args.noise_k, date=time.strftime("%Y-%m-%d"), shapes={})
    if hkk is not None:
        table["hipcc"] = hkk.hipcc_version()
    if args.rotate_mb != ROTATE_BYTES // 2**20:
        table["rotate_mb"] = args.rotate_mb
    if fly:
        table.update(optional={"flydsl": flydsl_gemm.status()}, opt_gain=args.opt_gain)
    confirmed = decisions(json.loads(Path(args.confirm).read_text())) if args.confirm else None
    if args.confirm:
        table["confirmed_by"] = args.confirm
    t0, remeasured = time.time(), 0
    for sname, n, k, _ in shapes:
        w_bytes = n * k * 2
        copies = max(2, -(-rotate_bytes // w_bytes))
        g = torch.Generator(device=dev).manual_seed(0)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
              * k ** -0.5 for _ in range(copies)]
        points = []

        def variants_for(m, x, out):
            variants = {"engine": lambda w: engine(x, w, None)}
            op = gemm_op(m, n, k)
            if op is not None:
                variants["hk"] = lambda w, op=op: op(x, w, out) or out
            dop = decode_op(m, n, k)
            if dop is not None:
                variants["hk_decode"] = lambda w, dop=dop: dop(x, w, out) or out
            if hkk is not None and hkk.gemm_fits(m, n, k):
                variants["hipkittens"] = lambda w: hkk.gemm_bf16(x, w, out=out)
            if fly and flydsl_gemm.variant(m, n, k) is not None:
                variants["flydsl"] = lambda w: flydsl_gemm.run(x, w, out) or out
            return variants

        def one_set(m, names):
            """{path: [ms per round]} on activations of their own."""
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ok = {p: c for p, c in variants_for(m, x, out).items() if p in names}
            per = {p: [] for p in ok}
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
                    per[name].append(s.elapsed_time(e) / copies)
            return per

        for m in grid:
            if sname == "lm_head" and m > MAX_SEQS:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            scale = ref.abs().max().item()
            tiles = {}
            if fly and flydsl_gemm.variant(m, n, k) is not None:
                tiles["flydsl"] = flydsl_gemm.variant(m, n, k)
            ok = []
            for name, call in variants_for(m, x, out).items():
                y = call(ws[0]).float()
                err = (y - ref).abs().max().item() / max(scale, 1e-30)
                if name != "engine" and not err < 2e-2:
                    print(f"  {sname} m={m}: {name} FAILED the gate "
                          f"(rel err {err:.3g}); not a candidate", flush=True)
                    continue
                ok.append(name)
            del x, out, ref
            for _ in range(args.warmup):
                one_set(m, ok)
            sets = [one_set(m, ok) for _ in range(args.sets)]
            times, noise = estimate(sets)
            if args.remeasure and close_call(times, noise, args.min_gain, args.opt_gain,
                                             args.close_k, args.noise_k):
                sets += [one_set(m, ok) for _ in range(args.remeasure)]
                times, noise = estimate(sets)
                remeasured += 1
            tiles = {p: v for p, v in tiles.items() if p in times}
            point = [m, {p: round(v, 5) for p, v in times.items()}, tiles,
                     {p: round(v, 4) for p, v in noise.items()}, len(sets)]
            if args.raw:
                point.append(sets)
            points.append(point)
        rs, above = runs(points, args.min_gain, args.noise_k)
        ent = dict(name=sname, runs=rs, above=above, points=points)
        conf = ({m for (key, m), (_, took) in confirmed.items() if key == f"{n}x{k}" and took}
                if confirmed is not None else None)
        pref = {name: prefer(points, name, args.min_gain, args.opt_gain, args.noise_k, conf)
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

    table["remeasured"] = remeasured
    Path(args.out).write_text(dump(table))
    print(f"wrote {args.out} ({remeasured} points re-measured as close calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
