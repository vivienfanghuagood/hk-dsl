"""End to end against the unmodified engine, in one command. Baseline 1.0x.

Take a serving engine as it ships, replace the kernels it has no good one
for, and see whether the *request* got faster -- not the kernel, not the
decode step. Each arm is a fresh vLLM in a child process; arms alternate,
round by round, and every point reports the best of `--reps`, because two
blocks of runs on this chip are not comparable. The first request's greedy
tokens are compared against stock's, so a patch that changes the arithmetic
beyond rounding shows up as a different continuation.

    python -m hk.integration ab --model Qwen/Qwen3-8B --arms hk,hk+gemm
    HK_VLLM_USER=my_kernels.py python -m hk.integration ab --arms user
    python -m hk.integration ab ... --out raw.json --json verdicts.json
    python -m hk.integration ab --from raw.json --json verdicts.json

`--json` writes one row per shape, metric and arm, `{shape, metric, arm, ms,
stock_ms, ratio, band, verdict, ...}`, the verdict `win`, `loss`, `tie` or
`noise` against a band measured from the rounds themselves (`judge`): a gap
inside max(2%, the noise) is not a result, and a band too wide to call a
tie is called noise.

Arms: `stock` (always run), `hk` (the attention backend), `gemm` (linear
layers), `moe` (MoE GEMMs), `int4` (AWQ layers), `user` (the modules
HK_VLLM_USER names, see `hk.integration.register`), joined with `+`; or a
vLLM attention backend's name. `HK_EAGER=0` runs with vLLM's CUDA graphs and
torch.compile, as it ships; `HK_VLLM_DEFAULTS=1` keeps prefix caching and
chunked prefill on, as it ships.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PKG_ROOT = str(Path(__file__).resolve().parents[2])


def apply_arm(backend: str, routes: str = ""):
    """Patch this process for one arm: (the attention backend to ask vLLM
    for, or None for its own choice; {name: Patch} of what was applied; the
    set of patch names wanted). `gemm@NAME` routes the linear layers by the
    table `routes` ("NAME=path,...") names NAME."""
    # The patches below go into this process. An engine core in a process
    # of its own would import vLLM afresh and run unpatched, timing like stock.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # hk is on by default: an installed hk-dsl's vLLM plugin would apply its default routes
    # to every arm, stock included. The arm alone decides what is patched here.
    os.environ["HK_DISABLE"] = "1"
    backend, _, table = backend.partition("@")
    if table:
        named = dict(kv.split("=", 1) for kv in routes.split(",") if kv)
        os.environ["HK_VLLM_ROUTES"] = named[table]
    attn, *extras = backend.split("+")
    want = {p for p in ("gemm", "moe", "int4", "user") if attn == p or p in extras}
    if attn == "hk":
        from . import vllm_backend

        slot = os.environ.get("HK_VLLM_BACKEND_SLOT", "TRITON_ATTN")
        print(vllm_backend.register(slot), flush=True)
        sel = slot
    elif attn in ("stock", "gemm", "moe", "int4", "user"):
        # None, not a name. The baseline is the engine as it ships, which
        # means letting the platform choose -- naming TRITON_ATTN here would
        # be a different experiment that happens to agree today.
        sel = None
    else:
        sel = attn
    patches = {}
    if "gemm" in want:
        from . import vllm_gemm

        print(vllm_gemm.apply(), flush=True)
        patches["gemm"] = vllm_gemm.PATCH
    if "moe" in want:
        from . import vllm_moe

        print(vllm_moe.apply(), flush=True)
        patches["moe"] = vllm_moe.PATCH
    if "int4" in want:
        from . import vllm_int4

        print(vllm_int4.apply_patch(), flush=True)
        patches["int4"] = vllm_int4.PATCH
    if "user" in want:
        from . import register

        print("user modules:", register.load_user(os.environ.get("HK_VLLM_USER", "")),
              flush=True)
    return sel, patches, want


def one_arm(args, backend: str) -> dict:
    """Run one arm in this process and return its timings."""
    sel, patches, want = apply_arm(backend, args.routes)

    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    ctxs = [int(x) for x in args.context.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    # HK_VLLM_DEFAULTS=1: prefix caching and chunked prefill as vLLM ships
    # them. Otherwise both off, the setting every number before them used.
    defaults = os.environ.get("HK_VLLM_DEFAULTS", "0") == "1"
    knobs = {} if defaults else dict(enable_prefix_caching=False,
                                     enable_chunked_prefill=False)
    t_start = time.perf_counter()
    llm = LLM(model=args.model, dtype=args.dtype, block_size=32,
              max_model_len=max(ctxs) + args.tokens + 64,
              max_num_seqs=max(batches),
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              attention_backend=sel,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1", **knobs)
    out = {"_startup": {"engine_s": round(time.perf_counter() - t_start, 1)}}
    if "gemm" in want:
        from . import vllm_autotune

        out["_startup"]["routes"] = dict(vllm_autotune.REPORT)
    print(f"  engine built in {out['_startup']['engine_s']:.1f} s"
          + (f"; linear routes at load: "
             f"{vllm_autotune.REPORT.get('line', 'nothing measured (no hook installed)')}, "
             f"{vllm_autotune.REPORT.get('seconds', 0.0):.1f} s" if "gemm" in want else ""),
          flush=True)
    tok = llm.get_tokenizer()
    salt = iter(range(1 << 30))
    for n_ctx in ctxs:
        for n_req in batches:
            def prompts():
                # A fresh first token per generate: with prefix caching on, a
                # repeated prompt would be a cache hit and the timing would
                # stop measuring attention.
                s = next(salt)
                return [{"prompt_token_ids":
                         tok.encode(f"{s}.{i} " + "the quick brown fox " *
                                    (n_ctx // 4))[:n_ctx]}
                        for i in range(n_req)]
            reqs = prompts()

            def params(n):
                return SamplingParams(temperature=0.0, max_tokens=n,
                                      ignore_eos=True)

            # Three lengths. One output token is the batch's prefill. The
            # difference between `tokens` and `short` is steady-state decode
            # at the full batch: with `short` >= the batch, both runs admit
            # every prompt identically, and the extra steps are decode.
            lengths = {"min": args.tokens, "short": args.short, "prefill": 1}
            warm = llm.generate(reqs, params(args.tokens))  # warm this shape
            got = {k: [] for k in lengths}
            for _ in range(args.reps):
                for k, n in lengths.items():
                    batch = prompts() if defaults else reqs
                    t0 = time.perf_counter()
                    llm.generate(batch, params(n))
                    got[k].append(time.perf_counter() - t0)
            out[f"{n_ctx}x{n_req}"] = {
                **{k: min(v) for k, v in got.items()}, "all": got["min"], "reps": got,
                "tokens": list(warm[0].outputs[0].token_ids)}
            print(f"  ctx {n_ctx} batch {n_req}: {min(got['min']) * 1e3:9.1f}"
                  f" ms, {args.short} tokens {min(got['short']) * 1e3:9.1f} "
                  f"ms, prefill {min(got['prefill']) * 1e3:8.1f} ms",
                  flush=True)
    # A patch that silently routed nothing would time exactly like stock.
    for name, p in patches.items():
        print(f"  {name} routing: {p.summary()}", flush=True)
        out[f"_{name}_stats"] = dict(p.stats)
        out[f"_{name}_captured"] = dict(p.captured)
    if "user" in want:
        from . import register

        for line in register.summary():
            print(f"  user routing: {line}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hk.integration ab",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16",
                    help="the engine's dtype; float16 for an AWQ model on ROCm")
    ap.add_argument("--context", default="1024,4096")
    ap.add_argument("--batch", default="1,4,16")
    ap.add_argument("--tokens", type=int, default=72)
    ap.add_argument("--short", type=int, default=24,
                    help="the shorter run the decode slope is taken against; "
                         "must be at least the largest batch")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arm", default="", help=argparse.SUPPRESS)
    ap.add_argument("--arms", default="stock,hk",
                    help="comma-separated: stock, hk, gemm, moe, int4, user, "
                         "joined with +, or a vLLM backend name; stock is "
                         "always included. gemm@NAME routes by the table "
                         "--routes names NAME")
    ap.add_argument("--routes", default="",
                    help="NAME=routes.json,... for gemm@NAME arms")
    ap.add_argument("--out", default="", help="every round's timings, raw")
    ap.add_argument("--json", default="",
                    help="one row per shape, metric and arm: "
                         "{shape, metric, arm, ms, ratio, band, verdict, ...}")
    ap.add_argument("--from", dest="from_", default="",
                    help="judge an earlier --out file instead of running")
    args = ap.parse_args(argv)

    if args.from_:
        return judge_file(Path(args.from_), args)
    if not max(int(b) for b in args.batch.split(",")) <= args.short < args.tokens:
        ap.error("need max(batch) <= --short < --tokens for the decode slope "
                 "to cancel the prefill steps")
    if args.arm:                       # child: one engine, one backend
        res = one_arm(args, args.arm)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0

    # The stack, before an hour of engines: stock and every Triton arm move
    # with the Triton, so a run on another one measures another baseline
    # (hk.integration.stack). The children run under this same Python.
    from . import stack

    try:
        print(stack.enforce("hk.integration ab", default="refuse"), flush=True)
    except (stack.StackMismatch, ValueError) as e:
        print(e, file=sys.stderr, flush=True)
        return 2

    # Parent: one child per arm and round, because two backends cannot share
    # one engine and a backend chosen at construction cannot be swapped.
    base = [sys.executable, "-m", "hk.integration.ab", "--model", args.model,
            "--dtype", args.dtype, "--context", args.context, "--batch", args.batch,
            "--tokens", str(args.tokens), "--short", str(args.short),
            "--reps", str(args.reps), "--routes", args.routes]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_PKG_ROOT, env.get("PYTHONPATH", "")) if p)
    arms = ["stock"] + [a for a in args.arms.split(",") if a and a != "stock"]
    files, failed = {}, {}
    for rnd in range(args.reps):
        # Rotate, so no arm always runs first or always follows the same one.
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
        for arm in order:
            f = f"/tmp/hk_e2e_{arm}_{rnd}.json"
            print(f"--- {arm} round {rnd}", flush=True)
            # An arm that dies -- a backend that cannot be captured, say --
            # is a result too; the others still get measured.
            r = subprocess.run(base + ["--arm", arm, "--out", f], env=env)
            if r.returncode:
                failed.setdefault(arm, []).append(r.returncode)
                print(f"--- {arm} round {rnd} FAILED, exit {r.returncode}",
                      flush=True)
                continue
            files.setdefault(arm, []).append((rnd, f))
    if "stock" not in files:
        print("stock failed every round; nothing to compare against")
        return 1

    best, toks, starts, rounds = {}, {}, {}, {}
    for arm, fs in files.items():
        for rnd, f in fs:
            got = json.loads(Path(f).read_text())
            rounds.setdefault(arm, []).append(
                {"round": rnd, **{k: {m: v[m] for m in ("min", "short", "prefill", "reps")
                                      if m in v}
                                  for k, v in got.items() if not k.startswith("_")}})
            for k, v in got.items():
                if k == "_startup":
                    starts.setdefault(arm, []).append(v)
                if k.startswith("_"):
                    continue
                b = best.setdefault(arm, {}).setdefault(
                    k, {"total": 1e9, "short": 1e9, "prefill": 1e9})
                b["total"] = min(b["total"], v["min"])
                b["short"] = min(b["short"], v["short"])
                b["prefill"] = min(b["prefill"], v["prefill"])
                toks.setdefault(arm, {}).setdefault(k, v["tokens"])
    for arm in best:
        for b in best[arm].values():
            b["decode"] = (b["total"] - b["short"]) / (args.tokens - args.short)

    print(f"\n{args.model}, {args.tokens} output tokens, enforce_eager="
          f"{os.environ.get('HK_EAGER', '1') == '1'}, prefix caching and "
          f"chunked prefill "
          f"{'on' if os.environ.get('HK_VLLM_DEFAULTS') == '1' else 'off'}, "
          f"baseline = stock vLLM "
          f"= 1.00x.\ntotal = the whole batch, prompt to last token; prefill "
          f"= the same batch generating one token; decode = (T({args.tokens})"
          f" - T({args.short})) / {args.tokens - args.short}, one step at the "
          f"full batch.")
    for arm, codes in failed.items():
        print(f"{arm}: {len(codes)} of {args.reps} rounds failed "
              f"(exit {', '.join(map(str, codes))})")
    shapes = sorted(best.get("stock", {}),
                    key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1])))
    others = [a for a in arms[1:] if a in best]
    for what, unit in (("total", "ms"), ("prefill", "ms"),
                       ("decode", "ms/tok")):
        print(f"\n{what} ({unit})")
        print(f"{'ctx x batch':>12}{'stock':>10}"
              + "".join(f"{a:>12}{'':>8}" for a in others))
        for k in shapes:
            a = best["stock"][k][what]
            row = f"{k:>12}{a * 1e3:10.2f}"
            for arm in others:
                b = best[arm][k][what]
                row += f"{b * 1e3:12.2f}{a / b:7.3f}x"
            print(row)
    print("\nleading output tokens identical to stock (first request, greedy)")
    for k in shapes:
        ref = toks["stock"][k]
        same = []
        for arm in others:
            got = toks[arm][k]
            n = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                     min(len(ref), len(got)))
            same.append(f"{arm} {n}/{len(ref)}")
        print(f"{k:>12}  " + ", ".join(same))
    if starts:
        print("\nengine start (s), round by round; of it, measuring linear routes at load")
        for arm in arms:
            if arm not in starts:
                continue
            row = "  ".join(
                f"{s['engine_s']:7.1f}" + (f" ({s['routes'].get('seconds', 0.0):.1f})"
                                           if "routes" in s else "")
                for s in starts[arm])
            print(f"{arm:>12}  {row}")
        for arm in arms:
            for i, s in enumerate(starts.get(arm, [])):
                if s.get("routes", {}).get("line"):
                    print(f"  {arm} round {i}: {s['routes']['line']}")
    meta = {"model": args.model, "tokens": args.tokens, "short": args.short,
            "reps": args.reps, "arms": arms,
            "eager": os.environ.get("HK_EAGER", "1") == "1",
            "vllm_defaults": os.environ.get("HK_VLLM_DEFAULTS") == "1",
            "failed": failed, "date": time.strftime("%Y-%m-%d %H:%M"),
            "stack": {k: v for k, v in stack.installed().items() if k != "python"}}
    if args.out:
        Path(args.out).write_text(json.dumps(
            {**best, "_startup": starts, "_rounds": rounds, "_tokens": toks,
             "_meta": meta}, indent=2))
    rows = verdicts(rounds, args.tokens, args.short, toks)
    print_verdicts(rows)
    if args.json:
        write_verdicts(Path(args.json), rows, meta)
    return 0


# ------------------------------------------------------------------ verdicts
#
# A ratio is a verdict only with its noise beside it: this project has twice
# taken a difference inside the noise for a finding (chapter 32, and the
# attention tuning whose two sweeps crowned contradictory winners at 0.1% and
# 2.8%). So a row says win or loss only past its band, and tie or noise are
# answers, not gaps in the table.

#: No band is narrower than this, however quiet the rounds: the scoreboard's
#: floor (tools/hk-bench/scoreboard.py BAND).
BAND = 0.02
#: A band wider than this says nothing: the performance gate's cap
#: (tools/hk-bench/check_regression.py CAP).
CAP = 0.15
#: A ratio inside a band wider than this is noise, not a tie: such a band
#: cannot tell a tie from the 2-5% gaps this project's kernels win or lose by.
TIE_MAX = 0.05
METRICS = ("total", "prefill", "decode")
VERDICTS = ("win", "loss", "tie", "noise")
#: Every row has these keys, in this order.
ROW_KEYS = ("shape", "metric", "arm", "ms", "stock_ms", "ratio", "band", "verdict",
            "why", "rounds", "noise", "tokens_same")
SCHEMA = "hk.integration.ab/verdicts-1"


def _value(v: dict, metric: str, tokens: int, short: int) -> Optional[float]:
    """A round's time for `metric`, in seconds: the best of its reps."""
    if metric == "total":
        return v["min"]
    if metric == "prefill":
        return v["prefill"]
    return (v["min"] - v["short"]) / (tokens - short)


def _rep_spread(v: dict, metric: str, tokens: int, short: int) -> float:
    """The median absolute deviation of a round's reps over their median, as
    the scoreboard's `spread`; 0 for a round that kept no reps."""
    reps = v.get("reps") or {}
    if metric == "decode":
        xs = [(a - b) / (tokens - short) for a, b in zip(reps.get("min", []),
                                                          reps.get("short", []))]
    else:
        xs = reps.get("min" if metric == "total" else "prefill", [])
    xs = [x for x in xs if x > 0]
    if len(xs) < 2:
        return 0.0
    med = statistics.median(xs)
    return statistics.median(abs(x / med - 1) for x in xs)


def judge(ratios: Sequence[float], within: float) -> Tuple[float, float, str, str, float]:
    """(ratio, band, verdict, why, across) from one arm's per-round paired
    ratios (stock / arm, above 1 the arm is faster) and its within-round noise.

    The noise is check_regression.py's: the larger of the ratio's spread
    across rounds (max |r_i / median - 1|) and within one (the two arms' rep
    spreads, summed, at their worst round). The band is max(BAND, noise).
      win / loss  the ratio past the band, the band at most CAP;
      tie    the ratio within a band of at most TIE_MAX: no difference this
             measurement could resolve;
      noise  one round (the spread across rounds is not measured), a band
             past CAP, or the ratio within a band wider than TIE_MAX."""
    rs = sorted(ratios)
    ratio = statistics.median(rs)
    across = max(abs(r / ratio - 1) for r in rs) if len(rs) > 1 else 0.0
    band = max(BAND, across, within)
    if len(rs) < 2:
        return ratio, band, "noise", "one round: the spread across rounds is not measured", across
    if band > CAP:
        return ratio, band, "noise", f"band {band:.1%} is past the cap {CAP:.0%}", across
    if abs(ratio - 1) > band:
        kind = "win" if ratio > 1 else "loss"
        return ratio, band, kind, f"{ratio - 1:+.1%}, past the band", across
    if band > TIE_MAX:
        return (ratio, band, "noise", f"{ratio - 1:+.1%} inside a band of {band:.1%}, too "
                f"wide to call a tie", across)
    return ratio, band, "tie", f"{ratio - 1:+.1%}, within the band", across


def _same(ref: Optional[list], got: Optional[list]) -> Optional[str]:
    if not ref or got is None:
        return None
    n = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
    return f"{n}/{len(ref)}"


def verdicts(rounds: Dict[str, list], tokens: int, short: int,
             toks: Optional[dict] = None) -> List[dict]:
    """One row per (shape, metric, arm other than stock), from each arm's
    rounds (`--out`'s `_rounds`: per round, per shape, the best rep of each
    length and the reps). Rounds pair by their number, which is when they
    ran: arms rotate within a round, so a pair shares the node's state."""
    def by_round(arm):
        return {r.get("round", i): r for i, r in enumerate(rounds.get(arm, []))}

    stock = by_round("stock")
    shapes = sorted({k for r in stock.values() for k in r if k != "round"},
                    key=lambda x: tuple(int(v) for v in x.split("x")))
    rows = []
    for arm in [a for a in rounds if a != "stock"]:
        mine = by_round(arm)
        for shape in shapes:
            for metric in METRICS:
                pairs = [(stock[n][shape], mine[n][shape]) for n in sorted(stock)
                         if n in mine and shape in stock[n] and shape in mine[n]]
                vals = [(_value(s, metric, tokens, short), _value(a, metric, tokens, short),
                         _rep_spread(s, metric, tokens, short)
                         + _rep_spread(a, metric, tokens, short)) for s, a in pairs]
                vals = [v for v in vals if v[0] and v[1] and v[0] > 0 and v[1] > 0]
                if not vals:
                    continue
                within = max(w for _, _, w in vals)
                ratio, band, verdict, why, across = judge([s / a for s, a, _ in vals], within)
                rows.append(dict(zip(ROW_KEYS, (
                    shape, metric, arm,
                    round(statistics.median(a for _, a, _ in vals) * 1e3, 4),
                    round(statistics.median(s for s, _, _ in vals) * 1e3, 4),
                    round(ratio, 4), round(band, 4), verdict, why, len(vals),
                    {"across": round(across, 4), "within": round(within, 4)},
                    _same((toks or {}).get("stock", {}).get(shape),
                          (toks or {}).get(arm, {}).get(shape))))))
    return rows


def print_verdicts(rows: List[dict]) -> None:
    if not rows:
        return
    print(f"\nverdicts: the median over rounds; ratio = stock / arm per round, paired, "
          f"median; band = max({BAND:.0%}, the noise across and within rounds)")
    print(f"{'ctx x batch':>12} {'metric':<8}{'arm':<14}{'stock ms':>11}{'arm ms':>11}"
          f"{'ratio':>8}{'band':>7}  verdict")
    for r in rows:
        print(f"{r['shape']:>12} {r['metric']:<8}{r['arm']:<14}{r['stock_ms']:>11.2f}"
              f"{r['ms']:>11.2f}{r['ratio']:>7.3f}x{r['band']:>6.1%}  {r['verdict']:<5} "
              f"({r['why']}; {r['rounds']} round{'s' * (r['rounds'] != 1)})")


def write_verdicts(path: Path, rows: List[dict], meta: dict) -> None:
    path.write_text(json.dumps({
        "schema": SCHEMA, "meta": meta,
        "rule": (f"ms: the arm's median over rounds (each round the best of its reps), "
                 f"decode as the slope (T(tokens) - T(short)) / (tokens - short); ratio: "
                 f"the median over rounds of stock / arm, rounds paired by number; band: "
                 f"max({BAND}, noise), noise the larger of the ratio's spread across "
                 f"rounds and the two arms' summed rep spread within one; verdict: "
                 f"win | loss past the band; tie within a band of at most {TIE_MAX}; "
                 f"noise for one round, a band past {CAP}, or a ratio within a band "
                 f"wider than {TIE_MAX}"),
        "rows": rows}, indent=1) + "\n")
    print(f"wrote {path}")


def judge_file(path: Path, args) -> int:
    """`--from`: the verdicts of an earlier run's `--out`."""
    d = json.loads(path.read_text())
    meta = d.get("_meta") or {"model": args.model, "tokens": args.tokens,
                              "short": args.short, "from": str(path)}
    rows = verdicts(d.get("_rounds", {}), meta.get("tokens", args.tokens),
                    meta.get("short", args.short), d.get("_tokens"))
    print_verdicts(rows)
    if args.json:
        write_verdicts(Path(args.json), rows, {**meta, "from": str(path)})
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
