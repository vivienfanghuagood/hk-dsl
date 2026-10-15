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
import subprocess
import sys
import time
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parents[2])


def one_arm(args, backend: str) -> dict:
    """Run one arm in this process and return its timings.
    `gemm@NAME` routes the linear layers by the table `--routes` names NAME."""
    # The patches below go into this process. An engine core in a process
    # of its own would import vLLM afresh and run unpatched, timing like stock.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    backend, _, table = backend.partition("@")
    if table:
        routes = dict(kv.split("=", 1) for kv in args.routes.split(",") if kv)
        os.environ["HK_VLLM_ROUTES"] = routes[table]
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

    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    ctxs = [int(x) for x in args.context.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    # HK_VLLM_DEFAULTS=1: prefix caching and chunked prefill as vLLM ships
    # them. Otherwise both off, the setting every number before them used.
    defaults = os.environ.get("HK_VLLM_DEFAULTS", "0") == "1"
    knobs = {} if defaults else dict(enable_prefix_caching=False,
                                     enable_chunked_prefill=False)
    llm = LLM(model=args.model, dtype=args.dtype, block_size=32,
              max_model_len=max(ctxs) + args.tokens + 64,
              max_num_seqs=max(batches),
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              attention_backend=sel,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1", **knobs)
    tok = llm.get_tokenizer()
    out = {}
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
                **{k: min(v) for k, v in got.items()}, "all": got["min"],
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
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if not max(int(b) for b in args.batch.split(",")) <= args.short < args.tokens:
        ap.error("need max(batch) <= --short < --tokens for the decode slope "
                 "to cancel the prefill steps")
    if args.arm:                       # child: one engine, one backend
        res = one_arm(args, args.arm)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0

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
            files.setdefault(arm, []).append(f)
    if "stock" not in files:
        print("stock failed every round; nothing to compare against")
        return 1

    best, toks = {}, {}
    for arm, fs in files.items():
        for f in fs:
            for k, v in json.loads(Path(f).read_text()).items():
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
    if args.out:
        Path(args.out).write_text(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
