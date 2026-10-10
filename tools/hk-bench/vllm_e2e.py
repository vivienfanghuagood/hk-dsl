#!/usr/bin/env python3
"""End to end against the unmodified engine. Baseline 1.0x.

This is the number the IR exists to move: take a serving engine as it ships,
replace the kernels it has no good one for, and see whether the *request* got
faster. Not the kernel, not the decode step -- the request.

`vllm_sweep.py` deliberately cancels prefill to isolate a decode step, which
was the right tool for asking "is the paged kernel any good" and the wrong one
for asking "is the patch worth installing". A profile of this exact workload
(tools/hk-bench/vllm_profile.py, Qwen3-8B, batch 16, ctx 4096) says where the
time is:

    57.6%  hipBLASLt GEMM
    39.1%  Triton prefill attention
     2.3%  silu-and-mul, RMSNorm, RoPE, cache writes, everything else

-- so a patch that only touches decode attention is arguing over the last
percent, and the dense prefill kernel is sitting in front of 39%.

Arms alternate and every point reports the min of `--reps`, because two blocks
of runs on this chip are not comparable.

    python3 tools/hk-bench/vllm_e2e.py --context 4096 --batch 16 --tokens 72
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))


def one_arm(args, backend: str) -> dict:
    """Run one backend in this process and return its timings."""
    if backend == "hk":
        from hk.integration import vllm_backend

        slot = os.environ.get("HK_VLLM_BACKEND_SLOT", "TRITON_ATTN")
        print(vllm_backend.register(slot), flush=True)
        sel = slot
    elif backend == "stock":
        # None, not a name. The baseline is the engine as it ships, which
        # means letting the platform choose -- naming TRITON_ATTN here would
        # be a different experiment that happens to agree today.
        sel = None
    else:
        sel = backend

    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    ctxs = [int(x) for x in args.context.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    llm = LLM(model=args.model, dtype="bfloat16", block_size=32,
              max_model_len=max(ctxs) + args.tokens + 64,
              max_num_seqs=max(batches),
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              attention_backend=sel,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1")
    tok = llm.get_tokenizer()
    out = {}
    for n_ctx in ctxs:
        for n_req in batches:
            reqs = [{"prompt_token_ids":
                     tok.encode(f"{i} " + "the quick brown fox " *
                                (n_ctx // 4))[:n_ctx]}
                    for i in range(n_req)]
            p = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                               ignore_eos=True)
            llm.generate(reqs, p)                       # warm this shape
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                llm.generate(reqs, p)
                ts.append(time.perf_counter() - t0)
            out[f"{n_ctx}x{n_req}"] = {"min": min(ts), "all": ts}
            print(f"  ctx {n_ctx} batch {n_req}: {min(ts) * 1e3:9.1f} ms",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--context", default="1024,4096")
    ap.add_argument("--batch", default="1,4,16")
    ap.add_argument("--tokens", type=int, default=72)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm:                       # child: one engine, one backend
        res = one_arm(args, args.arm)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0

    # Parent: two children, alternating, because two backends cannot share one
    # engine and a backend chosen at construction cannot be swapped.
    base = [sys.executable, str(HERE), "--model", args.model,
            "--context", args.context, "--batch", args.batch,
            "--tokens", str(args.tokens), "--reps", str(args.reps)]
    files = {}
    for rnd in range(args.reps):
        for arm in (["stock", "hk"] if rnd % 2 == 0 else ["hk", "stock"]):
            f = f"/tmp/hk_e2e_{arm}_{rnd}.json"
            print(f"--- {arm} round {rnd}", flush=True)
            subprocess.run(base + ["--arm", arm, "--out", f], check=True)
            files.setdefault(arm, []).append(f)

    best = {}
    for arm, fs in files.items():
        for f in fs:
            for k, v in json.loads(Path(f).read_text()).items():
                best.setdefault(arm, {})
                best[arm][k] = min(best[arm].get(k, 1e9), v["min"])
    print(f"\n{args.model}, {args.tokens} output tokens, "
          f"baseline = stock vLLM = 1.00x\n")
    print(f"{'shape':>12}{'stock ms':>12}{'hk ms':>12}{'speedup':>10}")
    for k in sorted(best.get("stock", {}),
                    key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1]))):
        a, b = best["stock"][k], best["hk"][k]
        print(f"{k:>12}{a * 1e3:12.1f}{b * 1e3:12.1f}{a / b:9.3f}x")
    if args.out:
        Path(args.out).write_text(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
