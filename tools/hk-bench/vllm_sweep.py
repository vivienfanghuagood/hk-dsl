#!/usr/bin/env python3
"""Where does the HK attention backend win, and where does it lose?

The first version of this measured four short prompts end to end and reported
one ratio. That number answers nothing: it mixes prefill with decode, it is a
single batch size at a single context length, and total wall time is neither
latency nor throughput. This measures the thing attention actually costs in
serving, as a function of the two axes that move it.

**Method: the slope, not the intercept.** Each point runs the same requests
twice -- once with `K1` output tokens, once with `K2` -- and takes

    per-step decode latency = (T(K2) - T(K1)) / (K2 - K1)

which cancels prefill, model load, sampling setup and every other fixed cost,
leaving the cost of one decode step for that (batch, context). Throughput is
that latency divided into the batch size. A backend can win on one and lose on
the other -- a kernel that is fast at batch 1 and does not scale, or the
reverse -- and reporting a single ratio would hide exactly that.

Each timing is the min of `--reps` runs; the spread is reported too, because
on this chip a difference smaller than the spread is not a difference.

    python3 tools/hk-bench/vllm_sweep.py --backend hk --model Qwen/Qwen3-8B \\
        --batches 1,4,16,32 --context 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--backend", default="hk")
    ap.add_argument("--batches", default="1,4,16,32")
    ap.add_argument("--context", default="1024")
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=72)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    slot = os.environ.get("HK_VLLM_BACKEND_SLOT", "TRITON_ATTN")
    if args.backend == "hk":
        from hk.integration import vllm_backend

        print(vllm_backend.register(slot))
        backend = slot
    else:
        backend = args.backend

    import torch  # noqa: F401 -- torch first, as vLLM expects
    from vllm import LLM, SamplingParams

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.context.split(",")]
    max_len = max(contexts) + args.k2 + 64

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        block_size=32,
        max_model_len=max_len,
        max_num_seqs=max(batches),
        gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
        attention_backend=backend,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=os.environ.get("HK_EAGER", "1") == "1",
    )
    tok = llm.get_tokenizer()

    def prompts(n_req, n_ctx):
        # Distinct prompts of an identical token length: the same filler with
        # a different index in front, so no two requests share a prefix the
        # scheduler could collapse even with prefix caching off.
        out = []
        for i in range(n_req):
            ids = tok.encode(f"{i} " + "the quick brown fox " * (n_ctx // 4))
            out.append({"prompt_token_ids": ids[:n_ctx]})
        return out

    def run(reqs, k):
        p = SamplingParams(temperature=0.0, max_tokens=k, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate(reqs, p)
        return time.perf_counter() - t0

    rows = []
    for n_ctx in contexts:
        for n_req in batches:
            reqs = prompts(n_req, n_ctx)
            run(reqs, args.k1)                       # warm this shape
            t1 = [run(reqs, args.k1) for _ in range(args.reps)]
            t2 = [run(reqs, args.k2) for _ in range(args.reps)]
            dk = args.k2 - args.k1
            step_ms = (min(t2) - min(t1)) / dk * 1e3
            # A band, not a +-: the slope is a difference of two measured
            # times, so its uncertainty is the worst and best pairings of
            # their ranges. Quoting the min-min slope alone would hide that a
            # slope built from two noisy endpoints is noisier than either.
            lo = (min(t2) - max(t1)) / dk * 1e3
            hi = (max(t2) - min(t1)) / dk * 1e3
            rows.append({
                "context": n_ctx, "batch": n_req,
                "step_ms": step_ms,
                "tok_s": n_req / (step_ms * 1e-3) if step_ms > 0 else 0.0,
                "lo_ms": lo, "hi_ms": hi,
                "t1_min": min(t1), "t2_min": min(t2),
            })
            print(f"ctx {n_ctx:>6}  batch {n_req:>3}  "
                  f"{step_ms:8.3f} ms/step  "
                  f"{rows[-1]['tok_s']:9.1f} tok/s  "
                  f"[{lo:.3f}, {hi:.3f}]", flush=True)

    print(f"\nbackend {args.backend} ({backend}) on {args.model}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"backend": args.backend, "model": args.model, "rows": rows},
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
