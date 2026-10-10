#!/usr/bin/env python3
"""Run a model on the HK attention backend, against vLLM's own.

The SDPA drop-in cannot reach a decoder -- vLLM V1's KV is paged and dispatch
goes through AttentionImpl, with no dense tensor on the path. So this runs the
other integration: `hk.integration.vllm_backend`, registered through vLLM's
own `register_backend` hook. On ROCm it has to override an existing enum
member -- the platform's allowlist rejects `CUSTOM` outright -- so `--backend
hk` registers over `TRITON_ATTN` and then selects it.

Two runs, one per backend, compared on the generated text. Greedy decoding, so
a backend that is merely *close* shows up as a different sentence rather than
as a tolerance; that is the point of comparing text here and numbers in
tests/hk/gpu/test_paged.py.

The flags are a real restriction and the script sets them deliberately: this
backend declares its own head-major KV layout, so it cannot hand a prefill to
vLLM's Triton kernel, and prefill is served by the dense kernel over the new
tokens. Prefix caching and chunked prefill both break that assumption and the
backend refuses them rather than attending over part of a context.

    python3 tools/hk-bench/vllm_decode.py --backend hk
    python3 tools/hk-bench/vllm_decode.py --backend TRITON_ATTN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

PROMPTS = [
    "The capital of France is",
    "In one sentence, what is a compiler?",
    "List three prime numbers:",
    "Water boils at",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", default="hk",
                    help="hk, or a vLLM backend name to compare against")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    hk_slot = os.environ.get("HK_VLLM_BACKEND_SLOT", "TRITON_ATTN")
    if args.backend == "hk":
        from hk.integration import vllm_backend

        print(vllm_backend.register(hk_slot))
        args.backend = hk_slot

    import torch  # noqa: F401 -- torch first, as vLLM expects
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        block_size=32,
        max_model_len=2048,
        gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.80")),
        attention_backend=args.backend,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=os.environ.get("HK_EAGER", "1") == "1",
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm.generate(PROMPTS, params)                     # warm
    best = None
    for _ in range(args.reps):
        t0 = time.perf_counter()
        outs = llm.generate(PROMPTS, params)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    texts = [o.outputs[0].text for o in outs]

    print(f"\nbackend {args.backend}  {best * 1e3:.1f} ms for "
          f"{len(PROMPTS)} x {args.max_tokens} tokens")
    for p, t in zip(PROMPTS, texts):
        print(f"  {p!r}\n    -> {t!r}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"backend": args.backend, "ms": best * 1e3, "texts": texts},
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
