#!/usr/bin/env python3
"""Where does a decode step actually spend its time?

The point of this IR is to write a kernel where an engine is missing a good
one -- so the first question is which ops those are, and the answer has to
come from the engine rather than from a guess. Attention was the guess, and
it was a bad one: vLLM's ROCm attention is a well-tuned Triton kernel, which
makes it the hardest target on the board rather than the most promising.

This runs a steady-state decode loop under the torch profiler and prints the
device kernels by total time, with the share of the step each one is. A
kernel that is 6% of the step and twice as fast as it needs to be is worth
more than one that is 25% and already at the bandwidth limit.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/vllm_profile.py \\
        --model Qwen/Qwen3-8B --batch 16 --context 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--top", type=int, default=28)
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", block_size=32,
              max_model_len=args.context + args.tokens + 64,
              max_num_seqs=args.batch,
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1")
    tok = llm.get_tokenizer()
    reqs = []
    for i in range(args.batch):
        ids = tok.encode(f"{i} " + "the quick brown fox " * (args.context // 4))
        reqs.append({"prompt_token_ids": ids[: args.context]})
    p = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)

    llm.generate(reqs, p)                                  # warm
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) \
            as prof:
        llm.generate(reqs, p)

    # The chrome trace rather than key_averages(). On this ROCm build
    # `self_device_time_total` comes back zero for every kernel -- the summary
    # path does not see the ROCTracer events -- while the exported trace has
    # them all, with a duration each. Parsing it is less elegant and it is the
    # one that works.
    import json
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "hk_trace.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        ev = json.load(f)["traceEvents"]

    tot = defaultdict(float)
    cnt = defaultdict(int)
    for e in ev:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            tot[e["name"]] += e.get("dur", 0)
            cnt[e["name"]] += 1
    grand = sum(tot.values()) or 1.0
    print(f"\n{args.model}  batch {args.batch}  ctx {args.context}  "
          f"{args.tokens} decode steps")
    print(f"total device time {grand / 1e3:.1f} ms over {len(tot)} kernels\n")
    print(f"{'%':>6}{'ms':>10}{'calls':>9}  kernel")
    for k in sorted(tot, key=tot.get, reverse=True)[: args.top]:
        print(f"{100 * tot[k] / grand:6.2f}{tot[k] / 1e3:10.2f}"
              f"{cnt[k]:9d}  {k[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
