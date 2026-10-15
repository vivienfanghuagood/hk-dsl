#!/usr/bin/env python3
"""SGLang's fused-MoE layer, stock against hk's GEMM, in one process.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/sglang_moe_ab.py \\
        [--shapes olmoe,qwen3-30b-a3b] [--tokens 1,2,4,...] [--block16-max N] [--out f.json]

At each token count the whole layer runs: SGLang's `fused_experts_impl`, with
its sort, both GEMMs, the activation and the sum over the top-k, on random
bf16 weights of the model's shape and top-k ids from a random router. The two
arms differ only in what `sglang_moe.apply()` rebinds. The routing is the
same torch top-k in both, since sgl_kernel's fails on gfx1100. Each arm's
output is held to the other's before anything is timed. They interleave,
min over rounds, with the weights rotated through three copies. `host` is
the time to enqueue one layer: what an eager server pays per layer when the
CPU, not the GPU, sets the pace.

`--block16-max` moves the patch's 16-row ceiling for the run (default: the
module's); a large value measures hk at every token count, which is how the
ceiling was chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

SHAPES = {   # experts, hidden, intermediate, top_k
    "olmoe": (64, 2048, 1024, 8),
    "qwen3-30b-a3b": (128, 2048, 768, 8),
}
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="olmoe,qwen3-30b-a3b")
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--block16-max", type=int, default=0)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct",
                    help="what the runtime context is published for; the MoE "
                    "code reads its config bags, which a server publishes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from sglang.srt import runtime_context
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sglang.srt.server_args import ServerArgs

    from hk.integration import sglang_moe

    # What a server sets up before the first MoE call: its config bags and a
    # tensor-parallel group, here of one.
    runtime_context.publish(ServerArgs(model_path=args.model, attention_backend="triton",
                                       disable_cuda_graph=True), role="scheduler")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                 distributed_init_method="tcp://127.0.0.1:29571")
    initialize_model_parallel(tensor_model_parallel_size=1)

    if args.block16_max:
        sglang_moe.BLOCK16_MAX_TOKENS = args.block16_max
    dev = torch.device("cuda")
    results = {}
    for name in args.shapes.split(","):
        e, hidden, inter, top_k = SHAPES[name]
        g = torch.Generator(device=dev).manual_seed(0)
        layers = [((torch.randn(e, 2 * inter, hidden, device=dev, generator=g)
                    * hidden ** -0.5).to(torch.bfloat16),
                   (torch.randn(e, hidden, inter, device=dev, generator=g)
                    * inter ** -0.5).to(torch.bfloat16)) for _ in range(3)]
        rows = []
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, hidden, device=dev, generator=g).to(torch.bfloat16)
            logits = torch.randn(m, e, device=dev, generator=g)
            tw, ti = torch.softmax(logits, -1).topk(top_k, -1)
            tw, ti = tw.contiguous(), ti.to(torch.int32).contiguous()

            def run(w1, w2):
                return fused_experts_impl(x, w1, w2, tw, ti)

            arms = {}
            for arm in ("stock", "hk"):
                sglang_moe.revert() if arm == "stock" else sglang_moe.apply()
                before = dict(sglang_moe.STATS)
                arms[arm] = run(*layers[0]).float()
                took = {p: sglang_moe.STATS.get(p, 0) - before.get(p, 0)
                        for p in ("hk", "fallback")}
            ref = arms["stock"]
            err = (arms["hk"] - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
            if not err < 2e-2:
                print(f"{name} m={m}: hk FAILED against stock (rel err {err:.3g})", flush=True)
                rows.append(dict(tokens=m, err=err, failed=True))
                continue
            best = {"stock": float("inf"), "hk": float("inf")}
            host = {"stock": float("inf"), "hk": float("inf")}
            for r in range(args.rounds):
                for arm in (("stock", "hk") if r % 2 == 0 else ("hk", "stock")):
                    sglang_moe.revert() if arm == "stock" else sglang_moe.apply()
                    run(*layers[-1])
                    torch.cuda.synchronize()
                    s, t = torch.cuda.Event(True), torch.cuda.Event(True)
                    h0 = time.perf_counter()
                    s.record()
                    for w1, w2 in layers:
                        run(w1, w2)
                    t.record()
                    h1 = time.perf_counter()
                    torch.cuda.synchronize()
                    best[arm] = min(best[arm], s.elapsed_time(t) / len(layers))
                    host[arm] = min(host[arm], (h1 - h0) * 1e3 / len(layers))
            rows.append(dict(tokens=m, stock_ms=round(best["stock"], 4),
                             hk_ms=round(best["hk"], 4),
                             x=round(best["stock"] / best["hk"], 3), err=err, hk_calls=took,
                             stock_host_ms=round(host["stock"], 4),
                             hk_host_ms=round(host["hk"], 4)))
            print(f"{name:>14} m={m:>5}: stock {best['stock']:8.3f} ms  hk {best['hk']:8.3f} ms"
                  f"  {best['stock'] / best['hk']:5.2f}x  err {err:.2e}  "
                  f"host {host['stock'] * 1e3:5.0f} / {host['hk'] * 1e3:5.0f} us  {took}",
                  flush=True)
        results[name] = rows
    sglang_moe.revert()
    if args.out:
        Path(args.out).write_text(json.dumps(dict(
            torch=torch.__version__, hip=torch.version.hip,
            block16_max=sglang_moe.BLOCK16_MAX_TOKENS, shapes=results), indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
