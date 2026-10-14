#!/usr/bin/env python3
"""hk's MoE GEMM against vLLM's Triton fused_moe kernel, on vLLM's own sort.

For each model shape and token count: vLLM picks its config
(`try_get_optimal_moe_config`) and sorts the batch with that BLOCK_SIZE_M
(`moe_align_block_size`); both kernels then run both GEMMs of the layer on
the same lists -- w13 on the hidden states, w2 on the activated intermediate
with the routing weights applied -- checked against each other before they
are timed. Eight routings rotate, so the chosen experts' weights are not
left in the Infinity Cache from the call before. One process, the arms
interleaved round by round, best round reported.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/moe_ab.py --model olmoe
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))

#: experts, top_k, hidden, moe intermediate
MODELS = {
    "olmoe": (64, 8, 2048, 1024),          # allenai/OLMoE-1B-7B
    "qwen3-30b-a3b": (128, 8, 2048, 768),  # Qwen/Qwen3-30B-A3B
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="olmoe", choices=sorted(MODELS))
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,512,1024,4096")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sets", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.triton_utils import tl

    from hk.ops import moe

    experts, top_k, hidden, inter = MODELS[args.model]
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    w1 = torch.randn(experts, 2 * inter, hidden, device=dev, dtype=torch.bfloat16,
                     generator=g) * hidden ** -0.5
    w2 = torch.randn(experts, hidden, inter, device=dev, dtype=torch.bfloat16,
                     generator=g) * inter ** -0.5
    rows = []
    for m in (int(t) for t in args.tokens.split(",")):
        config = fm.try_get_optimal_moe_config(w1.shape, w2.shape, top_k,
                                               None, m)
        bm = config["BLOCK_SIZE_M"]
        x = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16, generator=g)
        h = torch.randn(m * top_k, inter, device=dev, dtype=torch.bfloat16,
                        generator=g)
        config16 = dict(config, BLOCK_SIZE_M=16)
        sets, sets16 = [], []
        for _ in range(args.sets):
            scores = torch.rand(m, experts, device=dev, generator=g)
            tw, ti = scores.topk(top_k, dim=1)
            tw = tw.softmax(dim=1).float()
            ti = ti.to(torch.int32)
            # vLLM's own choice: the naive one-pair-a-block assignment when
            # tokens * top_k * 4 <= experts, moe_align_block_size otherwise.
            sets.append((tw, *fm._prepare_expert_assignment(
                ti, config, m, top_k, experts, None, ignore_invalid_experts=True)))
            sets16.append((tw, *fm._prepare_expert_assignment(
                ti, config16, m, top_k, experts, None, ignore_invalid_experts=True)))
        c1 = torch.empty(m, top_k, 2 * inter, device=dev, dtype=torch.bfloat16)
        c2 = torch.empty(m, top_k, hidden, device=dev, dtype=torch.bfloat16)
        d1, d2 = torch.empty_like(c1), torch.empty_like(c2)
        t1 = torch.empty_like(c1)
        t2 = torch.empty_like(c2)

        def triton_layer(s, out1=t1, out2=t2):
            tw, sid, eid, ntp = s
            fm.invoke_fused_moe_triton_kernel(
                x, w1, out1, None, None, None, sid, eid, ntp, False, top_k,
                config, tl.bfloat16, False, False, False, False, False, None)
            fm.invoke_fused_moe_triton_kernel(
                h, w2, out2, None, None, tw, sid, eid, ntp, True, 1,
                config, tl.bfloat16, False, False, False, False, False, None)

        def hk_layer(s, out1=c1, out2=c2, block_m=bm):
            tw, sid, eid, ntp = s
            moe.moe_gemm(x, w1, out1, sid, eid, ntp, top_k=top_k, block_m=block_m)
            moe.moe_gemm(h, w2, out2, sid, eid, ntp, top_k=1, block_m=block_m,
                         topk_weights=tw)

        def hk16_layer(s):
            hk_layer(s, d1, d2, 16)

        # hk on vLLM's BLOCK_SIZE_M where it has a schedule for it; hk16 on
        # a sort redone at 16 rows, which is what the patch asks vLLM for.
        arms = {"triton": (triton_layer, sets), "hk16": (hk16_layer, sets16)}
        if moe.schedule_for(bm, hidden) and moe.schedule_for(bm, inter) and bm != 16:
            arms["hk"] = (hk_layer, sets)
        triton_layer(sets[0])
        checks = []
        if "hk" in arms:
            hk_layer(sets[0])
            checks += [("hk w13", c1, t1), ("hk w2", c2, t2)]
        hk16_layer(sets16[0])
        checks += [("hk16 w13", d1, t1), ("hk16 w2", d2, t2)]
        torch.cuda.synchronize()
        for name, got, want in checks:
            err = ((got.float() - want.float()).abs().max()
                   / want.float().abs().max()).item()
            if not err < 2e-2:
                raise SystemExit(f"M={m} {name}: differs from Triton by {err:.3g}")

        best = {a: float("inf") for a in arms}
        for r in range(args.rounds):
            for a in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                fn, its = arms[a]
                it = itertools.cycle(its)
                for _ in range(3):
                    fn(next(it))
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                n = 20
                st.record()
                for _ in range(n):
                    fn(next(it))
                en.record()
                torch.cuda.synchronize()
                best[a] = min(best[a], st.elapsed_time(en) / n * 1e3)
        row = dict(tokens=m, block_m=bm, naive=sets[0][1] is None,
                   **{f"{a}_us": round(v, 1) for a, v in best.items()})
        rows.append(row)
        rest = "   ".join(f"{a} {best[a]:8.1f} us {best['triton'] / best[a]:5.2f}x"
                          for a in best if a != "triton")
        print(f"M={m:5d} BLOCK_M={bm:3d}{' naive' if row['naive'] else ''}: "
              f"Triton {best['triton']:8.1f} us   {rest}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(model=args.model, rows=rows), indent=1))


if __name__ == "__main__":
    main()
