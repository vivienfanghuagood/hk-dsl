#!/usr/bin/env python3
"""hk's 4-bit decode GEMM against vLLM's RDNA3 GPTQ kernel, on GPTQ checkpoints.

For each (N, K) and M: a random symmetric GPTQ v1 checkpoint (group 32 by
default), repacked into hk's layout (`hk.ops.w4a16.repack_gptq`) and shuffled
the way vLLM's `RDNA3W4A16LinearKernel` shuffles it (`ops.gptq_gemm_rdna3`).
Both results are checked against the fp32 product of the dequantized weights,
and then timed. The weight copies rotate past the Infinity Cache, the arms are
interleaved in one process, and the best of 5 rounds is reported. This is
where `vllm_gptq.MIN_M` comes from. It needs a vLLM with `gptq_gemm_rdna3`
(gfx1100).

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/gptq_w4a16.py \\
        --shapes 14336x5120,5120x6144,16384x5120,34816x5120,5120x17408   # Qwen3.8-27B, TP1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def checkpoint(n, k, group, dev):
    """A random symmetric GPTQ v1 checkpoint and the fp32 W [N, K] it means."""
    import torch

    q = torch.randint(0, 16, (k, n), device=dev, dtype=torch.int32)
    qweight = torch.zeros(k // 8, n, device=dev, dtype=torch.int32)
    for t in range(8):
        qweight |= (q[t::8] & 0xF) << (4 * t)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device=dev, dtype=torch.int32)
    scales = (torch.rand(k // group, n, device=dev) * 0.02 + 0.005).half()
    w = ((q.float() - 8) * scales.float().repeat_interleave(group, 0)).t().contiguous()
    return qweight, qzeros, scales, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="14336x5120,5120x6144,16384x5120,34816x5120,5120x17408",
                    help="NxK, comma-separated")
    ap.add_argument("--m", default="1,2,4,8,16,32,48,64")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--rotate-mb", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    import torch
    from vllm import _custom_ops as ops

    from hk.ops import w4a16 as W

    torch.manual_seed(0)
    dev, g = "cuda", args.group
    for nk in args.shapes.split(","):
        n, k = map(int, nk.split("x"))
        copies = max(2, -(-(args.rotate_mb << 20) // (n * k // 2)))
        ck = [checkpoint(n, k, g, dev) for _ in range(copies)]
        hk = [W.repack_gptq(qw, qz, s, g, dtype=torch.float16) for qw, qz, s, _ in ck]
        vl = []
        for qw, qz, s, _ in ck:
            w = qw.clone()
            ops.gptq_shuffle(w, 4)
            vl.append((w, qz, s))
        for m in map(int, args.m.split(",")):
            x = (torch.randn(m, k, device=dev) * 0.5).half()
            ref = x.float() @ ck[0][3].t()
            arms = {"hk": lambda i, x=x: W.matmul(x, *hk[i], g),
                    "vllm": lambda i, x=x: ops.gptq_gemm_rdna3(x, *vl[i], False)}
            err = {a: ((f(0).float() - ref).abs().max() / ref.abs().max()).item()
                   for a, f in arms.items()}
            if max(err.values()) > 2e-2:
                raise SystemExit(f"{nk} M={m}: relative error {err}")
            best = {a: float("inf") for a in arms}
            for r in range(args.rounds):
                for a in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                    f = arms[a]
                    for i in range(copies):
                        f(i)
                    torch.cuda.synchronize()
                    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                    st.record()
                    for _ in range(4):
                        for i in range(copies):
                            f(i)
                    en.record()
                    torch.cuda.synchronize()
                    best[a] = min(best[a], st.elapsed_time(en) / (4 * copies) * 1e3)
            gbs = n * k / 2 / (best["hk"] * 1e-6) / 1e9
            print(f"{nk} M={m:3d} hk kernel={W.plan(m, n, k, g, torch.float16) is not None}: "
                  f"hk {best['hk']:7.1f} us ({gbs:4.0f} GB/s of codes)  vllm {best['vllm']:7.1f} us  "
                  f"vllm/hk {best['vllm'] / best['hk']:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
