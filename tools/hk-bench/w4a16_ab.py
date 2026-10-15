#!/usr/bin/env python3
"""hk's 4-bit decode GEMM against what vLLM runs for 4-bit weights on gfx1100.

At each of Qwen3-8B's linear shapes and each decode batch, with group 128:

  awq       ops.awq_gemm on an AutoAWQ checkpoint -- what vLLM's
            AutoAWQLinearMethod calls below 256 tokens (Triton on ROCm when
            VLLM_USE_TRITON_AWQ is set, its C++ kernel otherwise)
  rdna3     ops.gptq_gemm_rdna3 on the same weights as a GPTQ checkpoint,
            shuffled as RDNA3W4A16LinearKernel shuffles it -- vLLM's GPTQ path
  hk        hk.ops.w4a16.matmul on the weights repacked by repack_awq

Every arm is checked against the dequantized product before it is timed;
the weights rotate through copies larger than the Infinity Cache; arms
interleave, best of rounds.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/w4a16_ab.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096),
          "down": (4096, 12288)}
ROTATE_BYTES = 3 * 96 * 2 ** 20


def _pack_along(v, order, dim):
    import torch

    v = v.to(torch.int64).movedim(dim, -1)
    shape = v.shape
    v = v.reshape(*shape[:-1], shape[-1] // 8, 8)
    words = (v << torch.tensor([4 * o for o in order], device=v.device)).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.movedim(-1, dim).contiguous()


def checkpoint(n, k, group, g, dtype):
    """One random 4-bit W as an AWQ checkpoint, a shuffled GPTQ one, and hk's."""
    import torch
    from vllm import _custom_ops as ops

    from hk.ops import w4a16

    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.004 + 0.001)
    z = torch.full((k // group, n), 8, device="cuda", dtype=torch.int32)
    awq = (_pack_along(q, w4a16.AWQ_ORDER, 1), _pack_along(z, w4a16.AWQ_ORDER, 1),
           s.to(dtype))
    gq = _pack_along(q, range(8), 0)                    # [K/8, N], along K
    gz = _pack_along(z - 1, range(8), 1)                # GPTQ v1's zero, one short
    ops.gptq_shuffle(gq, 4)
    rdna3 = (gq, gz, s.to(dtype))
    hk = w4a16.repack_awq(*awq, group, dtype=dtype)
    return awq, rdna3, hk


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"),
                    help="activations and scales; vLLM's Triton AWQ kernel takes fp16 only")
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    from vllm import _custom_ops as ops

    from hk.ops import w4a16

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    group = 128
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for sname, (n, k) in SHAPES.items():
        copies = max(2, -(-ROTATE_BYTES // (n * k // 2)))
        ws = [checkpoint(n, k, group, g, dtype) for _ in range(copies)]
        ref_w = w4a16.dequant(*ws[0][2], group).float()
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=dtype, generator=g)
            want = x.float() @ ref_w.t()
            arms = {
                "awq": lambda c: ops.awq_gemm(x, c[0][0], c[0][2], c[0][1], 8),
                "rdna3": lambda c: ops.gptq_gemm_rdna3(x, c[1][0], c[1][1], c[1][2], False),
                "hk": lambda c: w4a16.matmul(x, *c[2], group),
            }
            if m <= 16:
                # One token tile: the schedules that read it once for several
                # 16-row blocks of W (slices, warps, blocks).
                out = torch.empty(m, n, device="cuda", dtype=dtype)
                for sl, wp, bl in ((4, 8, 2),):
                    if n % (16 * bl) or k % (16 * sl * wp):
                        continue
                    op = w4a16._op_for(1, sl, wp, group, bl, args.dtype)
                    arms[f"hk_b{bl}w{wp}"] = (
                        lambda c, op=op, out=out: op(x, *c[2], out) or out)
            ok = {}
            for name, fn in arms.items():
                try:
                    got = fn(ws[0]).float()
                except Exception as e:  # noqa: BLE001 -- an arm this build lacks
                    print(f"  {sname} m={m}: {name} raised {type(e).__name__}: {e}")
                    continue
                err = (got - want).abs().max().item() / want.abs().max().item()
                if not err < 3e-2:
                    print(f"  {sname} m={m}: {name} FAILED the check ({err:.3g})")
                    continue
                ok[name] = fn
            best = {a: float("inf") for a in ok}
            for r in range(args.rounds):
                for a in (list(ok) if r % 2 == 0 else list(ok)[::-1]):
                    ok[a](ws[-1])
                    torch.cuda.synchronize()
                    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                    st.record()
                    for c in ws:
                        ok[a](c)
                    en.record()
                    torch.cuda.synchronize()
                    best[a] = min(best[a], st.elapsed_time(en) / len(ws) * 1e3)
            row = dict(shape=sname, n=n, k=k, m=m,
                       **{f"{a}_us": round(v, 1) for a, v in best.items()})
            rows.append(row)
            gbs = n * k / 2 / best["hk"] / 1e3 if "hk" in best else 0
            cmp = "  ".join(f"{a} {v:7.1f}" for a, v in best.items())
            vs = "  ".join(f"hk/{a} {best[a] / best['hk']:.2f}x" for a in best
                           if a != "hk" and "hk" in best)
            print(f"{sname:8s} m={m:3d}: {cmp} us   {vs}   hk {gbs:.0f} GB/s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
