#!/usr/bin/env python3
"""How accurate is the generated attention kernel, against what, and is the
difference a bug?

This exists because of a result from a real vLLM: with the kernel patched in,
InternVL2-2B described a test image differently from unpatched -- while the
unpatched run was reproducible, and vLLM's *other* vision backend (Triton
flash attention) agreed with the unpatched text exactly. Two independent
implementations agreeing and ours disagreeing is the shape of a bug, so it had
to be chased rather than written off as bf16 noise.

Three questions, three sections, in the order they rule things out:

  --tail      Does the error track `N % KV_BLOCK`? The two ViT shapes that
              changed the caption (577 and 1025 tokens) are both one past a
              multiple of the 32-wide KV block, and a tail block counted twice
              in the softmax denominator would be a real correctness bug.

  --backend   Is torch's SDPA here a flash kernel or its exact math fallback?
              Losing to an exact reference by 2x and losing to another flash
              kernel by 2x are different statements.

  --shape     Is the error concentrated or spread? A mishandled block or a
              missed rescale puts a few elements far out and leaves the rest
              at the noise floor. Rounding scales every quantile together.

Answers, on a W7900D at ROCm 7.2.4 (bf16, 16 heads, head_dim 64):

  tail      No. hk/torch is 1.5-2.4x at N = 512, 544, 576, 608, 1024, 1056
            (all exact multiples) and the same 1.5-2.4x at 577, 578, 1000,
            1025, 1026. The tail is handled.
  backend   torch's flash, mem-efficient and math backends all land within
            3e-6 of each other, so the comparison is against an effectively
            exact answer either way.
  shape     Spread. p50, p90 and p99 are each ~2.5x torch's, and the worst
            query rows are scattered across the sequence rather than
            clustered at one end.

So: the kernel is uniformly less precise than aotriton by about 2.5x, at every
quantile, with no structure to it. It stays inside one bf16 ulp of the fp32
answer (max 0.0025 against an ulp of 0.0039 at these magnitudes) -- but
aotriton is four times better than it needs to be, and the gap is enough to
flip a greedy token on an input where the top two are close. That is a real
limitation to know about before putting this in front of a model, and it is
not a correctness bug.

    python3 tools/hk-bench/attn_precision.py            # all three
    python3 tools/hk-bench/attn_precision.py --shape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch                                  # noqa: E402
import torch.nn.functional as F               # noqa: E402

from hk.ops import sdpa as hk_sdpa            # noqa: E402

H, D = 16, 64


def _case(n, heads=H, head_dim=D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(1, heads, n, head_dim, device="cuda",
                           dtype=torch.bfloat16, generator=g) for _ in range(3))
    s = head_dim ** -0.5
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         scale=s)
    hk = hk_sdpa.scaled_dot_product_attention(q, k, v, scale=s).float()
    th = F.scaled_dot_product_attention(q, k, v, scale=s).float()
    torch.cuda.synchronize()
    return q, k, v, s, ref, hk, th


def tail() -> None:
    print(f"{'N':>6} {'N%32':>5} {'hk vs fp32':>12} {'torch vs fp32':>14} "
          f"{'hk/torch':>9}")
    for n in (512, 544, 576, 577, 578, 608, 1000, 1024, 1025, 1026, 1056):
        _, _, _, _, ref, hk, th = _case(n)
        he = (hk - ref).abs().max().item()
        te = (th - ref).abs().max().item()
        print(f"{n:>6} {n % 32:>5} {he:>12.6f} {te:>14.6f} {he / te:>9.2f}")


def backend() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    names = {"flash": SDPBackend.FLASH_ATTENTION,
             "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
             "math": SDPBackend.MATH}
    for n in (577, 1025):
        q, k, v, s, ref, hk, th = _case(n)
        print(f"\n=== N={n}")
        print(f"  {'hk':<10} {(hk - ref).abs().max().item():.6f}")
        for name, be in names.items():
            try:
                with sdpa_kernel(be):
                    out = F.scaled_dot_product_attention(q, k, v, scale=s)
                torch.cuda.synchronize()
                d = (out.float() - ref).abs().max().item()
                print(f"  {name:<10} {d:.6f}")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<10} unavailable: {type(e).__name__}")
        print(f"  {'default':<10} {(th - ref).abs().max().item():.6f}")


def shape() -> None:
    for n in (577, 1025):
        _, _, _, _, ref, hk, th = _case(n)
        print(f"\n=== N={n}  ({n // 32} full KV blocks + {n % 32})")
        for name, out in (("hk", hk), ("torch", th)):
            e = (out - ref).abs()
            qs = [torch.quantile(e.flatten(), x).item()
                  for x in (0.5, 0.9, 0.99, 0.999)]
            print(f"  {name:<6} p50 {qs[0]:.6f}  p90 {qs[1]:.6f}  "
                  f"p99 {qs[2]:.6f}  p99.9 {qs[3]:.6f}  "
                  f"max {e.max().item():.6f}")
            worst = e.amax(dim=-1)[0].amax(dim=0)
            top = sorted(torch.topk(worst, 8).indices.tolist())
            print(f"         worst query rows: {top}  of {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--backend", action="store_true")
    ap.add_argument("--shape", action="store_true")
    args = ap.parse_args(argv)
    run_all = not (args.tail or args.backend or args.shape)
    if args.tail or run_all:
        print("--- does the error track the KV tail?")
        tail()
    if args.backend or run_all:
        print("\n--- which torch backend is the baseline?")
        backend()
    if args.shape or run_all:
        print("\n--- concentrated or spread?")
        shape()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
