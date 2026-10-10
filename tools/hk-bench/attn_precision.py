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


def emulate_once(q, k, v, scale, *, kv_block=32, p_dtype=torch.bfloat16,
                 exact_exp2=False, l_from_cast=False):
    """The kernel's algorithm, in torch, one switch at a time.

    Mirrors `hk/ops/attn.py`'s inner loop exactly: walk KV in blocks, keep a
    running max and sum, rescale the accumulator when the max grows, round the
    probability tile to `p_dtype` before the PV matmul, and divide once at the
    end. Everything that is fp32 in the kernel is fp32 here.

    The point is to find which step costs the precision without recompiling
    anything. If this reproduces the kernel's 2.5x, the cause is the algorithm
    and the switches say which part of it; if it does not, the cause is in the
    kernel's arithmetic -- the hardware `v_exp_f32`, or an accumulation order
    torch cannot express.
    """
    b, h, n_q, d = q.shape
    n_kv = k.shape[2]
    qf, kf, vf = q.float(), k.float(), v.float()

    log2e = 1.4426950408889634
    acc = torch.zeros(b, h, n_q, d, device=q.device, dtype=torch.float32)
    m = torch.full((b, h, n_q, 1), float("-inf"), device=q.device)
    l = torch.zeros(b, h, n_q, 1, device=q.device)

    for lo in range(0, n_kv, kv_block):
        hi = min(lo + kv_block, n_kv)
        s_blk = (qf @ kf[:, :, lo:hi].transpose(-1, -2)) * (scale * log2e)
        m_new = torch.maximum(m, s_blk.amax(dim=-1, keepdim=True))
        shifted = s_blk - m_new
        p = torch.exp2(shifted) if exact_exp2 else _exp2_like_hw(shifted)
        alpha = torch.exp2(m - m_new) if exact_exp2 else _exp2_like_hw(m - m_new)
        alpha = torch.nan_to_num(alpha, nan=0.0)      # first block: -inf - -inf
        p_cast = p.to(p_dtype)
        l = l * alpha + (p_cast.float() if l_from_cast else p).sum(-1,
                                                                  keepdim=True)
        acc = acc * alpha + p_cast.float() @ vf[:, :, lo:hi]
        m = m_new
    return acc / l


def _exp2_like_hw(x):
    """`v_exp_f32` to within what torch can model: the hardware instruction is
    specified to 1 ulp, so round the exact result to a 1-ulp neighbourhood.

    This is a stand-in, not the instruction. It exists to bound how much of
    the gap a 1-ulp exp2 *could* explain -- if even this does not reproduce
    the kernel's error, exp2 is not the cause."""
    e = torch.exp2(x.double())
    # Perturb by half an ulp of fp32 in a deterministic, value-dependent way,
    # so the result is reproducible but not correlated with the exact answer.
    ulp = torch.ldexp(torch.ones_like(e), torch.floor(torch.log2(
        e.abs().clamp_min(1e-30))).to(torch.int64) - 23)
    bits = (e.abs().view(torch.int64) & 1).to(e.dtype) * 2 - 1
    return (e + bits * ulp).float()


def emulate() -> None:
    """Two columns, and the second one is the one that matters.

    `internal` is the emulated algorithm's own error in fp32. `rounded` is
    that answer put through bf16, which is what a kernel writing a bf16 output
    tensor actually returns -- and what `hk` and `torch` are measured as. Half
    a bf16 ulp is printed beside them because it is the floor: no kernel
    writing bf16 can do better, and a path whose internal error is well under
    it returns the correctly-rounded value almost everywhere.
    """
    cfgs = [
        ("kernel as written", {}),
        ("P in fp16", dict(p_dtype=torch.float16)),
        ("P in fp32", dict(p_dtype=torch.float32)),
        ("exact exp2", dict(exact_exp2=True)),
        ("l from cast P", dict(l_from_cast=True)),
        ("kv_block 128", dict(kv_block=128)),
        ("P fp16 + exact exp2", dict(p_dtype=torch.float16, exact_exp2=True)),
    ]
    for n in (577, 1025):
        q, k, v, s, ref, hk, th = _case(n)
        half_ulp = ref.abs().max().item() * 2 ** -9
        print(f"\n=== N={n}   hk {(hk - ref).abs().max():.6f}   "
              f"torch {(th - ref).abs().max():.6f}   "
              f"half a bf16 ulp {half_ulp:.6f}")
        print(f"  {'':<22} {'internal':>10} {'rounded':>10}")
        pred = None
        for name, kw in cfgs:
            out = emulate_once(q, k, v, s, **kw)
            e = (out - ref).abs().max().item()
            rounded = out.to(torch.bfloat16).float()
            r = (rounded - ref).abs().max().item()
            print(f"  {name:<22} {e:>10.6f} {r:>10.6f}")
            if pred is None:
                pred = rounded

        # The sharp question: is the kernel doing what its algorithm says?
        # `pred` is the algorithm's answer put through bf16, which is what a
        # faithful kernel would return.
        d_hk = (pred - hk).abs()
        d_th = (pred - th).abs()
        print(f"  {'vs the algorithm:':<22} {'max':>10} {'differs on':>12}")
        for who, d in (("hk", d_hk), ("torch", d_th)):
            n_off = (d > 0).sum().item()
            print(f"    {who:<20} {d.max().item():>10.6f} "
                  f"{100 * n_off / d.numel():>11.1f}%")


def exact() -> None:
    """Probes whose answer is known exactly, so there is nothing to argue with.

    Each one pins a different part of the kernel:

      V = 1     softmax weights sum to one, so the output is exactly 1.0 for
                every query and every channel -- whatever Q and K were. Any
                deviation is the kernel disagreeing with *itself*: the
                numerator accumulates the bf16-rounded probability tile
                through WMMA while the denominator sums the fp32 one with
                col_sum, and with V = 1 nothing else can contribute.

      Q = 0     every score is equal, so P is uniformly 1/N and the output is
                the mean of V along the sequence, computable in fp64. This
                keeps the PV matmul and the V operand in play but removes the
                exponential and the running max.
    """
    for n in (577, 1025):
        for name in ("V=1", "Q=0"):
            g = torch.Generator(device="cuda").manual_seed(0)
            q, k, v = (torch.randn(1, H, n, D, device="cuda",
                                   dtype=torch.bfloat16, generator=g)
                       for _ in range(3))
            if name == "V=1":
                v = torch.ones_like(v)
                want = torch.ones(1, H, n, D, device="cuda")
            else:
                q = torch.zeros_like(q)
                want = v.double().mean(dim=2, keepdim=True).expand(
                    1, H, n, D).float()
            s = D ** -0.5
            hk = hk_sdpa.scaled_dot_product_attention(q, k, v, scale=s).float()
            th = F.scaled_dot_product_attention(q, k, v, scale=s).float()
            torch.cuda.synchronize()
            he = (hk - want).abs().max().item()
            te = (th - want).abs().max().item()
            rel_h = ((hk - want).abs() / want.abs().clamp_min(1e-20)).max()
            rel_t = ((th - want).abs() / want.abs().clamp_min(1e-20)).max()
            print(f"  N={n:<6} {name:<6} hk {he:.6f} (rel {rel_h:.2e})   "
                  f"torch {te:.6f} (rel {rel_t:.2e})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--backend", action="store_true")
    ap.add_argument("--shape", action="store_true")
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument("--exact", action="store_true")
    args = ap.parse_args(argv)
    run_all = not (args.tail or args.backend or args.shape
                   or args.emulate or args.exact)
    if args.tail or run_all:
        print("--- does the error track the KV tail?")
        tail()
    if args.backend or run_all:
        print("\n--- which torch backend is the baseline?")
        backend()
    if args.shape or run_all:
        print("\n--- concentrated or spread?")
        shape()
    if args.emulate or run_all:
        print("\n--- which step costs it? (the algorithm, in torch)")
        emulate()
    if args.exact or run_all:
        print("\n--- probes with an exactly known answer")
        exact()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
