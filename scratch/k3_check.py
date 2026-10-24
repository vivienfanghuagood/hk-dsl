"""hk.ops.attn_k3 checks (ragged lengths, both layouts) and interleaved timing.

    python scratch/k3_check.py [--time]
Correctness: max-rel <= 1e-2 vs fp32 math SDPA, V=1 gives exactly 1.0, 5 repeats bitwise, no NaN.
Timing: in place on BNHD views against hk.ops.sdpa (its in-place BNHD path) and the tuned
non-causal Triton FA-2 on the same strides.
"""
import argparse
import os
import sys

import torch

from hk.ops import attn_k3 as K3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party",
                                "HipKittens", "kernels", "rdna3", "attn", "baselines"))

# (B, H, N_q, N_kv)
CASES = [(1, 2, 512, 512), (2, 3, 1485, 1485), (1, 4, 640, 1001), (1, 2, 128, 33),
         (1, 24, 4307, 4307), (2, 2, 4096, 4096)]


def make(b, h, nq, nkv, bnhd, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def t(n):
        if bnhd:
            return torch.randn((b, n, h, 128), device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)
        return torch.randn((b, h, n, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    return t(nq), t(nkv), t(nkv)


def check(case, bnhd, offset=False):
    q, k, v = make(*case, bnhd)
    if offset:   # v a view at a nonzero storage offset (the second of two stacked tensors)
        v = torch.stack([torch.zeros_like(v), v])[1]
        assert v.storage_offset() > 0
    assert K3.why(q, k, v) == "", K3.why(q, k, v)
    o = K3.attention(q, k, v).clone()
    torch.cuda.synchronize()
    r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = ((o.float() - r).abs().max() / r.abs().max()).item()
    det = all(torch.equal(o, K3.attention(q, k, v)) for _ in range(5))
    ones = K3.attention(q, k, torch.ones_like(v))
    ones_ok = bool((ones == 1).all().item())
    nan = bool(torch.isnan(o).any().item())
    layout_ok = (o.transpose(1, 2).is_contiguous() if bnhd else o.is_contiguous())
    ok = err <= 1e-2 and det and ones_ok and not nan and layout_ok
    print(f"{'BNHD' if bnhd else 'BHND'} {case} max-rel {err:.2e} det {det} ones {ones_ok} nan {nan} "
          f"layout {layout_ok} -> {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def triton_strided(tf, q, k, v):
    import triton

    b, h, n, d = q.shape
    o = torch.empty_like(q)
    grid = lambda META: (triton.cdiv(n, META["BLOCK_M"]), b * h)  # noqa: E731
    tf._attn_fwd[grid](q, k, v, o, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
                       h, n, d ** -0.5, HEAD_DIM=d, IS_CAUSAL=False)
    return o


def timeit(case, trials=7):
    import hk.ops.sdpa as S
    import triton_fa

    b, h, n, nkv = case
    q, k, v = make(b, h, n, nkv, True)
    flops = 4 * b * h * n * nkv * 128
    fns = {"k3": lambda: K3.attention(q, k, v),
           "hk": lambda: S.scaled_dot_product_attention(q, k, v)}
    if n == nkv:
        fns["triton"] = lambda: triton_strided(triton_fa, q, k, v)
    iters = 10 if n * nkv <= 4096 * 4400 else 3
    for f in fns.values():
        f(); f()
    torch.cuda.synchronize()
    res = {nm: [] for nm in fns}
    for _ in range(trials):
        for nm, f in fns.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                f()
            e1.record()
            torch.cuda.synchronize()
            res[nm].append(e0.elapsed_time(e1) / iters)
    print(f"time BNHD {case}", flush=True)
    for nm, ts in res.items():
        t = sorted(ts)[len(ts) // 2]
        print(f"  {nm:7s} {t:9.3f} ms {flops / t / 1e9:6.1f} TF/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    ok = all([check(c, bnhd) for c in CASES for bnhd in (False, True)])
    ok = all([check((1, 3, 1001, 1001), False, True), check((2, 2, 777, 777), False, True)]) and ok
    if a.time and ok:
        for c in [(1, 24, 4307, 4307), (1, 32, 4096, 4096), (1, 56, 14720, 14720)]:
            timeit(c)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
