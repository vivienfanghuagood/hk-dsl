"""K3 c1 check: correctness vs fp32 math SDPA, V-all-ones == 1.0, repeat determinism, then interleaved
timing against hk.ops.sdpa.attention (current) and the tuned non-causal Triton FA-2
(third_party/HipKittens/kernels/rdna3/attn/baselines/triton_fa.py; bench/dx's triton_attn is causal), in one process.

    python scratch/k3_test.py [--time] [--shapes qwen h3]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "HipKittens", "kernels", "rdna3", "attn", "baselines"))
import k3_attn  # noqa: E402

SHAPES = {"small": (1, 2, 512, 128), "odd": (1, 3, 1408, 128), "qwen": (1, 32, 4096, 128),
          "h3": (1, 56, 14720, 128)}


def ref(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


def check(name, shape):
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = k3_attn.attention(q, k, v)
    torch.cuda.synchronize()
    r = ref(q, k, v)
    err = ((o.float() - r).abs().max() / r.abs().max()).item()
    outs = [k3_attn.attention(q, k, v).clone() for _ in range(5)]
    det = all(torch.equal(outs[0], x) for x in outs[1:]) and torch.equal(outs[0], o)
    ones = k3_attn.attention(q, k, torch.ones_like(v))
    ones_ok = bool((ones == 1).all().item())
    nan = bool(torch.isnan(o).any().item())
    ok = err <= 1e-2 and det and ones_ok and not nan
    print(f"{name:6s} {shape} max-rel {err:.2e} det {det} ones {ones_ok} nan {nan} -> {'OK' if ok else 'FAIL'}",
          flush=True)
    return ok


def timeit(shape, trials=7, iters=10):
    import hk.ops.sdpa as S
    import triton_fa

    b, h, n, d = shape
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flops = 4 * b * h * n * n * d
    fns = {"k3": lambda: k3_attn.attention(q, k, v),
           "hk": lambda: S.attention(q, k, v),
           "triton": lambda: triton_fa.triton_attention(q, k, v, causal=False)}
    for f in fns.values():
        f(); f()
    torch.cuda.synchronize()
    res = {n_: [] for n_ in fns}
    for _ in range(trials):
        for n_, f in fns.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                f()
            e1.record()
            torch.cuda.synchronize()
            res[n_].append(e0.elapsed_time(e1) / iters)
    out = {}
    for n_, ts in res.items():
        t = sorted(ts)[len(ts) // 2]
        out[n_] = t
        print(f"  {n_:7s} {t:8.3f} ms  {flops / t / 1e9:6.1f} TF/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--shapes", nargs="*", default=["small", "odd", "qwen", "h3"])
    a = ap.parse_args()
    ok = all([check(s, SHAPES[s]) for s in a.shapes])
    if a.time and ok:
        for s in a.shapes:
            if s in ("qwen", "h3"):
                print(s, SHAPES[s], flush=True)
                timeit(SHAPES[s])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
