"""K3 c1's knobs, every variant checked (max-rel, V=1) then timed interleaved with c1 in one process.

    python scratch/k3_sweep.py [--shape qwen|h3]
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_attn  # noqa: E402

SHAPES = {"qwen": (1, 32, 4096, 128), "h3": (1, 56, 14720, 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen")
    a = ap.parse_args()
    shape = SHAPES[a.shape]
    b, h, n, d = shape
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    variants = {"c1": k3_attn.ATTN}
    for kvb, qt, pt, sp in itertools.product((32, 64), (1, 2, 4), (2, 4, 8), (True, False)):
        if kvb % (16 * qt) or (kvb, qt, pt, sp) == (32, 2, 2, True):
            continue
        name = f"kv{kvb}_q{qt}_p{pt}_{'sp' if sp else 'nosp'}"
        try:
            variants[name] = k3_attn.attn_kernel(f"k3s_{name}", kv_block=kvb, qk_tiles=qt, pv_tiles=pt, setprio=sp)
        except Exception as e:  # noqa: BLE001
            print(f"{name:24s} build error {type(e).__name__}: {str(e)[:120]}", flush=True)
    good = {}
    for name, kern in variants.items():
        try:
            o = k3_attn.attention(q, k, v, attn=kern)
            ones = k3_attn.attention(q, k, torch.ones_like(v), attn=kern)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"{name:24s} run error {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        err = ((o.float() - r).abs().max() / r.abs().max()).item()
        ok = err <= 1e-2 and bool((ones == 1).all().item())
        if not ok:
            print(f"{name:24s} WRONG max-rel {err:.2e}", flush=True)
            continue
        good[name] = kern
    flops = 4 * b * h * n * n * d
    res = {nm: [] for nm in good}
    for _ in range(5):
        for nm, kern in good.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                k3_attn.attention(q, k, v, attn=kern)
            e1.record()
            torch.cuda.synchronize()
            res[nm].append(e0.elapsed_time(e1) / 5)
    rows = sorted((sorted(ts)[len(ts) // 2], nm) for nm, ts in res.items())
    for t, nm in rows:
        print(f"{nm:24s} {t:8.3f} ms {flops / t / 1e9:6.1f} TF/s", flush=True)


if __name__ == "__main__":
    main()
