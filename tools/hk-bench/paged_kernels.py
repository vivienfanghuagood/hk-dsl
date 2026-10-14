#!/usr/bin/env python3
"""Per-kernel GPU time of one decode layer, stock against hk, with 36 layers'
caches so that K/V is read cold the way a decode step reads it.

  stock      vLLM's Triton backend: cache write, 3-D kernel, reduction
  hk         this backend's forward, as routed (vllm_backend.TRITON_DECODE_MAX)
  hk split   hk's own split decode, walk and merge, whatever the batch
  tri_on_hk  Triton's kernel reading hk's head-major cache

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/paged_kernels.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import paged_ab as ab  # noqa: E402

LAYERS = 36


def kernels(fn, calls=3):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    tot, cnt = defaultdict(float), defaultdict(int)
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0:
            tot[e.name] += e.device_time_total
            cnt[e.name] += 1
    per = calls * LAYERS
    return {k: (tot[k] / per, cnt[k] / per) for k in tot}


def main():
    for ctx in (1024, 4096):
        for n in (1, 4):
            gen = torch.Generator(device="cuda").manual_seed(n * 100003 + ctx)
            key = torch.randn(n, ctx, ab.HKV, ab.D, device="cuda",
                              dtype=ab.BF16, generator=gen)
            value = torch.randn_like(key)
            q = torch.randn(n, ab.HQ, ab.D, device="cuda", dtype=ab.BF16,
                            generator=gen)
            k_new, v_new = key[:, -1].contiguous(), value[:, -1].contiguous()
            stock, _ = ab.stock_arms(n, ctx, key, value, q, k_new, v_new, gen,
                                     LAYERS)
            hk, _, _, splits = ab.hk_arms(n, ctx, key, value, q, k_new, v_new,
                                          gen, LAYERS)
            del key, value
            print(f"\nctx {ctx} batch {n} (hk splits {splits}): us per layer")
            for arm, fn in (("stock", stock["stock_layer"]),
                            ("hk", hk["hk_layer"]),
                            ("hk split", hk["hk_kernel"]),
                            ("tri_on_hk", hk["tri_on_hk"])):
                ks = kernels(fn)
                total = sum(t for t, _ in ks.values())
                print(f"  {arm:<9} total {total:6.2f}")
                for name, (t, c) in sorted(ks.items(), key=lambda kv: -kv[1][0]):
                    print(f"         {t:6.2f}  x{c:.0f}  {name[:90]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
