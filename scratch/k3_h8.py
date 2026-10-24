"""K3 at the 4-GPU (Ulysses, 8 heads a GPU) and 1-GPU Qwen-Image 2.1 shapes: q 4096, kv 4141."""
import os
import torch
from hk.ops import attn_k3 as K3
import hk.ops.sdpa as S


def t_ms(f, iters=20, trials=7):
    f(); f(); torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            f()
        e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    return sorted(ts)[len(ts) // 2]


print("HK_K3_WG_PER_WGP", os.environ.get("HK_K3_WG_PER_WGP", "3"))
for h in (8, 16, 32):
    nq, nkv = 4096, 4141
    q = torch.randn((1, nq, h, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k, v = (torch.randn((1, nkv, h, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(2))
    fl = 4 * h * nq * nkv * 128
    tk = t_ms(lambda: K3.attention(q, k, v))
    th = t_ms(lambda: S.scaled_dot_product_attention(q, k, v))
    print(f"H{h} q{nq} kv{nkv}: k3 {tk:.3f} ms {fl / tk / 1e9:5.1f} TF/s | hk {th:.3f} ms {fl / th / 1e9:5.1f} "
          f"| {th / tk:.2f}x | plan {K3.plan(h * 32, q.device)}", flush=True)
