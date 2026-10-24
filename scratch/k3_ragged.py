"""Where K3 loses at ragged N: V^T (hk kernel vs torch), attn alone, by N (even/odd, multiple of 128)."""
import torch

from hk.ops import attn_k3 as K3
import hk.ops.sdpa as S


def t_ms(f, iters=10, trials=5):
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


import os
print('HK_K3_WG_PER_WGP', os.environ.get('HK_K3_WG_PER_WGP', '3'))
for h, n in [(24, 4224), (24, 4307), (24, 4308), (32, 4096), (32, 4307), (32, 4352)]:
    q, k, v = (torch.randn((1, n, h, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3))
    fl = 4 * h * n * n * 128
    tk = t_ms(lambda: K3.attention(q, k, v))
    th = t_ms(lambda: S.scaled_dot_product_attention(q, k, v))
    n_pad = -(-n // 128) * 128
    tvt = t_ms(lambda: K3._vt_torch(v, n_pad))
    nqb = -(-n // 128)
    print(f"H{h} N{n}: k3 {tk:.3f} ms {fl / tk / 1e9:5.1f} TF/s | hk {th:.3f} ms {fl / th / 1e9:5.1f} | "
          f"torch V^T {tvt:.3f} ms | plan {K3.plan(h * nqb, q.device)}", flush=True)
