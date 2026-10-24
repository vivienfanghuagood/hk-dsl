"""Per-call accuracy at Qwen-Image 2.1's real shape: K3 vs hk.ops.sdpa vs torch (aotriton), against
fp32 math SDPA. Bias (mean signed error) matters most: it accumulates over 30 steps."""
import os
import torch
import torch.nn.functional as F
import hk.ops.sdpa as S
from hk.ops import attn_k3 as K3

torch.manual_seed(0)
b, h, nq, nkv, d = 1, 32, 4096, 4141, 128
mk = lambda n, sc: (torch.randn((b, n, h, d), device="cuda") * sc).to(torch.bfloat16).transpose(1, 2)
for sc in (1.0, 3.0):     # 3.0: peakier softmax
    q, k, v = mk(nq, sc), mk(nkv, sc), mk(nkv, 1.0)
    r = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    outs = {"k3": K3.attention(q, k, v), "hk": S.scaled_dot_product_attention(q, k, v),
            "torch": F.scaled_dot_product_attention(q, k, v)}
    rb = r.to(torch.bfloat16).float()
    print(f"scale {sc}: bf16(ref) itself: mean abs {(rb - r).abs().mean():.3e}")
    for nm, o in outs.items():
        e = o.float() - r
        print(f"  {nm:6s} mean abs {e.abs().mean():.3e}  max abs {e.abs().max():.3e}  "
              f"bias {e.mean():+.3e}  bias/|o| {(e * r.sign()).mean() / r.abs().mean():+.3e}", flush=True)
