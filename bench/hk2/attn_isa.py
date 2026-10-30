"""Hot-loop instruction mix of the attention kernel: python bench/hk2/attn_isa.py [BMxBNxW] [--asm out/a.s]"""
import argparse
import os

import torch

from hk2 import isa
from hk2.kernels.attention import attention, attn_fwd

ap = argparse.ArgumentParser()
ap.add_argument("cfg", nargs="?", default="128x64x8")
ap.add_argument("--asm")
ap.add_argument("--d", type=int, default=128)
a = ap.parse_args()
bm, bn, w = map(int, a.cfg.split("x"))
q, k, v = (torch.randn(1, 2, 512, a.d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
attention(q, k, v, BM=bm, BN=bn, num_warps=w)
kern = list(attn_fwd.cache.values())[-1]
print(kern.stats)
print(isa.summary(kern.asm))
if a.asm:
    os.makedirs(os.path.dirname(a.asm), exist_ok=True)
    open(a.asm, "w").write(kern.asm)
    open(a.asm.replace(".s", ".ll"), "w").write(kern.ir)
