"""Print the hot-loop instruction mix of the GEMM benchmark kernel for one config.

    python bench/hk2/isa_mix.py 128x128x64x4 [--nn] [--asm out/k.s]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from gemm import matmul  # noqa: E402

import hk2 as hk  # noqa: E402
from hk2 import isa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cfg")
ap.add_argument("--nn", action="store_true")
ap.add_argument("--asm")
args = ap.parse_args()
bm, bn, bk, nw = map(int, args.cfg.split("x"))
m = n = k = 1024
a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) if args.nn else torch.randn(n, k, device="cuda",
                                                                                       dtype=torch.bfloat16)
c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
kern = matmul.warmup(a, b, c, BM=bm, BN=bn, BK=bk, TRANS_B=not args.nn, num_warps=nw)
print(kern.stats)
print(isa.summary(kern.asm))
if args.asm:
    os.makedirs(os.path.dirname(args.asm), exist_ok=True)
    open(args.asm, "w").write(kern.asm)
    open(args.asm.replace(".s", ".ll"), "w").write(kern.ir)
