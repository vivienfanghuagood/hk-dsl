"""hk's tuned GEMM, then the epilogue in torch: hk.ops.matmul (on gfx950 it
routes per shape among hk's CDNA kernels, the decode GEMV included; on gfx1100
it is the tuned RDNA GEMM), then `+ bias` and GELU(tanh) as two torch ops.
hk has no GEMM with an epilogue (G2's hk.templates.gemm is meant to be one)."""

import torch.nn.functional as F

import hk

KIND = "hk_ops"
LABEL = "hk.ops.matmul + torch bias/GELU epilogue"


def solve(x, weight, bias):
    y = hk.ops.matmul(x, weight)
    return F.gelu(y.add_(bias), approximate="tanh")
