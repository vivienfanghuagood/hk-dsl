"""hk's decode GEMV (hk.ops.decode_gemm.matmul, the kernel vLLM's patch routes
decode-sized batches to) where it takes the shape, then the epilogue in torch.
On gfx950 hk.ops.matmul already routes small M here; on gfx1100 it does not,
so this is the hk.ops route an expert would pick at M <= 64."""

import torch.nn.functional as F

import hk
from hk.ops import decode_gemm

KIND = "hk_ops"
LABEL = "hk.ops.decode_gemm.matmul + torch bias/GELU epilogue (small M only)"


def supports(x, weight, bias):
    m, k = x.shape
    n = weight.shape[0]
    return True if decode_gemm.fits(m, n, k) else f"decode GEMM does not take M={m}, N={n}, K={k}"


def solve(x, weight, bias):
    y = decode_gemm.matmul(x, weight)
    return F.gelu(y.add_(bias), approximate="tanh")
