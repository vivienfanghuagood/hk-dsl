"""Triton baseline: the matmul tutorial, autotuned, with bias + GELU(tanh)
fused into the epilogue (bench/dx/baselines/triton_gemm.py)."""

from bench.dx.baselines.triton_gemm import gemm

KIND = "baseline"
LABEL = "Triton matmul tutorial (autotuned) + fused bias/GELU epilogue"


def solve(x, weight, bias):
    return gemm(x, weight, 1, bias)
