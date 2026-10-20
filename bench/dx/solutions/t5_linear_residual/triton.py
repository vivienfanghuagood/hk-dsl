"""Triton baseline: the matmul tutorial, autotuned, with the residual add
fused into the epilogue (bench/dx/baselines/triton_gemm.py)."""

from bench.dx.baselines.triton_gemm import gemm

KIND = "baseline"
LABEL = "Triton matmul tutorial (autotuned) + fused residual epilogue"


def solve(x, weight, residual):
    return gemm(x, weight, 2, residual)
