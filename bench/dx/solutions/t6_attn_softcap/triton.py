"""Triton baseline: the fused-attention tutorial's forward with GQA and the
tanh soft-cap (bench/dx/baselines/triton_attn.py), autotuned."""

from bench.dx.baselines.triton_attn import attention

KIND = "baseline"
LABEL = "Triton fused-attention tutorial + GQA + soft-cap (autotuned)"


def solve(q, k, v, scale, cap):
    return attention(q, k, v, scale, cap=cap)
