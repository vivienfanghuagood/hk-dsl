"""Triton baseline: the fused-attention tutorial's forward with GQA and a
sliding window that skips the key blocks outside it
(bench/dx/baselines/triton_attn.py), autotuned."""

from bench.dx.baselines.triton_attn import attention

KIND = "baseline"
LABEL = "Triton fused-attention tutorial + GQA + window, blocks skipped (autotuned)"


def solve(q, k, v, scale, window):
    return attention(q, k, v, scale, window=window)
