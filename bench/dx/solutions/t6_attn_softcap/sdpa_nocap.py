"""torch's fused SDPA (causal, GQA) WITHOUT the soft-cap: SDPA has no score
modifier, so this is torch's fastest attention at this size, a lower bound
on the time, not a solution."""

import torch.nn.functional as F

KIND = "proxy"
PROXY = "causal attention without the tanh soft-cap (SDPA has no score_mod)"
LABEL = "F.scaled_dot_product_attention(is_causal=True), no soft-cap: a lower bound"


def solve(q, k, v, scale, cap):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
