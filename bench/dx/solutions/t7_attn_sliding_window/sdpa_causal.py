"""torch's fused SDPA, causal, where the window covers every causal key
(N <= window): there plain causal attention is the task, and this is the
fastest torch path (what a user would call once they notice)."""

import torch.nn.functional as F

KIND = "baseline"
LABEL = "F.scaled_dot_product_attention(is_causal=True), exact while N <= window"


def supports(q, k, v, scale, window):
    return True if q.shape[2] <= window else f"N={q.shape[2]} > window={window}: SDPA's causal flag has no window"


def solve(q, k, v, scale, window):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
