"""hk.attention (hk's tuned flash attention, causal, GQA) where the window
covers every causal key (N <= window): there it computes the task exactly.
Past the window it would do the full causal work and give a different
answer; see hk_attention_full_causal.py for that proxy."""

import hk
from hk.ops import attn

KIND = "hk_ops"
LABEL = "hk.attention(causal=True), exact while N <= window"


def supports(q, k, v, scale, window):
    if q.shape[2] > window:
        return f"N={q.shape[2]} > window={window}: hk.attention has no window (mask_mod)"
    return True if attn.supported(q, k, v, causal=True) else \
        f"hk.attention does not take this shape: {attn._why(q, k, v, True)}"


def solve(q, k, v, scale, window):
    return hk.attention(q, k, v, causal=True, scale=scale)
