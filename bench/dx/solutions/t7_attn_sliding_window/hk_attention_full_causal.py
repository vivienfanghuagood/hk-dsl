"""hk.attention (causal, GQA) past the window: it attends to every earlier
key instead of the last `window`, so it does more work and computes different
math -- an upper bound on what an hk kernel that skips the blocks outside the
window should take. Timed for scale only, N > window."""

import hk
from hk.ops import attn

KIND = "proxy"
PROXY = "full causal attention, no window: more work than the task"
LABEL = "hk.attention(causal=True) past the window: full causal work"


def supports(q, k, v, scale, window):
    if q.shape[2] <= window:
        return "N <= window: exact there, see hk_attention.py"
    return True if attn.supported(q, k, v, causal=True) else \
        f"hk.attention does not take this shape: {attn._why(q, k, v, True)}"


def solve(q, k, v, scale, window):
    return hk.attention(q, k, v, causal=True, scale=scale)
