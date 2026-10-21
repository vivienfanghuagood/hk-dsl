"""hk.attention (hk's tuned flash attention, causal, GQA) WITHOUT the
soft-cap: hk has no score modifier, so this is a lower bound on what an hk
kernel with the cap would take, not a solution. Head dims 64 and 128 only."""

import hk
from hk.ops import attn

KIND = "proxy"
PROXY = "causal attention without the tanh soft-cap (hk has no score_mod)"
LABEL = "hk.attention(causal=True), no soft-cap: a lower bound"


def supports(q, k, v, scale, cap):
    return True if attn.supported(q, k, v, causal=True) else \
        f"hk.attention does not take this shape: {attn._why(q, k, v, True)}"


def solve(q, k, v, scale, cap):
    return hk.attention(q, k, v, causal=True, scale=scale)
