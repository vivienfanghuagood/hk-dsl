"""T7 attn_sliding_window: the reference, the inputs and the check. task.md is the problem."""

import functools

import torch
import torch.nn.functional as F

from bench.dx.tasks import attn_common

_MISTRAL = dict(Hq=32, Hkv=8, D=128, scale=128 ** -0.5)
SHAPES = [
    dict(name="B1-N2048", B=1, N=2048, **_MISTRAL),
    dict(name="B1-N4096", B=1, N=4096, **_MISTRAL),
    dict(name="B1-N8192", B=1, N=8192, **_MISTRAL),
    dict(name="B1-N16384", B=1, N=16384, **_MISTRAL),
    dict(name="B2-N8192", B=2, N=8192, **_MISTRAL),
]
QUICK = ("B1-N2048", "B1-N8192")
SMALL = [dict(name="small", B=1, N=512, Hq=4, Hkv=1, D=128, scale=128 ** -0.5, window=128),
         dict(name="small-ragged", B=1, N=300, Hq=4, Hkv=2, D=128, scale=128 ** -0.5, window=100),
         dict(name="small-inwindow", B=1, N=256, Hq=4, Hkv=2, D=128, scale=128 ** -0.5, window=256)]
WINDOW = 4096


def make_inputs(shape, device="cuda", seed=0):
    ins = attn_common.make_inputs(shape, device, seed)
    return {**ins, "scale": float(shape["scale"]), "window": int(shape.get("window", WINDOW))}


def reference(q, k, v, scale, window):
    return attn_common.attention_ref(q, k, v, scale, window=window)


@functools.lru_cache(maxsize=4)
def _window_mask(n, window, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (i - j < window)


def torch_impl(q, k, v, scale, window):
    """SDPA with a boolean mask, built once per length (as a model builds it once per forward)."""
    mask = _window_mask(q.shape[2], window, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)


def compile_setup(q, k, v, scale, window):
    """torch.compile's attention with a custom mask: flex_attention, its block
    mask built once per length, outside the timing."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex = torch.compile(flex_attention, dynamic=False)
    n = q.shape[2]

    def sliding(b, h, qi, ki):
        return (qi >= ki) & (qi - ki < window)

    mask = create_block_mask(sliding, None, None, n, n, device=q.device)
    return lambda: flex(q, k, v, block_mask=mask, scale=scale, enable_gqa=True)


def check(out, ref, inputs):
    return attn_common.check(out, ref, inputs)


def work(shape):
    return {"flops": attn_common.flops(shape, shape.get("window", WINDOW))}
