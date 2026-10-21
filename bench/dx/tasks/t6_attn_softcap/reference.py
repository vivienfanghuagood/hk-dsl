"""T6 attn_softcap: the reference, the inputs and the check. task.md is the problem."""

import torch

from bench.dx.tasks import attn_common

_G27 = dict(Hq=32, Hkv=16, D=128, scale=144 ** -0.5)
_G9 = dict(Hq=16, Hkv=8, D=256, scale=256 ** -0.5)
SHAPES = [
    dict(name="27b-B1-N1024", B=1, N=1024, **_G27),
    dict(name="27b-B1-N4096", B=1, N=4096, **_G27),
    dict(name="27b-B1-N8192", B=1, N=8192, **_G27),
    dict(name="27b-B4-N2048", B=4, N=2048, **_G27),
    dict(name="9b-B1-N2048", B=1, N=2048, **_G9),
    dict(name="9b-B1-N8192", B=1, N=8192, **_G9),
]
QUICK = ("27b-B1-N1024", "27b-B1-N4096", "9b-B1-N2048")
SMALL = [dict(name="small", B=1, N=256, Hq=4, Hkv=2, D=128, scale=144 ** -0.5),
         dict(name="small-ragged", B=1, N=200, Hq=4, Hkv=2, D=128, scale=144 ** -0.5)]
CAP = 50.0


def make_inputs(shape, device="cuda", seed=0):
    ins = attn_common.make_inputs(shape, device, seed, qk_std=3.0)
    return {**ins, "scale": float(shape["scale"]), "cap": CAP}


def reference(q, k, v, scale, cap):
    return attn_common.attention_ref(q, k, v, scale, cap=cap)


def torch_impl(q, k, v, scale, cap):
    """Hugging Face's eager Gemma-2 attention, with the scores in fp32: HF's own
    (scores rounded to bf16 before the soft-cap) is 7% off at these logits and
    fails the check."""
    g = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    s = torch.tanh(s / cap) * cap
    n = q.shape[2]
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
    s = s.masked_fill(causal, float("-inf"))
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, v)


def compile_setup(q, k, v, scale, cap):
    """torch.compile's attention with a modified score: flex_attention. The
    block mask is built once per sequence length, outside the timing, as a
    model builds it once per forward."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex = torch.compile(flex_attention, dynamic=False)
    n = q.shape[2]
    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, n, n, device=q.device)

    def softcap(score, b, h, qi, ki):
        return cap * torch.tanh(score / cap)

    return lambda: flex(q, k, v, score_mod=softcap, block_mask=mask, scale=scale, enable_gqa=True)


def check(out, ref, inputs):
    return attn_common.check(out, ref, inputs)


def work(shape):
    return {"flops": attn_common.flops(shape)}
