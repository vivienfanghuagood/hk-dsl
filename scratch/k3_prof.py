"""hk.profile entry for K3 (Qwen shape):

    python -m hk.profile scratch/k3_prof.py:run --inputs scratch/k3_prof.py:inputs --flops 274877906944
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_attn  # noqa: E402

SHAPE = (1, 32, 4096, 128)


def inputs():
    import torch

    torch.manual_seed(0)
    return tuple(torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16) for _ in range(3))


def run(q, k, v):
    return k3_attn.attention(q, k, v)


def hk_sdpa(q, k, v):
    import hk.ops.sdpa as S

    return S.attention(q, k, v)


_S2 = None


def run_s2(q, k, v):
    global _S2
    if _S2 is None:
        _S2 = k3_attn.attn_kernel("k3p_s2", stages=2)
    return k3_attn.attention(q, k, v, attn=_S2)
