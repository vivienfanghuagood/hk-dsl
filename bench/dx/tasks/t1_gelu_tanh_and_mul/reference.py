"""T1 gelu_tanh_and_mul: the reference, the inputs and the check. task.md is the problem."""

import torch
import torch.nn.functional as F

from bench.dx.common import assert_close

SHAPES = [
    dict(name="M1-I14336", M=1, I=14336),
    dict(name="M16-I14336", M=16, I=14336),
    dict(name="M37-I9216", M=37, I=9216),
    dict(name="M128-I14336", M=128, I=14336),
    dict(name="M1024-I14336", M=1024, I=14336),
    dict(name="M4096-I14336", M=4096, I=14336),
    dict(name="M4096-I9216", M=4096, I=9216),
    dict(name="M1024-I21504", M=1024, I=21504),
]
QUICK = ("M16-I14336", "M37-I9216", "M4096-I14336")
SMALL = [dict(name="small", M=16, I=1024), dict(name="small-ragged", M=5, I=512)]


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape["M"], 2 * shape["I"], generator=g, device=device, dtype=torch.bfloat16)
    return {"x": x}


def reference(x):
    """Golden: fp32 throughout."""
    gate, up = x.float().chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


def torch_impl(x):
    """What a PyTorch user writes (vLLM's GeluAndMul.forward_native)."""
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]


def check(out, ref, inputs):
    return assert_close("out", out, ref, rtol=2e-2, atol=1e-2, dtype=torch.bfloat16)


def work(shape):
    return {"bytes": 3 * shape["M"] * shape["I"] * 2}
