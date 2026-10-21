"""T4 linear_bias_gelu: the reference, the inputs and the check. task.md is the problem."""

import torch
import torch.nn.functional as F

from bench.dx.common import assert_close

SHAPES = [dict(name=f"M{m}-K4096-N{n}", M=m, K=4096, N=n)
          for n in (12288, 14336) for m in (16, 128, 1024, 4096)]
QUICK = ("M16-K4096-N12288", "M1024-K4096-N12288", "M4096-K4096-N14336")
SMALL = [dict(name="small", M=64, K=512, N=1024), dict(name="small-ragged", M=5, K=256, N=512)]


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    m, k, n = shape["M"], shape["K"], shape["N"]
    x = torch.randn(m, k, generator=g, device=device).to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device=device) * k ** -0.5).to(torch.bfloat16)
    bias = (0.5 * torch.randn(n, generator=g, device=device)).to(torch.bfloat16)
    return {"x": x, "weight": weight, "bias": bias}


def reference(x, weight, bias):
    return F.gelu(x.float() @ weight.float().T + bias.float(), approximate="tanh")


def torch_impl(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")


def check(out, ref, inputs):
    return assert_close("out", out, ref, rtol=2e-2, atol=2e-2, dtype=torch.bfloat16)


def work(shape):
    m, k, n = shape["M"], shape["K"], shape["N"]
    return {"flops": 2 * m * n * k, "bytes": 2 * (m * k + n * k + m * n + n)}
