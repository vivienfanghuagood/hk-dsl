"""T5 linear_residual: the reference, the inputs and the check. task.md is the problem."""

import torch
import torch.nn.functional as F

from bench.dx.common import assert_close

SHAPES = ([dict(name=f"oproj-M{m}", M=m, K=4096, N=4096) for m in (16, 128, 1024, 4096)]
          + [dict(name=f"down-M{m}-K12288", M=m, K=12288, N=4096) for m in (16, 128, 1024, 4096)]
          + [dict(name=f"down-M{m}-K14336", M=m, K=14336, N=4096) for m in (128, 4096)])
QUICK = ("oproj-M16", "oproj-M1024", "down-M4096-K12288")
SMALL = [dict(name="small", M=64, K=512, N=1024), dict(name="small-ragged", M=5, K=256, N=512)]


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    m, k, n = shape["M"], shape["K"], shape["N"]
    x = torch.randn(m, k, generator=g, device=device).to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device=device) * k ** -0.5).to(torch.bfloat16)
    residual = torch.randn(m, n, generator=g, device=device).to(torch.bfloat16)
    return {"x": x, "weight": weight, "residual": residual}


def reference(x, weight, residual):
    return x.float() @ weight.float().T + residual.float()


def torch_impl(x, weight, residual):
    return F.linear(x, weight) + residual


def check(out, ref, inputs):
    return assert_close("out", out, ref, rtol=2e-2, atol=2e-2, dtype=torch.bfloat16)


def work(shape):
    m, k, n = shape["M"], shape["K"], shape["N"]
    return {"flops": 2 * m * n * k, "bytes": 2 * (m * k + n * k + 2 * m * n)}
