"""hk.templates.rowwise: the residual add, the RMSNorm and the per-token int8
quantization as one row's math -- two row reductions (the mean square, then
the absmax of the normed row), three outputs. The updated residual is written
into `residual` in place, as vLLM does (MUTABLE_INPUTS)."""

import torch

import hk
from hk import templates as T

KIND = "template"
LABEL = "hk.templates.rowwise: add + RMSNorm + absmax int8, one kernel"


@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32), hk.bf16))
def add_rms_quant(x, r, w, *, EPS=1e-6):
    h = x + r
    y = h * (T.row_mean(h * h) + EPS).rsqrt() * w
    amax = T.row_max(abs(y)).maximum(1e-10)
    return y * (127.0 / amax), amax / 127.0, h


def setup(x, residual, weight, eps):
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    return lambda: add_rms_quant(x, residual, weight, EPS=eps, out=(q, scale, residual))


def solve(x, residual, weight, eps):
    return setup(x, residual, weight, eps)()
