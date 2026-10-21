"""Triton baseline: one program per row, the add, the norm and the quantization
fused, the row held in registers (vLLM's Triton-style fused kernels)."""

import torch
import triton
import triton.language as tl

KIND = "baseline"
LABEL = "Triton: one program per row, all three steps fused"


@triton.jit
def _add_rms_quant(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, r_ptr, H, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=m, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + row * H + cols, mask=m, other=0.0).to(tl.float32)
    r = (x + res).to(r_ptr.dtype.element_ty)
    tl.store(r_ptr + row * H + cols, r, mask=m)
    rf = r.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(rf * rf, axis=0) / H + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    y = rf * rstd * w
    scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 127.0, 1e-10)
    q = tl.floor(y / scale + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(q_ptr + row * H + cols, q.to(tl.int8), mask=m)
    tl.store(s_ptr + row, scale)


def solve(x, residual, weight, eps):
    t, h = x.shape
    q = torch.empty((t, h), device=x.device, dtype=torch.int8)
    s = torch.empty((t, 1), device=x.device, dtype=torch.float32)
    r = torch.empty_like(x)
    block = triton.next_power_of_2(h)
    _add_rms_quant[(t,)](x, residual, weight, q, s, r, h, eps, BLOCK=block,
                         num_warps=min(16, max(4, block // 512)))
    return q, s, r
