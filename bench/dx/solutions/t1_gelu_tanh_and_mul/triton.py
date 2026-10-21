"""Triton baseline: one program per (row, BLOCK columns), as vLLM's and Liger's
GELU-and-mul kernels are written. tanh is spelled through exp, which every
Triton backend has (libdevice's tanh is per-vendor)."""

import torch
import triton
import triton.language as tl

KIND = "baseline"
LABEL = "Triton: row x block program, fp32 math"


@triton.jit
def _gelu_and_mul(x_ptr, o_ptr, I, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * stride_x + I + cols, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    t = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * g * (1.0 + t) * u
    tl.store(o_ptr + row * stride_o + cols, y.to(o_ptr.dtype.element_ty), mask=mask)


def solve(x):
    m, two_i = x.shape
    i = two_i // 2
    out = torch.empty((m, i), device=x.device, dtype=x.dtype)
    block = 1024
    _gelu_and_mul[(m, triton.cdiv(i, block))](x, out, i, x.stride(0), out.stride(0),
                                              BLOCK=block, num_warps=4)
    return out
