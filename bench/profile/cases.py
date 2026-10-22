"""The known cases hk.profile is validated on (docs/profile.md#validation).

Each is a target `python -m hk.profile` takes (bench/profile/cases.py:NAME)
and an inputs generator; bench/profile/run.py profiles them and writes the
reports under bench/profile/results/.

(a) gfx950, Llama-3-8B gate_up (N=28672, K=4096) at M=1024: hk.ops.matmul
    (the 256x256 pingpong, 448 workgroups on 256 CUs) against the same GEMM
    from hk.templates.gemm with a bias + GELU epilogue, 1.09x its time.
(b) gfx950, 2048x16384x4096, a shape outside routes_gemm_gfx950.json:
    route()'s rule picks dp128, 0.76x hipBLASLt; against F.linear
    (hipBLASLt) and against the pingpong forced (`pp`).
(c) gfx1100 (and gfx950): GELU-and-mul at 4096x14336, the 16x64-tile kernel
    the first page of the docs teaches against the register-strip one
    (llms.txt), streaming kernels both, the tile one far below the DRAM peak.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import hk

# ---------------------------------------------------------------- (a) the tail


def gate_up_inputs(device="cuda", m=1024, n=28672, k=4096):
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g) * 0.02
    bias = torch.randn(n, device=device, dtype=torch.bfloat16, generator=g)
    return a, w, bias


def gemm_plain(a, w, bias):
    """hk.ops.matmul: the GEMM the template is (bias ignored)."""
    return hk.ops.matmul(a, w)


def gemm_bias_gelu(a, w, bias):
    """hk.templates.gemm with gelu(acc + bias) at the store."""
    return hk.templates.gemm(a, w, epilogue=hk.templates.gemm.bias_gelu, bias=bias)


# ---------------------------------------------------------------- (b) off the table


def shape_b_inputs(device="cuda", m=2048, n=16384, k=4096):
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g) * 0.02
    return a, w


def square_inputs(device="cuda", m=4096, n=4096, k=4096):
    """The gfx1100 case: hk.ops.matmul at M = 4096 against hipBLASLt."""
    return shape_b_inputs(device, m, n, k)


def matmul(a, w):
    """hk.ops.matmul at whatever route() picks."""
    return hk.ops.matmul(a, w)


def hipblaslt(a, w):
    """torch's F.linear: hipBLASLt on ROCm."""
    return F.linear(a, w)


def matmul_pp(a, w):
    """The 256x256 pingpong forced, for the shape route() sends to dp128."""
    from hk.ops import gemm_cdna  # noqa: PLC0415

    out = torch.empty(a.shape[0], w.shape[0], device=a.device, dtype=torch.bfloat16)
    gemm_cdna.run("pp", a, w, out)
    return out


# ---------------------------------------------------------------- (c) streaming


@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def gelu_mul_tiles(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    """docs/check.md's kernel: one warp a 16x64 register tile."""
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    x = a[idx].load(hk.RT(hk.fp32, 16, 64))
    x.gelu(into=x)
    x *= b[idx].load(hk.RT(hk.fp32, 16, 64))
    o[idx] = x.to(hk.bf16)


@hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
def gelu_mul_strips(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *,
                    CW=512):
    """llms.txt's kernel: one row's CW columns on a register strip."""
    S = hk.RS(hk.fp32, 1, CW, 8)
    per_row = hk.cdiv(o.cols, CW)
    row = hk.arch.block_idx.x // per_row
    c0 = (hk.arch.block_idx.x - row * per_row) * CW
    x = a.at[row, c0].load(S, bounded=True)
    x.gelu(into=x)
    x *= b.at[row, c0].load(S, bounded=True)
    o.at[row, c0].store(x, bounded=True)


def gelu_mul_inputs(device="cuda", shape=(4096, 14336)):
    a = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    return a, torch.randn_like(a), torch.empty_like(a)
