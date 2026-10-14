"""Triton kernels for the scoreboard, written the way a Triton user would.

Not tuned for this card beyond what each kernel's own autotune does: the
point of the comparison is the DSL a kernel author would reach for, so the
matmul is the official tutorial's with its configs, the softmax the tutorial's,
and the norms one program per row with the row in a single block -- the
shape every Triton RMSNorm in the wild has.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _mm_configs():
    out = []
    for bm, bn, bk, w, s in ((128, 128, 32, 4, 2), (128, 128, 64, 8, 2),
                             (256, 128, 32, 8, 2), (128, 256, 32, 8, 2),
                             (64, 128, 64, 4, 2), (128, 64, 64, 4, 2),
                             (128, 128, 64, 4, 1)):
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                  "GROUP_M": 8}, num_warps=w, num_stages=s))
    return out


@triton.autotune(configs=_mm_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_abt(a_ptr, b_ptr, c_ptr, M, N, K, s_am, s_ak, s_bn, s_bk, s_cm,
                s_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm[:, None] * s_am + rk[None, :] * s_ak
    b = b_ptr + rn[None, :] * s_bn + rk[:, None] * s_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        ka = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K),
                     other=0.0)
        kb = tl.load(b, mask=(rn[None, :] < N) & (rk[:, None] < K - k * BLOCK_K),
                     other=0.0)
        acc += tl.dot(ka, kb)
        a += BLOCK_K * s_ak
        b += BLOCK_K * s_bk
    c = c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul_abt(a, b, out=None):
    """a @ b.T, b given as (N, K) -- F.linear's layout."""
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty(m, n, device=a.device, dtype=a.dtype) if out is None else out
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"])         # noqa: E731
                         * triton.cdiv(n, meta["BLOCK_N"]),)
    _matmul_abt[grid](a, b, out, m, n, k, a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1), out.stride(0), out.stride(1))
    return out


@triton.jit
def _rmsnorm(x_ptr, r_ptr, w_ptr, o_ptr, ro_ptr, N, eps,
             RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    if RESIDUAL:
        x += tl.load(r_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(ro_ptr + row * N + cols, x.to(ro_ptr.dtype.element_ty), mask=m)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * N + cols, (x * rstd * w).to(o_ptr.dtype.element_ty),
             mask=m)


def rms_norm(x, w, eps=1e-6, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.shape[-1]
    _rmsnorm[(x.numel() // n,)](x, x, w, out, out, n, eps, RESIDUAL=False,
                                BLOCK=triton.next_power_of_2(n),
                                num_warps=8 if n >= 2048 else 4)
    return out


def fused_add_rms_norm(x, r, w, eps=1e-6):
    n = x.shape[-1]
    _rmsnorm[(x.numel() // n,)](x, r, w, x, r, n, eps, RESIDUAL=True,
                                BLOCK=triton.next_power_of_2(n),
                                num_warps=8 if n >= 2048 else 4)
    return x, r


@triton.jit
def _softmax(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(o_ptr + row * N + cols,
             (e / tl.sum(e, axis=0)).to(o_ptr.dtype.element_ty), mask=m)


def softmax(x, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.shape[-1]
    _softmax[(x.numel() // n,)](x, out, n, BLOCK=triton.next_power_of_2(n),
                                num_warps=8 if n >= 2048 else 4)
    return out
