"""The Triton matmul tutorial (03-matrix-multiplication, with its AMD configs
and a few for decode-sized M), C = A @ W.T with an epilogue fused before the
store: bias + GELU(tanh) for t4, a residual add for t5. Autotuned per (M, N, K)."""

import torch
import triton
import triton.language as tl


def _configs():
    cfg = []
    for bm, bn, bk, g, w, wpe in [
        (128, 256, 32, 4, 8, 2), (256, 256, 32, 4, 8, 2), (128, 128, 64, 4, 8, 2),
        (128, 128, 32, 8, 4, 2), (64, 128, 64, 8, 4, 3), (64, 64, 64, 8, 4, 4),
        (32, 128, 128, 1, 4, 2), (16, 128, 128, 1, 4, 2), (16, 64, 256, 1, 4, 2),
        (32, 64, 128, 1, 4, 2),
    ]:
        cfg.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": g,
                                  "waves_per_eu": wpe}, num_warps=w, num_stages=2))
    return cfg


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_epilogue(a_ptr, w_ptr, c_ptr, e_ptr, M, N, K,
                   stride_am, stride_wn, stride_cm, stride_em,
                   EPILOGUE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % group) % gsize)
    pid_n = (pid % group) // gsize
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None], other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    if EPILOGUE == 1:  # + bias, GELU(tanh)
        b = tl.load(e_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        z = acc + b[None, :]
        inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        acc = 0.5 * z * (2.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0))
    elif EPILOGUE == 2:  # + residual
        r = tl.load(e_ptr + rm[:, None] * stride_em + rn[None, :], mask=cmask, other=0.0)
        acc = acc + r.to(tl.float32)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc.to(c_ptr.dtype.element_ty), mask=cmask)


def gemm(x, weight, epilogue: int, extra):
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _gemm_epilogue[grid](x, weight, out, extra, m, n, k, x.stride(0), weight.stride(0),
                         out.stride(0), extra.stride(0) if extra.dim() == 2 else 0,
                         EPILOGUE=epilogue)
    return out
