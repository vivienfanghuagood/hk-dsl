"""A Triton flash-attention forward, after Triton's fused-attention tutorial
(one program per BLOCK_M queries of one head, an online softmax over BLOCK_N
keys), extended the way vLLM's and the tutorial's users extend it: GQA, an
optional tanh soft-cap on the scores (t6) and an optional sliding window whose
out-of-window key blocks are skipped (t7). Autotuned per (N, D)."""

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


def _configs():
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "waves_per_eu": wpe}, num_warps=w, num_stages=1)
            for bm, bn, w, wpe in [(128, 64, 4, 2), (128, 32, 4, 2), (64, 64, 4, 2), (64, 32, 4, 2),
                                   (128, 64, 8, 2), (64, 16, 4, 2)]]


@triton.autotune(configs=_configs(), key=["N", "HEAD_D", "SOFTCAP", "WINDOWED"])
@triton.jit
def _attn_fwd(Q, K, V, O, sm_scale, cap, window, N, H, GROUP,
              sqb, sqh, sqn, skb, skh, skn, sob, soh, son,
              HEAD_D: tl.constexpr, SOFTCAP: tl.constexpr, WINDOWED: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    hkv = h // GROUP
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_D)
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)
    kbase = K + b * skb + hkv * skh
    vbase = V + b * skb + hkv * skh
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_D], tl.float32)
    hi = tl.minimum((start_m + 1) * BLOCK_M, N)
    lo = 0
    if WINDOWED:
        lo = tl.maximum(0, start_m * BLOCK_M - window + 1)
        lo = (lo // BLOCK_N) * BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(kbase + cols[None, :] * skn + offs_d[:, None], mask=cols[None, :] < N, other=0.0)
        s = tl.dot(q, k) * sm_scale
        if SOFTCAP:
            t = s / cap
            s = cap * (1.0 - 2.0 / (tl.exp(2.0 * t) + 1.0))
        keep = (cols[None, :] <= offs_m[:, None]) & (cols[None, :] < N)
        if WINDOWED:
            keep = keep & (offs_m[:, None] - cols[None, :] < window)
        s = tl.where(keep, s * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.math.exp2(s - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + cols[:, None] * skn + offs_d[None, :], mask=cols[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


def attention(q, k, v, scale, cap=None, window=None):
    b, h, n, d = q.shape
    out = torch.empty_like(q)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_M"]), b * h)

    _attn_fwd[grid](q, k, v, out, scale, float(cap or 1.0), int(window or 0), n, h, h // k.shape[1],
                    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    HEAD_D=d, SOFTCAP=cap is not None, WINDOWED=window is not None)
    return out
