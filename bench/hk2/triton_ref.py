"""Triton baselines for the hk2 comparisons: the tutorial matmul and flash attention forward, autotuned over
the configurations Triton's own AMD tutorials use."""
import torch
import triton
import triton.language as tl


def _mm_configs():
    out = []
    for bm, bn, bk, w, wpe in ((256, 256, 64, 8, 2), (256, 128, 64, 8, 2), (128, 256, 64, 8, 2),
                               (128, 128, 64, 4, 2), (128, 128, 32, 4, 0), (64, 128, 64, 4, 0)):
        for st in (2, 3):
            out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8, "waves_per_eu": wpe,
                                      "matrix_instr_nonkdim": 16, "kpack": 1}, num_warps=w, num_stages=st))
    return out


@triton.autotune(configs=_mm_configs(), key=["M", "N", "K"])
@triton.jit
def _mm(a, b, c, M, N, K, sam, sak, sbn, sbk, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = GROUP_M * npn
    gid = pid // group
    first = gid * GROUP_M
    gm = min(npm - first, GROUP_M)
    pm = first + (pid % group) % gm
    pn = (pid % group) // gm
    rm = (pm * BM + tl.arange(0, BM)) % M
    rn = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    pa = a + rm[:, None] * sam + rk[None, :] * sak
    pb = b + rn[None, :] * sbn + rk[:, None] * sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(pa, mask=rk[None, :] < K - k * BK, other=0.0)
        y = tl.load(pb, mask=rk[:, None] < K - k * BK, other=0.0)
        acc += tl.dot(x, y)
        pa += BK * sak
        pb += BK * sbk
    cm = pm * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(c + cm[:, None] * scm + cn[None, :] * scn, acc.to(c.dtype.element_ty),
             mask=(cm[:, None] < M) & (cn[None, :] < N))


def matmul_nt(a, b):
    """a (M, K) @ b (N, K)^T"""
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)
    _mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def _fa_configs():
    out = []
    for bm, bn, w in ((256, 64, 8), (128, 64, 4), (128, 128, 8), (128, 32, 4), (64, 64, 4)):
        for st in (1, 2):
            out.append(triton.Config({"BM": bm, "BN": bn, "waves_per_eu": 2}, num_warps=w, num_stages=st))
    return out


@triton.jit
def _fa_inner(acc, l_i, m_i, q, kb, vb, sk, sv, m0, lo, hi, S, qk_scale,
              BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, MASK: tl.constexpr, CAUSAL: tl.constexpr):
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    for n0 in range(lo, hi, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(kb + offs_n[None, :] * sk + offs_d[:, None], mask=offs_n[None, :] < S, other=0.0)
        s = tl.dot(q, k) * qk_scale
        if MASK:
            keep = offs_n[None, :] < S
            if CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None])
            s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vb + offs_n[:, None] * sv + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.autotune(configs=_fa_configs(), key=["S", "D", "CAUSAL"])
@triton.jit
def _fa(q, k, v, o, sm_scale, S, H, G, sqb, sqh, sqs, skb, skh, sks, sob, soh, sos,
        D: tl.constexpr, CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m0 = tl.program_id(0) * BM
    h = tl.program_id(1)
    b = tl.program_id(2)
    kh = h // G
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    qt = tl.load(q + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :], mask=offs_m[:, None] < S,
                 other=0.0)
    kb = k + b * skb + kh * skh
    vb = v + b * skb + kh * skh
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    if CAUSAL:
        full = m0 // BN * BN
        hi = tl.minimum(S, m0 + BM)
    else:
        full = S // BN * BN
        hi = S
    acc, l_i, m_i = _fa_inner(acc, l_i, m_i, qt, kb, vb, sks, sks, m0, 0, full, S, qk_scale, BM, BN, D, False,
                              CAUSAL)
    acc, l_i, m_i = _fa_inner(acc, l_i, m_i, qt, kb, vb, sks, sks, m0, full, hi, S, qk_scale, BM, BN, D, True,
                              CAUSAL)
    out = acc / l_i[:, None]
    tl.store(o + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :], out.to(o.dtype.element_ty),
             mask=offs_m[:, None] < S)


def attention(q, k, v, causal=False):
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    grid = lambda m: (triton.cdiv(S, m["BM"]), H, B)
    _fa[grid](q, k, v, o, D ** -0.5, S, H, H // k.shape[1], q.stride(0), q.stride(1), q.stride(2),
              k.stride(0), k.stride(1), k.stride(2), o.stride(0), o.stride(1), o.stride(2), D=D, CAUSAL=causal)
    return o
