"""Flash attention, forward: O = softmax(Q K^T * scale) V over (batch, heads, seq, head_dim) tensors.

One source for CDNA and RDNA. Grouped-query attention: K and V may have fewer heads than Q.
"""
import torch

import hk2 as hk


@hk.func
def _step(qt, k, v, b, kh, n0, m_i, l_i, acc, scale, rows, S, BN, D, MASK, CAUSAL):
    """One block of keys: scores, online softmax, and the P @ V update."""
    kt = hk.load(k, (b, kh, n0, 0), (BN, D))
    s = hk.dot(qt, kt.T)
    if MASK:  # only the blocks at the end of the sequence or on the diagonal pay for the mask
        cols = n0 + hk.arange(BN)
        keep = cols[None, :] < S
        if CAUSAL:
            keep = keep & (cols[None, :] <= rows[:, None])
        s = hk.where(keep, s, float("-inf"))
    m_new = hk.maximum(m_i, s.max(1) * scale)  # scale > 0: scaling the max, not every score
    p = hk.exp2(s * scale - m_new[:, None])  # one fused multiply-add per score
    alpha = hk.exp2(m_i - m_new)
    l_i = l_i * alpha + p.sum(1)
    vt = hk.load(v, (b, kh, n0, 0), (BN, D))
    acc = hk.dot(p.to(vt.dtype), vt, acc * alpha[:, None])
    return m_new, l_i, acc


@hk.jit
def attn_fwd(q, k, v, o, sm_scale, BM: hk.constexpr = 128, BN: hk.constexpr = 64, D: hk.constexpr = 128,
             CAUSAL: hk.constexpr = False, GROUP: hk.constexpr = 1):
    m0 = hk.program_id(0) * BM
    b, h = hk.program_id(2), hk.program_id(1)
    kh = h // GROUP
    S = q.shape[2]
    qt = hk.load(q, (b, h, m0, 0), (BM, D))
    m_i = hk.full((BM,), float("-inf"))
    l_i = hk.zeros((BM,))
    acc = hk.zeros((BM, D))
    scale = sm_scale * 1.4426950408889634  # exp2 instead of exp
    rows = m0 + hk.arange(BM)
    if CAUSAL:  # keys left of the diagonal block need no mask
        full, hi = m0 // BN * BN, hk.minimum(S, m0 + BM)
    else:
        full, hi = S // BN * BN, S
    for n0 in range(0, full, BN):
        m_i, l_i, acc = _step(qt, k, v, b, kh, n0, m_i, l_i, acc, scale, rows, S, BN, D, False, CAUSAL)
    for n0 in range(full, hi, BN):
        m_i, l_i, acc = _step(qt, k, v, b, kh, n0, m_i, l_i, acc, scale, rows, S, BN, D, True, CAUSAL)
    hk.store(o, (b, h, m0, 0), acc / l_i[:, None])


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
              sm_scale: float = None, BM: int = None, BN: int = None, num_warps: int = None) -> torch.Tensor:
    """q: (B, H, S, D), k/v: (B, H_kv, S, D) with H % H_kv == 0; bf16 or f16; D a power of two."""
    B, H, S, D = q.shape
    Hkv = k.shape[1]
    o = torch.empty_like(q)
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    arch = torch.cuda.get_device_properties(q.device).gcnArchName
    rdna = "gfx1" in arch
    # measured defaults (bench/hk2/attention.py): MI355X 256x64 at D=128, 256x128 at D=64 (128x64 causal);
    # W7900D 128x32; all 8 warps, no spills
    if rdna:
        bm, bn = 128, 32
    elif D <= 64:
        bm, bn = (128, 64) if causal else (256, 128)
    else:
        bm, bn = 256, 64 if D <= 128 else 32
    BM, BN = BM or bm, BN or bn
    num_warps = num_warps or 8
    attn_fwd[(hk.cdiv(S, BM), H, B)](q, k, v, o, sm_scale, BM=BM, BN=BN, D=D, CAUSAL=causal, GROUP=H // Hkv,
                                     num_warps=num_warps)
    return o
