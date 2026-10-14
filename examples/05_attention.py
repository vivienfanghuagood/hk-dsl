"""Flash-attention forward: Q·Kᵀ, an online softmax, P·V, without ever
materialising the N x N scores.

One warp per 16 queries of one (batch, head), walking K and V in blocks of
32 keys. Three ideas carry it:

  * The scores are computed transposed, Sᵀ = K·Qᵀ: rows are keys, columns
    are queries. WMMA accumulates into col layout, and in Sᵀ a col-layout
    reduction over keys is the cheap axis -- `s.max(axis=0)` gives one value
    per query, a row vector.
  * The softmax is online. A running max m and sum l per query; when a block
    raises the max, the output so far is rescaled by exp2(m_old - m). That
    rescale sits behind `if m.any_ne(m_old):`, a wave-uniform branch that is
    rarely taken after the first few blocks.
  * Oᵀ += Vᵀ·Pᵀ keeps the output transposed too, and `hk.transpose` relabels
    it back to [query, d] for the store, which costs no instructions.

`hk.scope()` bounds register live ranges, so the K and V fragments do not
all stay live at once. K and V are read straight from global memory; the
shipped kernel (hk.attention, hk/ops/attn.py) shares them across 12 warps
through LDS and is several times faster. Sequence lengths have to be
multiples of 32 here.

    python3 examples/05_attention.py
    python3 examples/05_attention.py --causal
"""

import math

import _common

import hk

KV = 32                                   # keys per block
NEG_INF = -1e30
T_Q = hk.RT(hk.bf16, 16, 16, "row")       # 16 queries x 16 of d
T_K = hk.RT(hk.bf16, 16, 16, "row")       # 16 keys x 16 of d
T_S = hk.RT(hk.fp32, 16, 16, "col")       # Sᵀ: 16 keys x 16 queries
T_O = hk.RT(hk.fp32, 16, 16, "col")       # Oᵀ: 16 of d x 16 queries
T_V = hk.RT(hk.bf16, 16, 16, "row")       # 16 keys x 16 of d
VEC = hk.row_vec(T_S)                     # one value per query


def attention_kernel(head_dim: int = 128, causal: bool = False):
    d_frags, kv_frags = head_dim // 16, KV // 16
    scale = head_dim ** -0.5 * math.log2(math.e)   # exp2, so fold log2(e) in

    @hk.kernel(warps=1, name=f"attention_d{head_dim}{'_causal' if causal else ''}",
               grid=lambda p: (hk.cdiv(p.q.rows, 16), p.q.depth, p.q.batch))
    def attention(q: hk.Tensor[hk.bf16], k: hk.Tensor[hk.bf16],
                  v: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        b, h = hk.arch.block_idx.z, hk.arch.block_idx.y
        q0 = hk.arch.block_idx.x * 16
        q_r = [q[hk.elem_coord(b, h, q0, 16 * kk)].load(T_Q) for kk in range(d_frags)]
        o_c = [hk.zeros(T_O) for _ in range(d_frags)]
        m = hk.neg_infty(VEC)
        l = hk.zeros(VEC)

        # Causal: no key past this tile's last query is needed at all.
        kv_end = hk.min(k.rows, q0 + 16) if causal else k.rows
        for kv0 in range(0, kv_end, KV):
            with hk.scope():
                s = [hk.zeros(T_S) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            kf = k[hk.elem_coord(b, h, kv0 + 16 * n, 16 * kk)].load(T_K)
                            s[n] += kf @ q_r[kk].T                 # Sᵀ += K·Qᵀ
                for n in range(kv_frags):
                    s[n] *= scale
                if causal:
                    # Sᵀ row = key, column = query: mask key > query.
                    for n in range(kv_frags):
                        s[n].triu_(q0 - kv0 - 16 * n, NEG_INF)

                m_old = m.copy()
                for n in range(kv_frags):
                    m.max_(s[n].max(axis=0))
                for n in range(kv_frags):
                    s[n] -= m[None, :]
                    s[n].exp2_()
                if m.any_ne(m_old):                               # rarely, after warm-up
                    alpha = (m_old - m).exp2()
                    l *= alpha
                    for c in o_c:
                        c *= alpha[None, :]
                for n in range(kv_frags):
                    l += s[n].sum(axis=0)

                for n in range(kv_frags):
                    p = s[n].to(hk.bf16)
                    for g in range(d_frags):
                        with hk.scope():
                            vf = v[hk.elem_coord(b, h, kv0 + 16 * n, 16 * g)].load(T_V)
                            o_c[g] += hk.transpose_sep(vf) @ p     # Oᵀ += Vᵀ·Pᵀ

        for i, c in enumerate(o_c):
            c /= l[None, :]
            o[hk.elem_coord(b, h, q0, 16 * i)] = hk.transpose(c)

    return attention


def main():
    args = _common.parse(__doc__, lambda p: p.add_argument(
        "--causal", action="store_true", help="mask keys after each query"))
    kernel = attention_kernel(causal=args.causal)
    if _common.static(args, kernel):
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    b, heads, n, d = 1, 32, 4096, 128
    q, k, v = (torch.randn(b, heads, n, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    o = torch.empty_like(q)
    mode = "causal" if args.causal else "non-causal"
    print(f"attention forward, batch {b}, {heads} heads, N = {n}, d = {d}, {mode}")
    kernel(q, k, v, o)
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                          is_causal=args.causal)
    _common.check("this kernel vs fp32 torch", o, want, rel=2e-2)
    if args.no_bench:
        return
    flops = 4.0 * b * heads * n * n * d * (0.5 if args.causal else 1.0)
    best = _common.compare({
        "hk (this kernel)": lambda: kernel(q, k, v, o),
        "hk.attention": lambda: hk.attention(q, k, v, causal=args.causal),
        "torch sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal),
    }, args.rounds)
    _common.report("time", best, baseline="torch sdpa", work=flops, unit="TFLOP/s")


if __name__ == "__main__":
    main()
