"""T7 (Mistral causal sliding-window GQA prefill attention), written from the docs only.

The t6 docs-only kernel (bench/dx/solutions/t6_attn_softcap/docs_only.py) without the soft-cap,
with the online softmax back (the logits are not bounded here) and the window:
- gfx950: one workgroup per (QB query rows, KV head) holding all G query heads of the GQA group
  (G x WQ warps of 32 query rows each), so a K/V block copied into LDS once serves G*QB query
  rows; K and V double-buffered by the DMA (`g.load(..., wait=False)` / `g.wait_for` / barrier);
  Sᵀ = K·Qᵀ in 32x32 accumulators, Pᵀ read in place for Oᵀ += Vᵀ·Pᵀ.
- The window: a workgroup walks only key blocks from floor((q0 - window + 1) / 64) * 64 to its last
  query, so its work is bounded by window + QB keys whatever N is -- linear in N, not quadratic.
  Masks run only on the blocks that need them: the causal `triu_` on blocks reaching past a warp's
  first query, the window's `tril_` on blocks reaching below its last query's window.
- A warp whose first block is wholly outside its window takes exp2(0) = 1 for masked scores at
  first (its running max is the -1e30 fill); the first real block raises the max by ~1e30, so the
  rescale multiplies the garbage by exp2(-1e30) = 0. Checked by the small shapes (window 100/128).
- gfx1100: examples/05_attention.py's wave32 body with GQA and the window (start block and mask).

DX log -- what had to be looked up, and where (about 40 min on top of t6):
- Everything in t6's DX log (the kernel is the same; see that file's GAP lines, which all apply
  here too: the stale "no CDNA attention" statements, no multi-warp attention example, the shipped
  schedule off limits, `count=` stacks, bounded store spelling, `rows=` semantics).
- The window's lower edge needs the other triangle: dir(hk.Tile) lists tril_ / triu_ /
  lower_fill_ / upper_fill_ / left_fill_ / right_fill_, none documented.
GAP: Tile.triu_ / tril_ (and the *_fill_ family) have no docstring and are not in docs/; the
     convention had to be probed with a scratch kernel: on a tile with row r, column c,
     `triu_(d, v)` writes v where r - c > d and `tril_(d, v)` where r - c < d (so tril_(0)
     excludes the diagonal, and the sign of d is the opposite of numpy's triu/tril k). ~10 min.
GAP: no mask_mod / window option on hk.attention (and no doc saying how to add one); a sliding
     window means writing attention from scratch, as for t6's soft-cap.
GAP: nothing in the docs on skipping work in a runtime loop whose bounds differ per workgroup
     beyond "a for over a bound every warp agrees on is fine" (language.md Limits); the start
     block computed from block_idx is such a bound, and it works.

Performance iterations (gfx950, device us; B1-N2048 / N4096 / N8192 / N16384 / B2-N8192):
- v1 = t6's kernel minus the soft-cap plus the window, 4 warps (WQ=1): 94 / 266 / 600 / 1336 / 1174.
- 8 warps (WQ=2, 64 rows a head): 80 / 281 / 571 / 1195 / 1096.
- + lazy rescale (Vec.any_above(m, 8.0), docs/language.md and help(Vec.any_above); a max up to
  2^8 stale): 75 / 279 / 574 / 1148 / 1100 in the suite (5% in a sweep). Kept.
- torch.compile (flex_attention with a block mask): 229 / 879 / 1721 / 3672 / 3592 -> 2.9-3.2x.
  Triton (the suite's, blocks skipped): 1.4-1.5x. hk.attention where it is exact (N <= 4096): 50 /
  128, so 0.67x / 0.46x of the shipped kernel's schedule; past the window this kernel beats
  hk.attention doing the full causal work at N=16384 (1798 us) and loses at 8192 (471 / 933).
- gfx1100 (the one-warp RDNA body with the window, untuned; --quick): B1-N2048 1427 us, B1-N8192
  13120 us against torch.compile 2284 / 22408 (1.6x / 1.7x).
- Not tried: hk.arch.pingpong. With two buffers the lagging half would still read buffer 0 while
  the leading half's next copy overwrites it; it needs a third buffer and the phase split of
  chapter 34, which the docs describe in prose only.
"""

import functools
import math

import torch

import hk

KIND = "docs_only"
LABEL = "from the docs: t6's LDS/DMA kernel, window blocks skipped, masks only on edge blocks"

QW = 32           # query rows per warp (gfx950)
KVB = 64          # keys per block (gfx950)
NEG_INF = -1e30
LOG2E = math.log2(math.e)


@functools.lru_cache(maxsize=None)
def _kernel_cdna(D, G, scale, window, WQ=2, THR=8.0):
    W = G * WQ
    QB = QW * WQ
    DC = 64 if D <= 128 else 128
    ND = D // DC
    sc = scale * LOG2E

    T_Q = hk.RT(hk.bf16, QW, DC, "row")
    T_K = hk.RT(hk.bf16, KVB, DC, "row")
    T_V = hk.RT(hk.bf16, KVB, DC, "col")
    T_S = hk.RT(hk.fp32, KVB, QW, "col", frag=(32, 32))
    T_P = hk.RT(hk.bf16, KVB, QW, "col", frag=(16, 32, 4))
    T_O = hk.RT(hk.fp32, DC, QW, "col", frag=(32, 32))
    VEC = hk.row_vec(T_S)
    ST_KV = hk.ST(hk.bf16, KVB, D)

    @hk.func
    def step(Kb, Vb, kv0, qw0, q_r, o_t, m, l, s, p):
        s.zero_()
        for c in range(ND):
            with hk.scope():
                kf = Kb.subtile(KVB, DC, 0, c).load(T_K)
                s += kf @ q_r[c].T                      # Sᵀ = K·Qᵀ
        s *= sc
        if kv0 + KVB > qw0 + 1:                         # a key past one of this warp's queries
            s.triu_(qw0 - kv0, NEG_INF)                 # key > query
        if kv0 + window <= qw0 + QW - 1:                # a key outside one query's window
            s.tril_(qw0 - kv0 - window + 1, NEG_INF)    # query - key >= window
        if THR:
            mb = s.max(axis=0)
            if mb.any_above(m, THR):                    # lazy: a stale max up to THR below
                m_old = m.copy()
                m.max_(mb)
                alpha = (m_old - m).exp2()
                l *= alpha
                for c in range(ND):
                    o_t[c] *= hk.as_vec(alpha, T_O)[None, :]
            s -= m[None, :]
            s.exp2_()
        else:
            m_old = m.copy()
            m.max_(s.max(axis=0))
            s -= m[None, :]
            s.exp2_()
            if m.any_ne(m_old):
                alpha = (m_old - m).exp2()
                l *= alpha
                for c in range(ND):
                    o_t[c] *= hk.as_vec(alpha, T_O)[None, :]
        l += s.sum(axis=0)
        s.as_operand(hk.bf16, into=p)                   # Pᵀ in place
        for c in range(ND):
            with hk.scope():
                vf = Vb.subtile(KVB, DC, 0, c).load(T_V)
                o_t[c] += vf.T @ p                      # Oᵀ += Vᵀ·Pᵀ

    @hk.kernel(warps=W, name=f"swa_attn_d{D}_g{G}_w{WQ}_win{window}_t{int(THR)}",
               grid=lambda p: (hk.cdiv(p.q.rows, QB), p.k.depth, p.q.batch))
    def attn(q: hk.Tensor[hk.bf16], k: hk.Tensor[hk.bf16],
             v: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        g = hk.group(W)
        Ks = hk.lds.alloc(ST_KV, count=2, name="Ks")
        Vs = hk.lds.alloc(ST_KV, count=2, name="Vs")
        b, hkv = hk.arch.block_idx.z, hk.arch.block_idx.y
        wid = hk.arch.warp_id()
        h = hkv * G + wid // WQ
        q0 = (hk.cdiv(q.rows, QB) - 1 - hk.arch.block_idx.x) * QB   # heaviest first
        qw0 = q0 + (wid % WQ) * QW
        n = k.rows
        q_r = [q[hk.elem_coord(b, h, qw0, c * DC)].load(T_Q, bounded=True) for c in range(ND)]
        o_t = [hk.zeros(T_O) for _ in range(ND)]
        m = hk.neg_infty(VEC)
        l = hk.zeros(VEC)
        s = hk.zeros(T_S)
        p = hk.zeros(T_P)
        kv_end = hk.min(n, q0 + QB)
        kv_beg = (hk.max(q0 - window + 1, 0) // KVB) * KVB     # first block any query can see
        g.load(Ks[0], k[hk.elem_coord(b, hkv, kv_beg, 0)], wait=False, rows=n - kv_beg)
        g.load(Vs[0], v[hk.elem_coord(b, hkv, kv_beg, 0)], wait=False, rows=n - kv_beg)
        for kv0 in range(kv_beg, kv_end, 2 * KVB):     # unrolled by 2: buffers by Python int
            kv1 = kv0 + KVB
            g.load(Ks[1], k[hk.elem_coord(b, hkv, kv1, 0)], wait=False, rows=hk.max(n - kv1, 0))
            g.load(Vs[1], v[hk.elem_coord(b, hkv, kv1, 0)], wait=False, rows=hk.max(n - kv1, 0))
            g.wait_for(Ks[0], Vs[0])
            hk.arch.barrier()
            step(Ks[0], Vs[0], kv0, qw0, q_r, o_t, m, l, s, p)
            hk.arch.barrier()
            kv2 = kv0 + 2 * KVB
            g.load(Ks[0], k[hk.elem_coord(b, hkv, kv2, 0)], wait=False, rows=hk.max(n - kv2, 0))
            g.load(Vs[0], v[hk.elem_coord(b, hkv, kv2, 0)], wait=False, rows=hk.max(n - kv2, 0))
            g.wait_for(Ks[1], Vs[1])
            hk.arch.barrier()
            if kv1 < kv_end:
                step(Ks[1], Vs[1], kv1, qw0, q_r, o_t, m, l, s, p)
            hk.arch.barrier()
        g.wait_for(Ks[0], Vs[0])
        for c in range(ND):
            o_t[c] /= hk.as_vec(l, T_O)[None, :]
            o[hk.elem_coord(b, h, qw0, c * DC)].store(hk.transpose(o_t[c]), bounded=True)

    return attn


@functools.lru_cache(maxsize=None)
def _kernel_rdna(D, G, scale, window):
    """examples/05_attention.py's wave32 body + GQA + the window (one warp, 16 queries)."""
    KV = 32
    d_frags, kv_frags = D // 16, KV // 16
    sc = scale * LOG2E
    T_Q = hk.RT(hk.bf16, 16, 16, "row")
    T_K = hk.RT(hk.bf16, 16, 16, "row")
    T_S = hk.RT(hk.fp32, 16, 16, "col")
    T_O = hk.RT(hk.fp32, 16, 16, "col")
    T_V = hk.RT(hk.bf16, 16, 16, "row")
    VEC = hk.row_vec(T_S)

    @hk.kernel(warps=1, name=f"swa_attn_rdna_d{D}_g{G}_win{window}",
               grid=lambda p: (hk.cdiv(p.q.rows, 16), p.q.depth, p.q.batch))
    def attn(q: hk.Tensor[hk.bf16], k: hk.Tensor[hk.bf16],
             v: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        b, h = hk.arch.block_idx.z, hk.arch.block_idx.y
        hkv = h // G
        q0 = hk.arch.block_idx.x * 16
        q_r = [q[hk.elem_coord(b, h, q0, 16 * kk)].load(T_Q, bounded=True) for kk in range(d_frags)]
        o_c = [hk.zeros(T_O) for _ in range(d_frags)]
        m = hk.neg_infty(VEC)
        l = hk.zeros(VEC)
        kv_end = hk.min(k.rows, q0 + 16)
        kv_beg = (hk.max(q0 - window + 1, 0) // KV) * KV
        for kv0 in range(kv_beg, kv_end, KV):
            with hk.scope():
                s = [hk.zeros(T_S) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            kf = k[hk.elem_coord(b, hkv, kv0 + 16 * n, 16 * kk)].load(T_K, bounded=True)
                            s[n] += kf @ q_r[kk].T
                for n in range(kv_frags):
                    s[n] *= sc
                    s[n].triu_(q0 - kv0 - 16 * n, NEG_INF)
                    s[n].tril_(q0 - kv0 - 16 * n - window + 1, NEG_INF)
                m_old = m.copy()
                for n in range(kv_frags):
                    m.max_(s[n].max(axis=0))
                for n in range(kv_frags):
                    s[n] -= m[None, :]
                    s[n].exp2_()
                if m.any_ne(m_old):
                    alpha = (m_old - m).exp2()
                    l *= alpha
                    for c in o_c:
                        c *= alpha[None, :]
                for n in range(kv_frags):
                    l += s[n].sum(axis=0)
                for n in range(kv_frags):
                    p = s[n].to(hk.bf16)
                    for gi in range(d_frags):
                        with hk.scope():
                            vf = v[hk.elem_coord(b, hkv, kv0 + 16 * n, 16 * gi)].load(T_V, bounded=True)
                            o_c[gi] += hk.transpose_sep(vf) @ p
        for i, c in enumerate(o_c):
            c /= l[None, :]
            o[hk.elem_coord(b, h, q0, 16 * i)].store(hk.transpose(c), bounded=True)

    return attn


def _wave():
    from hk.target import current_arch, get_target
    return get_target(current_arch()).wave


def _pick(q, k, scale, window, **kw):
    D, G = q.shape[-1], q.shape[1] // k.shape[1]
    if _wave() == 64:
        kw.setdefault("WQ", 2 if G <= 4 else 1)          # at most 8 warps, or G for G > 8
        return _kernel_cdna(D, G, float(scale), int(window), **kw)
    return _kernel_rdna(D, G, float(scale), int(window))


def supports(q, k, v, scale, window):
    D = q.shape[-1]
    if q.shape[1] % k.shape[1]:
        return "Hq must be a multiple of Hkv"
    if window < 1:
        return "window must be >= 1"
    if _wave() == 64:
        if D % 64 or D > 256:
            return "gfx950 kernel: head dim 64, 128, 192 or 256"
        if q.shape[1] // k.shape[1] > 16:
            return "gfx950 kernel: GQA group of at most 16 (a warp per query head)"
    elif D % 16:
        return "head dim must be a multiple of 16"
    try:
        _pick(q, k, scale, window).build()
    except hk.ResourceError as e:
        return f"does not fit the register file on this GPU: {str(e).splitlines()[1].strip()}"
    return True


def solve(q, k, v, scale, window):
    out = torch.empty_like(q)
    _pick(q, k, scale, window)(q, k, v, out)
    return out
