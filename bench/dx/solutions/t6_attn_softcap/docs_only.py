"""T6 (Gemma-2 causal GQA prefill attention with tanh logit soft-capping), written from the docs only.

gfx950: one workgroup per (64 query rows, KV head) with both query heads of the GQA group in it
(G x 2 warps of 32 query rows each), so every K/V block is copied into LDS once for 2*G*32 query
rows. K and V are double-buffered in LDS by the gfx950 DMA (`g.load(..., wait=False)`,
`g.wait_for`, a barrier) -- docs/language.md "Copies left in flight". Scores are computed
transposed in a 32x32 accumulator (Sᵀ = K·Qᵀ, `frag=(32, 32)`) and Pᵀ is read in place as the
operand of Oᵀ += Vᵀ·Pᵀ (`as_operand`) -- docs/language.md "An accumulator read as an operand".
The causal mask runs only on blocks that reach past a warp's first query; workgroups are
launched heaviest (last query rows) first.

Soft-cap with no tanh: hk.math has none (dir(hk.math): exp, exp2, log, log2, rsqrt, sqrt, ...).
    cap*tanh(x/cap) = cap - 2*cap / (exp(2x/cap) + 1)
and 1/(E+1) = rsqrt(E+1)**2 (E+1 >= 1, rsqrt(inf) = 0, so no inf/nan anywhere): one exp2, one
add, one rsqrt and two multiplies an element instead of an IEEE division. Because the capped logit
is bounded by +-cap, the softmax needs no running max: p = exp(capped logit) is in [e^-cap, e^cap]
and the sum over 16384 keys of e^50 is 8e25, far inside fp32 (and bf16, whose exponent range is
fp32's). That removes the max reduction, the rescale vote and the rescale (12% at N=4096). For
cap > 60 the kernel keeps the online softmax of examples/05_attention.py instead.

gfx1100 (wave32): examples/05_attention.py's RDNA one-warp body with the same soft-cap, the same
fixed-offset softmax and GQA. Not tuned.

DX log -- what had to be looked up, and where (about 1h45 for this task):
- The starting point: examples/05_attention.py (its wave64 body: Sᵀ = K·Qᵀ, as_operand, the
  online softmax, triu_ as the causal mask in the transposed scores).
- 32x32 accumulators and the stride-4 operand: docs/language.md "An accumulator read as an
  operand (gfx950)"; the exact T_S / T_P spellings were copied from there.
- LDS staging: examples/04_gemm.py (`hk.group(W).load`, `hk.lds.alloc`, `.subtile(r, c, i, j)`,
  barriers); the double buffer from docs/language.md "Copies left in flight" and help(hk.lds.alloc)
  (`count=` gives a stack, buffers picked by a Python int) and help(hk.Group.load) (`rows=`).
- Which math exists: dir(hk.math) and dir(hk.Tile) (runtime lookup): no tanh, no reciprocal.
- What the shipped kernel's schedule is: docs/history/34-cdna-attention.md (linked from
  language.md): 8 warps x 32 rows, 64-key blocks double-buffered by DMA, pingpong. It says nothing
  about head dim 256 (the shipped kernel takes D 64/128 only).
- Resource gate output: `k.build().kernels[i].line()` (examples/_common.py `--build`).
GAP: no tanh (nor sigmoid, nor a reciprocal) in hk.math, and no list of what hk.math has in the
     docs: language.md says only "hk.math: elementwise functions". Found by dir(hk.math). Worked
     around with exp2 + rsqrt (above). ~10 min.
GAP: no documented way to apply a score modifier to the shipped attention (hk.attention takes no
     score_mod / mask_mod), so soft-capping means writing attention from scratch.
GAP: the fast attention's source (hk/ops/attn_cdna.py) is the docs' "read it next" (examples/README.md
     "What they leave out", the 05_attention.py docstring) -- it is off limits here, so the schedule
     (pingpong phases, setprio, the split softmax of chapter 34) could only be guessed from prose.
     The docs teach no multi-warp attention: everything between the one-warp example (167 TFLOP/s
     here) and the shipped kernel (~1080) had to be designed by me.
GAP: (wrong) examples/README.md "On an MI355X ... attention has no CDNA path yet and is refused
     while tracing", and docs/history/34-cdna-attention.md "examples/05_attention.py ... is still
     skipped on gfx950", and docs/language.md Limits "On gfx950, hk.ops' GEMM and attention have
     no native form yet": all stale. examples/05_attention.py --causal runs and passes on gfx950
     (0.82 ms, 167 TFLOP/s at 1x32x4096x128), and hk.attention runs natively (design/cdna.md).
GAP: the bounded store spelling `o[idx].store(t, bounded=True)` is still not shown anywhere
     (language.md shows only bounded loads); guessed from t1's docs-only attempt.
GAP: help(hk.lds.alloc) mentions `count` but no doc says how to index the stack (Ks[0], Ks[1]
     worked, from language.md's "As[1]" in the DMA example); and nothing says a runtime loop
     must be unrolled by hand by 2 to alternate buffers (buffers must be picked by a Python int).
GAP: help(hk.Group.load) says `rows=` "ends the copy that many rows from the tile's first"; it does
     not say what a negative value does, so the copies past the end clamp with hk.max(n - kv, 0)
     (hk.max on scalars is undocumented too; it works).
GAP: Tile.triu_(diag, value) has no docstring and no doc; its meaning on a transposed score
     tile (rows = keys) was taken on trust from 05_attention.py. hk.full has no docstring either.
GAP: occupancy facts needed for tuning (8 warps cap a wave at 256 VGPRs; the gate reports
     occ=waves/SIMD) are not in the docs; learned from the gate's refusals (D=256 with 8 warps
     spills: vgpr=256, spill=47). help(k.build) / report fields are undocumented beyond the
     getting-started example line.

Performance iterations (gfx950, us, 27b-B1-N1024 / N4096 / N8192, 9b-B1-N2048):
- v1 05_attention.py's one-warp walk + soft-cap: 91 / 831 / - / 407 (torch.compile flex: 88 / 1274 / - / 551).
- v2 K/V through LDS shared by 4-8 warps, 32x32 MFMA, both GQA heads in one workgroup: 55 / 457 / 1562 / 144.
- v3 + DMA double buffer, diagonal-only mask, heaviest blocks first: 44 / 352 / 1197 / 122.
- v4 + fixed-offset softmax (no running max): 36 / 302 / 861 / 98 (4 warps; 8 warps was slower).
- tried: 32-key blocks (occupancy 3; +8% at N=4096, -2% at 8192), head-dim chunks of 32/128 (noise).
- the same structure without the soft-cap (wrong math, timing only) is 30 / 219 / 682 / 87: the
  soft-cap's two extra transcendentals cost ~30%; the rest of the gap to hk.attention (~1080
  TFLOP/s vs ~460-640 here) is the schedule (pingpong, phase split), which the docs do not teach.
- v5 (kept): with no running max the dominant key's p is rounded to bf16 for P·V while l summed
  the fp32 p, so the max error doubled (3.07e-2 vs torch.compile's 1.55e-2, i.e. one bf16 ulp at
  |out| in [4, 8); the check allows 3e-2 + 3e-2|ref|, so it passed, thinly). l is now taken on the
  MFMA from the same bf16 P (lt += onesᵀ·Pᵀ, then a max over lt's identical rows): error back to
  1.55e-2, 0-7% slower at large N, and the row sum leaves the VALU.
- Final, in the suite (device us, all six shapes; 27b N1024 / N4096 / N8192 / B4-N2048, 9b N2048 /
  N8192): 34 / 342 / 993 / 344 / 96 / 1066. torch.compile 87 / 1282 / 4086 / 1322 / 537 / 5979
  (2.5-5.6x); the suite's Triton 78 / 793 / 2524 / 798 / 211 / 2083 (2.0-2.5x); hk.attention
  WITHOUT the soft-cap (a lower bound) 29 / 137 / 501 / 154 / - / - (it has no head dim 256).
- Not tried: hk.arch.pingpong (with two buffers the lagging half would read a buffer the leading
  half's next copy overwrites: it needs three buffers and chapter 34's phase split, prose only).
- Head dim 256 runs (4 warps; 8 warps spill: a wave of an 8-warp workgroup gets 256 VGPRs and
  Oᵀ alone is 128 of them), at 350-500 TFLOP/s.
- gfx1100 (the one-warp RDNA body, untuned; --quick): 27b N1024 434 us, N4096 4826 us against
  torch.compile 1118 / 10998 (2.6x / 2.3x). Head dim 256 is declined by supports(): the one-warp
  body holds Q and Oᵀ in registers (16 + 16 16x16 tiles) and the gate refuses it (spill vgpr=51).
  Its l is still the fp32 sum (max error 2.8e-2 at |out| ~5, inside that element's 0.18 bound).
"""

import functools
import math

import torch

import hk

KIND = "docs_only"
LABEL = "from the docs: LDS-shared K/V by DMA, 32x32 MFMA, GQA packed, tanh via exp2+rsqrt, no running max"

QW = 32           # query rows per warp (gfx950)
KVB = 64          # keys per block (gfx950)
NEG_INF = -1e30
LOG2E = math.log2(math.e)


def _consts(scale, cap):
    # exp2(c1 * qk) = exp(2 * scale * qk / cap); capped logit in log2 units = c0 + c2 * rsqrt(E + 1)^2
    return 2.0 * scale / cap * LOG2E, -2.0 * cap * LOG2E, cap * LOG2E


@functools.lru_cache(maxsize=None)
def _kernel_cdna(D, G, scale, cap, WQ=2, FIXED=1, LMMA=1):
    W = G * WQ
    QB = QW * WQ
    DC = 64 if D <= 128 else 128          # head-dim chunk read out of LDS at a time
    ND = D // DC
    c1, c2, c0 = _consts(scale, cap)

    T_Q = hk.RT(hk.bf16, QW, DC, "row")
    T_K = hk.RT(hk.bf16, KVB, DC, "row")
    T_V = hk.RT(hk.bf16, KVB, DC, "col")
    T_S = hk.RT(hk.fp32, KVB, QW, "col", frag=(32, 32))
    T_P = hk.RT(hk.bf16, KVB, QW, "col", frag=(16, 32, 4))
    T_O = hk.RT(hk.fp32, DC, QW, "col", frag=(32, 32))
    VEC = hk.row_vec(T_S)
    ST_KV = hk.ST(hk.bf16, KVB, D)
    T_1 = hk.RT(hk.bf16, KVB, 32, "col")                 # ones: l = 1ᵀ·Pᵀ on the MFMA
    T_L = hk.RT(hk.fp32, 32, QW, "col", frag=(32, 32))

    @hk.func
    def step(Kb, Vb, kv0, qw0, q_r, o_t, m, l, s, p, ones, lt):
        s.zero_()
        for c in range(ND):
            with hk.scope():
                kf = Kb.subtile(KVB, DC, 0, c).load(T_K)
                s += kf @ q_r[c].T                      # Sᵀ = K·Qᵀ
        s *= c1                                         # soft-cap, in log2 units
        s.exp2_()
        s += 1.0
        s.rsqrt_()
        s *= s
        s *= c2
        if FIXED:
            s += c0
        if kv0 + KVB > qw0 + 1:                         # only blocks reaching past a query
            s.triu_(qw0 - kv0, NEG_INF)
        if FIXED:
            s.exp2_()                                   # bounded logits: no running max
        else:
            m_old = m.copy()
            m.max_(s.max(axis=0))
            s -= m[None, :]
            s.exp2_()
            if m.any_ne(m_old):
                alpha = (m_old - m).exp2()
                l *= alpha
                lt *= hk.as_vec(alpha, T_L)[None, :]
                for c in range(ND):
                    o_t[c] *= hk.as_vec(alpha, T_O)[None, :]
        if not LMMA:
            l += s.sum(axis=0)
        s.as_operand(hk.bf16, into=p)                   # Pᵀ in place
        if LMMA:
            lt += ones.T @ p                            # every row of lt is l, from the bf16 P
        for c in range(ND):
            with hk.scope():
                vf = Vb.subtile(KVB, DC, 0, c).load(T_V)
                o_t[c] += vf.T @ p                      # Oᵀ += Vᵀ·Pᵀ

    @hk.kernel(warps=W, name=f"softcap_attn_d{D}_g{G}_w{WQ}_f{FIXED}_l{LMMA}",
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
        ones = hk.full(T_1, 1.0)
        lt = hk.zeros(T_L)
        kv_end = hk.min(n, q0 + QB)
        g.load(Ks[0], k[hk.elem_coord(b, hkv, 0, 0)], wait=False, rows=n)
        g.load(Vs[0], v[hk.elem_coord(b, hkv, 0, 0)], wait=False, rows=n)
        for kv0 in range(0, kv_end, 2 * KVB):          # unrolled by 2: buffers by Python int
            kv1 = kv0 + KVB
            g.load(Ks[1], k[hk.elem_coord(b, hkv, kv1, 0)], wait=False, rows=hk.max(n - kv1, 0))
            g.load(Vs[1], v[hk.elem_coord(b, hkv, kv1, 0)], wait=False, rows=hk.max(n - kv1, 0))
            g.wait_for(Ks[0], Vs[0])
            hk.arch.barrier()
            step(Ks[0], Vs[0], kv0, qw0, q_r, o_t, m, l, s, p, ones, lt)
            hk.arch.barrier()
            kv2 = kv0 + 2 * KVB
            g.load(Ks[0], k[hk.elem_coord(b, hkv, kv2, 0)], wait=False, rows=hk.max(n - kv2, 0))
            g.load(Vs[0], v[hk.elem_coord(b, hkv, kv2, 0)], wait=False, rows=hk.max(n - kv2, 0))
            g.wait_for(Ks[1], Vs[1])
            hk.arch.barrier()
            if kv1 < kv_end:
                step(Ks[1], Vs[1], kv1, qw0, q_r, o_t, m, l, s, p, ones, lt)
            hk.arch.barrier()
        g.wait_for(Ks[0], Vs[0])
        if LMMA:
            l.zero_()
            l.max_(lt.max(axis=0))                      # every row of lt is the row sum
        for c in range(ND):
            o_t[c] /= hk.as_vec(l, T_O)[None, :]
            o[hk.elem_coord(b, h, qw0, c * DC)].store(hk.transpose(o_t[c]), bounded=True)

    return attn


@functools.lru_cache(maxsize=None)
def _kernel_rdna(D, G, scale, cap, FIXED=1):
    """examples/05_attention.py's wave32 body + soft-cap + GQA (one warp, 16 queries)."""
    KV = 32
    d_frags, kv_frags = D // 16, KV // 16
    c1, c2, c0 = _consts(scale, cap)
    T_Q = hk.RT(hk.bf16, 16, 16, "row")
    T_K = hk.RT(hk.bf16, 16, 16, "row")
    T_S = hk.RT(hk.fp32, 16, 16, "col")
    T_O = hk.RT(hk.fp32, 16, 16, "col")
    T_V = hk.RT(hk.bf16, 16, 16, "row")
    VEC = hk.row_vec(T_S)

    @hk.kernel(warps=1, name=f"softcap_attn_rdna_d{D}_g{G}_f{FIXED}",
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
        for kv0 in range(0, kv_end, KV):
            with hk.scope():
                s = [hk.zeros(T_S) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            kf = k[hk.elem_coord(b, hkv, kv0 + 16 * n, 16 * kk)].load(T_K, bounded=True)
                            s[n] += kf @ q_r[kk].T
                for n in range(kv_frags):
                    s[n] *= c1
                    s[n].exp2_()
                    s[n] += 1.0
                    s[n].rsqrt_()
                    s[n] *= s[n]
                    s[n] *= c2
                    if FIXED:
                        s[n] += c0
                    s[n].triu_(q0 - kv0 - 16 * n, NEG_INF)
                if FIXED:
                    for n in range(kv_frags):
                        s[n].exp2_()
                else:
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


def _pick(q, k, scale, cap, **kw):
    D, G = q.shape[-1], q.shape[1] // k.shape[1]
    fixed = int(cap <= 60.0)          # e^cap * N must stay far inside fp32
    if _wave() == 64:
        return _kernel_cdna(D, G, float(scale), float(cap), FIXED=fixed, **kw)
    return _kernel_rdna(D, G, float(scale), float(cap), FIXED=fixed)


def supports(q, k, v, scale, cap):
    D = q.shape[-1]
    if q.shape[1] % k.shape[1]:
        return "Hq must be a multiple of Hkv"
    if _wave() == 64:
        if D % 64 or D > 256:
            return "gfx950 kernel: head dim 64, 128, 192 or 256"
        if (q.shape[1] // k.shape[1]) * 2 > 16:
            return "gfx950 kernel: GQA group of at most 8 (2 warps per query head)"
    elif D % 16:
        return "head dim must be a multiple of 16"
    try:
        _pick(q, k, scale, cap).build()
    except hk.ResourceError as e:
        return f"does not fit the register file on this GPU: {str(e).splitlines()[1].strip()}"
    return True


def solve(q, k, v, scale, cap):
    out = torch.empty_like(q)
    _pick(q, k, scale, cap)(q, k, v, out)
    return out
