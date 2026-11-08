"""Decode attention over vLLM's own paged cache (TRITON_ATTN), split along the KV axis.

hk.ops.paged is the decode for hk's backend, over a cache laid out for it. This one reads the
cache vLLM's TRITON_ATTN backend keeps, in place -- the layouts hk.ops.paged_prefill reads with
layout="flash" -- so a decode step stays on vLLM's backend and only its attention kernel
changes:

* "planes": (blocks, 2, page, kv_heads * head_dim), K then V a block;
* "tokens": a head's K and V side by side, token-major, (blocks, 1, page, kv_heads * 2 * head_dim);
* "heads": the same head-major, (blocks, kv_heads, page, 2 * head_dim);
* "rocm": vLLM ROCM_ATTN's own cache, K x-split -- passed as the strided view (blocks, kv_heads,
  page, head_dim / 8, 8) -- and V as per-page V^T (blocks, kv_heads, head_dim, page), the extra
  last argument `vt`. A strided tensor is read by staging, so K goes through LDS: each step the
  workgroup stages every walker's block; V^T fragments are the PV operand as they are. Pages of
  16 tokens, head_dim up to 128.

Pages are any multiple of 16 tokens (hybrid models' 544, 784 and 2064 included): the walk steps
a KV block of 32 keys, in bands of gcd(page, 32) rows, each band through the block table on its
own. With pages of whole blocks (a hybrid model's), a page's rows past the sequence's end can
hold another layer's state, NaN included, so V is read as zero there and K's scores are masked.

**Why split.** Decode attention reads the whole KV cache and does one query row a head of maths
on it. vLLM's Triton kernel cuts each sequence into 16 segments, so a batch of one on a model with
2 kv heads is 32 programs, a fraction of the card's bandwidth (MiniCPM5-2B, 16k tokens: ~200 GB/s
of the W7900D's ~820). Here a workgroup is a (split, kv head, request), and the split count is
planned from the batch alone (`plan_splits`): a split's page range is arithmetic on the device's
sequence length, so a split count fixed when a CUDA graph was captured is right for every replay.

**A workgroup.** `warps` warps, each walking every `warps`-th block of the split (a warp pair
each, with dsplit). The query tile is the request's token's `group` query heads (GQA-packed,
padded to 16 rows), staged once into LDS and read from there for each WMMA: held in registers,
its 64 VGPRs are what keeps a block's K loads from being in flight together. K and V come from
global memory straight into registers, V transposed there. At the end the warps' (O, m, l) are
folded pairwise through LDS; with one split the workgroup writes the answer, with more it writes
its normalised partial and (m, l), and `merge_kernel` combines a request's splits.

**head_dim 256 (dsplit).** A d256 row's O alone is 128 VGPRs, and with a block of K or V in
flight it spills. So, as hk.ops.paged_prefill does, a warp pair splits head_dim: each warp holds
its half of Q's columns and of O, computes its half's partial S, and the pair adds the two halves
through LDS (one barrier a block, the workgroup's warps in step); both then hold the same S, the
same softmax, and accumulate their halves of O.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp16 as fp16_t, fp32, i32
from . import _arch
from .paged_prefill import _kv_at

#: Keys a step of the walk takes (two 16-row WMMA fragments).
KV_BLOCK = 32
#: Query rows a workgroup holds: the GQA group, padded.
Q_TILE = 16
NEG_INF = float("-inf")


def walk_kernel(name: str, *, head_dim: int = 128, group: int = 8, page: int = 16,
                kv_order: str = "planes", splits: int = 1, scale: Optional[float] = None,
                warps: int = 4, v_both: bool = False, max_vgprs: int = 256,
                min_occupancy: int = 1, dtype=bf16):
    """The split walk. Tensors: `qh` (1, tokens, heads, head_dim) -- the engine's query rows,
    which may be row-strided (a fused qkv's) -- and `out` (tokens, kv_heads, group, head_dim),
    the engine's contiguous output viewed by kv head; `kv` the cache as its `kv_order` view,
    `block_table` (requests, pages), `seq_lens` (requests,), `q_start` (requests + 1,) -- the
    batch's query_start_loc: a request has one query token, or none (a CUDA graph's padding),
    which then writes nothing -- and the split scratch `o_part` (requests * splits, kv_heads,
    group, head_dim) bf16 and `ml_part` (requests * splits, kv_heads, 2, 16) fp32, unused with
    one split. head_dim above 128 is split across warp pairs (`warps` even)."""
    if head_dim % 16 or page % 16 or not 1 <= group <= Q_TILE:
        raise ValueError(f"head_dim {head_dim} and page {page}: multiples of 16; group "
                         f"{group}: 1..{Q_TILE}")
    if kv_order not in ("planes", "tokens", "heads", "rocm"):
        raise ValueError(f"kv_order={kv_order!r}")
    rocm = kv_order == "rocm"
    if rocm and (head_dim > 128 or page != 16):
        raise ValueError(f"kv_order='rocm': head_dim up to 128 and pages of 16 tokens "
                         f"(ROCM_ATTN's), not {head_dim} / {page}")
    dsplit = head_dim > 128
    if warps & (warps - 1) or (dsplit and warps < 2):
        raise ValueError(f"warps={warps}: a power of two (at least 2 above head_dim 128)")
    dh = head_dim // 2 if dsplit else head_dim       # a warp's share of head_dim
    if dh > 128 or dh % 16:
        raise ValueError(f"head_dim {head_dim}: up to 256")
    halves = 2 if dsplit else 1
    walkers = warps // halves                        # warps (or pairs) walking blocks apart
    d_frags, kv_frags = dh // 16, KV_BLOCK // 16
    band = math.gcd(page, KV_BLOCK)
    ppb = KV_BLOCK // band
    # a hybrid model's pages (whole blocks): rows past the end may be NaN
    masked = page >= KV_BLOCK and page > 16
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)
    # LDS slots the fold uses at once, (d_frags + 2) fp32 tiles each
    slots = max(1, min(walkers // 2, 2 if d_frags <= 8 and not dsplit else 1)) * halves

    dt = dtype           # the activations' and cache's: bf16, or fp16 for fp16-served models
    ty_q = hk.RT(dt, 16, 16, "row")
    ty_k = hk.RT(dt, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")
    ty_v = hk.RT(dt, 16, 16, "row")
    ty_vec = hk.row_vec(ty_s)

    kv_ty = hk.Tensor[dt, "strided"] if rocm else hk.Tensor[dt]

    def body(qh: hk.Tensor[bf16, "strided"], kv: kv_ty, block_table: hk.Tensor[i32],
             seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32], vt: hk.Tensor[bf16] = None):
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z

        tok = q_start[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        q_len = q_start[hk.elem_coord(0, 0, 0, req + 1)].load_scalar(invariant=True) - tok
        rows = group * hk.min(q_len, 1)          # 0 for a graph's padding: nothing is written
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_blk = hk.cdiv(seq_len, KV_BLOCK)
        n_pg = hk.max(hk.cdiv(seq_len, page), 1)
        per = hk.max(hk.cdiv(n_blk, splits), 1)
        lo = split * per
        hi = hk.min(n_blk, lo + per)

        g = hk.group(warps)
        q_s = hk.lds.alloc(hk.ST(dt, 16, head_dim), name="q_s")
        if walkers > 1:
            # the fold's slots: a giving warp's O, and its m and l as tiles
            xo = hk.lds.alloc(hk.ST(fp32, 16, Q_TILE, slots * d_frags), name="xo")
            xm = hk.lds.alloc(hk.ST(fp32, 16, Q_TILE, max(slots, 2)), name="xm")
            xl = hk.lds.alloc(hk.ST(fp32, 16, Q_TILE, max(slots, 2)), name="xl")
        if dsplit:
            # each warp's partial S, double-buffered by block parity: one barrier a block
            xs = hk.lds.alloc(hk.ST(fp32, 16, Q_TILE, 2 * warps * kv_frags), name="xs")
        wid = hk.arch.warp_id()
        half = wid % 2 if dsplit else 0
        walker = wid // 2 if dsplit else wid
        if rocm:
            # ROCM_ATTN's x-split K goes through LDS (a strided tensor is read by g.stage): each
            # step the workgroup stages every walker's block, a 16-token page a slot
            # exactly the walkers' blocks: LDS is what bounds the workgroups a WGP holds here
            k_s = hk.lds.alloc(hk.ST(dt, KV_BLOCK, head_dim, walkers), name="k_s")
            k_w = [k_s[w] if walkers > 1 else k_s for w in range(walkers)]
            k_pg = [[k_w[w].subtile(16, head_dim, p, 0) for p in range(ppb)]
                    for w in range(walkers)]
            # one 16-token page staged at a time: a step's whole block in registers at once
            # (64 VGPRs a walker at one warp) spills under some compilers (ROCm 7.2.4's)
            buf_k = g.stage_buffer(k_pg[0][0], depth=walkers * ppb, name="buf_k")
            gr_k = (k_s[walker] if walkers > 1 else k_s).granules(name="gr_k")
            # V^T too: register loads of its fragments spill under ROCm 7.2.4's compiler
            vt_s = hk.lds.alloc(hk.ST(dt, head_dim, KV_BLOCK, walkers), name="vt_s")
            v_w = [vt_s[w] if walkers > 1 else vt_s for w in range(walkers)]
            v_pg = [[v_w[w].subtile(head_dim, 16, 0, p) for p in range(ppb)]
                    for w in range(walkers)]
            buf_v = g.stage_buffer(v_pg[0][0], depth=walkers * ppb, name="buf_v")
            gr_vt = (vt_s[walker] if walkers > 1 else vt_s).granules(name="gr_vt")

            def stage_pages(st, buf, src):
                # every walker's block of step `st`, a page a slot: loads issued, not waited on
                for w in range(walkers):
                    bw = hk.max(hk.min(lo + w + st * walkers, n_blk - 1), 0)
                    for p in range(ppb):
                        g.stage(buf, src[hk.elem_coord(page_at(bw, p), kv_head, 0, 0)],
                                slot=w * ppb + p)

            def commit_pages(dst, buf):
                hk.arch.vm_wait(0)
                for w in range(walkers):
                    for p in range(ppb):
                        g.commit(dst[w][p], buf, slot=w * ppb + p)
            gr_q = q_s.granules(name="gr_q")
            k_dep = k_col = v_dep = v_col = 0
        else:
            k_dep, k_col = _kv_at(kv_order, head_dim, kv_head, False)
            v_dep, v_col = _kv_at(kv_order, head_dim, kv_head, True)
        if dsplit:
            k_col = k_col + half * dh
            v_col = v_col + half * dh

        def page_at(b, p):
            # band p of block b: its page (clamped into the request's: past the end the
            # prefetch rereads the last)
            i = (b * (KV_BLOCK // page) + p if page <= KV_BLOCK
                 else (b * KV_BLOCK + p * band) // page)
            return block_table[hk.elem_coord(0, 0, req, hk.min(i, n_pg - 1))].load_scalar(
                invariant=True)

        # the first block's page indices, in flight with Q's staging below
        first = hk.max(hk.min(lo + walker if walkers > 1 else lo, n_blk - 1), 0)
        nx0 = page_at(first, 0)
        nx1 = page_at(first, 1) if ppb > 1 else nx0
        # the token's query heads of this kv head, staged through LDS (q is a strided view, a
        # fused qkv's rows) and read from there at each WMMA. Rows past the group are the next
        # heads' (or zeros past the tensor): their columns of S are computed, never stored.
        buf_q = g.stage_buffer(q_s, name="buf_q")
        g.stage(buf_q, qh[hk.elem_coord(0, tok, kv_head * group, 0)])
        hk.arch.vm_wait(0)
        g.commit(q_s, buf_q)
        hk.arch.barrier(drain=True)

        def q_frag(kk):
            return q_s.subtile(16, 16, 0, kk + half * d_frags if dsplit else kk).load(ty_q)

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        if dsplit or rocm:
            # finite: a pair's step past its blocks is all masked, and -inf - -inf is NaN
            m_run = hk.zeros(ty_vec)
            m_run -= 1e30
        else:
            m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        def row_in(b, n):
            # fragment n of block b: its band, and its first row in that band's page
            if page <= KV_BLOCK:
                return (16 * n) // band, (16 * n) % page
            return (16 * n) // band, (b * KV_BLOCK + 16 * n) % page

        def k_dot(pg, b, n, acc):
            if rocm:
                # K's and Q's fragments both from LDS, two of each in flight, waited on by name
                for k0 in range(0, d_frags, 2):
                    kk_ = range(k0, min(k0 + 2, d_frags))
                    ks = [gr_k.load_frag(n, kk, ty_k) for kk in kk_]
                    qs = [gr_q.load_frag(0, kk, ty_q) for kk in kk_]
                    hk.lds.wait_for(*ks, *qs)
                    for i in range(len(ks)):
                        acc += ks[i] @ qs[i].T
                return acc
            ks = [kv[hk.elem_coord(pg[row_in(b, n)[0]], k_dep, row_in(b, n)[1],
                                   k_col + 16 * kk)].load(ty_k, bounded=True)
                  for kk in range(d_frags)]
            for kk in range(d_frags):
                acc += ks[kk] @ q_frag(kk).T
            return acc

        def softmax_pv(pg, b, s_f):
            nonlocal m_run, l_run

            def v_row(n):
                if rocm:
                    return None          # V^T is in LDS, read fragment by fragment below
                return [kv[hk.elem_coord(pg[row_in(b, n)[0]], v_dep, row_in(b, n)[1],
                                         v_col + 16 * g0)].load(
                    ty_v, bounded=(seq_len - (b * KV_BLOCK + 16 * n)) if masked else True)
                    for g0 in range(d_frags)]
            # V in flight during the softmax: the block's first 16-row band, or with `v_both`
            # both (60 VGPRs more, occupancy 8 -> 5; `plan` asks for it at small batches, where
            # MiniCPM5-2B at one request and 8k tokens took 25.4-25.7 us in two processes
            # against 26.0 in one and 30.4 in the other with the second band loaded after P).
            # It does not build everywhere: unsplit, it spills; with bounded V (hybrid pages) the
            # LDS-read hazard check refuses it.
            vs = v_row(0)
            vs1 = v_row(1) if v_both and kv_frags > 1 and not rocm else None

            for n in range(kv_frags):
                s_f[n] *= scale_l2e
            # keys at or past the sequence's end, and (dsplit) a step past the split's
            valid = seq_len - b * KV_BLOCK
            if dsplit or rocm:
                valid = valid * hk.min(hk.max(hi - b, 0), 1)
            if valid < KV_BLOCK:
                for n in range(kv_frags):
                    s_f[n].lower_fill_(valid - 16 * n, NEG_INF)
            m_old = m_run.copy()
            for n in range(kv_frags):
                m_run.max_(s_f[n].max(axis=0))
            for n in range(kv_frags):
                s_f[n] -= m_run
                s_f[n].exp2_()
            if m_run.any_ne(m_old):
                alpha = (m_old - m_run).exp2()
                l_run *= alpha
                for c in o_c:
                    c *= alpha
            for n in range(kv_frags):
                l_run += s_f[n].sum(axis=0)
            if rocm:
                return          # PV after V^T's commit (a workgroup op): pv_rocm
            for n in range(kv_frags):
                p_cvt = s_f[n].to(dt)
                if n:
                    vs = vs1 if vs1 is not None else v_row(n)
                for g0 in range(d_frags):
                    o_c[g0] += hk.transpose_sep(vs[g0]) @ p_cvt

        # walker w takes the split's blocks lo + w, lo + w + walkers, ... With dsplit (or rocm)
        # every walker takes the same number of steps (the barriers), a step past its blocks
        # masked whole.
        if rocm:
            steps = hk.cdiv(hk.max(hi - lo, 0), walkers)

            def pv_rocm(s_f):
                for n in range(kv_frags):
                    p_cvt = s_f[n].to(dt)
                    # V^T fragments (16 of head_dim by the page's 16 tokens), two in flight
                    for g0 in range(0, d_frags, 2):
                        gs = list(range(g0, min(g0 + 2, d_frags)))
                        vf = [gr_vt.load_frag(gg, n, ty_v) for gg in gs]
                        hk.lds.wait_for(*vf)
                        for i in range(len(gs)):
                            o_c[gs[i]] += vf[i] @ p_cvt

            # The staged block is not prefetched a step ahead: held across the step's maths its
            # 64 VGPRs spill. The other workgroups on the WGP cover the load.
            for bi in range(steps):
                b = lo + walker + bi * walkers
                bc = hk.max(hk.min(b, n_blk - 1), 0)
                pg = (page_at(bc, 0), page_at(bc, 1))
                # K, then V^T, each staged and committed before the maths: a stage buffer held
                # across it (64 VGPRs) spills under one compiler or the other
                hk.arch.barrier()
                stage_pages(bi, buf_k, kv)
                commit_pages(k_pg, buf_k)
                stage_pages(bi, buf_v, vt)
                commit_pages(v_pg, buf_v)
                hk.arch.barrier(drain=True)
                with hk.scope():
                    s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                    for n in range(kv_frags):
                        s_f[n] = k_dot(pg, b, n, s_f[n])
                    softmax_pv(pg, b, s_f)
                    pv_rocm(s_f)
        elif dsplit:
            steps = hk.cdiv(hk.max(hi - lo, 0), walkers)
        elif walkers > 1:
            steps = hk.cdiv(hk.max(hi - lo - walker, 0), walkers)
        else:
            steps = hk.max(hi - lo, 0)
        for bi in range(0 if rocm else steps):
            b = lo + walker + bi * walkers if walkers > 1 else lo + bi
            pg0 = nx0
            nx0 = page_at(hk.min(b + walkers, n_blk - 1), 0)
            pg1 = nx1
            if ppb > 1:
                nx1 = page_at(hk.min(b + walkers, n_blk - 1), 1)
            pg = (pg0, pg1)
            if dsplit:
                # the pair's two halves of S, added through LDS (the barrier outside a scope:
                # it is the workgroup's)
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                with hk.scope():
                    for n in range(kv_frags):
                        s_f[n] = k_dot(pg, b, n, s_f[n])
                par = (bi % 2) * (warps * kv_frags)
                for n in range(kv_frags):
                    xs[par + wid * kv_frags + n].store(s_f[n])
                hk.arch.barrier(drain=True)
                mate = wid + 1 - 2 * half
                for n in range(kv_frags):
                    s_f[n] += xs[par + mate * kv_frags + n].load(ty_s)
                with hk.scope():
                    softmax_pv(pg, b, s_f)
            else:
                with hk.scope():
                    s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                    for n in range(kv_frags):
                        s_f[n] = k_dot(pg, b, n, s_f[n])
                    softmax_pv(pg, b, s_f)

        if walkers > 1:
            # fold the walkers' (O, m, l) pairwise through LDS (each half with its own half):
            # in a round, walkers [h, 2h) give to [0, h), slots / halves at a time
            floor = hk.zeros(ty_vec)
            floor -= 1e30
            per_round = slots // halves
            h = walkers
            while h > 1:
                h //= 2
                for j0 in range(0, h, per_round):
                    js = min(per_round, h - j0)
                    sl = (walker - h - j0) * halves + half      # a giver's slot
                    if walker >= h + j0:
                        if walker < h + j0 + js:
                            for i in range(d_frags):
                                xo[sl * d_frags + i].store(o_c[i])
                            mt = hk.zeros(ty_s)
                            mt += m_run
                            xm[sl].store(mt)
                            lt = hk.zeros(ty_s)
                            lt += l_run
                            xl[sl].store(lt)
                    hk.arch.barrier(drain=True)
                    sr = (walker - j0) * halves + half           # a receiver's
                    if walker >= j0:
                        if walker < j0 + js:
                            mb = xm[sr].load(ty_s).max(axis=0)
                            lb = xl[sr].load(ty_s).max(axis=0)
                            mx = m_run.copy()
                            mx.max_(mb)
                            mx.max_(floor)
                            wa = (m_run - mx).exp2()
                            wb = (mb - mx).exp2()
                            m_run.max_(mb)
                            l_run *= wa
                            lb *= wb
                            l_run += lb
                            for i in range(d_frags):
                                ob = xo[sr * d_frags + i].load(ty_o)
                                ob *= wb
                                o_c[i] *= wa
                                o_c[i] += ob
                    hk.arch.barrier()
            writer = 1 - hk.min(walker, 1)         # walker 0 (a pair, with dsplit) writes
        else:
            writer = 1

        col0 = half * dh if dsplit else 0
        l_safe = l_run.copy()
        l_safe += 1e-30
        if splits == 1:
            for i, c in enumerate(o_c):
                c /= l_safe
                out[hk.elem_coord(tok, kv_head, 0, col0 + 16 * i)].store(
                    hk.transpose(c), bounded=rows * writer)
            return
        # normalised, bf16, the group's rows; an empty split has l = 0 and O = 0
        row = req * splits + split
        if writer > 0:
            for i, c in enumerate(o_c):
                c /= l_safe
                o_part[hk.elem_coord(row, kv_head, 0, col0 + 16 * i)].store(hk.transpose(c),
                                                                            bounded=True)
            if half == 0:
                ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m_run
                ml_part[hk.elem_coord(row, kv_head, 1, 0)] = l_run

    body.__annotations__["kv"] = kv_ty
    for nm in ("out", "o_part", "vt"):
        body.__annotations__[nm] = hk.Tensor[dt]
    body.__annotations__["qh"] = hk.Tensor[dt, "strided"]
    if not rocm:
        import inspect  # noqa: PLC0415

        sig = inspect.signature(body)
        body.__signature__ = sig.replace(parameters=list(sig.parameters.values())[:8])
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (splits, p.out.depth, p.seq_lens.cols),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


def merge_kernel_def(name: str, *, head_dim: int = 128, group: int = 8, splits: int = 8,
                     step: int = 8, dtype=bf16):
    """Combine a request's `splits` partials into `out` at the request's token; a request with
    no query (a graph's padding) writes nothing. One workgroup a (16 columns of head_dim, kv
    head, request), a warp a group of `step` splits: every warp reads its splits' partials in one
    round -- a launch that is latency, not bandwidth -- and the groups meet through LDS. Small
    workgroups, spread over the card: one of 32 warps a (kv head, request) took 4 us here, as
    long as the walk's own tail at one request."""
    d_frags = head_dim // 16
    # split groups: `step` splits a warp, as many groups as 32 warps hold
    groups = max(1, min(-(-splits // min(step, splits)), 32))
    step = -(-splits // groups)
    warps = groups                     # a workgroup a 16-column fragment, a warp a split group
    ty_ot = hk.RT(fp32, Q_TILE, 16, "row")
    ty_ob = hk.RT(dtype, Q_TILE, 16, "row")
    cvec = hk.col_vec(ty_ot)

    def body(seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
        # seq_lens: the grid's extent (a request a workgroup); its values are not read
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        if groups > 1:
            xa = hk.lds.alloc(hk.ST(fp32, Q_TILE, 16, max(groups, 2)), name="xa")
            xm = hk.lds.alloc(hk.ST(fp32, Q_TILE, 16, max(groups, 2)), name="xm")
            xl = hk.lds.alloc(hk.ST(fp32, Q_TILE, 16, max(groups, 2)), name="xl")
        cf = hk.arch.block_idx.x
        sg = hk.arch.warp_id() if groups > 1 else 0
        tok = q_start[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        q_len = q_start[hk.elem_coord(0, 0, 0, req + 1)].load_scalar(invariant=True) - tok
        s1 = sg * step
        # this warp's splits, read at once and from the start: every split of the walk wrote
        # its partial, an empty one weightless (l = 0), so nothing here waits on the request's
        # length. Past the last split (a ragged last group): the last reread, weighed zero.
        rows = [req * splits + min(s1 + j, splits - 1) if isinstance(s1, int)
                else req * splits + hk.min(s1 + j, splits - 1) for j in range(step)]
        ws = [ml_part[hk.elem_coord(r, kv_head, 0, 0)].load(cvec) for r in rows]
        ls = [ml_part[hk.elem_coord(r, kv_head, 1, 0)].load(cvec) for r in rows]
        os_ = [o_part[hk.elem_coord(r, kv_head, 0, 16 * cf)].load(ty_ob, bounded=True)
               for r in rows]
        def emit(a, l):
            # a request with no context (a graph's padding) has l = 0: zero, not 0/0
            l += 1e-30
            a /= l
            out[hk.elem_coord(tok, kv_head, 0, 16 * cf)].store(
                a, bounded=group * hk.min(q_len, 1))

        m_run = hk.zeros(cvec)
        m_run -= 1e30
        for w in ws:
            m_run.max_(w)
        acc = hk.zeros(ty_ot)
        l_acc = hk.zeros(cvec)
        for j, (w, l_s, o_b) in enumerate(zip(ws, ls, os_)):
            w -= m_run
            w.exp2_()
            w *= l_s
            w *= hk.min(hk.max(splits - s1 - j, 0), 1)     # 0 for a reread of the last
            l_acc += w
            o_s = o_b.to(fp32)
            o_s *= w
            acc += o_s
        if groups > 1:
            xa[sg].store(acc)
            mt = hk.zeros(ty_ot)
            mt += m_run
            xm[sg].store(mt)
            lt = hk.zeros(ty_ot)
            lt += l_acc
            xl[sg].store(lt)
            hk.arch.barrier(drain=True)
            if sg == 0:
                ms = [xm[g].load(ty_ot).max(axis=1) for g in range(groups)]
                big = ms[0].copy()
                for g in range(1, groups):
                    big.max_(ms[g])
                tot = hk.zeros(ty_ot)
                l_tot = hk.zeros(cvec)
                for g in range(groups):
                    wg = ms[g] - big
                    wg.exp2_()
                    lg = xl[g].load(ty_ot).max(axis=1)
                    lg *= wg
                    l_tot += lg
                    ag = xa[g].load(ty_ot)
                    ag *= wg
                    tot += ag
                emit(tot, l_tot)
        else:
            emit(acc, l_acc)

    body.__annotations__["out"] = hk.Tensor[dtype]
    body.__annotations__["o_part"] = hk.Tensor[dtype]
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (d_frags, p.out.depth, p.seq_lens.cols))


_WALK: Dict[tuple, object] = {}
_MERGE: Dict[tuple, object] = {}


def _tag(scale):
    return "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)


def walk_for(head_dim: int, group: int, page: int, kv_order: str, splits: int,
             scale: Optional[float] = None, warps: int = 4, v_both: bool = False,
             fp16: bool = False):
    """The walk for one (shape, layout, split count, bf16 or fp16), built on first use."""
    _arch.require("hk.ops.paged_decode")
    key = (head_dim, group, page, kv_order, splits, scale, warps, v_both, fp16)
    if key not in _WALK:
        _WALK[key] = walk_kernel(
            f"paged_decode_flash_d{head_dim}_g{group}_p{page}_{kv_order}_s{splits}_w{warps}"
            f"{'_vb' if v_both else ''}{'_f16' if fp16 else ''}{_tag(scale)}", head_dim=head_dim,
            group=group, page=page, kv_order=kv_order, splits=splits, scale=scale, warps=warps,
            v_both=v_both, dtype=fp16_t if fp16 else bf16)
    return _WALK[key]


def merge_for(head_dim: int, group: int, splits: int, step: int = 8, fp16: bool = False):
    """The merge of `splits` partials, built on first use."""
    _arch.require("hk.ops.paged_decode")
    key = (head_dim, group, splits, step, fp16)
    if key not in _MERGE:
        _MERGE[key] = merge_kernel_def(
            f"paged_decode_merge_d{head_dim}_g{group}_s{splits}_m{step}{'_f16' if fp16 else ''}",
            head_dim=head_dim, group=group, splits=splits, step=step,
            dtype=fp16_t if fp16 else bf16)
    return _MERGE[key]


#: workgroups to aim for, and the most splits a request takes
TARGET_WORKGROUPS = 64
MAX_SPLITS = 32


def plan_splits(n_reqs: int, n_kv_heads: int, max_blocks: Optional[int] = None) -> int:
    """Splits for a batch: about TARGET_WORKGROUPS (request, kv head, split) workgroups,
    from the batch alone (a CUDA graph's replays see other lengths), capped by MAX_SPLITS and,
    when known, the longest request's KV blocks."""
    have = max(1, n_reqs * n_kv_heads)
    want = max(1, min(-(-TARGET_WORKGROUPS // have), MAX_SPLITS))
    if max_blocks is not None:
        want = max(1, min(want, max_blocks))
    return want


#: The splits a small batch at head_dim 128 takes, as many as the GPU has WGPs, in this range
SMALL_SPLITS = (16, 64)
_WGPS: dict = {}


def device_wgps(device) -> int:
    """The WGPs of `device` (torch's multi_processor_count on RDNA: 48 on a W7900D, 32 on an
    R9700), read once a device."""
    import torch  # noqa: PLC0415

    i = torch.device(device).index
    i = torch.cuda.current_device() if i is None else i
    if i not in _WGPS:
        _WGPS[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _WGPS[i]


def plan(n_reqs: int, n_kv_heads: int, head_dim: int, page: int = 16,
         wgps: Optional[int] = None):
    """(splits, warps, v_both) for a decode batch. At head_dim 128 and up to 4 (request, kv
    head) pairs: 2 warps a workgroup, both of a block's V bands in flight on pages of 16 tokens,
    and as many splits as the GPU has WGPs (`wgps`; MAX_SPLITS when not given), within
    SMALL_SPLITS. On a W7900D (48 WGPs), MiniCPM5-2B at one request, 3 processes
    (tools/hk-bench/attention_bench.py): 48 splits 17.7-18.3 / 25.0-25.3 / 36.1-36.5 /
    57.0-57.4 us at 4k / 8k / 16k / 32k tokens, against 17.5-19.3 / 25.4-29.3 / 36.8-40.5 /
    59.6-62.8 for 32 -- whose time depended on the process (where 64 workgroups landed); at two
    requests 2-4% faster. From the batch and the device alone, so a CUDA graph's replays keep it.
    Otherwise `plan_splits`, 4 warps."""
    pairs = max(1, n_reqs * n_kv_heads)
    if head_dim <= 128 and pairs <= 4:
        lo, hi = SMALL_SPLITS
        return (MAX_SPLITS if wgps is None else min(max(wgps, lo), hi)), 2, page == 16
    return plan_splits(n_reqs, n_kv_heads), 4, False


#: kv_order="rocm" (ROCM_ATTN's own cache): one-warp workgroups, this many a WGP. Measured on a
#: W7900D (48 WGPs), MiniCPM5-2B, B1-B8 x 2k-128k against ROCM_ATTN's paged decode (ll4mi), one
#: process (tools/hk-bench/attention_bench.py): 4 a WGP -- 192 workgroups, B1 96 splits, B2 48,
#: B4 24 -- is 0.95-1.29x; 200 (B5 at 20 splits) was 0.70-0.80x: a fifth workgroup on a few WGPs
#: is a whole extra wave of the walk.
ROCM_WGS_PER_WGP = 4
#: and the most splits a request takes there
ROCM_MAX_SPLITS = 96


def plan_rocm(n_reqs: int, n_kv_heads: int, wgps: Optional[int] = None) -> int:
    """Splits for a decode batch on ROCM_ATTN's cache (one warp a workgroup): as many
    (split, kv head, request) workgroups as ROCM_WGS_PER_WGP a WGP, never more. From the batch
    and the device alone, so a CUDA graph's replays keep it."""
    have = max(1, n_reqs * n_kv_heads)
    return max(1, min(ROCM_MAX_SPLITS, ROCM_WGS_PER_WGP * (wgps or 48) // have))


def scratch(rows: int, n_kv_heads: int, group: int, head_dim: int, device, dtype=None):
    """(o_part, ml_part) for `rows` = requests x splits; o_part in the activations' dtype."""
    import torch  # noqa: PLC0415

    return (torch.empty(rows, n_kv_heads, group, head_dim, device=device,
                        dtype=dtype or torch.bfloat16),
            torch.empty(rows, n_kv_heads, 2, Q_TILE, device=device, dtype=torch.float32))
