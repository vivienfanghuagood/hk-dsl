"""Attention over a paged KV cache -- the decode half.

vLLM V1 does not keep K and V as dense `(B, H, N, D)` tensors. It keeps them
in fixed-size pages and hands the kernel a *page table*: `block_table[req][i]`
is where request `req`'s i-th page lives. There is no contiguous tensor on
that path, which is why an SDPA drop-in cannot reach a decoder however it is
configured, and why this file exists.

**The layout is ours, and that is the point.** `gl`'s strides are derived from
its dims, so a `(block, token, head, dim)` cache -- vLLM's default -- cannot
produce a contiguous `(block_size, head_dim)` tile for one head. Since a
backend declares its own `get_kv_cache_shape`, this one declares

    (num_blocks, num_kv_heads, block_size, head_dim)

which is exactly `gl`'s `(b, d, r, c)`, so one page of one head is a plain
tile load. The cost is that the cache update becomes a strided scatter instead
of a contiguous one -- and that is the right way round, because a decode step
*reads* `seq_len` pages per layer and *writes* one. The read is thousands of
times the traffic.

**Page size is the KV block.** `KV_BLOCK` here is the cache's page size rather
than a tiling choice, so one loop iteration is one page and the within-page
offset is always zero. vLLM's ROCm backends offer 16 and 32; 32 is what the
dense kernel already uses.

Its first form had no split across the KV axis -- one workgroup per (request,
kv head), walking the whole sequence -- and that was most of the gap to
Triton. `paged_decode_split_kernel` and `merge_splits` close it, with the
split count planned per call (`plan_splits`). The prefill half is
hk.ops.paged_prefill. The measurements are in
docs/history/06-phase-7-paged-decode.md.
"""

import math
from typing import Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32, i32

#: The cache's page size. Also the kernel's KV block, by construction.
KV_BLOCK = 32

#: Query rows per workgroup. A decode step has one query token per request, so
#: the rows are the GQA group -- the query heads that share this kv head --
#: padded to a tile. Packing the group rather than one row is what keeps the
#: WMMA from being 1/16 occupied.
Q_TILE = 16

NEG_INF = -1e30


def paged_decode_kernel(name: str, *, head_dim: int = 128,
                        kv_block: int = KV_BLOCK,
                        scale: Optional[float] = None,
                        max_vgprs: int = 256, min_occupancy: int = 1):
    """One workgroup per (request, kv head); it walks that request's pages.

    Single warp on purpose in this form. Decode attention is bandwidth bound --
    it reads the whole KV cache and does one query row of maths against it --
    so the schedule that matters is the read, and splitting the KV axis across
    warps buys parallelism at the cost of a cross-warp merge of (o, m, l). The
    merge is the next step, not this one; what this has to be first is right.
    """
    if head_dim % 16:
        raise ValueError(f"head_dim {head_dim} must be a multiple of 16")
    if kv_block % 16:
        raise ValueError(f"kv_block {kv_block} must be a multiple of 16")
    d_frags = head_dim // 16
    kv_frags = kv_block // 16

    # exp2, with log2(e) folded into the scale, and the scale applied to S
    # rather than to Q -- Q is bf16 and scaling it would round the input twice.
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    ty_q = hk.RT(bf16, 16, 16, "row")        # one d-slice of the Q group
    ty_k = hk.RT(bf16, 16, 16, "row")        # one [kv, d] fragment of a page
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")    # one [kv, q] fragment of S^T
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")    # one [d, q] fragment of O^T
    ty_v = hk.RT(bf16, 16, 16, "row")        # a [kv, d] fragment of V
    ty_vt = hk.RT(bf16, 16, 16, "row")       # the same, transposed
    ty_vec = hk.row_vec(ty_s)                # one entry per query row

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16], v_cache: hk.Tensor[bf16],
             block_table: hk.Tensor[i32], seq_lens: hk.Tensor[i32],
             o: hk.Tensor[bf16]):
        req = hk.arch.block_idx.z
        kv_head = hk.arch.block_idx.y

        # The only two numbers that are not in the shapes: how long this
        # request is, and where its pages are. Both come out of memory.
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)

        q_r = [q[hk.elem_coord(req, kv_head, 0, 16 * kk)].load(ty_q)
               for kk in range(d_frags)]

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        for pi in range(n_pages):
            page = block_table[hk.elem_coord(0, 0, req, pi)].load_scalar()

            with hk.scope():
                # S^T = K . Q^T, one [kv, d] fragment of the page at a time.
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            k_f = k_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * kk)].load(ty_k)
                            s_f[n] += k_f @ q_r[kk].T

                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                # The tail. A row of S^T is a kv index, so positions at or past
                # the sequence length are rows to fill -- not a predicate, and
                # not a backed-up block either: pages are allocated whole and
                # the slots past the end hold whatever the allocator left.
                # Counting them would corrupt the softmax sum.
                valid = seq_len - pi * kv_block
                if valid < kv_block:
                    for n in range(kv_frags):
                        s_f[n].lower_fill_(valid - 16 * n, NEG_INF)

                # ---- online softmax, the dense kernel's, unchanged ----
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

                # ---- O^T += V^T . P^T ----
                # V arrives [kv, d] and the matmul reads [d, kv], so each
                # fragment is transposed in registers on the way in. With one
                # warp there is nobody to share a staged copy with, so it does
                # not go through LDS at all.
                for n in range(kv_frags):
                    p_cvt = s_f[n].to(bf16)
                    for g0 in range(d_frags):
                        # One V fragment live at a time. Without the scope the
                        # allocator keeps all d_frags loads in flight -- eight
                        # at head_dim 128 -- on top of the O and Q tiles that
                        # are pinned for the whole kernel, and that is exactly
                        # the register file.
                        with hk.scope():
                            v_f = v_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * g0)].load(ty_v)
                            o_c[g0] += hk.transpose_sep(v_f) @ p_cvt

        for i, c in enumerate(o_c):
            c /= l_run
            o[hk.elem_coord(req, kv_head, 0, 16 * i)] = hk.transpose(c)

    body.__name__ = name
    return hk.kernel(body, warps=1, name=name,
                     grid=lambda p: (1, p.q.depth, p.q.batch),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


def paged_decode_split_kernel(name: str, *, head_dim: int = 128,
                              splits: int = 4, kv_block: int = KV_BLOCK,
                              scale: Optional[float] = None,
                              max_vgprs: int = 256, min_occupancy: int = 1):
    """The same walk, cut `splits` ways along the KV axis.

    Why this exists, measured rather than assumed. Without it one workgroup
    per (request, kv head) walks the whole context, so a batch-1 step on
    Qwen3-8B launches 8 workgroups of one warp each on a 48-WGP GPU. Going
    from 1024 to 4096 tokens of context cost that kernel +28.2 ms/step against
    vLLM's Triton backend's +0.97 -- 16 GB/s against 469, or 1.9% of this
    card's bandwidth against 54%. Decode attention is a bandwidth problem and
    the only way to use the bandwidth is to put the machine on it.

    Each workgroup produces a *partial* softmax over its page range: the
    unnormalised O, the running max m, and the running sum l. It does not
    divide, because a split's l is not the sequence's l. `merge_splits` below
    combines them.

    `splits` is a constexpr rather than a grid dimension read from a tensor,
    so the page range is compile-time arithmetic and a split past the end is
    a branch the workgroup takes once. A handful of instantiations covers
    every batch size; they are cached like any other schedule.
    """
    if head_dim % 16 or kv_block % 16:
        raise ValueError("head_dim and kv_block must be multiples of 16")
    d_frags, kv_frags = head_dim // 16, kv_block // 16
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    ty_q = hk.RT(bf16, 16, 16, "row")
    ty_k = hk.RT(bf16, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")
    ty_v = hk.RT(bf16, 16, 16, "row")
    ty_vec = hk.row_vec(ty_s)

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16], v_cache: hk.Tensor[bf16],
             block_table: hk.Tensor[i32], seq_lens: hk.Tensor[i32],
             o_part: hk.Tensor[fp32], ml_part: hk.Tensor[fp32]):
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z

        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)
        per = hk.cdiv(n_pages, splits)
        lo = split * per
        hi = hk.min(n_pages, lo + per)

        q_r = [q[hk.elem_coord(req, kv_head, 0, 16 * kk)].load(ty_q)
               for kk in range(d_frags)]
        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        # A split past the end of a short sequence contributes nothing. It
        # still has to write, because the merge reads every split's slot --
        # m = -inf and l = 0 make it weightless there.
        for pi in range(hk.max(hi - lo, 0)):
            page = block_table[hk.elem_coord(0, 0, req, lo + pi)].load_scalar()
            with hk.scope():
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            k_f = k_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * kk)].load(ty_k)
                            s_f[n] += k_f @ q_r[kk].T
                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                valid = seq_len - (lo + pi) * kv_block
                if valid < kv_block:
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

                for n in range(kv_frags):
                    p_cvt = s_f[n].to(bf16)
                    for g0 in range(d_frags):
                        with hk.scope():
                            v_f = v_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * g0)].load(ty_v)
                            o_c[g0] += hk.transpose_sep(v_f) @ p_cvt

        # Unnormalised. The merge divides once, by the sum of the rescaled
        # per-split sums; dividing here would throw away the only thing that
        # makes the splits combinable.
        row = req * splits + split
        for i, c in enumerate(o_c):
            o_part[hk.elem_coord(row, kv_head, 0, 16 * i)] = hk.transpose(c)
        ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m_run
        ml_part[hk.elem_coord(row, kv_head, 1, 0)] = l_run

    body.__name__ = name
    return hk.kernel(body, warps=1, name=name,
                     grid=lambda p: (splits, p.q.depth, p.q.batch),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


def merge_splits(o_part, ml_part, splits: int, out):
    """Combine the per-split partial softmaxes into the answer.

    In torch, for now, and deliberately: it is a few elementwise passes over a
    tensor whose size is `splits` times the output, which is small next to the
    KV traffic the splits exist to parallelise. If it shows up in a profile it
    becomes a kernel; writing it as one first would be optimising a thing
    nobody has measured.
    """
    import torch  # noqa: PLC0415

    r_s, hkv, tile, d = o_part.shape
    reqs = r_s // splits
    o = o_part.view(reqs, splits, hkv, tile, d)
    m = ml_part.view(reqs, splits, hkv, 2, tile)[:, :, :, 0]      # (r,s,h,t)
    l = ml_part.view(reqs, splits, hkv, 2, tile)[:, :, :, 1]

    m_max = m.amax(dim=1, keepdim=True)
    # A split that saw nothing has m = -inf and l = 0; exp2(-inf - m) is 0, so
    # it drops out -- unless *every* split saw nothing, which cannot happen
    # because split 0 always covers page 0.
    #
    # The numerator and the denominator take *different* weights, and getting
    # that wrong is the whole difficulty of a split softmax. O_s is already a
    # sum of (probability x V) under split s's own maximum, so rescaling it to
    # the global maximum takes exp2(m_s - m) and nothing else; l_s is a sum of
    # those same probabilities and takes the same factor. Weighting O by
    # exp2(m_s - m) * l_s -- which is what this did first -- multiplies every
    # numerator by its split's sum a second time. The weights still sum to
    # something, so the answer looks like an answer: 77x off, not NaN.
    rescale = torch.exp2(m - m_max)                                # (r,s,h,t)
    num = (o * rescale.unsqueeze(-1)).sum(dim=1)                   # (r,h,t,d)
    den = (l * rescale).sum(dim=1).clamp_min(1e-30)                # (r,h,t)
    out.copy_((num / den.unsqueeze(-1)).to(out.dtype))
    return out


def _native_cdna() -> bool:
    """Whether calls go to hk.ops.paged_cdna, gfx950's own decode: the
    kernels here are RDNA's (16-wide WMMA operands, V transposed in
    registers), and gfx950's is another schedule (K and V by DMA through
    LDS, Pᵀ read in place), not a port."""
    from ..target import current_arch  # noqa: PLC0415

    return current_arch() == "gfx950"


class _Kernels(dict):
    """head_dim -> the unsplit decode; on gfx950, paged_cdna's."""

    def __getitem__(self, head_dim):
        if _native_cdna():
            from . import paged_cdna  # noqa: PLC0415

            return paged_cdna.kernel_for(head_dim, 1, "plain")
        return dict.__getitem__(self, head_dim)

    def get(self, head_dim, default=None):
        """As `[]`, so gfx950 gets paged_cdna's kernel here too (dict.get
        does not go through __getitem__)."""
        return self[head_dim] if head_dim in self else default


#: head_dim -> kernel. Nothing is traced or compiled at import.
KERNELS: Dict[int, object] = _Kernels({
    64: paged_decode_kernel("paged_decode_d64", head_dim=64),
    128: paged_decode_kernel("paged_decode_d128", head_dim=128),
})

#: (head_dim, splits) -> kernel, built on demand.
_SPLIT: Dict[tuple, object] = {}


def split_kernel(head_dim: int, splits: int):
    """The split decode's walk for (head_dim, splits): each split's
    partial output and its (max, sum) into the scratch `merge_kernel`
    combines. On gfx950, paged_cdna's (head_dim 64 or 128)."""
    if _native_cdna():
        from . import paged_cdna  # noqa: PLC0415

        return paged_cdna.kernel_for(head_dim, splits, "split")
    key = (head_dim, splits)
    if key not in _SPLIT:
        _SPLIT[key] = paged_decode_split_kernel(
            f"paged_decode_d{head_dim}_s{splits}",
            head_dim=head_dim, splits=splits)
    return _SPLIT[key]


@hk.func
def _merge_splits(out, o_part, ml_part, req, kv_head, splits, active, cols,
                  step):
    """Fold `active` normalised split partials of one (request, kv head) into
    the answer, for the 16-column fragments of head_dim starting at `cols`.

    One pass, online across steps of `step` splits: a step's m's, l's and
    partials are all read before any is used, the running max takes the
    step's, the accumulator is rescaled once, and the step's partials go in
    at exp2(m_s - max) * l_s -- one round of L2 latency a step, where a pass
    for the max and a pass for the sum paid two. Vectors are per row of the
    (q, d) tiles. The max starts finite, so a split that saw nothing weighs
    zero rather than NaN; a step past the last split rereads it and weighs it
    zero.

    The partials are read bounded, as bf16, and widened by copy(). Read
    unbounded straight into fp32 -- no descriptor per read, so a step of 16
    fits -- the merge was slower at every point measured, 7.8-8.8 us to
    11.6-14.8 (tools/hk-bench/paged_kernels.py, 36 layers' caches): each
    unbounded read carries its alignment branch and its elementwise
    fallback, and the larger code is what this latency-bound launch pays.
    """
    d_frags = len(cols)
    ty_ot = hk.RT(fp32, Q_TILE, 16, "row")
    ty_ob = hk.RT(bf16, Q_TILE, 16, "row")
    cvec = hk.col_vec(ty_ot)
    m_run = hk.zeros(cvec)
    m_run -= -NEG_INF
    top = active - 1
    acc = [hk.zeros(ty_ot) for _ in range(d_frags)]
    l_acc = hk.zeros(cvec)
    for s1 in hk.range(0, active, step):
        rows = [req * splits + hk.min(s1 + j2, top) for j2 in range(step)]
        ws = [ml_part[hk.elem_coord(r, kv_head, 0, 0)].load(cvec) for r in rows]
        ls = [ml_part[hk.elem_coord(r, kv_head, 1, 0)].load(cvec) for r in rows]
        os_ = [[o_part[hk.elem_coord(r, kv_head, 0, cols[fi])].load(
            ty_ob, bounded=True) for r in rows] for fi in range(d_frags)]
        m_old = m_run.copy()
        for w in ws:
            m_run.max_(w)
        alpha = m_old - m_run
        alpha.exp2_()
        l_acc *= alpha
        for fi in range(d_frags):
            acc[fi] *= alpha
        for j3, (w, l_s) in enumerate(zip(ws, ls)):
            w -= m_run
            w.exp2_()
            w *= l_s
            if j3:
                # 1 for a real split, 0 for a reread of the last.
                w *= hk.min(hk.max(active - s1 - j3, 0), 1)
            l_acc += w
        for fi in range(d_frags):
            for w2, o_b in zip(ws, os_[fi]):
                o_s = o_b.to(fp32)
                o_s *= w2
                acc[fi] += o_s
    # A request with no context at all -- a CUDA graph's padding -- has l = 0;
    # the floor makes its rows zero rather than 0/0.
    l_acc += 1e-30
    for i in range(d_frags):
        acc[i] /= l_acc
        out[hk.elem_coord(req, kv_head, 0, cols[i])].store(acc[i], bounded=True)


def paged_merge_kernel(name: str, *, head_dim: int = 128, splits: int = 8,
                       kv_block: int = KV_BLOCK, merge_step: int = 0,
                       warps: int = 0):
    """The split decode's merge as a launch of its own: one workgroup per
    (request, kv head), one warp per 16 columns of head_dim.

    Measured apart (tools/hk-bench/paged_attrib.py), the merge one warp ran
    at the end of the walk was slow code, not slow placement: alone it took
    18 us over 8 splits and 36-47 over 32, about 400 instructions a split for
    eight fragments, in series. A warp a fragment shares that eight ways, and
    the warps need nothing from each other."""
    d_frags = head_dim // 16
    # Splits read per round of latency. More would be fewer rounds, but each
    # bounded read builds a 4-SGPR buffer descriptor: 16 spilled SGPRs under
    # both hipcc 7.1 and 7.2, and 12 under 7.2 only.
    step = merge_step or min(splits, MERGE_STEP)
    warps = warps or d_frags
    if d_frags % warps:
        raise ValueError(f"{d_frags} fragments do not split over {warps} warps")
    per_warp = d_frags // warps

    def body(seq_lens: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)
        per = hk.max(hk.cdiv(n_pages, splits), MIN_SPLIT_PAGES)
        active = hk.cdiv(n_pages, per)
        base = hk.arch.warp_id() * (16 * per_warp)
        _merge_splits(out, o_part, ml_part, req, kv_head, splits, active,
                      [base + 16 * f for f in range(per_warp)], step)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (1, p.out.depth, p.out.batch))


def paged_decode_fused_kernel(name: str, *, head_dim: int = 128,
                              splits: int = 1, kv_block: int = KV_BLOCK,
                              scale: Optional[float] = None,
                              max_vgprs: int = 256, min_occupancy: int = 1,
                              ablate: tuple = (),
                              defines: Optional[dict] = None):
    """The decode a serving engine can capture: the engine's query in, and
    with one split the answer out; with more, `paged_merge_kernel` after it.

    Differences from the two kernels above, each a launch or a copy the vLLM
    backend no longer makes per layer per step:

    * The query and the output are the engine's own `(tokens, kv_heads, group,
      head_dim)` views. A group of four rows loads into a 16-row tile bounded
      by the global, so the pad rows read zeros and are never stored, where
      the backend used to copy into a zero-padded tile and slice back out.
    * With `splits > 1` each split writes its partial normalised, as bf16,
      for the group's rows only, and the merge is one launch with a warp per
      head_dim fragment instead of torch's dozen. Folding it into the split
      that finishes last (`wg_arrive`) was measured and lost: that is one
      warp doing all of it in series, 18-47 us on every decode.
    * Nothing comes from the host but the grid. A split's page range is
      arithmetic on the device `seq_len`, so a split count fixed when a CUDA
      graph was captured is right for every replay: a split past the end of a
      short sequence is empty and merges with weight zero.

    `ablate` deletes stages for tools/hk-bench/paged_attrib.py, and makes the
    answer wrong on purpose: "page" (pages are contiguous, no table read),
    "kload"/"vload" (one K or V fragment loaded once and reused, so the math
    stays and the stream goes), "qk", "softmax", "pv".
    """
    if head_dim % 16 or kv_block % 16:
        raise ValueError("head_dim and kv_block must be multiples of 16")
    bad = set(ablate) - {"page", "kload", "vload", "qk", "softmax", "pv"}
    if bad:
        raise ValueError(f"unknown ablation {sorted(bad)}")
    d_frags, kv_frags = head_dim // 16, kv_block // 16
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    ty_q = hk.RT(bf16, 16, 16, "row")
    ty_k = hk.RT(bf16, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")
    ty_v = hk.RT(bf16, 16, 16, "row")
    ty_vec = hk.row_vec(ty_s)

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16], v_cache: hk.Tensor[bf16],
             block_table: hk.Tensor[i32], seq_lens: hk.Tensor[i32],
             out: hk.Tensor[bf16], o_part: hk.Tensor[bf16],
             ml_part: hk.Tensor[fp32]):
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z

        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)
        # At least MIN_SPLIT_PAGES a split, so a short sequence under a split
        # count planned for a long one uses a few splits instead of many
        # one-page ones -- each of which the merge would have to read.
        per = hk.max(hk.cdiv(n_pages, splits), MIN_SPLIT_PAGES)
        lo = split * per
        hi = hk.min(n_pages, lo + per)

        q_r = [q[hk.elem_coord(req, kv_head, 0, 16 * kk)].load(ty_q, bounded=True)
               for kk in range(d_frags)]
        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)
        k_keep = (k_cache[hk.elem_coord(0, kv_head, 0, 0)].load(ty_k)
                  if "kload" in ablate else None)
        v_keep = (v_cache[hk.elem_coord(0, kv_head, 0, 0)].load(ty_v)
                  if "vload" in ablate else None)

        # The page index is a load the page's K loads wait on, so it is read
        # one page ahead: the next index is in flight while this page is
        # worked. Every column read is clamped into the request's pages -- an
        # empty split's `lo` can be past the end of the table's row.
        nxt = block_table[hk.elem_coord(
            0, 0, req, hk.max(hk.min(lo, n_pages - 1), 0))].load_scalar()
        for pi in range(hk.max(hi - lo, 0)):
            if "page" in ablate:
                page = req * n_pages + lo + pi
            else:
                page = nxt
                nxt = block_table[hk.elem_coord(
                    0, 0, req, hk.min(lo + pi + 1, hi - 1))].load_scalar()
            # Loads are issued a 16-token row of the page at a time -- eight
            # fragments, then their eight WMMAs -- and through the bounded
            # (buffer) path, which is straight-line code. One fragment per
            # `hk.scope()` made every load an opaque branch of its own: load,
            # wait the full latency, multiply, sixteen times a page for K
            # and sixteen for V.
            def k_row(n):
                return [k_keep if k_keep is not None else
                        k_cache[hk.elem_coord(page, kv_head, 16 * n, 16 * kk)]
                        .load(ty_k, bounded=True) for kk in range(d_frags)]

            def v_row(n):
                return [v_keep if v_keep is not None else
                        v_cache[hk.elem_coord(page, kv_head, 16 * n, 16 * g0)]
                        .load(ty_v, bounded=True) for g0 in range(d_frags)]

            with hk.scope():
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for n in range(kv_frags if "qk" not in ablate else 0):
                    ks = k_row(n)
                    for kk in range(d_frags):
                        s_f[n] += ks[kk] @ q_r[kk].T
                # V's first row does not wait on the softmax; in flight
                # during it.
                vs = v_row(0) if "pv" not in ablate else None

                if "softmax" not in ablate:
                    for n in range(kv_frags):
                        s_f[n] *= scale_l2e

                    valid = seq_len - (lo + pi) * kv_block
                    if valid < kv_block:
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

                for n in range(kv_frags if "pv" not in ablate else 0):
                    p_cvt = s_f[n].to(bf16)
                    if n:
                        vs = v_row(n)
                    for g0 in range(d_frags):
                        o_c[g0] += hk.transpose_sep(vs[g0]) @ p_cvt

        if splits == 1:
            for i, c in enumerate(o_c):
                c /= l_run
                out[hk.elem_coord(req, kv_head, 0, 16 * i)].store(
                    hk.transpose(c), bounded=True)
            return

        # The partial goes out normalised, transposed to (q, d) and in bf16,
        # bounded to the group's rows: a quarter of the 16-row tile, and in
        # the one register layout a load vectorises (a bf16 row reads as two
        # b128 per lane; an fp32 accumulator tile is eight scalar loads).
        # Normalised, its values are a convex mix of V's, so bf16 costs the
        # partial what bf16 costs the output. An empty split has l = 0 and
        # O = 0; the floor keeps 0/0 out of what the merge multiplies by 0.
        row = req * splits + split
        l_safe = l_run.copy()
        l_safe += 1e-30
        for i, c in enumerate(o_c):
            c /= l_safe
            o_part[hk.elem_coord(row, kv_head, 0, 16 * i)].store(
                hk.transpose(c), bounded=True)
        ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m_run
        ml_part[hk.elem_coord(row, kv_head, 1, 0)] = l_run

    body.__name__ = name
    return hk.kernel(body, warps=1, name=name,
                     grid=lambda p: (splits, p.q.depth, p.q.batch),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy,
                     defines=defines)


#: The fewest pages a split of the fused decode takes. Above 1 it trades the
#: merge's reads for a longer serial walk per split, and at 4 that lost on
#: both ends (tools/hk-bench/paged_attrib.py: a ctx-1024 decode went from
#: 56 to 67 us with the main loop 16 -> 42 us); the merge is cheapened
#: instead, by what it reads.
MIN_SPLIT_PAGES = 1

#: Splits whose partials the merge reads at once.
MERGE_STEP = 8

#: (head_dim, splits) -> the fused kernel, built on demand.
_FUSED: Dict[tuple, object] = {}


def fused_kernel(head_dim: int, splits: int, ablate: tuple = (),
                 defines: Optional[dict] = None):
    """`defines` builds the walk with library knobs, for an A/B in one
    process (tools/hk-bench/paged_variants.py): HK_RDNA_XPOSE_WORDS=0 is the
    generic register transpose V took before the packed one."""
    if _native_cdna():
        if ablate or defines:
            raise ValueError("ablate= and defines= build variants of the RDNA "
                             "walk; gfx950's decode is hk.ops.paged_cdna's")
        from . import paged_cdna  # noqa: PLC0415

        return paged_cdna.kernel_for(head_dim, splits, "fused")
    dkey = tuple(sorted((defines or {}).items()))
    key = (head_dim, splits, tuple(sorted(ablate)), dkey)
    if key not in _FUSED:
        tag = "".join(f"_no{a}" for a in key[2])
        tag += "".join(f"_{k.lower()}{v}" for k, v in dkey)
        _FUSED[key] = paged_decode_fused_kernel(
            f"paged_decode_fused_d{head_dim}_s{splits}{tag}",
            head_dim=head_dim, splits=splits, ablate=key[2], defines=defines)
    return _FUSED[key]


#: (head_dim, splits) -> the merge launch, built on demand.
_MERGE: Dict[tuple, object] = {}


def merge_kernel(head_dim: int, splits: int):
    """The kernel that combines `splits` partial outputs of the split decode
    into the answer. On gfx950, paged_cdna's: it reads splits eight at a
    time, so above 8 `splits` is a multiple of 8."""
    if _native_cdna():
        from . import paged_cdna  # noqa: PLC0415

        return paged_cdna.merge_for(head_dim, splits)
    key = (head_dim, splits)
    if key not in _MERGE:
        _MERGE[key] = paged_merge_kernel(
            f"paged_merge_d{head_dim}_s{splits}", head_dim=head_dim,
            splits=splits)
    return _MERGE[key]


def fused_scratch(rows: int, n_kv_heads: int, head_dim: int, device,
                  group: int = 4):
    """(o_part, ml_part) for `rows` = requests x splits: each split's
    normalised partial for the `group` real query rows, and its (m, l)."""
    import torch  # noqa: PLC0415

    o_part = torch.empty(rows, n_kv_heads, group, head_dim, device=device,
                         dtype=torch.bfloat16)
    ml = torch.empty(rows, n_kv_heads, 2, Q_TILE, device=device,
                     dtype=torch.float32)
    return o_part, ml


def decode(walk, merge, q, k_cache, v_cache, block_table, seq_lens, out,
           scratch, splits: int) -> None:
    """The split decode's two launches, with whatever callables run them --
    Kernels for a bench, torch ops for an engine."""
    walk(q, k_cache, v_cache, block_table, seq_lens, out, *scratch)
    if splits > 1:
        merge(seq_lens, out, *scratch)


#: Workgroups to aim for, measured rather than reasoned
#: (tools/hk-bench/paged_variants.py, Qwen3-8B's shape on a W7900D: 8 kv
#: heads, head_dim 128, on vLLM 0.31's cache -- K and V side by side in one
#: tensor -- with every layer's cache cold, as a CUDA graph's replay). Each
#: column is a split count, each cell the walk plus its merge in ms a layer,
#: and the last column `n_reqs * n_kv_heads * splits` at the best split:
#:
#:     ctx  batch     s1      s2      s4      s8     s16     s32   best  wgs
#:    1024      1  0.0946  0.0520  0.0322  0.0237  0.0218  0.0237    16  128
#:    1024      4  0.0976  0.0571  0.0451  0.0430  0.0517  0.0647     8  256
#:    1024      8  0.1027  0.0805  0.0637  0.0817  0.1010  0.1110     4  256
#:    1024     16  0.1360  0.1121  0.1220  0.1524  0.1699  0.1774     2  256
#:    1024     32  0.1989  0.2157  0.2434  0.2693  0.2913  0.3052     1  256
#:    1024     64  0.3922  0.4266  0.4552  0.4785  0.5115  0.5458     1  512
#:    4096      1  0.3512  0.1852  0.1024  0.0593  0.0462  0.0438    32  256
#:    4096      4  0.3589  0.1971  0.1410  0.1150  0.1251  0.1516     8  256
#:    4096      8  0.3738  0.2615  0.2041  0.2177  0.2480  0.2646     4  256
#:    4096     16  0.5023  0.3824  0.3981  0.4270  0.4511  0.4709     2  256
#:    4096     32  0.7447  0.7679  0.7862  0.8167  0.8435  0.8781     1  256
#:    4096     64  1.4931  1.4983  1.5329  1.5629  1.5998  1.6643     1  512
#:
#: The best split at eleven of the twelve points puts about 256 one-warp
#: workgroups on the card, one or two a SIMD; the twelfth is a single request,
#: which vLLM's Triton kernel serves (vllm_backend.TRITON_DECODE_MAX). This
#: was a thousand until V's transpose in registers was made packed
#: (docs/history/38-rdna-losses.md): with a page's VALU work cut by 57%, more
#: workgroups walking fewer pages each stopped paying for their merge reads
#: and their own setup. At 16 requests and 4096 tokens a thousand was 0.427
#: ms a layer and 256 is 0.382.
TARGET_WORKGROUPS = 256

#: Past this the merge's traffic grows faster than the split saves, and every
#: value is a separate compile.
MAX_SPLITS = 32


def plan_splits(n_reqs: int, n_kv_heads: int, n_pages: int) -> int:
    """How many ways to cut the KV axis, given how empty the machine is.

    On gfx950 the merge reads splits eight at a time, so above 8 a plan must
    be a multiple of 8 there (vllm_backend.decode_splits rounds to a power of
    two); gfx1100's merge takes any count.

    Without a split the grid is `n_reqs * n_kv_heads` one-warp workgroups.
    This asks for enough splits to reach `TARGET_WORKGROUPS`, capped by the
    page count -- a split with no pages is pure overhead -- and by
    `MAX_SPLITS` (on gfx950, paged_cdna's target_workgroups and MAX_SPLITS).
    """
    import os  # noqa: PLC0415

    forced = os.environ.get("HK_SPLITS", "auto")
    if forced in ("off", "0", "1"):
        # One split: the switch that ruled this path in when vLLM at batch 16
        # / context 4096 aborted on its second `generate` with
        # HSA_STATUS_ERROR_EXCEPTION 0x1016 inside an unrelated torch kernel.
        # Forcing one split made it go away and forcing prefill onto torch did
        # not; 2000 calls of the kernel with ragged lengths, padded block
        # tables and guard pages either side of both outputs left every margin
        # byte untouched, so it was not the kernel but how it was driven. The
        # backend's scratch was keyed on the decode request count, which moves
        # as a batch ramps, so it allocated a fresh set per count, per split,
        # per layer, until the card ran out. It is keyed on the split count
        # now (vllm_backend._buffers), and the planned split is the default.
        return 1
    target, cap = TARGET_WORKGROUPS, MAX_SPLITS
    have = max(1, n_reqs * n_kv_heads)
    if _native_cdna():
        from . import paged_cdna  # noqa: PLC0415

        target, cap = paged_cdna.target_workgroups(have), paged_cdna.MAX_SPLITS
    if forced in ("auto", "plan"):
        want = -(-target // have)                 # ceil
        return max(1, min(want, n_pages, cap))
    return max(1, min(int(forced), n_pages, cap))
