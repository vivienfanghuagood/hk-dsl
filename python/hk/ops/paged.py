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

What this does not do, deliberately, in its first form: no split across the KV
axis (one workgroup per (request, kv head), walking the whole sequence), and
no prefill. Both are measurable next steps rather than guesses -- see
`python/README.md`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .. import lang as _ops
from ..lang import GL, const, kernel as _kernel
from ..ir.nodes import bf16, fp32, i32

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

    ty_q = _ops.rt(bf16, 16, 16, "row")        # one d-slice of the Q group
    ty_k = _ops.rt(bf16, 16, 16, "row")        # one [kv, d] fragment of a page
    ty_s = _ops.rt(fp32, 16, Q_TILE, "col")    # one [kv, q] fragment of S^T
    ty_o = _ops.rt(fp32, 16, Q_TILE, "col")    # one [d, q] fragment of O^T
    ty_v = _ops.rt(bf16, 16, 16, "row")        # a [kv, d] fragment of V
    ty_vt = _ops.rt(bf16, 16, 16, "row")       # the same, transposed
    ty_vec = _ops.row_vec(ty_s)                # one entry per query row

    def body(q, k_cache, v_cache, block_table, seq_lens, o):
        req = _ops.block_idx.z
        kv_head = _ops.block_idx.y

        # The only two numbers that are not in the shapes: how long this
        # request is, and where its pages are. Both come out of memory.
        seq_len = _ops.load_scalar(seq_lens, _ops.elem_coord(0, 0, 0, req))
        n_pages = _ops.s_cdiv(seq_len, kv_block)

        q_r = [_ops.load(q, _ops.elem_coord(req, kv_head, 0, 16 * kk), ty_q)
               for kk in range(d_frags)]

        o_c = [_ops.zeros(ty_o) for _ in range(d_frags)]
        m_run = _ops.neg_infty_vec(ty_vec)
        l_run = _ops.zeros_vec(ty_vec)

        for pi in _ops.range(n_pages):
            page = _ops.load_scalar(block_table, _ops.elem_coord(0, 0, req, pi))

            with _ops.scope():
                # S^T = K . Q^T, one [kv, d] fragment of the page at a time.
                s_f = [_ops.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with _ops.scope():
                            k_f = _ops.load(
                                k_cache,
                                _ops.elem_coord(page, kv_head, 16 * n,
                                                16 * kk),
                                ty_k)
                            _ops.mma_ABt(k_f, q_r[kk], s_f[n], out=s_f[n])

                for n in range(kv_frags):
                    _ops.mul(s_f[n], scale_l2e, out=s_f[n])

                # The tail. A row of S^T is a kv index, so positions at or past
                # the sequence length are rows to fill -- not a predicate, and
                # not a backed-up block either: pages are allocated whole and
                # the slots past the end hold whatever the allocator left.
                # Counting them would corrupt the softmax sum.
                valid = _ops.s_sub(seq_len, pi * kv_block)
                with _ops.if_(_ops.s_lt(valid, kv_block)):
                    for n in range(kv_frags):
                        _ops.lower_fill(s_f[n], _ops.s_sub(valid, 16 * n),
                                        NEG_INF, out=s_f[n])

                # ---- online softmax, the dense kernel's, unchanged ----
                m_old = _ops.copy(m_run)
                for n in range(kv_frags):
                    _ops.col_max(s_f[n], out=m_run, accumulate=True)
                for n in range(kv_frags):
                    _ops.sub_col(s_f[n], m_run, out=s_f[n])
                    _ops.exp2(s_f[n], out=s_f[n])
                with _ops.if_(_ops.s_any_ne(m_run, m_old)):
                    alpha = _ops.exp2(_ops.sub(m_old, m_run))
                    _ops.mul(l_run, alpha, out=l_run)
                    for c in o_c:
                        _ops.mul_col(c, alpha, out=c)
                for n in range(kv_frags):
                    _ops.col_sum(s_f[n], out=l_run, accumulate=True)

                # ---- O^T += V^T . P^T ----
                # V arrives [kv, d] and the matmul reads [d, kv], so each
                # fragment is transposed in registers on the way in. With one
                # warp there is nobody to share a staged copy with, so it does
                # not go through LDS at all.
                for n in range(kv_frags):
                    p_cvt = _ops.cast(s_f[n], bf16)
                    for g0 in range(d_frags):
                        # One V fragment live at a time. Without the scope the
                        # allocator keeps all d_frags loads in flight -- eight
                        # at head_dim 128 -- on top of the O and Q tiles that
                        # are pinned for the whole kernel, and that is exactly
                        # the register file.
                        with _ops.scope():
                            v_f = _ops.load(
                                v_cache,
                                _ops.elem_coord(page, kv_head, 16 * n, 16 * g0),
                                ty_v)
                            _ops.mma_AB(_ops.transpose_sep(v_f), p_cvt,
                                        o_c[g0], out=o_c[g0])

        for i, c in enumerate(o_c):
            _ops.div_col(c, l_run, out=c)
            _ops.store(o, _ops.transpose(c),
                       _ops.elem_coord(req, kv_head, 0, 16 * i))

    body.__name__ = name
    # Set rather than written in the signature, because the dtypes differ by
    # parameter and PEP 563 would make them strings resolved in this module.
    body.__annotations__ = {
        "q": GL[bf16], "k_cache": GL[bf16], "v_cache": GL[bf16],
        "block_table": GL[i32], "seq_lens": GL[i32], "o": GL[bf16],
    }
    return _kernel(body, arch="gfx1100", warps=1, name=name,
                   grid=lambda p: (1, p.q.depth, p.q.batch),
                   max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: head_dim -> kernel. Nothing is traced or compiled at import.
KERNELS: Dict[int, object] = {
    64: paged_decode_kernel("paged_decode_d64", head_dim=64),
    128: paged_decode_kernel("paged_decode_d128", head_dim=128),
}
