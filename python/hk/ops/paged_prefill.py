"""Attention over a paged KV cache -- the prefill half.

What makes vLLM's defaults work. With chunked prefill and prefix caching on,
a request's new tokens are a suffix of its sequence: `ctx` tokens are already
in the cache (a cached prefix, or the chunks before this one), and the new
`q_len` attend to all `ctx + q_len` of them, causally. The dense kernel
(hk.ops.attn) cannot see a cached prefix, so without this the backend had to
refuse both features.

This is that kernel with three things changed, and its schedule otherwise
the dense one's -- Q in registers, K staged into LDS and V transposed into it,
one barrier pair a KV block:

* K and V come through the page table. The KV block *is* the page, so a
  block's staging reads one page of one kv head, contiguous.
* The diagonal sits at `ctx`: query row i of a request is position ctx + i.
  The causal mask then also masks the cache slots past the sequence's end,
  since no query of the request is past it.
* A query tile is one request's, and its rows end where the request's do --
  the next rows are another request's tokens. Loads and stores are bounded by
  the rows the request has left (`bounded=n`), not by the tensor's end, so a
  short chunk computes on zeros past its end and never writes them.

The grid is a work list built per batch: one entry per (request, 192-row
query tile of it), so a batch of ragged chunks launches once.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32, i32
from . import _arch
from .attn import _derive

NEG_INF = float("-inf")

#: Query rows a gfx1100 workgroup owns (gfx950: attn_cdna.Q_TILE, 256; build
#: the work list with `work_list`, which knows), and the KV step -- the
#: cache's page.
Q_TILE = 192
KV_BLOCK = 32


def paged_prefill_kernel(name: str, *, head_dim: int = 128, group: int = 4,
                         q_block: int = 16, kv_block: int = KV_BLOCK,
                         warps: int = 12, vt_d_chunk: int = 16,
                         qk_tiles: int = 2, pv_tiles: int = 2,
                         scale: Optional[float] = None,
                         max_vgprs: int = 256, min_occupancy: int = 1):
    """Causal paged prefill: `q` and `o` are the engine's (tokens, heads *
    head_dim) rows, `tiles` the (request, tile) work list, `q_start` the
    batch's query_start_loc."""
    d = _derive(head_dim, q_block, kv_block, warps, vt_d_chunk, qk_tiles, pv_tiles)
    q_tile, d_frags, kv_frags = d["q_tile"], d["d_frags"], d["kv_frags"]
    d_chunks, v_bands, vt_frags = d["d_chunks"], d["v_bands"], d["vt_frags"]
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    st_k = hk.ST(bf16, kv_block, head_dim)
    st_vt = hk.ST(bf16, head_dim, kv_block)
    ty_q = hk.RT(bf16, 16, 16, "row")
    ty_k = hk.RT(bf16, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, q_block, "col")
    ty_o = hk.RT(fp32, 16, q_block, "col")
    ty_v = hk.RT(bf16, 16, vt_d_chunk, "row")
    ty_vr = hk.RT(bf16, 16, 16, "row")
    ty_vec = hk.row_vec(ty_s)

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
             v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
             seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32],
             tiles: hk.Tensor[i32], o: hk.Tensor[bf16]):
        g = hk.group(warps)
        k_s = hk.lds.alloc(st_k, name="k_s")
        vt_s = hk.lds.alloc(st_vt, name="vt_s")
        buf_k = g.stage_buffer(k_s, name="buf_k")
        gr_k = k_s.granules(name="gr_k")
        gr_vt = vt_s.granules(name="gr_vt")

        t = hk.arch.block_idx.x
        head = hk.arch.block_idx.y
        req = tiles[hk.elem_coord(0, 0, t, 0)].load_scalar()
        qb = tiles[hk.elem_coord(0, 0, t, 1)].load_scalar()
        q0 = q_start[hk.elem_coord(0, 0, 0, req)].load_scalar()
        q_len = q_start[hk.elem_coord(0, 0, 0, req + 1)].load_scalar() - q0
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        ctx = seq_len - q_len
        head_kv = head // group
        col = head * head_dim

        tile_lo = qb * q_tile
        wid = hk.arch.warp_id()
        q_local = tile_lo + wid * q_block           # this warp's first row
        left = q_len - q_local                      # its rows the request has

        q_r = [q[hk.elem_coord(0, 0, q0 + q_local, col + 16 * kk)]
               .load(ty_q, bounded=left) for kk in range(d_frags)]
        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)
        v_rows = [hk.zeros(ty_v) for _ in range(v_bands)]
        c_chunk = wid
        stages_v = c_chunk < d_chunks

        # Up to the workgroup's last real query, uniform across it: staging
        # has barriers in it. A warp whose rows end sooner is masked per
        # fragment below.
        last_q = ctx + hk.min(tile_lo + q_tile, q_len) - 1
        kv_blocks = hk.cdiv(last_q + 1, kv_block)

        def page_of(kb):
            # Clamped: past the end the prefetch rereads the last page.
            return block_table[hk.elem_coord(
                0, 0, req, hk.min(kb, kv_blocks - 1))].load_scalar()

        def load_kv(page):
            g.stage(buf_k, k_cache[hk.elem_coord(page, head_kv, 0, 0)])
            if stages_v:
                for w in range(v_bands):
                    v_cache[hk.elem_coord(page, head_kv, 16 * w,
                                          c_chunk * vt_d_chunk)].load(
                        ty_v, into=v_rows[w])

        def commit_kv():
            hk.arch.vm_wait(0)
            g.commit(k_s, buf_k)
            if stages_v:
                for w in range(v_bands):
                    v_t = hk.transpose_sep(v_rows[w])
                    for tt in range(vt_frags):
                        gr_vt.store_frag(v_t, c_chunk * vt_frags + tt, w)

        load_kv(page_of(0))

        for kb in range(kv_blocks):
            kv_lo = kb * kv_block
            hk.arch.barrier()
            commit_kv()
            hk.arch.barrier(drain=True)
            load_kv(page_of(kb + 1))

            with hk.scope():
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n0 in range(0, kv_frags, qk_tiles):
                        k_win = [gr_k.load_frag(n0 + i, kk, ty_k)
                                 for i in range(qk_tiles)]
                        hk.lds.wait_for(*k_win)
                        hk.arch.setprio(1)
                        for i in range(qk_tiles):
                            s_f[n0 + i] += k_win[i] @ q_r[kk].T
                        hk.arch.setprio(0)

                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                # kv > position: S^T's rows are kv, its columns this warp's
                # queries, at positions ctx + q_local + column.
                dd = ctx + q_local - kv_lo
                for n in range(kv_frags):
                    s_f[n].triu_(dd - 16 * n, NEG_INF)

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
                    for g0 in range(0, d_frags, pv_tiles):
                        v_win = [gr_vt.load_frag(g0 + i, n, ty_vr)
                                 for i in range(pv_tiles)]
                        hk.lds.wait_for(*v_win)
                        hk.arch.setprio(1)
                        for i in range(pv_tiles):
                            o_c[g0 + i] += v_win[i] @ p_cvt
                        hk.arch.setprio(0)

        # A warp past the request's end saw only masked keys past its own
        # position and has l = 0 in no row it writes: its stores are bounded
        # to nothing.
        for i, c in enumerate(o_c):
            c /= l_run
            o[hk.elem_coord(0, 0, q0 + q_local, col + 16 * i)].store(
                hk.transpose(c), bounded=left)

    body.__name__ = name

    def _grid(p):
        return (p.tiles.rows, p.o.cols // head_dim, 1)

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: (head_dim, group, scale) -> kernel.
_KERNELS: Dict[tuple, object] = {}


def _native_cdna() -> bool:
    """Whether calls go to gfx950's own prefill, hk.ops.attn_cdna's schedule
    over the paged cache: this kernel's tiles are RDNA's (16-wide WMMA
    operands, LDS granules)."""
    from ..target import current_arch  # noqa: PLC0415

    return current_arch() == "gfx950"


def kernel_for(head_dim: int, group: int, scale: Optional[float] = None):
    """The paged prefill for (head_dim, group, scale). On gfx1100 `group`
    (query heads a KV head) is compiled in and must match the tensors, and
    head_dim is a multiple of 32 up to 192. On gfx950 this is
    attn_cdna.paged_kernel_for: `group` is read off the shapes (the argument
    is ignored) and head_dim is 64 or 128."""
    _arch.require("hk.ops.paged_prefill")
    if _native_cdna():
        from . import attn_cdna  # noqa: PLC0415

        return attn_cdna.paged_kernel_for(head_dim, scale)
    key = (head_dim, group, None if scale is None or scale == head_dim ** -0.5
           else float(scale))
    k = _KERNELS.get(key)
    if k is None:
        import struct  # noqa: PLC0415

        tag = "" if key[2] is None else "_s%016x" % struct.unpack(
            "<Q", struct.pack("<d", key[2]))[0]
        k = _KERNELS[key] = paged_prefill_kernel(
            f"paged_prefill_d{head_dim}_g{group}{tag}", head_dim=head_dim,
            group=group, scale=key[2])
    return k


def work_list(q_lens, device):
    """The (request, tile) entries for a batch's prefill requests: one per
    Q_TILE rows of each, ints on the host, one int32 copy to the device."""
    import torch  # noqa: PLC0415

    if _native_cdna():
        # gfx950's tile, and a request's tiles last first: causal, the last
        # reads the most of the cache, so the longest start first.
        from .attn_cdna import Q_TILE as tile  # noqa: PLC0415

        rows = [(r, t) for r, n in q_lens for t in reversed(range(-(-n // tile)))]
        return torch.tensor(rows or [(0, 0)], dtype=torch.int32).to(
            device, non_blocking=True)
    rows = [(r, t) for r, n in q_lens for t in range(-(-n // Q_TILE))]
    return torch.tensor(rows or [(0, 0)], dtype=torch.int32).to(
        device, non_blocking=True)


def split_plan(reqs, n_heads: int, device, target: int = 256, min_blocks: int = 16):
    """gfx950's KV-split prefill work for requests [(r, ctx, q_len, tok0)]:
    (tiles, mlist, n_slots) for attn_cdna.split_kernels_for.

    Each request's query tiles (attn_cdna.Q_TILE rows), longest first. When
    the tiles times the heads come short of `target` workgroups, a tile's KV
    walk is cut into splits over its prefix -- the blocks every one of its
    queries sees -- each at least `min_blocks` 64-key blocks; the last split
    takes the rest, its diagonal included. A tile entry is (request, tile,
    first block, end block or -1, slot or -1); a split tile's splits write
    partials to consecutive slots, and its merge entry is (first token,
    rows, first slot, splits)."""
    import torch  # noqa: PLC0415

    from .attn_cdna import KV_BLOCK as blk, Q_TILE as tile  # noqa: PLC0415

    tiles = []
    for r, ctx, n, tok0 in reqs:
        for y in reversed(range(-(-n // tile))):
            q0 = y * tile
            rows = min(tile, n - q0)
            work = -(-(ctx + q0 + rows) // blk)
            tiles.append((work, r, y, ctx, q0, rows, tok0))
    tiles.sort(key=lambda t: -t[0])
    want = -(-target // max(1, len(tiles) * n_heads))
    entries, merges, slots = [], [], 0
    for work, r, y, ctx, q0, rows, tok0 in tiles:
        prefix = (ctx + q0) // blk
        s = max(1, min(want, prefix // min_blocks))
        if s == 1:
            entries.append((work, (r, y, 0, -1, -1)))
            continue
        per = -(-prefix // s)
        s = -(-prefix // per)            # no empty split: its walk would be all masked
        cuts = [i * per for i in range(s)]
        for i, lo in enumerate(cuts):
            hi = cuts[i + 1] if i + 1 < s else -1
            entries.append(((hi if hi >= 0 else work) - lo, (r, y, lo, hi, slots + i)))
        merges.append((tok0 + q0, rows, slots, s))
        slots += s
    entries.sort(key=lambda e: -e[0])
    t = torch.tensor([e for _, e in entries] or [(0, 0, 0, -1, -1)], dtype=torch.int32)
    m = torch.tensor(merges or [(0, 0, 0, 0)], dtype=torch.int32)
    return t.to(device, non_blocking=True), m.to(device, non_blocking=True), len(merges), slots


_SPLIT_SCRATCH: dict = {}


def split_scratch(n_slots: int, n_heads: int, head_dim: int, device):
    """The KV-split prefill's partials: (n_slots * Q_TILE, heads * head_dim)
    bf16 and their (m, l), (1, heads, 2, n_slots * Q_TILE) fp32 -- kept and
    grown by powers of two, so a batch reuses the last one's."""
    import torch  # noqa: PLC0415

    from .attn_cdna import Q_TILE as tile  # noqa: PLC0415

    key = (str(device), n_heads, head_dim)
    have = _SPLIT_SCRATCH.get(key)
    if have is None or have[0] < n_slots:
        cap = 1 << max(0, (max(1, n_slots) - 1).bit_length())
        have = _SPLIT_SCRATCH[key] = (
            cap,
            torch.empty(cap * tile, n_heads * head_dim, device=device, dtype=torch.bfloat16),
            torch.empty(1, n_heads, 2, cap * tile, device=device, dtype=torch.float32))
    return have[1], have[2]


__all__ = ["paged_prefill_kernel", "kernel_for", "work_list", "split_plan", "split_scratch",
           "Q_TILE", "KV_BLOCK"]
