"""Flash attention backward for RDNA (gfx1201 first), written in the DSL: dQ, dK and dV.

FA-2's backward, in the orientations hk's forward (`hk.ops.attn`) already uses, so no layout is
new:

  * the forward keeps one more thing, the base-2 log-sum-exp of each query row's scaled scores
    (`attn_kernel(..., lse=True)`), and the backward rebuilds P = exp2(S * scale * log2 e - LSE)
    from it instead of redoing the online softmax;
  * Delta = rowsum(dO . O) per query row, once, before the two kernels below (torch, for now);
  * `dq_kernel` is the forward's structure with K and V in LDS and this warp's Q *and* dO in
    registers: S^T = K Q^T and dP^T = V dO^T are [kv, q] col tiles, so LSE and Delta are row
    vectors of them -- one value per query, on the lane axis, as the forward's running max is --
    and dS^T = P^T . (dP^T - Delta) is the B operand of dQ^T += K^T dS^T, with K^T staged
    transposed in LDS the way the forward stages V^T;
  * `dkdv_kernel` is the same algebra the other way round: a workgroup owns a block of KV rows
    and walks the Q blocks; S = Q K^T and dP = dO V^T are [q, kv] col tiles (the warp's own K
    and V in registers as the row-layout B operands), so P and dS are B operands of
    dV^T += dO^T P and dK^T += Q^T dS, with Q^T and dO^T staged transposed in LDS.

The softmax scale enters S once (in log2 units, folded with log2 e as in the forward) and the
outputs once: dQ and dK carry a factor `scale`, dV none.
"""

import math
from typing import Optional

from .. import dsl as hk
from ..types import bf16, fp32
from .attn import _derive

NEG_INF = float("-inf")
LOG2E = math.log2(math.e)


def dq_kernel(name: str, *, head_dim: int = 128, q_block: int = 16, kv_block: int = 32,
              warps: int = 12, causal: bool = False, scale: Optional[float] = None,
              bnhd: bool = False, setprio: bool = True, max_vgprs: int = 256,
              min_occupancy: int = 1):
    """dQ for one (head_dim, causal, scale). Arguments: q, k, v, do (bf16, the forward's
    layout), lse and delta (fp32 (B, H, 1, N): base-2 LSE of the scaled scores, rowsum(dO . O)),
    dq (bf16, q's layout)."""
    d = _derive(head_dim, q_block, kv_block, warps, 16, 2, 2)
    q_tile, d_frags, kv_frags = d["q_tile"], d["d_frags"], d["kv_frags"]
    d_chunks, v_bands, vt_frags = d["d_chunks"], d["v_bands"], d["vt_frags"]
    sc = scale if scale is not None else head_dim ** -0.5
    scale_l2e = sc * LOG2E

    st_k = hk.ST(bf16, kv_block, head_dim)
    st_kt = hk.ST(bf16, head_dim, kv_block)

    ty_q = hk.RT(bf16, 16, 16, "row")          # one d-slice of this warp's Q (and of its dO)
    ty_k = hk.RT(bf16, 16, 16, "row")          # one [kv, d] fragment of K or V
    ty_s = hk.RT(fp32, 16, q_block, "col")     # one [kv, q] fragment of S^T / P^T / dP^T / dS^T
    ty_dq = hk.RT(fp32, 16, q_block, "col")    # one [d, q] fragment of dQ^T
    ty_slab = hk.RT(bf16, 16, 16, "row")       # a [kv, d] slab of K, in flight to K^T
    ty_kt = hk.RT(bf16, 16, 16, "row")         # one [d, kv] fragment of K^T
    ty_vec = hk.row_vec(ty_s)                  # one entry per query

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
             dout: hk.Tensor[bf16], lse: hk.Tensor[fp32], delta: hk.Tensor[fp32],
             dq: hk.Tensor[bf16]):
        g = hk.group(warps)

        k_s = hk.lds.alloc(st_k, name="k_s")
        v_s = hk.lds.alloc(st_k, name="v_s")
        kt_s = hk.lds.alloc(st_kt, name="kt_s")
        buf_k = g.stage_buffer(k_s, name="buf_k")
        buf_v = g.stage_buffer(v_s, name="buf_v")
        gr_k = k_s.granules(name="gr_k")
        gr_v = v_s.granules(name="gr_v")
        gr_kt = kt_s.granules(name="gr_kt")

        b = hk.arch.block_idx.z
        head, qb = hk.arch.block_idx.y, hk.arch.block_idx.x
        n_q = q.rows
        n_kv = k.rows
        if causal:
            qb = hk.cdiv(n_q, q_tile) - 1 - qb      # longest first, as the forward
        if bnhd:
            head_kv = head // ((q.cols // head_dim) // (k.cols // head_dim))
        else:
            head_kv = head // (q.depth // k.depth)

        def at(t, h, row, col):
            if bnhd:
                return t[hk.elem_coord(b, 0, row, h * head_dim + col)]
            return t[hk.elem_coord(b, h, row, col)]

        q_tile_start = hk.max(hk.min(qb * q_tile, n_q - q_tile), 0)
        wid = hk.arch.warp_id()
        q_row = q_tile_start + wid * q_block

        q_r = [at(q, head, q_row, 16 * kk).load(ty_q) for kk in range(d_frags)]
        do_r = [at(dout, head, q_row, 16 * kk).load(ty_q) for kk in range(d_frags)]
        lse_v = lse[hk.elem_coord(b, head, 0, q_row)].load(ty_vec)
        dl_v = delta[hk.elem_coord(b, head, 0, q_row)].load(ty_vec)

        dq_c = [hk.zeros(ty_dq) for _ in range(d_frags)]
        k_rows = [hk.zeros(ty_slab) for _ in range(v_bands)]
        c_chunk = wid
        stages_kt = c_chunk < d_chunks

        kv_blocks = hk.cdiv(n_kv, kv_block)
        if causal:
            kv_blocks = hk.min(kv_blocks, hk.cdiv(q_tile_start + q_tile, kv_block))

        def block_start(kb):
            kb = hk.min(kb, kv_blocks - 1)
            return hk.min(kb * kv_block, n_kv - kv_block)

        def load_kv(kv_start):
            g.stage(buf_k, at(k, head_kv, kv_start, 0))
            g.stage(buf_v, at(v, head_kv, kv_start, 0))
            if stages_kt:
                for w in range(v_bands):
                    at(k, head_kv, kv_start + 16 * w, c_chunk * 16).load(ty_slab, into=k_rows[w])

        def commit_kv():
            hk.arch.vm_wait(0)
            g.commit(k_s, buf_k)
            g.commit(v_s, buf_v)
            if stages_kt:
                for w in range(v_bands):
                    k_t = hk.transpose_sep(k_rows[w])
                    for t in range(vt_frags):
                        gr_kt.store_frag(k_t, c_chunk * vt_frags + t, w)

        def block_math(kv_start, skip):
            with hk.scope():
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                # ---- S^T = K . Q^T ----------------------------------------------------
                for kk in range(d_frags):
                    for n0 in range(0, kv_frags, 2):
                        k_win = [gr_k.load_frag(n0 + i, kk, ty_k) for i in range(2)]
                        hk.lds.wait_for(*k_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(2):
                            s_f[n0 + i] += k_win[i] @ q_r[kk].T
                        if setprio:
                            hk.arch.setprio(0)
                for n in range(kv_frags):
                    s_f[n] *= scale_l2e
                if causal:
                    dd = q_row - kv_start
                    for n in range(kv_frags):
                        s_f[n].triu_(dd - 16 * n, NEG_INF)
                if skip > 0:
                    for n in range(kv_frags):
                        s_f[n].upper_fill_(skip - 16 * n, NEG_INF)
                # ---- P^T = exp2(S^T - LSE) -------------------------------------------
                for n in range(kv_frags):
                    s_f[n] -= lse_v
                    s_f[n].exp2_()
                # ---- dP^T = V . dO^T --------------------------------------------------
                dp_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n0 in range(0, kv_frags, 2):
                        v_win = [gr_v.load_frag(n0 + i, kk, ty_k) for i in range(2)]
                        hk.lds.wait_for(*v_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(2):
                            dp_f[n0 + i] += v_win[i] @ do_r[kk].T
                        if setprio:
                            hk.arch.setprio(0)
                # ---- dS^T = P^T . (dP^T - Delta); dQ^T += K^T . dS^T ------------------
                for n in range(kv_frags):
                    dp_f[n] -= dl_v
                    dp_f[n] *= s_f[n]
                    ds_cvt = dp_f[n].to(bf16)
                    for g0 in range(0, d_frags, 2):
                        kt_win = [gr_kt.load_frag(g0 + i, n, ty_kt) for i in range(2)]
                        hk.lds.wait_for(*kt_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(2):
                            dq_c[g0 + i] += kt_win[i] @ ds_cvt
                        if setprio:
                            hk.arch.setprio(0)

        load_kv(block_start(0))
        for kb in range(kv_blocks):
            kv_lo = kb * kv_block
            kv_start = hk.min(kv_lo, n_kv - kv_block)
            skip = kv_lo - kv_start
            hk.arch.barrier()
            commit_kv()
            hk.arch.barrier(drain=True)
            load_kv(block_start(kb + 1))
            block_math(kv_start, skip)

        for i, c in enumerate(dq_c):
            c *= sc
            at(dq, head, q_row, 16 * i).store(hk.transpose(c))

    def _grid(p):
        heads = p.q.cols // head_dim if bnhd else p.q.depth
        return (hk.cdiv(p.q.rows, q_tile), heads, p.q.batch)

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)




def dkdv_kernel(name: str, *, head_dim: int = 128, q_step: int = 16, warps: int = 4,
                causal: bool = False, scale: Optional[float] = None, bnhd: bool = False,
                setprio: bool = True, max_vgprs: int = 256, min_occupancy: int = 1):
    """dK and dV for one (head_dim, causal, scale). Arguments: q, k, v, do (bf16), lse and delta
    (fp32 (B, H, 1, N)), dk and dv (bf16, k's layout). A workgroup owns 16 * warps KV rows of one
    KV head -- 16 a warp, its K and V rows in registers -- and walks the Q blocks of every query
    head reading that KV head (GQA sums over them here, no atomics), q_step rows at a time staged
    through LDS as Q, dO and their transposes."""
    if head_dim % 16 or q_step % 16:
        raise ValueError(f"head_dim={head_dim}, q_step={q_step}: multiples of 16")
    d_frags, q_frags = head_dim // 16, q_step // 16
    kv_tile = 16 * warps
    sc = scale if scale is not None else head_dim ** -0.5
    scale_l2e = sc * LOG2E

    # Each pair that is read the same way shares one tile, side by side, and so one granule set:
    # Q | dO, Q^T | dO^T, and a warp's K | V. Six sets of address registers cost the 18 VGPRs
    # that spilled at head_dim 128; three do not.
    st_qo = hk.ST(bf16, q_step, 2 * head_dim)
    st_kv = hk.ST(bf16, 16, 2 * head_dim)     # one warp's K | V rows: in LDS, not registers
    ty_w = hk.RT(bf16, 16, head_dim, "row")


    ty_op = hk.RT(bf16, 16, 16, "row")         # a 16x16 operand fragment
    ty_s = hk.RT(fp32, 16, 16, "col")          # one [q, kv] fragment of S / P / dP / dS
    ty_acc = hk.RT(fp32, 16, 16, "col")        # one [d, kv] fragment of dK^T / dV^T
    ty_cv = hk.col_vec(ty_s)                   # one entry per query (a row of S)

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
             dout: hk.Tensor[bf16], lse: hk.Tensor[fp32], delta: hk.Tensor[fp32],
             dk: hk.Tensor[bf16], dv: hk.Tensor[bf16]):
        g = hk.group(warps)
        qo_s = hk.lds.alloc(st_qo, name="qo_s")
        gr_qo = qo_s.granules(name="gr_qo")

        # One 16-row slab a warp, stacked: a warp's own slab picked by its id has a granule set,
        # where a runtime-indexed subtile of one big tile does not (its reads would each hold an
        # address register, and at head_dim 128 those spill).
        kv_s = hk.lds.alloc(st_kv, name="kv_s", count=warps)

        b = hk.arch.block_idx.z
        head_kv, kb = hk.arch.block_idx.y, hk.arch.block_idx.x
        n_q = q.rows
        n_kv = k.rows
        if bnhd:
            group = (q.cols // head_dim) // (k.cols // head_dim)
        else:
            group = q.depth // k.depth

        def at(t, h, row, col):
            if bnhd:
                return t[hk.elem_coord(b, 0, row, h * head_dim + col)]
            return t[hk.elem_coord(b, h, row, col)]

        kv_tile_start = hk.max(hk.min(kb * kv_tile, n_kv - kv_tile), 0)
        wid = hk.arch.warp_id()
        kv_row = kv_tile_start + wid * 16

        # K and V are read once per Q fragment as the B operands of S and dP. Holding them in
        # registers next to the two accumulators (128 VGPRs at head_dim 128) spills, so the
        # workgroup's K and V rows live in LDS and each warp reads its own 16 rows of them.
        kv_s[wid].subtile(16, head_dim, 0, 0).store(at(k, head_kv, kv_row, 0).load(ty_w))
        kv_s[wid].subtile(16, head_dim, 0, 1).store(at(v, head_kv, kv_row, 0).load(ty_w))
        gr_kv = kv_s[wid].granules(name="gr_kv")
        dk_c = [hk.zeros(ty_acc) for _ in range(d_frags)]
        dv_c = [hk.zeros(ty_acc) for _ in range(d_frags)]

        q_blocks = hk.cdiv(n_q, q_step)
        qb0 = 0
        if causal:
            qb0 = kv_tile_start // q_step     # earlier Q blocks see none of these KV rows

        def block_start(qi):
            qi = hk.min(qi, q_blocks - 1)
            return hk.min(qi * q_step, n_q - q_step)

        def block_math(h, q_start, skip):
            for f in range(q_frags):
                with hk.scope():
                    qr0 = q_start + 16 * f
                    s = hk.zeros(ty_s)
                    for kk in range(d_frags):
                        qa = gr_qo.load_frag(f, kk, ty_op)
                        ka = gr_kv.load_frag(0, kk, ty_op)
                        hk.lds.wait_for(qa, ka)
                        s += qa @ ka.T
                    s *= scale_l2e
                    if causal:
                        # keep kv <= q: rows q = qr0 + r, columns kv = kv_row + c; fill c > r + qr0 -
                        # kv_row, which is tril's `c > r - row_idx` at row_idx = kv_row - qr0
                        s.tril_(kv_row - qr0, NEG_INF)
                    if skip > 0:
                        s.upper_fill_(skip - 16 * f, NEG_INF)
                    s -= lse[hk.elem_coord(b, h, 0, qr0)].load(ty_cv)
                    s.exp2_()                                    # P
                    # One read of each dO fragment feeds both dP = dO V^T and, transposed in
                    # registers (transpose_sep), dV^T += dO^T P: no second, transposed copy in
                    # LDS, whose 16 KB kept the workgroups a WGP holds at two.
                    p16 = s.to(bf16)
                    dp = hk.zeros(ty_s)
                    for kk in range(d_frags):
                        oa = gr_qo.load_frag(f, d_frags + kk, ty_op)
                        va = gr_kv.load_frag(0, d_frags + kk, ty_op)
                        hk.lds.wait_for(oa, va)
                        dp += oa @ va.T
                        dv_c[kk] += hk.transpose_sep(oa) @ p16
                    dp -= delta[hk.elem_coord(b, h, 0, qr0)].load(ty_cv)
                    dp *= s                                      # dS
                    ds16 = dp.to(bf16)
                    for kk in range(d_frags):
                        qa = gr_qo.load_frag(f, kk, ty_op)
                        hk.lds.wait_for(qa)
                        dk_c[kk] += hk.transpose_sep(qa) @ ds16

        # Not prefetched: the next block's Q and dO in flight across the math would hold their
        # staging registers (64 a lane at head_dim 128, four warps) next to the two
        # accumulators, and that spills. Several workgroups a WGP cover the load instead.
        for hh in range(group):
            h = head_kv * group + hh
            for qi in range(qb0, q_blocks):
                q_lo = qi * q_step
                q_start = hk.min(q_lo, n_q - q_step)
                skip = q_lo - q_start
                hk.arch.barrier()
                # synchronous workgroup copies: no staging buffer held for the kernel's life
                g.load(qo_s.subtile(q_step, head_dim, 0, 0), at(q, h, q_start, 0))
                g.load(qo_s.subtile(q_step, head_dim, 0, 1), at(dout, h, q_start, 0))
                hk.arch.barrier(drain=True)
                block_math(h, q_start, skip)

        for i in range(d_frags):
            c = dk_c[i]
            c *= sc
            at(dk, head_kv, kv_row, 16 * i).store(hk.transpose(c))
            at(dv, head_kv, kv_row, 16 * i).store(hk.transpose(dv_c[i]))

    def _grid(p):
        heads = p.k.cols // head_dim if bnhd else p.k.depth
        return (hk.cdiv(p.k.rows, kv_tile), heads, p.k.batch)

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)




def dkv_kernel(name: str, *, mode: str, head_dim: int = 128, kv_rows: int = 16, q_block: int = 32,
               warps: int = 12, causal: bool = False, scale: Optional[float] = None,
               bnhd: bool = False, setprio: bool = True, max_vgprs: int = 256,
               min_occupancy: int = 1):
    """dK (mode="dk") or dV (mode="dv"): `dq_kernel` with the roles of Q and KV swapped. A warp
    owns 16 KV rows -- K (and for dK, V) in registers -- and the workgroup walks the Q blocks,
    staging Q and dO through LDS, prefetched one block ahead exactly as the forward and dQ stage
    K and V. Two kernels rather than one fused dK/dV: the fused one holds both accumulators
    (128 VGPRs at head_dim 128), cannot keep K and V in registers or prefetch, and measured
    ~29 TF; each of these holds one accumulator and recomputes S (dV: S, dV; dK: S, dP, dK).

    Orientation: S = Q K^T and dP = dO V^T are [q, kv] col tiles (the warp's KV rows on the
    lane axis), so LSE and Delta are column vectors of them, and P / dS are the B operands of
    dV^T += dO^T P and dK^T += Q^T dS, with dO^T / Q^T staged transposed in LDS.

    Arguments: q, k, v, do (bf16), lse and delta (fp32 (B, H, 1, N)), out (dk or dv, bf16, k's
    layout). GQA sums over the query heads of a KV head inside the kernel."""
    if mode not in ("dk", "dv"):
        raise ValueError(f"mode is 'dk' or 'dv', not {mode!r}")
    if kv_rows != 16:
        raise ValueError("kv_rows: 16 (one fragment a warp)")
    dk_mode = mode == "dk"
    d = _derive(head_dim, kv_rows, q_block, warps, 16, 2, 2)
    kv_tile, d_frags, q_frags = d["q_tile"], d["d_frags"], d["kv_frags"]
    d_chunks, q_bands, t_frags = d["d_chunks"], d["v_bands"], d["vt_frags"]
    sc = scale if scale is not None else head_dim ** -0.5
    scale_l2e = sc * LOG2E

    st_q = hk.ST(bf16, q_block, head_dim)
    st_qt = hk.ST(bf16, head_dim, q_block)

    ty_op = hk.RT(bf16, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, 16, "col")          # one [q, kv] fragment of S / P / dP / dS
    ty_acc = hk.RT(fp32, 16, 16, "col")        # one [d, kv] fragment of dK^T or dV^T
    ty_cv = hk.col_vec(ty_s)                   # one entry per query: a row of S

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
             dout: hk.Tensor[bf16], lse: hk.Tensor[fp32], delta: hk.Tensor[fp32],
             out: hk.Tensor[bf16]):
        g = hk.group(warps)
        # Q is S's A operand in both modes; dO is dP's in dK mode; the transposed tile is
        # Q^T (dK) or dO^T (dV).
        q_s = hk.lds.alloc(st_q, name="q_s")
        buf_q = g.stage_buffer(q_s, name="buf_q")
        gr_q = q_s.granules(name="gr_q")
        if dk_mode:
            o_s = hk.lds.alloc(st_q, name="do_s")
            buf_o = g.stage_buffer(o_s, name="buf_do")
            gr_o = o_s.granules(name="gr_do")
        t_s = hk.lds.alloc(st_qt, name="t_s")
        gr_t = t_s.granules(name="gr_t")

        b = hk.arch.block_idx.z
        head_kv, kb = hk.arch.block_idx.y, hk.arch.block_idx.x
        n_q = q.rows
        n_kv = k.rows
        if bnhd:
            group = (q.cols // head_dim) // (k.cols // head_dim)
        else:
            group = q.depth // k.depth

        def at(t, h, row, col):
            if bnhd:
                return t[hk.elem_coord(b, 0, row, h * head_dim + col)]
            return t[hk.elem_coord(b, h, row, col)]

        kv_tile_start = hk.max(hk.min(kb * kv_tile, n_kv - kv_tile), 0)
        wid = hk.arch.warp_id()
        kv_row = kv_tile_start + wid * 16

        k_r = [at(k, head_kv, kv_row, 16 * kk).load(ty_op) for kk in range(d_frags)]
        if dk_mode:
            v_r = [at(v, head_kv, kv_row, 16 * kk).load(ty_op) for kk in range(d_frags)]
        acc = [hk.zeros(ty_acc) for _ in range(d_frags)]
        t_rows = [hk.zeros(ty_op) for _ in range(q_bands)]
        c_chunk = wid
        stages_t = c_chunk < d_chunks
        src_t = q if dk_mode else dout        # what is staged transposed

        q_blocks = hk.cdiv(n_q, q_block)
        qb0 = 0
        if causal:
            qb0 = kv_tile_start // q_block    # earlier Q blocks see none of these KV rows

        def block_start(h, qi):
            qi = hk.min(qi, q_blocks - 1)
            return hk.min(qi * q_block, n_q - q_block)

        def load_q(h, q_start):
            g.stage(buf_q, at(q, h, q_start, 0))
            if dk_mode:
                g.stage(buf_o, at(dout, h, q_start, 0))
            if stages_t:
                for w in range(q_bands):
                    at(src_t, h, q_start + 16 * w, c_chunk * 16).load(ty_op, into=t_rows[w])

        def commit_q():
            hk.arch.vm_wait(0)
            g.commit(q_s, buf_q)
            if dk_mode:
                g.commit(o_s, buf_o)
            if stages_t:
                for w in range(q_bands):
                    tt = hk.transpose_sep(t_rows[w])
                    for t in range(t_frags):
                        gr_t.store_frag(tt, c_chunk * t_frags + t, w)

        def block_math(h, q_start, skip):
            with hk.scope():
                # ---- S = Q . K^T, P = exp2(S - LSE) ------------------------------------
                s_f = [hk.zeros(ty_s) for _ in range(q_frags)]
                for kk in range(d_frags):
                    for n0 in range(0, q_frags, 2):
                        q_win = [gr_q.load_frag(n0 + i, kk, ty_op) for i in range(2)]
                        hk.lds.wait_for(*q_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(2):
                            s_f[n0 + i] += q_win[i] @ k_r[kk].T
                        if setprio:
                            hk.arch.setprio(0)
                for n in range(q_frags):
                    s_f[n] *= scale_l2e
                    if causal:
                        # rows q = q_start + 16n + r, columns kv = kv_row + c: fill kv > q
                        s_f[n].tril_(kv_row - q_start - 16 * n, NEG_INF)
                if skip > 0:
                    for n in range(q_frags):
                        s_f[n].upper_fill_(skip - 16 * n, NEG_INF)
                for n in range(q_frags):
                    s_f[n] -= lse[hk.elem_coord(b, h, 0, q_start + 16 * n)].load(ty_cv)
                    s_f[n].exp2_()
                if dk_mode:
                    # ---- dP = dO . V^T; dS = P . (dP - Delta) ----------------------------
                    dp_f = [hk.zeros(ty_s) for _ in range(q_frags)]
                    for kk in range(d_frags):
                        for n0 in range(0, q_frags, 2):
                            o_win = [gr_o.load_frag(n0 + i, kk, ty_op) for i in range(2)]
                            hk.lds.wait_for(*o_win)
                            if setprio:
                                hk.arch.setprio(1)
                            for i in range(2):
                                dp_f[n0 + i] += o_win[i] @ v_r[kk].T
                            if setprio:
                                hk.arch.setprio(0)
                    for n in range(q_frags):
                        dp_f[n] -= delta[hk.elem_coord(b, h, 0, q_start + 16 * n)].load(ty_cv)
                        dp_f[n] *= s_f[n]
                    b_f = dp_f
                else:
                    b_f = s_f
                # ---- dK^T += Q^T . dS  |  dV^T += dO^T . P ----------------------------------
                for n in range(q_frags):
                    b16 = b_f[n].to(bf16)
                    for g0 in range(0, d_frags, 2):
                        t_win = [gr_t.load_frag(g0 + i, n, ty_op) for i in range(2)]
                        hk.lds.wait_for(*t_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(2):
                            acc[g0 + i] += t_win[i] @ b16
                        if setprio:
                            hk.arch.setprio(0)

        for hh in range(group):
            h = head_kv * group + hh
            load_q(h, block_start(h, qb0))
            for qi in range(qb0, q_blocks):
                q_lo = qi * q_block
                q_start = hk.min(q_lo, n_q - q_block)
                skip = q_lo - q_start
                hk.arch.barrier()
                commit_q()
                hk.arch.barrier(drain=True)
                load_q(h, block_start(h, qi + 1))
                block_math(h, q_start, skip)

        for i, c in enumerate(acc):
            if dk_mode:
                c *= sc
            at(out, head_kv, kv_row, 16 * i).store(hk.transpose(c))

    def _grid(p):
        heads = p.k.cols // head_dim if bnhd else p.k.depth
        return (hk.cdiv(p.k.rows, kv_tile), heads, p.k.batch)

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)




def delta_kernel(name: str, *, head_dim: int = 128, warps: int = 4, bnhd: bool = False):
    """Delta = rowsum(dO . O) in fp32, the backward's per-row constant: o and do (bf16, the
    forward's layout) -> delta, fp32 (B, H, 1, N_pad) with N_pad a multiple of 16 * warps
    (rows past N read zeros and write zeros there, which nothing reads). 16 rows a warp."""
    ty = hk.RT(bf16, 16, head_dim, "row")
    rows = 16 * warps

    def body(o: hk.Tensor[bf16], dout: hk.Tensor[bf16], delta: hk.Tensor[fp32]):
        b = hk.arch.block_idx.z
        head, blk = hk.arch.block_idx.y, hk.arch.block_idx.x
        row = blk * rows + hk.arch.warp_id() * 16

        def at(t):
            if bnhd:
                return t[hk.elem_coord(b, 0, row, head * head_dim)]
            return t[hk.elem_coord(b, head, row, 0)]

        x = at(o).load(ty, bounded=True).to(fp32)
        x *= at(dout).load(ty, bounded=True).to(fp32)
        delta[hk.elem_coord(b, head, 0, row)].store(x.sum(axis=1))

    def _grid(p):
        heads = p.o.cols // head_dim if bnhd else p.o.depth
        return (hk.cdiv(p.o.rows, rows), heads, p.o.batch)

    return hk.kernel(body, warps=warps, grid=_grid, name=name)


__all__ = ["dq_kernel", "dkdv_kernel", "dkv_kernel", "delta_kernel"]
