"""K3 candidate c1: hk's gfx1100 DiT attention restructured as the Sage experiment's hk arm wrote it
(~/kda-sage-hk/kda/sage_hk.py, its own kernel, credited there), with bf16 Q/K on the bf16 WMMA instead of
int8: S^T = K . Q^T kept as [kv, q]; online softmax in log2 units; V pre-transposed once into fp16 V^T
(the even/odd banded column order kda::pair_p's P^T operands use); O^T += V^T . P^T on the fp16 WMMA;
the grid's tail split over KV with a combine. One escape hatch remains: kda::pair_p, a source rewrite
of the fp32 -> fp16 P^T pair conversion (hk has no op for that exchange).
"""
import math
import re

import hk
from hk import dsl as _dsl
from hk.dsl.collective import cross_warp, fold_rows
from hk.types import bf16, fp16, fp32, i8, i32

HEAD_DIM = 128
Q_BLK = 128                  # rows of Q a work unit
_TINY = 1e-12

_PAIR_HELPER = r"""
namespace kda {
typedef float f2_t __attribute__((ext_vector_type(2)));
typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
// P^T for the PV WMMA, two S^T fragments at a time (pair_p). src: rt<float, 32, 16, col>, the
// fp32 accumulators of kv rows [0, 16) and [16, 32) of a 32-row pair: lane l, element e holds kv
// row 2e + h of its fragment (h = l >> 4), query l & 15. dst: rt<half, 32, 16, col>; tile 0 is
// the operand whose K axis is the pair's 16 even rows (0, 2, .., 30), tile 1 the odd rows -- the
// order vt_kernel writes V^T's columns in. A lane's own values are then already one of the two
// operands, in order; the other comes from the mirror lane: per packed dword one cvt pair, one
// permlanex16 and two selects, against hk's per-element exchange (cvt pair + permlane + two
// selects per *element*) for the natural order.
template<typename DT, typename ST>
__device__ static inline void pair_p(DT &dst, const ST &src) {
    static_assert(ST::height == 2 && ST::width == 1 && DT::height == 2 && DT::width == 1, "a pair");
    static_assert(sizeof(src.tiles[0][0].data) == 32 && sizeof(dst.tiles[0][0].data) == 32, "16x16");
    const bool h = (__lane_id() >> 4) != 0;
    #pragma unroll
    for (int f = 0; f < 2; f++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            f2_t x;
            __builtin_memcpy(&x, (const char *)&src.tiles[f][0].data[0] + 8 * q, 8);
            const h2_t hh = __builtin_convertvector(x, h2_t);     // v_cvt_f16_f32: nearest even
            int own;
            __builtin_memcpy(&own, &hh, 4);
            const int oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210, 0xfedcba98, false, false);
            const int ev = h ? oth : own, od = h ? own : oth;
            __builtin_memcpy((char *)&dst.tiles[0][0].data[0] + 4 * (4 * f + q), &ev, 4);
            __builtin_memcpy((char *)&dst.tiles[1][0].data[0] + 4 * (4 * f + q), &od, 4);
        }
    }
}
}  // namespace kda
"""


class PairPKernel(_dsl.Kernel):
    """An hk kernel whose 32-row P^T conversions are kda::pair_p (see the module docstring)."""

    def __init__(self, fn, *, n_pair_sites: int = 0, **kw):
        super().__init__(fn, **kw)
        self.n_pair_sites = n_pair_sites

    def with_defines(self, src: str) -> str:
        # Every build path (Kernel.source for pybind, hk.runtime.codeobj.load for the
        # device-only code object) passes the generated C++ through here.
        src = super().with_defines(src)
        inc = '#include "kittens.cuh"\n'
        if inc not in src:
            raise RuntimeError(f"{self.name}: no kittens include to anchor the pair_p helper")
        src = src.replace(inc, inc + _PAIR_HELPER, 1)
        # `pair.to(fp16)` on a [32, 16] fp32 S^T pair -> kda::pair_p (the even/odd band order).
        names = set(re.findall(r"rt_fp16_32x16_col (\w+);", src))
        pat = re.compile(r"kittens::copy\((\w+), (\w+)\);")
        n = 0

        def sub(m):
            nonlocal n
            if m.group(1) in names:
                n += 1
                return f"kda::pair_p({m.group(1)}, {m.group(2)});"
            return m.group(0)
        src = pat.sub(sub, src)
        if n != self.n_pair_sites:
            raise RuntimeError(f"{self.name}: expected {self.n_pair_sites} P pair conversions, "
                               f"found {n}")
        return src

    def preloaded(self):
        # hk's preloaded() rebuilds the kernel through type(self)(...) without our arguments.
        return self


# ---------------------------------------------------------------- V^T


def vt_kernel(name: str, warps: int = 2):
    """V^T for the attention: vt [B, H, 128, N] fp16 from bf16 V, with the columns of every 32-row
    pair in kda::pair_p's order (the 16 even kv rows, then the 16 odd). A WMMA against the identity
    (bf16 V . I in fp32, exact) lands V in the accumulator layout, stored transposed and narrowed
    to fp16 (the hk arm's vt_kernel, bf16 in)."""
    ta = hk.RT(bf16, 16, 16, "row")
    te = hk.RT(bf16, 16, 16, "col")
    tacc = hk.RT(fp32, 64, 16, "col")

    def body(v: hk.Tensor[bf16], eye: hk.Tensor[bf16], vt: hk.Tensor[fp16]):
        b = hk.arch.block_idx.z
        h = hk.arch.block_idx.y
        row = hk.arch.block_idx.x * (64 * warps) + hk.arch.warp_id() * 64
        e = eye[hk.elem_coord(0, 0, 0, 0)].load(te)
        for dd in range(HEAD_DIM // 16):
            acc = hk.zeros(tacc)
            for pr in range(2):
                for par in range(2):
                    x = v[hk.elem_coord(b, h, (row + 32 * pr) // 2, HEAD_DIM * par + 16 * dd)].load(ta)
                    sub = acc.subtile(16, 2 * pr + par)
                    sub += x @ e
            vt[hk.elem_coord(b, h, 16 * dd, row)].store(hk.transpose(acc))

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (hk.cdiv(p.v.rows * 2, 64 * warps), p.v.depth, p.v.batch))


# ---------------------------------------------------------------- attention


def attn_kernel(name: str, *, kv_block: int = 32, warps: int = 8, qk_tiles: int = 2,
                pv_tiles: int = 2, setprio: bool = True, max_vgprs: int = 256, stages: int = 1,
                lazy: float = 0.0, _ablate: str = ""):
    """One workgroup a work unit (a 128-row Q block, or a KV part of one for the grid's tail).
    q: [1, 1, R, 128] bf16 (R = B*H*N rows); k: [1, G, N_kv, 128] bf16; v: [1, G, 128, N_kv] fp16
    V^T (banded); o: [1, 1, R, 128] bf16; wp/wu/op/ml as the hk arm's (split plan, partials).
    stages=2: two KV blocks in flight in registers (the loop unrolled by two over the buffer's slots,
    a part's block count rounded up to even), so a block's commit waits only for its own loads.
    lazy > 0: keep the running max stale until a row's block max exceeds it by more than `lazy`
    (log2 units): P <= 2^lazy (fp16-safe for lazy <= 15), and the rescale of O and l -- 64 + 8
    multiplies a lane -- runs on a handful of blocks instead of every block where some max moved."""
    head_dim = HEAD_DIM
    q_block = 16
    q_tile = q_block * warps
    assert q_tile == Q_BLK
    d_frags = head_dim // 16
    kv_frags = kv_block // 16
    assert kv_block % (16 * qk_tiles) == 0 and d_frags % pv_tiles == 0 and kv_block % 32 == 0
    scale_l2e = head_dim ** -0.5 * math.log2(math.e)
    assert stages in (1, 2)
    # global loads a lane issues for one KV block (16 B each), K and V^T
    lane_loads = 2 * (kv_block * head_dim * 2) // (warps * 32 * 16)

    st_k = hk.ST(bf16, kv_block, head_dim)
    st_vt = hk.ST(fp16, head_dim, kv_block)
    ty_q = hk.RT(bf16, 16, 16, "row")
    ty_k = hk.RT(bf16, 16, 16, "row")
    ty_sw = hk.RT(fp32, kv_block, q_block, "col")
    ty_o = hk.RT(fp32, 16, q_block, "col")
    ty_vr = hk.RT(fp16, 16, 16, "row")
    ty_vec = hk.row_vec(hk.RT(fp32, 16, q_block, "col"))

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[fp16], o: hk.Tensor[bf16],
             wp: hk.Tensor[i32], wu: hk.Tensor[i32], op: hk.Tensor[fp32], ml: hk.Tensor[fp32]):
        g = hk.group(warps)
        k_s = hk.lds.alloc(st_k, name="k_s")
        vt_s = hk.lds.alloc(st_vt, name="vt_s")
        buf_k = g.stage_buffer(k_s, depth=stages, name="buf_k")
        buf_vt = g.stage_buffer(vt_s, depth=stages, name="buf_vt")
        gr_k = k_s.granules(name="gr_k")
        gr_vt = vt_s.granules(name="gr_vt")

        u = hk.arch.block_idx.x
        n_full = wp[hk.elem_coord(0, 0, 0, 0)].load_scalar(invariant=True)
        parts = wp[hk.elem_coord(0, 0, 0, 1)].load_scalar(invariant=True)
        nqb = wp[hk.elem_coord(0, 0, 0, 2)].load_scalar(invariant=True)
        ts = hk.max(u - n_full + 1, 0)
        is_s = hk.min(ts, 1)
        t = ts - is_s
        blk = hk.min(u, n_full) + t // parts
        part = t % parts
        head = blk // nqb
        n_kv = k.rows
        kv_total = n_kv // kv_block
        per = hk.cdiv(kv_total, 1 + (parts - 1) * is_s)
        if stages == 2:
            per = 2 * hk.cdiv(per, 2)
        kb_lo = part * per
        kb_hi = hk.min(kb_lo + per, kv_total)

        wid = hk.arch.warp_id()
        q_row = blk * q_tile + wid * q_block
        q_r = [q[hk.elem_coord(0, 0, q_row, 16 * kk)].load(ty_q) for kk in range(d_frags)]

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        def load_kv(kb, slot=0):
            kv0 = hk.min(kb, kb_hi - 1) * kv_block
            g.stage(buf_k, k[hk.elem_coord(0, head, kv0, 0)], slot)
            g.stage(buf_vt, v[hk.elem_coord(0, head, 0, kv0)], slot)

        def commit_kv(slot=0):
            # the other slot's loads (issued after this one's) may stay in flight
            hk.arch.vm_wait(lane_loads * (stages - 1))
            g.commit(k_s, buf_k, slot)
            g.commit(vt_s, buf_vt, slot)

        for sl in range(stages):
            load_kv(kb_lo + sl, sl)

        for i_kb in range((kb_hi - kb_lo) // stages):
            for half in range(stages):
                kb = kb_lo + i_kb * stages + half
                hk.arch.barrier()
                commit_kv(half)
                hk.arch.barrier(drain=True)
                load_kv(kb + stages, half)

                with hk.scope():
                    k_win, v_win = [], []
                    s_w = hk.zeros(ty_sw)
                    s_f = [s_w.subtile(16, n) for n in range(kv_frags)]
                    for kk in range(d_frags):
                        for n0 in range(0, kv_frags, qk_tiles):
                            if not ("kread" in _ablate and kk % 2):   # timing-only ablation: wrong
                                k_win[:] = [gr_k.load_frag(n0 + i, kk, ty_k) for i in range(qk_tiles)]
                                hk.lds.wait_for(*k_win)
                            if setprio:
                                hk.arch.setprio(1)
                            for i in range(qk_tiles):
                                s_f[n0 + i] += k_win[i] @ q_r[kk].T
                            if setprio:
                                hk.arch.setprio(0)

                    # log2 units: y = S * c, c = sm_scale * log2e; p = exp2(S * c - m)
                    m_old = m_run.copy()
                    m_f = s_f[0].max(axis=0).value()
                    for n in range(1, kv_frags):
                        m_f.max_(s_f[n].max(axis=0))
                    m_f *= scale_l2e
                    if lazy:
                        if m_f.any_above(m_run, lazy):
                            m_run.max_(m_f)
                            alpha = (m_old - m_run).exp2()
                            l_run *= alpha
                            for c in o_c:
                                c *= alpha
                    else:
                        m_run.max_(m_f)
                    for n in range(kv_frags):
                        s_f[n] *= scale_l2e
                        s_f[n] -= m_run
                        if "noexp" not in _ablate:
                            s_f[n].exp2_()
                    if not lazy:
                        if m_run.any_ne(m_old):
                            alpha = (m_old - m_run).exp2()
                            l_run *= alpha
                            for c in o_c:
                                c *= alpha
                    for n in range(kv_frags):
                        l_run += s_f[n].sum(axis=0)

                    for j in range(kv_frags // 2):
                        p_pair = s_w.subtile(32, j).to(fp16)
                        for par in range(2):
                            p_cvt = p_pair.subtile(16, par)
                            for g0 in range(0, d_frags, pv_tiles):
                                if not ("vread" in _ablate and g0 % (2 * pv_tiles)):
                                    v_win[:] = [gr_vt.load_frag(g0 + i, 2 * j + par, ty_vr)
                                                for i in range(pv_tiles)]
                                    hk.lds.wait_for(*v_win)
                                if setprio:
                                    hk.arch.setprio(1)
                                for i in range(pv_tiles):
                                    o_c[g0 + i] += v_win[i] @ p_cvt
                                if setprio:
                                    hk.arch.setprio(0)

        if is_s > 0:
            p_row = t * q_tile + wid * q_block
            for i, c in enumerate(o_c):
                op[hk.elem_coord(0, 0, p_row, 16 * i)].store(hk.transpose(c))
            ml[hk.elem_coord(0, 0, 0, p_row)].store(m_run)
            ml[hk.elem_coord(0, 0, 1, p_row)].store(l_run)
        if is_s == 0:
            for i, c in enumerate(o_c):
                c /= l_run
                o[hk.elem_coord(0, 0, q_row, 16 * i)].store(hk.transpose(c))

    body.__name__ = name
    return PairPKernel(body, n_pair_sites=stages * (kv_frags // 2), warps=warps, name=name,
                       grid=lambda p: (p.wu.cols, 1, 1), max_vgprs=max_vgprs, min_occupancy=1)


def combine_kernel(name: str, warps: int = 8):
    t = hk.RT(fp32, 16, HEAD_DIM)
    vc = hk.col_vec(t)

    def body(op: hk.Tensor[fp32], ml: hk.Tensor[fp32], wp: hk.Tensor[i32], wc: hk.Tensor[i32],
             o: hk.Tensor[bf16]):
        n_full = wp[hk.elem_coord(0, 0, 0, 0)].load_scalar(invariant=True)
        parts = wp[hk.elem_coord(0, 0, 0, 1)].load_scalar(invariant=True)
        tb = hk.arch.block_idx.x
        wid = hk.arch.warp_id()
        row = (n_full + tb) * (16 * warps) + wid * 16
        base = tb * parts * (16 * warps) + wid * 16
        m = hk.neg_infty(vc)
        for p in range(parts):
            m.max_(ml[hk.elem_coord(0, 0, 0, base + p * (16 * warps))].load(vc))
        acc = hk.zeros(t)
        l = hk.zeros(vc)
        for p in range(parts):
            r = base + p * (16 * warps)
            w = ml[hk.elem_coord(0, 0, 0, r)].load(vc)
            w -= m
            w.exp2_()
            lp = ml[hk.elem_coord(0, 0, 1, r)].load(vc)
            lp *= w
            l += lp
            x = op[hk.elem_coord(0, 0, r, 0)].load(t)
            x *= w
            acc += x
        acc /= l
        o[hk.elem_coord(0, 0, row, 0)] = acc

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, grid=lambda p: (p.wc.cols, 1, 1))


VT = vt_kernel("k3_vt")
ATTN = attn_kernel("k3_attn")
COMBINE = combine_kernel("k3_combine")
_EYE = {}
_PLANS = {}
_CO = {}


def plan(nblk: int, device) -> tuple:
    import torch

    wgps = torch.cuda.get_device_properties(device).multi_processor_count
    slots = 4 * wgps
    rem = nblk % slots
    parts = min(slots // rem, 4) if rem else 1
    if parts < 2:
        return nblk, 1, 0
    return nblk - rem, parts, rem


def _launch(kern, *tensors):
    import torch
    from hk.runtime import codeobj

    kk = _CO.get(id(kern))
    if kk is None:
        kk = _CO[id(kern)] = codeobj.load(kern)
    shapes, ptrs, rows = codeobj.tensor_args(kk, tensors, kern.name)
    dev = tensors[0].device
    kk.launch(dev.index, torch.cuda.current_stream(dev).cuda_stream, shapes, ptrs, rows)


def _plan_tensors(nblk, nqb, dev):
    import torch

    key = (nblk, nqb, dev)
    pl = _PLANS.get(key)
    if pl is None:
        n_full, parts, n_split = plan(nblk, dev)
        units = n_full + n_split * parts
        pl = _PLANS[key] = (
            torch.tensor([n_full, parts, nqb, n_split], device=dev, dtype=torch.int32).view(1, 1, 1, 4),
            torch.empty((1, 1, 1, units), device=dev, dtype=torch.int32),
            torch.empty((1, 1, 1, max(n_split, 1)), device=dev, dtype=torch.int32), n_split, parts)
    return pl


def attention(q, k, v, *, attn=None):
    """Non-causal attention, q, k, v [B, H, N, 128] bf16 (N, N_kv multiples of 128) -> o bf16."""
    import torch

    attn = attn or ATTN
    b, h, n, d = q.shape
    n_kv = k.shape[2]
    assert d == HEAD_DIM and n % Q_BLK == 0 and n_kv % 128 == 0, (q.shape, k.shape)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dev = q.device
    g = b * h
    vt = torch.empty((1, g, d, n_kv), device=dev, dtype=torch.float16)
    eye = _EYE.get(dev)
    if eye is None:
        eye = _EYE[dev] = torch.eye(16, device=dev, dtype=torch.bfloat16).view(1, 1, 16, 16)
    _launch(VT, v.view(1, g, n_kv // 2, 2 * d), eye, vt)
    o = torch.empty((1, 1, g * n, d), device=dev, dtype=torch.bfloat16)
    wp, wu, wc, n_split, parts = _plan_tensors(g * n // Q_BLK, n // Q_BLK, dev)
    su = max(n_split * parts, 1) * Q_BLK
    op = torch.empty((1, 1, su, d), device=dev, dtype=torch.float32)
    ml = torch.empty((1, 1, 2, su), device=dev, dtype=torch.float32)
    _launch(attn, q.view(1, 1, g * n, d), k.view(1, g, n_kv, d), vt, o, wp, wu, op, ml)
    if n_split:
        _launch(COMBINE, op, ml, wp, wc, o)
    return o.view(b, h, n, d)
