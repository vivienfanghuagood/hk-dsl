"""gfx1100 non-causal attention, head_dim 128, bf16: the DiT kernel (K3).

The structure the KDA SageAttention experiment's hk arm arrived at, in bf16
(docs/design/e2e-goal.md, K3): S^T = K . Q^T kept [kv, q] on the bf16 WMMA, so
the softmax's reductions over kv are within a lane plus one exchange; online
softmax in log2 units with a lazy running max (rescale O and l only when a
row's block max passes the stale one by more than LAZY, as FA-4 does); V
transposed once, before the attention, into fp16 V^T whose columns run in the
order the P^T operand is cheapest to build in; O^T += V^T . P^T on the fp16
WMMA; the grid's tail of Q blocks split over KV with a combine.

Three launches: `vt` (V^T, ~3% of a Qwen-Image call), `attn`, and `combine`
when the Q blocks do not fill the last wave.

Layouts: (B, H, N, D) contiguous, or (B, H, N, D) views of contiguous
(B, N, H, D) memory -- what SGLang-Diffusion hands SDPA -- read and written in
place. Both are addressed as (batch', 1, N, heads' * 128): BHND is batch' =
B * H, heads' = 1; BNHD is batch' = B, heads' = H.

Any N >= 128 (Q) and N_kv >= 32: the last Q block backs up to end at N and
recomputes the overlap (idempotent: a row's output depends on that row only),
the last KV block masks its rows past N_kv, and V^T is zero-padded to a
multiple of 128: vt_kernel converts the whole 128-row chunks (its loads are not
bounded by the view's rows, so it is never pointed past N_kv) and torch the last
partial one, padded with zeros. With an odd N_kv the even/odd row views have no batch stride
hk can carry (it derives one from rows x row stride, (N+1) rows' worth), so V^T
takes a launch a batch -- BNHD's B, small -- or, past 8 batches, torch's padded
permute-copy (4x the kernel's time).

One escape hatch: `kda::pair_p`, the fp32 -> fp16 conversion of a 32-row S^T
pair into its two P^T operands, is spliced into the generated C++ (`PairPKernel`)
-- hk has no op for that exchange.

Measured on a W7900D (interleaved, in place on BNHD): Qwen-Image B1 H32 N4096
76.7 TF/s, MiniMax-H3 B1 H56 N14720 81.7 TF/s, against 63.2 / 68.8 for
hk.ops.attn's kernel and 54.9 / 56.6 for tuned Triton FA-2; the attainable
WMMA rate on the card is ~100 TF/s.
"""

from __future__ import annotations

import math
import re

from .. import dsl as _dsl
from ..types import bf16, fp16, fp32, i32

HEAD_DIM = 128
Q_BLK = 128          # rows of Q a work unit: 8 warps x 16
KV_BLOCK = 32
LAZY = 8.0           # log2 units: P <= 2^8 before a rescale, fp16-safe
NEG_INF = float("-inf")

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
    """An hk kernel whose 32-row P^T conversions are kda::pair_p (module docstring)."""

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


def vt_kernel(name: str = "attn_k3_vt", warps: int = 2):
    """vt [1, G, 128, N_pad] fp16 from bf16 V, G = batch' * heads', the columns of every 32-row
    group in kda::pair_p's order (its 16 even kv rows, then the 16 odd). ve / vo: V's even / odd
    rows, [batch', 1, ceil(N/2) / floor(N/2), heads' * 128] with a row stride of two rows. A WMMA
    against the identity (bf16 V . I in fp32, exact) lands V in the accumulator layout, stored
    transposed and narrowed to fp16. Past N the loads read zeros: V^T's padding is 0."""
    import hk  # noqa: PLC0415

    ta = hk.RT(bf16, 16, 16, "row")
    te = hk.RT(bf16, 16, 16, "col")
    tacc = hk.RT(fp32, 64, 16, "col")

    def body(ve: hk.Tensor[bf16], vo: hk.Tensor[bf16], eye: hk.Tensor[bf16], vt: hk.Tensor[fp16]):
        gh = hk.arch.block_idx.y
        heads = ve.cols // HEAD_DIM
        b = gh // heads
        col0 = (gh % heads) * HEAD_DIM
        row = hk.arch.block_idx.x * (64 * warps) + hk.arch.warp_id() * 64
        e = eye[hk.elem_coord(0, 0, 0, 0)].load(te)
        for dd in range(HEAD_DIM // 16):
            acc = hk.zeros(tacc)
            for pr in range(2):
                for par, src in enumerate((ve, vo)):
                    x = src[hk.elem_coord(b, 0, (row + 32 * pr) // 2, col0 + 16 * dd)].load(ta)
                    sub = acc.subtile(16, 2 * pr + par)
                    sub += x @ e
            vt[hk.elem_coord(0, gh, 16 * dd, row)].store(hk.transpose(acc))

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (hk.cdiv(p.vt.cols, 64 * warps), p.vt.depth, 1))


# ---------------------------------------------------------------- attention


def attn_kernel(name: str = "attn_k3", *, kv_block: int = KV_BLOCK, warps: int = 8,
                qk_tiles: int = 2, pv_tiles: int = 2, setprio: bool = True,
                lazy: float = LAZY, max_vgprs: int = 256):
    """One workgroup a work unit: a 128-row Q block, or (the grid's tail) one KV part of one.
    q, o: [batch', 1, N, heads' * 128] bf16; k: [batch', 1, N_kv, heads' * 128] bf16;
    vt: [1, G, 128, N_pad] fp16 (vt_kernel); wp: [n_full, parts, nqb, n_split] int32;
    wu: the grid (its length is the unit count); op, ml: the split units' partial O and
    (max, sum)."""
    import hk  # noqa: PLC0415

    head_dim = HEAD_DIM
    q_block = 16
    q_tile = q_block * warps
    assert q_tile == Q_BLK
    d_frags = head_dim // 16
    kv_frags = kv_block // 16
    assert kv_block % (16 * qk_tiles) == 0 and d_frags % pv_tiles == 0 and kv_block % 32 == 0
    scale_l2e = head_dim ** -0.5 * math.log2(math.e)

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
        buf_k = g.stage_buffer(k_s, name="buf_k")
        buf_vt = g.stage_buffer(vt_s, name="buf_vt")
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
        heads = q.cols // head_dim
        bb = head // heads
        col0 = (head % heads) * head_dim
        n_q = q.rows
        n_kv = k.rows
        kv_total = hk.cdiv(n_kv, kv_block)
        per = hk.cdiv(kv_total, 1 + (parts - 1) * is_s)
        kb_lo = part * per
        kb_hi = hk.min(kb_lo + per, kv_total)

        wid = hk.arch.warp_id()
        # the last Q block backs up to end at n_q
        q_row = hk.min((blk % nqb) * q_tile, n_q - q_tile) + wid * q_block
        q_r = [q[hk.elem_coord(bb, 0, q_row, col0 + 16 * kk)].load(ty_q) for kk in range(d_frags)]

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        def load_kv(kb):
            kv0 = hk.min(kb, kb_hi - 1) * kv_block
            g.stage(buf_k, k[hk.elem_coord(bb, 0, kv0, col0)])
            g.stage(buf_vt, v[hk.elem_coord(0, head, 0, kv0)])

        def commit_kv():
            hk.arch.vm_wait(0)
            g.commit(k_s, buf_k)
            g.commit(vt_s, buf_vt)

        load_kv(kb_lo)

        for i_kb in range(kb_hi - kb_lo):
            kb = kb_lo + i_kb
            hk.arch.barrier()
            commit_kv()
            hk.arch.barrier(drain=True)
            load_kv(kb + 1)

            with hk.scope():
                s_w = hk.zeros(ty_sw)
                s_f = [s_w.subtile(16, n) for n in range(kv_frags)]
                for kk in range(d_frags):
                    for n0 in range(0, kv_frags, qk_tiles):
                        k_win = [gr_k.load_frag(n0 + i, kk, ty_k) for i in range(qk_tiles)]
                        hk.lds.wait_for(*k_win)
                        if setprio:
                            hk.arch.setprio(1)
                        for i in range(qk_tiles):
                            s_f[n0 + i] += k_win[i] @ q_r[kk].T
                        if setprio:
                            hk.arch.setprio(0)

                # the last block's rows past n_kv (a row of S^T is a kv index)
                valid = n_kv - kb * kv_block
                if valid < kv_block:
                    for n in range(kv_frags):
                        s_f[n].lower_fill_(valid - 16 * n, NEG_INF)

                # log2 units: p = exp2(S * c - m), c = sm_scale * log2e; m stale by up to `lazy`
                m_old = m_run.copy()
                m_f = s_f[0].max(axis=0).value()
                for n in range(1, kv_frags):
                    m_f.max_(s_f[n].max(axis=0))
                m_f *= scale_l2e
                if m_f.any_above(m_run, lazy):
                    m_run.max_(m_f)
                    alpha = (m_old - m_run).exp2()
                    l_run *= alpha
                    for c in o_c:
                        c *= alpha
                for n in range(kv_frags):
                    s_f[n] *= scale_l2e
                    s_f[n] -= m_run
                    s_f[n].exp2_()
                for n in range(kv_frags):
                    l_run += s_f[n].sum(axis=0)

                for j in range(kv_frags // 2):
                    p_pair = s_w.subtile(32, j).to(fp16)       # -> kda::pair_p
                    for par in range(2):
                        p_cvt = p_pair.subtile(16, par)
                        for g0 in range(0, d_frags, pv_tiles):
                            v_win = [gr_vt.load_frag(g0 + i, 2 * j + par, ty_vr)
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
                o[hk.elem_coord(bb, 0, q_row, col0 + 16 * i)].store(hk.transpose(c))

    body.__name__ = name
    return PairPKernel(body, n_pair_sites=kv_frags // 2, warps=warps, name=name,
                       grid=lambda p: (p.wu.cols, 1, 1), max_vgprs=max_vgprs, min_occupancy=1)


def combine_kernel(name: str = "attn_k3_combine", warps: int = 8):
    """The split units' partials -> o rows, one 128-row Q block a workgroup."""
    import hk  # noqa: PLC0415

    t = hk.RT(fp32, 16, HEAD_DIM)
    vc = hk.col_vec(t)

    def body(op: hk.Tensor[fp32], ml: hk.Tensor[fp32], wp: hk.Tensor[i32], wc: hk.Tensor[i32],
             o: hk.Tensor[bf16]):
        n_full = wp[hk.elem_coord(0, 0, 0, 0)].load_scalar(invariant=True)
        parts = wp[hk.elem_coord(0, 0, 0, 1)].load_scalar(invariant=True)
        nqb = wp[hk.elem_coord(0, 0, 0, 2)].load_scalar(invariant=True)
        tb = hk.arch.block_idx.x
        wid = hk.arch.warp_id()
        blk = n_full + tb
        head = blk // nqb
        heads = o.cols // HEAD_DIM
        ob = head // heads
        col0 = (head % heads) * HEAD_DIM
        row = hk.min((blk % nqb) * (16 * warps), o.rows - 16 * warps) + wid * 16
        base = tb * parts * (16 * warps) + wid * 16
        m = hk.neg_infty(vc)
        for p in range(parts):
            m.max_(ml[hk.elem_coord(0, 0, 0, base + p * (16 * warps))].load(vc))
        acc = hk.zeros(t)
        lsum = hk.zeros(vc)
        for p in range(parts):
            r = base + p * (16 * warps)
            w = ml[hk.elem_coord(0, 0, 0, r)].load(vc)
            w -= m
            w.exp2_()
            lp = ml[hk.elem_coord(0, 0, 1, r)].load(vc)
            lp *= w
            lsum += lp
            x = op[hk.elem_coord(0, 0, r, 0)].load(t)
            x *= w
            acc += x
        acc /= lsum
        o[hk.elem_coord(ob, 0, row, col0)] = acc

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, grid=lambda p: (p.wc.cols, 1, 1))


# ---------------------------------------------------------------- host side

_K = {}
_EYE = {}
_PLANS = {}
_CO = {}


def _kernels():
    if not _K:
        _K.update(vt=vt_kernel(), attn=attn_kernel(), combine=combine_kernel())
    return _K


def plan(nblk: int, device) -> tuple:
    """(n_full, parts, n_split): the Q blocks that do not fill the last wave split over KV into
    `parts` (2..4) units each, when that fills it better. A wave is 3 workgroups a WGP: the
    attention kernel's 208 VGPRs leave room for 6 of its waves a SIMD (torch's
    multi_processor_count is the WGP count on gfx1100)."""
    import os  # noqa: PLC0415

    import torch  # noqa: PLC0415

    wgps = torch.cuda.get_device_properties(device).multi_processor_count
    slots = int(os.environ.get("HK_K3_WG_PER_WGP", "3")) * wgps
    rem = nblk % slots
    parts = min(slots // rem, 4) if rem else 1
    if parts < 2:
        return nblk, 1, 0
    return nblk - rem, parts, rem


def _launch(kern, *tensors):
    import torch  # noqa: PLC0415

    from ..runtime import codeobj  # noqa: PLC0415

    kk = _CO.get(id(kern))
    if kk is None:
        kk = _CO[id(kern)] = codeobj.load(kern)
    shapes, ptrs, rows = codeobj.tensor_args(kk, tensors, kern.name)
    dev = tensors[0].device
    kk.launch(dev.index, torch.cuda.current_stream(dev).cuda_stream, shapes, ptrs, rows)


def _plan_tensors(nblk, nqb, dev):
    import torch  # noqa: PLC0415

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


def _vt_torch(v, n_pad: int):
    """vt_kernel's output by torch: V^T [1, B*H, 128, n_pad] fp16, zero-padded, every 32-column
    group its 16 even kv rows then its 16 odd."""
    import torch  # noqa: PLC0415

    b, h, n, d = v.shape
    vp = torch.zeros((b * h, n_pad, d), device=v.device, dtype=v.dtype)
    vp[:, :n].view(b, h, n, d).copy_(v)
    vt = vp.view(b * h, n_pad // 32, 16, 2, d).permute(0, 4, 1, 3, 2).to(
        torch.float16, memory_format=torch.contiguous_format)
    return vt.reshape(1, b * h, d, n_pad)


def is_bnhd(*ts) -> bool:
    """(B, H, N, D) views of contiguous (B, N, H, D) memory."""
    return all(t.dim() == 4 and not t.is_contiguous() and t.transpose(1, 2).is_contiguous()
               for t in ts)


def why(q, k, v, causal: bool = False, scale=None) -> str:
    """Why this call cannot run here, or '' if it can (shapes, dtypes and layout only)."""
    import torch  # noqa: PLC0415

    if causal:
        return "causal"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "bf16 only"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or k.shape != v.shape:
        return "(B, H, N, D) q, k, v with k and v alike"
    b, h, n, d = q.shape
    if d != HEAD_DIM or k.shape[0] != b or k.shape[1] != h or k.shape[3] != d:
        return f"head_dim {HEAD_DIM}, MHA (got q {tuple(q.shape)}, k {tuple(k.shape)})"
    if scale is not None and abs(scale - d ** -0.5) > 1e-12 * d ** -0.5:
        return "the default scale only"
    if n < Q_BLK or k.shape[2] < KV_BLOCK:
        return f"n_q >= {Q_BLK} and n_kv >= {KV_BLOCK}"
    contiguous = q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    if not contiguous and not is_bnhd(q, k, v):
        return "contiguous (B, H, N, D) or views of contiguous (B, N, H, D)"
    return ""


def attention(q, k, v):
    """softmax(q k^T / sqrt(128)) v, non-causal; see `why` for what fits. Returns the output in
    q's layout (a (B, H, N, D) view of (B, N, H, D) memory when q is one)."""
    import torch  # noqa: PLC0415

    ks = _kernels()
    b, h, n, d = q.shape
    n_kv = k.shape[2]
    dev = q.device
    bnhd = is_bnhd(q, k, v)
    if bnhd:
        bp, hp = b, h
        out = torch.empty((b, n, h, d), device=dev, dtype=torch.bfloat16)
        ret = out.transpose(1, 2)
        qf, kf, vf = (t.transpose(1, 2).reshape(b, 1, t.shape[2], h * d) for t in (q, k, v))
        of = out.view(b, 1, n, h * d)
    else:
        bp, hp = b * h, 1
        out = torch.empty_like(q)
        ret = out
        qf, kf, vf = (t.reshape(b * h, 1, t.shape[2], d) for t in (q, k, v))
        of = out.view(b * h, 1, n, d)
    g = bp * hp
    n_pad = -(-n_kv // 128) * 128
    rs, off = vf.stride(), vf.storage_offset()
    ve = vf.as_strided((bp, 1, (n_kv + 1) // 2, hp * d), (rs[0], rs[0], 2 * rs[2], 1), off)
    vo = vf.as_strided((bp, 1, n_kv // 2, hp * d), (rs[0], rs[0], 2 * rs[2], 1), off + rs[2])
    n_full = n_kv // 128 * 128        # vt_kernel's chunks: whole ones only, no read past n_kv
    if n_full and (n_kv % 2 == 0 or bp <= 8):
        vt = torch.empty((1, g, d, n_pad), device=dev, dtype=torch.float16)
        eye = _EYE.get(dev)
        if eye is None:
            eye = _EYE[dev] = torch.eye(16, device=dev, dtype=torch.bfloat16).view(1, 1, 16, 16)
        body = vt[..., :n_full]
        if n_kv % 2 == 0:
            # rows x row stride is then the views' true batch stride
            _launch(ks["vt"], ve, vo, eye, body)
        else:
            # it is not (hk derives the batch stride, (n + 1) rows' worth): a launch a batch
            for bi in range(bp):
                _launch(ks["vt"], ve[bi:bi + 1], vo[bi:bi + 1], eye, body[:, bi * hp:(bi + 1) * hp])
        if n_full < n_kv:
            vt[..., n_full:] = _vt_torch(v[:, :, n_full:], n_pad - n_full)
    else:
        vt = _vt_torch(v, n_pad)
    nqb = -(-n // Q_BLK)
    wp, wu, wc, n_split, parts = _plan_tensors(g * nqb, nqb, dev)
    su = max(n_split * parts, 1) * Q_BLK
    op = torch.empty((1, 1, su, d), device=dev, dtype=torch.float32)
    ml = torch.empty((1, 1, 2, su), device=dev, dtype=torch.float32)
    _launch(ks["attn"], qf, kf, vt, of, wp, wu, op, ml)
    if n_split:
        _launch(ks["combine"], op, ml, wp, wc, of)
    return ret
