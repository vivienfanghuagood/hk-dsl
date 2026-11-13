"""4-bit weights, bf16 activations: the decode GEMM of an AWQ or GPTQ model.

At decode M is the batch and the cost is reading W, so 4-bit codes read a
quarter of what bf16 weights do. This is `decode_gemm` with W dequantized
on its way into registers (`hk.load_w4`): one 16-row block of W per
workgroup, its K split across the warps, the partial sums meeting in LDS.

**The layout** is hk's own, written once when the checkpoint is loaded.
Codes are packed eight to an int32 in K order, and the words interleaved by
16-row block and 32-column chunk, so that one 16-byte load per lane reads 16
rows' codes as 256 contiguous bytes. Row-major, the same load touched 16
lines 2 KB apart, and the kernel streamed codes at 223-328 GB/s. Per group
of `group` columns it keeps a scale s and the zero point times the scale,
zs, so a code dequantizes as q * s - zs, one FMA:

    qw  int32 [N, K / 8], ordered [N/16][K/32][16][4]: nibble t of word w of
        row n is column 8w + t
    s   bf16  [N, K / group]
    zs  bf16  [N, K / group]

`repack_awq` and `repack_gptq` write it from either checkpoint format.

**Symmetric.** A checkpoint with one zero point for every group, GPTQ's
uint4b8 (8), need not keep zs: `repack_gptq(..., symmetric=True)` returns zs
None and s in 16-row blocks,

    s   [N/16, K/group * 16]: block b, group g, row r at [b][16 g + r]

so the 16 rows a k-step reads take 32 contiguous bytes. The kernels compute
(q - 8) * s (`hk.load_w4(..., zero=8, s_blocked=True)`), in packed fp16 for
fp16 activations: q - 8 exact, the product rounded once to nearest even.
12.5% fewer bytes at GPTQ's 32-column groups, and half the decode work.
"""

from typing import Any, Dict, Optional, Tuple

from .. import dsl as hk
from ..types import bf16, fp16, fp32, i32
from . import _arch

_DTYPES = {"bf16": bf16, "fp16": fp16}

#: AWQ packs eight output columns to an int32, column c in nibble AWQ_ORDER[c].
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_cols(packed, order=None):
    """int32 [R, C/8] -> uint8 [R, C], column 8j + c from nibble order[c] of
    word j (natural order when None)."""
    import torch  # noqa: PLC0415

    order = order or tuple(range(8))
    shifts = torch.tensor([4 * o for o in order], device=packed.device, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF
    return q.reshape(packed.shape[0], -1).to(torch.uint8)


def _pack_k(q):
    """Codes [N, K] -> int32 [N, K/8], nibble t of word w being column 8w + t."""
    import torch  # noqa: PLC0415

    n, k = q.shape
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64)
    words = (q.to(torch.int64).reshape(n, k // 8, 8) << shifts).sum(dim=-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
    return words.to(torch.int32).contiguous()


def _interleave(words):
    """Row-major words [N, K/8] -> [N/16][K/32][16][4], kept as [N, K/8]."""
    n, w = words.shape
    return words.reshape(n // 16, 16, w // 4, 4).permute(0, 2, 1, 3).reshape(n, w).contiguous()


def _deinterleave(qw):
    n, w = qw.shape
    return qw.reshape(n // 16, w // 4, 16, 4).permute(0, 2, 1, 3).reshape(n, w)


def _finish(q_nk, z_gn, s_gn, group: int, dtype):
    """Codes [N, K] and per-group zeros and scales [K/group, N] -> hk's layout,
    the scales in `dtype` (the activations' type; None: the checkpoint's
    scales' if 16-bit, else bf16)."""
    import torch  # noqa: PLC0415

    n, k = q_nk.shape
    if n % 16 or k % 32:
        raise ValueError(f"hk's 4-bit layout takes N a multiple of 16 and K of 32, "
                         f"got {n} x {k}")
    if dtype is None:
        dtype = s_gn.dtype if s_gn.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    s = s_gn.t().float()
    zs = z_gn.t().float() * s
    return _interleave(_pack_k(q_nk)), s.to(dtype).contiguous(), zs.to(dtype).contiguous()


def repack_awq(qweight, qzeros, scales, group: int, dtype=None):
    """An AutoAWQ GEMM checkpoint -- qweight int32 [K, N/8], qzeros int32
    [K/group, N/8], both packed along N in AWQ's order, scales [K/group, N] --
    as (qw, s, zs) in hk's layout."""
    q = _unpack_cols(qweight, AWQ_ORDER)            # [K, N]
    z = _unpack_cols(qzeros, AWQ_ORDER)             # [K/group, N]
    return _finish(q.t().contiguous(), z, scales, group, dtype)


#: The zero point of a symmetric 4-bit checkpoint (GPTQ's uint4b8).
SYMMETRIC_ZERO = 8


def repack_gptq(qweight, qzeros, scales, group: int, *, zero_offset: int = 1,
                dtype=None, symmetric: bool = False):
    """A GPTQ checkpoint without act-order -- qweight int32 [K/8, N] packed
    along K, qzeros int32 [K/group, N/8] packed along N, scales [K/group, N]
    -- as (qw, s, zs). `zero_offset` 1 is GPTQ v1's stored zero, one short.

    `symmetric`: (qw, s, None) for the kernels' symmetric form, which takes
    every zero point to be 8. Refused (ValueError) unless every stored zero
    point is: a checkpoint is checked, not assumed."""
    import torch  # noqa: PLC0415

    k8, n = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(k8 * 8, n)
    z = _unpack_cols(qzeros).to(torch.int32) + zero_offset
    if symmetric:
        if not bool((z == SYMMETRIC_ZERO).all()):
            raise ValueError(f"repack_gptq(symmetric=True): the checkpoint's zero points are not "
                             f"all {SYMMETRIC_ZERO} (with zero_offset {zero_offset})")
        qw, s, _ = _finish(q.t().contiguous().to(torch.uint8), z, scales, group, dtype)
        return qw, block_scales(s), None
    return _finish(q.t().contiguous().to(torch.uint8), z, scales, group, dtype)


def repack_quark(qweight, scales, group: int, *, qzeros=None, pack_reorder: bool = True,
                 dtype=None):
    """An AMD Quark int4 weight-only checkpoint -- `weight` int32 [K, N/8] packed along N,
    `weight_scale` [K/group, N] -- in hk's layout. `pack_reorder` is Quark's `pack_method:
    reorder` (AWQ's nibble order; else natural order).

    Symmetric (qzeros None): Quark stores signed codes, so the code is the nibble ^ 8 with every
    zero point 8; (qw, s, None) in the kernels' symmetric form, as repack_gptq(symmetric=True).
    Asymmetric: unsigned codes and `weight_zero_point` int32 [K/group, N/8] packed the same way;
    (qw, s, zs) as repack_awq."""
    import torch  # noqa: PLC0415

    order = AWQ_ORDER if pack_reorder else None
    q = _unpack_cols(qweight, order)                # [K, N]
    if qzeros is None:
        z = torch.full((q.shape[0] // group, q.shape[1]), SYMMETRIC_ZERO, device=q.device,
                       dtype=torch.int32)
        qw, s, _ = _finish((q ^ SYMMETRIC_ZERO).t().contiguous(), z, scales, group, dtype)
        return qw, block_scales(s), None
    return _finish(q.t().contiguous(), _unpack_cols(qzeros, order), scales, group, dtype)


def block_scales(s):
    """Scales [N, G] -> the symmetric form's 16-row blocks [N/16, G * 16]."""
    n, g = s.shape
    return s.reshape(n // 16, 16, g).permute(0, 2, 1).reshape(n // 16, g * 16).contiguous()


def unblock_scales(s):
    """`block_scales`' inverse: [N/16, G * 16] -> [N, G]."""
    b, g16 = s.shape
    return s.reshape(b, g16 // 16, 16).permute(0, 2, 1).reshape(b * 16, g16 // 16)


def dequant(qw, s, zs, group: int):
    """hk's layout back to a bf16 [N, K]: the reference, and the path for
    batches the decode kernel does not take. zs None: symmetric -- (q - 8) * s,
    s in 16-row blocks."""
    q = _unpack_cols(_deinterleave(qw)).float()     # [N, K]
    if zs is None:
        s = unblock_scales(s)
    sc = s.float().repeat_interleave(group, dim=1)
    z = sc * SYMMETRIC_ZERO if zs is None else zs.float().repeat_interleave(group, dim=1)
    return (q * sc - z).to(s.dtype)


def w4a16_kernel(name: str, *, tokens: int = 1, slices: int = 4, warps: int = 16,
                 group: int = 128, blocks: int = 1, dtype=bf16, symmetric: bool = False,
                 exllama: bool = False):
    """`tokens` 16-row token tiles (M <= 16 * tokens), `slices` 16-deep K
    slices per load, `warps` waves sharing each workgroup's `blocks` 16-row
    blocks of W, `group` columns per scale. K is a multiple of
    16 * slices * warps, and of group; group is a multiple of 16 * slices,
    so a load's codes share one scale.

    `blocks` > 1 reads each token tile once for several blocks of W: at four
    bits a 16 x 64 block of W is 512 bytes and the tokens' tile beside it
    2 KB, so the token reads, not the weights, can be what fills the L2.

    `symmetric`: the kernel takes (x, qw, s, c), every zero point 8, no zs, s
    in 16-row blocks (module docstring). `exllama` (symmetric, fp16): W is vLLM's
    RDNA skinny layout read in place (`hk.load_w4`'s `exllama`), s [N, K/group]."""
    m_rows = 16 * tokens
    kc = 16 * slices
    rows = 16 * blocks
    if group % kc:
        raise ValueError(f"group {group} must be a multiple of the {kc}-column load")
    ty_w = hk.RT(dtype, rows, kc, "row")
    ty_x = hk.RT(dtype, m_rows, kc, "row")
    ty_acc = hk.RT(fp32, rows, m_rows, "col")    # C^T: W rows x tokens
    ty_out = hk.RT(fp32, m_rows, rows, "row")
    st_part = hk.ST(fp32, m_rows, rows)
    in_flight = max(2, 8 // (tokens * blocks))

    @hk.func
    def gemm(x, qw, s, zs, c):
        n0 = hk.arch.block_idx.x * rows
        wid = hk.arch.warp_id()
        acc = hk.zeros(ty_acc)
        for t in hk.range(x.cols // (kc * warps)):
            k = (t * warps + wid) * kc
            w_t = (hk.load_w4(ty_w, qw, s, None, n0, k, group=group, zero=SYMMETRIC_ZERO,
                              s_blocked=not exllama, exllama=exllama)
                   if zs is None else hk.load_w4(ty_w, qw, s, zs, n0, k, group=group))
            x_t = x[hk.elem_coord(0, 0, 0, k)].load(ty_x, bounded=True)
            acc += w_t @ x_t.T
        out = hk.transpose(acc)
        if warps > 1:
            parts = hk.lds.alloc(st_part, count=warps, name="parts")
            parts[wid].store(out)
            hk.arch.barrier(drain=True)
        if wid == 0:
            if warps > 1:
                for i in range(1, warps):
                    if i > 1 and (i - 1) % in_flight == 0:
                        hk.arch.sched_barrier()
                    out += parts[i].load(ty_out)
            c[hk.elem_coord(0, 0, 0, n0)].store(out, bounded=True)

    if symmetric:
        def body(x: hk.Tensor[dtype], qw: hk.Tensor[i32], s: hk.Tensor[dtype],
                 c: hk.Tensor[dtype]):
            gemm(x, qw, s, None, c)
    else:
        def body(x: hk.Tensor[dtype], qw: hk.Tensor[i32], s: hk.Tensor[dtype],
                 zs: hk.Tensor[dtype], c: hk.Tensor[dtype]):
            gemm(x, qw, s, zs, c)

    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (p.qw.rows // rows, 1, 1))


def w4a16_large_kernel(name: str, *, group: int = 32, block_m: int = 128, block_n: int = 128,
                       warps: int = 8, warp_rows: int = 2, asym: bool = False):
    """The large-batch W4A16 GEMM on vLLM's RDNA skinny layout read in place (prefill): C (M, N)
    = x (M, K) @ W.T, fp16, W symmetric (zero point 8) as `exllama_matmul` reads it. A workgroup
    takes a block_m x block_n tile of C; every 64-deep K step its warps stage x's rows into LDS and
    decode W's block_n rows once into LDS (`hk.load_w4(exllama=True)`), then each warp multiplies
    its (block_m / warp_rows) x (block_n / warp_cols) share from both -- the decode is paid once
    per block_m tokens, not once per warp. K a multiple of 64; N of block_n.

    `asym`: W in hk's own layout with zero points instead (`repack_awq`: codes interleaved by
    16-row block, s and zs [N, K/group]), the kernel taking (x, qw, s, zs, c) -- AWQ as
    vllm_int4 holds it."""
    ks = 64
    warp_cols = warps // warp_rows
    rm, rn = block_m // warp_rows, block_n // warp_cols
    if block_m % (16 * warps) or block_n % (16 * warps) or rm % 16 or rn % 16:
        raise ValueError(f"w4a16_large_kernel: block {block_m}x{block_n} does not split over "
                         f"{warps} warps ({warp_rows} rows)")
    st_a = hk.ST(fp16, block_m, ks)
    st_b = hk.ST(fp16, block_n, ks)
    ty_ld_a = hk.RT(fp16, block_m // warps, ks, "row")
    ty_ld_b = hk.RT(fp16, block_n // warps, ks, "row")
    ty_a = hk.RT(fp16, rm, 16, "row")
    ty_b = hk.RT(fp16, rn, 16, "row")
    ty_c = hk.RT(fp32, rm, rn, "col")

    @hk.func
    def gemm(x, qw, s, zs, c):
        As = hk.lds.alloc(st_a, name="As")
        Bs = hk.lds.alloc(st_b, name="Bs")
        n0 = hk.arch.block_idx.x * block_n
        m0 = hk.arch.block_idx.y * block_m
        wid = hk.arch.warp_id()
        wr = wid // warp_cols
        wc = wid % warp_cols
        acc = hk.zeros(ty_c)
        for t in hk.range(x.cols // ks):
            k = t * ks
            a_t = x[hk.elem_coord(0, 0, m0 + wid * (block_m // warps), k)].load(ty_ld_a, bounded=True)
            w_t = (hk.load_w4(ty_ld_b, qw, s, zs, n0 + wid * (block_n // warps), k, group=group)
                   if zs is not None else
                   hk.load_w4(ty_ld_b, qw, s, None, n0 + wid * (block_n // warps), k, group=group,
                              zero=SYMMETRIC_ZERO, exllama=True))
            hk.arch.barrier()
            As.subtile(block_m // warps, ks, wid, 0).store(a_t)
            Bs.subtile(block_n // warps, ks, wid, 0).store(w_t)
            hk.arch.barrier()
            for sl in range(ks // 16):
                a_s = As.subtile(rm, 16, wr, sl).load(ty_a, wait=False)
                b_s = Bs.subtile(rn, 16, wc, sl).load(ty_b, wait=False)
                # the operands' LDS reads retired before the WMMA reads them: the compiler may
                # otherwise copy a ds_load destination ahead of the wait (gfx1100 caught it)
                hk.lds.wait_for(a_s, b_s)
                acc += a_s @ b_s.T
        c[hk.elem_coord(0, 0, m0 + wr * rm, n0 + wc * rn)].store(acc, bounded=True)

    if asym:
        def body(x: hk.Tensor[fp16], qw: hk.Tensor[i32], s: hk.Tensor[fp16],
                 zs: hk.Tensor[fp16], c: hk.Tensor[fp16]):
            gemm(x, qw, s, zs, c)
    else:
        def body(x: hk.Tensor[fp16], qw: hk.Tensor[i32], s: hk.Tensor[fp16], c: hk.Tensor[fp16]):
            gemm(x, qw, s, None, c)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (p.qw.rows // block_n, hk.cdiv(p.x.rows, block_m), 1))


def w4a16_gemv_kernel(name: str, *, tokens: int = 1, warps: int = 16, group: int = 128,
                      symmetric: bool = False, exllama: bool = False):
    """The small-batch form: up to `tokens` (1..8) rows of fp16 x against one 16-row block of W a
    workgroup, each warp a share of K through `hk.gemv_w4` -- no WMMA, so each lane decodes only
    its own row -- and the warps' partials meeting in LDS as `w4a16_kernel`'s do. K is a multiple
    of 64 * warps; group of 32.

    `exllama`: symmetric, W in vLLM's RDNA skinny layout read in place (`hk.gemv_w4`'s
    `exllama`): qw int32 [N, K/8] row-major in ExLlama's nibble order, s [N, K/group]."""
    ty_out = hk.RT(fp32, 16, 16, "row")
    st_part = hk.ST(fp32, 16, 16)

    @hk.func
    def gemv(x, qw, s, zs, c):
        n0 = hk.arch.block_idx.x * 16
        wid = hk.arch.warp_id()
        count = x.cols // (64 * warps)
        acc = (hk.gemv_w4(x, qw, s, None, n0, wid * 64, 64 * warps, count, group=group,
                          tokens=tokens, zero=SYMMETRIC_ZERO, s_blocked=not exllama,
                          exllama=exllama)
               if zs is None else
               hk.gemv_w4(x, qw, s, zs, n0, wid * 64, 64 * warps, count, group=group, tokens=tokens))
        out = hk.transpose(acc)
        if warps > 1:
            parts = hk.lds.alloc(st_part, count=warps, name="parts")
            parts[wid].store(out)
            hk.arch.barrier(drain=True)
        if wid == 0:
            if warps > 1:
                for i in range(1, warps):
                    if i > 1 and (i - 1) % 8 == 0:
                        hk.arch.sched_barrier()
                    out += parts[i].load(ty_out)
            c[hk.elem_coord(0, 0, 0, n0)].store(out, bounded=True)

    if exllama and not symmetric:
        raise ValueError("w4a16_gemv_kernel(exllama=True) is the symmetric form")
    if symmetric:
        def body(x: hk.Tensor[fp16], qw: hk.Tensor[i32], s: hk.Tensor[fp16], c: hk.Tensor[fp16]):
            gemv(x, qw, s, None, c)
    else:
        def body(x: hk.Tensor[fp16], qw: hk.Tensor[i32], s: hk.Tensor[fp16],
                 zs: hk.Tensor[fp16], c: hk.Tensor[fp16]):
            gemv(x, qw, s, zs, c)

    return hk.kernel(body, warps=warps, name=name, grid=lambda p: (p.qw.rows // 16, 1, 1))


#: The most tokens the small-batch form (`w4a16_gemv_kernel`) takes, and its warps.
GEMV_MAX_M = 4
GEMV_WARPS = 16
#: Where it is measured faster than the WMMA form and the engine's kernel.
GEMV_ARCHES = ("gfx1100",)


def gemv_takes(m: int, n: int, k: int, group: int, dtype: str) -> bool:
    """Whether the small-batch form runs an (M, N, K) GEMM: fp16, M <= GEMV_MAX_M, N a multiple
    of 16, K of 64 * GEMV_WARPS, group a multiple of 32, on an arch it is measured on."""
    from ..target import current_arch  # noqa: PLC0415

    return (dtype == "fp16" and 1 <= m <= GEMV_MAX_M and n % 16 == 0 and group % 32 == 0
            and k % group == 0 and k % (64 * GEMV_WARPS) == 0 and current_arch() in GEMV_ARCHES)


def gemv_kernel_for(tokens: int, group: int, symmetric: bool, warps: int = GEMV_WARPS):
    key = ("gemv", tokens, group, symmetric, warps)
    if key not in _KERNELS:
        _KERNELS[key] = w4a16_gemv_kernel(
            f"w4a16_gemv_t{tokens}_w{warps}_g{group}" + ("_sym" if symmetric else ""),
            tokens=tokens, warps=warps, group=group, symmetric=symmetric)
    return _KERNELS[key]


#: The most tokens one launch of the in-place small-batch form takes (`exllama_matmul`'s gemv path,
#: in chunks of this many, each reading W again), and the most tokens it runs that way: above,
#: the in-place WMMA form (`w4a16_kernel(exllama=True)`), one launch up to 64 tokens.
EXL_CHUNK = 8
EXL_GEMV_MAX = 4


#: ExLlama's nibble order in a skinny word: nibble i holds column 8w + EXL_ORDER[i] of the row.
EXL_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def gptq_to_skinny(qweight, chunk: int = 2048):
    """A GPTQ checkpoint's codes (int32 [K/8, N], packed along K, nibble i = row 8w + i) in vLLM's RDNA
    skinny layout (int32 [N, K/8], nibble i of word w = column 8w + EXL_ORDER[i] -- what vLLM's
    `pack_skinny_int4` gives and `exllama_matmul` / `exllama_large_matmul` read in place). N is done
    `chunk` columns at a time, so the transient is a few hundred MB at most, not 8x the codes."""
    import torch  # noqa: PLC0415

    k8, n = qweight.shape
    out = torch.empty((n, k8), device=qweight.device, dtype=torch.int32)
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    for n0 in range(0, n, chunk):
        q = qweight[:, n0:n0 + chunk]
        codes = ((q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(k8 * 8, -1)   # [K, c]
        g = codes.t().reshape(-1, k8, 8).to(torch.int64)                               # [c, K/8, 8]
        words = sum(g[:, :, col] << (4 * i) for i, col in enumerate(EXL_ORDER))
        out[n0:n0 + chunk] = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return out


def exllama_takes(m: int, n: int, k: int, group: int) -> bool:
    """Whether `exllama_matmul` runs an (M, N, K) GEMM: fp16, N a multiple of 16, K of
    64 * GEMV_WARPS, group a multiple of 32 (the small-batch form's needs; the WMMA form's are
    checked per call)."""
    return m >= 1 and n % 16 == 0 and k % (64 * GEMV_WARPS) == 0 and group % 32 == 0 and k % group == 0


def exllama_matmul(x, w_q, s, group: int, out=None):
    """x @ W.T, fp16, for a symmetric 4-bit W (zero point 8) in vLLM's RDNA skinny layout, read in
    place: w_q int32 [N, K/8] (its int8 [N, K/2] viewed as int32) in ExLlama's nibble order, s
    [N, K/group]. Up to EXL_GEMV_MAX tokens the small-batch form; above, the WMMA form at the
    decode schedules (one launch, to 64 tokens) where `takes` holds, else the small-batch form
    `EXL_CHUNK` tokens a launch. `exllama_takes` must hold."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    n = w_q.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    xc = x if x.is_contiguous() else x.contiguous()
    if EXL_GEMV_MAX < m and takes(m, n, k, group):
        sch = schedule_for(m, k, group)
        key = ("exl-wmma", tokens_for(m), sch["slices"], sch["warps"], group, sch["blocks"])
        if key not in _OPS:
            from ..runtime.torch_ext import launcher  # noqa: PLC0415

            _OPS[key] = launcher(kernel_for(tokens_for(m), sch["slices"], sch["warps"], group,
                                            sch["blocks"], "fp16", True, True))
        _OPS[key](xc, w_q, s, out)
        return out
    for r0 in range(0, m, EXL_CHUNK):
        t = min(EXL_CHUNK, m - r0)
        key = ("exl", t, group)
        if key not in _OPS:
            from ..runtime.torch_ext import launcher  # noqa: PLC0415

            _OPS[key] = launcher(w4a16_gemv_kernel(
                f"w4a16_gemv_t{t}_w{GEMV_WARPS}_g{group}_exl", tokens=t, warps=GEMV_WARPS,
                group=group, symmetric=True, exllama=True))
        _OPS[key](xc[r0:r0 + t], w_q, s, out[r0:r0 + t])
    return out


#: The large-batch form's tile (`w4a16_large_kernel`): measured best of 128x128 with 8 warps in
#: 2 or 4 rows and 128x64 with 4 on Qwen3.8-27B-GPTQ's shapes on gfx1201 (65-4096 tokens).
LARGE_TILE = dict(block_m=128, block_n=128, warps=8, warp_rows=4)


def exllama_large_takes(n: int, k: int, group: int) -> bool:
    """Whether `exllama_large_matmul` takes an [N, K] W: N a multiple of the tile's block_n, K of
    64, group a multiple of 16 dividing K."""
    return n % LARGE_TILE["block_n"] == 0 and k % 64 == 0 and group % 16 == 0 and k % group == 0


def exllama_large_matmul(x, w_q, s, group: int, out=None):
    """x @ W.T, fp16, any number of tokens, W in vLLM's RDNA skinny layout read in place (as
    `exllama_matmul`): the large-batch form, W decoded once per 128 tokens into LDS.
    `exllama_large_takes` must hold."""
    import torch  # noqa: PLC0415

    m = x.shape[0]
    n = w_q.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    key = ("exl-large", group)
    if key not in _OPS:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        _OPS[key] = launcher(w4a16_large_kernel(f"w4a16_large_g{group}_exl", group=group,
                                                **LARGE_TILE))
    _OPS[key](x if x.is_contiguous() else x.contiguous(), w_q, s, out)
    return out


def large_matmul(x, qw, s, zs, group: int, out=None):
    """x @ W.T, fp16, any number of tokens, W in hk's layout with zero points (`repack_awq`): the
    large-batch form's `asym` variant. `exllama_large_takes` must hold for W's [N, K]."""
    import torch  # noqa: PLC0415

    m = x.shape[0]
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    key = ("large-asym", group)
    if key not in _OPS:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        _OPS[key] = launcher(w4a16_large_kernel(f"w4a16_large_g{group}_asym", group=group,
                                                asym=True, **LARGE_TILE))
    _OPS[key](x if x.is_contiguous() else x.contiguous(), qw, s, zs, out)
    return out


#: Per token-tile count: decode_gemm's measured schedules (chapter 9), with
#: the load kept at a whole group's divisor.
SCHEDULES: Dict[int, Dict[str, int]] = {
    1: dict(slices=4, warps=16, blocks=1),
    2: dict(slices=2, warps=16, blocks=1),
    4: dict(slices=2, warps=8, blocks=1),
}
MAX_TOKENS = max(SCHEDULES)

#: One token tile at K up to this: two 16-row blocks of W a workgroup on 8
#: warps, 5-22% faster than one on 16 at Qwen3-8B's 4096-deep layers
#: (tools/hk-bench/w4a16_ab.py) and 6% slower at its 12288-deep one.
SHALLOW_K = 4096


def schedule_for(m: int, k: int, group: int = 128) -> Optional[Dict[str, int]]:
    """The schedule for M rows at depth K. A load's 16 x slices columns share
    one scale, so a group narrower than the load (GPTQ's 32) narrows it."""
    t = tokens_for(m)
    if t is None:
        return None
    s = dict(slices=4, warps=8, blocks=2) if t == 1 and k <= SHALLOW_K else dict(SCHEDULES[t])
    if 16 * s["slices"] > group and group % 16 == 0:
        s["slices"] = group // 16
    return s

_KERNELS: Dict[Tuple[Any, ...], Any] = {}
_OPS: Dict[Tuple[Any, ...], Any] = {}


def tokens_for(m: int) -> Optional[int]:
    t = -(-m // 16)
    return next((k for k in sorted(SCHEDULES) if k >= t), None)


def kernel_for(tokens: int, slices: int, warps: int, group: int, blocks: int = 1,
               dtype: str = "bf16", symmetric: bool = False, exllama: bool = False):
    key = (tokens, slices, warps, group, blocks, dtype, symmetric, exllama)
    if key not in _KERNELS:
        _KERNELS[key] = w4a16_kernel(
            f"w4a16_t{tokens}_s{slices}_w{warps}_g{group}_b{blocks}_{dtype}"
            + ("_sym" if symmetric else "") + ("_exl" if exllama else ""),
            tokens=tokens, slices=slices, warps=warps, group=group, blocks=blocks,
            dtype=_DTYPES[dtype], symmetric=symmetric, exllama=exllama)
    return _KERNELS[key]


def _op_for(*key):
    """The kernel's launch: its torch op outside torch.compile's tracing
    called without the dispatcher (`torch_ext.launcher`)."""
    if key not in _OPS:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        _OPS[key] = launcher(kernel_for(*key))
    return _OPS[key]


def takes(m: int, n: int, k: int, group: int) -> bool:
    """Whether the decode kernel runs an (M, N, K) GEMM at this group size."""
    s = schedule_for(m, k, group)
    if s is None or k % group:
        return False
    kc = 16 * s["slices"]
    return (n % (16 * s["blocks"]) == 0 and group % kc == 0
            and k % (kc * s["warps"]) == 0)


#: (M, N, K, group, dtype) -> the launch that takes the call, or None for the
#: dequantize path: `takes`, `schedule_for` and the kernel's name, decided once.
_PLANS: Dict[Tuple[Any, ...], Any] = {}
#: Shapes memoized; past it a call decides afresh.
_PLAN_CAP = 4096


def plan(m: int, n: int, k: int, group: int, dtype, symmetric: bool = False):
    """The decode kernel's launch for this call -- (x, qw, s, zs, out), or (x, qw, s, out)
    `symmetric` -- or None where `matmul` takes the dequantize path: memoized `takes` and
    schedule."""
    key = (m, n, k, group, dtype, symmetric)
    try:
        return _PLANS[key]
    except KeyError:
        pass
    import torch  # noqa: PLC0415

    names = {torch.bfloat16: "bf16", torch.float16: "fp16"}
    run = None
    if dtype in names and gemv_takes(m, n, k, group, names[dtype]):
        if ("gemv", m, group, symmetric) not in _OPS:
            from ..runtime.torch_ext import launcher  # noqa: PLC0415

            _OPS[("gemv", m, group, symmetric)] = launcher(gemv_kernel_for(m, group, symmetric))
        run = _OPS[("gemv", m, group, symmetric)]
    elif dtype in names and takes(m, n, k, group):
        sch = schedule_for(m, k, group)
        run = _op_for(tokens_for(m), sch["slices"], sch["warps"], group, sch["blocks"],
                      names[dtype], symmetric)
    if len(_PLANS) < _PLAN_CAP:
        _PLANS[key] = run
    return run


def matmul(x, qw, s, zs, group: int, out=None):
    """x @ W.T for W in hk's 4-bit layout: the decode kernel up to 64 rows of
    x, dequantize-then-F.linear above (or for a shape it does not take).
    zs None: the symmetric form (every zero point 8; `repack_gptq(symmetric=True)`)."""
    import torch  # noqa: PLC0415

    _arch.require("hk.ops.w4a16.matmul")
    from ..runtime import hostpath  # noqa: PLC0415

    if not hostpath.ENABLED:
        return _matmul_legacy(x, qw, s, zs, group, out)
    m, k = x.shape
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=s.dtype)
    run = plan(m, n, k, group, x.dtype, zs is None) if x.dtype == s.dtype else None
    if run is not None:
        xc = x if x.is_contiguous() else x.contiguous()
        if zs is None:
            run(xc, qw, s, out)
        else:
            run(xc, qw, s, zs, out)
        return out
    out.copy_(torch.nn.functional.linear(x.to(s.dtype), dequant(qw, s, zs, group)))
    return out


def _op_for_legacy(*key):
    if key not in _OPS_LEGACY:
        from ..runtime.torch_ext import torch_op  # noqa: PLC0415

        _OPS_LEGACY[key] = torch_op(kernel_for(*key))
    return _OPS_LEGACY[key]


_OPS_LEGACY: Dict[Tuple[Any, ...], Any] = {}


def _matmul_legacy(x, qw, s, zs, group: int, out=None):
    """`matmul` before the host fast paths, line for line (HK_HOST_FASTPATH=0)."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=s.dtype)
    names = {torch.bfloat16: "bf16", torch.float16: "fp16"}
    if x.dtype == s.dtype and x.dtype in names and takes(m, n, k, group):
        t = tokens_for(m)
        sch = schedule_for(m, k, group)
        op = _op_for_legacy(t, sch["slices"], sch["warps"], group, sch["blocks"], names[x.dtype],
                            zs is None)
        xc = x if x.is_contiguous() else x.contiguous()
        if zs is None:
            op(xc, qw, s, out)
        else:
            op(xc, qw, s, zs, out)
        return out
    out.copy_(torch.nn.functional.linear(x.to(s.dtype), dequant(qw, s, zs, group)))
    return out


__all__ = ["AWQ_ORDER", "EXL_CHUNK", "EXL_GEMV_MAX", "GEMV_ARCHES", "LARGE_TILE", "GEMV_MAX_M", "GEMV_WARPS", "MAX_TOKENS", "SCHEDULES",
           "SHALLOW_K", "SYMMETRIC_ZERO", "block_scales", "dequant", "exllama_large_matmul", "exllama_large_takes", "exllama_matmul",
           "exllama_takes", "gemv_kernel_for", "gemv_takes", "large_matmul",
           "kernel_for", "unblock_scales", "w4a16_gemv_kernel", "w4a16_large_kernel",
           "matmul", "plan", "schedule_for",
           "repack_awq", "repack_gptq", "repack_quark", "takes", "tokens_for", "w4a16_kernel"]
