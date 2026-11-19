"""fp8 casts for training on gfx1201: bf16 -> fp8 rowwise and columnwise (transposed), with amax.

Delayed scaling, Transformer Engine's semantics: out = fp8(x * scale), saturating; amax = max |x|
folded into the caller's slot with a max (TE's recipe zeroes it after each update; micro-batches
in between all fold into it), never overwritten; scale_inv = 1 / scale on request.

* `cast_transpose(x, ...)`: the plain cast, TE's `quantize` with rowwise and columnwise usage.
* `swiglu_bwd_cast_transpose(g, y, ...)`: Megatron-LM's `swiglu_back(g, y)` and that cast in one
  pass -- the e5m2 gradient of an MLP's fc1 output, which its dgrad and wgrad GEMMs read. It
  replaces Megatron's fused elementwise kernel and TE's quantize, and never writes the bf16
  gradient.

The rowwise copy has a padded row stride (`pad`, default 64 bytes) and the columnwise one can
(`pad_col`): on gfx1201, rows a power of two of bytes apart share a memory channel
(hk.ops.gemm.FP8_POW2's note), which costs the GEMM that reads them and this kernel's own stores.
hk.ops.gemm.scaled_mm takes the padded operands as they are (dgrad A @ W^T, and wgrad with
`accumulate=True` into an fp32 main_grad).

R9700, swiglu_back + cast (Megatron: its compiled swiglu_back, then TE's quantize), F = 8192, us:

    M        Megatron   this, columnwise dense   this, columnwise padded (64 B)
    2048       577            495                       489
    4096      1490           1368                       973

Workgroups walk the columns fastest so the reads stay whole row segments; the columnwise stores
are what is left (rowwise only: 273 / 509 us). Drafted 2026-11-19 (tools/rdna4-probes/fp8_cast.py).
"""

from .. import dsl as hk
from ..dsl.collective import cross_warp, fold_rows
from ..types import bf16, fp32, fp8e4m3, fp8e5m2

ROWS = 16      # a register tile's rows
COLS = 64      # a workgroup's columns
WARPS = 4      # a workgroup: WARPS * 2 * ROWS = 128 rows


def cast_transpose_kernel(name: str, *, dtype=fp8e4m3, warps: int = WARPS, cols: int = COLS):
    """One workgroup: 128 rows x `cols` columns of x. Each warp casts two 16-row tiles and stores
    them rowwise; their transposes go to an fp32 LDS tile [cols, 128], from which each warp stores
    16 of the columnwise rows, 128 bytes each."""
    T = hk.RT(fp32, ROWS, cols)
    rows_wg = warps * 2 * ROWS
    st = hk.ST(fp32, cols, rows_wg)
    TR = hk.RT(fp32, ROWS, rows_wg)

    def body(x: hk.Tensor[bf16], qr: hk.Tensor[dtype], qc: hk.Tensor[dtype],
             sc: hk.Tensor[fp32], part: hk.Tensor[fp32]):
        Ts = hk.lds.alloc(st, name="Ts")
        w = hk.arch.warp_id()
        rb = hk.arch.block_idx.y * rows_wg
        c0 = hk.arch.block_idx.x * cols
        s = sc.at[0, 0].load(hk.row_vec(T))
        a = hk.zeros(hk.col_vec(T))     # |x| >= 0, so 0 is the identity of the max
        for i in range(2):
            r0 = rb + (2 * w + i) * ROWS
            t = x.at[r0, c0].load(T)
            a.max_(t.abs().max(axis=1))
            t *= s
            qr.at[r0, c0].store(t)
            Ts.subtile(cols, ROWS, 0, 2 * w + i).store(hk.transpose(t))
        hk.arch.barrier(drain=True)
        for j in range(cols // ROWS // warps):
            row = j * warps + w
            u = Ts.subtile(ROWS, rows_wg, row, 0).load(TR)
            qc.at[c0 + row * ROWS, rb].store(u)
        a = cross_warp(a, "max", warps)
        a = fold_rows(a, "max", warps)
        part.at[0, hk.arch.block_idx.y * (x.cols // cols) + hk.arch.block_idx.x].store_scalar(a)

    body.__annotations__.update(qr=hk.Tensor[dtype], qc=hk.Tensor[dtype])
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (p.x.cols // cols, p.x.rows // rows_wg, 1))


def swiglu_bwd_cast_transpose_kernel(name: str, *, dtype=fp8e5m2, warps: int = WARPS,
                                      cols: int = 32, bf16_round: bool = True,
                                      rowwise: bool = True, colwise: bool = True,
                                      col_lds: bool = True):
    """Megatron's swiglu_back fused with the cast: g (M, F) and y = [y1 | y2] (M, 2F), bf16 ->
    dy = [g s (1 + y1 (1 - s)) y2 | g y1 s], s = sigmoid(y1), computed in fp32, rounded to bf16
    (`bf16_round`: the value Megatron's kernel would have written), then cast as
    cast_transpose_kernel does: rowwise (M, 2F) and columnwise (2F, M), amax partials.
    A workgroup: 128 rows x `cols` columns of each half; each half's transpose goes to its own
    fp32 LDS tile [cols, 128], and the warps store the 2 * cols columnwise rows, 128 bytes each.
    `rowwise` / `colwise` False drop that store, `col_lds` False stores the transposes straight
    from registers (ablations; tools/rdna4-probes/fp8_cast.py)."""
    T = hk.RT(fp32, ROWS, cols)
    rows_wg = warps * 2 * ROWS
    st = hk.ST(fp32, cols, rows_wg)
    TR = hk.RT(fp32, ROWS, rows_wg)
    per = cols // ROWS
    blocks = 2 * per                   # columnwise 16-row blocks, both halves
    if blocks % warps:
        raise ValueError(f"2 * cols / 16 = {blocks} blocks must split over {warps} warps")

    def body(g: hk.Tensor[bf16], y: hk.Tensor[bf16], qr: hk.Tensor[dtype], qc: hk.Tensor[dtype],
             sc: hk.Tensor[fp32], part: hk.Tensor[fp32]):
        Ts = [hk.lds.alloc(st, name=f"Ts{h}") for h in range(2)]
        w = hk.arch.warp_id()
        f = g.cols
        # Workgroups walk the columns fastest, so the ones in flight together read whole row
        # segments of g and y. Measured: M-fastest, diagonal and grouped orders all read 1.5-4x
        # slower (y's rows are 2F * 2 bytes apart, a power of two), more than they saved on the
        # columnwise stores.
        nx = f // cols
        lin = hk.arch.block_idx.x
        bx, by = lin % nx, lin // nx
        rb = by * rows_wg
        c0 = bx * cols
        s = sc.at[0, 0].load(hk.row_vec(T))
        a = hk.zeros(hk.col_vec(T))
        for i in range(2):
            r0 = rb + (2 * w + i) * ROWS
            gt = g.at[r0, c0].load(T)
            y1 = y.at[r0, c0].load(T)
            y2 = y.at[r0, f + c0].load(T)
            sg = hk.full(T, 1.0) / ((y1 * -1.0).exp() + 1.0)
            for h in range(2):
                if h == 0:
                    d = gt * sg * ((hk.full(T, 1.0) - sg) * y1 + 1.0) * y2
                else:
                    d = gt * y1 * sg
                if bf16_round:
                    d = d.to(bf16).to(fp32)
                a.max_(d.abs().max(axis=1))
                d *= s
                if rowwise:
                    qr.at[r0, h * f + c0].store(d)
                if colwise and col_lds:
                    Ts[h].subtile(cols, ROWS, 0, 2 * w + i).store(hk.transpose(d))
                elif colwise:
                    qc.at[h * f + c0, r0].store(hk.transpose(d))
        if colwise and col_lds:
            hk.arch.barrier(drain=True)
            # warp w stores columnwise blocks w, w + warps, ...; block b is half b // per, 16-row
            # block b % per of it (a warp-uniform branch)
            for j in range(blocks // warps):
                for h in range(2):
                    for r in range(per):
                        b = h * per + r
                        with hk.if_((j * warps + w) == b):
                            u = Ts[h].subtile(ROWS, rows_wg, r, 0).load(TR)
                            qc.at[h * f + c0 + r * ROWS, rb].store(u)
        a = cross_warp(a, "max", warps)
        a = fold_rows(a, "max", warps)
        part.at[0, hk.arch.block_idx.x].store_scalar(a)

    body.__annotations__.update(qr=hk.Tensor[dtype], qc=hk.Tensor[dtype])
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: ((p.g.cols // cols) * (p.g.rows // rows_wg), 1, 1))


_K: dict = {}


def _outs(out, shapes, tdt, op: str):
    """The caller's (rowwise, columnwise) buffers, fp8 or uint8 views (TE's Float8Tensor data), as fp8."""
    import torch  # noqa: PLC0415

    res = []
    for t, shape in zip(out, shapes):
        if t.dtype == torch.uint8:
            t = t.view(tdt)
        if t.dtype != tdt or tuple(t.shape) != shape or t.stride(1) != 1:
            raise ValueError(f"{op}: out must be {tdt} (or uint8) {shape} with unit column stride, "
                             f"got {tuple(t.shape)}/{t.dtype}")
        res.append(t)
    return res


def _scale_inv(scale, scale_inv):
    import torch  # noqa: PLC0415

    scale_inv.copy_(torch.reciprocal(scale).reshape(scale_inv.shape))


def _check(x, rows: int, cols: int, op: str):
    import torch  # noqa: PLC0415

    from ..target import current_arch, get_target  # noqa: PLC0415

    t = get_target(current_arch())
    if "fp8e4m3" not in t.mma or t.wave != 32:
        raise ValueError(f"{op}: gfx1201 (an fp8 WMMA target), not {current_arch()}")
    if x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError(f"{op}: a bf16 (M, N) tensor contiguous along N, got "
                         f"{tuple(x.shape)}/{x.dtype}")
    if x.shape[0] % rows or x.shape[1] % cols:
        raise ValueError(f"{op}: M={x.shape[0]} must be a multiple of {rows} and "
                         f"N={x.shape[1]} of {cols}")


def cast_transpose(x, scale, amax, *, dtype="e4m3", pad: int = 64, scale_inv=None, out=None):
    """(rowwise (M, N) with row stride N + pad, columnwise (N, M)) of bf16 x (M, N), fp8 of x * scale;
    amax <- max(amax, max |x|) in place; scale_inv <- 1 / scale if given. M a multiple of 128,
    N of 64; `dtype` "e4m3" or "e5m2"."""
    import torch  # noqa: PLC0415

    _check(x, 128, COLS, "cast_transpose")
    m, n = x.shape
    tdt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[dtype]
    hdt = {"e4m3": fp8e4m3, "e5m2": fp8e5m2}[dtype]
    k = _K.get(dtype)
    if k is None:
        k = _K[dtype] = cast_transpose_kernel(f"fp8_cast_transpose_{dtype}", dtype=hdt)
    if out is not None:
        rw, cw = _outs(out, ((m, n), (n, m)), tdt, "cast_transpose")
    else:
        rw = torch.empty(m, n + pad, device=x.device, dtype=tdt)[:, :n]
        cw = torch.empty(n, m, device=x.device, dtype=tdt)
    s64 = scale.reshape(1, 1).expand(1, COLS).contiguous()
    part = torch.empty(1, (m // (WARPS * 2 * ROWS)) * (n // COLS), device=x.device, dtype=torch.float32)
    k(x, rw, cw, s64, part)
    torch.maximum(amax, part.amax(), out=amax)
    if scale_inv is not None:
        _scale_inv(scale, scale_inv)
    return rw, cw


_KS: dict = {}


def swiglu_bwd_cast_transpose(g, y, scale, amax, *, dtype="e5m2", pad: int = 64, pad_col: int = 0,
                              scale_inv=None, out=None, **knobs):
    """Megatron's swiglu_back(g, y) cast to fp8 in the same pass: g (M, F) and y (M, 2F) bf16 ->
    (rowwise (M, 2F), row stride 2F + pad; columnwise (2F, M), row stride M + pad_col); amax <-
    max(amax, max |dy|); scale_inv <- 1 / scale if given. dy is computed in fp32 and rounded to
    bf16 before the cast, as Megatron writes it. `out=(rowwise, columnwise)`: the caller's buffers
    (fp8, or uint8 views of them; any row strides) instead of new ones. M a multiple of 128, F of 32."""
    import torch  # noqa: PLC0415

    _check(g, 128, knobs.get("cols", 32), "swiglu_bwd_cast_transpose")
    if y.shape != (g.shape[0], 2 * g.shape[1]) or y.dtype != g.dtype or y.stride(1) != 1:
        raise ValueError(f"swiglu_bwd_cast_transpose: y must be bf16 (M, 2F) = "
                         f"({g.shape[0]}, {2 * g.shape[1]}), got {tuple(y.shape)}/{y.dtype}")
    m, f = g.shape
    tdt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[dtype]
    hdt = {"e4m3": fp8e4m3, "e5m2": fp8e5m2}[dtype]
    key = (dtype, tuple(sorted(knobs.items())))
    k = _KS.get(key)
    if k is None:
        sfx = "".join(f"_{a}{int(b) if isinstance(b, bool) else b}" for a, b in key[1])
        k = _KS[key] = swiglu_bwd_cast_transpose_kernel(f"fp8_swiglu_bwd_ct_{dtype}{sfx}",
                                                         dtype=hdt, **knobs)
    if out is not None:
        rw, cw = _outs(out, ((m, 2 * f), (2 * f, m)), tdt, "swiglu_bwd_cast_transpose")
    else:
        rw = torch.empty(m, 2 * f + pad, device=g.device, dtype=tdt)[:, :2 * f]
        cw = torch.empty(2 * f, m + pad_col, device=g.device, dtype=tdt)[:, :m]
    s64 = scale.reshape(1, 1).expand(1, COLS).contiguous()
    part = torch.empty(1, (m // (knobs.get("warps", WARPS) * 2 * ROWS)) * (f // knobs.get("cols", 32)),
                       device=g.device,
                       dtype=torch.float32)
    k(g, y, rw, cw, s64, part)
    torch.maximum(amax, part.amax(), out=amax)
    if scale_inv is not None:
        _scale_inv(scale, scale_inv)
    return rw, cw


__all__ = ["cast_transpose", "swiglu_bwd_cast_transpose", "cast_transpose_kernel",
           "swiglu_bwd_cast_transpose_kernel"]
