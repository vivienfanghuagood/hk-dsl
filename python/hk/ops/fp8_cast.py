"""fp8 cast with transpose and amax for training on gfx1201 (draft).

Delayed scaling, Transformer Engine's semantics: out = fp8(x * scale) rowwise (M, N) -- optionally
with a padded row stride, which the dgrad GEMM wants (hk.ops.gemm.FP8_POW2's note) -- and
columnwise (N, M); amax = max |x| folded into the caller's slot with a max, never overwritten.
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


_K: dict = {}


def cast_transpose(x, scale, amax, *, dtype="e4m3", pad: int = 64, scale_inv=None):
    """(rowwise (M, N) with row stride N + pad, columnwise (N, M)) of bf16 x (M, N), fp8 of x * scale;
    amax <- max(amax, max |x|) in place; scale_inv <- 1 / scale if given. Draft: M a multiple of 128,
    N of 64."""
    import torch  # noqa: PLC0415

    m, n = x.shape
    tdt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[dtype]
    hdt = {"e4m3": fp8e4m3, "e5m2": fp8e5m2}[dtype]
    k = _K.get(dtype)
    if k is None:
        k = _K[dtype] = cast_transpose_kernel(f"fp8_cast_transpose_{dtype}", dtype=hdt)
    rw = torch.empty(m, n + pad, device=x.device, dtype=tdt)[:, :n]
    cw = torch.empty(n, m, device=x.device, dtype=tdt)
    s64 = scale.reshape(1, 1).expand(1, COLS).contiguous()
    part = torch.empty(1, (m // (WARPS * 2 * ROWS)) * (n // COLS), device=x.device, dtype=torch.float32)
    k(x, rw, cw, s64, part)
    torch.maximum(amax, part.amax(), out=amax)
    if scale_inv is not None:
        torch.reciprocal(scale, out=scale_inv)
    return rw, cw
