"""RMSNorm over the last axis: a runtime loop, a row reduction, a cross-warp
combine and a broadcast -- on a register tile, and on a register strip.

o = x * rsqrt(mean(x^2) + eps). The tile kernel takes one workgroup per 16
rows. The row is split across WARPS warps, each walking every WARPS-th
64-column block; the per-warp sums meet through LDS
(`hk.collective.cross_warp`), and the scale is broadcast back along the rows
with `scale[:, None]`. Columns have to be a multiple of 64 and rows of 16.

The strip kernel takes one workgroup per row and holds it: warp w owns
columns [w*CW, (w+1)*CW) as an `hk.RS` strip, in which each lane owns runs of
8 contiguous columns -- one 16-byte load each, which is the layout a row
kernel wants and a WMMA tile does not have. It reads the row once. Strips
are include/rdna3's, so on another target the strip kernel is skipped.
hk.ops.norm.rmsnorm, which both are compared against, runs that strip with a
planned warp count, at any width.

    python3 examples/02_rmsnorm.py
    python3 examples/02_rmsnorm.py --dump ir
"""

import _common

import hk

T = hk.RT(hk.fp32, 16, 64)


def rmsnorm_kernel(warps: int = 4):
    @hk.kernel(warps=warps, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), p.x.batch))
    def rmsnorm(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
                WARPS: hk.Constexpr[int] = warps, EPS: hk.Constexpr[float] = 1e-6):
        row0, wid = hk.arch.block_idx.y * 16, hk.arch.warp_id()
        sq = hk.zeros(hk.col_vec(T))
        for cb in range(wid, hk.cdiv(x.cols, 64), WARPS):        # runtime loop -> scf.for
            v = x.at[row0, cb * 64].load(T)                       # bf16 -> fp32 on the way in
            sq += (v * v).sum(axis=1)                             # row_sum, accumulated in place
        scale = hk.math.rsqrt(hk.collective.cross_warp(sq, "add", WARPS) / x.cols + EPS)
        for cb in range(wid, hk.cdiv(x.cols, 64), WARPS):
            o.at[row0, cb * 64] = (x.at[row0, cb * 64].load(T) * scale[:, None]).to(hk.bf16)

    return rmsnorm


def rmsnorm_strip_kernel(cols: int = 4096, warps: int = 8):
    cw = cols // warps
    S = hk.RS(hk.fp32, 1, cw)                                     # 1 row, cw columns a warp

    @hk.kernel(warps=warps, grid=lambda p: (p.x.rows, 1, 1))
    def rmsnorm_strip(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
                      EPS: hk.Constexpr[float] = 1e-6):
        row, c0 = hk.arch.block_idx.x, hk.arch.warp_id() * cw
        v = x.at[row, c0].load(S)                                 # cw/256 x 16 bytes a lane
        sq = hk.collective.cross_warp((v * v).sum(axis=1), "add", warps)
        v *= hk.math.rsqrt(sq / x.cols + EPS)                     # the col_vec broadcasts
        o.at[row, c0].store(v)

    return rmsnorm_strip


def main():
    args = _common.parse(__doc__)
    kernel = rmsnorm_kernel()
    if _common.static(args, kernel):
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    rows, cols = 8192, 4096
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    print(f"RMSNorm, {rows} x {cols} bf16, eps 1e-6")
    kernel(x, o)
    xf = x.float()
    want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    _common.check("this kernel vs fp32 torch", o, want, rel=2e-2)
    contenders = {"hk (this kernel)": lambda: kernel(x, o)}
    from hk.target import current_arch, get_target  # noqa: PLC0415

    if get_target(current_arch()).kittens_define == "KITTENS_RDNA3":
        strip = rmsnorm_strip_kernel(cols)
        strip(x, o)
        _common.check("the strip kernel vs fp32 torch", o, want, rel=2e-2)
        contenders["hk (the strip kernel)"] = lambda: strip(x, o)
    else:
        print(f"  the strip kernel: skipped, strips are include/rdna3's "
              f"and this is {current_arch()}")
    if args.no_bench:
        return
    best = _common.compare({
        **contenders,
        "hk.ops.norm.rmsnorm": lambda: hk.ops.norm.rmsnorm(x, out=o),
        "torch rms_norm": lambda: F.rms_norm(x, (cols,), eps=1e-6),
    }, args.rounds)
    _common.report("time", best, baseline="torch rms_norm",
                   work=2 * x.numel() * x.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
