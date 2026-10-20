"""LayerNorm backward: Triton's tutorial 05 backward pass, as two kernels.

From dy, x, the weight w and each row's mean and rstd saved by the forward:

    xhat = (x - mean) * rstd                 wdy = w * dy
    dx   = (wdy - (xhat * c1 + c2)) * rstd   c1 = mean(xhat * wdy), c2 = mean(wdy)
    dw   = sum over rows of dy * xhat        db = sum over rows of dy

dx is a row's business: a row reduction for c1 and c2, then the row again.
dw and db sum over every row, which no workgroup holds. Triton's tutorial
adds each row's share into GROUP_SIZE_M buffers under a spin lock, then sums
those in a second kernel. Here each workgroup sums its own sixteen rows in
registers -- `sum(axis=0)`, a column reduction -- and writes them to a row of
the partials that is its alone: no lock, no atomic. The second kernel sums
the partials' rows.

Columns have to be a multiple of 64 and rows of 16 here.

    python3 examples/10_layernorm_backward.py
    python3 examples/10_layernorm_backward.py --dump cpp
"""

import _common

import hk

T = hk.RT(hk.fp32, 16, 64)
VT, RV = hk.col_vec(T), hk.row_vec(T)      # one value per row / per column


def plan(cols: int):
    """(warps, tiles held per warp) for a row of `cols`. Held, x-hat and dy
    are read once. A 16 x 64 fp32 pair is 64 VGPRs and two of them with
    their temporaries ~208, which 16 warps afford (256 a lane) and 32 do not
    (192): a held row is at most 2048 wide. Wider, it is read twice -- the
    second time, at 8192, from HBM. Sixteen rows of 4096 are more registers
    than a WGP has to give them; Triton's program, one row, holds it."""
    blocks = cols // 64
    if blocks <= 32 and blocks % 2 == 0:
        return blocks // 2, 2
    return 8, 0


def ln_bwd_dx_kernel(warps: int = 8, tpw: int = 0):
    @hk.kernel(warps=warps, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1),
               name=f"ln_bwd_dx_w{warps}_t{tpw}")
    def ln_bwd_dx(x: hk.Tensor[hk.bf16], dy: hk.Tensor[hk.bf16],
                  w: hk.Tensor[hk.bf16], mean: hk.Tensor[hk.fp32],
                  rstd: hk.Tensor[hk.fp32], dx: hk.Tensor[hk.bf16],
                  pdw: hk.Tensor[hk.fp32], pdb: hk.Tensor[hk.fp32],
                  WARPS: hk.Constexpr[int] = warps, TPW: hk.Constexpr[int] = tpw):
        rb, wid = hk.arch.block_idx.y, hk.arch.warp_id()
        r0 = rb * 16
        mu = mean.at[0, r0].load(VT)
        rs = rstd.at[0, r0].load(VT)

        def operands(cb):
            xh = (x.at[r0, cb * 64].load(T, bounded=True) - mu[:, None]) * rs[:, None]
            return cb, xh, dy.at[r0, cb * 64].load(T, bounded=True)

        def wdy_of(cb, g):
            return g * w.at[0, cb * 64].load(RV)[None, :]

        # TPW tiles held (a Python list: unrolled), or a runtime loop that
        # reads each block again in the second pass.
        held = [operands(s * WARPS + wid) for s in range(TPW)]

        def visit(fn):
            if TPW:
                for rec in held:
                    fn(*rec)
            else:
                for cb in range(wid, hk.cdiv(x.cols, 64), WARPS):
                    fn(*operands(cb))

        c1, c2 = hk.zeros(VT), hk.zeros(VT)

        def reduce(cb, xh, g):
            nonlocal c1, c2
            wdy = wdy_of(cb, g)
            c1 += (xh * wdy).sum(axis=1)
            c2 += wdy.sum(axis=1)

        def write(cb, xh, g):
            d = (wdy_of(cb, g) - (xh * c1[:, None] + c2[:, None])) * rs[:, None]
            dx.at[r0, cb * 64].store(d, bounded=True)
            pdw.at[rb, cb * 64] = (g * xh).sum(axis=0)    # this block's rows, summed
            pdb.at[rb, cb * 64] = g.sum(axis=0)

        visit(reduce)
        c1 = hk.collective.cross_warp(c1, "add", WARPS) / x.cols
        c2 = hk.collective.cross_warp(c2, "add", WARPS) / x.cols
        visit(write)

    return ln_bwd_dx


def strip_plan(cols: int):
    """(warps, chunks) for a row of `cols` on strips: each lane holds two
    16-byte runs of bf16 (16 columns) of a row, as hk.ops.norm's strips do,
    and the warps cover the row exactly; None where they cannot."""
    span = 32 * RUN
    if cols % (2 * span):
        return None
    warps = cols // (2 * span)
    return (warps, 2) if warps <= 16 else None


RUN = 8                                     # bf16 a lane reads at once: 16 bytes


def ln_bwd_strip_kernel(warps: int, chunks: int = 2):
    """dx on strips, and each workgroup's share of dw and db.

    The tile kernel's 16-row tile cannot hold a wide row in registers, so
    past 2048 columns it reads the row twice. A strip is one row, each lane
    holding 16-byte runs of it and nothing twice; `warps` of them cover the
    row, so it is read once at any width. A workgroup walks rows g, g + G,
    ... (G workgroups, one partial row each), the next row's x and dy in
    flight while this one is reduced, and keeps its rows' dy * x-hat and dy
    in registers: one partial row a workgroup, written once, which the
    second kernel sums."""
    cw = chunks * 32 * RUN
    S = hk.RS(hk.fp32, 1, cw, RUN)
    CV, RS_ = hk.col_vec(S), hk.row_vec(S)

    @hk.kernel(warps=warps, grid=lambda p: (p.pdw.rows, 1, 1),
               name=f"ln_bwd_strip_w{warps}_c{chunks}")
    def ln_bwd_strip(x: hk.Tensor[hk.bf16], dy: hk.Tensor[hk.bf16],
                     w: hk.Tensor[hk.bf16], mean: hk.Tensor[hk.fp32],
                     rstd: hk.Tensor[hk.fp32], dx: hk.Tensor[hk.bf16],
                     pdw: hk.Tensor[hk.fp32], pdb: hk.Tensor[hk.fp32]):
        g, groups = hk.arch.block_idx.x, pdw.rows
        wid = hk.arch.warp_id()
        c0 = wid * cw
        n = x.cols
        # Both row sums of every warp, for two rows in turn: a row's partials
        # go out, one barrier, and every warp reads all of them. The next
        # row writes the other half, so nothing overwrites what a slower warp
        # is still reading without a barrier in between.
        xw = hk.lds.alloc_vec(hk.lds.shared_vec(CV, 4 * warps), name="xw")
        wv = w.at[0, c0].load(RS_)
        acc_w, acc_b = hk.zeros(S), hk.zeros(S)
        xs, gs = hk.zeros(S), hk.zeros(S)
        # Rows past the last read zero; a workgroup with no rows writes zeros.
        xn = x.at[g, c0].load(S, bounded=True)
        gn = dy.at[g, c0].load(S, bounded=True)
        it = 0
        for r in range(g, x.rows, groups):
            xn.copy(into=xs)
            gn.copy(into=gs)
            x.at[r + groups, c0].load(S, bounded=True, into=xn)
            dy.at[r + groups, c0].load(S, bounded=True, into=gn)
            mu = mean.at[0, r].load(CV)
            rs = rstd.at[0, r].load(CV)
            xh = (xs - mu) * rs
            wdy = gs * wv
            base = (it % 2) * (2 * warps)
            xw.store((xh * wdy).sum(axis=1).value(), base + wid)
            xw.store(wdy.sum(axis=1).value(), base + warps + wid)
            hk.arch.barrier(drain=True)
            c1 = xw.load(CV, base)
            c2 = xw.load(CV, base + warps)
            for k in range(1, warps):
                c1 += xw.load(CV, base + k)
                c2 += xw.load(CV, base + warps + k)
            c1 /= n
            c2 /= n
            it = it + 1
            d = (wdy - (xh * c1 + c2)) * rs
            dx.at[r, c0].store(d)
            acc_w += gs * xh
            acc_b += gs
        pdw.at[g, c0].store(acc_w)
        pdb.at[g, c0].store(acc_b)

    return ln_bwd_strip


def ln_bwd_dwdb_kernel(warps: int = 4):
    @hk.kernel(warps=warps, grid=lambda p: (hk.cdiv(p.pdw.cols, 64), 1, 1),
               name=f"ln_bwd_dwdb_w{warps}")
    def ln_bwd_dwdb(pdw: hk.Tensor[hk.fp32], pdb: hk.Tensor[hk.fp32],
                    dw: hk.Tensor[hk.fp32], db: hk.Tensor[hk.fp32],
                    WARPS: hk.Constexpr[int] = warps):
        c0 = hk.arch.block_idx.x * 64
        aw, ab = hk.zeros(T), hk.zeros(T)
        for r in range(hk.arch.warp_id(), hk.cdiv(pdw.rows, 16), WARPS):
            aw += pdw.at[r * 16, c0].load(T, bounded=True)
            ab += pdb.at[r * 16, c0].load(T, bounded=True)
        sw, sb = hk.zeros(RV), hk.zeros(RV)
        sw += aw.sum(axis=0)
        sb += ab.sum(axis=0)
        dw.at[0, c0] = hk.collective.cross_warp(sw, "add", WARPS)
        db.at[0, c0] = hk.collective.cross_warp(sb, "add", WARPS)

    return ln_bwd_dwdb


ln_bwd_dwdb = ln_bwd_dwdb_kernel()


def _triton_ln_bwd():
    """The tutorial's two kernels, as published (bf16 partials and all)."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def bwd_dx_fused(DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride, N,
                     GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + cols, dx, mask=mask)
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)
        tl.debug_barrier()
        tl.atomic_xchg(Lock, 0)

    @triton.jit
    def bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        tl.store(FINAL_DW + cols, tl.sum(dw, axis=0), mask=cols < N)
        tl.store(FINAL_DB + cols, tl.sum(db, axis=0), mask=cols < N)

    def make(x, w):
        m, n = x.shape
        group = 64
        if n <= 8192:
            group = 96
        if n <= 4096:
            group = 128
        if n <= 1024:
            group = 256
        dev = x.device
        locks = torch.zeros(2 * group, dtype=torch.int32, device=dev)
        pdw = torch.zeros((group, n), dtype=x.dtype, device=dev)
        pdb = torch.zeros((group, n), dtype=x.dtype, device=dev)
        dw = torch.empty((n,), dtype=w.dtype, device=dev)
        db = torch.empty((n,), dtype=w.dtype, device=dev)
        block = triton.next_power_of_2(n)
        warps = min(max(block // 256, 1), 8)

        def run(dy, x, mean, rstd, dx):
            locks.zero_()                  # the lock protocol counts from zero
            bwd_dx_fused[(m,)](dx, dy, pdw, pdb, x, w, mean, rstd, locks,
                               x.stride(0), n, GROUP_SIZE_M=group,
                               BLOCK_SIZE_N=block, num_warps=warps)
            bwd_dwdb[(triton.cdiv(n, 128),)](pdw, pdb, dw, db, min(group, m), n,
                                            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
            return dw, db
        return run
    return make


def groups_for(rows: int, warps: int) -> int:
    """Workgroups (= partial rows) for the strip path: about 768 waves in
    flight, measured on a W7900D at 4096 rows (192 workgroups were best at
    1024 and 2048 columns, 96 at 4096 and 8192)."""
    return max(1, min(rows, 768 // warps))


def main():
    args = _common.parse(__doc__, lambda p: p.add_argument(
        "--groups", default="", help="strip path: workgroup counts to time, "
        "comma separated (default: groups_for)"))
    dx_kernel = ln_bwd_dx_kernel()
    if any([_common.static(args, k) for k in (dx_kernel, ln_bwd_dwdb)]):
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    eps = 1e-5
    for m, n in ((4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192)):
        g = torch.Generator(device="cuda").manual_seed(n)
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)
        dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)
        w = (1 + 0.1 * torch.randn(n, device="cuda", generator=g)).to(torch.bfloat16)
        b = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
        xf = x.float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        xr = xf.clone().requires_grad_()
        wr = w.float().requires_grad_()
        br = b.float().requires_grad_()
        F.layer_norm(xr, (n,), wr, br, eps).backward(dy.float())

        dx = torch.empty_like(x)
        pdw = torch.empty(m // 16, n, device="cuda", dtype=torch.float32)
        pdb = torch.empty_like(pdw)
        dw = torch.empty(n, device="cuda", dtype=torch.float32)
        db = torch.empty_like(dw)

        warps, tpw = plan(n)
        k = ln_bwd_dx_kernel(warps, tpw)

        def hk_bwd():
            k(x, dy, w, mean, rstd, dx, pdw, pdb)
            ln_bwd_dwdb(pdw, pdb, dw, db)

        print(f"\nLayerNorm backward, {m} x {n} bf16 (tiles: {warps} warps, "
              f"{'row held' if tpw else 'row read twice'})")
        hk_bwd()
        _common.check("tiles: dx vs fp32 autograd", dx, xr.grad, rel=2e-2)
        _common.check("tiles: dw vs fp32 autograd", dw, wr.grad, rel=1e-3)
        _common.check("tiles: db vs fp32 autograd", db, br.grad, rel=1e-3)

        mean2, rstd2 = mean[:, None], rstd[:, None]
        mask = [True, True, True]
        fns = {"hk tiles": hk_bwd}

        sp = strip_plan(n)
        if sp is not None:
            sk = ln_bwd_strip_kernel(*sp)
            gs = ([int(v) for v in args.groups.split(",")] if args.groups
                  else [groups_for(m, sp[0])])
            for G in gs:
                s_dx = torch.empty_like(x)
                s_pdw = torch.empty(G, n, device="cuda", dtype=torch.float32)
                s_pdb = torch.empty_like(s_pdw)
                s_dw, s_db = torch.empty_like(dw), torch.empty_like(db)

                def hk_strip(s_dx=s_dx, s_pdw=s_pdw, s_pdb=s_pdb, s_dw=s_dw,
                             s_db=s_db):
                    sk(x, dy, w, mean, rstd, s_dx, s_pdw, s_pdb)
                    ln_bwd_dwdb(s_pdw, s_pdb, s_dw, s_db)

                hk_strip()
                tag = f"hk strips G={G}"
                _common.check(f"strips G={G}: dx", s_dx, xr.grad, rel=2e-2)
                _common.check(f"strips G={G}: dw", s_dw, wr.grad, rel=1e-3)
                _common.check(f"strips G={G}: db", s_db, br.grad, rel=1e-3)
                fns[tag] = hk_strip
        fns["torch native_layer_norm_backward"] = (
            lambda: torch.ops.aten.native_layer_norm_backward(
                dy, x, (n,), mean2, rstd2, w, b, mask))
        try:
            tri = _triton_ln_bwd()(x, w)
            t_dx = torch.empty_like(x)
            tri(dy, x, mean, rstd, t_dx)
            torch.cuda.synchronize()
            _common.check("Triton tutorial dx vs fp32", t_dx, xr.grad, rel=2e-2)
            fns["Triton tutorial"] = lambda: tri(dy, x, mean, rstd, t_dx)
        except ImportError:
            pass
        if args.no_bench:
            continue
        best = _common.compare(fns, args.rounds)
        base = "Triton tutorial" if "Triton tutorial" in fns else \
            "torch native_layer_norm_backward"
        _common.report("time", best, baseline=base,
                       work=3 * x.numel() * x.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
