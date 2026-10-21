"""T5 from the docs only: out = x @ W.T + residual, the GEMM written in the DSL with the
residual added to the fp32 accumulators and rounded to bf16 once.

The same GEMM as t4's docs_only.py (copied: one file per solution). gfx950: a ring of S
LDS buffers filled by the DMA S-1 K-tiles ahead, one barrier a K-tile, `s_setprio` around
the MFMAs, block shape per M. N = 4096 gives too few blocks at small M (16x64 blocks: 64
workgroups for 256 CUs), so M <= 256 and the M=1024 down_proj split K over grid z
(KS = 4): fp32 partials to a workspace, then `reduce_res` adds them and the residual and
rounds once (history/33's "split-K as two launches"). Other targets (gfx1100):
examples/04_gemm.py's single-buffered 64x64 kernel with the residual epilogue.

DX log -- what had to be looked up, and where: as t4's docs_only.py, plus
- the residual read straight into the accumulator's layout,
  `r.at[...].load(hk.RT(hk.fp32, TM, TN, "col"), bounded=True)`: by analogy with
  language.md's "global -> registers, bf16 -> fp32" loads; not stated for col layout.
- split-K: history/33 ("Split-K as two launches (fp32 shares to a scratch, then
  reduce_kernel)") and history/41; the in-kernel alternative (`hk.arch.wg_arrive`) is
  documented there as 2-8x slower, so it was not tried.
GAP: every GAP in t4's docs_only.py applies (no fast GEMM outside library source, no gfx950
     GEMM example, multi-buffer spelling only in help(), pipelined-loop tail, the 16-byte
     copy rule, no column bounds, no per-shape tiling guidance, nontemporal slower here,
     pingpong scheduling undocumented, the stale "no gfx950 GEMM" statements, gfx1100 has
     no documented double-buffered GEMM).
GAP: split-K is described (history/33, 41) but never shown: the workspace layout, the grid
     z split, and the reduce kernel were written from scratch (~15 min). Whether
     `hk.cdiv` works on device scalars is not said (used `(a.rows + BM - 1) // BM * BM`).
GAP: a split-K reduce plus the residual had to be a second launch; nothing in the docs says
     whether an fp32 `atomic_add` into a workspace would be faster, and the bf16 output
     cannot take atomics (language.md: "atomic_add takes fp32 tiles only").
GAP: the large-M, deep-K points (down_proj M=4096, 0.81-0.86x torch.compile, which runs
     hipBLASLt at ~1500 TF/s) need what history/33 calls the pingpong schedule; see t4's
     GAP on pingpong.
Performance tried (gfx950): split-K 2/4/8 at M=16-1024 (4 best at M<=256, and at M=1024
with K=12288; none at K=4096, M=1024), 16x32 vs 16x64 blocks at M=16, 128x64 vs 64x64 vs
128x128 at M=128, 128x128 8 warps at M=1024, 256x256 with/without split at M=4096,
B read in halves (spilled), pingpong (slower).
"""

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: LDS GEMM, DMA ring (gfx950) / 04_gemm (gfx1100), residual epilogue, split-K"

_KERNELS = {}
T_R = hk.RT(hk.fp32, 16, 64)


def make_kernel(BM, BN, BK, WM, WN, S, KS):
    """gfx950: a ring of S LDS buffers filled by the DMA S-1 K-tiles ahead, one barrier a
    K-tile. KS > 1 splits K over grid z and writes fp32 partials for reduce_res."""
    W = WM * WN
    TM, TN = BM // WM, BN // WN
    T_A = hk.RT(hk.bf16, TM, BK, "row")
    T_B = hk.RT(hk.bf16, TN, BK, "row")
    T_C = hk.RT(hk.fp32, TM, TN, "col")
    CT = hk.fp32 if KS > 1 else hk.bf16

    @hk.kernel(warps=W, name=f"lin_res_{BM}x{BN}x{BK}_{WM}x{WN}_s{S}_k{KS}",
               grid=lambda p: (hk.cdiv(p.c.cols, BN), hk.cdiv(p.a.rows, BM), KS))
    def lin_res(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], r: hk.Tensor[hk.bf16],
                c: hk.Tensor[CT]):
        g = hk.group(W)
        As = hk.lds.alloc(hk.ST(hk.bf16, BM, BK), name="As", count=S)
        Bs = hk.lds.alloc(hk.ST(hk.bf16, BN, BK), name="Bs", count=S)
        row0 = hk.arch.block_idx.y * BM
        col0 = hk.arch.block_idx.x * BN
        kspan = a.cols // KS
        kb = hk.arch.block_idx.z * kspan
        wid = hk.arch.warp_id()
        wr, wc = wid // WN, wid % WN
        acc = hk.zeros(T_C)

        def copy(j, k):             # the DMA, left in flight; rows past M read zero
            g.load(As[j], a.at[row0, k], wait=False)
            g.load(Bs[j], b.at[col0, k], wait=False)

        def step(j, kn):            # K-tile in buffer j; then refill the buffer read last step
            nonlocal acc
            g.wait_for(As[j], Bs[j])
            hk.arch.barrier()
            if kn is not None:
                copy((j + S - 1) % S, kn)
            at = As[j].subtile(TM, BK, wr, 0).load(T_A, wait=False)
            bt = Bs[j].subtile(TN, BK, wc, 0).load(T_B, wait=False)
            hk.lds.wait_for(at, bt)
            hk.arch.setprio(1)
            acc += at @ bt.T
            hk.arch.setprio(0)

        for j in range(S - 1):
            copy(j, kb + j * BK)
        kl = kb + kspan - S * BK
        for k0 in range(kb, kl, S * BK):
            for j in range(S):
                step(j, k0 + (j + S - 1) * BK)
        step(0, kl + (S - 1) * BK)              # the last S tiles: one copy left to issue
        for j in range(1, S):
            step(j, None)
        if KS > 1:          # fp32 partials: split z owns rows [z * Mp, (z + 1) * Mp)
            mp = (a.rows + BM - 1) // BM * BM
            c.at[hk.arch.block_idx.z * mp + row0 + wr * TM, col0 + wc * TN] = acc
        else:
            acc += r.at[row0 + wr * TM, col0 + wc * TN].load(T_C, bounded=True)
            c.at[row0 + wr * TM, col0 + wc * TN].store(acc, bounded=True)

    return lin_res


@hk.kernel(warps=1, grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def reduce_res(ws: hk.Tensor[hk.fp32], r: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
               KS: hk.Constexpr[int] = 2):
    """o = residual + the KS fp32 partials, rounded once."""
    row = hk.arch.block_idx.y * 16
    col = hk.arch.block_idx.x * 64
    mp = ws.rows // KS
    acc = r.at[row, col].load(T_R, bounded=True)
    for z in range(KS):
        acc += ws.at[z * mp + row, col].load(T_R)
    o.at[row, col].store(acc, bounded=True)


def make_rdna():
    """Other targets (gfx1100): examples/04_gemm.py's single-buffered 64x64 kernel, with A
    read straight from global (bounded, for a ragged M) and the residual epilogue."""
    T_A = hk.RT(hk.bf16, 32, 32, "row")
    T_B = hk.RT(hk.bf16, 16, 32, "row")
    T_C = hk.RT(hk.fp32, 32, 16, "col")

    @hk.kernel(warps=4, grid=lambda p: (hk.cdiv(p.c.cols, 64), hk.cdiv(p.c.rows, 64), 1))
    def lin_res_rdna(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], r: hk.Tensor[hk.bf16],
                     c: hk.Tensor[hk.bf16]):
        g = hk.group(4)
        Bs = hk.lds.alloc(hk.ST(hk.bf16, 64, 32), name="Bs")
        row0 = hk.arch.block_idx.y * 64 + hk.arch.warp_id() // 2 * 32
        col0 = hk.arch.block_idx.x * 64
        wc = hk.arch.warp_id() % 2
        acc = [hk.zeros(T_C) for _ in range(2)]
        for k0 in range(0, a.cols, 32):
            g.load(Bs, b.at[col0, k0])
            hk.arch.barrier()
            bt = [Bs.subtile(16, 32, 2 * wc + h, 0).load(T_B, wait=False) for h in range(2)]
            at = a.at[row0, k0].load(T_A, bounded=True)
            hk.lds.wait_for(bt[0])
            acc[0] += at @ bt[0].T
            hk.lds.wait_for(bt[1])
            acc[1] += at @ bt[1].T
            hk.arch.barrier()
        for h in range(2):
            col = col0 + wc * 32 + 16 * h
            acc[h] += r.at[row0, col].load(T_C, bounded=True)
            c.at[row0, col].store(acc[h], bounded=True)

    return lin_res_rdna


def pick(m, n, k):
    if current_arch() != "gfx950":
        return ()
    if m <= 16:
        cfg = [16, 32 if k <= 4096 else 64, 64, 1, 2, 4, 4]
    elif m <= 256:
        cfg = [128, 64, 64, 2, 2, 4, 4]
    elif m <= 1024 and k <= 4096:
        cfg = [128, 128, 64, 2, 4, 2, 1]
    elif m <= 1024:
        cfg = [256, 256, 64, 4, 2, 2, 4]
    else:
        cfg = [256, 256, 64, 4, 2, 2, 1]
    while cfg[6] > 1 and k % (cfg[6] * cfg[5] * cfg[2]):
        cfg[6] //= 2
    return tuple(cfg)


def _kernel(cfg):
    if cfg not in _KERNELS:
        _KERNELS[cfg] = make_kernel(*cfg) if cfg else make_rdna()
    return _KERNELS[cfg]


def supports(x, weight, residual):
    cfg = pick(x.shape[0], weight.shape[0], x.shape[1])
    bn, kq = (cfg[1], cfg[2] * cfg[5] * cfg[6]) if cfg else (64, 32)
    if weight.shape[0] % bn or x.shape[1] % kq:
        return f"N must be a multiple of {bn} and K of {kq}"
    return True


def run(cfg, x, weight, residual):
    m, n = x.shape[0], weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if not cfg or cfg[6] == 1:
        _kernel(cfg)(x, weight, residual, out)
        return out
    mp = -(-m // cfg[0]) * cfg[0]
    ws = torch.empty((cfg[6] * mp, n), device=x.device, dtype=torch.float32)
    _kernel(cfg)(x, weight, residual, ws)
    reduce_res(ws, residual, out, KS=cfg[6])
    return out


def solve(x, weight, residual):
    return run(pick(x.shape[0], weight.shape[0], x.shape[1]), x, weight, residual)
