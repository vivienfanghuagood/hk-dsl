"""T4 from the docs only: out = gelu_tanh(x @ W.T + b), the GEMM written in the DSL with the
bias and GELU applied to the fp32 accumulators before the single bf16 store.

gfx950: examples/04_gemm.py's shape (C = A.B^T through LDS, `acc += a @ b.T`), rebuilt
with what language.md "Copies left in flight (gfx950)" and history/33 describe: a ring of S
LDS buffers per operand filled by the DMA (`g.load(..., wait=False)`) S-1 K-tiles ahead,
`g.wait_for` + one barrier a K-tile, `s_setprio` around the MFMAs, block shape picked per M
(16x64 for decode, 128x64 at M=128, 256x192/256x256 with 8 warps at M>=1024). Other
targets (gfx1100): 04_gemm's single-buffered 64x64 kernel with the same epilogue.

DX log -- what had to be looked up, and where:
- GEMM structure, LDS staging, async LDS reads + `hk.lds.wait_for`: examples/04_gemm.py.
- gfx950 DMA left in flight, `g.wait_for`, the cross-warp barrier rule: language.md
  "Copies left in flight (gfx950)"; the ring schedule ("a ring of S LDS buffers with the
  copies S-1 K-tiles ahead, and one barrier a K-tile") is one sentence in history/33.
- 16x32 operand fragments, wave64, the 160 KB LDS, AGPRs: design/cdna.md, history/29.
- `hk.ST(..., count)` / `hk.lds.alloc(..., count=)` (multi-buffer), `g.load`'s keywords,
  `SharedTile.subtile`, `hk.arch.setprio`: help() (runtime lookups).
- The bias as a row vector, `e.at[0, c].load(hk.row_vec(T))[None, :]`: examples/
  10_layernorm_backward.py (`w.at[0, cb * 64].load(RV)[None, :]`); a 1-D tensor is
  (1, N) by language.md "A tensor with fewer than four dimensions fills the trailing ones".
- gelu is the tanh form and fp32-only: from the t1 docs-only attempt (see GAP below).
- Ragged M: history/33 "the buffer descriptor ... ends at the tensor's last row, so a
  partial last tile reads zeros" + `store(..., bounded=True)` (language.md "Memory").
GAP: examples/04_gemm.py and examples/README.md send the reader to hk/ops/gemm.py ("read it
     next") for everything past a single-buffered GEMM; that is library source, so for a
     docs-only user there is no documented fast GEMM at all. The pieces are scattered over
     language.md, design/cdna.md and history/28, 29, 33, 41; ~30 min of reading.
GAP: 04_gemm's tile facts (32x16 col accumulators, 16-row B halves, BK=32) are gfx1100's
     WMMA ones. Nothing in the example says what to change on gfx950 (16x32 operand
     fragment, wave64, the DMA, 160 KB LDS); there is no gfx950 GEMM example. Worked it out
     from design/cdna.md + history/33; the first gfx950 kernel was correct at once (the
     passes catch the hazards), the cost was choosing shapes.
GAP: how to declare a double buffer is not shown anywhere: language.md says "a buffer of a
     stack picked by a Python int" and shows `As[1]`, ir.md shows `!hk.st<bf16, 128x64, 2>`,
     but the spelling had to come from help(hk.ST) (`count`) and help(hk.lds.alloc)
     (`count=`) -- two ways, no word on which to use.
GAP: no documented pattern for the tail of a pipelined K loop. A clamped (redundant) last
     copy is refused: "the kernel ends with a copy into As[0] still in flight" (a good
     error). Fixed by peeling the last S steps (`step(j, None)`); ~5 min.
GAP: `g.load(wait=False)` needs the tile to be a whole number of 16-byte copies per thread
     (VerifyError "the tile is not a whole number of 16-byte copies per thread"): not in the
     docs. It ruled out 256x224 with 8 warps (a block shape that fills the 256 CUs at
     N=14336, which history/41 recommends) and 16x64 with 4 warps.
GAP: nothing bounds a tile's columns (bounded= is rows only for register tiles), so BN must
     divide N; history/41's 96- and 224-wide blocks therefore need N to suit. supports()
     declines other N.
GAP: no guidance on block shape / warps / BK / ring depth per M on gfx950. history/33/41's
     route tables name candidates ("dp64x64", "pp", "dp32x48nt") but not their parameters.
     Swept ~40 configurations by hand (~40 min).
GAP: history/41 says weights should be streamed `nontemporal` from a `shape="16x64"` LDS
     tile; here (inputs hot in the 256 MB MALL between calls, as the suite measures) it was
     20-35% SLOWER at every M (M=16: 21.3 vs 16.8 us). The docs do not say when not to.
GAP: `s_setprio` around the math (history/33: "worth 10%") also decided whether the
     256x256 8-warp (2x4) kernel spilled (ResourceError without it, 1141 TF/s with it);
     nothing explains that. The spill gate's message is clear.
GAP: `hk.arch.pingpong` is documented (language.md, help) but not how to schedule a loop
     inside it; my try (S=2, next copy waited inside the memory phase so the two-barrier
     rule holds) passed the pass's checks and was correct, but 2-8% slower than no
     pingpong. The working schedule is only in hk/ops/gemm_cdna.py (library source).
GAP: help(hk.math.gelu) says only "kittens::gelu": not which GELU (tanh) nor that it is
     fp32-only (language.md "Errors and locations" implies it). Taken from t1's finding.
GAP: (wrong) language.md "Limits": "On gfx950, hk.ops' GEMM and attention have no native
     form yet", and examples/README.md: "attention has no CDNA path yet ... validate.py
     skips the GEMM" -- contradicted by design/cdna.md's status table (rows 9 and 11),
     history/33 and examples/05_attention.py's wave64 body.
GAP: gfx1100: the DMA (`wait=False`) is gfx950-only, and the register staging pipeline
     (`g.stage_buffer`/`stage`/`commit`, `hk.arch.vm_wait`) is named only in help(); no doc
     or example shows a double-buffered RDNA GEMM. Used 04_gemm's single-buffered schedule.
     Whether `g.load` past the tensor's last row is safe on gfx1100 is not documented, so A
     is read straight from global with `bounded=True` there.
Performance tried (gfx950): 128x128 4 warps S=2 (0.77-0.80x torch.compile at M>=1024) ->
256x256 8 warps + setprio (0.90x at M=4096) -> CU-filling 256x192 at M=1024 (1.04x);
small M: 16x64 2 warps, 4-deep ring (1.3-1.4x); S=4, BK=32, B in halves (spilled),
16 warps (resource gate), nontemporal, pingpong: no gain.
"""

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: LDS GEMM, DMA ring (gfx950) / 04_gemm (gfx1100), bias+gelu epilogue"

_KERNELS = {}


def make_kernel(BM, BN, BK, WM, WN, S):
    """gfx950: a ring of S LDS buffers filled by the DMA S-1 K-tiles ahead, one barrier a K-tile."""
    W = WM * WN
    TM, TN = BM // WM, BN // WN
    T_A = hk.RT(hk.bf16, TM, BK, "row")
    T_B = hk.RT(hk.bf16, TN, BK, "row")
    T_C = hk.RT(hk.fp32, TM, TN, "col")

    @hk.kernel(warps=W, name=f"lin_gelu_{BM}x{BN}x{BK}_{WM}x{WN}_s{S}",
               grid=lambda p: (hk.cdiv(p.c.cols, BN), hk.cdiv(p.c.rows, BM), 1))
    def lin_gelu(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], e: hk.Tensor[hk.bf16],
                 c: hk.Tensor[hk.bf16]):
        g = hk.group(W)
        As = hk.lds.alloc(hk.ST(hk.bf16, BM, BK), name="As", count=S)
        Bs = hk.lds.alloc(hk.ST(hk.bf16, BN, BK), name="Bs", count=S)
        row0 = hk.arch.block_idx.y * BM
        col0 = hk.arch.block_idx.x * BN
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
            copy(j, j * BK)
        kl = a.cols - S * BK
        for k0 in range(0, kl, S * BK):
            for j in range(S):
                step(j, k0 + (j + S - 1) * BK)
        step(0, kl + (S - 1) * BK)              # the last S tiles: one copy left to issue
        for j in range(1, S):
            step(j, None)
        acc += e.at[0, col0 + wc * TN].load(hk.row_vec(T_C))[None, :]
        hk.math.gelu(acc, into=acc)
        c.at[row0 + wr * TM, col0 + wc * TN].store(acc, bounded=True)

    return lin_gelu


def make_rdna():
    """Other targets (gfx1100): examples/04_gemm.py's single-buffered 64x64 kernel, with A
    read straight from global (bounded, for a ragged M) and the epilogue."""
    T_A = hk.RT(hk.bf16, 32, 32, "row")
    T_B = hk.RT(hk.bf16, 16, 32, "row")
    T_C = hk.RT(hk.fp32, 32, 16, "col")

    @hk.kernel(warps=4, grid=lambda p: (hk.cdiv(p.c.cols, 64), hk.cdiv(p.c.rows, 64), 1))
    def lin_gelu_rdna(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], e: hk.Tensor[hk.bf16],
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
            acc[h] += e.at[0, col].load(hk.row_vec(T_C))[None, :]
            hk.math.gelu(acc[h], into=acc[h])
            c.at[row0, col].store(acc[h], bounded=True)

    return lin_gelu_rdna


def pick(m, n, k):
    if current_arch() != "gfx950":
        return ()
    if m <= 16:
        return (16, 64, 64, 1, 2, 4)
    if m <= 256:
        return (128, 64, 64, 2, 2, 4)
    if m <= 1024 and n % 192 == 0:
        return (256, 192, 64, 4, 2, 2)
    return (256, 256, 64, 4, 2, 2)


def _kernel(cfg):
    if cfg not in _KERNELS:
        _KERNELS[cfg] = make_kernel(*cfg) if cfg else make_rdna()
    return _KERNELS[cfg]


def supports(x, weight, bias):
    cfg = pick(x.shape[0], weight.shape[0], x.shape[1])
    bn, kq = (cfg[1], cfg[2] * cfg[5]) if cfg else (64, 32)
    if weight.shape[0] % bn or x.shape[1] % kq:
        return f"N must be a multiple of {bn} and K of {kq}"
    return True


def solve(x, weight, bias):
    m, n = x.shape[0], weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _kernel(pick(m, n, x.shape[1]))(x, weight, bias.view(1, n), out)
    return out
