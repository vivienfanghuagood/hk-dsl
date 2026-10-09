"""bf16 GEMM with fp32 accumulate, written in the DSL.

A transcription of kernels/rdna3/gemm/bf16fp32/gemm.cpp, which is 995 lines of
C++ and twenty `#define` knobs. The gate for this file is not "it works": it is
that the generated kernel matches the handwritten one on ScratchSize,
occupancy and TFLOPs, because an IR that can only express the slow version of a
kernel is an IR nobody will use for the fast one.

What the schedule is, in one paragraph. The workgroup holds two LDS buffers
each of A (BLOCK_M x K_STEP) and B (BLOCK_N x K_STEP), and alternates between
them. Each K-tile: read operand slices out of the resident buffer and multiply,
then write the *other* buffer from registers that were filled by global loads
issued an iteration ago, then issue the next iteration's loads. One barrier per
K-tile, not two -- the write-after-read hazard on the buffer being overwritten
is already covered by the previous iteration's barrier, so only the
write-before-read hazard needs a fresh one.

Three things the DSL does that the C++ cannot:

  * **The lgkmcnt immediates are derived.** The C++ computes them with a
    constexpr `reads_in_flight()` plus a `TAIL_OPS` term for the interleaved
    ds_writes, and a paragraph explaining why. Here `lds_wait_for` names the
    tiles it retires and `hk.ir.passes.lds_pipeline` walks the issue queue. An
    immediate that is too large does not over-wait, it fails to wait at all --
    so this is the difference between a schedule you can edit and one you can
    only admire.
  * **The operand binding is not optional.** There is no bare `lds_wait` in the
    DSL to reach for.
  * **The spill gate is in the build.** `max_vgprs`/`min_occupancy` below are
    part of compiling this kernel, so a change that costs a wave fails to
    produce a `.so` rather than quietly producing a slower one.

Shapes. B is passed pre-transposed as (N, K): it keeps both shared loads
contiguous and makes `mma_ABt` the right primitive, which is the whole reason
the operand tiles are both row-layout and therefore both on the ds_read_b128
fast path.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .. import lang as _ops
from ..lang import GL, const, kernel as _kernel
from ..ir.nodes import bf16, fp32


def _derive(block_m: int, block_n: int, k_step: int, dot_slice: int,
            warps: int, warp_rows: int, n_split: int) -> Dict[str, int]:
    """The config arithmetic, in one place, with the C++'s static_asserts."""
    warp_cols = warps // warp_rows
    if warp_rows * warp_cols != warps:
        raise ValueError(f"warp grid {warp_rows}x{warp_cols} does not cover {warps} warps")
    reg_m = block_m // warp_rows
    reg_n = block_n // warp_cols
    split_n = reg_n // n_split
    if split_n % 16 or split_n * n_split != reg_n:
        raise ValueError(
            f"N_SPLIT={n_split} must cut the {reg_n}-wide warp tile into whole "
            f"16-column chunks"
        )
    if k_step % dot_slice:
        raise ValueError(f"DOT_SLICE={dot_slice} must divide K_STEP={k_step}")
    lds = 2 * (block_m + block_n) * k_step * 2
    if lds > 65536:
        raise ValueError(f"{lds} B of LDS; a gfx1100 workgroup has 65536")
    return dict(
        warp_cols=warp_cols, reg_m=reg_m, reg_n=reg_n, split_n=split_n,
        slices=k_step // dot_slice, lds=lds,
    )


def gemm_kernel(name: str, *, block_m: int = 128, block_n: int = 128,
                k_step: int = 64, dot_slice: int = 16, warps: int = 8,
                warp_rows: int = 4, n_split: int = 4, wgm: int = 8,
                write_pos: int = 2,
                max_vgprs: int = 256, min_occupancy: int = 1):
    """The default tiling is `big_config` from the C++: 128x128x64, 8 warps.

    That is a 32x64 warp tile rather than the 64x64 everything else on this
    GPU runs, and the README next to the C++ has the sweep that justifies it:
    every 64x64 variant either spills or falls to 5-6 waves/SIMD, and the ones
    that avoid both do it by taking K_STEP down to 16, straight back into the
    global-latency penalty. 1.5x the LDS traffic per WMMA, bought with the
    registers to run a 64-deep K-tile at 7 waves/SIMD.

    `write_pos` is where the staging drain lands relative to the math: 2 puts
    it inside the last-but-one slice, so the ds_writes complete behind a whole
    slice of WMMAs and the waits downstream carry them. The C++ computes that
    carry by hand and calls it TAIL_OPS; here lds_pipeline derives it, which
    is the difference between a schedule you can edit and one you can only
    admire.
    """
    d = _derive(block_m, block_n, k_step, dot_slice, warps, warp_rows, n_split)
    warp_cols, reg_m, reg_n = d["warp_cols"], d["reg_m"], d["reg_n"]
    split_n, slices = d["split_n"], d["slices"]

    st_a = _ops.st(bf16, block_m, k_step)
    st_b = _ops.st(bf16, block_n, k_step)

    def body(a, b, c, *, WGM: const = wgm):
        g = _ops.group(warps)

        As = _ops.alloc_shared(st_a, count=2, name="As")
        Bs = _ops.alloc_shared(st_b, count=2, name="Bs")
        buf_a = g.stage_buffer(As, name="buf_a")
        buf_b = g.stage_buffer(Bs, name="buf_b")

        A_tile = _ops.zeros(_ops.rt(bf16, reg_m, dot_slice, "row"))
        B_tile = [_ops.zeros(_ops.rt(bf16, split_n, dot_slice, "row"))
                  for _ in range(n_split)]
        C_acc = [_ops.zeros(_ops.rt(fp32, reg_m, split_n, "col"))
                 for _ in range(n_split)]

        M = _ops.rows(a)
        m_blocks = _ops.s_cdiv(M, block_m)
        n_blocks = _ops.rows(b) // block_n

        # L2 swizzle: walk WGM block-rows at a time so the B panel a group of
        # workgroups reads stays resident. One XCD here, so no chiplet term.
        wgid = _ops.block_idx.x
        per_group = WGM * n_blocks
        group_id = wgid // per_group
        first_m = group_id * WGM
        group_m = _ops.s_min(m_blocks - first_m, WGM)
        row = first_m + ((wgid % per_group) % group_m)
        col = (wgid % per_group) // group_m

        # M remainder by backing the last block up, not by predicating it: the
        # staging path has no bounds check, and predicating the issue would
        # make the right vmcnt a runtime value, which no instruction accepts.
        # The overlap recomputes rows from the same A and the same full K, so
        # both workgroups write identical bytes.
        m_base = _ops.s_max(_ops.s_min(row * block_m, M - block_m), 0)

        wid = _ops.warp_id()
        warp_row = wid // warp_cols
        warp_col = wid % warp_cols

        n_tiles = _ops.cols(a) // k_step

        def a_coord(t):
            return _ops.elem_coord(0, 0, m_base, t * k_step)

        def b_coord(t):
            return _ops.tile_coord(0, 0, col, t)

        def stage(t):
            # Clamped rather than predicated: past the end this re-reads the
            # last tile into staging registers that are never committed, which
            # is a handful of L2-hot loads, and it keeps exactly one batch in
            # flight at all times so vmcnt stays a compile-time immediate.
            tt = _ops.s_min(t, n_tiles - 1)
            g.stage(buf_a, a, a_coord(tt))
            g.stage(buf_b, b, b_coord(tt))

        def dot_tile(buf, tail=None):
            """One K-tile of math out of LDS buffer `buf`, with `tail` woven in.

            The whole slice's reads go in flight at once and the waits step
            down, so chunk s's WMMAs issue on top of chunks s+1.. still moving.
            No tile is duplicated: the overlapping reads are reads this slice
            had to do anyway, just not fenced in front of it. Double-buffering
            the operand tiles instead was tried in the C++ and is slower -- the
            second set costs 48 VGPRs and takes occupancy from 9 to 7.
            """
            A_s = _ops.shared_at(As, buf)
            B_s = _ops.shared_at(Bs, buf)

            def load_A(k):
                _ops.load_shared(_ops.subtile(A_s, reg_m, dot_slice, warp_row, k),
                                 A_tile.type, wait=False, out=A_tile)

            def load_B(k, s):
                _ops.load_shared(
                    _ops.subtile(B_s, split_n, dot_slice, warp_col * n_split + s, k),
                    B_tile[s].type, wait=False, out=B_tile[s])

            def run_tail():
                if tail is not None:
                    tail()

            load_A(0)
            for s in range(n_split):
                load_B(0, s)
            if slices == 1:
                # Nothing left to issue for this tile, so there is no later
                # slot to hide the drain in.
                run_tail()

            for k in range(slices):
                for s in range(n_split):
                    _ops.lds_wait_for(A_tile, B_tile[s])
                    _ops.setprio(1)
                    _ops.mma_ABt(A_tile, B_tile[s], C_acc[s], out=C_acc[s])
                    _ops.setprio(0)
                    # B_tile[s] is dead the instant this chunk's WMMAs have
                    # issued, so the next slice's read of it goes in flight
                    # here, a full chunk of math early, at no register cost.
                    # A_tile dies only after the slice's last chunk -- and goes
                    # out *before* that chunk's B, so the next slice's first
                    # chunk is gated on A rather than on a read behind it.
                    if k + 1 < slices:
                        if s == n_split - 1:
                            load_A(k + 1)
                        load_B(k + 1, s)
                        # The tile's last read is now in flight: hand the LDS
                        # pipe to the store and let the final slice's math
                        # cover it. Every wait after this point is downstream
                        # of two ds_writes and has to say so -- which is the
                        # one number in this kernel nobody has to write down.
                        if k + 2 == slices and s == n_split - 1:
                            run_tail()

        # Prologue: tile 0 resident, tile 1 in flight.
        g.load(_ops.shared_at(As, 0), a, a_coord(0))
        g.load(_ops.shared_at(Bs, 0), b, b_coord(0))
        stage(1)
        _ops.barrier()

        for t in _ops.range(n_tiles - 1):
            tic = t % 2
            toc = 1 - tic

            def store_tile():
                # Land the batch issued an iteration ago, drain it into the
                # buffer nobody is reading, and refill the registers at once so
                # they are never idle. GPREFETCH is 1, so vmcnt keeps nothing.
                _ops.vm_wait(0)
                g.commit(_ops.shared_at(As, toc), buf_a)
                g.commit(_ops.shared_at(Bs, toc), buf_b)
                stage(t + 2)

            if write_pos == 0:
                store_tile()
                dot_tile(tic)
            elif write_pos == 1:
                dot_tile(tic)
                store_tile()
            else:
                dot_tile(tic, store_tile)

            # The writes are still moving: dot_tile's last chunk stopped at the
            # two of them rather than at zero, on purpose, and s_barrier does
            # not order memory. Nearly free here -- a whole K-slice of WMMAs
            # has run since they issued -- and the pass refuses the barrier
            # without it, so it cannot be dropped by accident.
            _ops.barrier(drain=True)

        # Epilogue: the last tile is already resident and there is nothing left
        # to stage, so no tail -- and therefore no write for the waits to
        # carry. Counting writes that were never issued would be the dangerous
        # direction, and here that is structural rather than an argument the
        # caller has to remember to pass.
        dot_tile((n_tiles - 1) % 2)

        # Columns are in units of the chunk width; rows in elements, because
        # m_base need not be a multiple of BLOCK_M.
        m_row = m_base + warp_row * reg_m
        col_tile = (col * warp_cols + warp_col) * n_split
        for s in range(n_split):
            _ops.store(c, C_acc[s],
                       _ops.elem_coord(0, 0, m_row, (col_tile + s) * split_n))

    body.__name__ = name
    body.__annotations__ = {"a": GL[bf16], "b": GL[bf16], "c": GL[bf16]}

    def _grid(p):
        return (_ops.cdiv(p.a.rows, block_m) * (p.b.rows // block_n), 1, 1)

    if write_pos not in (0, 1, 2):
        raise ValueError(f"write_pos={write_pos} must be 0, 1 or 2")

    return _kernel(body, arch="gfx1100", warps=warps, grid=_grid, name=name,
                   max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: The shape this file is named for. More tilings go here as they are measured;
#: no single one wins across shapes, which is what Phase 5's autotune is for.
KERNELS = {
    "gemm_bf16_128x128": gemm_kernel("gemm_bf16_128x128"),
}

#: What the default tiling requires of a shape. K and N have to tile exactly --
#: the staging path has no bounds check and predicating it would make vmcnt a
#: runtime value. M does not, because the last block backs up instead.
BLOCK_M, BLOCK_N, K_STEP = 128, 128, 64


def matmul(a, b, out=None):
    """A @ B.T for bf16, fp32 accumulate.

    `b` is (N, K), not (K, N). That is not an inconvenience to be hidden: it
    keeps both shared loads contiguous and makes `mma_ABt` the primitive, which
    is what puts both operand tiles in row layout and therefore on the
    `ds_read_b128` fast path. Transposing here instead would copy, and the copy
    is a bigger cost than the GEMM saves.
    """
    import torch  # noqa: PLC0415 -- hk traces and compiles without torch

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"matmul: want 2D, got {a.dim()}D and {b.dim()}D")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(f"matmul: A is (M={m}, K={k}) and B is (N={n}, K={kb}); "
                         f"B is passed pre-transposed, so its second axis is K")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"matmul: bf16 only for now, got {a.dtype} and {b.dtype}")
    if k % K_STEP or n % BLOCK_N:
        raise ValueError(
            f"matmul: K={k} must be a multiple of {K_STEP} and N={n} a multiple "
            f"of {BLOCK_N}. Only M may be ragged -- the last block backs up to "
            f"end at M and recomputes the overlap, which it can do because the "
            f"rows it repeats are a function of A and the full K alone."
        )
    if m < BLOCK_M:
        raise ValueError(
            f"matmul: M={m} is below one block ({BLOCK_M}). Backing up needs a "
            f"whole block of rows to back into; a short-M shape wants the thin "
            f"tiling, which is not ported yet."
        )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16:
        raise ValueError(f"matmul: out is {tuple(out.shape)}/{out.dtype}, "
                         f"expected ({m}, {n})/bfloat16")
    KERNELS["gemm_bf16_128x128"](a, b, out)
    return out


__all__ = ["KERNELS", "gemm_kernel", "matmul", "BLOCK_M", "BLOCK_N", "K_STEP"]
