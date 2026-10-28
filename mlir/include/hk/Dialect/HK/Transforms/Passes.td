#ifndef HK_PASSES_TD
#define HK_PASSES_TD

include "mlir/Pass/PassBase.td"

def HKVerify : Pass<"hk-verify", "::mlir::func::FuncOp"> {
  let summary = "Structural rules: warp count, nothing workgroup-wide in a branch, fp8 only where there is an fp8 WMMA";
  let description = [{
    Errors, not warnings: a workgroup op skipped by some warps is a hang, an
    fp8 tile on a target without fp8 WMMA does not exist.
  }];
}

def HKFragments : Pass<"hk-fragments", "::mlir::func::FuncOp"> {
  let summary = "Name the matrix-core fragment in the type of every tile that needs one";
  let description = [{
    On a target whose bf16 MFMA operand is not 16x16 (gfx950: 16x32), set
    `frag` on every tile type an mma reads as A or B and on every type tied
    to one by a cast, a transpose or a broadcast. A type tied to two
    fragments -- an accumulator cast into an operand -- is an error. A no-op
    on RDNA and gfx942.
  }];
}

def HKLdsAlloc : Pass<"hk-lds-alloc", "::mlir::func::FuncOp"> {
  let summary = "Assign LDS offsets: declaration order, power-of-two alignment";
  let description = [{
    A bump allocator over the dynamic shared segment. Each allocation is
    aligned to its own size when that is a power of two, which is what makes
    the XOR buffer swap legal. Records hk.lds_bytes and hk.lds_static on the
    kernel.
  }];
}

def HKLdsPipeline : Pass<"hk-lds-pipeline", "::mlir::func::FuncOp"> {
  let summary = "Derive every lgkmcnt and vmcnt, prove every async tile is bound, check copies into LDS across warps";
  let description = [{
    Walks the LDS issue queue. A tile in flight may not be read; a tile
    retired by somebody else's wait may not be read; a barrier may not be
    reached with LDS writes, or reads of a buffer picked at run time, in
    flight. Async group loads are a second queue, on vmcnt: a buffer may not
    be read until this warp's copy is retired and a barrier (two inside an
    hk.pingpong) has passed, nor copied into until every warp's reads of it
    have landed as long ago. Loop bodies are walked twice and must agree.
    Sets `n` on every hk.lds_wait_for and hk.group_wait_for.
  }];
}

def HKMmaFence : Pass<"hk-mma-fence", "::mlir::func::FuncOp"> {
  let summary = "A scheduling fence between a matrix-core chain and the first other read of its accumulator (CDNA)";
  let description = [{
    Walks each block in order. An mma marks its accumulator (the tile it
    writes, through any rt_subtile alias); a full hk.sched_barrier clears the
    marks. The first op other than an mma accumulating into the same tile that
    touches a marked tile -- reads it, writes it, stores it -- gets an
    hk.sched_barrier in front of it, and the marks clear. Branches are walked
    with a copy of the marks and merge their union; loop bodies are walked
    twice, so what the end of one trip leaves marked reaches the next.

    Why (gfx950, ROCm 7.2.3 to 10.1): left free, LLVM scheduled the paged
    decode so that an MFMA result was read one branch after the MFMA, and
    its hazard recognizer, measuring the longer path into that block,
    padded 2 wait states where 8 are needed: some scores came out 2-4e-2
    wrong, differently from one build to the next. Fixed in LLVM by
    llvm/llvm-project#225989, not yet in a ROCm release
    (ROCm/llvm-project#5153). The fence does not fix that; it keeps a
    chain's results from being read where the scheduler would interleave
    them, which is where it struck. A fence at this boundary costs nothing
    measurable in the kernels that had one by hand. CDNA only; a kernel
    opts out with `hk.mma_fence = false`.
  }];
}

def HKCheckBudgets : Pass<"hk-check-budgets", "::mlir::func::FuncOp"> {
  let summary = "LDS limit, register estimate and slow-layout warnings";
  let description = [{
    The LDS limit is an error. The register estimate counts tile storage
    only and is reported, with the slow-layout findings, in hk.warnings.
  }];
}

#endif // HK_PASSES_TD
