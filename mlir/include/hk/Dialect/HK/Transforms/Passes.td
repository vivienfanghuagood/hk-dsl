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

def HKCheckBudgets : Pass<"hk-check-budgets", "::mlir::func::FuncOp"> {
  let summary = "LDS limit, register estimate and slow-layout warnings";
  let description = [{
    The LDS limit is an error. The register estimate counts tile storage
    only and is reported, with the slow-layout findings, in hk.warnings.
  }];
}

#endif // HK_PASSES_TD
