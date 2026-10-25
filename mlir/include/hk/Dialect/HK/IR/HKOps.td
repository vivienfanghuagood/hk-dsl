#ifndef HK_OPS_TD
#define HK_OPS_TD

include "hk/Dialect/HK/IR/HKBase.td"
include "hk/Dialect/HK/IR/HKTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def HK_Axis : I32EnumAttr<"Axis", "grid axis", [
    I32EnumAttrCase<"x", 0>, I32EnumAttrCase<"y", 1>, I32EnumAttrCase<"z", 2>]> {
  let cppNamespace = "::mlir::hk";
}

def HK_Extent : I32EnumAttr<"Extent", "global tensor extent", [
    I32EnumAttrCase<"batch", 0>, I32EnumAttrCase<"depth", 1>,
    I32EnumAttrCase<"rows", 2>, I32EnumAttrCase<"cols", 3>]> {
  let cppNamespace = "::mlir::hk";
}

//===----------------------------------------------------------------------===//
// Indices and extents
//===----------------------------------------------------------------------===//

def HK_BlockIdxOp : HK_Op<"block_idx", [Pure]> {
  let summary = "blockIdx.{x,y,z}";
  let arguments = (ins HK_Axis:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$axis attr-dict";
}

def HK_WarpIdOp : HK_Op<"warp_id", [Pure]> {
  let summary = "this warp's index in the workgroup (kittens::warpid_uniform)";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def HK_LaneIdOp : HK_Op<"lane_id", [Pure]> {
  let summary = "kittens::laneid()";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def HK_CoordOp : HK_Op<"coord", [Pure]> {
  let summary = "a (b, d, r, c) index";
  let arguments = (ins I32:$b, I32:$d, I32:$r, I32:$c);
  let results = (outs HK_CoordType:$result);
  let assemblyFormat = "$b `,` $d `,` $r `,` $c attr-dict `:` qualified(type($result))";
}

def HK_ExtentOp : HK_Op<"extent", [Pure]> {
  let summary = "one extent of a global, at runtime (g.x.rows())";
  let arguments = (ins HK_GlobalType:$tensor, HK_Extent:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$tensor $axis attr-dict `:` qualified(type($tensor))";
}

def HK_AnyNeOp : HK_Op<"any_ne"> {
  let summary = "wave-level: did any entry of lhs differ from rhs";
  let arguments = (ins HK_RegVecType:$lhs, HK_RegVecType:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))";
  let hasVerifier = 1;
}

def HK_AnyAboveOp : HK_Op<"any_above"> {
  let summary = "wave-level: did any entry of lhs exceed rhs's by more than `threshold`";
  let description = [{
    1 in every lane if `lhs[i] - rhs[i] > threshold` for some entry in some
    lane, else 0: one vote across the wave, so a branch on it is uniform.
    The online softmax's lazy rescale asks it of the new running max against
    the one the probabilities are taken against.
  }];
  let arguments = (ins HK_RegVecType:$lhs, HK_RegVecType:$rhs,
                       F64Attr:$threshold);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $threshold attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Global memory
//===----------------------------------------------------------------------===//

def HK_LoadOp : HK_Op<"load"> {
  let summary = "global -> register tile or vector";
  let description = [{
    `bounded` predicates the tile on the global's rows: rows past the last
    read as zero and their memory is never touched.

    `period` (strips, unbounded) reads column `idx.c + c % period` into a
    strip's column c: a row of `period` values repeated along the strip, as
    a 16-byte RoPE wants the one token's angles under each of its heads.

    `aux` (register tiles): the loads' cache-policy bits -- 16 device scope,
    17 system scope (gfx950): a scoped tensor's data, which another XCD or
    GPU writes (hk.comm).
  }];
  let arguments = (ins Optional<HK_Elementwise>:$dst, HK_GlobalType:$src,
                       HK_CoordType:$idx, UnitAttr:$bounded,
                       OptionalAttr<I64Attr>:$period,
                       DefaultValuedAttr<I64Attr, "0">:$aux);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $src `[` $idx `]` (`into` $dst^)? attr-dict `:` qualified(type($src)) `,` qualified(type($idx))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreOp : HK_Op<"store"> {
  let summary = "register tile or vector -> global";
  let description = [{
    `aux` (register tiles): the stores' cache-policy bits -- 16 device
    scope, 17 system scope (gfx950): data another XCD or GPU reads after an
    hk.signal (hk.comm). A plain store stays in this XCD's L2.
  }];
  let arguments = (ins HK_GlobalType:$dst, HK_Elementwise:$value,
                       HK_CoordType:$idx, UnitAttr:$bounded,
                       DefaultValuedAttr<I64Attr, "0">:$aux);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_LoadRowsOp : HK_Op<"load_rows"> {
  let summary = "global -> register tile, rows at or past `rows` read as zero";
  let description = [{
    `load ... bounded` with a runtime limit as well as the global's: a tile
    whose rows end before the tensor's -- a query tile of one request running
    into the next request's tokens -- reads zeros past `rows` and never
    touches that memory. `rows` counts from the tile's first row.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GlobalType:$src,
                       HK_CoordType:$idx, I32:$rows);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `[` $idx `]` `rows` $rows (`into` $dst^)? attr-dict `:` qualified(type($src)) `,`
    qualified(type($idx)) (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreRowsOp : HK_Op<"store_rows"> {
  let summary = "register tile -> global, rows at or past `rows` not written";
  let arguments = (ins HK_GlobalType:$dst, HK_RegTileType:$value,
                       HK_CoordType:$idx, I32:$rows);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` `rows` $rows attr-dict `:` qualified(type($value)) `,`
    qualified(type($dst)) `,` qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_StoreScalarOp : HK_Op<"store_scalar"> {
  let summary = "one entry of a uniform register vector -> one global element";
  let arguments = (ins HK_GlobalType:$dst, HK_RegVecType:$value,
                       HK_CoordType:$idx);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_LoadScalarOp : HK_Op<"load_scalar"> {
  let summary = "one int32 element of a global -> a wave-uniform scalar";
  let description = [{
    `invariant`: the kernel never writes the element, so on a wave64 target
    it is read through the scalar cache (s_load_dword, on lgkmcnt) rather
    than by a vector load, whose wait is in vmcnt's queue behind every copy
    issued before it.
  }];
  let arguments = (ins HK_GlobalType:$src, HK_CoordType:$idx,
                       UnitAttr:$invariant);
  let results = (outs I32:$result);
  let assemblyFormat = "$src `[` $idx `]` attr-dict `:` qualified(type($src)) `,` qualified(type($idx))";
  let hasVerifier = 1;
}

def HK_WgArriveOp : HK_Op<"wg_arrive"> {
  let summary = "count this workgroup's arrival; how many arrived before it";
  let description = [{
    Device-wide by default: fences make every global store issued before it
    visible to the last arrival, wherever it runs. `xcd`: every workgroup
    that arrives at this counter runs on one XCD of a gfx950 (shares its
    L2) -- the stores are retired into the L2 and the last arrival's L1 is
    invalidated, with no L2 writeback or invalidate, which a device-wide
    fence costs every workgroup on the XCD.
  }];
  let arguments = (ins HK_GlobalType:$counter, HK_CoordType:$idx,
                       I32:$expected, UnitAttr:$xcd);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $counter `[` $idx `]` `,` $expected attr-dict `:` qualified(type($counter)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_GatherRowsOp : HK_Op<"gather_rows"> {
  let summary = "global -> row-layout register tile, row r from row idx[at + r] / div";
  let description = [{
    Tile row r is row `idx[at + r] / div` of `src` (rows flattened over batch
    and depth), from column `col`. An index outside the tensor, or a position
    at or past `count`, reads zeros. `idx` is an int32 global; `div` is how
    many index values name one row -- a MoE's top-k, whose sorted list counts
    (token, choice) pairs.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GlobalType:$src,
                       HK_GlobalType:$idx, I32:$at, I32:$count, I32:$div,
                       I32:$col);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `[` $idx `[` $at `,` $count `]` `,` $col `]` `div` $div (`into` $dst^)? attr-dict `:`
    qualified(type($src)) `,` qualified(type($idx)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_GatherPrepOp : HK_Op<"gather_prep"> {
  let summary = "stage a gathered copy's source rows in LDS (ids[at..at+rows) & 0xFFFFFF), barrier";
  let description = [{
    The workgroup reads `rows` sort ids from `ids` at `at` into a static LDS
    array, masks them to rows (AITER's token | choice << 24), and passes a
    barrier; hk.group_load with gather_staged = rows reads them from there.
  }];
  let arguments = (ins HK_GlobalType:$ids, I32:$at, I64Attr:$rows);
  let assemblyFormat = "$ids `[` $at `]` attr-dict `:` qualified(type($ids))";
}

def HK_ScatterRowsOp : HK_Op<"scatter_rows"> {
  let summary = "row-layout register tile -> global, row r to row idx[at + r]";
  let description = [{
    `sort_topk` > 0: idx holds AITER's MoE sort ids (token | choice << 24) and
    dst is [sort_topk * M, ...], choice-major: row r goes to row choice * M +
    token, and nowhere for a token >= M (the sort's padding).
    `add`: the rows are added into dst atomically instead of stored (fp32 or
    bf16 dst; bf16 adds pairs with global_atomic_pk_add_bf16 on gfx950); with
    sort_topk, dst is [M, ...] and row r goes to its token -- the top-k sum
    done by the scatter, in no fixed order.
  }];
  let arguments = (ins HK_GlobalType:$dst, HK_RegTileType:$value,
                       HK_GlobalType:$idx, I32:$at, I32:$count, I32:$col,
                       DefaultValuedAttr<I64Attr, "0">:$sort_topk,
                       DefaultValuedAttr<BoolAttr, "false">:$add);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `[` $at `,` $count `]` `,` $col `]` attr-dict `:`
    qualified(type($value)) `,` qualified(type($dst)) `,` qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_ScatterAddLdsOp : HK_Op<"scatter_add_lds"> {
  let summary = "a warp's LDS tile added into global rows by index, atomically, row-contiguous";
  let description = [{
    The atomic twin of scatter_rows for an epilogue: src (bf16, a warp's own LDS tile, written
    by that warp) row r is added into dst row idx[at + r] from column `col` with gfx950's
    packed bf16 atomic add. Lanes take consecutive column pairs of one row, so one instruction
    covers 64 / (cols / 2) rows of 4 * (cols / 2) contiguous bytes each -- a register tile's
    own layout would touch 16 rows of 16 bytes. sort_topk > 0: idx holds AITER's MoE sort ids
    (token | choice << 24) and the row is the token's (dst [M, ...]), padding (token >= M)
    nowhere. Positions at or past `count` add nothing. The sum is in no fixed order.
  }];
  let arguments = (ins HK_GlobalType:$dst, HK_SharedTileType:$src, HK_GlobalType:$idx,
                       I32:$at, I32:$count, I32:$col, DefaultValuedAttr<I64Attr, "0">:$sort_topk);
  let assemblyFormat = [{
    $src `,` $dst `[` $idx `[` $at `,` $count `]` `,` $col `]` attr-dict `:`
    qualified(type($src)) `,` qualified(type($dst)) `,` qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_AtomicAddOp : HK_Op<"atomic_add"> {
  let summary = "add an fp32 register tile into a float global, atomically";
  let arguments = (ins HK_GlobalType:$dst, HK_RegTileType:$value,
                       HK_CoordType:$idx);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($dst)) `,` qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_AtomicScalarOp : HK_Op<"atomic_scalar"> {
  let summary = "one int32 atomic (add or max) per warp; the value before it";
  let arguments = (ins HK_GlobalType:$dst, HK_CoordType:$idx, I32:$value,
                       StrAttr:$kind, DefaultValuedAttr<I64Attr, "0">:$aux);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $kind $dst `[` $idx `]` `,` $value attr-dict `:` qualified(type($dst)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_SignalOp : HK_Op<"signal"> {
  let summary = "publish: the workgroup's stores complete, then one int32 element := value";
  let description = [{
    Every wave waits for its stores' acknowledgements (vmcnt(0)), the
    workgroup passes a barrier, and thread 0 stores `value` with the
    cache-policy bits `aux` (16 device scope, 17 system scope, gfx950). The
    data it announces must have been stored with that scope -- every store
    to a scoped tensor is (hk.comm). Contains a barrier.
  }];
  let arguments = (ins HK_GlobalType:$dst, HK_CoordType:$idx, I32:$value,
                       I64Attr:$aux);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($dst)) `,` qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_WaitOp : HK_Op<"wait", [AttrSizedOperandSegments]> {
  let summary = "poll one int32 element until it compares with value; the value seen";
  let description = [{
    Every wave polls with the cache-policy bits `aux` (16 device, 17 system
    scope) and s_sleep between polls; `cmp` is "ge" (signed >=) or "eq".
    After `timeout` ticks of the 100 MHz s_memrealtime clock (0: never) it
    gives up: with `err` (an int32 element) lane 0 ORs 1 into it and the
    value seen is returned; without, the kernel traps.
  }];
  let arguments = (ins HK_GlobalType:$src, HK_CoordType:$idx, I32:$value,
                       Optional<HK_GlobalType>:$err, Optional<HK_CoordType>:$err_idx,
                       StrAttr:$cmp, I64Attr:$timeout, I64Attr:$aux);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $cmp $src `[` $idx `]` `,` $value (`err` $err^ `[` $err_idx `]`)? attr-dict `:`
    qualified(type($src)) `,` qualified(type($idx)) (`,` qualified(type($err))^ `,` qualified(type($err_idx)))?
  }];
  let hasVerifier = 1;
}

def HK_PhiloxOp : HK_Op<"philox_uniform"> {
  let summary = "uniform [0, 1) draws into an fp32 tile, as Triton's tl.rand";
  let description = [{
    Element (r, c) is Philox4x32-10 of counter `offset + r * row_stride + c`
    under key (seed, 0), converted as triton.language.rand converts it.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, I32:$seed, I32:$offset,
                       I32:$row_stride);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $seed `,` $offset `,` $row_stride (`into` $dst^)? attr-dict
    (`:` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_LoadW4Op : HK_Op<"load_w4"> {
  let summary = "4-bit weight codes -> a dequantized bf16 row-layout tile";
  let description = [{
    Tile element (r, c) is q * s - zs for W row n0 + r and column k0 + c: q is
    nibble (k % 8) of row n's word k / 8 (int32 [N, K/8], interleaved by
    16-row block and 32-column chunk), s and zs the group's scale and
    zero-times-scale `s[n][k / group]`, `zs[n][k / group]` (bf16
    [N, K/group]). hk's repack of an AWQ or GPTQ checkpoint.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GlobalType:$qw,
                       HK_GlobalType:$s, HK_GlobalType:$zs, I32:$n0, I32:$k0,
                       I64Attr:$group);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $qw `,` $s `,` $zs `[` $n0 `,` $k0 `]` `group` $group (`into` $dst^)? attr-dict `:`
    qualified(type($qw)) `,` qualified(type($s)) `,` qualified(type($zs))
    (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LDS
//===----------------------------------------------------------------------===//

def HK_AllocSharedOp : HK_Op<"alloc_shared"> {
  let summary = "reserve an LDS tile, or `count` of them contiguously";
  let description = [{
    `shape` (gfx950) names the st_shape the allocation is laid out in, where
    the translator would otherwise take the one its reads' fragment wants:
    a layout the reads also support whose rows a DMA copy reads in longer
    runs (st_16x64 for a 16x32 row operand, st_32x32 for a 32x16 stride-4
    col operand).
  }];
  let arguments = (ins OptionalAttr<I64Attr>:$offset,
                       OptionalAttr<I64Attr>:$nbytes,
                       OptionalAttr<StrAttr>:$shape);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def HK_SharedAtOp : HK_Op<"shared_at", [Pure]> {
  let summary = "one buffer of a multi-buffer LDS allocation";
  let arguments = (ins HK_SharedTileType:$buf, I32:$index);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = "$buf `[` $index `]` attr-dict `:` qualified(type($buf)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_SubtileOp : HK_Op<"subtile", [Pure]> {
  let summary = "a window of an LDS tile, indexed in units of itself";
  let arguments = (ins HK_SharedTileType:$src, I32:$row, I32:$col);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = [{
    $src `[` $row `,` $col `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def HK_LoadSharedOp : HK_Op<"load_shared"> {
  let summary = "LDS -> register tile; `wait = false` leaves the reads in flight";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_SharedTileType:$src,
                       DefaultValuedAttr<BoolAttr, "true">:$wait,
                       I64Attr:$lds_loads);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreSharedOp : HK_Op<"store_shared"> {
  let summary = "register tile -> LDS";
  let arguments = (ins HK_SharedTileType:$dst, HK_RegTileType:$value);
  let assemblyFormat = "$value `,` $dst attr-dict `:` qualified(type($value)) `,` qualified(type($dst))";
  let hasVerifier = 1;
}

def HK_GranulesOp : HK_Op<"granules"> {
  let summary = "factor one LDS tile's addressing into swizzle_bytes/16 registers";
  let arguments = (ins HK_SharedTileType:$src);
  let results = (outs HK_GranulesType:$result);
  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_LoadFragOp : HK_Op<"load_frag"> {
  let summary = "one 16x16 operand fragment out of a granule set; never waits";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GranulesType:$granules,
                       I64Attr:$row, I64Attr:$col, I64Attr:$lds_loads,
                       DefaultValuedAttr<BoolAttr, "false">:$wait);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $granules (`into` $dst^)? attr-dict `:` qualified(type($granules))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreFragOp : HK_Op<"store_frag"> {
  let summary = "one 16x16 fragment into a granule set's tile";
  let arguments = (ins HK_GranulesType:$granules, HK_RegTileType:$value,
                       Optional<I32>:$row_index, OptionalAttr<I64Attr>:$row,
                       I64Attr:$col);
  let assemblyFormat = [{
    $value `,` $granules (`,` $row_index^)? attr-dict `:` qualified(type($value)) `,`
    qualified(type($granules))
  }];
  let hasVerifier = 1;
}

def HK_LdsWaitForOp : HK_Op<"lds_wait_for"> {
  let summary = "s_waitcnt lgkmcnt(n), then re-bind each retired tile";
  let description = [{
    `n` is derived by the hk-lds-pipeline pass from the issue queue, never
    written by hand.
  }];
  let arguments = (ins Variadic<HK_RegTileType>:$tiles,
                       OptionalAttr<I64Attr>:$n);
  let assemblyFormat = "$tiles attr-dict `:` qualified(type($tiles))";
  let hasVerifier = 1;
}

def HK_AllocSharedVecOp : HK_Op<"alloc_shared_vec"> {
  let summary = "reserve `count` LDS vectors, contiguous";
  let arguments = (ins OptionalAttr<I64Attr>:$offset,
                       OptionalAttr<I64Attr>:$nbytes);
  let results = (outs HK_SharedVecType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def HK_StoreSharedVecOp : HK_Op<"store_shared_vec"> {
  let summary = "register vector -> one entry of an LDS vector stack";
  let arguments = (ins HK_SharedVecType:$dst, HK_RegVecType:$value,
                       I32:$index);
  let assemblyFormat = [{
    $value `,` $dst `[` $index `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst))
  }];
}

def HK_LoadSharedVecOp : HK_Op<"load_shared_vec"> {
  let summary = "one entry of an LDS vector stack -> a register vector";
  let arguments = (ins HK_SharedVecType:$src, I32:$index);
  let results = (outs HK_RegVecType:$result);
  let assemblyFormat = "$src `[` $index `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// Staging: global -> VGPRs -> LDS
//===----------------------------------------------------------------------===//

def HK_StageBufferOp : HK_Op<"stage_buffer"> {
  let summary = "declare the register buffer for a group copy into an LDS tile";
  let arguments = (ins HK_SharedTileType:$dst);
  let results = (outs HK_StageType:$result);
  let assemblyFormat = "$dst attr-dict `:` qualified(type($dst)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_StageLoadOp : HK_Op<"stage_load"> {
  let summary = "issue the global loads for one tile into a stage buffer";
  let arguments = (ins HK_StageType:$buf, HK_GlobalType:$src,
                       HK_CoordType:$idx, I64Attr:$slot);
  let assemblyFormat = [{
    $src `[` $idx `]` `into` $buf attr-dict `:` qualified(type($src)) `,` qualified(type($idx))
    `into` qualified(type($buf))
  }];
  let hasVerifier = 1;
}

def HK_StageCommitOp : HK_Op<"stage_commit"> {
  let summary = "write a staged buffer to LDS";
  let arguments = (ins HK_SharedTileType:$dst, HK_StageType:$buf,
                       I64Attr:$slot, DefaultValuedAttr<BoolAttr, "false">:$wait);
  let assemblyFormat = "$buf `into` $dst attr-dict `:` qualified(type($buf)) `into` qualified(type($dst))";
  let hasVerifier = 1;
}

def HK_VmWaitOp : HK_Op<"vm_wait"> {
  let summary = "s_waitcnt vmcnt(keep)";
  let arguments = (ins I64Attr:$keep);
  let assemblyFormat = "$keep attr-dict";
}

def HK_GroupLoadOp : HK_Op<"group_load", [AttrSizedOperandSegments]> {
  let summary = "global -> LDS by the whole workgroup, issued and waited unless `async`";
  let description = [{
    `async` (gfx950): the copy is the DMA, buffer_load ... lds, issued and
    left in flight. It counts on vmcnt; an hk.group_wait_for that names the
    tile retires this warp's share, with the count derived by
    hk-lds-pipeline, and a barrier after that makes the whole tile visible.
    The tile is a whole allocation or one buffer of a stack picked by a
    constant, so the pass can tell which tile a read or a later copy touches.
    `rows` (async only) ends the copy that many rows from the tile's first:
    rows past it read zero, as past the tensor's last row.
    `nontemporal` (async only): the copy reads past the caches' keeping
    (sc0 nt), for data read once.
    `gather_ids` / `gather_at` (async only): the tile's rows come by index --
    row r is the source's row `gather_ids[gather_at + r] & 0xFFFFFF` (AITER's
    MoE sort ids, token | choice << 24; its padding names a row past the
    tensor's end and reads zero); `idx` gives the column, its row is ignored.
    `gather_staged` N > 0: the rows were staged in LDS by hk.gather_prep (N
    rows): read from there, on lgkmcnt -- not from the index list, on vmcnt.
  }];
  let arguments = (ins HK_SharedTileType:$dst, HK_GlobalType:$src,
                       HK_CoordType:$idx, Optional<I32>:$rows,
                       Optional<HK_GlobalType>:$gather_ids, Optional<I32>:$gather_at,
                       I64Attr:$threads, UnitAttr:$async,
                       UnitAttr:$nontemporal, DefaultValuedAttr<I64Attr, "0">:$gather_staged);
  let assemblyFormat = [{
    $src `[` $idx `]` (`rows` $rows^)? (`gather` $gather_ids^ `[` $gather_at `]` `:` qualified(type($gather_ids)))?
    `into` $dst attr-dict `:` qualified(type($src)) `,`
    qualified(type($idx)) `into` qualified(type($dst))
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// buffer_load ... lds every thread issues for the tile, 16 bytes a
    /// lane: the fewest any warp does when the tile is not a whole number of
    /// them per thread (the first warps then issue one more), 0 when it is
    /// not a whole number per warp.
    int64_t dmaCalls();
  }];
}

def HK_GroupWaitForOp : HK_Op<"group_wait_for"> {
  let summary = "s_waitcnt vmcnt(n): this warp's async copies into the named LDS tiles have landed";
  let description = [{
    `n` is derived by hk-lds-pipeline from the order the copies were issued,
    never written by hand. It retires this warp's share only: the other
    warps' shares are visible after a barrier (two inside an hk.pingpong,
    where half the workgroup runs a barrier behind the other), and the pass
    refuses a read of the tile before that. `exact`: n also counts the
    register loads the pass tracks (E8M0 scales, FP4 fragments) issued after
    the copy, so they stay in flight -- vmcnt retires LDS-DMA copies and
    register loads in issue order on gfx950 (probes/vmcnt_order.hip); without
    it they are retired with the copy (a stricter wait).
  }];
  let arguments = (ins Variadic<HK_SharedTileType>:$tiles,
                       OptionalAttr<I64Attr>:$n, UnitAttr:$exact);
  let assemblyFormat = "$tiles attr-dict `:` qualified(type($tiles))";
  let hasVerifier = 1;
}

def HK_PingPongBeginOp : HK_Op<"pingpong_begin"> {
  let summary = "s_barrier in the warps where `cond` holds: they run a barrier behind the rest";
  let description = [{
    The half of the workgroup where `cond` holds waits at one extra barrier,
    so from here every s_barrier pairs its program point in one half with
    the next one in the other: while one half is between two barriers doing
    math, the other is between the same two doing memory. `cond` is wave
    uniform. hk.pingpong_end, later in the same block, gives the other half
    its extra barrier and the halves meet again.
  }];
  let arguments = (ins I1:$cond);
  let assemblyFormat = "$cond attr-dict";
  let hasVerifier = 1;
}

def HK_PingPongEndOp : HK_Op<"pingpong_end"> {
  let summary = "s_barrier in the warps where `cond` does not hold: the halves meet again";
  let arguments = (ins I1:$cond);
  let assemblyFormat = "$cond attr-dict";
  let hasVerifier = 1;
}

def HK_GroupStoreOp : HK_Op<"group_store"> {
  let summary = "LDS -> global by the whole workgroup";
  let arguments = (ins HK_GlobalType:$dst, HK_SharedTileType:$src,
                       HK_CoordType:$idx, I64Attr:$threads);
  let assemblyFormat = [{
    $src `into` $dst `[` $idx `]` attr-dict `:` qualified(type($src)) `into` qualified(type($dst))
    `,` qualified(type($idx))
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization and scheduling
//===----------------------------------------------------------------------===//

def HK_BarrierOp : HK_Op<"barrier"> {
  let summary = "s_barrier; `drain` puts an lgkmcnt(0) in front of it";
  let arguments = (ins UnitAttr:$drain);
  let assemblyFormat = "attr-dict";
}

def HK_SchedBarrierOp : HK_Op<"sched_barrier"> {
  let summary = "__builtin_amdgcn_sched_barrier(mask): only `mask` crosses this line";
  let description = [{
    `mask` names the instruction classes the scheduler may still move across
    (the sched_group_barrier classes below); 0, the default, is none.
  }];
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$mask);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def HK_SchedGroupBarrierOp : HK_Op<"sched_group_barrier"> {
  let summary = "__builtin_amdgcn_sched_group_barrier(mask, size, syncid)";
  let description = [{
    Asks the scheduler for a group of `size` instructions of the classes in
    `mask`, placed in program order after the groups before it that share
    `syncid`. A run of these is an interleave: "one VMEM read, then two
    WMMAs, then ..." The classes: 0x1 all ALU, 0x2 VALU, 0x4 SALU, 0x8
    MFMA/WMMA, 0x10 all VMEM, 0x20 VMEM read, 0x40 VMEM write, 0x80 all DS,
    0x100 DS read, 0x200 DS write, 0x400 transcendental. It orders only
    what the scheduler can see: an inline-asm instruction is in no class.
  }];
  let arguments = (ins I64Attr:$mask, I64Attr:$size, I64Attr:$syncid);
  let assemblyFormat = "$mask `,` $size `,` $syncid attr-dict";
  let hasVerifier = 1;
}

def HK_PinOp : HK_Op<"pin"> {
  let summary = "keep register values here: their definitions stay above, their uses below";
  let description = [{
    No instruction. Each value is bound with an empty volatile asm that
    LLVM takes to read and write its registers, so nothing that computes
    the value moves below this point and nothing that reads it moves above
    it. sched_barrier fences only the scheduler; LLVM's sinking still moves
    a computation down to its first use, across s_barrier and into another
    phase of a schedule -- attention's P conversions, sunk from their math
    phase into the next memory phase. A value whose LDS read is still in
    flight is refused, as any other read of it.
  }];
  let arguments = (ins Variadic<HK_Elementwise>:$values);
  let assemblyFormat = "$values attr-dict `:` qualified(type($values))";
}

def HK_SetPrioOp : HK_Op<"setprio"> {
  let summary = "s_setprio";
  let arguments = (ins I64Attr:$level);
  let assemblyFormat = "$level attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tile creation
//===----------------------------------------------------------------------===//

class HK_CreateOp<string mnemonic, dag attrs = (ins)>
    : HK_Op<mnemonic> {
  let arguments = !con((ins Optional<HK_Elementwise>:$dst), attrs);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    (`into` $dst^)? attr-dict (`:` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_ZeroOp : HK_CreateOp<"zero">;
def HK_FullOp : HK_CreateOp<"full", (ins F64Attr:$value)>;
def HK_NegInftyOp : HK_CreateOp<"neg_infty">;

//===----------------------------------------------------------------------===//
// Elementwise maps
//===----------------------------------------------------------------------===//

class HK_BinaryMapOp<string mnemonic>
    : HK_Op<mnemonic, [AttrSizedOperandSegments]> {
  let summary = "elementwise " # mnemonic # " on a tile or a vector";
  let description = [{
    The right operand is a tile or vector of the same type, a runtime i32
    scalar (cast to the element type), or absent with a literal `scalar`.
  }];
  let arguments = (ins Optional<HK_Elementwise>:$dst, HK_Elementwise:$lhs,
                       Optional<AnyTypeOf<[HK_RegTileType, HK_RegVecType, I32]>>:$rhs,
                       OptionalAttr<F64Attr>:$scalar);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $lhs (`,` $rhs^)? (`into` $dst^)? attr-dict `:` qualified(type($lhs))
    (`,` qualified(type($rhs))^)? (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_AddOp : HK_BinaryMapOp<"add">;
def HK_SubOp : HK_BinaryMapOp<"sub">;
def HK_MulOp : HK_BinaryMapOp<"mul">;
def HK_DivOp : HK_BinaryMapOp<"div">;
def HK_MaxOp : HK_BinaryMapOp<"max">;
def HK_MinOp : HK_BinaryMapOp<"min">;

class HK_UnaryMapOp<string mnemonic, dag attrs = (ins)> : HK_Op<mnemonic> {
  let summary = "elementwise " # mnemonic;
  let arguments = !con((ins Optional<HK_Elementwise>:$dst, HK_Elementwise:$src),
                       attrs);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_ExpOp : HK_UnaryMapOp<"exp">;
def HK_Exp2Op : HK_UnaryMapOp<"exp2">;
def HK_LogOp : HK_UnaryMapOp<"log">;
def HK_Log2Op : HK_UnaryMapOp<"log2">;
def HK_AbsOp : HK_UnaryMapOp<"abs">;
def HK_ReluOp : HK_UnaryMapOp<"relu">;
def HK_CopyOp : HK_UnaryMapOp<"copy">;
def HK_GeluOp : HK_UnaryMapOp<"gelu">;
def HK_DgeluOp : HK_UnaryMapOp<"dgelu">;
def HK_SqrtOp : HK_UnaryMapOp<"sqrt">;
def HK_RsqrtOp : HK_UnaryMapOp<"rsqrt">;
def HK_SiluOp : HK_UnaryMapOp<"silu">;
def HK_StepOp : HK_UnaryMapOp<"step">;
def HK_SwapHalvesOp : HK_UnaryMapOp<"swap_halves"> {
  let description = [{
    A strip's rows with their two halves exchanged: element (r, c) of the
    result is element (r, (c + cols/2) mod cols) of the operand -- RoPE's
    rotate_half without its sign. Strips only, and a row of one chunk or of
    an even number of them: a register swap between chunks, or one lane
    exchange across the wave's halves.
  }];
}
def HK_RotateHalfOp : HK_UnaryMapOp<"rotate_half", (ins I64Attr:$span)> {
  let description = [{
    NeoX's rotate_half on each `span`-column segment of a strip's rows: for
    c in the segment's first half the result is -x[c + span/2], in its second
    x[c - span/2]. Strips only. The halves are 8, 16 or 32 lanes apart -- a
    DPP row_ror:8, a v_permlane16_swap or a v_permlane32_swap on gfx950 --
    or whole chunks apart, a register swap; the sign is one XOR a value.
  }];
}

//===----------------------------------------------------------------------===//
// Reductions and broadcasts
//===----------------------------------------------------------------------===//

class HK_ReduceOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": reduce a tile to one value per row or column";
  let description = [{
    `row_*` reduces along each row and gives one value per row -- a column
    vector. `accumulate` folds into `dst` instead of overwriting it.
  }];
  let arguments = (ins Optional<HK_RegVecType>:$dst, HK_RegTileType:$src,
                       UnitAttr:$accumulate);
  let results = (outs Optional<HK_RegVecType>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_RowMaxOp : HK_ReduceOp<"row_max">;
def HK_RowMinOp : HK_ReduceOp<"row_min">;
def HK_RowSumOp : HK_ReduceOp<"row_sum">;
def HK_RowProdOp : HK_ReduceOp<"row_prod">;
def HK_ColMaxOp : HK_ReduceOp<"col_max">;
def HK_ColMinOp : HK_ReduceOp<"col_min">;
def HK_ColSumOp : HK_ReduceOp<"col_sum">;
def HK_ColProdOp : HK_ReduceOp<"col_prod">;

class HK_BroadcastOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": apply vec[i] to every element of row/col i";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$src,
                       HK_RegVecType:$vec);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `,` $vec (`into` $dst^)? attr-dict `:` qualified(type($src)) `,` qualified(type($vec))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_AddRowOp : HK_BroadcastOp<"add_row">;
def HK_SubRowOp : HK_BroadcastOp<"sub_row">;
def HK_MulRowOp : HK_BroadcastOp<"mul_row">;
def HK_DivRowOp : HK_BroadcastOp<"div_row">;
def HK_AddColOp : HK_BroadcastOp<"add_col">;
def HK_SubColOp : HK_BroadcastOp<"sub_col">;
def HK_MulColOp : HK_BroadcastOp<"mul_col">;
def HK_DivColOp : HK_BroadcastOp<"div_col">;

class HK_BroadcastFillOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": a tile whose every row/col is the vector";
  let arguments = (ins HK_RegVecType:$vec);
  let results = (outs HK_RegTileType:$result);
  let assemblyFormat = "$vec attr-dict `:` qualified(type($vec)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_BroadcastRowOp : HK_BroadcastFillOp<"broadcast_row">;
def HK_BroadcastColOp : HK_BroadcastFillOp<"broadcast_col">;

def HK_RetypeVecOp : HK_Op<"retype_vec"> {
  let summary = "re-attach a register vector to a tile of another shape";
  let arguments = (ins HK_RegVecType:$src);
  let results = (outs HK_RegVecType:$result);
  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Masks
//===----------------------------------------------------------------------===//

class HK_FillOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": overwrite the masked part of a tile with `value`";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$src,
                       I32:$idx, F64Attr:$value);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `,` $idx (`into` $dst^)? attr-dict `:` qualified(type($src))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_LeftFillOp : HK_FillOp<"left_fill">;
def HK_RightFillOp : HK_FillOp<"right_fill">;
def HK_UpperFillOp : HK_FillOp<"upper_fill">;
def HK_LowerFillOp : HK_FillOp<"lower_fill">;
def HK_TriuOp : HK_FillOp<"triu">;
def HK_TrilOp : HK_FillOp<"tril">;

def HK_MakeCausalOp : HK_UnaryMapOp<"make_causal", (ins F64Attr:$value)>;
def HK_MakeCausalTOp : HK_UnaryMapOp<"make_causal_t", (ins F64Attr:$value)>;

//===----------------------------------------------------------------------===//
// Conversions and WMMA
//===----------------------------------------------------------------------===//

def HK_CastOp : HK_UnaryMapOp<"cast">;
def HK_BitcastOp : HK_UnaryMapOp<"bitcast">;
def HK_TransposeOp : HK_UnaryMapOp<"transpose">;
def HK_TransposeSepOp : HK_UnaryMapOp<"transpose_sep">;

def HK_RegSubtileOp : HK_Op<"rt_subtile"> {
  let summary = "rows [index*R, (index+1)*R) of a register tile, as a tile of R rows: an alias";
  let description = [{
    include/cdna4's subtile_inplace on a register tile: the same registers,
    so an op that writes into the subtile writes the tile. What attention
    needs it for is one 64-row score tile -- one column reduction over it,
    one conversion -- whose two 32-row halves take their exp2 in different
    phases of the schedule. `index` is a constant: a runtime one would index
    the register file. hk-lds-pipeline takes a read of the subtile as a read
    of the tile, and refuses an LDS load into a subtile.
  }];
  let arguments = (ins HK_RegTileType:$src, I64Attr:$index);
  let results = (outs HK_RegTileType:$result);
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def HK_AsOperandOp : HK_Op<"as_operand"> {
  let summary = "an MFMA accumulator's registers read as an operand fragment, no data moved";
  let description = [{
    A 16-bit col-layout tile in an accumulator fragment, read as the
    stride-4 operand fragment whose lanes hold the same rows: 32x32 to
    16x32 stride 4, 16x16 to 32x16 stride 4. The reduction index then runs
    down the rows in the accumulator's order rather than the operand's, and
    an MMA pairs it with an operand laid out with the same permutation (the
    verifier holds A's and B's strides equal), so the product is the same.
    This is how P, the softmax of an MFMA result, feeds P·V on gfx950
    without a cross-lane reshuffle: the scores are computed transposed,
    Sᵀ = K·Qᵀ, and Oᵀ += Vᵀ·Pᵀ reads Pᵀ in place.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$src);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

class HK_MmaOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": acc + A x B on the matrix cores (WMMA on RDNA, MFMA on CDNA)";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$a,
                       HK_RegTileType:$b, HK_RegTileType:$c);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c (`into` $dst^)? attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `,`
    qualified(type($c)) (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_LoadMxScalesOp : HK_Op<"load_mx_scales"> {
  let summary = "E8M0 scales [rows, K/32] (uint8) -> the lanes of an mma operand that use them";
  let description = [{
    Rows row0.. and 32-column blocks kb0.. of a scale tensor of one byte a
    row and block (uint8 [R, K/32], row-major, the OCP MX layout), into the
    per-lane registers `mma_ABt_mx` reads: base tile i's row (lane % 16) and
    block (lane / 16) of each 128-wide K step. An i32 source is the packed
    layout (include/hk/cdna.cuh): one dword a lane per 64 rows and K step.
    `layout` 1 and 2 read AITER's shuffled MoE weight scales (uint8, [E *
    group_rows, K/32]; gfx950 `shuffle_scale(is_guinterleave=True)`): 1 for a
    gate|up stage-1 weight (the tile's 16-row base tiles alternate gate and
    up block row0's), 2 for a plain one. row0 is the tile's first row in
    [E * group_rows].
  }];
  let arguments = (ins Optional<HK_MxScalesType>:$dst, HK_GlobalType:$src, I32:$row0,
                       I32:$kb0, DefaultValuedAttr<I64Attr, "0">:$layout,
                       DefaultValuedAttr<I64Attr, "0">:$group_rows);
  let results = (outs Optional<HK_MxScalesType>:$result);
  let assemblyFormat = [{
    $src `[` $row0 `,` $kb0 `]` (`into` $dst^)? attr-dict `:` qualified(type($src))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_MxScalesCopyOp : HK_Op<"mx_scales_copy"> {
  let summary = "dst = src, a register copy of E8M0 scales (a one-stage loop's rotation)";
  let arguments = (ins HK_MxScalesType:$dst, HK_MxScalesType:$src);
  let assemblyFormat = "$src `into` $dst attr-dict `:` qualified(type($src)) `,` qualified(type($dst))";
  let hasVerifier = 1;
}

def HK_MmaABtMxOp : HK_Op<"mma_ABt_mx"> {
  let summary = "acc + (sa . A) x (sb . B)^T on gfx950's scaled f8f6f4 MFMA";
  let description = [{
    A and B are fp8 e4m3 row-layout tiles (16x128 fragments), C the fp32
    accumulator; sa and sb their E8M0 block scales (`load_mx_scales`):
    element (r, k) of A counts as A[r][k] * 2^(sa[r][k / 32] - 127).
    `a_fmt` / `b_fmt` 4: that operand is FP4 e2m1 (`load_fp4_frags`: the first
    four dwords of each fragment; its lane group g holds K [32g, 32g+32) of a
    128-wide step, where an FP8 operand's holds [16g, 16g+16) and [64+16g,
    64+16g+16) -- HipKittens' rt_16x128 order).
    `sa_tile` / `sa_step` (and sb's): sa may hold more than A's scales -- A's
    base tile i and K step s use sa's base tile i + sa_tile and step s +
    sa_step (one load of a 32-row block's two-step dwords feeding two tiles
    and two steps).
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$a,
                       HK_RegTileType:$b, HK_RegTileType:$c,
                       HK_MxScalesType:$sa, HK_MxScalesType:$sb,
                       DefaultValuedAttr<I64Attr, "0">:$a_fmt,
                       DefaultValuedAttr<I64Attr, "0">:$b_fmt,
                       DefaultValuedAttr<I64Attr, "0">:$sa_tile,
                       DefaultValuedAttr<I64Attr, "0">:$sa_step,
                       DefaultValuedAttr<I64Attr, "0">:$sb_tile,
                       DefaultValuedAttr<I64Attr, "0">:$sb_step);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c `scales` $sa `,` $sb (`into` $dst^)? attr-dict `:` qualified(type($a)) `,`
    qualified(type($b)) `,` qualified(type($c)) `,` qualified(type($sa)) `,` qualified(type($sb))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_LoadFp4FragsOp : HK_Op<"load_fp4_frags"> {
  let summary = "AITER-shuffled MXFP4 weight fragments -> an mma_ABt_mx B operand (b_fmt 4)";
  let description = [{
    AITER's gfx950 MoE weight layout (`shuffle_weight(is_guinterleave=True)`):
    per expert, 1 KB chunks each one 16x128 FP4 operand in MFMA lane order
    (lane l = 16 * kgroup + row, 16 bytes). Fragment (i, s) of the fp8-typed
    container tile gets, in its first four dwords, the chunk of 16-row block
    i and 128-wide K step kc0 + s; with gate_up, base tiles alternate the
    gate and up halves of block (row0 % group_rows) / 16 + i / 2. src is
    uint8 [E * group_rows, K/2]; row0 the tile's first row in it.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GlobalType:$src, I32:$row0,
                       I32:$kc0, I64Attr:$gate_up, I64Attr:$group_rows);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `[` $row0 `,` $kc0 `]` `gate_up` $gate_up `group_rows` $group_rows (`into` $dst^)?
    attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_MxQuantStoreOp : HK_Op<"mx_quant_store"> {
  let summary = "an LDS tile's rows to FP8 e4m3 + E8M0 scales (AITER's MX rule), stored";
  let description = [{
    Every thread of the workgroup takes (row, 32-column group) pairs of src (bf16 or fp32,
    plain LDS reads -- write it, then barrier, before this) and computes the group's E8M0
    exponent exactly as AITER's per_1x32 FP8 quant does: e = ceil(log2(amax / 448)) from
    amax's bits. It writes the 32 FP8 values (x * 2^-e, round to nearest even) to
    q[row0 + row, col0 + 32 g ..] and the byte e + 127 to s: a uint8 s gets AITER's 32-row
    two-step layout (hk.mx.aiter_scales of q's [R, K/32], K/32 padded to 8) -- the layout
    stage-2 / mma_ABt_mx(layout="aiter") reads; an int32 s the packed layout load_mx_scales
    reads (hk.mx.pack_scales). Rows past q's are skipped (with an int32 s their scale bytes get
    the unit scale, 127); AITER's pad scale bytes are not written.
  }];
  let arguments = (ins HK_SharedTileType:$src, HK_GlobalType:$q, HK_GlobalType:$s,
                       I32:$row0, I32:$col0);
  let assemblyFormat = [{
    $src `,` $q `,` $s `[` $row0 `,` $col0 `]` attr-dict `:` qualified(type($src)) `,`
    qualified(type($q)) `,` qualified(type($s))
  }];
  let hasVerifier = 1;
}

def HK_LoadFp4FragsLdsOp : HK_Op<"load_fp4_frags_lds"> {
  let summary = "FP4 operand fragments from an LDS tile of AITER-shuffled weight bytes";
  let description = [{
    The LDS side of load_fp4_frags: src is an fp8-typed tile of raw weight bytes, a row a
    (expert, 16-row block, gate|up) and 1 KB a 128-wide K step (the [E * N/16 * 2, K/2 * 16]
    view of AITER's layout, copied in by a group load). Fragment (i, s) of the container gets row
    row0 + i, 1 KB chunk chunk0 + s, 16 bytes a lane at lane * 16 -- with gate|up rows
    interleaved, base tiles alternate gate and up as load_fp4_frags(gate_up) does. Plain loads:
    the compiler waits for them.
  }];
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_SharedTileType:$src, I32:$row0,
                       I32:$chunk0);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `[` $row0 `,` $chunk0 `]` (`into` $dst^)? attr-dict `:` qualified(type($src))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_MmaABOp : HK_MmaOp<"mma_AB">;
def HK_MmaABtOp : HK_MmaOp<"mma_ABt">;
def HK_MmaAtBOp : HK_MmaOp<"mma_AtB">;
def HK_MmaAtBtOp : HK_MmaOp<"mma_AtBt">;

#endif // HK_OPS_TD
