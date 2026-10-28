# Design: converging the DSL surface

Status: **approved** (2026-10-28; the decisions are in §9). Nothing on this page is built yet.
The work is timeboxed to one week, until 2026-11-04 00:00Z; §8 says which steps fit and which
come later.

#115 (`dsv41_index`) and #121 (`lane_strip`, `lane_coords`, `run_xor`) merged after the review.
Their call sites, counted under "PRs" in §2, are now in the repository; #120 is still open.

hk-dsl is the DSL and the kernels written in it. The language has grown one hand-written C++
helper per hard kernel, each exposed as a DSL op:
- `include/hk/cdna.cuh` went from nothing to 4,071 lines in 13 days.
- The ops it exports include `mx_quant_store`, `copy_mx_scales`, the scatter forms, the strip and
  element ops of the decode megakernel, and `map_cpp`.

This page does four things:
- takes stock of every kernel-specific op and helper (§2);
- proposes the few general abstractions they should fold into (§3);
- gives an ordered, bit-identical migration (§5-§7);
- states the rule that stops the growth (§4, and [CONTRIBUTING.md](../../CONTRIBUTING.md)).

- [0. Summary](#0-summary)
- [1. Scope and method](#1-scope-and-method)
- [2. Inventory](#2-inventory)
- [3. The target API surface](#3-the-target-api-surface)
- [4. The surface registry and the rule for new kernels](#4-the-surface-registry-and-the-rule-for-new-kernels)
- [5. Migration path](#5-migration-path)
- [6. Impact on shipped kernels](#6-impact-on-shipped-kernels)
- [7. Verification: bit-identical and not slower](#7-verification-bit-identical-and-not-slower)
- [8. Effort, risk and the timebox](#8-effort-risk-and-the-timebox)
- [9. Open questions for review, and the decisions](#9-open-questions-for-review-and-the-decisions)
- [Appendix: cdna.cuh by section](#appendix-cdnacuh-by-section)

## 0. Summary

**Inventory: 47 items. 17 MERGE, 28 KEEP, 2 DELETE.**
- Four of the KEEP items are PR #121's additions (proposed): `lane_strip`, `lane_coords`,
  `run_xor`, and group reductions over any xor offsets up to 32.
- The 28 KEEPs are not all escape hatches. Each is registered as one of these kinds:
  - **15 primitives**: general; any kernel may use them.
  - **3 formats**: a data layout of hk's or of a third-party library.
  - **4 escape hatches**: allowed only inside a written boundary.
  - **1 experiment**: off everywhere.
  - **5 internal C++**: never a DSL name.

**Six general abstractions absorb the 17 merges:**

| # | abstraction | absorbs |
|---|---|---|
| G1 | **Elementwise integer and bit expressions on strips**: operators on i32 strips, `hk.where`, shifts, bit ops, comparisons, `hk.math.fma`, as one typed `hk.map` region that lowers like `map_cpp` does | 93% of `map_cpp`'s 746 calls (§2.6); `mx_quant_store` as a composition; the E8M0 repack |
| G2 | **Register views** both ways: `Tile.lane_strip(run)` (#121) and its inverse `Tile.as_fragment(ty)` / `Tile.as_scales()` | `fp4_operand`, `fp8_operand`, `mx_scales_loose`; `mfma_strip`'s operands |
| G3 | **Lane coordinates**: `hk.lane_coords(tile_or_type, run)` (#121) | `iota` |
| G4 | **Row index sources**: `hk.RowIndex(ids, at, count, div=, sort_topk=)` and `g.stage(ix, rows)` | `div=`, `sort_topk=`, `staged=`, `gather_prep(raw=)`, `gather=(ids, at)` |
| G5 | **One MMA entry**: `hk.mma(variant, a, b, acc)` with scaled operands `a.scaled(sa, fmt=)`; MX scales are ordinary register values | `mma_ABt_mx`, `copy_mx_scales`, `load_fp4_frags_lds` |
| G6 | **Lane exchanges as one family**: `lane_gather`, `swap_halves(span, negate_first=)`, `run_xor`; on RDNA3, `as_operand(k_order=)` | `rotate_half`; attn_k3's rewrite of its own generated C++ |

**Two deletions:**
- `Global.scatter_add_lds`: no shipped user; `scatter(add=True)` covers it.
- `Tile.make_causal`: no user; `tril_` / `triu_` cover it.

**The rule** ([CONTRIBUTING.md](../../CONTRIBUTING.md)):
- A new kernel adds no kernel-specific C++ helper and no one-off DSL op.
- If one is truly needed, it goes through review with a written reason why the existing
  abstractions cannot express it.
- It enters the surface registry (§4) as an escape hatch with a usage boundary.
- A registry test (S0) makes the rule mechanical.

**Verification.** Every migrated kernel must stay bit-identical and must not get slower. The
check depends on what a step changes (§7):
- **V0**: the generated C++ is byte-identical, CPU only. Most steps are front-end only and stop
  here.
- **V1**: the code object's ISA is identical, function by function.
- **V2**: when the ISA changes, all of:
  - bitwise outputs against the pre-migration build;
  - a same-process interleaved A/B on the scoreboard harness;
  - the fence suite;
  - the spill and ScratchSize gate.

**In the week, after review:**
- S0, the registry and its lint;
- S1, the deletions;
- S2-S5, the front-end merges: MX, row index, rope, views;
- S12, the split of `cdna.cuh` by kind.

All of them are V0 or V1. The merges that change code a kernel runs come later, behind V2:
- G1's `hk.map`;
- `mx_quant_store` as a composition;
- the sync group copy;
- `mfma_strip`;
- attn_k3's pair exchange.

**Where this disagrees with the assessment.** The assessment is the 2026-10-28 review of what
hk-dsl buys over HipKittens C++, Triton and FlyDSL (`<workspace>/dsl-assessment.md`); this
page's "§4.x" references are to it. Its §4.3 differs from this design on three items:
- **`map_cpp` is not "2 calls" to delete.** In this repository it is 2 calls (`paged_cdna`), but
  there are many more elsewhere:
  - open PRs: 185 calls (#115: 173, #120: 12);
  - the out-of-tree DeepSeek-V4.1 decode stages written in hk: 559 calls.

  80% of those 746 calls are arithmetic, compares, selects and bit operations with no builtin,
  mostly i32 index arithmetic.
  With the fp32 builtins G1 also names (fma, min / max, abs, bitcast, exp2, rcp, rsq), G1 covers
  93%. 17 calls (2%) name an intrinsic hk has no op for, and 38 (5%) pass an expression built
  elsewhere. So `map_cpp` is kept, narrowed to what G1 cannot write.
- **`hk.arch.agpr`** is kept as an experiment, with a sunset condition, rather than deleted now.
- **`scatter(add=True)`** has no shipped user, but it is the general atomic scatter-add. It is kept
  as a primitive. Its bespoke LDS variant is deleted.

## 1. Scope and method

**What is inventoried.** Every DSL op, method or keyword, and every C++ helper in `include/hk/`,
that matches at least one of these:
- it exists for one kernel family, or for a few;
- its semantics encode one kernel's or one library's data layout;
- it rewrites or injects generated C++.

Each item is classified into exactly one of three classes:
- **MERGE** into a general abstraction. The item names the target and how today's call sites map
  onto it. The dedicated dialect op and C++ helper go away. A Python spelling may remain as sugar
  if it is a composition of general ops.
- **KEEP** as its own op. The item gives the reason and a usage boundary, and the op is registered
  as a *primitive*, *format*, *escape hatch*, *experiment* or *internal* (§4).
- **DELETE**: unused, or covered by a general abstraction (the coverage is shown).

**Sources.** Each call site is counted where it is:
- shipped ops: `python/hk/ops`, `python/hk/templates`, `python/hk/mx.py`;
- examples and `bench/`;
- tests and tools;
- open PRs #115 (`dsv41_index`), #118 (LDS-pointer stores), #120 (`sparse_mla`) and #121 (lane
  strips);
- the out-of-tree DeepSeek-V4.1 decode stages written in hk: the seam / attention front, the MoE,
  and the attention back.

Out-of-tree code is a real user, and its counts are given. It is not in this repository, so it
is not a shipped kernel. It is migrated through deprecation shims, not edits.

**The surface today, at main `af17846`:**
- `hk` exports 74 names; `docs/api/hk.md` documents 203 functions and methods;
- the dialect has 137 ops;
- `include/hk/cdna.cuh` has 4,071 lines (sections in the [appendix](#appendix-cdnacuh-by-section));
- `include/hk/cdna_agpr.cuh` has 183 lines.

**Out of scope** (sibling designs, named so nothing is lost):
- **Scheduling knobs** (`setprio`, `sched_barrier`, `sched_group_barrier`, `pin`, `mma_fence`).
  The assessment's recommendation is to wrap them in schedule templates, not delete them. They
  are knobs, not kernel-specific ops.
- **Per-kernel codegen attributes as kernel options**: AGPR allocation, waves per EU, max VGPRs
  (assessment §4.2 item 1). This would retire `Kernel.build_flags` (D7).
- **The backend decision** (assessment §4.2 item 5).
- **`hk.ops.hipkittens`**: kept as the baseline, not grown.
- **The HipKittens fork's own `include/rdna3` and `include/rdna4` additions.** They are reached
  only through DSL ops inventoried here (`granules`, `load_w4`, `transpose_sep`, the staging
  ops).

## 2. Inventory

Columns:
- **Users** counts call sites. Ops: shipped `hk.ops` / templates / examples. PRs: the open PRs.
  Out of tree: the decode stages.
- **Tests** names the test files that exercise the item directly. *Via ops* means only through a
  shipped op's tests.
- **Public?**: *API* means documented in `docs/api`. *Exported* means in `hk.__all__` but listed
  only under "Also exported". *C++* means no DSL name.
- **What general ops cannot say** is why the item exists today.

### 2.1 MX block-scaled operands (gfx950)

| # | item (definition) | public? | users | tests | what general ops cannot say today | class: target |
|---|---|---|---|---|---|---|
| A1 | `hk.load_mx_scales(src, rows, kblocks, row0, kb0, *, into, layout="ocp"\|"aiter"\|"aiter_gate_up", group_rows, loose)` (`values.py`, `hk.load_mx_scales` op, `kittens::load_mx_scales` / `load_mx_scales_aiter`, `hk_mxs<R, KB, F>`) | API | ops: moe_a8w4 45, mxfp8 6, gemm_cdna 2, mx.py 2; PRs: #115 2; out of tree: 18 | test_cdna_mx_ops, test_k2_ops, test_mx_fp4_aiter, ir/test_cdna_fp8 | the scaled MFMA reads E8M0 scales from lane-specific bytes (opsel). OCP, packed and AITER's shuffled layouts all land in those registers | **KEEP, format.** The scaled MFMA's operand. A new scale layout is a `layout=` entry, never a new op |
| A2 | `hk.copy_mx_scales(dst, src)` (`hk.mx_scales_copy` op) | API | ops: mxfp8 6 | via ops (test_mxfp8_gemm, test_vllm_mxfp8) | `MxScales` is not a `Tile`, so `Tile.copy(into=)` does not apply | **MERGE: G5.** `src.copy(into=dst)` on `MxScales`; the same op is emitted |
| A3 | `hk.mma_ABt_mx(a, b, acc, sa, sb, *, a_fmt, b_fmt, sa_at, sb_at)` (`hk.mma_ABt_mx` op, `kittens::mma_ABt_mx`) | API | ops: moe_a8w4 13, mxfp8 3, gemm_cdna 1; PRs: #115 3; out of tree: 14 | test_cdna_mx_ops, test_k2_ops, test_mx_fp4_aiter, test_select_ops, ir/test_cdna_fp8, ir/test_strip | `hk.mma` has no scales | **MERGE: G5.** `hk.mma("ABt", a.scaled(sa), b.scaled(sb, fmt="fp4", at=(t, s)), acc)`; the same op is emitted |
| A4 | `hk.load_fp4_frags(ty, src, row0, kc0, *, group_rows, gate_up)` (`kittens::load_fp4_frags`) | API | ops: moe_a8w4 8 | test_cdna_mx_ops, test_mx_fp4_aiter | AITER's `shuffle_weight` layout (1 KB chunks in lane order, gate/up interleave), with a 64-bit base per expert (a 4.5 GB weight passes a 32-bit buffer offset) | **KEEP, format.** AITER's pre-shuffled MXFP4 weights only |
| A5 | `hk.load_fp4_frags_lds(ty, src, row0, chunk0)` | API | ops: moe_a8w4 2 | test_cdna_mx_ops | the same bytes, from an LDS tile | **MERGE: A4.** One `load_fp4_frags` whose `src` may be a `SharedTile`; it dispatches to the existing op |
| A6 | `hk.mx_quant_store(src, q, s, row0, col0)` (`kittens::mx_quant_store`, 70 lines) | API | ops: moe_a8w4 1 (stage-1 epilogue, `mxq`), mxfp8 1 (`quant`) | test_cdna_mx_ops, test_mxfp8_quant, test_moe_a8w4, test_fenced | floor and ceil on exponent bits, byte scatter of scales into AITER's 32-row or the packed layout, packed fp8 conversion: no integer or bit ops on tiles | **MERGE: G1 composition**, after G1, behind V2. If it is slower (K1's +3-4% came from it), reclassify as KEEP-format with the measurement |
| A7 | `hk.fp4_operand(words)` (`hk.fp4_operand` op) | exported | PRs: #115 2; out of tree: 2 | test_select_ops, ir/test_strip | an i32 strip's dwords reinterpreted as an MFMA operand fragment | **MERGE: G2.** `words.as_fragment(hk.RT(fp8e4m3, 16, 128), fmt="fp4")` |
| A8 | `hk.fp8_operand(words)` | exported | out of tree: 5 | test_k2_ops, ir/test_strip | as A7, 8 dwords | **MERGE: G2.** `words.as_fragment(hk.RT(fp8e4m3, 16, 128))` |
| A9 | `hk.mx_scales_loose(bytes_)` | exported | PRs: #115 2; out of tree: 7 | test_select_ops, ir/test_strip | an i32 strip's byte as a loose scale register | **MERGE: G2.** `bytes_.as_scales()` |

### 2.2 Rows by index (MoE sort ids)

| # | item | public? | users | tests | what general ops cannot say today | class: target |
|---|---|---|---|---|---|---|
| B1 | `Global.gather(tile, ids, at, *, count, col, div)`: the `div` decode | API | ops: moe (RDNA, `div=top_k`) 1; gather itself: moe 3, moe_a8w4 2, example 08 | test_indexed, test_sglang_moe_topk, test_templates_rowwise | an index list whose values are not rows | **MERGE: G4.** `RowIndex(ids, at, count=, div=)`; gather keeps its name |
| B2 | `Global.scatter(..., sort_topk=)` (`ScatterRowsOp.sort_topk`, `store_scatter<TOPK>`) | API | ops: moe_a8w4 3 | test_scatter_add | AITER sort ids (token \| choice << 24) decoded choice-major | **MERGE: G4.** `RowIndex(..., sort_topk=k)` |
| B3 | `Global.scatter(..., staged=N)` | API | ops: moe_a8w4 1 | test_scatter_add | ids read from an LDS copy, so no vmcnt wait drains the prefetch (K1 c36) | **MERGE: G4.** A staged `RowIndex` returned by `g.stage` |
| B4 | `Group.gather_prep(ids, at, rows, *, raw)` (`hk.gather_prep` op, `kittens::gather_prep<N, RAW>`, function-static `__shared__` arrays) | API | ops: moe_a8w4 3 (1 raw) | test_cdna_fp8_lds, test_gather_load, test_scatter_add, ir/test_cdna_gemm | staging the ids in LDS. Nothing ties `raw=True` to a later `scatter(staged=)`: a scatter after a non-raw prep reads an LDS array no one wrote, silently | **MERGE: G4.** `g.stage(ix, rows, keep_choice=)` returns the staged index, so the pairing is by construction |
| B5 | `Group.load(..., gather=(ids, at), staged=N)` (`GroupLoadOp.gather_staged`, `group_load_async_gather<NW, N>`) | API | ops: moe_a8w4 2; PRs: #120 2 | test_cdna_fp8_lds, test_gather_load, ir/test_cdna_gemm | rows by index for the LDS DMA | **MERGE: G4.** `g.load(dst, src, wait=False, gather=ix)` |
| B6 | `Global.scatter(..., add=True)` (`store_scatter<.., ADD>`) | API | ops: none; tools: the MFMA_SCALE sink probe | test_scatter_add, codegen/test_mfma_scale_sink | atomic top-k reduce (nondeterministic order) | **KEEP, primitive.** The general atomic scatter-add. Boundary: not for routes that promise determinism. In G4 it is `scatter(t, ix, add=True)`. Kept only with a GPU test (§9, decision 7) |
| B7 | `Global.scatter_add_lds(src, ids, at, *, count, col, sort_topk)` (`kittens::scatter_add_lds`) | API | none (K1 c33 measured it and did not ship it) | test_scatter_add | packed bf16 atomics from an LDS tile, lanes on column pairs (310 vs 85 G adds/s) | **DELETE.** No user. Covered functionally by `SharedTile.load` + `scatter(add=True)`. Measured faster, but no kernel wants an atomic reduce (K1 chose the deterministic sum) |
| B8 | `detail::hk_pinned(t)` (C++) | C++ | `store_scatter<.., ADD>` only | codegen/test_mfma_scale_sink (`HK_UNPINNED`) | roc-7.2.3 sinks `V_MFMA_SCALE` into a divergent `if` and its operand copies run under partial EXEC | **KEEP, internal.** A compiler-bug guard. Boundary: only inside a primitive that consumes an MFMA result in a lane-divergent branch; each use has a sink test |

### 2.3 Strips, elements and lanes (gfx950)

| # | item | public? | users | tests | what general ops cannot say today | class: target |
|---|---|---|---|---|---|---|
| C1 | `hk.map_cpp(expr, *args, dtype, name)` (`hk.map_cpp` op, a C++ lambda compiled with contraction off) | API | ops: paged_cdna 2; PRs: #120 12, #115 173; out of tree: 559 (seam 184, MoE 149, attention back 226) | test_elems, test_k2_ops, test_select_ops, test_strip, ir/test_strip | integer arithmetic on strips is refused ("written in hk.map_cpp"); no bit ops, select or fma; no way to name an intrinsic | **KEEP, escape hatch, narrowed after G1.** Of all 746 calls (§2.6), 93% are expressions G1 writes. 17 call an intrinsic: `__builtin_amdgcn_cvt_scalef32_pk_*`, `popcount`, `ctz`, `readlane`, `s_memrealtime`. Boundary in §4 |
| C2 | `hk.iota(ty)` (`hk.iota` op, `kittens::iota`) | API | ops: paged_cdna 1; PRs: #120 2, #115 21; out of tree: 103 | test_elems, test_k2_ops, test_select_ops, test_strip, ir/test_comm, ir/test_strip | each element's column | **MERGE: G3.** `hk.lane_coords(ty)[1]`, the same formula (#121's `lane_coord<1>` on a strip is `iota`'s body). `hk.iota` stays as a one-line alias |
| C3 | `hk.mfma_strip("f32_16x16x4", a, b, acc)` | API | out of tree: 5 | test_strip, ir/test_strip | one raw `v_mfma_f32_16x16x4_f32` on strip registers, to match another kernel bit for bit | **MERGE: G2 + G5, later.** `hk.mma` gains the fp32 16x16x4 form; strips reach it through `as_fragment` / `lane_strip`. Until then, alias |
| C4 | `Tile.lane_gather(lanes)` (`ds_bpermute` an element) | API | PRs: #120 1, #115 5; out of tree: 13 | test_strip, ir/test_strip | arbitrary lane permutation | **KEEP, primitive** (G6) |
| C5 | `Tile.first_lane()` | API | PRs: #115 8; out of tree: 8 | test_select_ops, ir/test_strip | lane 0's element as a uniform scalar | **KEEP, primitive.** May grow a `lane=` argument to absorb `map_cpp`'s `readlane` calls |
| C6 | `Tile.group_sum` / `group_max` / `group_min(cols, *, lanes, xor)` (`hk.group_reduce`) | API | PRs: #120 1, #115 6; out of tree: 10 | test_strip, ir/test_strip | a reduction over a lane group, in a stated order, broadcast back | **KEEP, primitive.** Segmented reductions: an MX group's amax, a 64-dim block |
| C7 | element-indexed access: `Global.load_elems` / `store_elems(idx, *, runs, row)` (scoped variants), `SharedTile.load_elems` / `store_elems` / `atomic_add_elems` | API | ops: paged_cdna 2; PRs: #120 4, #115 76; out of tree: 261 | test_elems, test_k2_ops, test_select_ops, test_strip, ir/test_comm, ir/test_strip | element gather and scatter at flat indices, out of range masked | **KEEP, primitive.** The general per-element access (Triton's pointer-tensor loads). Nothing to merge |
| C8 | `Global.post_elems` / `take_elems(idx, tag, *, runs, bf16, timeout_s)` | API | PRs: #115 12; out of tree: 67 | test_elems, ir/test_comm, ir/test_strip | `hk.comm.post` / `take` per element: a tagged mailbox pair, one 8-byte scoped store | **KEEP, primitive** (the hk.comm family). The API reference moves them next to `hk.comm.post` / `take` |
| C9 | `Tile.swap_halves()` (`hk.swap_halves`) | API | ops: fused (rope) 2 | ir/test_strip; rope GPU tests (test_cdna_rows, test_templates_rowwise, test_fenced) | a row's halves exchanged (register swap or `v_permlane32_swap`) | **KEEP, primitive** (G6), and the target of C10 |
| C10 | `Tile.rotate_half(span)` (`hk.rotate_half`) | API | ops: fused (rope) 1 | as C9 | NeoX's rotate_half: the halves of each span exchanged (DPP `row_ror:8`, `permlane16` / `permlane32` swaps) and the first half negated | **MERGE: G6.** `swap_halves(span=, negate_first=True)`: one op, one C++ template. `rotate_half` stays as sugar |
| C11 | `Tile.lane_strip(run)` (#121, proposed) | API (#121) | PRs: #120 1 | #121: test_strip, ir/test_strip | an MMA result read as a strip where it sits | **KEEP, primitive** (G2). Its inverse is `as_fragment` |
| C12 | `hk.lane_coords(tile, run)` (#121, proposed) | API (#121) | PRs: #120 0: it writes the coordinates out by hand (`iota` + `map_cpp`) "so nothing but the values stays live" | #121: test_strip | each lane-strip element's row and column in the tile | **KEEP, primitive** (G3). To also take a type, so it absorbs `iota`. #120's choice says G1's fusion must let coordinates fold into the index expressions that use them (S7) |
| C13 | `Tile.run_xor(mask)` (#121, proposed) | API (#121) | PRs: #120 4 | #121: test_strip, ir/test_strip | element e of a run from element e ^ mask (a rope pair inside a run) | **KEEP, primitive** (G6) |
| C14 | group reductions over any distinct power-of-two xor offsets up to 32 (#121, proposed) | API (#121) | PRs: #120 1 | #121: test_strip, ir/test_strip | a 32-row column block of a 32x32 MFMA result is lane l's run plus lane l ^ 32's | **KEEP**: an extension of C6, no new op |

### 2.4 Attention, RDNA and the kernel object

| # | item | public? | users | tests | what general ops cannot say today | class: target |
|---|---|---|---|---|---|---|
| D1 | attn_k3's `PairPKernel`, a `Kernel` subclass whose `with_defines` splices `kda::pair_p` (20 lines of C++) into the generated source and regex-replaces `kittens::copy(P, S)` | none (inside a shipped op: the `sdpa_k3` route, default-on on gfx1100) | ops: attn_k3 | test_attn_k3 (GPU, bitwise) | RDNA3's paired-lane accumulator-to-operand exchange (Pᵀ) in the even/odd K order, one `permlanex16` a packed dword (assessment §4.1 item 4) | **MERGE: G6.** `S.as_operand(fp16, k_order="even_odd")` on gfx1100, an RDNA3 form of the CDNA `as_operand`. Then `PairPKernel` and its source rewrite are deleted |
| D2 | `SharedTile.granules()`, `Granules.load_frag` / `store_frag` (RDNA3) | API | ops: attn 2+2+1, attn_k3 2+2, paged_prefill 2+2+1 | ir/test_granules | an LDS tile's addressing factored into swizzle-bytes/16 registers, by hand, to save VGPRs | **KEEP, escape hatch.** Boundary: gfx11 kernels whose address registers are the measured limiter; not on CDNA. The 1-byte-tile extension (assessment §4.1 item 4) goes through review |
| D3 | `hk.arch.agpr(tile)` + `include/hk/cdna_agpr.cuh` (asm MFMAs with `"+a"`, hand-placed `s_nop`s) | API | ops: attn_cdna, behind `acc="agpr"` (off) | ir/test_agpr_codegen, gpu/test_agpr | an accumulator in AGPRs. LLVM chooses the register file | **KEEP, experiment.** Bitwise the VGPR form's, measured 0.93-0.95x. Boundary: never on by default. Sunset (§9, decision 6): deleted after 2026-10-31 12:00Z unless the task evaluating it has adopted it |
| D4 | `Tile.as_operand(dtype)` (CDNA; `kittens::as_operand`, two shapes) | API | ops: attn_cdna, paged_cdna, templates (elementwise, rowwise 2); examples 05; bench solutions | ir/test_cdna_attn; via ops | an MFMA accumulator's registers read as an operand fragment, no data moved | **KEEP, primitive** (several kernel families and the templates). The target of D1 |
| D5 | `Tile.make_causal(value, *, transposed)` (`hk.make_causal`, `hk.make_causal_t`; the fork's rdna3 `make_causal`) | API | none | codegen/test_transpose_causal | the causal mask of a diagonal block | **DELETE.** Covered: `make_causal(v)` is `tril_(0, v)` and `transposed=True` is `triu_(0, v)`. Every kernel already masks with `triu_` / `lower_fill_`. One release as a deprecated alias, with a test that the two agree |
| D6 | `Kernel.with_defines(src)` used as a source hook (D1 is the only override) | API | ops: attn_k3 (D1); tests: 3 codegen; tools: 1 probe | test_cdna_builds, test_lds_copy_gate, test_strip_cdna | any C++ at all | **KEEP, escape hatch, narrowed.** `#define NAME value` knobs only (`Kernel.defines`). Overriding `with_defines` or `source` in an `hk.ops` kernel is refused by a test (S0). Every past use was a missing op |
| D7 | `Kernel.build_flags` | API | tests only | ir/test_flat_args, codegen/test_cdna_fp32_lds, codegen/test_strip_cdna | hipcc flags per kernel | **KEEP, escape hatch.** Boundary: tests and probes. Retired by per-kernel codegen options (§1, out of scope) |

### 2.5 C++ only: library overrides, specializations, plumbing

| # | item (in `include/hk/cdna.cuh`) | users | tests | why it exists | class: target |
|---|---|---|---|---|---|
| E1 | `group_load_sync<NW>` (gfx950): include/cdna4's synchronous group copy with a bounded descriptor (PR #47's fault) | emitted for every synchronous `g.load` on gfx950. In hk.ops: `mxfp8_kernel` (`mxfp8.gemm`, not on the route) | ir/test_cdna | the library's copy read past a partial last tile | **MERGE** into the asynchronous DMA body (`group_load_async_rows<NW>` with `limit = ST::rows`) plus the wait HKToCpp already emits. One copy body instead of two. V2, later |
| E2 | `group_store_bounded<NW>` (and both gfx942 forms, which call the library's unbounded copies) | every `g.store` on CDNA | ir/test_cdna | the library's store wrote past a partial last tile | **KEEP, internal** library fix. Upstream to the HipKittens fork, then delete. gfx942 stays unbounded (no hardware to test on) |
| E3 | LDS-pointer overrides: fp32 `rt` ↔ `st` (main); fp32 `sv` ↔ `rv` and the 16-bit col `rt` → `st` store (#118, also in #120) | every such access on gfx950 | codegen/test_cdna_fp32_lds | the library reaches LDS through a generic pointer: FLAT ops, counted in vmcnt, so an LDS wait also waits for the DMA | **KEEP, internal** library fixes. Upstream to the fork |
| E4 | type-selected fast paths: the 32x32 fp32 → bf16 tile store (`permlane32_swap`, attention's O); the 16x16 fp32 → bf16 scatter (`permlane16_swap`, K1 c35); the 16-column `col_reduce` (permlane swaps, paged decode); the col-layout masks (asm `v_cmp` / `v_cndmask`, the sign-bit add); `neg_infty` by asm `v_mov` | attn_cdna, paged_cdna, moe_a8w4 | via ops; codegen/test_strip_cdna | measured: halved store counts, no LDS shuffle, no VCC wait states, no copies behind a branch | **KEEP, internal** specializations. Boundary: chosen by type alone, never by a flag, and bitwise equal to the generic path. S0 adds one test each: specialized against generic on random inputs |
| E5 | gap fills of the CDNA trees: `rsqrt`, `sqrt`, `silu`, `step`, `gelu` / `dgelu` (CDNA3), bf16 round-to-nearest-even copies, fp16 × fp16 → fp32 `mma_ABt`, `one()` | any kernel | via ops | the library lacks them, or truncates where hk rounds | **KEEP, internal.** Upstream the fixes |
| E6 | `load_scalar_invariant` (`View.load_scalar(invariant=True)`: an `s_load` through the scalar cache) | attn_cdna 13, attn_k3 6, fused 2, gemm_cdna 1, paged_cdna 4; #120 3 | ir/test_cdna_paged, ir/test_strip | a global scalar load waits on vmcnt with the DMA | **KEEP, primitive** |
| E7 | `wg_arrive_xcd` (`hk.arch.wg_arrive(..., scope=)`) | gemm_cdna (stream-K), decode_gemm | ir/test_wg_arrive, ir/test_cdna_gemm | an arrival without a device-wide fence | **KEEP, primitive** |
| E8 | `hk.load_w4(ty, qw, s, zs, n0, k0, *, group)` (`kittens::load_w4`; the fork's rdna3 `quant.cuh`) | w4a16 on gfx950 / gfx1100 / gfx1201 | test_w4a16 | hk's own int4 layout, dequantized into a register operand | **KEEP, format** |
| E9 | `hk.rand` (`philox_uniform`, Triton's Philox) | example 07; tests | test_indexed and others | draws equal to `tl.rand`'s | **KEEP, primitive** |

### 2.6 Totals, and how the counts were taken

| class | count | items |
|---|---|---|
| MERGE | 17 | A2, A3, A5, A6, A7, A8, A9, B1, B2, B3, B4, B5, C2, C3, C10, D1, E1 |
| KEEP, primitive | 15 | B6, C4, C5, C6, C7, C8, C9, C11, C12, C13, C14, D4, E6, E7, E9 |
| KEEP, format | 3 | A1, A4, E8 |
| KEEP, escape hatch | 4 | C1, D2, D6, D7 |
| KEEP, experiment | 1 | D3 |
| KEEP, internal | 5 | B8, E2, E3, E4, E5 |
| DELETE | 2 | B7, D5 |
| **total** | **47** | of which #121's proposed additions: C11-C14 |

**How the counts were taken.**
- Call sites were counted by pattern over the sources in §1.
- Out-of-tree counts are from the latest file of each stage: the seam / attention front, the MoE
  and the attention back.
- The `map_cpp` breakdown parses every call (Python's `ast`) in paged_cdna, #120, #115 and the
  three stages: 746 calls. By what the expression needs:

  | needs | calls | share |
  |---|---|---|
  | only arithmetic, compares and selects (no builtin; mostly i32 indices) | 486 | 65% |
  | bit ops, possibly with those | 114 | 15% |
  | an fp32 builtin or cast G1 names (`fmaxf`, `fmaf`, `fabsf`, `bit_cast`, `exp2f`, `rcp`, `rsq`, `(float)`) | 91 | 12% |
  | an intrinsic hk has no op for (`cvt_scalef32_pk_*`, `popcount`, `ctz`, `readlane`, `s_memrealtime`) | 17 | 2% |
  | an expression built elsewhere and passed in a variable | 38 | 5% |

## 3. The target API surface

Each abstraction below has:
- its signatures;
- its semantics and lowering;
- one current call site written in today's form and in the target form.

Names that change keep a deprecation shim (§5).

### G1. Elementwise integer and bit expressions on strips

```python
# operators on i32 strips (fp32 strips keep theirs); results are strips of the same shape and run
a + b, a - b, a * b, a // b, a % b        # i32: C semantics (truncation), as hk.Int's arith.divsi / remsi
a << n, a >> n, a & b, a | b, a ^ b, ~a   # i32; >> is arithmetic (signed)
a < b, a <= b, a == b, a != b, ...        # -> a Bool strip (a lane mask a element)
m1 & m2, m1 | m2, ~m                      # on Bool strips
hk.where(m, a, b)                         # select; a, b strips or scalars (Int, Python numbers)
hk.min(a, b), hk.max(a, b)                # exist for fp32 strips; i32 too
x.bitcast(i32), n.bitcast(fp32)           # exists for tiles; strips too
hk.math.fma(a, b, c)                      # one rounding
hk.math.rcp(x)                            # v_rcp_f32 (rsqrt exists)
hk.math.popcount(n), hk.math.ctz(n)       # the two bit intrinsics map_cpp is used for
x.to(fp8e4m3)                             # packed v_cvt_pk_fp8_f32 (exists in store_elems(runs=True))
```

**Semantics.**
- Every operand is a strip of one shape and run, or a scalar (`hk.Int` or a Python number),
  broadcast.
- Integer ops are 32-bit and signed.
- Float ops inside an expression are compiled with contraction off, as `map_cpp`'s are:
  - `a * b + c` is two roundings;
  - `hk.math.fma` is one.
- The existing float tile operators outside these expressions keep their current lowering. This
  design changes no float result.

**Lowering.**
- The operators build an `hk.map` op whose body is a region of `arith` / `math` ops on scalars:
  one element. This is `linalg.generic`'s shape, on strips.
- A fusion pattern merges a chain of single-use `hk.map`s into one region.
- HKToCpp prints a region exactly as it prints `map_cpp` today: one `strip_each` lambda, with
  contraction off.
- A migrated call site therefore compiles to the same C++ expression up to its text, and to the
  same ISA (V1).
- `map_cpp` becomes the same `hk.map` op with an opaque string body, so the escape hatch and the
  typed form share one emitter path.

**Today → target** (`paged_cdna`, the split merge's partial sums):

```python
col = hk.iota(T)
parts.store_elems(acc, hk.map_cpp(f"x1 * {D} + x0", col, hk.arch.warp_id()))
acc += parts.load_elems(hk.map_cpp(f"{w * D} + x0", col))
# target
col = hk.lane_coords(T)[1]
parts.store_elems(acc, hk.arch.warp_id() * D + col)
acc += parts.load_elems(col + w * D)
```

**Today → target** (#120, the scale index of one 32-d MX group):

```python
idx = hk.map_cpp(f"(x0 & 15) == 0 && ((x0 >> 4) & 63) < 32 ? ((x0 >> 4) & 31) * {D // 32} + "
                 f"{DW // 32} * x1 + (x0 >> 10) : -1", c, w, dtype=i32)
# target
lane = (c >> 4) & 63
idx = hk.where(((c & 15) == 0) & (lane < 32), (lane & 31) * (D // 32) + (DW // 32) * w + (c >> 10), -1)
```

**`mx_quant_store` as a composition** (A6). This is AITER's rule, e = ceil(log2(amax / 448)) from
amax's bits, on an fp32 strip of run 8, where a 32-column MX group is four lanes' runs:

```python
amax = hk.max(abs(x).group_max(32), 2.0 ** -126)
b = amax.bitcast(i32)
e = hk.min(hk.max((b >> 23) - 8 + hk.where((b & 0x7FFFFF) > 0x600000, 1, 0), 0), 254)  # e + 127
inv = hk.where(e < 254, ((254 - e) << 23).bitcast(fp32), 2.0 ** -127)
q = hk.min(hk.max(x * inv, -448.0), 448.0)
qo.store_elems(q, q_idx, runs=True)             # fp32 -> e4m3, four at a time (exists)
so.store_elems(e, s_idx)                        # one byte a group: s_idx is -1 but in its first lane
```

`q_idx` and `s_idx` are G1 expressions over `hk.lane_coords`. They give the AITER 32-row scale
layout or the packed one: `hk.mx.aiter_scales` / `pack_scales`' formulas, which `mx_quant_store`
spells in C++ today. #120 already quantizes its output this way, with `map_cpp` in place of G1.

**What A6 still needs.** Either a strip load from a 16-bit swizzled LDS tile (K1 keeps h in LDS)
or the register path #120 takes. That is open question 3.

**Beneficiary outside the kernels.** The E8M0 repack that stays in torch (23% of a short MXFP8
call) becomes writable as a kernel.

### G2. Register views

```python
s = t.lane_strip(run=8)                        # (#121) a register tile's registers as a 1-row strip, no data moved
f = s.as_fragment(ty, *, fmt="fp8" | "fp4")    # a strip's i32 elements as ty's per-lane dwords, in register order
sc = s.as_scales()                             # a strip's element (the E8M0 byte, bits 0..7) as a loose MxScales
```

**Semantics.** `as_fragment` is `lane_strip`'s inverse on 32-bit data. Lane l's elements, in the
strip's per-lane order (rows, chunks, runs), become lane l's registers of `ty`. Each of these
costs no instruction; they are reinterpretations.
- **fp8** (`ty` = `hk.RT(fp8e4m3, 16, 128)`): 8 dwords, a strip of 1 x 512 at run 4. The K order
  is the hardware's segmented one, which `hk.RT`'s 16x128 shape already is.
- **fp4**: 4 dwords; the upper half of the operand is left undefined.

**Lowering.**
- Front end first (S5): `as_fragment` and `as_scales` emit today's `hk.fp8_operand`,
  `hk.fp4_operand` and `hk.mx_scales_loose`, so in-tree kernels are V0.
- Later, one `hk.lane_view` op replaces the three. Its C++ is generic over #121's `hk_lane_each`.

```python
a = hk.fp8_operand(words);  sa = hk.mx_scales_loose(bytes_)                  # today
a = words.as_fragment(hk.RT(fp8e4m3, 16, 128));  sa = bytes_.as_scales()     # target
```

**`mfma_strip`** (C3) becomes `hk.mma` on views once `hk.mma` takes the fp32 16x16x4 form.

### G3. Lane coordinates

```python
rows, cols = hk.lane_coords(tile_or_type, run=8)    # i32 strips: each element's row and column
```

**Semantics.** The arguments differ:
- **a register tile**: as #121 defines it, the coordinates of `tile.lane_strip(run)`'s elements;
- **a strip type or strip**: its own coordinates. `cols` is today's `hk.iota(ty)`, by the same
  formula: `j * 64 * run + lane * run + e`.

It is the index arithmetic of every element op, layout-independent.

```python
c = hk.iota(ST16)            # today
c = hk.lane_coords(ST16)[1]  # target; hk.iota stays as this one line
```

### G4. Row index sources

```python
ix = hk.RowIndex(ids, at, *, count=None, div=1, sort_topk=0)   # ids an int32 Global
ix.offset(n)                                                   # the same list from at + n
sx = g.stage(ix, rows, *, keep_choice=False)                   # an LDS copy of [at, at + rows), then a barrier
t = src.gather(ty, ix, col=c)                                  # rows by index -> a register tile
dst.scatter(t, ix, col=c, add=False)                           # a register tile -> rows by index
g.load(dst_lds, src.at[0, k], wait=False, gather=ix)           # the LDS DMA, rows by index
```

**Semantics.** One object says which rows; each consumer applies the decode it documents today.
- **Plain ids**: the row is `ids[at + r]`.
- **`div=k`**: `ids[at + r] // k`.
- **`sort_topk=k`**, AITER's sort ids (token | choice << 24):
  - a gather reads the token's row;
  - a scatter writes row `choice * M + token`;
  - a scatter with `add=True` adds into the token's row.
- Positions at or past `count`, and padding, read zero and write nothing.

**Staging.** `g.stage` returns a staged index, and only a staged index can be used where the LDS
copy is meant:
- with `keep_choice=False` it holds the decoded token rows, which gathers read (today's
  `gather_prep`);
- with `keep_choice=True` it holds the raw ids, which scatters read (`gather_prep(raw=True)`).

A consumer refuses a staged index of the wrong form. Today that mismatch is a silent read of an
LDS array no one wrote.

**Lowering.** Unchanged: `GatherRowsOp` (`div`), `ScatterRowsOp` (`sort_topk`, `staged`),
`GroupLoadOp` (`gather_staged`), `GatherPrepOp` (`raw`). V0.

```python
# today (moe_a8w4)
g.gather_prep(sids, r0, BM)
g.load(Xs[slot], x.at[0, kc * KS], wait=False, gather=(sids, r0), staged=BM)
g.gather_prep(sids, r0, BM, raw=True)
out.scatter(o, sids, r0 + i * TILE, col=c0, sort_topk=topk, staged=BM)
# target
ix = hk.RowIndex(sids, r0, sort_topk=topk)
rows_in = g.stage(ix, BM)
g.load(Xs[slot], x.at[0, kc * KS], wait=False, gather=rows_in)
rows_out = g.stage(ix, BM, keep_choice=True)
out.scatter(o, rows_out.offset(i * TILE), col=c0)
```

### G5. One MMA entry; MX scales as register values

```python
acc = hk.mma("ABt", a.scaled(sa), b.scaled(sb, fmt="fp4", at=(tile, step)), acc)
src.copy(into=dst)                                   # MxScales, as Tile.copy(into=)
w = hk.load_fp4_frags(ty, src, row0, kc0, ...)       # src a Global or a SharedTile
```

**Semantics.**
- `Tile.scaled(scales, *, fmt="fp8", at=(0, 0))` makes a scaled operand. `hk.mma` takes one only
  as both operands of the `ABt` variant on gfx950. It emits today's `hk.mma_ABt_mx` with
  `a_fmt` / `b_fmt` / `sa_at` / `sb_at` taken from the operands.
- Unscaled calls are unchanged.

```python
acc = hk.mma_ABt_mx(a, w, acc, sa, sw, b_fmt="fp4", sb_at=(t, s))            # today
acc = hk.mma("ABt", a.scaled(sa), w.scaled(sw, fmt="fp4", at=(t, s)), acc)   # target
hk.copy_mx_scales(S[0][0][i], S[1][0][i])                                    # today
S[1][0][i].copy(into=S[0][0][i])                                             # target
```

### G6. Lane exchanges as one family; RDNA3's paired operand

```python
t.lane_gather(lanes)                              # any source lane (ds_bpermute)
t.swap_halves(span=None, *, negate_first=False)   # within each span-column segment, the halves exchanged
t.run_xor(mask)                                   # (#121) within a lane's run
S.as_operand(fp16, k_order="even_odd")            # gfx1100: an fp32 [32, 16] S^T pair as two P^T operands
```

**Semantics.**
- `swap_halves(span)` exchanges the halves of each `span`-column segment. With
  `negate_first=True` the first half is negated, which is NeoX's `rotate_half`.
- The lowering is today's, picked by the same rules:
  - halves whole chunks apart swap registers;
  - halves 8, 16 or 32 lanes apart use DPP `row_ror:8`, `v_permlane16_swap` or
    `v_permlane32_swap`.
- `rotate_half(span)` stays as `swap_halves(span, negate_first=True)`.

**RDNA3's paired operand.** `as_operand(k_order="even_odd")` is the RDNA3 counterpart of the CDNA
`as_operand`:
- tile 0 holds the operand whose K axis is the pair's even rows, tile 1 the odd rows;
- each packed dword is one conversion, one `permlanex16` and two selects;
- that is `kda::pair_p`'s body, moved into hk's headers unchanged.

attn_k3 then writes `P = S.as_operand(fp16, k_order="even_odd")`. Its `PairPKernel` subclass and
the regex over generated C++ are deleted.

## 4. The surface registry and the rule for new kernels

**The registry.**
- **The table.** Every public DSL name, every dialect op and every C++ helper a kernel can reach
  gets one entry: name, kind, boundary, and the tests that pin it. §2's table is the first
  version.
- **Where it lives.** S0 puts it in the code as `python/hk/dsl/surface.py`: `ENTRIES` (the
  inventoried names that are not plain primitives, with kind, item, boundary, target and step),
  `PRIMITIVES`, `OPS` (every dialect op) and `CPP` (every function in `include/hk/`). The API
  reference shows each non-primitive name's kind and boundary.
- **Kinds:**
  - **primitive**: general; any kernel.
  - **format**: one data layout of hk's or of a third-party library. A new layout of the same
    family is an argument value, not a new op.
  - **escape hatch**: only inside its boundary, and each call site in `hk.ops`, the templates
    and the examples is listed with a one-line reason (`scripts/escape_hatches_allow.txt`, as
    the public-hygiene allowlist is).
  - **experiment**: never default-on; it has a sunset condition.
  - **internal**: C++ only, never a DSL name. A library fix or a type-selected specialization,
    which must be bitwise the generic path's.
  - **deprecated**: a shim with the release it goes away in.

**What S0's tests check:**
- every name in `hk.__all__`, every documented method and every op in `HKOps.td` has a registry
  entry;
- every escape-hatch call site in `python/hk/ops`, the templates and the examples is in a file
  the allowlist lists, within the line's ceiling (below);
- no `hk.ops` kernel subclasses `Kernel` to override `with_defines` or `source`;
- each internal specialization has a test against the generic path.

**The allowlist and the registry are ratchets** (decided 10-29, after the first merge trains). An
allowlist line is (file, name, ceiling, reason). A use in a file with no line fails, and so do
more uses than the ceiling: bringing a hatch into a new file, or more of it into a listed one, is
the reviewed change -- the PR edits the line and the reviewer reads it. Fewer uses than the
ceiling (or none) pass with a hint to tighten the line. The registry checks work the same way: a
new name, dialect op, C++ helper or header that is not registered fails; a registered one that is
gone only hints. The first version demanded exact counts and full equality, and every PR that
touched an op file then conflicted with S0 in the merge train without anything to review (#137's
dsv41_index went 173 -> 190 `map_cpp` uses; #120 brought 12 in a new file). A rewriting `Kernel`
subclass is a line too, with ceiling 1 a class.

**Escape-hatch boundaries:**

| escape hatch | may be used | may not be used |
|---|---|---|
| `hk.map_cpp` (C1) | for an intrinsic or instruction hk has no op for (scaled fp8/fp4 conversions, `readlane`, `s_memrealtime`); to reproduce another kernel's exact instruction sequence | for index arithmetic, selects, bit ops or float expressions G1 can write (once G1 lands); in a template |
| `granules` / `load_frag` / `store_frag` (D2) | gfx11 attention-class kernels whose address VGPRs are the measured limiter | CDNA; any kernel that has not measured the limit |
| `Kernel.with_defines` (D6) | `#define NAME value` knobs, A/B variants | any other C++; overriding it in a subclass |
| `Kernel.build_flags` (D7) | tests and probes | `hk.ops` kernels (until codegen options exist) |
| `hk.arch.agpr` (D3, experiment) | behind an option that is off by default | a default-on route |

**The rule for new kernels.** It is in [CONTRIBUTING.md](../../CONTRIBUTING.md), with the reviewer
checklist.

## 5. Migration path

The steps are ordered. Each is one PR, small, and mergeable on its own. Each regenerates the API
docs (`scripts/gen_api_docs.py`) and states its verification tier (§7).

**The shim policy.** A public name that changes keeps its old spelling as a shim that emits the new
form:
- The shim warns (`DeprecationWarning`) from the release after the step.
- It is removed one minor release later.
- In-tree callers, examples and docs move in the step itself.
- Out-of-tree callers (the decode stages) move at their owners' pace within that window.

| step | change | shims | tier | depends on |
|---|---|---|---|---|
| **S0** | Guard rails only, no kernel change: the registry (`hk.dsl.surface`) from §2; the tests of §4; the escape-hatch allowlist; one bitwise test per E4 specialization; `tools/surface/snapshot.py`, which traces every shipped kernel at its route-table configurations and hashes the generated C++ (V0's tool); a PR-template checklist pointing at CONTRIBUTING.md | none | none (tests only) | this design |
| **S1** | DELETE B7 `scatter_add_lds` (DSL method, dialect op, C++, test) and D5 `make_causal` (alias to `tril_` / `triu_` for one release, with a test that they agree; the two dialect ops go) | D5's alias | V0 on every shipped kernel | S0 |
| **S2** | G5 front end: `Tile.scaled`, `hk.mma` with scaled operands, `MxScales.copy(into=)`, `load_fp4_frags` from a `SharedTile`. Same dialect ops | `mma_ABt_mx`, `copy_mx_scales`, `load_fp4_frags_lds` | V0: K1, K2, gemm_cdna's MX mode | S0 |
| **S3** | G4: `hk.RowIndex`, `g.stage`, the `gather=ix` form; the staged-form check. Same dialect ops and attributes | `div=`, `sort_topk=`, `staged=`, `gather=(ids, at)`, `gather_prep` | V0: K1, RDNA3 / RDNA4 MoE, example 08 / 09 | S0 |
| **S4** | G6, rope: `rotate_half` folds into `swap_halves(span, negate_first)`: one dialect op, one C++ template | `rotate_half` (sugar, kept) | V1: fused rope kernels at their plans | S0 |
| **S5** | G2 front end: `as_fragment`, `as_scales`, emitting today's three ops | `fp4_operand`, `fp8_operand`, `mx_scales_loose` | V0 (no in-tree kernel uses them; #115 if merged) | S0 |
| **S12** | Split `cdna.cuh` by kind, a pure move: `cdna/lib.cuh` (library fixes, gap fills), `cdna/primitives.cuh`, `cdna/formats.cuh` (MX, W4), `cdna/escape.cuh` (`map_cpp`, `mfma_strip`), `cdna/special.cuh` (E4); each definition headed by its registry name | none | V0 (same generated source, which includes one umbrella header) and V1 | S0 |
| S6 | G3: `lane_coords` takes a type; `iota` becomes its alias; the `hk.iota` op goes | `iota` (alias, kept) | V1: paged_cdna; #120 / #115 if merged | #121 |
| S7 | G1: integer and bit operators on strips, `hk.where`, `hk.math.fma` / `rcp` / `popcount` / `ctz`, the `hk.map` region op and its fusion pattern; `map_cpp` becomes `hk.map` with a string body; the escape-hatch allowlist starts. Migrate paged_cdna's 2 calls (and #120's 12 if merged) | `map_cpp` stays (escape hatch) | V1 (ISA identical expected); V2 where not | S0 |
| S8 | E1: `group_load_sync` → the DMA body + wait | none (C++) | V2: `mxfp8.gemm` (128x128) bitwise, A/B, fence | S12 |
| S9 | C3: `hk.mma` takes the fp32 16x16x4 form; `mfma_strip` becomes an alias over views; G2's single `hk.lane_view` op | `mfma_strip` | V1 on the strip tests; out-of-tree users bitwise | S5, S6 |
| S10 | A6: `mx_quant_store` as a G1 composition, in `hk.mx` (Python), for K1's epilogue and `mxfp8.quant` | `mx_quant_store` (alias to the composition) | V2: K1 and K2 quant bitwise and A/B; fence; spill. Not adopted if slower | S7 (and open question 3) |
| S11 | D1: RDNA3 `as_operand(k_order="even_odd")`; attn_k3 drops `PairPKernel` | none (internal to attn_k3) | V1 on gfx1100 (the same C++ body); V2 on a W7900D if the ISA differs | S0 |
| S13 | Upstream E2, E3 and E5 into the HipKittens fork; delete hk's copies when the submodule moves | none | V1 | S12 |

## 6. Impact on shipped kernels

Every row must come out **bit-identical and not slower**. §7 says how each tier proves it.

| kernel | inventory items it uses | steps that touch it | verification |
|---|---|---|---|
| **K1** `moe_a8w4`: stage 1 / stage 2, the sorted ring, the `mxq` epilogue (gfx950) | A1, A3, A4, A5, A6, B2-B5, E4 (16x16 scatter), E6 | S2, S3, S12: V0. S10: V2 | V0: the generated C++ at the scoreboard's M set (64 … 16384 plus the ragged M) and both stage forms. S10: outputs bitwise against the pre-migration build on real layer 20 / 30 weights, then the same-process K1 A/B (`k1_ab`), the fence suite, ScratchSize 0 |
| **K2** `mxfp8`: `gemm_4w`, `gemm_8w_tail`, `quant` (gfx950) | A1, A2, A3; A6 (`quant`) | S2, S12: V0. S10 (quant): V2. S8 touches only `mxfp8.gemm`'s 128x128 kernel, which is not on the route | V0 at the 18 DeepSeek-V4.1 (N, K) x M points of the K2 scoreboard. S10: `quant` bitwise with `hk.mx.quant` plus `pack_scales` (test_mxfp8_quant already asserts it), the K2 route's same-process A/B, the fence |
| **attention** `attn_cdna`: dense, causal, paged prefill (gfx950) | D3 (off), D4, E4 (32x32 store, masks, `neg_infty`), E6 | S12: V0 / V1 | V0; V1 with resources (VGPR / AGPR / occupancy) unchanged |
| **paged decode** `paged_cdna` (gfx950) | C1 (2), C2, C7, E4 (`col_reduce`), E6 | S6: V1. S7: V1. S12: V0 | V1: ISA hash of every plan the 120-point scoreboard uses. If an ISA differs: outputs bitwise and the `paged_sb` same-process A/B |
| **sdpa / attn / attn_k3** (gfx1100) | D1 (attn_k3), D2 | S11: V1 | `kda::pair_p`'s body moves unchanged, so the ISA should be identical (V1, gfx1100 codegen tier). If not: test_attn_k3 bitwise and a same-process A/B on a W7900D. attn and attn_k3's other ops are untouched |
| **int4** `w4a16` (gfx950, gfx1100, gfx1201) | E8 | S12: V0 (gfx950) | V0 |
| **convrot** (gfx1100) | none (`hk.ops.gemm` + `hk.ops.quant`) | none | not touched |
| **RDNA4** (gfx1201): GEMM, attention, MoE | B1 (MoE `div`), D2 (attention) | S3: V0 | V0 on gfx1201 and gfx1100 MoE configurations |
| **RDNA3 MoE** (`vllm_moe`, `sglang_moe`) | B1 | S3: V0 | V0 |
| **fused rope** (gfx950) | C9, C10 | S4: V1 | V1 at every `_rotary*` plan; the rope GPU tests |
| **sparse MLA** `sparse_mla` (#120, if merged) | C1 (12), C2, C4, C6, C7, C11, C13, C14, E6, B5 | S3: V0. S6, S7: V1 | V1. If not identical: test_sparse_mla bitwise and the route's same-process A/B |
| **DeepSeek-V4.1 index** `dsv41_index` (#115, if merged) | C1 (173), C2, C5, C6, C7, A1, A3, A7, A9 | S2, S5: V0. S6: V1. S7 migration later | as sparse MLA |

## 7. Verification: bit-identical and not slower

Each step names its tier. A step passes its tier on every kernel in its §6 rows before it is
marked READY, and on a head that contains the latest main.

The tiers are the repository's existing
[gates for a change to the generated code](../testing.md#the-gates-for-a-change-to-the-generated-code),
applied per step:
- V0 is Gate A;
- V1 is Gate B's compile-and-resources half, plus an ISA hash;
- V2 is the rest of Gate B, with a bitwise check against the old build added. A V2 step on a
  default-on route (S10: K1 and K2) also runs Gate C, the engine.

**V0: the generated source is byte-identical (CPU only).**
- **Two worktrees.** The pre-migration build (the merge base) and the step's build each trace every
  shipped kernel at every configuration its route table, schedule records or plan functions
  produce. The kernels are listed in §6; #115 and #120 count once merged.
- **The check.** `Kernel.source()` must match byte for byte, per (kernel, configuration).
- **Why it is enough.** Identical source, compiled by the same hipcc against the same headers,
  gives an identical code object, so bitwise outputs and speed follow. A step that edits
  `include/hk/` beyond deleting unused definitions is therefore at least V1.
- **The tool** is `tools/surface/snapshot.py` (S0). It is the CPU half of what the #107 A/B's dump
  mode did on the GPU.

**V1: the ISA is identical.**
- **When.** The step changes generated C++, by renaming a helper or changing the emitter, but
  should not change code.
- **The check.** Both builds are compiled, and per function the code object's ISA hash and its
  resources must match: VGPR, SGPR, AGPR, LDS, ScratchSize and occupancy.
- **How.** The #107 A/B recorded these per build by hooking `hk.runtime.build`; the same hook is
  reused.
- **If any function differs, the step drops to V2 for that kernel.**

**V2: the ISA changes. Every check below must pass:**
1. **Bitwise against the pre-migration build.**
   - Inputs: fixed inputs at the kernel's scoreboard shapes, with real weights where the
     scoreboard uses them (K1: layers 20 and 30).
   - The check: the sha256 of every output against the pre-migration build's.
   - Run twice, for determinism.
   - Where the kernel promises bitwise equality with a reference (K2 with vLLM's kernel,
     `mxfp8.quant` with `hk.mx.quant`), that test passes too.
2. **Not slower, in one process.**
   - The arms: the pre-migration kernels are loaded under different module names next to the
     step's, as the #107 A/B loaded its `_nf` arms. A cross-process A/B is not valid on these
     GPUs.
   - The harness: the MI355X scoreboard (`<workspace>/kda/scoreboard-mi355x`: `k1_ab`, `k2_sb`,
     `attn_sb`, `paged_sb`):
     - 7 interleaved rounds;
     - weights rotated through at least 1.2 GB, so no call finds them warm;
     - one locked GPU, with no co-tenant (recorded).
   - **Proposed acceptance:** the median of new over old is at most 1.005 at every point, with no
     point above 1.01 even inside the spread, and the geomean is at most 1.002. The thresholds are
     open question 4.
   - For gfx1100 / gfx1201 kernels: the same-process A/B on that GPU, plus the performance gate's
     baselines (`tests/perf/baselines`).
3. **The fence suite** (`tests/hk/gpu/test_fenced.py`): no access past any tensor, at ragged
   shapes.
4. **The build gates.**
   - ScratchSize 0 and no spill (the build refuses a spill), on every configuration.
   - Register counts and occupancy are not worse.
   - The LDS-read wait gate, the MFMA hazard gate (#119 once merged) and `cocheck` run clean on
     the new code object.
5. **The tiers.** IR, codegen and GPU tiers on gfx950, gfx1100 and gfx1201, on a head with the
   latest main.
6. **The engine (Gate C)**, for a step that changes a default-on route's kernel (S10): vLLM with
   the route, from the tree before and after. The routing counters must match call for call, and
   the outputs must match the pre-migration tree's.

**Where the GPUs come from.** V0 and V1 need none. V2 on gfx950 uses an MI355X under its lock, never
a CI-only or reserved GPU. gfx1100 / gfx1201 A/Bs use those runners under their locks.

## 8. Effort, risk and the timebox

| step | effort | risk | in the week (after review) | why |
|---|---|---|---|---|
| S0 registry, lint, snapshot tool | 1 day | none: tests only | yes | everything else is checked by it |
| S1 deletions | 0.5 day | low: no user | yes | V0 |
| S2 G5 front end | 0.5 day | low | yes | V0 |
| S3 G4 row index | 1 day | low; fixes a silent-misuse hole | yes | V0 |
| S4 rope `swap_halves` | 0.5 day | low | yes | V1 |
| S5 G2 front end | 0.5 day | low | yes | V0 |
| S12 header split | 1 day | low: a mechanical move | yes, if review lands by 10-30 | V0 / V1 |
| S6 `iota` → `lane_coords` | 0.5 day | low | after #121 merges | V1 |
| S7 G1 `hk.map` | 3-4 days | medium: a new IR construct, fusion, emitter | later (the design detail can start in the week) | the largest single lever: it narrows `map_cpp` |
| S8 sync copy | 0.5 day + A/B | medium: speed | later | V2 |
| S9 `mfma_strip` | 1 day | low; out-of-tree users only | later | after S5 / S6 |
| S10 `mx_quant_store` composition | 2 days + A/B | high: K1's +3-4% depends on it | later | V2; may end as KEEP |
| S11 attn_k3 pair exchange | 1 day + gfx1100 A/B | low-medium | later | needs the gfx1100 runner |
| S13 upstream library fixes | 1 day | low | later | fork review |

The week's set (S0-S5, S12) is about 5 days of work, all V0 or V1. It needs no GPU except the
tiers' own runs.

**What it delivers:**
- 12 of the 17 merges: A2, A3, A5, A7-A9 and B1-B5 at the front end (V0), and C10 (V1);
- both deletions;
- the registry enforcing the rule;
- `cdna.cuh` split by kind.

**What is left, and when.** The other five merges come after the timebox:
- A6, C3, D1 and E1 change code a kernel runs, and need V1 or V2 evidence;
- C2 (S6) waits only for #121.

G1 (S7), which narrows `map_cpp` and enables A6, is the main item for the next phase.

## 9. Open questions for review, and the decisions

1. **`map_cpp`**: keep it as a narrowed escape hatch (this design), or set a date to delete it once
   G1 exists? Deleting it needs an op for each intrinsic the remaining 2% of calls name (and a
   look at the 5% passed in variables).

   **Decision (10-28):** keep `map_cpp` as the narrowed escape hatch, with no deletion date. Revisit
   the remaining 2% (intrinsics) and 5% (expressions passed in variables) after S7.
2. **Out-of-tree callers.** The names that change reach few out-of-tree calls:
   - the decode stages: 33 calls (`mma_ABt_mx` 14, `mx_scales_loose` 7, `fp8_operand` 5,
     `mfma_strip` 5, `fp4_operand` 2);
   - #115: 7 calls.

   `iota` and `rotate_half` keep permanent aliases, and `map_cpp` stays. Is one minor release of
   deprecation warnings enough, or should the shims stay until those stages move into `hk.ops`?

   **Decision (10-28):** shims warn (`DeprecationWarning`) from the next release on. They are removed
   once calls from outside the repository reach zero, and at most two minor releases later. The
   decode stages and #115 are the project's own: their migration patches are prepared during S2 /
   S5, so no shim is left hanging.
3. **A6's source.** Should `mx_quant_store`'s composition read K1's h from LDS (needs a strip load
   from a 16-bit swizzled tile), or quantize from registers through `lane_strip` as #120 does
   (changes K1's epilogue structure)? And if the composition measures slower, is KEEP-format with
   the measurement acceptable?

   **Decision (10-28):** S10 is not this week. When it runs, prefer the register path (`lane_strip`,
   as #120 does), provided it is not slower. If it measures slower, it stays KEEP-format with the
   measurements attached.
4. **"Not slower" thresholds**: ≤ 1.005 per point and ≤ 1.002 geomean (time ratio), or something
   else?

   **Decision (10-28):** per point ≤ 1.005, geomean ≤ 1.002 (time ratio, new over old). Every A/B
   records the GPU's co-tenant state before and after it.
5. **Staged row indices** live in function-static `__shared__` arrays outside `hk.lds`'s
   allocator. Should G4 move them into it? That makes the LDS budget honest but moves LDS
   offsets, so it is V2, not V0.

   **Decision (10-28):** yes, move them into the `hk.lds` allocator so the LDS budget is real, but as
   a separate V2 step after this week, not folded into S3.
6. **`hk.arch.agpr`**: keep as an experiment with a sunset (this design), or delete now?

   **Decision (10-28):** keep as an experiment. The sunset is the timebox of the task evaluating
   it (2026-10-31 12:00Z): if that task has not adopted it by then, it is deleted.
7. **`scatter(add=True)`**: keep as the general primitive (this design), or delete with
   `scatter_add_lds`, since no shipped kernel uses it?

   **Decision (10-28):** keep as a primitive, on condition that a GPU test covers it. Without one it
   is deleted with `scatter_add_lds`.
8. **Registry enforcement**: should S0's tests fail CI from day one, or warn for a release while
   the allowlist settles?

   **Decision (10-28):** a hard CI failure from S0 on, with the allowlist seeded from this inventory so
   existing code stays green.
9. **#121**: approve as primitives (C11-C14), with `lane_coords` taking a type in S6 and
   `as_fragment` as its inverse in S5?

   **Decision (10-28):** C11-C14 approved as primitives. S6 makes `lane_coords` take a type; S5 adds
   `as_fragment` as the inverse of `lane_strip`.

## Appendix: cdna.cuh by section

Line ranges at main `af17846`. S12 regroups these by kind; nothing is renamed.

| lines | section | kind after S12 |
|---|---|---|
| 1-376 | int8 packing, `strided_gl`, waits and binds, base ops, bf16 round-to-nearest-even copies | lib (gap fills), primitives |
| 377-762 | register tile ↔ global through per-tile buffer descriptors; the 32x32 bf16 store | primitives; special (E4) |
| 763-975 | scoped accesses, signals, mailbox (`hk.comm`) | primitives |
| 977-1035 | fp32 `rt` ↔ `st` through an LDS pointer | lib (E3) |
| 1037-1375 | the LDS DMA (async, rows, gather, `gather_prep`); `group_load_sync`, `group_store_bounded` | primitives; lib (E1, E2) |
| 1377-1600 | col-layout masks and fills, `neg_infty` | special (E4) |
| 1602-1653 | `as_operand` | primitives |
| 1655-1748 | W4 word decode, the 16-column `col_reduce`, fp16 → fp32 `mma_ABt` | formats; special; lib |
| 1750-2156 | MX: `hk_mxs`, `load_mx_scales` (+AITER), `mx_quant_store`, `load_fp4_frags` (+LDS), `mma_ABt_mx` | formats |
| 2158-2204 | `load_w4` | formats |
| 2206-2360 | gfx942 group copies and register vectors | lib |
| 2362-2497 | scalars, `wg_arrive` (+XCD), atomics | primitives |
| 2498-2747 | rows by index: `load_gather`, `store_scatter` (+16x16 path, `hk_pinned`), `scatter_add_lds` | primitives; special; deleted (B7) |
| 2748-2800 | Philox | primitives |
| 2802-3072 | strips: types, butterflies, maps | primitives |
| 3073-3482 | element ops, mailbox pairs, `lane_gather`, `mfma_strip`, operand views, `iota`, group reductions | primitives; escape (`mfma_strip`) |
| 3483-3600 | `map_cpp`, strip copies, `swap_halves`, `rotate_half` | escape (`map_cpp`); primitives |
| 3600-4071 | strip reductions, broadcasts, fills, strip memory | primitives |
