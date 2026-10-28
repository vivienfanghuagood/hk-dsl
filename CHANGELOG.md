# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## Unreleased

What is on main after 0.8.0. `pip install hk-dsl` (0.8.0) does not have it;
the API reference marks such names *New after 0.8.0*.

**`include/hk/cdna.cuh` split by kind (DSL-surface S12).** It is an umbrella over
`include/hk/cdna/{core,lib,formats,special,primitives,escape}.cuh`, a pure move of the definitions
in their dependency order; every gfx950 kernel's code object is unchanged, function by function.
The build cache sees the parts through the preprocessor's dependency list (#136).

**Register views: `Tile.as_fragment`, `Tile.as_scales` (DSL-surface S5).**
- `words.as_fragment(hk.RT(hk.fp8e4m3, 16, 128), fmt="fp8" | "fp4")` reads an i32 strip's
  registers as a block-scaled MMA operand fragment, and `bytes.as_scales()` reads one as loose E8M0
  scales. No data moves. They are the inverse of the planned `lane_strip`.
- They replace `hk.fp8_operand`, `hk.fp4_operand` and `hk.mx_scales_loose`, now deprecated shims
  that trace to the same IR. `dsv41_index` uses the new spellings.

**`rotate_half` is `swap_halves(span, negate_first=True)` (DSL-surface S4).**
- `Tile.swap_halves(span=None, negate_first=False)` exchanges the halves of each `span`-column
  segment (the whole row by default) and optionally negates the first. `rotate_half(span)` stays
  as sugar for it. The two dialect ops are one (`hk.swap_halves {span, negate_first}`), and the two
  C++ templates are one, `kittens::swap_halves<SPAN, NEG>`.
- Evidence: the 70 rotary kernels' code objects are identical on gfx950, function by function
  (ISA hash and resources).

**Rows by index take an `hk.RowIndex` (DSL-surface S3).**
- `hk.RowIndex(ids, at, count=, div=, sort_topk=)` says which rows a gather, a scatter or a
  gathered LDS copy moves, and how an index names a row: plain, `div=k`, or AITER's sort ids. It
  replaces the `div=` / `sort_topk=` / `staged=` keywords and the `(ids, at)` pair of
  `g.load(gather=)`. `Global.gather(ty, ids, at)` / `scatter(t, ids, at)` with plain ids stay.
- `g.stage_rows(ix, rows, keep_choice=)` copies an index into LDS and returns the staged index the
  copies read there. It replaces `g.gather_prep(ids, at, rows, raw=)`.
- A consumer now refuses a staged index of the other form, and one no `stage_rows` has copied.
  Both used to be a silent read of an LDS array no one wrote.
- `ix.staged(rows)` describes the staged index ahead of the loop body that copies it, and
  `ix.offset(n)` gives a later tile's positions.
- A gathered copy's row limit (#125) carries over: `g.load(..., gather=ix, rows=n)`, or the
  index's own `count` (rows = max(count - at, 0)) -- the same IR as `gather=(ids, at), rows=`.
- The old keywords and `gather_prep` are deprecated shims that trace to the same IR.
- K1 and the RDNA MoE GEMM use the new form; every shipped kernel's generated source is
  byte-identical.

**The block-scaled MMA is `hk.mma`; MX scales are register values (DSL-surface S2).**
- `hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb, fmt="fp4", at=(tile, step)), acc)` is gfx950's
  block-scaled MFMA: `Tile.scaled(scales, fmt=, at=)` makes an MX operand (`hk.Scaled`), and both
  operands carry one. `hk.mma_ABt_mx` is a deprecated shim for it.
- `MxScales.copy(into=dst)` replaces `hk.copy_mx_scales(dst, src)`, and `hk.load_fp4_frags`
  takes an LDS tile as its source, replacing `hk.load_fp4_frags_lds`. Both old names are
  deprecated shims (DeprecationWarning), removed once nothing outside the repository calls them.
- `hk.MxScales` and `hk.Scaled` are exported and documented. K1, K2, gemm_cdna's MX mode and the
  DeepSeek-V4.1 index front (`dsv41_index`) use the new spellings. Every shipped kernel's generated source is byte-identical.
- Migrating a call with inline loads in its arguments: bind them to locals in the old argument
  order first (`hk.mma_ABt_mx` evaluated a, b, acc, sa, sb; `hk.mma` with `.scaled` evaluates a,
  sa, b, sb, acc), or the loads issue in another order and the kernel changes.
  `tools/surface/migrate.py` rewrites call sites and does this; [docs/migrating.md](docs/migrating.md).

**Removed: `Global.scatter_add_lds`; deprecated: `Tile.make_causal` (DSL-surface S1).**
- `scatter_add_lds` had no user since K1 chose the deterministic sum (K1 c33). Its dialect op
  and `kittens::scatter_add_lds` are gone. `SharedTile.load` + `Global.scatter(..., add=True)`
  does the same; `scatter(add=True)` stays, held by its GPU test (`test_scatter_add.py`).
- `Tile.make_causal(v)` is now `tril_(0, v)` on a copy, and `transposed=True` is `triu_(0, v)`.
  The call warns (`DeprecationWarning`). The `hk.make_causal` / `hk.make_causal_t` dialect ops
  are gone; no kernel used them.
  - On RDNA the old ops were the same C++ as `tril(0)` / `triu(0)`.
  - On gfx950 the old op only compiled as `make_causal` on fp32 tiles: there it matched
    `tril_(0, v)` and torch bit for bit (col layout, 16 to 64 square). include/cdna4 has no
    `make_causal_t` and no bf16 form, so the alias now covers more.
- Every shipped kernel's generated source is byte-identical (`tools/surface/snapshot.py
  sources`: gfx950 1145, gfx1100 1049, gfx1201 1049 kernels).

**The surface registry (DSL-surface S0).** `python/hk/dsl/surface.py` registers every name of
the language, every dialect op and every function in `include/hk/` with its kind (primitive,
format, escape hatch, experiment, internal, merge / delete / deprecated). The IR tier
(`tests/hk/ir/test_surface.py`) now fails on anything unregistered, and on an escape-hatch use in
a file of hk.ops, the templates or the examples that `scripts/escape_hatches_allow.txt` does not
list with its reason, or above that line's ceiling. A `Kernel` subclass may not override
`with_defines` / `source` unless listed there. Both ratchet: a registered name that is gone, or
uses under a ceiling, print a hint to tighten them and never fail.
- The API reference marks each non-primitive name with its kind and boundary, or its migration
  step.
- `tools/surface/snapshot.py` takes the Gate A / B evidence for a change to the generated code:
  every shipped kernel's generated source hashed (`sources`), or its resources and an ISA hash a
  function (`builds`), diffed between two checkouts.
- `tests/hk/gpu/test_specializations.py`: the 32x32 fp32 -> bf16 store, the 16-column column
  reduction and the -inf column masks (cdna.cuh's type-selected paths) give the generic results
  bit for bit on gfx950.
- A pull-request template with the reviewer checklist.

**The compile cache served kernels built from old headers.** The key hashed a
fixed set of files -- kittens.cuh, pyutils/, the target's arch tree and its
extra_includes (include/hk/cdna.cuh) -- and nothing they include. An edit to
include/hk/cdna_agpr.cuh (which cdna.cuh includes) or include/pyutils/hk_bind.cuh
(which every pybind module includes), in a checkout or an installed wheel, kept
every key, so the next build loaded the binary from before the edit with no
sign of it: an A/B or benchmark after a header fix could measure the old code.
The key now covers the contents of every header the kernel's compile reads, as
the compiler lists them: on a miss, one preprocessor pass per side (device, and
host for a `.so` -- hipcc's `-MD` lists the host's only) writes clang's
dependency file, kept beside the cache as a manifest; a hit re-hashes the listed
files. An edit to a header a kernel does not include still leaves it cached.
torch's and pybind11's versions are in the key when their headers are on the
include path (a torch upgraded in place kept its key), and so are the flags
hipcc appends from HIPCC_COMPILE_FLAGS_APPEND / HIPCC_LINK_FLAGS_APPEND and
HIP_CLANG_PATH. Warm path: +0.14-0.18 ms a cached kernel, against a 1.1-2.6 ms
hit. Every existing entry misses once (a new
key scheme); old entries are never read again and can be deleted.
`runtime.compile.target_headers_fingerprint` is gone. `tests/hk/ir/test_compile_cache.py`,
`tests/hk/codegen/test_compile.py`.

**gfx950: fp32 max and min reductions are IEEE 754-2019 maximum and minimum.**
Row and column `max` / `min` of register tiles and strips, `group_max` /
`group_min`, the combine of `cross_warp(v, "max" | "min")` and `fold_rows`,
and the online fold `m.max_(...)` / `m.min_(...)` are `llvm.maximum` /
`llvm.minimum` on gfx950 (`v_maximum3_f32`, `v_minimum3_f32`), where they were
`fmaxf` / `fminf`.
- Two semantic changes, on values no shipped kernel reduces: a NaN anywhere
  in what is reduced now gives NaN, however the reduction is split across
  lanes, warps or blocks (as `torch.amax` does; maxNum skipped it), and -0
  orders below +0 (the max of -0 and +0 is +0, the min -0; maxNum may return
  either). `tests/hk/gpu/test_reduce_nan.py` holds both, within a wave, across
  warps and across blocks.
- Unchanged: elementwise `hk.max` / `hk.min` (`maximum`, `minimum`, clamps)
  stay `fmaxf` / `fminf` everywhere; gfx942 (no `v_maximum3_f32`: LLVM would
  expand each maximum into compares and selects) and RDNA keep `fmaxf` for
  reductions too. docs/language.md has the two kinds.
- Why: in IEEE mode LLVM quiets a maxNum operand it cannot prove quiet with a
  `v_max_f32 x, x`. A reduction of 2k + 1 values pairs into `v_max3_f32`s but
  for one two-operand max, whose operands (MFMA results, loop-carried values)
  it never can: attention's row max took 3 instructions for that 1. The d128
  main loop is 693 -> 680 instructions, 348 -> 338 VALU.
- Outputs: every launch of every `tests/hk/gpu` test, re-run as a twin built
  from the previous headers and emitter, is bit-identical (1182 launches; the
  6 that differ also differ from themselves run to run: atomics). Speed, one
  process against the previous code (MI355X, d128 attention, 6 shapes):
  0.999-1.010x, median +0.35%. GEMM, K1 and K2 compile to the same
  instructions as before.

**CONTRIBUTING.md: new kernels add no kernel-specific C++ helpers or one-off DSL ops.**
- A needed exception goes through review with a written reason and is registered as an escape
  hatch with a usage boundary. The file also has a reviewer checklist.
- [docs/design/dsl-surface.md](docs/design/dsl-surface.md), the approved design, inventories
  every kernel-specific op and helper and classifies each: 17 to merge, 28 to keep, 2 to delete.
  It names the six general abstractions the merges fold into and gives a migration in which every
  kernel stays bit-identical and no kernel gets slower. The decisions on its open questions are
  recorded in it.

**gfx950: an LDS tile narrower than its read's layout was never copied.** A
tile with fewer rows or columns than the include/cdna4 st shape its read's
fragment picks -- a 32x16 bf16 tile read as the 32x16 row operand of the
32x32x16 MFMA, which picks st_32x32 -- held no whole subtile of that shape, and
include/cdna4's copies loop over subtiles: every store and load of it compiled
to nothing, and the MFMA read whatever the LDS held, with no error. The
translator now gives such an allocation the fragment's own shape (st_32x16
here) when that one tiles it, and refuses one that no shape tiles.
`tests/hk/ir/test_lds_narrow_tile.py`, `tests/hk/gpu/test_lds_narrow_tile.py`.

**gfx950: `hk.ops.sparse_mla`, DeepSeek-V4.1's sparse MLA prefill, and a vLLM route
for it (`HK_VLLM_SPARSE_MLA`, off by default).** The attention vLLM's ROCm path runs
for a prefill chunk (AITER's opus kernel at T >= 1024): 32 heads of 512, each
query's sliding window and top-k rows gathered by index into LDS by the DMA.
`prefill_mxfp8` writes instead the layer's inverse-RoPE'd MXFP8 rows for wo_a,
made from O in registers (`lane_strip`) and bit for bit vLLM's quant of `prefill`'s
bf16 rows; vLLM's bf16 rows are scratch, so none are written. On real-weight
DSV4.1-Flash TP2 calls at T 8192 on an MI355X: attention 1.11-1.13x opus; with the
MXFP8 rows, top-k layers 1.22x AITER's fused fp8 kernel and 1.39x opus + vLLM's
quant pass, sliding-window layers 1.12x and 1.97x. Below T 512 opus is faster and
the route leaves the call to vLLM. The kernel's gathered copies are bounded by each
query's rows (`rows=`), so no index past a query's list is read: a real-weight server
faulted on a mixed step before. Serving scorecard (TP2, real weights): prefill 8k
+6.1% and TTFT -5.8% against hk's default routes, GSM8K equal. The route is opt-in
(`HK_VLLM_SPARSE_MLA=1`). `tests/hk/gpu/test_sparse_mla.py`,
`tests/hk/gpu/test_vllm_sparse_mla.py`.

**gfx950: fp32 LDS vectors and 16-bit col-layout LDS stores are ds ops, not FLAT.**
`hk.SharedVec`'s load and store (what `hk.collective.cross_warp` combines
through) and the store of a 16-bit col-layout register tile into LDS (an
accumulator converted to bf16 -- attention's P handed to other warps) went
through include/cdna4's generic-pointer paths, so the compiler emitted FLAT
loads and stores. FLAT counts in vmcnt as well as lgkmcnt: a sparse-MLA kernel's
per-block max exchange waited for its next KV block's DMA. `include/hk/cdna.cuh`
now has the same accesses through LDS pointers, as the fp32 tiles got in #59.
`tests/hk/codegen/test_cdna_fp32_lds.py`.

**gfx950: a matrix-core result as a strip, where it sits: `Tile.lane_strip`,
`hk.lane_coords`, `Tile.run_xor`, and group reductions over any xor offsets.**
`acc.lane_strip(run)` takes a register tile's registers, lane by lane, as a 1-row
strip. No data moves. `hk.lane_coords(acc, run)` gives each element's row and
column in the tile, so `hk.map_cpp`, `group_max` and `store_elems` work on an MMA
result without a round trip through memory: an epilogue that quantizes
attention's output to MXFP8 per 32-row group. `run_xor(mask)` takes an element's
partner inside its lane's run, such as a rotary pair. `group_max`/`group_min`/
`group_sum` take any distinct power-of-two xor offsets up to 32, not only the
neighbouring 1 .. 2^(k-1). A 32x32 result's 32-row column block is a lane's
run plus lane l ^ 32's: `group_max(32, xor=(32,))`.
`tests/hk/gpu/test_strip.py`, `tests/hk/ir/test_strip.py`.

**vLLM, DeepSeek-V4.1 under `--monokernel`: the index layers as one hk launch
(`HK_VLLM_DSV41_INDEX`, on by default on gfx950).** Between dsv41_mono's two
launches each index layer ran vLLM's 9-12-kernel chain (`mono_decode._select`).
At decode steps of up to 8 rows, `hk.integration.vllm_dsv41_index` runs it as
one resident launch, `hk.ops.dsv41_index`: compressor, compressed-cache insert,
indexer K and Q, and, for rows of up to 16640 entries, the logits and the
top-k -- the candidate source's block top-k and the consumers' candidate mask
included. The caches, q and q_scale are vLLM's byte for byte and the top-k sets
equal (docs/integration.md, "The DeepSeek-V4.1 index launch"). Serving on
MI355X (TP2, max_model_len 16640, with hk's other default routes): TPOT 4.5 /
4.4 / 3.1% below mono's at batch 1 / 4 / 8, GSM8K-1319 within stock's noise.
The route patches through an import hook, since the plugin must not import the
model package, and its contract is read from the module's source.

**CDNA: the MFMA hazard gate pads what LLVM leaves short.** The hardware
does not interlock on an MFMA's result. LLVM before llvm/llvm-project#225989
-- every ROCm release so far, 7.2.3 and the 10.x images included -- pads too
little on the shortest path into a block it first reached the long way
(ROCm/llvm-project#5153), and gfx950's paged decode computed some scores
wrong from it. hk's build now checks every CDNA code object
(`hk.runtime.mfmahaz`: a shortest-distance walk backwards over the control
flow from each instruction that touches an MFMA's registers, against LLVM's
windows) and, by default, rebuilds a kernel that is short anywhere with the
`s_nop` padding the fixed compiler would emit (`HK_MFMA_GATE=fix`; `refuse`,
`report` and `0` are the other modes). The build log, the cache entry's
`MFMAHAZ` and `python3 -m hk.integration --status` name each padded kernel
and the places. A compiler that pads the gate's two-path probe is not
checked. Swept over every kernel the test tiers and the default routes build:
- ROCm 7.2.3 (CI's GPU image): gfx950's dense attention, head_dim 64 and 128,
  not causal -- `hk.attention`'s kernel -- reads S for the softmax's max 3-4
  wait states after the last QK MFMA on the taken path of a branch, where 12
  are needed (2 places a kernel). Padded, its output is bitwise the unpadded
  one's and its time 0.993-1.003x (6 shapes, one process).
- ROCm 10 (the vLLM image): HipKittens' causal attention (`attend_ker`,
  hk.ops.hipkittens), 3-6 wait states short at the join after an exec branch
  (6 places at head_dim 128, 3 at 64); and one `hk.ops.gemm_cdna_dp`
  candidate with a bias+GELU epilogue (an MFMA reading an overlapping
  accumulator 1 wait state early).
- The default-on routes' kernels (vllm_mxfp8, vllm_moe_a8w4, vllm_int4) are
  clean under both; gfx1100 / gfx1201 have no MFMAs.

**`warm-routes` builds every K2 kernel a server can pick.**
- `python3 -m hk.integration warm-routes` called K2 (the MXFP8 linear route) at
  M = 2048 / 5120 / 8192 / 16384 only. The four-warp `stages=2` schedule that
  `vllm_mxfp8.RANGES` takes at M 832-1376 of (5120, 4096) was never built, so
  every vLLM server compiled it on its first request in that range, in every
  TP worker (the `[hk] JIT: compiling kernel
  'mxfp8_4w_block_m128_block_n128_split_m1_split_n1_stages2'` line).
- `warm_routes.k2_plans(n, k)` now enumerates, for each routed (N, K), every
  (kind, schedule, tail recut) the route can launch at M up to 32768, and
  warm-routes makes one call at the smallest M of each.
  `tests/hk/ir/test_warm_routes.py` checks the cover against an independent
  scan of `schedule` and `mxfp8.tail_split`.

**gfx950: a gathered LDS copy takes a row limit, and reads no index past it.**
`g.load(..., wait=False, gather=(ids, at))` read `ids[at + r]` for every row r of the
tile. When an index list ended inside the tile (the last query of a CSR), the copy
read up to a tile of ids past the list's end. Usually that lands in the allocator's
slack; when the list ends its segment, it is a memory fault. In vLLM that was the
sparse-MLA prefill, under a real-weight server's mixed batch. `rows=` (refused with
`gather=` until now) now bounds a gathered copy: rows from the limit on read zero,
and their ids are not read; and no gathered copy or `g.gather_prep` reads an id past
the ids tensor (those rows read zero too). The copy stays branch-free, so its index
loads still issue together. `tests/hk/gpu/test_gather_load.py`,
`tests/hk/gpu/test_fenced.py::test_gathered_copy_reads_no_index_past_its_rows`.

**CDNA: `hk-mma-fence`, a DSL rule for matrix-core accumulators; `hk.arch.agpr`
to hold one in AGPRs.**
- `hk-mma-fence`, a pass in the kernel pipeline, puts an `hk.sched_barrier`
  between a matrix-core chain and the first other op that touches its
  accumulator. On gfx950 LLVM threaded the softmax's reads of AGPR scores
  through the two-page paged decode's MFMA chains and computed some scores
  2-4e-2 wrong, differently from one build to the next: one of those reads
  sat a branch after its MFMA, and LLVM's hazard recognizer, measuring the
  longer path into that block, padded 2 wait states where 8 are needed
  (fixed in LLVM by llvm/llvm-project#225989; ROCm 7.2.3 to 10.1 do not have
  it, ROCm/llvm-project#5153). The decode's hand-written fence is gone; the
  rule covers every CDNA kernel. `hk.arch.mma_fence(False)` turns it off for an
  A/B (`attn_kernel(mma_fence=)`, `decode_kernel(fence_qk=)`).
  - Where it lands: attention's prologue (its main loop fences its phases
    already), the decode's per-step scores and epilogue, GEMM epilogues.
  - Cost, measured in one process against main's code (MI355X): dense
    attention 0.993-1.004x (8 shapes, outputs bitwise equal); paged decode
    0.986-1.013x over 120 points, geometric mean 0.999x -- with one request on
    1 kv head at 1k-4k tokens 1% slower, the per-step fence of the one-page
    walk.
- `hk.arch.agpr(tile)` holds an fp32 accumulator that `hk.zeros` just made in
  AGPRs (CDNA): mmas into it are inline asm with C/D in AGPRs, other accesses
  go through a VGPR copy after the MFMAs' wait states
  (`include/hk/cdna_agpr.cuh`). Outputs are bitwise the VGPR form's.
  `attn_kernel(acc="agpr")` puts attention's O there; it stays off by default:
  0.93-0.95x on the 26 full-attention shapes where hk trails FlyDSL/CK, because
  LLVM cannot schedule around an asm MFMA (AGPR placement itself measured
  +1.5-2.4% like for like). LLVM 22's register split spills the d128 kernel;
  the causal pairs walk (two O tiles) spills on either toolchain.

**Attention tests hold the kernels to their own error.** Every attention
test (test_attn, test_sdpa, test_paged, test_paged_fused, test_paged_prefill,
test_attn_k3, test_hipkittens' GQA, test_schedules' attention,
test_vllm_backend_decode, test_fenced's attention cases) compares against
attention in float64 (`tests/hk/gpu/_attn_ref.py`) instead of torch's fp32
SDPA, which moved by more than the kernel's error from run to run on ROCm,
through one check, `_attn_ref.check`:
- every output row within 4 bf16 ulps of that row's largest value (worst
  measured: 2.31 on gfx950, CI and vLLM images; 1.32 on gfx1100);
- and 1e-2 of the whole output's largest value (worst 7.1e-3 / 5.3e-3),
  where the tests asked 2e-2 to 5e-2.
The gfx950 decode that computed some scores wrong (ROCm/llvm-project#5153)
was 16-30 ulps off in its worst row and passed the old bounds. The value
caches of the paged tests now come from the test's own generator, so a case
sees the same data in any order.

**gfx950 paged decode: two pages a step, a cheaper merge, and no point
slower than before the 512-split plan.** The previous entry's plan was up to
13% slower than the one before it on short sequences (1-2k tokens); this one
is not, and keeps the plan's gains on long ones.
- Where (requests x kv heads) is under 64 or 256-511 -- about one workgroup a
  CU -- the walk takes two pages a step on splits of 8 pages or more and one
  on shorter ones, in one kernel (`paged_cdna.pages_per_step`,
  `decode_kernel(ppstep=2)`, `hk.ops.paged.fused_kernel(ppstep=)`): 1.025x
  one page a step on those points.
- Two pages a step need a scheduling fence after QK (`decode_kernel(fence_qk=)`,
  on by default, no measurable cost). Without it the current LLVM threads the
  softmax's reads of the AGPR scores through the two pages' MFMA chains, and
  some scores come out wrong: 2-4e-2 off on a 128-token split, a different
  answer from one build to the next.
- Splits take at least 2 pages where the merge reads every split (up to 64
  splits, `paged_cdna.merge_reads_all`, `min_pages_for`) and, with more
  splits, on sequences of up to 128 pages (`floor_pages`); 4 otherwise. A
  short sequence on a plan for long ones walked 4 pages a split, 1.2 us more.
- The merge: up to a round of 8 splits a warp it reads every split without
  first loading the sequence length (the walk writes empty splits' zeros); it
  folds on at most 8 warps (`MERGE_WARPS`, 3-10% faster than 16 from 4k
  tokens at 256 splits); and it sums only the warps that folded something.
- On many-round plans an empty split skips its query load and copies, and
  split indices are turned by the (request, kv head) pair so that a short
  sequence's splits spread over the XCDs.
- Measured on the attention task's decode grid (MI355X; 32/8, 64/8, 32/4 and
  8/1 heads; 1-64 requests; 1k, 2k, 4k-128k tokens; CUDA graphs; KV rotated
  past 1.2 GB), main before the 512-split plan, the 512-split plan, this and
  the rivals in one process, compared round by round:
  - against main before the 512-split plan: 1.107x (geometric mean of the
    per-point medians), up to 3.5x; the lowest four points, 1k tokens, are
    0.991-0.998x with rounds either side of 1.0, where the 512-split plan had
    0.87-0.98x on 8 points;
  - against the 512-split plan: 1.039x, lowest 0.994x;
  - against the best of AITER's Gluon unified attention, FlyDSL `pa_decode` and
    asm `pa_fwd_asm`, on the 120 points without 2k: 14 WIN, 93 tie, 13 LOSS,
    geometric mean 1.009x, where the 512-split plan had 14 / 60 / 46 and
    0.977x the same day.
- Outputs change by the order pages and splits are folded in (within the fp32
  check; deterministic under another process on the GPU).

**gfx950 paged decode: a merge on several warps, and up to 512 splits.** The
split decode's merge (`hk.ops.paged_cdna.merge_kernel`) folded every split of
a (request, kv head, query row) on one warp, 8 splits a round. That capped
the plan at 32 splits, and a request on few kv heads walked its whole context
on 32 to 128 workgroups.
- The merge now folds on up to 16 warps (`paged_cdna.merge_warps`, about one
  round of 8 splits each), combined through LDS; one round stays one warp's.
- The plan (`hk.ops.paged.plan_splits` on gfx950, `paged_cdna.target_workgroups`)
  aims at 512 one-warp workgroups from 64 to 255 (request, kv head) pairs and
  256 otherwise, up to 512 splits (`paged_cdna.MAX_SPLITS`).
- The walk no longer writes partials for splits the merge does not read.
- Measured on the attention task's decode grid (MI355X; 32/8, 64/8, 32/4 and 8/1
  heads; 1-64 requests; 1k-128k tokens; CUDA graphs; KV rotated past 1.2 GB),
  same process, against main's plan and merge:
  - 1.08x geometric mean, up to 3.3x: one request on 1 kv head at 128k tokens
    is 25 µs, where it was 83 (AITER FlyDSL `pa_decode` 27, AITER Gluon 57);
  - against the best of AITER's Gluon unified attention, FlyDSL `pa_decode` and
    asm `pa_fwd_asm`: 14 WIN, 62 tie, 44 LOSS, where main had 11 / 38 / 71;
    geometric mean 0.98x, where it was 0.91x;
  - 1k tokens at 1-4 requests are up to 7% slower than main (more workgroups
    launched for a short sequence); hk is still 1.16-1.32x the best there.
- Outputs change by the order splits are folded in (within the fp32 check;
  deterministic). `HK_SPLITS` still forces a count.

**gfx950 paged prefill: long KV walks cut over the cached prefix.** A
short chunk on a long prefix, or a batch on few heads, is a few query tiles
of 256 rows walking the whole context, with most of the machine idle: 4096
cached + 512 new tokens on 32/8 heads is 64 workgroups for 256 CUs.
- When the tiles times the heads come short of 256 workgroups,
  `hk.ops.paged_prefill.split_plan` cuts each tile's walk over the prefix
  every one of its queries sees, at least 16 64-key blocks a piece, the last
  piece taking the rest with the diagonal. The cut tiles write normalised
  partials and (m, l) (`attn_cdna.attn_kernel(kv_split=True)`,
  `split_kernels_for`), and `attn_cdna.prefill_merge_kernel` weighs them.
- Its entries go longest first across requests. On one or two kv heads the
  vLLM backend takes that order even with nothing to cut: 1.2-1.4x on a batch
  of prompts there. On more kv heads that order cost 2-4% of L2 reuse, and
  the backend keeps the single-tile work list unless the plan cuts.
- The vLLM backend plans it in the metadata builder, from
  `seq_lens_cpu_upper_bound` (exact for prefill rows); without it the
  prefill is the single launch as before.
- Measured on the attention task's prefill grid (MI355X; 32/8, 64/8, 32/4 and
  8/1 heads; 13 batches from fresh prompts to 65536 + 2048 chunks), same
  process, against AITER's Gluon unified attention, FlyDSL's paged varlen and
  AITER's varlen:
  - hk was 0 WIN, 18 tie, 34 LOSS; it is 18 / 16 / 18;
  - geometric mean 0.93x -> 1.20x;
  - 1.29x main on average, up to 6.9x: 32768 + 512 on 8/1 heads takes 101 µs,
    where it took 697 (AITER Gluon 602);
  - no batch slower than main beyond 1.1%.
- Cut tiles' outputs change by the order partials are folded in (within the
  fp32 check; deterministic). Uncut ones are bitwise as before.

**gfx950 causal attention walks query blocks in pairs.** A causal workgroup of
`hk.ops.attn_cdna` (what `hk.attention` and the SDPA drop-in run on gfx950)
now walks query block nqb - 1 - y and then block y, so every workgroup does
the same work. Before, longest-first order inside each batch still ended
grids with few workgroups a batch on a round of long tiles.
- Outputs are bitwise the same (all 42 causal points of the attention grid,
  head_dim 128, 8/1 to 64/8 heads, N 1k-32k).
- Same process, Q/K/V rotated, against the single-block walk:
  - 8 query / 1 KV heads (a 70B-class model at TP8): 1.26-1.30x at N up to 4k;
  - 16 / 2 heads: 1.10-1.12x at 1k-2k;
  - 40 / 8 heads: 1.01-1.03x;
  - the rest within ±1.4%.
- At 8/1 and 16/2 heads, short N, hk now ties AITER's asm `fmha_v3`, where it
  had been at 0.74-0.92x.
- `attn_cdna.kernel_for(..., pairs=False)` and the `pairs` knob of
  `hk.schedules.AttnCdna` give back the single-block walk. The spatial
  numbering and the paged prefill do not change.

**K2 below its MIN_M** (gfx950, `vllm_mxfp8.RANGES`). DeepSeek-V4.1's MXFP8 linears at decode
and mid-size M were measured against vLLM's kernel, FlyDSL's, AITER Triton's and hipBLASLt's, route
level with each side's activation quant, weights rotated, two sweeps: hk now takes engram wkv
(12800 x 6144) from M 160, wo_b (5120 x 4096) from 416, the shared expert's w2 (5120 x 1152) from
512, wq_b (16384 x 1280) from 576, the shared expert's w1|w3 (2304 x 5120) from 960 and fused
wq_a|wkv (1792 x 5120) from 1216, in the ranges where it won by >= 3% in every round of both
sweeps -- 1.04-1.61x the best of the four -- on the four-warp kernel at 128 x 128 or the
eight-warp one. Bitwise vLLM's output at every point. At M <= 128, and for the indexer's wq_b
below its MIN_M, no hk kernel wins; those calls stay with vLLM.

**Host time per call** (#91, cross-arch). An engine route's Python no longer
re-decides per call what only the call's shapes decide, and code-object
kernels launch from a per-shape memo without the torch dispatcher
(`hk.runtime.hostpath`, [integration](docs/integration.md#host-time-per-call)).
Same kernels, same arguments, bitwise the same outputs; `HK_HOST_FASTPATH=0`
restores every call's previous path.
- Host µs per call, before -> now: `vllm_gemm` decode 38.5 -> 14.0 and
  prefill 46.5 -> 17.6 (W7900D); `vllm_int4` 33.6 -> 12.5 (W7900D) and
  19.9 -> 9.4 (MI355X); K3 attention 82.5 -> 49.2; K1 167.1 -> 122.6, and a
  MoE call K1 leaves to AITER 252.6 -> 208.6 (AITER's own: 202.7).
- Eager decode of Qwen3-8B-AWQ: 1.27x tokens/s on MI355X, 1.29-1.37x on
  W7900D, at 1 and 16 requests; DeepSeek-V4.1-Flash TP2 1.04-1.07x. Prefill
  and GPU-bound decode (Qwen3-8B bf16) unchanged.
- Code-object launches in C (#92) where hk._mlir has `_hkLaunch` (built with the bindings): 2.9 µs a
  launch on the W7900D and 4.6 on the MI355X, under the pybind entry point's 3.4 / 5.3 (the
  Python launcher: 5.4 / 6.6).
- New: `hk.runtime.hostpath` (`ENABLED`, `set_enabled`),
  `hk.runtime.codeobj.direct`, `hk.runtime.torch_ext.launcher`,
  `CodeObjectKernel.launcher` (`native=`, #92), `hk.ops.w4a16.plan`. They are runtime and
  integration names, which the API reference does not cover, so they carry
  no mark there; [the runtime](docs/runtime.md) and
  [integration](docs/integration.md#host-time-per-call) describe them.

**DeepSeek-V4.1 routes, retuned from the MI355X scoreboard** (#84, gfx950, real weights, TP2; every
point against the best of AITER's tuned FlyDSL, FlyDSL, AITER Triton, hipBLASLt and vLLM's kernel).
- K1 (`vllm_moe_a8w4`) takes M >= 2560, not 8192 (`HK_MOE_A8W4_MIN_M`), with #85's kernels: on
  the real router's choices 1.03-1.08x AITER at 2560, 1.08-1.20x at 3072-4096 and 1.27-1.38x
  from 4608 to 8192, layers 20 and 30. With uniform routing, which DSV4.1's router does not
  produce, AITER is 3-11% faster from 2304 to 3584; from 4096 hk wins there too (the reasons
  are at `MIN_M`).
- K2 (`vllm_mxfp8`) takes the indexer's wq_b (N 4096, K 1280) from M 2825, 1.08-1.20x, and
  engram's wkv (12800 x 6144) from M 1024 instead of 2048, 1.29-1.79x; per (N, K) in
  `MIN_M_NK`. Bit-identical to vLLM's kernel at every new point, ragged M included.

**fp8 GEMM on gfx1201** (#76, RDNA4). `hk.ops.scaled_mm(a, b, scale_a, scale_b,
*, bias=None, out=None)` is `torch._scaled_mm`'s product, bf16
`(A @ B.T) * scale_a * scale_b (+ bias)` from OCP fp8 operands with fp32 accumulation, in
`hk.ops.matmul`'s (N, K) layout, with per-tensor or per-token x per-channel scales. It runs on
gfx1201's fp8 WMMA: `hk.ops.gemm`'s kernel takes `op_dtype=hk.fp8e4m3` / `hk.fp8e5m2` there,
and gfx1100 (no fp8 WMMA) and gfx950 (whose fp8 GEMMs are gemm_cdna's) refuse it. On a
Radeon AI PRO R9700 against `torch._scaled_mm` (hipBLASLt), in one process: 240 vs 236 TFLOP/s
at 4096³ per-tensor; per-token x per-channel 1.28-1.99x at M 128-4096 (N 4096 / 14336); a tie
at M 1-64 with K 4096, and 0.89x at M 1-64 with K 14336 (decode shapes, not tuned yet).

**K1 MoE a8w4 at prefill M: 1.34-1.49x AITER** (gfx950). On DeepSeek-V4.1-Flash's TP2 shapes with
real weights and real routing, against `aiter.fused_moe`'s tuned FlyDSL kernels (MI355X scoreboard
harness, four runs): 1.42-1.49x at M 8192 and 1.34-1.36x at 16384, from 1.35-1.37x / 1.22-1.24x
before; 1.14-1.47x at M 4096-7707, and 1-4% less time than before at M 2560-3584. Outputs
are bit for bit those of before; decode M unchanged.
The route's thresholds are not touched.
- `store_scatter` stores 16x16 fp32 fragments as bf16 in 16-byte runs (`v_permlane16_swap` pairs
  lanes l and l+16): half the store instructions. Stage 2's scattered top-k rows were a third of
  its time.
- `Global.scatter(..., staged=N)` reads its row ids from an LDS copy made by
  `g.gather_prep(ids, at, N, raw=True)` (both new after 0.8.0). A global id load in an epilogue is a vmcnt
  wait, and on gfx950 that retires, in order, every load and store issued before it: in the
  resident stage 2, the next column tile's weight prefetch and the epilogue's own stores. Stage 2
  now stages its block's ids, and loads its rows' routing weights once, not once a column tile.
- Both stages issue the next token fragment's LDS read before the current fragment's MFMAs
  (`stage1_sorted(xpipe=True)`, the default).
- `load_fp4_frags` builds its per-expert buffer resource with a 32-bit compare and a select: the
  64-bit bounds check was two scalar branches at every K step.

**Attention on gfx1201** (#77, RDNA4). `hk.attention` (and the SDPA drop-in) runs its own
tiling on gfx1201 from 256 query rows: 16 warps (a 256-row Q tile), a head-first grid
that starts the longest rows of every head first, and, causal, warps that skip the KV
blocks they precede. At n 4096, GQA 4, head_dim 128 it ties AITER's gfx1201 FlyDSL kernel
(1.40 vs 1.41 ms causal, 2.65 vs 2.69 full; gfx1100's tiling took 1.71 / 2.78). The causal
KV block depends on the compiler: 64 rows under ROCm 7.2.3's clang 22, 32 under the
R9700 vLLM image's clang 23, where 64 is 16-20% slower. A shorter Q keeps gfx1100's
192-row tiling, so what `hk.attention` accepts is the same on every RDNA arch.
`attn_kernel(warp_skip=True)` without `causal` raises.

**Element ops for a whole cache** (#88, gfx950). The forms DeepSeek-V4.1's
index chain needed as one launch on vLLM's caches:
- `Global.load_elems` / `store_elems(..., row=r)`: indices count from row `r`,
  its base formed in 64 bits and its descriptor that one row -- a paged cache
  of ~100 GB addressed a block at a time (a descriptor over the tensor wraps
  at 4 GB).
- `Global.load_elems(..., runs=True)`: a lane's run is one load of its bytes
  (masked by its first index), not one `buffer_load_dword` a value.
- `SharedTile.load_elems(..., runs=True)`: 4 elements a 16-byte LDS read. Index
  a 16-column tile: a wider `hk.ST` is 16x16 subtiles 1 KB apart, and a
  flat-indexed strip conflicts 8 ways on every read.
- `SharedTile.store_elems` into a bf16 tile (round to nearest even): an MMA
  operand built element by element. LDS element reads are branch-free (an
  index out of range still reads 0); a guarded read was one exec-masked block
  each, and a strip's reads ran one at a time.

**Norms on gfx1201** (#78, RDNA4). The strip plans take per-lane targets measured on the
R9700: rms_norm at 2048 rows goes from 0.96x to 1.04x the best other implementation, and
the fused add + rms_norm at 8192 rows from 0.93x to a tie (447 vs 443 us). gfx1100's and
gfx950's plans are unchanged.

**Ops for an index top-k in one launch** (#98, gfx950), from DeepSeek-V4.1's
index logits and top-512 inside the same launch:
- `SharedTile.atomic_add_elems(val, idx)`: i32 adds into an LDS tile
  (`ds_add_u32`), every add to one element landing -- a radix select's
  histogram.
- `hk.fp4_operand(words)` / `hk.mx_scales_loose(bytes)`: an MXFP4 MMA operand
  and its E8M0 scales from a lane's own gathered dwords / byte, for operands in
  a layout no tile load knows (logits straight off a preshuffled paged K
  cache).
- `Tile.first_lane()`: lane 0's element of an i32 strip as a scalar
  (`v_readfirstlane`); one LDS word read by every wave after a barrier may
  then drive an `hk.uniform` branch (a select's exact-fit early exit).

**Ops for K2's MoE in one launch** (#104, gfx950), from DeepSeek-V4.1's MoE as one
resident hk launch, bitwise vs dsv41_mono's:
- `while hk.uniform(t < n):` -- a runtime loop every warp runs the same
  iterations of may hold barriers: a work queue's loop over `hk.comm.claim`.
- `hk.fp8_operand(words)`: an MXFP8 MMA operand from a lane's eight dwords, for
  fp8 operands kept in LDS as words (an fp8 shared tile has no register path).
- `hk.Tensor[..., "stream"]`: non-temporal accesses for a weight stream a launch
  reads once (K2's MoE: 135 -> 123 us a launch). Not a scope: mailboxes, flags
  and symmetric memory refuse it.

**vLLM on gfx1201** (#80, RDNA4), in the Radeon AI PRO R9700 vLLM image
(`ghcr.io/inferstation/vllm-rocm-r9700-main`: TheRock ROCm 7.14, clang 23, torch 2.13).
- `vllm_gemm` is on by default on gfx1201, with a measured route table
  (`routes_vllm_linear_gfx1201.json`): tokens identical to stock and routing counts equal
  round to round on Qwen3-1.7B and Qwen3.5-9B; prefill 1.11-1.16x on the 9B.
- `vllm_int4` (AWQ) is on by default on gfx1201: decode 2.5-4.5x the image's AWQ path on
  Qwen3-8B-AWQ.
- `vllm_moe` and `vllm_sdpa` stay off there: the MoE route is right but 0.90-0.93x at 16
  tokens, and the SDPA drop-in's per-call host time hid its kernel's 1.10-1.40x.
- The decode GEMM's four-tile schedule on gfx1201 is combine depth 2
  (`decode_gemm.SCHEDULES_AT`), the one that builds without a spill under both compilers.

**Attention on gfx1201 at head_dim 64** (#94, RDNA4). The 16-warp tilings now cover
head_dim 64 too: causal from 1024 query rows, full from 2048; at n 1024-8192, GQA 4, under
either compiler, causal 1.10-1.84x and full 1.03-1.14x gfx1100's tiling. Below those lengths
gfx1100's tiling stays, which is faster at a ViT's 577 and 1025 tokens.

**gfx1201's bf16 GEMM at 32-512 rows** (#102, RDNA4). hk ships tuned schedules for an
RDNA arch in `python/hk/tuned/routes_gemm_<arch>.json`, in `hk.tune`'s record format,
read behind a machine's own records (`hk.tune.records.shipped_route`); `hk.ops.gemm.matmul`,
`schedule_for`, `hk.templates.gemm` and the vLLM route take them. gfx1201's 18, tuned under
both ROCm 7.2.3's clang 22 and the R9700 vLLM image's clang 23 on Qwen3's four (N, K):
1.03-1.40x the default schedule; with W from GDDR6, 1.28-5.7x hipBLASLt's NT and 0.94-1.09x
its best layout. The gfx1201 vLLM table, re-measured with them, routes most of M 33-256 to
hk: Qwen3.5-9B at 32-128 requests, tokens identical and requests 1.08-1.11x stock (the old
table's 1.01-1.08x). `HK_AUTOTUNE=0` ignores the shipped records.

**vllm_moe on gfx1201, up to 2 tokens a step** (#108, RDNA4). `vllm_moe.BLOCK16_MAX_TOKENS_AT`
gives an arch its own token limit for hk's MoE GEMM; gfx1201's is 2, where vLLM skips the
sort and gives each (token, choice) pair a block of its own. There, OLMoE decode is
1.10-1.12x stock at one request and 1.06-1.07x at two (tokens identical, routing counts
equal round for round); past it vLLM's Triton kernel runs, which from 16 tokens a step
streams the chosen experts at ~96% of GDDR6's rate. `vllm_moe` is on by default on gfx1201.

**gfx1201's bf16 GEMM at 1024-8192 rows** (#114, RDNA4). 11 more shipped gfx1201 schedules
in `routes_gemm_gfx1201.json`, for Qwen3's four (N, K): 4.1-6.1% faster than the default
schedule under the R9700 vLLM image's clang 23, and under ROCm 7.2.3's clang 22 2.3-3.4%
faster at four shapes and within 1% at the rest (each compiler's own champions disagree at
these sizes; the shipped one is the one that holds under both).

**Fixes**
- **`hk.schedules.lookup(arch=...)`, `hk.templates.gemm.plan(arch=...)` and hk.tune's
  incumbent read the current arch's tuned records, not the asked arch's** (#114): under
  `HK_ARCH=gfx1201`, `lookup(arch="gfx1100")` could return a gfx1201 schedule illegal on
  gfx1100. `hk.ops.gemm.user_route` / `schedule_for` take an `arch` now.
- **gfx950: `hk.ops.gemm.matmul` raised `ResourceError` in the vLLM image** (#84) at M 192-512 for
  N 28672 / 14336, K 4096: its table routes there to `dp128x224`, which ROCm 10's hipcc builds
  with a 13-VGPR spill, and the spill gate refuses it. The route now records a refused candidate
  (once a process, with a line on stderr) and takes the next-fastest one the table measured at
  that point (`pp128`, `ppk2`), else the rule's next; a refused kernel is never run.
- **A kernel registered as a torch op through libtorch's headers did not
  compile against TheRock's torch 2.13** (#79), the torch of the Radeon AI PRO
  R9700 vLLM image (ROCm 7.14), whose ROCm headers are hipified the newer way
  (`c10::cuda::CUDAGuard`, no `c10::hip::HIPGuard`): the C++ shim
  `hk.codegen.scaffold` writes for `hk.runtime.torch_ext` failed with
  `no type named 'HIPGuard' in namespace 'c10::hip'`. It now picks the names by
  `C10_HIP_API`, which only the older headers define; with those (rocm/pytorch's
  torch 2.10, the vLLM 0.31 venv's 2.13, the gfx950 images) it builds the same
  code as before.
- **The LDS wait check refused a correct kernel** (#87). `hk.runtime.cocheck`'s scan walked the code in
  address order and carried a block's in-flight LDS loads past an unconditional `s_branch` into a block
  that code never reaches. It now follows branches: a forward branch carries its states to its target.
- **An LDS tile allocated after one inside a branch not every warp takes landed at two addresses** (#96).
  HipKittens' `shared_allocator` hands out offsets in execution order, so the warps that skipped the branch saw
  every later tile elsewhere -- silently (DeepSeek-V4.1's index-chain top-k read two histograms). `hk.lds.alloc` now
  refuses an allocation that follows such a branch outside it; inside the branch, or in a workgroup-uniform one,
  stays allowed.

**CI and tests**
- A fourth GPU-tier job, `gfx1201`, on a shared RDNA4 box (Radeon AI PRO R9700), with an
  ephemeral runner registered for this repository. It is continue-on-error and not a required
  check until the RDNA4 port's tier is green (#67,
  [testing](docs/testing.md#the-gfx1201-runner)).
- The `gfx1201` job is no longer continue-on-error, now that the RDNA4 port's GPU tier is green
  there: a red gfx1201 fails the run (#97). Whether it is one of main's required checks is a
  branch-protection setting.
- gfx1201 has CI legs of its own for the tiers that need no GPU: `IR tier, gfx1201` and
  `codegen tier (I/4), gfx1201`, with `HK_ARCH=gfx1201` and the same ROCm 7.1 hipcc (#106).
- A performance-gate baseline for gfx1201, `tests/perf/baselines/gfx1201.json` (34 pairs:
  gemm, attention, norms, softmax, fp8 and int8 against torch, Inductor and Triton);
  `check_regression.py` takes each arch's scoreboard arguments (`ARGS_AT`). Run by hand on an
  R9700, as gfx1100's is, not in the gfx1201 job (#106,
  [testing](docs/testing.md#the-gfx1201-runner)).
- Pull requests merge through a merge train, `scripts/ci/merge_train.py` (#113,
  [testing](docs/testing.md#the-merge-train)).
  - Twice a day, at 01:00Z and 13:00Z, every READY PR is stacked in READY order on
    `train/<YYYYMMDD>-<HHMM>`, as GitHub's squash merge will apply it.
  - CI and the GPU tier's four jobs run once on that branch.
  - If they are green, the PRs are squash-merged in order, and main's tree is checked
    against the train's after each merge.
  - A PR that conflicts, or that turns the train red, is kicked out and the train is
    rebuilt without it.
  - READY is a PR comment that names the current head.
- A gfx950 job's wait for an idle MI355X is `HK_CI_PICK_WAIT_MIN` minutes (runner `.env`, default 60,
  as before), so CI can wait out other users' processes on its GPUs instead of failing; the gfx950
  and gfx950-vllm jobs' timeout is 360 minutes, which covers the wait (#123).
- The merge train resolves a conflict in `CHANGELOG.md` by union instead of kicking the PR.
  The union is checked before it is kept: no conflict markers, no entry twice, and each
  version section kept as it was. To merge such a PR, its branch is first brought to the
  train's tree with an admin bypass, which is logged. `readmit` puts back the PRs kicked
  only for such a conflict (#128, [testing](docs/testing.md#the-merge-train)).

## 0.8.0

hk-dsl 0.8.0: workgroup placement as a compiler pass on gfx950; `hk.map_cpp`,
i32 strips, and the strip ops and element exchange a megakernel's roles need; gfx1201 (RDNA4) on
hardware; and reproducible wheels, one `manylinux_2_35` wheel per interpreter.
hk stays on by default in vLLM and SGLang for the routes validated on your GPU:
`HK_DISABLE=1` turns every route off, `HK_<ROUTE>=0` one of them
([integration](docs/integration.md#on-by-default-and-turning-it-off)).

On PyPI: `pip install hk-dsl`, for CPython 3.10 and 3.12 on Linux x86_64 with
glibc 2.35 or newer (Ubuntu 22.04 and later, and vLLM's ROCm images): one
`manylinux_2_35` wheel per interpreter. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed. The API reference marks what 0.8.0 adds *New after
0.7.0*.

    pip install hk_dsl-0.8.0-cp312-cp312-manylinux_2_35_x86_64.whl

**Workgroup placement as a pass** (#63, gfx950; design in
[docs/design/placement-pass.md](docs/design/placement-pass.md), #62).
Which XCD runs which workgroup, as one mechanism instead of hand-written
remaps:
- `hk.kernel(placement=None | "auto" | hk.Placement)`. `hk.Placement.cluster()`
  runs one sharing cluster an XCD at a time, its key found by the analysis
  (`hk.placement.analyze`: each global read's sharing key over the grid axes)
  or named with `hk.Digit`s; `hk.Placement.chunked(chunk, tail)` renumbers
  `block_idx.x` alone. `"auto"` launches the cluster placement where
  `hk.placement.choose` takes it: not for order-dependent or resident
  kernels, from five KV streams an XCD, and not for non-uniform work under
  twelve rounds an XCD (thresholds measured in #55). `Kernel.placed()` and
  `Kernel.placement_choice()` give the placed kernel and the decision.
- Outputs are bitwise the unplaced kernel's. `attn_cdna`'s `mapping="spatial"`
  (#55) is now `hk.Placement.cluster()`, and `gemm_cdna` / `gemm_cdna_dp`
  take `Placement.chunked` instead of their inline remaps: 11 GEMM shapes
  bitwise, new/old 0.997-1.007; 15 attention points bitwise, 0.993-1.003 of
  #55's hand-written numbering.
- `hk.arch.xcd_count()`: KFD's `num_xcc` (MI350P / DPX: 4, MI300A: 6), the
  arch's default otherwise, `HK_XCDS` to override. With 1 (gfx1100) every
  placement is the identity.
- `python -m hk.check FILE.py:NAME --placement --inputs ...` checks that a
  launch's placement covers its grid exactly once, evaluating the decode the
  kernel's own emitted ops compute ([check](docs/check.md#which-xcd-runs-which-workgroup---placement)).
- **M2** (#65): K2 (`mxfp8`), K1 (`moe_a8w4`) and the stream-K plan take it too, so hk.ops has no
  hand-written XCD remap left (and no `NUM_XCDS`); persistent kernels use
  `hk.placement.renumber` / `hk.placement.xcd_range`. K2 at the 18 DSV4.1 points bitwise,
  0.992-1.011; K1 bitwise, and **1.19-1.24x faster** at M where its old remap fell back to the
  hardware's order (stage-1 grid not a multiple of 8: M 9500, 10000, 14000, 15001), within 0.1%
  where it applied. `HK_XCDS=1` turns every placement off.
- A user guide, [docs/placement.md](docs/placement.md) (when to use it, examples, the measured
  tables, limits, turning it off); the design document rewritten as built; an
  [hk.placement](docs/api/hk.placement.md) API page. `hk.Digit.of`'s radix may be spelled
  `"tensor.dim"` (`"k.depth"`).

**`hk.map_cpp` and i32 strips** (#53, gfx950;
[the language](docs/language.md#elementwise-c-over-strips-and-i32-strips-gfx950)).
- `hk.map_cpp(expr, *args, dtype=None)` evaluates one C++ expression for each
  element of its strip arguments (`x0`, `x1`, ... bound to the elements, or
  to an `hk.Int` / Python int), into an fp32 or i32 strip. It is for what a
  kernel must reproduce bit for bit and the operators cannot say: bit
  manipulation, a fused multiply-add, one particular instruction. The
  expression compiles with floating-point contraction off (`a * b + c` is two
  roundings; an fma is `__builtin_fmaf`). Strips of fp32 and i32 together
  must say `dtype=`; wave32 targets refuse it.
- Strips may hold i32: they load, store, convert with `.to` and are
  `hk.map_cpp`'s arguments and results. Operators and reductions on them are
  refused, so integer arithmetic is written in `hk.map_cpp`.

**Strip ops for megakernel roles** (#72, gfx950), which DeepSeek-V4.1's K1
written in hk uses (bitwise to the hand-written K1 over ~496k layer calls a
rank):
- `hk.iota(ty)`: an i32 strip whose element (r, c) is c.
- `Global.load_elems(idx)` / `store_elems(val, idx)`: element gather and
  scatter at flat indices; an index below zero or past the tensor reads 0 and
  writes nothing.
- `Tile.group_sum` / `group_max` / `group_min(cols, lanes=, xor=)`:
  reductions over groups of lanes, in a stated order like
  `sum(axis=1, lanes=, xor=)`.
- `Tile.lane_gather(lanes)` (`ds_bpermute` of each element), and
  `hk.mfma_strip("f32_16x16x4", a, b, acc)`, one raw MFMA on strip registers
  for reproducing another kernel's MFMA bit for bit.

**Element exchange for megakernel hand-offs** (#73, gfx950), also from K1 in
hk:
- `Global.post_elems(val, idx, tag, runs=, bf16=)` / `take_elems(idx, tag,
  runs=, bf16=, timeout_s=)`: tagged mailbox pairs, each value and its tag in
  one 8-byte scoped store, as `hk.comm.post` / `take` per element. `runs=True`
  moves a lane's run as 16-byte pairs of pairs; `bf16=True` packs two bf16 a
  pair (values must already be bf16; refused without `runs`). A wave with no
  live pair polls nothing, and a timeout traps.
- `SharedTile.load_elems` / `store_elems`: an fp32 LDS tile addressed element
  by element, with `ds` operations (not FLAT, which would also count in vmcnt).
- Scoped `load_elems` / `store_elems`; `store_elems(runs=True)`, one store a
  run, fp32 to e4m3 packed four at a time; scoped register-tile loads take
  `bounded=n`.
- `hk.uniform(cond)` / `hk.if_(..., uniform=True)`: a branch declared
  workgroup-uniform may hold barriers, which is a persistent kernel's role
  dispatch; a divergent enclosing branch is still refused.

**gfx1201 (RDNA4) on hardware** (#70, #71). gfx1201 goes from compile-only to
hk's GPU tier passing on a Radeon AI PRO R9700 (914 passed, 0 failed;
[chapter 47](docs/history/47-gfx1201-on-hardware.md)):
- HipKittens moves to `9d182f93`: `include/rdna4` with gfx12's accumulator
  layout as measured (every MMA unit test had failed on the old one) and
  rdna3's work since the fork re-derived for gfx12. gfx1100 and gfx950 are
  unaffected: `include/rdna3` and the CDNA trees are byte-identical.
- The LDS reads a tile load issues come from the target: one on gfx1201, where
  hk assumed gfx1100's two. hk-lds-pipeline derives every LDS wait from that
  count, and a count too high makes every wait too shallow.
- The wait gate reads gfx12's waits (hex immediates, `s_wait_loadcnt_dscnt`,
  `s_wait_storecnt_dscnt`), the target is measured (VGPR granule 24, LDS pool
  128 KB, i8 WMMA, strips), and the schedules' register model is fitted on
  gfx1201's own builds.
- Its ceilings are measured on the R9700 (#74): bf16 / f16 WMMA 197 TFLOP/s,
  fp8 372, iu8 374 TOP/s; an add at 1.81 TB/s inside the 64 MB Infinity Cache,
  0.62 TB/s past it. `hk.check`, `hk.profile`, `hk.tune` and the scoreboard
  use them (they fell back to gfx1100's, which would have called hk's own 4096³
  GEMM faster than the card can), and `hk.tune` has a GEMM space there.
- The scoreboard on gfx1201, out of the box (gfx1100's plans), every arm in one
  process, with FlyDSL and AITER arms (#74,
  [chapter 48](docs/history/48-gfx1201-ceilings-and-first-scoreboard.md),
  [performance](docs/performance.md#part-c-rdna4-gfx1201-one-radeon-ai-pro-r9700)):
  hk's GEMM is the fastest at the five shapes (1.03-1.07x the best other arm).
  It loses attention to AITER's (0.83x causal, 0.97x full, at 4096 tokens,
  GQA 4) and fused add + rms_norm at 8192 rows (0.93x).
- No engine route is validated on gfx1201, so none is on by default there, and
  nothing is tuned there yet (its tables are gfx1100's).

**K1 (MoE a8w4) on gfx950, measured for this release** on the release's code
(after #85): real DeepSeek-V4.1-Flash weights, two layers, real, gate-only and
uniform routing, weights rotated past the caches, interleaved rounds in one
process; hk deterministic and within AITER's own error against an exact fp32
MoE everywhere.

| M | hk / AITER (tuned FlyDSL) | K1 now / K1 at #65 | K1 now / K1 before #65 |
|---|---|---|---|
| 64 | 0.77-0.84x | 1.26-1.45x | 1.12-1.20x |
| 128-512 | 0.74-0.85x | 1.05-1.30x | 1.04-1.31x |
| 2048 | 0.97-0.99x | 1.08-1.15x | |
| 4096 | 1.13-1.20x | 1.10-1.12x | |
| 8192 (routed) | 1.27-1.35x | 1.09-1.12x | 1.08-1.12x |
| 16384 (routed) | 1.16-1.23x | 1.04-1.06x | |

(The "K1 at #65" figures at M 2048, 4096 and 16384 compare each commit in its
own process, each normalised by its own in-process AITER; the other columns are
one process. The two methods agree within about 3%.)

- vLLM's K1 route takes M >= 8192, where hk wins at 12 of 12 points (before
  #85: 1.11-1.23x). Below that the route leaves the MoE to AITER, so decode in
  production is AITER's either way.
- At decode M (64-512) hk's K1 is still slower than AITER's tuned FlyDSL
  kernels, 0.74-0.85x, up from 0.53-0.78x at #65.

**Build gates**
- **The spill gate passes a scratch frame the code never touches** (#68). A
  kernel with ScratchSize > 0 and no VGPR spill builds when its code object has
  no `scratch_*` instruction, no `flat_*` memory instruction, no flat scratch
  and no private segment buffer (`cocheck.unused_scratch_frame`); the build
  records `scratch_note = "reserved-unused scratch frame"` and says so on
  stderr and in `build.log`. Found on K1's indexer build under ROCm 10 (36 B
  reserved, never used). A VGPR spill, any scratch access, and a code object
  that cannot be read are still refused.

**Fixes**
- **K1 MoE a8w4 at decode M ran slower after #65** (#85; M 64: 0.82-0.89x of
  K1 before #65, found by the MI355X scoreboard). The sorted kernels' grids are
  sized for the most sorted blocks a launch can have, and #65's
  `Placement.chunked(0, "spread")` dealt contiguous ranges of the whole grid, so
  the live blocks (235 of 390 at M 64) landed on the first five XCDs. They now
  renumber in the body over the live blocks only
  (`hk.placement.renumber(..., live=)`, new). Outputs are bitwise the same, and
  every M got faster (numbers in *K1 on gfx950* below).
- **gfx950: fp32 register tiles through LDS were FLAT loads and stores**
  (#59). `store(st<float>, rt<float>)` / `load(rt<float>, st<float>)` addressed
  the tile through a generic pointer, so every access was FLAT: slower to
  issue than `ds_read` / `ds_write`, and counted in vmcnt as well as lgkmcnt,
  so each wait for an LDS value also waited for every global load in flight.
  They are `ds_read` / `ds_write` now. Results are unchanged; affected are the
  cross-warp combines of fp32 partials (`hk.ops.decode_gemm`, `w4a16`, `moe`)
  and user kernels that move fp32 tiles through LDS on gfx950.

**Wheels** (#66)
- cp312 is one `manylinux_2_35` wheel, built on glibc 2.35 in vLLM's ROCm
  image, like cp310. Through 0.6.0 cp312 was built on Ubuntu 24.04 and needed
  glibc 2.38, so `pip install hk-dsl` found no cp312 wheel in vLLM's ROCm
  images; 0.7.0 added a second, `manylinux_2_35` cp312 wheel. (0.7.0's PyPI
  description still says CPython 3.12 needs glibc 2.38: that page cannot be
  edited after upload, and 0.8.0's states the wheels above.)
- `scripts/check_wheel_glibc.py` fails a wheel that needs a newer glibc (or
  GLIBCXX / CXXABI, with auditwheel) than its manylinux tag; CI's IR leg
  builds the cp310 wheel and runs it.
- Reproducible wheels: no build path in the binaries (`-ffile-prefix-map`,
  RUNPATH `$ORIGIN` at link time), entry times from `SOURCE_DATE_EPOCH`
  (the last commit's), fixed modes; the same commit builds byte-identical
  wheels ([building](docs/building.md#reproducible-wheels)).

**Docs**
- The API reference marks names against the 0.7.0 wheel (it marked them
  against 0.6.0, so `placement=` and `hk.arch.xcd_count` read as if 0.7.0 had
  them), and documents `hk.Placement`, `hk.Digit` and
  `hk.runtime.SymmetricBuffer`.
- [The language](docs/language.md) gains `hk.comm`, workgroup placement, the
  stated strip-reduction order (#52, #57), the bounded group copies on gfx950
  (#49) and LDS allocated in a runtime branch (#56).
- The docs site builds with `mkdocs build --strict` in CI: every page is in
  the nav (hk.profile, schedules, hk.tune and the end-to-end loop were not),
  heading ids follow GitHub's (so `#which-xcd-runs-which-workgroup---placement`
  resolves on both), and the links #54 broke (`#turning-them-on`) point at
  `#on-by-default-and-turning-it-off`.
- The install pages state 0.7.0's three wheels and that CPython 3.12 needs
  glibc 2.35, not 2.38; [the runtime](docs/runtime.md#which-gpu) has the arch
  detection order 0.7.0 introduced.
- [Releasing](docs/releasing.md): the release steps, and the documentation
  sync that must pass before a tag.
- Examples [14](examples/README.md) (an hk.comm task graph, gfx950) and
  [15](examples/README.md) (workgroup placement).
- `hk.map_cpp` and i32 strips in the language and the API reference (#81).
- No internal address, host name or user path in the repository (#83): docs,
  bench records and scripts use placeholders and environment variables, and
  `scripts/check_public_hygiene.py` (generic rules, an explicit allowlist)
  runs in CI's IR leg and before every tag.
- [docs/design/e2e-goal.md](docs/design/e2e-goal.md), the end-to-end goal
  refocused on three core kernels, and [docs/design/kda.md](docs/design/kda.md)
  (#20); bench/tune's record of T4 run 20261024-a (#22).

**CI and tests**
- Several gfx950 runners in parallel, each with its own GPUs and state, and
  GPU locks shared with manual measurements (#60,
  [testing](docs/testing.md#several-gfx950-runners)).
- A cancelled GPU job no longer leaves its container on the GPU (#69,
  [testing](docs/testing.md#the-gpu-tier)).
- Tests of gfx1100's measured data are pinned to gfx1100
  (`@pytest.mark.arch`, #64).
- The fenced paged-attention checks (`test_fenced.py`) build their reference
  on the host from one readback of each tensor, read twice and compared, and a
  relative-error failure prints both maxima. The one intermittent paged failure
  seen on gfx950, `relative error 9.73e+29`, matches an all-zero reference
  (0.97265625 / 1e-30), not a wrong output: the paged decode did not fail in
  thousands of repetitions with poisoned allocations and polluted LDS (#89;
  tests only, no kernel change).

**Compatibility**
- No public DSL name or parameter was removed or renamed since 0.7.0 (checked
  name by name against the 0.7.0 wheel). The modules' `NUM_XCDS` constants in
  `hk.ops.attn_cdna`, `gemm_cdna`, `gemm_cdna_dp` and `moe_a8w4` are gone
  (#65): the XCD count is `hk.arch.xcd_count()`.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-6`. 0.7.0's cached builds are rebuilt once.
- The spill gate can build a kernel that 0.7.0 refused: one whose only
  scratch is a reserved frame its code never touches (#68).
- gfx950's attention, GEMMs, K1 and K2 now number their workgroups through
  the placement pass: outputs bitwise as before. `HK_XCDS=1` turns every
  placement off.
- cp312 has one wheel, `manylinux_2_35`; 0.7.0's cp312 `manylinux_2_38` wheel
  has no 0.8.0 counterpart, and pip on a newer glibc takes the 2_35 one.

**Known issues**
- A GPU-tier CI run on MI355X (gfx950) once reported a mismatch in the
  fenced-allocator test of the paged-decode split path
  (`tests/hk/gpu/test_fenced.py::test_paged_decode_and_prefill`, fused decode,
  head_dim 64, 32 splits). It did not recur in later runs or in several
  thousand targeted repetitions with poisoned allocations and polluted LDS; the
  reported error value matches an all-zero host-side reference rather than a
  wrong kernel output. The paged decode is used only by the opt-in vLLM
  attention backend (`HK_VLLM_BACKEND=1`, off by default) and is not reached by
  any default-on route. The test has since been hardened (#89, tests only): it
  builds its reference on the host from one stable readback of each tensor, so
  a recurrence names its cause.
- gfx942: the synchronous group copies are still unbounded (#49): on gfx942 a
  `g.load` / `g.store` at a partial last tile reads and writes past the
  tensor. There is no gfx942 here to test the bounded form on.
- The gfx1100 defaults were not run end to end. Each gfx1100 route's evidence
  is its earlier opt-in run with real models; the default-on configuration
  itself was checked by unit tests, the GPU tier and the engine contract.
- The vLLM decoder attention backend stays off by default: it replaces ROCm's
  default decoder attention for every model and raises rather than falls back
  on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and other head
  sizes; validated on Qwen3-8B only, and on gfx950 slower at one request on
  long contexts (0.71-0.88x).
- SGLang-Diffusion installs made before 0.7.0 stay opt-in until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  spills and the spill gate refuses it (rocm/pytorch 7.2.3 builds it). There,
  `hk.ops.gemm.matmul`'s gfx950 route picks it for N 28672 / 14336, K 4096 at
  M 192-512 and raises `ResourceError` instead of falling back. vLLM's linear
  route (`HK_VLLM_GEMM`) is off on gfx950, so no default route reaches it.
- gfx1201 runs hk's GPU tier on a dev box, not in CI's required tiers; no
  route is validated or on by default there and nothing is tuned. gfx942 is
  compile-only.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

## 0.7.0

hk-dsl 0.7.0: hk is on by default in vLLM and SGLang, for the routes validated
with real weights on your GPU; `HK_DISABLE=1` turns it off. Also `hk.comm` for
hand-offs between gfx950's XCDs and between GPUs, and fixes for two silent
errors that 0.6.0's DSL has in user kernels: on gfx950 a synchronous `g.store`
at a partial last tile overwrites the memory after the tensor, and LDS
allocated inside a runtime branch reads back as 0. No shipped op or route was
affected by either.

On PyPI: `pip install hk-dsl` (CPython 3.10 or 3.12 on glibc 2.35+, Linux
x86_64). Three wheels: cp310 `manylinux_2_35`, and cp312 both
`manylinux_2_35` and `manylinux_2_38`; pip picks the newest tag the system's
glibc takes. The cp312 `manylinux_2_35` wheel was added on release day, for
glibc 2.35 systems such as vLLM's ROCm images (Ubuntu 22.04), where the
`manylinux_2_38` one does not install. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed (and a ROCm build of torch to launch).

    pip install hk_dsl-0.7.0-cp312-cp312-manylinux_2_35_x86_64.whl

(This paragraph first said "3.12 on glibc 2.38+", as do 0.7.0's PyPI page and
the GitHub release notes; the `manylinux_2_35` cp312 wheel makes 2.35 enough.)

**hk is on by default** (#54). With hk-dsl installed, its vLLM and SGLang
plugins (and the SGLang-Diffusion boot) apply every route that was validated
with real weights on your GPU: outputs bit-identical to the engine's, or error
within the engine's own noise. Before, every route needed its own `HK_*=1`.
Turning hk off is now for comparison experiments:

```bash
HK_DISABLE=1 vllm serve <model>        # stock: nothing patched, in every engine
HK_VLLM_MXFP8=0 vllm serve <model>     # one route off (HK_<ROUTE>=0)
python3 -m hk.integration --status     # every route on or off, and which rule decided
```

- **On by default** (the evidence for each route is in
  [integration](docs/integration.md#on-by-default-and-turning-it-off)):

  | GPU | engine | routes (switch) |
  |---|---|---|
  | gfx950 (MI355X) | vLLM | K2 MXFP8 linear for DeepSeek-V4.1 at M >= 2048 (`HK_VLLM_MXFP8`); K1 MoE a8w4 for DeepSeek-V4.1 at M >= 8192 (`HK_VLLM_MOE_A8W4`); AWQ 4-bit (`HK_VLLM_INT4`) |
  | gfx1100 (W7900D) | vLLM | unquantized linear (`HK_VLLM_GEMM`); bf16 MoE (`HK_VLLM_MOE`); AWQ 4-bit (`HK_VLLM_INT4`); the encoder SDPA (`HK_VLLM_SDPA`) |
  | gfx1100 | SGLang | linear (`HK_SGLANG_LINEAR`); MoE and the gfx1100 top-k fix (`HK_SGLANG_MOE`) |
  | gfx1100 | SGLang-Diffusion | DiT attention (`HK_SGLD_ATTN`); DiT linear (`HK_SGLD_LINEAR`) |
  | gfx1100 | `hk.ops.sdpa` | K3 DiT attention (`HK_SDPA_K3`) |

  On a GPU a route was not validated on, it stays off: the gfx1100 routes on
  gfx950 (vLLM's linear, MoE and encoder SDPA among them), and every route on
  gfx942 and gfx1201.
- **Stays opt-in:**
  - **ConvRot INT8** (`HK_SGLD_CONVROT=1`, or `HK_SGLD=attn,linear,convrot`):
    it is lossy by design (PSNR 30.4 dB from bf16), a model-level choice and
    not a toolchain default.
  - **The vLLM decoder attention backend** (`HK_VLLM_BACKEND=1`): see the
    known issues.
  - **`HK_MXFP8_ANY`**, K2 at shapes it did not measure (no real-weight run).
- **Turning routes off.** The first rule that applies decides:
  1. `HK_DISABLE=1` turns every route off, in every engine. Nothing beats it.
  2. An engine's switch at `0` (`HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0`,
     `HK_SGLD=0`) turns off that engine's routes.
  3. A route's own switch decides that route: `HK_<ROUTE>=0` off, `=1` on (also
     a default-off route, or a GPU it was not validated on).
  4. The legacy switches still mean what they meant: `HK_SGLD=attn,linear,convrot`
     turns on exactly those parts, `HK_SGLD=1` attention and linear,
     `HK_PATCH_VLLM=1` also the encoder SDPA; `HK_SGLANG_MOE=topk` and
     `HK_VLLM_BACKEND=<slot>` work as before.
  5. Otherwise the default applies.

  `0`, `off`, `no` and `false` are off; an empty value counts as unset.
  `python3 -m hk.integration --status` prints every route, whether it is on,
  and which rule decided.
- **The engine contract is checked at startup** (`hk.integration.contract`).
  Before it patches anything, a plugin checks every route it is about to apply
  against the installed vLLM / AITER / SGLang: each patched or fallback entry
  point exists, with the parameters the route uses. If any check fails, for
  example after a version change, the plugin patches **nothing** in that
  engine, prints one warning line, and the engine runs fully stock:

  ```
  [hk] WARNING: vllm: engine contract FAILED -- hk patched NOTHING in this engine, it runs stock (...)
  ```

  It is all or nothing per engine, never half-patched: a route that raises
  while it is applied undoes the ones applied before it, with the same line.
- **The first JIT compile says so.** A route's kernels compile on first use,
  seconds each, in every worker. If they were not pre-built, the first
  compile in each process prints one line that suggests
  `python3 -m hk.integration warm-routes` (and `--warm` for attention), run
  with the server's `HK_CACHE_DIR` before the server starts.
- **GPU detection reads the KFD topology.** The vLLM ROCm image sets
  `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`. Without `HK_ARCH`, hk took the
  list's first entry, so it read an MI355X as gfx90a, built for gfx1100, and
  with default-on it would have left every gfx950 route off.
  `hk.runtime.arch.detect()` now goes `HK_ARCH`, then a variable naming exactly
  one arch, then the KFD topology in sysfs (no HIP context), then torch and
  rocminfo, and uses a list's first entry only as the last resort.
- **Comparison scripts must say stock.** Unset no longer means stock, so an
  A/B's stock arm sets `HK_DISABLE=1`. `python3 -m hk.integration ab` /
  `headroom` and `tools/hk-bench/` do this themselves.
- **SGLang-Diffusion:** an install made before 0.7.0 keeps its old boot, which
  needs `HK_SGLD=1`. Run `python3 -m hk.integration sglang-diffusion install`
  again for the defaults; `... sglang-diffusion status` says when the boot is
  the old one.
- **Fixed:** with several routes on in one vLLM (int4 and MXFP8 on gfx950), its
  compile cache root was nested again in every worker
  (`hk-int4/hk-mxfp8/hk-int4/...`). It is now one directory named for all the
  routes (`hk-int4+mxfp8`), the same in every process.
- `hk.ops.sdpa.reset_stats()` also resets K3's counters.

**Fixes**
- **gfx950: synchronous `g.load` / `g.store` went out of bounds at a partial
  last tile** (#49). This affects 0.6.0's DSL. In 0.6.0 on gfx950, in a
  user-written kernel whose synchronous `g.load` covers a partial last tile
  (the tile's rows run past the tensor's last row):
  - the load reads up to a tile's rows past the tensor: an intermittent memory
    fault (10 of 10 fenced cases faulted, at M = 1, 63, 65, 130 and 1000);
  - its `g.store` **silently overwrites the memory after the tensor**: with
    slack after the output, a probe found 63 of 63 rows past it overwritten,
    with no error.

  No shipped `hk.ops` op and no vLLM / SGLang route was affected, in 0.6.0 or
  in this release: all run clean under an end-fenced allocator on gfx950 and
  gfx1100, and no shipped kernel issues a `g.store`. The fix: on CDNA, the
  synchronous `g.load` stops at the tensor's last row (rows past it read zero),
  and `g.store` drops the rows past it (`group_load_sync`, `group_store_bounded`
  in `include/hk/cdna.cuh`). `hk.ops.mxfp8.gemm`, the one shipped CDNA kernel
  with a synchronous group copy, is bit-identical and within 0.6% of its speed.
  gfx942 still uses the unbounded copies (no hardware to test). RDNA3 already
  bounded them. `tests/hk/gpu/test_fenced.py` now runs every `hk.ops` family at
  ragged shapes under the end-fenced allocator, in CI on gfx950 and gfx1100.
- **LDS allocated inside a runtime branch was not part of the launch's LDS**
  (#56). The `hk-lds-alloc` pass sized a kernel's launch from the
  allocations in the kernel's own block only, while the generated C++ bumps
  its allocator along the path the kernel takes. So a tile that a kernel
  allocated with `hk.lds.alloc` inside a runtime `if` (a branch on a device
  value) lay past the end of the LDS the launch requested: its reads returned
  0 and its writes were dropped, with no error. An allocation inside a runtime
  loop was past the end the same way. Found porting DeepSeek-V4.1's K1 to hk,
  where a reduction tile allocated in a role branch read back as zeros (87.5%
  of q wrong). Every released version through 0.6.0 has the pass in that form,
  so a user kernel that allocates LDS in a runtime branch or loop is affected
  there. No shipped kernel is: a scan of every kernel the IR and codegen tiers
  translate (both arches) and the gfx950 GPU suite builds found no LDS
  allocation outside a kernel's own block, in 0.6.0 or in this release. The
  pass now walks every region: the two arms of an `if` start at the same
  offset and the larger one continues, and an allocation inside a runtime
  `for` / `while` is refused, with a message to allocate it before the loop.
  `tests/hk/gpu/test_lds_branch.py` reads such a tile back across lanes (all
  zeros before the fix, exact after).
- **`mxfp8.quant` read up to 63 rows past the activation** (#47). At
  M % 64 != 0, K2's fused quant (#41, after 0.6.0) copied whole 64-row tiles of
  x. When x ended an allocator segment, that read took a GPU memory fault in
  vLLM with `HK_VLLM_MXFP8` on: the first three real-weight DeepSeek-V4.1 runs
  with K1 and K2 on faulted, and since it depends on the allocator's layout,
  later runs of the same arm completed. It only ever affected main after
  0.6.0; the released 0.6.0 does
  not contain `mxfp8.quant`. The last block's copy now stops at row M, with the
  output unchanged: 48 of 48 ragged-M cases are bitwise equal to vLLM's quant +
  GEMM. The end-fenced allocator test (`test_fenced.py`, which #49 extended)
  came with it.

- **A stated strip-reduction order was not the documented one for products**
  (#57). With #52's `lanes=` / `xor=`, `(x * x).sum(axis=1, lanes="seq",
  xor=...)` compiled to fused multiply-adds: HIP's FMA contraction fused each
  product into the running sum across hk's inlined ops, so the sum was not the
  documented one of rounded products (4288 of 32768 lanes differed from it on
  MI355X). Only an ordered reduction of products was affected, and no `hk.ops`
  kernel uses one. #52 came after 0.6.0, so 0.6.0 is not affected. The ordered
  reduction now keeps each summand's rounding (no fma reaches in), and
  `sum(..., times=y)` is the fused form, stated.

- **Tests: `test_sdpa` compares against fp32 attention on torch's math
  backend** (#61), test-only: no kernel or library change. Its intermittent
  head_dim-64 failure on a shared MI355X was torch's reference, not hk: in the
  vLLM ROCm 10 image, torch's fused bf16 SDPA (AOTriton 0.13.50) gave wrong,
  nondeterministic tiles under same-GPU load, while hk's kernel gave identical
  bits on every call.

**New in the language: hk.comm (gfx950)** -- hand-offs between the XCDs of
one GPU and between GPUs, each lowered to what a memory-model litmus suite
(20000 rounds a combination, across XCDs and across two MI355X) found correct.
The memory model is in [docs/design/comm.md](docs/design/comm.md).
- **Scoped tensors, signal / wait, symmetric memory** (#38):
  - `hk.Tensor[dtype, "device" | "system"]`: every load, store and scalar
    atomic through it carries the scope's cache-policy bits (on gfx950 both
    sides of a hand-off need them, so the scope belongs to the tensor);
  - `hk.comm.signal` (drain, barrier, one scoped flag store) and
    `hk.comm.wait` (a scoped poll with a timeout that sets an error element,
    or traps, instead of hanging);
  - `hk.comm.arrive` / `hk.comm.barrier`: an epoch'd grid barrier;
  - `hk.kernel(..., resident=True)` and `Kernel.resident_capacity()`: a grid
    that cannot all be resident raises `ResourceError` instead of deadlocking;
  - `hk.runtime.SymmetricBuffer`: one buffer per rank with every peer's mapped;
    while one is open, a launch refuses it bound to an unscoped parameter.
- **Every scalar fetch-op, and scoped col_vecs** (#45): `View.atomic_min_scalar`,
  `atomic_or_scalar`, `atomic_and_scalar`, `atomic_xor_scalar` and
  `atomic_exch_scalar` beside `atomic_add_scalar` / `atomic_max_scalar`, each
  returning the old value, plain or at a scoped tensor's scope; a row tile's 32-bit `col_vec` loaded from or
  stored to a scoped tensor (a per-row scale next to its tile).
- **Work queues and per-XCD aggregation** (#48): `hk.comm.claim` (a work
  queue's take: one scoped atomic add, no drain) for persistent kernels and
  task graphs; `hk.arch.xcd_id()`, read from the hardware; and
  `hk.comm.xcd_barrier`, a grid barrier with one global atomic per XCD a round
  (2.53 us against 3.71 us flat at 256 workgroups). The tile-signal pattern
  through a double-buffered symmetric buffer is tested across two GPUs.
- **Tagged mailboxes** (#50): `hk.comm.post` / `hk.comm.take` send a one-int32
  message and its tag as one 8-byte scoped store and one poll, no drain: a
  1.51 us round trip between two MI355X, against 3.40 us for a tile with
  signal / wait. A mailbox orders nothing else.

**New in the language: strips**
- **A stated row-reduction order** (#52, #57, gfx950): `t.sum(axis=1,
  lanes="seq" | "tree", xor=(32, 16, 8, 4, 2, 1))`, and the same for `max`,
  `min` and `prod`, for a result that must match another kernel's bit for bit.
  The guaranteed order:
  - a lane's own values are combined left to right (`"seq"`) or as a balanced
    tree (`"tree"`), then the lanes over the given xor offsets in that order,
    each of 1, 2, 4, 8, 16 and 32 once;
  - the strip's values are summed as they are: `(x * y).sum(...)` adds the
    rounded products, and the compiler does not fuse them into the sum;
  - `x.sum(axis=1, lanes="seq", xor=..., times=y)` is the fused form, spelled
    with fma: `s = fma(x0, y0, r(x1 * y1))`, then `s = fma(xi, yi, s)`, then the
    xor sums. That is what a compiled `s += a * b` loop computes, and it is
    dsv41_mono's and aiter's sum of squares in their norms (`times=x`).

  Without the arguments nothing changes (a tree, then 1 .. 32). Refused on
  wave32 targets, on columns and on WMMA tiles.

**Kernels and serving**
- **gfx950 attention: a spatial workgroup numbering, routed by measurement**
  (#55). `attn_cdna` (dense flash attention and paged prefill) gains
  `mapping="spatial"`: each XCD takes a contiguous range of (batch, KV head)
  clusters, so the workgroups resident on an XCD read one K/V stream. Outputs
  are bitwise the default numbering's. `hk.ops.attn` and the SDPA ops pick it
  when an XCD would otherwise hold five or more KV heads' streams, except for
  causal grids with fewer than twelve rounds of workgroups an XCD. Against the
  default, MHA at D 128 and 8K-128K tokens: 1.05-1.18x at 128 KV heads, up to
  1.08x at 64 and 1.05x at 40 (at 128 heads, 32K, the L2 hit rate goes from
  0.47 to 0.96). The routed choice was never below the default at the 74 timed
  points. The paged prefill accepts the option but is not
  routed (unmeasured).
- **K2's activation quant and its scale layout in one launch** (#41):
  `hk.ops.mxfp8.quant(x, packed=True)`, and `hk.mx_quant_store` writes the
  packed scale layout. The vLLM route makes one opaque call
  (`hk_vllm::mxfp8_linear`) instead of vLLM's quant, hk's scale packing and the
  GEMM: 1.01-1.58x per call at the 12 DeepSeek-V4.1 TP2 shapes, bit-identical.
- **Serving readiness** (#43, #44): the engine-contract test, a gfx950 GPU tier
  in the vLLM deployment image (ROCm 10), and `python3 -m hk.integration
  warm-routes --tp 2`, which builds the DeepSeek-V4.1 routes' kernels before
  the engine starts (31.2 s on an empty cache, 0.9 s warm).
  [docs/design/serving-readiness.md](docs/design/serving-readiness.md) lists what
  0.6 lacked for a deployment; [docs/design/comm.md](docs/design/comm.md) has
  hk.comm's roadmap (#51); [docs/design/placement-pass.md](docs/design/placement-pass.md)
  designs workgroup placement as a compiler pass (#62, design only).
- CI: pushes run only on main, and a PR's superseded runs are cancelled (#42);
  GPU-tier runs write no core dumps (#46).

**Compatibility**
- **Unset now means on.** With hk-dsl installed, vLLM and SGLang (and
  SGLang-Diffusion, after `install`) run with hk's validated routes unless
  `HK_DISABLE=1` or a route's switch is `0`. Every existing `HK_*=1` still
  works.
- No public DSL name or parameter was removed or renamed since 0.6.0 (checked
  name by name against the 0.6.0 wheel). Changed signatures only gained
  parameters: the strip reductions' `lanes=` and `xor=` (and `sum`'s
  `times=`), and `hk.kernel`'s `resident=`. One integration op was
  replaced: vLLM's MXFP8 route calls `hk_vllm::mxfp8_linear`, which quantizes
  inside, instead of `hk_vllm::mxfp8_block32` (#41).
- On gfx950, a synchronous `g.load` now reads zero for a tile's rows past the
  tensor, and `g.store` no longer writes them. In-bounds rows are unchanged.
- An LDS allocation inside a runtime loop is now refused at build time (it
  was silently past the launch's LDS); allocate it before the loop. One inside
  a runtime branch now counts toward the launch's LDS, so such a kernel asks
  for more LDS than 0.6.0's did.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-5`. 0.6.0's cached builds are rebuilt once.

**Known issues**
- **gfx942: the synchronous group copies are still unbounded** (#49). There
  is no gfx942 here to test the bounded form on, so on gfx942 a `g.load` /
  `g.store` at a partial last tile still reads and writes past the tensor.
- **The gfx1100 defaults were not run end to end.** Each
  gfx1100 route's evidence is its earlier opt-in run with real models. The
  default-on configuration itself was checked by the unit tests, the gfx1100
  GPU tier and the engine contract in each engine's environment, not by a
  model run on a W7900D with nothing set. The real-weight default-on smoke ran
  on MI355X (DeepSeek-V4.1-Flash, TP2).
- **The vLLM decoder attention backend stays off by default.** It replaces
  ROCm's default decoder attention for every model, and it raises rather than
  falls back on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and
  other head sizes. It was validated on Qwen3-8B only, and on gfx950 it is
  slower at one request on long contexts (0.71-0.88x).
- **SGLang-Diffusion installs made before 0.7.0 stay opt-in** until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  at 130x1792x768 spills and is refused. rocm/pytorch 7.2.3 builds it.
- gfx1201 and gfx942 are compile-only, and no route is on by default there.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

See docs/api (marks "New after 0.6.0"), [integration](docs/integration.md),
[docs/design/comm.md](docs/design/comm.md) and
[docs/testing.md](docs/testing.md).
