# Design: workgroup placement as a pass (`hk-placement`)

Status: M0 and M1 implemented (`hk.dsl.placement`; attention and the bf16 GEMMs on it); M2
(K1, K2, stream-K) next. The owner decided on 2026-10-26 to build it, in the milestones of
"Merging with what exists" below, one PR on the latest main each. As built, `Placement.chunked`
renumbers block_idx.x alone, in place (a GEMM's y is its split-K share), and a cluster
placement flattens the grid. The evidence is the
attention experiment of 2026-10-26 (PR #55), described in
`~/hk-dx/research/2511.02132/report.md` §5.

## The problem

### How the hardware places workgroups

On gfx950, workgroup `w` runs on the XCD of `w % 8`. This holds in every wave of a grid, whatever
order the workgroups finish in: PR #55's test
`test_workgroups_are_dealt_to_the_xcds_by_id_mod_8_in_every_wave` checks it on 64Ki workgroups.

Each XCD has a private 4 MB L2. So the order in which a kernel numbers its work decides two things:

- how many times shared bytes are fetched, once per XCD that reads them;
- how many unrelated streams each L2 has to hold at once.

### What it costs (MI355X, PR #55)

gfx950's attention used HipKittens' numbering: a query block of every head before the next. At 128
KV heads that holds 16 K/V streams per XCD. Changing only the numbering, with bitwise-identical
outputs, gave:

| KV heads | gain |
|---|---|
| 128 | +11.7% to +17.5% (N 8K-128K) |
| 64 | +5.5% to +8.0% |
| 40 | +2% to +5% |

The cost is clock, not bandwidth. At 128 heads and 32K, block-first moves 149 GB past the L2s at
1.43 GHz. Spatial moves 9.7 GB at 1.64 GHz. The matrix core is 74% busy in both.

### The same idea, written by hand many times

hk.ops writes this numbering out by hand 13 times, with `NUM_XCDS = 8` hard-coded:

| where | form |
|---|---|
| `gemm_cdna.py:413` | chunked round robin (`xcd_chunk`); ids past the last whole round stay where they are; then `tile_of` (`wgm` L2 swizzle) |
| `gemm_cdna.py:662-701` | the stream-K plan on the host: a contiguous range of blocks per XCD |
| `gemm_cdna_dp.py:87` | chunked round robin, `xcd_chunk = 0` meaning an eighth of the grid |
| `mxfp8.py:192` (mx4w) and `mxfp8.py:396` (persistent) | chunked round robin + `wgm` |
| `moe_a8w4.py:110, 210, 365, 653, 903, 1025, 1243` | contiguous eighths, **dropped to identity when `total % 8 != 0`** |
| `moe_a8w4.py:776` | persistent stage 2: a contiguous chunk of units per XCD |
| `attn_cdna.py` `_spatial` (PR #55) | contiguous ranges, a bijection for any total |

Each copy is written once, for one kernel. A new kernel gets none of them unless its author knows to
add one. In the KDA MX experiment the Triton arm did; the hk arm did not.

### What this document proposes

One mechanism:

1. **an analysis** that finds the sharing structure from the IR;
2. **a rewrite** that installs one placement function at kernel entry;
3. **a policy** that decides whether and how.

Each part, its limits, and how to validate and migrate are below.

## 1. Analysis: who shares what

### The access model

After tracing, a kernel is one MLIR function. Every global read is one of these ops:

- `hk.load`, `hk.load_rows`, `hk.group_load`;
- `hk.gather_rows`, `hk.load_scalar`;
- `hk.load_mx_scales`, `hk.load_fp4_frags`, `hk.load_w4`.

Each takes a global (`!hk.gl`) and a coordinate (`hk.coord`: batch, depth, row, col). Every
coordinate is an SSA expression over these leaves:

| leaf | role |
|---|---|
| `hk.block_idx x/y/z` | grid axes |
| `hk.warp_id`, lane ids | within the workgroup |
| `scf.for` induction variables | the K / KV walk |
| `hk.extent` | shapes, runtime-uniform |
| constants and constexprs | |
| results of `hk.load_scalar` / gather indices | indirect |

The grid itself is a second function in the same module. It is built from `hk.extent` of the
arguments; the attention dump shows it next to the kernel.

### The analysis

For each read op `r`, walk its coordinate expression and project out the walk:

- **loop induction variables, warp and lane ids** are the dimensions the workgroup itself sweeps.
  They become universally quantified.
- What remains is the **sharing key** `key_r(block_idx)`, a symbolic function of the grid axes.
- Two workgroups read the same bytes through `r` exactly when their keys are equal. This is exact
  when the coordinate is a function of (key, walk) only, which is the case for affine and
  `div`/`mod` expressions over the block ids.

Each read also gets a byte weight `bytes_r` = tile bytes × trip count. Both are symbolic in extents
and constexprs; they are evaluated at launch, when the shapes are known.

### Examples

| kernel | read | key | key cardinality |
|---|---|---|---|
| attention (`attn_cdna`) | K and V (`group_load %k[(z, x % h_kv, t*64, 0)]`) | `(z, x % h_kv)` | B · H_KV |
| attention | Q | `(z, head(x), y)` | unique per workgroup |
| GEMM (`gemm_cdna`, `mxfp8`) | A | `row(block)` | m_blocks |
| GEMM | B | `col(block)` | n_blocks |
| decode GEMV, paged decode | weights / pages | unique per workgroup | — |
| MoE stage 1 (sorted) | x tile | `b = block // nblk` | sorted blocks |
| MoE stage 1 (sorted) | expert weights | `seids[b]` | experts used |

For the decode GEMV and the paged decode the result is "no sharing", so the pass leaves the kernel
alone.

The MoE weight key `seids[b]` goes through a `load_scalar`, so it is indirect; see below.

### Classification

For the grid, combine the reads by bytes:

- **cluster**: one key carries the dominant share of the bytes. In attention, K and V are read once
  per KV block of the walk, while Q is read once per workgroup. Place by that key.
- **raster**: two keys over disjoint axes with comparable bytes (GEMM's A row panel and B column
  panel). The sharing is two-dimensional. Place by grouping blocks (`wgm`) and by chunking per XCD.
- **none**: no key worth it (decode GEMV, elementwise, norms). Leave the kernel alone.

### What it cannot see, and the annotation

- **Indirect keys.** A key that goes through `hk.load_scalar` / `gather_rows` / `load_fp4_frags` row
  indices is data, not shape:
  - the paged block table (`block_table[req, pos]`);
  - MoE's `sorted_expert_ids[b]` and `sorted_ids`;
  - the paged prefill work list (`tiles[t]`).

  When the indirect value is itself a function of a grid axis (e.g. `seids[b]` with `b` from the
  block id), the pass can still cluster by that axis: consecutive `b` are runs of one expert,
  because the sort makes them so. That is a heuristic, and the pass reports it as one. Otherwise the
  user states the key.

- **Annotation.** `hk.kernel(..., placement=hk.Placement.cluster(key=lambda p, b: ...))` gives the key
  as a function of the decoded logical coordinates. `hk.Placement.raster(wgm=, chunk=)` and
  `hk.Placement.none()` are the other two. An explicit placement overrides the analysis; the pass
  still checks it (§5).

- **Persistent and queue-driven kernels.** The workgroup id is not the unit of work, so this
  analysis does not apply. They are handled in §4.

## 2. Rewrite: one decode at kernel entry

### What the pass does

The pass `hk-placement` runs after `hk-verify` and before `hk-fragments`.

1. **Flatten the grid.** The grid function becomes `(Gx·Gy·Gz, 1, 1)`. Every `hk.block_idx` use
   becomes a use of the logical `(x, y, z)` decoded at entry:

       w      = block_idx.x                       # flat
       u      = place(w, total, xcds, chunk)      # logical unit, a bijection
       x, y, z = order(u)                         # the logical numbering, cluster-major

   - `place` is the bijection from PR #55 (`attn_cdna._spatial` / `draft/place.py`). With `chunk = 0`,
     XCD `d` takes one contiguous range of units, and the first `total % xcds` XCDs take one more.
     With `chunk > 0`, units are dealt in rounds of `xcds · chunk`, and the remainder is dealt as with
     `chunk = 0` instead of staying in round-robin order. Loads per XCD differ by at most one, for
     any total.
   - `order` is cluster-major: the key's axes outermost, the remaining axes inside.

2. **Keep the kernel's own order inside a cluster.** LPT under causal (`qb = n − 1 − qb`) and the
   paged work list's longest-first order are kernel code applied to the decoded `y`, so they survive
   unchanged.

3. **Raster.** `order` becomes `tile_of(u, m_blocks, n_blocks, wgm)`, the existing `wgm` group walk,
   applied after `place(..., chunk)`. That is exactly today's GEMM path, now as one function.

4. **Take the XCD count and the dispatch model from the target.**
   - `TargetInfo` gets `xcds` (gfx950: 8; MI350P / DPX: 4; gfx942 MI300X: 8, MI300A: 6) and
     `dispatch = "round_robin"`.
   - At runtime, `hk.runtime` reads `hipDeviceAttributeNumberOfXccs` (10018, ROCm ≥ 7.0) once per
     device. If it disagrees with the build's constexpr, the kernel is rebuilt with the device's
     value; `xcds` is part of the kernel cache key, like `arch`.
   - `hk.arch.xcd_count()` exposes it to hand-written code.
   - gfx1100 and other single-L2 parts: `xcds = 1`, and the pass is a no-op.

### Merging with what exists (migration)

| step | change | must hold |
|---|---|---|
| M0 | the mechanism: `hk.grid.place` (the function), `hk.arch.xcd_count()` and the target's `xcds`; the analysis (sharing keys, bytes, classification) and the rewrite as an opt-in pass (`placement="auto"` or an explicit `hk.Placement`); `hk.check --placement` (the decoded grid covers the original exactly once) | IR property tests and the pass's analysis on the shipped kernels (§5) |
| M1 | attention and the GEMMs on it: `attn_cdna`'s hand-written `mapping` (PR #55) becomes `placement="auto"`, which must find the key `(z, x % h_kv)` and choose as `mapping_for` does on the 74 measured points or better; `gemm_cdna` (pingpong) and `gemm_cdna_dp` take `Placement.raster(wgm, chunk)` instead of their inline remap. `xcd_chunk` and `wgm` stay as schedule knobs and are passed through | bitwise equal to before; in-process interleaved A/B within ±1% on the GEMM route shapes; §5's attention regression |
| M2 | the rest of the hand-written remaps: K2 (`mxfp8` mx4w and persistent), K1 (`moe_a8w4`'s seven sorted / grid kernels and the persistent stage 2), the stream-K plan. moe_a8w4's `whole` fallback goes away because `place` is a bijection for any total | bitwise; in-process A/B within ±1% (K2 at the 18 DSV4.1 points, K1 at M 8192 and 16384 plus ragged M) |
| M3 (later) | `hk.schedules` gains a `placement` knob (`auto` / `none` / `cluster` / `raster(wgm, chunk)`) that hk.tune searches | tuned route tables unchanged or better |

The GEMM's `xcd_chunk = 0` ("an eighth of the grid") is `place(chunk = 0)`. A positive `xcd_chunk` is
`place(chunk = xcd_chunk)`. Both become the same code path.

## 3. Policy: when, and which

The choice never changes results, because `place` is a bijection and each workgroup computes the
same thing. That is why "on by default" is possible at all. But it can change time in both
directions, so the default needs a policy.

### Heuristic, in order

1. **Refuse** (keep the kernel's own numbering) when the kernel's correctness or progress depends on
   which workgroup runs when:
   - `resident=True`;
   - `hk.comm.wait` / `claim` / `xcd_barrier`;
   - spin loops on a global;
   - `wg_arrive` where the last arrival does the fix-up;
   - atomics into floating-point outputs (order changes the sum).

   Reordering can deadlock a grid that waits on later workgroups. The fix-up order is
   deterministic in hk, but it is tied to arrival. These kernels can still take an explicit placement
   if the author vouches for it.

2. **No key, or key cardinality ≤ xcds** (GQA with 8 KV heads, TP-sharded heads, a GEMM with ≤ 8
   row blocks): keep the default. Measured: GQA-8 and 16 heads were within ±2%.

3. **Streams per XCD.** Under the default numbering, count the distinct keys among the first
   `resident_per_xcd` workgroups of each XCD. This is static, from `place` and the occupancy the
   build gate already knows. Let `S` be that count.
   - Below 5: keep the default. Measured: 3-4 streams gained 0-3%, 2 streams ±2%.
   - From 5 up, and when `S · bytes_per_stream_window` exceeds the L2's share: cluster.

4. **Non-uniform work** (a trip count that depends on a grid axis, e.g. causal attention's `n_blk`
   from `qb`): estimate an XCD's tail as one longest unit over its work, using rounds =
   units / (xcds · resident). Keep the default when rounds < 12. Measured: −10% at 5 rounds, −2% at
   10, from +1% at 12.

5. **Raster kernels:** search `wgm` × `chunk` with hk.tune as today, with `placement` added as a knob.

The thresholds are the ones measured on attention. On other kernels they are hypotheses until §5's
checks confirm them.

### When it gets slower

| case | why | guard |
|---|---|---|
| short causal or otherwise non-uniform grids | an XCD's last cluster ends on its longest unit; the default's last round is short units | rule 4 (−10% measured without it) |
| uneven work across contiguous ranges (MoE tiles of different experts, ragged requests) | `place` balances unit counts, not work | treat a work list as non-uniform; prefer `chunk > 0` (interleaved rounds) when units differ in cost |
| fewer clusters than XCDs | contiguous ranges split clusters across XCDs anyway; nothing to win, some locality lost if the kernel relied on round-robin | rule 2 |
| interleaving several clusters per XCD | tried as a fix for causal grids. It did not fix them (0.92 and 0.88 at 40 heads, 8K, for 2 and 4 clusters). Suspected cause: windows that straddle XCD ranges, giving one XCD a window's long units and another its short ones | do not interleave across range boundaries; one cluster at a time |
| a kernel already balanced by its own order (global LPT) | the new order breaks it | rule 4; per-kernel opt-out |
| partitions / other parts | wrong XCD count, wrong groups | `xcds` from the device (§2) |
| profiling | hk.profile's per-kernel counters are unchanged, but the kernel name gains a placement suffix | documented |

## 4. The same framework, other shapes of work

- **GEMM L2 rasterization.** The analysis finds two keys, so it classifies the kernel as raster. The
  rewrite is today's `xcd_chunk` + `wgm` as one `Placement.raster`. Nothing new for performance; the
  gain is one implementation and the non-multiple-of-8 tail.

- **Persistent kernels.** A persistent kernel loops over units (`mxfp8` persistent, `moe_a8w4`
  persistent stage 2, the stream-K plan), so placement is the unit order each workgroup walks, not
  the workgroup id. The same `place` gives each XCD a contiguous, cluster-major range; this is what
  `moe_a8w4.py:776` and the stream-K plan already do by hand. The pass exposes it as
  `hk.grid.units(total, cluster=...)`, an iterator for the loop.

- **Megakernel queues (I6, comm roadmap R5).** Tasks are pushed to one queue per XCD. The rules carry
  over unchanged:
  - **push by cluster key:** tasks that read the same bytes go to one XCD's queue;
  - **pop cluster-major:** a queue drains one cluster before the next;
  - **steal whole clusters from the tail:** stealing single tasks splits clusters and re-fetches
    their bytes on the thief's XCD.

  The key comes from the task descriptor's annotation, since the analysis cannot see through the
  queue. This is a library rule for R5, not a compiler pass.

- **Per-XCD broadcast of activations (decode L5).** Here every workgroup reads the same bytes: the
  key is empty and the cardinality is 1. Placement cannot help, because the data is already shared.
  What helps is publishing one copy per XCD.
  - The analysis can **flag** it: a read with an empty key, a `"device"`/`"system"` scope, and a
    large fan-out is a candidate for `hk.comm`'s per-XCD publish.
  - The transform itself is a comm-level change. It needs the "XCD-local" visibility proof from the
    litmus suite: a leader's plain stores, the followers' sc0 loads after an `xcd`-scope flag. It
    stays out of this pass.

- **hk.profile.** The `l2_refetch` finding gains the suggestion "the analysis finds key K with S
  streams per XCD; `placement=auto` would cluster it". The data is the same as the pass's.

## 5. Validation

1. **The dispatch policy.** PR #55's GPU test, which fails if `w % 8` stops naming the XCD in any
   wave. It runs on every gfx950 GPU tier.
2. **Bijection and balance, statically.**
   - `place` and every `order` are pure integer functions, so the IR tier evaluates them in Python
     over every total in 0-300 plus large totals, xcds ∈ {1, 2, 4, 6, 8} and chunk ∈ {0, 1, 16, 32,
     64}. It checks a permutation, per-XCD loads within one, and one cluster per XCD at a time
     (PR #55 has the attention instance).
   - `hk.check` gains `--placement`: it decodes the grid of a concrete call on the host and checks
     the decoded coordinates cover the original grid exactly once.
3. **Bitwise.** Every kernel the pass touches must give `torch.equal` outputs with and without it, on
   the shapes of its GPU tests. A kernel that is not bitwise is in rule 1's refusal set by
   construction, and the check finds it.
4. **The attention regression.**
   - Run the pass with `placement="auto"` on `attn_cdna` with its hand-written `mapping` removed.
   - Over the 74 points of PR #55 (`runs/attn-mapping-*.json`), the auto choice must equal
     `attn_cdna.mapping_for` or measure at least as fast, in one process, interleaved.
   - The key it finds must be `(z, x % h_kv)`.
5. **No regression on migrated kernels.**
   - **K2** `mxfp8` mx4w at the 18 DSV4.1 points: bitwise, within ±1%.
   - **K1** `moe_a8w4` sorted stage 1/2 at M 8192, 16384 and ragged M: bitwise, within ±1%. Ragged M
     is where the old `whole` fallback turned placement off, so a gain is possible there (I5).

## Cost

| step | days | GPU |
|---|---|---|
| M0 `place`, `xcd_count`, analysis, rewrite pass, annotation, `hk.check --placement` | 4-6 | 10 min (the dispatch test, a smoke run) |
| M1 attention + GEMM on the pass | 1-2 | ~3 h of A/B on 1 MI355X |
| M2 K1, K2, stream-K | 2-3 | ~3 h of A/B |
| M3 schedules knob + hk.tune | 1-2 | tune runs as usual |

M2 pays for itself with or without the analysis (the K1 tail, the XCD count). The analysis's value
depends on how many new kernels will have sharing structure, and on whether the heuristic's
thresholds hold beyond attention -- which M1's attention regression checks first.
