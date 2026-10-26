# Design: workgroup placement as a pass (`hk.dsl.placement`)

Status: built (M0-M2, PRs #63 and #65). Attention, the bf16 GEMMs, K2 (`mxfp8`), K1
(`moe_a8w4`) and the stream-K plan use it. hk.ops has no hand-written XCD remap and no
`NUM_XCDS` left. M3, a `placement` knob in `hk.schedules` searched by hk.tune, is not built.

The user's view is [placement.md](../placement.md); the API is
[hk.placement](../api/hk.placement.md). This page is how it works and why it is built this way.
The evidence is the attention experiment of 2026-10-26 (PR #55), described in
`<workspace>/research/2511.02132/report.md` §5-6.

- [The problem](#the-problem)
- [1. Analysis: who shares what](#1-analysis-who-shares-what)
- [2. Rewrite: one decode at kernel entry](#2-rewrite-one-decode-at-kernel-entry)
- [3. Policy: when, and which](#3-policy-when-and-which)
- [4. The XCD count](#4-the-xcd-count)
- [5. Checking a placement: hk.check --placement](#5-checking-a-placement-hkcheck---placement)
- [6. Other shapes of work](#6-other-shapes-of-work)
- [7. What was migrated, and what it measured](#7-what-was-migrated-and-what-it-measured)
- [8. Limits, and what is not built](#8-limits-and-what-is-not-built)

## The problem

On gfx950, workgroup `w` runs on the XCD of `w % 8`. This holds in every wave of a grid, whatever
order the workgroups finish in. `tests/hk/gpu/test_comm.py::test_workgroups_are_dealt_to_the_xcds_by_id_mod_8_in_every_wave`
checks it on 64Ki workgroups finishing out of order.

Each XCD has a private 4 MB L2. The order in which a kernel numbers its work therefore decides two
things:

- how many times shared bytes are fetched (once per XCD that reads them);
- how many unrelated streams each L2 holds at once.

gfx950's attention used HipKittens' numbering: a query block of every head before the next. At 128
KV heads, that holds 16 K/V streams per XCD. Changing only the numbering made it 12-17% faster.
The cost is clock, not bandwidth: at 128 heads and 32K, block-first moves 149 GB past the L2s and
runs at 1.43 GHz, while one-cluster-per-XCD moves 9.7 GB and runs at 1.64 GHz. The matrix core is
74% busy in both.

Before this pass, hk.ops wrote such numberings out by hand 13 times, with `NUM_XCDS = 8` fixed:

- **GEMMs:** chunked round robin.
- **MoE:** contiguous eighths that fell back to the hardware's order whenever the grid was not a
  multiple of 8.
- **Attention:** a hand-written spatial order.

A new kernel got none of them unless its author knew to add one.

The pass makes it one mechanism:

1. an analysis of the traced kernel;
2. a rewrite of its entry and grid;
3. a policy for when to use it.

## 1. Analysis: who shares what

`hk.placement.analyze(module)` runs on the traced module before the hk passes. It is pure: the
module is not changed.

**Reads.** Every global read op is examined:

- `hk.load`, `load_rows`, `group_load`;
- `gather_rows`, `load_scalar`, `stage_load`;
- `load_mx_scales`, `load_fp4_frags(_lds)`, `load_w4`.

**Dependence of an address on the grid.** For each integer operand of a read (its `hk.coord` and
indices), a memoised walk over the SSA def chain computes:

- **digits**: which parts of the grid the value depends on. A digit is a grid axis whole (`x`), or
  `axis // r` or `axis % r` where the kernel divides a `hk.block_idx` directly by `r`. `r` must be
  free of the grid: a constant, a tensor extent (`hk.extent`) or arithmetic of those. It is kept as
  a host expression, e.g. `g.k.depth()`.
- **flags**:
  - `loop`: an `scf.for` induction variable or iter_arg. This is the dimension the workgroup sweeps
    itself, and is projected out.
  - `warp`: the warp or lane id. Projected out.
  - `indirect`: the value went through a `load_scalar` (an index read from memory).

**Sharing key.** What remains is the read's sharing key. Two workgroups read the same bytes through
that read exactly when their key digits are equal. Keys are normalised: an axis taken both as
`// r` and `% r` is the whole axis.

**Weight.** Each read is weighted by its tile's bytes times the trip counts of the loops around it.
A loop with a non-constant trip count counts as 16.

**Classification.** Reads are grouped by key:

- A read whose key holds every axis the kernel uses is the workgroup's own.
- A read with an empty key is a broadcast (every workgroup reads it).
- The heaviest remaining key is the cluster key if it carries at least a quarter of the bytes.
- If a second key over disjoint axes weighs at least a quarter of it, the kernel is a **raster**
  (a GEMM's A panel and B panel). The analysis reports raster but does not act on it; GEMMs say
  what they want explicitly (§2).

**Flags on the kernel.** Order-dependent ops are recorded: `wg_arrive`, `signal`, `arrive`,
`wait`, `post`, `take`, `atomic_add`, `atomic_scalar` and `scatter_add_lds`. A loop whose bounds
depend on the grid makes the kernel **non-uniform** (causal attention).

What it found on the shipped kernels:

| kernel | key, as the analysis prints it | class |
|---|---|---|
| `attn_cdna`, dense | `(z, x % g.k.depth())`: (batch, KV head), no annotation | cluster; non-uniform under causal |
| `attn_cdna`, paged | `(y, x % g.k_cache.depth())`, flagged indirect: the page goes through the block table, and `y` (the work list entry) is not the request. The kernel gives the key itself: `Digit.of("x", "mod", "k_cache.depth")` | cluster, by annotation |
| a copy or a GEMV | every read the workgroup's own | none |
| a test kernel with `atomic_max` | — | order-dependent |

## 2. Rewrite: one decode at kernel entry

`hk.placement.rewrite` runs before verification and the hk passes, so the passes see an ordinary
kernel.

**`Placement.cluster(key=None)`**

1. **The plan.** The key comes from the analysis, or from `key=` digits. Every axis the key splits
   (`x % r`) becomes two digits, `x // r` and `x % r`. The logical order lists the key's digits
   first, then the rest. Each group keeps the original significance (z, y, x; `//` before `%`), so
   the kernel's own order inside a cluster is unchanged; causal attention's longest-first query
   blocks stay longest-first.
2. **The grid** becomes one axis of `Gx·Gy·Gz` workgroups, in both the kernel's grid function and
   the host's launch expression.
3. **The decode at entry.** A new `hk.block_idx x` (`pl_wg`) is mapped through
   `place(w, total, xcds)`. XCD `d = w % xcds` takes one contiguous range of the logical units; the
   first `total % xcds` XCDs take one more. The unit is cut into its digits in plan order (mixed
   radix), and each axis is put back together (`x = (x // r)·r + x % r`).
4. **The replacement.** The three values replace every use of the old `hk.block_idx` ops, which are
   erased. The body then computes on the logical `(x, y, z)` as before.

Every emitted value is named `pl_*`, so the generated C++ reads as what it is. The three results
carry an `hk.placement_axis` attribute, so the checker (§5) can find what was emitted.

**`Placement.chunked(chunk, tail)`** renumbers `block_idx.x` alone, in place: `x' = place(x, Gx,
xcds, chunk, tail)`. The grid is unchanged. This is the GEMM form: its `y` is the split-K share,
and the old hand-written remaps also renumbered `x` within each share.

**`place`** is written once, over an arithmetic backend:

- emitted as `arith` ops by the rewrite;
- evaluated on host ints by checks and tests;
- run on device ints inside a kernel body (`hk.placement.renumber`, `xcd_range`, for persistent
  kernels).

| chunk | tail | what XCD `d` gets |
|---|---|---|
| 0 | — | one contiguous range; loads differ by at most one |
| c > 0 | `"stay"` | rounds of `xcds·c` ids, `c` consecutive per XCD; ids past the last whole round in round-robin order (HipKittens' `chiplet_transform_chunked`, what the GEMMs did) |
| c > 0 | `"spread"` | the same rounds; the remainder dealt as contiguous ranges |
| `"even"` | either | c = max(total // xcds, 1) (`gemm_cdna_dp`'s `xcd_chunk = 0`) |

`place` is a bijection on `[0, total)` for every total, xcds and chunk. The IR tier checks this
over totals 0-129 plus four large ones, xcds ∈ {1, 2, 4, 6, 8}, and every chunk/tail combination.
It also checks that `place` reproduces the GEMMs', dp's and MoE's hand-written remaps exactly where
those applied.

**On the kernel object.**

- `hk.kernel(placement=...)` takes `None`, `"auto"` or an `hk.Placement`. The analysis and the plan
  are kept on the `TracedKernel` (`analysis`, `plan`, `logical_grid`).
- `Kernel.placed(p)` is a sibling kernel with placement `p`, named with a suffix such as
  `_x8cl0p`, `_x8ch32s` or `_x8ches`.

## 3. Policy: when, and which

`hk.placement.choose(analysis, plan, grid, radii, xcds, resident_per_xcd)` decides, at one launch
geometry, whether to launch the cluster placement. Rules, in order:

1. **One L2** (xcds 1): no.
2. **Order-dependent kernel** (any op of §1's list), or `resident=True`: no. Reordering can deadlock
   a grid whose workgroups wait on later ones, and changes the order of atomic floating-point sums.
3. **Not a cluster kernel**, or **a split axis is not a whole number of its radix** (the mixed-radix
   decode would not be a bijection): no.
4. **Streams per XCD.** Under the default order, count the distinct key values among the first
   `resident_per_xcd` workgroups of each XCD, and take the maximum. Below `MIN_STREAMS` (5): no.
5. **Non-uniform work.** rounds = total / (xcds · resident_per_xcd). If the kernel is non-uniform
   and has fewer than `MIN_ROUNDS_NONUNIFORM` (12) rounds: no. An XCD's last cluster would end on
   its longest unit, while the default's last round is short units.
6. Otherwise: yes.

`resident_per_xcd` is the kernel's measured residency, `Kernel.resident_capacity()` / xcds, from
its build's registers and LDS.

`placement="auto"` applies `choose` at each launch through `Kernel.__call__`, cached per tensor
shapes. `Kernel.placement_choice(*tensors)` returns the `Choice` (`placed`, `reason`, `streams`,
`rounds`). A torch-registered op is launched by torch, not through `__call__`, so it routes itself:
`attn_cdna.mapping_for` is `choose` on the dense kernel's own analysis, with 32 workgroups per XCD.

**Where the thresholds come from.** They were measured on attention (PR #55, 74 points), and
reproduce its routing at every measured point.

| case | measured |
|---|---|
| 128 KV heads, 16 streams per XCD | +11-18% |
| 64 KV heads, 8 streams per XCD | +5-8% |
| 40 KV heads, 5 streams per XCD | +2-5% |
| 2-4 streams per XCD | within 3% |
| causal, 40 KV heads, N 8K (5 rounds) | −10% |
| causal, 10 rounds | −2% |

**When it can be slower.**

| case | why | guard |
|---|---|---|
| short non-uniform grids | an XCD ends on its longest unit | rule 5 |
| units of different cost in contiguous ranges (MoE experts, ragged requests) | `place` balances counts, not work | measure; `chunk > 0` interleaves rounds |
| clusters ≤ xcds | nothing to win | rule 4 |
| interleaving several clusters per XCD | tried for causal grids: 0.92 and 0.88 at 40 heads, 8K. Suspected cause: windows straddle XCD ranges | not offered |

## 4. The XCD count

`hk.placement.xcd_count(arch)` returns the number of L2 domains, from the first of:

1. `HK_XCDS`, if set.
2. KFD's `num_xcc` (`/sys/class/kfd/kfd/topology/nodes/*/properties`), when every node of that arch
   agrees. It reads 8 on an MI355X, 4 on an MI350P or a DPX partition, 6 on an MI300A. This needs
   no HIP context.
3. The arch's default: 8 for gfx950 and gfx942, 1 for every other arch.

The count is baked into the emitted decode as a constant, so it is part of the kernel's source and
of its compile-cache key. `hk.arch.xcd_count()` gives it to a kernel body as a constexpr. A count
of 1 makes every placement the identity.

## 5. Checking a placement: hk.check --placement

`hk.placement.emitted_decode(module, name)` builds a host function from the ops the rewrite tagged
(`hk.placement_axis`). It evaluates `arith` constants, `addi`, `subi`, `muli`, `divsi`, `remsi`,
`minsi`, `maxsi`, `ceildivsi`, `hk.block_idx` and `hk.extent`. So it checks what was emitted, not
the Python that emitted it.

`check_emitted(traced, extents)` runs it over every workgroup id of a launch and raises unless each
logical `(x, y, z)` of the grid comes out exactly once.

`python -m hk.check TARGET --placement --inputs SPEC`:

- **a raw `@hk.kernel`:** checked at the shapes of `--inputs`, with no GPU and no torch (only the
  spec's shapes are read). For `placement="auto"`, both the kernel and its `.placed()` sibling are
  checked.
- **a function:** run on the GPU; every hk kernel it launches is recorded (`hk.placement.recording`)
  and checked.

Output, one line per kernel per case:

```
case 0: kernel: a bijection over 270 workgroups, grid (45, 3, 2), order (z, y, x), 8 XCDs
case 0: kern_x8cl0p: not applicable at grid (30, 8, 1) (an axis is not a whole number of its radix); placement='auto' does not pick it
case 0: kern: the hardware's numbering (cluster key (z, x % g.k.depth()); non-uniform work)
```

A failure prints `FAILED <how many grid points ran twice and never>` and exits 1.

## 6. Other shapes of work

- **GEMM rasterization.** It is `Placement.chunked(xcd_chunk, "stay")` on the block id, then the
  kernel's own `wgm` group walk (`tile_of`) on the result. `xcd_chunk` and `wgm` stay schedule
  knobs.
- **Persistent kernels** loop over units that are not workgroups:
  - `mxfp8`'s persistent GEMM calls `hk.placement.renumber(wg + i·G, blocks, chunk=xcd_chunk)` on
    each virtual block id.
  - `moe_a8w4`'s persistent stage 2 takes its XCD's units from `hk.placement.xcd_range(xcd,
    units)`: contiguous, balanced to one.
  - `moe_a8w4`'s sorted kernels (the vLLM route's stage 1 and stage 2) size their grid for the
    most sorted blocks a launch can have. Only the first `cdiv(num_valid, BM)` blocks are live,
    and the rest exit at once. At decode M that is about 60% of the grid (M 64: 235 of 390
    blocks). They renumber in the body: `hk.placement.renumber(wg, total, live=...)` places
    `[0, live)` among themselves and leaves the rest alone, so the live blocks are dealt over
    every XCD.
    - A grid placement over all of `[0, total)` (#65's `chunked(0, "spread")`) handed the live
      prefix to the first XCDs' ranges and left the others idle. At M 64 that was 0.89x of the
      hardware's order (found by the MI355X scoreboard); `live=` fixed it.
  - The stream-K plan (host Python) reads `xcd_count`.
- **Megakernel queues** (comm roadmap R5): push a task to the queue of the XCD its sharing key maps
  to, drain one cluster before the next, and steal whole clusters. These are library rules, not
  this pass: the analysis cannot see through a queue.
- **Per-XCD broadcast** (the decode megakernel's activation): a read with an empty key is shared by
  every workgroup already, so placement does nothing for it. What helps is one copy per XCD (an
  hk.comm transform). The analysis's `broadcast_bytes` marks such reads.

## 7. What was migrated, and what it measured

MI355X, one process, the old and new kernels interleaved, `rocm-smi` snapshots around each run
(`<workspace>/research/2511.02132/runs/m1-ab*`, `m2-ab*`):

| kernel | before | after | outputs | new / old |
|---|---|---|---|---|
| `attn_cdna` spatial | hand-written `_spatial` (PR #55) | `Placement.cluster()` (dense), explicit key (paged) | bitwise, 15 points | 0.993-1.003 against PR #55's |
| `gemm_cdna`, `gemm_cdna_dp` | inline chunked remaps | `Placement.chunked(xcd_chunk \| "even", "stay")` | bitwise, 11 shapes | 0.997-1.007 |
| `mxfp8` (K2) | inline chunked remaps | `chunked`, `renumber` | bitwise, 18 DSV4.1 points | 0.992-1.011 |
| `moe_a8w4` (K1) | contiguous eighths, identity if the grid % 8 ≠ 0 | sorted kernels: `renumber(live=)` in the body (live sorted blocks only); tile-list kernels `chunked(0, "spread")`; persistent `xcd_range` | bitwise | against main before #65, M 64-16384: **1.04-1.33x** at every M (M 64 1.14, M 2048 1.15, M 8192 1.09, M 10000 1.33). #65's first form, `chunked(0, "spread")` over the whole grid, was 0.89x at M 64 |
| stream-K plan | `NUM_XCDS` | `xcd_count` | same plan on 8 XCDs | — |

The default attention kernel's generated C++ is byte-identical before and after.

## 8. Limits, and what is not built

- **Addresses the analysis can follow:** built from block ids by `+ - * // %` with constants,
  extents and arithmetic of those. Anything through memory is indirect and needs `key=`.
- **One radix per axis.** An axis divided by two different values is taken whole.
- **Raster kernels are classified but not placed automatically.** The GEMMs state
  `Placement.chunked`.
- **"auto" only through `Kernel.__call__`.** Torch ops route themselves.
- **The thresholds were measured on one kernel family (attention).**
- **M3, not built:** `placement` as a searched schedule knob; auto-raster (choosing `wgm` and
  `chunk` from L2 capacity and panel bytes); placement for persistent kernels from the analysis.
