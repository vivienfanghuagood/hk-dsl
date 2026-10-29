# The scoreboard again, with every baseline tuned: what FlyDSL wins, what hk wins, and why

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 18](18-the-scoreboard.md) ended with hk's GEMM at 1.03-1.08x "the
faster of Triton (autotuned) and FlyDSL". An audit of what each arm had been
asked to do found that neither opponent had tried:
- Triton ran the matmul tutorial's seven configs.
- FlyDSL's GEMM was called through `create_wmma_gemm_module` at its bare
  128x128x32 default. FlyDSL's own entry points were not used: its
  dispatcher (`pick_tile` plus per-tile options) and its autotuner
  (`rdna3_gemm_autotuned`).
- FlyDSL's norm and softmax autotuners (`rmsnorm_autotuned`,
  `softmax_autotuned`) were not used either.
- hk's own GEMM record file shipped empty, so hk ran its default schedule
  too.

So this chapter rebuilds the scoreboard first, with every DSL out of the box
and tuned. It then attributes each loss to what the winner does, and asks
whether hk-dsl can express that. hk-dsl is meant to sit beside FlyDSL, not
replace it, so the question throughout is where each one wins and why.

## The harness, and four ways a tuned arm was not tuned

`tools/hk-bench/scoreboard.py` now has eight arms. Each records what it
searched, how many candidates, how it timed them and what won (`--json`
keeps all of it):
- `hk`: hk.ops on its default schedule (GEMM) or its planner (norms).
- `hk_tuned`: hk's GEMM after its own tuner. `--tune` runs
  `hk_gemm.TUNER.tune` for any shape without a record: 96 points, stage 1
  compiles and gates spills, stage 2 times the survivors.
- `torch`: eager torch: hipBLASLt, aotriton, ATen.
- `inductor`: `torch.compile` of the same expression.
- `triton`: the kernel as a Triton user first writes it (`triton_ref.py`).
- `triton_tuned`: the same kernel autotuned over gfx1100's space.
  - Matmul: BLOCK_M/N in {64,128,256}, BLOCK_K in {32,64,128}, num_warps
    {4,8}, num_stages {1,2}, waves_per_eu {0,2}. That is 120 configs, after
    pruning those whose staged tiles pass 64 KB of LDS or whose accumulator
    passes 128 VGPRs a lane. `matrix_instr_nonkdim` and `kpack` are left out
    on purpose: they choose among MFMA instructions, and gfx1100 has only
    WMMA 16x16x16.
  - Row kernels: num_warps {1,2,4,8,16} x waves_per_eu {0,1,2,4}, keyed on
    rows and columns.
- `flydsl`: FlyDSL's dispatcher with no measurement: `pick_tile` and
  `tile_opts` for the GEMM, the default block for the norms.
- `flydsl_tuned`: FlyDSL's autotuner, forced to search
  (`FLYDSL_AUTOTUNE=1`).

Timing is unchanged in kind: every arm is checked against fp32 before it is
timed, then timed in a torch-profiler session of its own per round. Rounds
are interleaved and rotated, and the best round counts. An arm's noise is how
far its second-best round sits from its best. A case gets two verdicts, hk's
best arm against the best other: out of the box (hk against torch, inductor,
triton, flydsl) and tuned (hk and hk_tuned against everything). It is a tie
when the ratio is within the larger of 2% and the two arms' noise.

Building the tuned arms turned up four ways for a tuner not to tune, each of
which would have handed hk a win:
- **FlyDSL's GEMM autotuner never searched.** `rdna3_gemm_autotuned` first
  returns whatever launcher `rdna3_gemm_dispatch` last stored for the
  signature. The dispatcher stores `pick_tile`'s launcher on every call. With
  the dispatcher arm in the same process, the "tuned" arm timed `pick_tile`'s
  kernel: both read 1.53995 ms at 4096³, to five digits. Now the stored
  launcher is dropped before a forced search, and the launcher the search
  leaves (its last dispatch is the winner) is kept and called directly. That
  is what the autotuned entry does on a hit. FlyDSL then searches all five
  feasible tiles, and its pick matches its own `[autotune] best:` line.
- **`FLYDSL_AUTOTUNE=1` forces a new search on every call.** It is set for
  each arm's checked first call only.
- **Triton's `do_bench` ranks a different regime.** It flushes the cache
  before every call and keeps the median. The scoreboard times warm,
  back-to-back calls. A 2048x4096 bf16 softmax is 33 MB and stays in the
  96 MB Infinity Cache when run back to back; ranked cold, Triton's tuned
  pick ran in 0.0362 ms against the untuned kernel's 0.0258. Now the tuned
  launchers pass `do_bench=` the scoreboard's own timing (warm, behind a GPU
  sleep, best of 5 rounds of 10), and search in every process. The softmax
  pick now runs in 0.0256 ms.
- **`restore_value` timed four copy kernels per norm.** Triton restores the
  in-place fused add's inputs by cloning and copying both operands inside
  every timed call, so the ranking measured mostly copies. The search now
  runs on scratch copies of the operands.

One label bug too: FlyDSL's "chosen" configs were read out of its tuner cache
by key prefix. That matched stale entries loaded from earlier processes' disk
caches, ahead of the search just run. They are now read off `resolve_config`.

## The scoreboard, tuned against tuned

After the fixes below (n_split=2 as the GEMM default, the folded norms and
softmax moving each row once): GPU ms per call, best of 5 rounds of 10 calls.
The rows marked * are best of 9 rounds of 50 calls, because at a few
microseconds 10 calls left hk's arm with 37% noise. Software: torch
2.10.0+rocm7.2.4, triton 3.6.0, FlyDSL 0.4.0.dev970, hk 0.3.1, on gfx1100.

| case | hk | hk tuned | torch | Inductor | Triton | Triton tuned | FlyDSL | FlyDSL tuned | tuned: hk vs best | chapter 18 |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.575 | 1.574 | 1.971 | -- | 1.674 | 1.669 | 1.547 | 1.540 | 0.98x loss (FlyDSL tuned) | 1.05x |
| gemm 2048x6144x4096 | 1.183 | 1.178 | 1.571 | -- | 1.285 | 1.257 | 1.139 | 1.139 | 0.97x loss (FlyDSL) | 1.08x |
| gemm 2048x4096x4096 | 0.8109 | 0.8106 | 0.9975 | -- | 0.8766 | 0.8582 | 0.7722 | 0.7721 | 0.95x loss (FlyDSL tuned) | 1.08x |
| gemm 2048x24576x4096 | 4.786 | 4.766 | 5.952 | -- | 5.084 | 4.999 | 4.631 | 4.641 | 0.97x loss (FlyDSL) | 1.05x |
| gemm 2048x4096x12288 | 2.427 | 2.401 | 2.943 | -- | 2.612 | 2.539 | 2.305 | 2.296 | 0.96x loss (FlyDSL tuned) | 1.03x |
| rms_norm 1x4096* | 0.0051 | -- | 0.0043 | 0.0026 | 0.0026 | 0.0026 | 0.0025 | 0.0027 | 0.48x loss (FlyDSL) | 0.41x |
| rms_norm 32x4096* | 0.0039 | -- | 0.0044 | 0.0027 | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.66x, "tie" (FlyDSL) | 0.47x |
| rms_norm 2048x4096 | 0.0248 | -- | 0.0262 | 0.0198 | 0.0193 | 0.0196 | 0.0191 | 0.0194 | 0.77x loss (FlyDSL) | 0.70x |
| rms_norm 8192x4096 | 0.1901 | -- | 0.1788 | 0.1769 | 0.1817 | 0.1737 | 0.1744 | 0.1768 | 0.91x loss (Triton tuned) | 0.92x |
| fused add + rms_norm 1x4096* | 0.0060 | -- | 0.0061 | 0.0027 | 0.0027 | 0.0027 | 0.0030 | -- | 0.44x loss (Inductor) | 0.37x |
| fused add + rms_norm 32x4096* | 0.0045 | -- | 0.0066 | 0.0029 | 0.0029 | 0.0029 | 0.0031 | -- | 0.64x, "tie" (Triton tuned) | 0.51x |
| fused add + rms_norm 2048x4096 | 0.0401 | -- | 0.0489 | 0.0395 | 0.0391 | 0.0365 | 0.0360 | -- | 0.90x loss (FlyDSL) | 0.78x |
| fused add + rms_norm 8192x4096 | 0.3478 | -- | 0.4545 | 0.3656 | 0.3522 | 0.3360 | 0.3689 | -- | 0.97x loss (Triton tuned) | 0.97x |
| softmax 2048x4096 | 0.0231 | -- | 0.0307 | 0.0576 | 0.0261 | 0.0258 | 0.0192 | 0.0192 | 0.83x loss (FlyDSL) | 0.63x |
| softmax 8192x4096 | 0.1846 | -- | 0.1739 | 0.2240 | 0.1739 | 0.1733 | 0.1746 | 0.1760 | 0.94x loss (Triton tuned) | 0.93x |
| attention n4096 causal, GQA 4 | 2.847 | -- | 10.831 | -- | -- | -- | -- | -- | 3.80x win (torch) | 3.65x |
| attention n4096, GQA 4 | 4.677 | -- | 12.762 | -- | -- | -- | -- | -- | 2.73x win (torch) | 2.75x |

How to read some of the cells:
- **The GEMM is a loss at every shape, 0.95-0.98x.** That is 84.8-87.5
  TFLOP/s against FlyDSL's 89.0-90.5. FlyDSL's dispatcher alone beats hk at
  every shape; its autotuner, searching its five feasible tiles, adds nothing
  beyond noise. Against tuned Triton hk is 1.05-1.07x, and against hipBLASLt
  1.23-1.33x.
- **hk's tuner** found at most 2.5% over the old default in its 96 points.
  It was adopted at two of five shapes (2048x24576x4096 and
  2048x4096x12288, +1.4% and +2.5%). With n_split=2 now the default, its
  arm is within 1.1% of the default everywhere.
- **The two "ties" at 32 rows are the band rule, not a near miss.** hk's
  arm there has 35-36% between its best and second-best rounds; its best
  round still loses to everything. One-row kernels are the least stable
  measurement on this card: hk's 1x4096 RMSNorm measured 3.7-5.3 us across
  these harnesses and runs (3.67 us in `norm_layout.py`, 5.1 here). Triton's
  and FlyDSL's stayed at 2.5-2.7 us. So it is a loss in every measurement,
  0.48-0.71x, whose size is not pinned down.
- **No Triton or FlyDSL attention.** FlyDSL's flash attention stages K/V
  through a global-to-LDS load gfx11 does not have. The Triton attention
  torch runs is aotriton, which is the torch column.
- **Out of the box**, the verdicts are the same, except that the fused add at
  8192 rows ties untuned Triton (1.01x) and the GEMM at 4096³ ties FlyDSL's
  dispatcher (0.98x).

## GEMM: what FlyDSL's dispatcher does that hk does not

FlyDSL's tile at every shape but 2048x24576 is 128x128x32 of four waves: a
64x64 warp tile, K-step 32, two padded LDS buffers (40 KB) and one barrier a
K-tile. Four options ride on it:
- `sched_hint`: `sched_group_barrier` groups that interleave the K-tile's
  global loads, LDS reads, WMMAs and LDS writes in a fixed pattern;
- `stagger`: workgroup w starts its K loop at k-tile w and wraps;
- the "persistent" path, taken at 256 tiles or more. It turns out to be one
  tile a workgroup again; what it adds is that same rotation, wrapped with a
  select rather than a remainder, so it also works when the K-tile count is
  not a power of two;
- `pad`, the padded LDS layout.

At 2048x24576 it picks 256x256x32 of sixteen waves with an unpadded k-major
("kblock") layout.

`tools/hk-bench/gemm_vs_flydsl.py` builds FlyDSL's kernel with each option
taken away, beside hk's own variants, and times them in one process the way
the scoreboard does. TFLOP/s:

| shape | hk default (n_split 4) | hk n_split 2 | FlyDSL, all options | without persist | without sched_hint | without stagger | bare |
|---|---|---|---|---|---|---|---|
| 4096x4096x4096 | 85.8 | 87.3 | 89.3 | 86.9 | 84.1 | 82.8 | 79.6 |
| 2048x6144x4096 | 87.9 | 89.2 | 90.7 | 89.0 | 86.4 | 84.4 | 80.3 |
| 2048x4096x4096 | 85.5 | 86.0 | 88.7 | 87.1 | 83.7 | 79.4 | 76.3 |
| 2048x24576x4096 | 87.5 | 89.0 | 91.5 | 87.6 | 87.1 | 83.9 | 82.1 |
| 2048x4096x12288 | 83.9 | 85.2 | 89.6 | 81.6 | 85.0 | 82.4 | 81.2 |

("Without stagger" also drops the persistent path. At K=12288 the
non-persistent path has no stagger, since 384 K-tiles is not a power of two,
so "without persist" there is the kernel without stagger.)

Read that as two levers on a kernel that is otherwise slower than hk's.
`sched_hint` is worth 5-6% at every shape. The K rotation is worth 5-12%, and
1.7-4.5% of that is the select wrap's advantage over a remainder. Bare (the
kernel chapter 18 measured) is 0.89-0.97x hk. Without its scheduling hints,
FlyDSL's kernel is at hk's level (0.98-1.01x).

What hk-dsl can express of that, each tried:
- **The rotation: expressible, and not a lever for hk.**
  `gemm_kernel(stagger=)` rotates the K start per workgroup. Written with a
  remainder, it measured 0.94-0.96x hk's default. Written branch-free, with
  scalar min/max and a multiply (the remainder is computed once per
  workgroup), it measured 0.91-1.00x at ten of eleven shapes, M 512 to 16384,
  and 1.02x at 8192³ inside that run's 2.6% noise. hk's eight-warp schedule,
  with one LDS buffer and an 8-row-group swizzle, does not have the access
  pattern the rotation repairs in FlyDSL's four-wave one. It stays a
  parameter and is not in the tuning space.
- **The 256x256x32 tile: expressible, builds, slower.** Sixteen warps of
  64x64 fit hk's schedule at 252 VGPRs and occupancy 5, with no spill. That
  is one workgroup a WGP, and it measured 0.77-0.88x hk's default across the
  five shapes, with one LDS buffer or two. FlyDSL's own 256 tile is
  0.85-1.03x hk's default and wins only at the two widest N.
- **The four-wave 128x128x32 tile: the DSL can say it, hk's schedule
  cannot fit it.** `gemm_kernel(k_step=32, warps=4, warp_rows=2)` needs 281
  VGPRs and spills 25; the build gate refuses it (a spill in a hand-waited
  kernel is a wrong answer). The cause is in `dot_tile`, not the language:
  it keeps the warp's whole A slice live (four 16-row fragments, each held in
  both wave halves, 32 VGPRs) plus every B chunk, beside 32 VGPRs of
  staging. FlyDSL streams one A fragment ahead of the WMMAs that use it.
  `dot_tile` could do that with per-fragment subtile loads, which the
  language has. It is a rewrite of the inner loop and its derived LDS waits;
  estimate one to two days. It only pays together with the next item: this
  tile is 0.89-0.97x hk until its scheduling hints are on.
- **`sched_group_barrier`: not expressible.** `hk.arch` has
  `sched_barrier()`, a full fence, and `setprio`, but nothing that says
  "interleave these N WMMAs with these M LDS reads". Adding it takes one op
  in `mlir/include/hk/Dialect/HK/IR/HKOps.td`, one case in
  `mlir/lib/Target/Cpp/HKToCpp.cpp` emitting
  `__builtin_amdgcn_sched_group_barrier(mask, size, syncid)`, a wrapper in
  `hk.arch`, and a rebuild of the dialect bindings for each Python the pods
  use (3.10 and 3.12). Estimate half a day for the op, then one to three days
  to find a plan that pays for hk's own schedule. That is the largest lever
  measured here: 5-6% to FlyDSL, against a 2-5% gap.

The one change that paid within hk's existing space: **n_split=2 is now the
default.** That is the warp's 64 columns as two 32-wide accumulators instead
of four 16-wide ones, at the same 211 VGPRs and occupancy 7. hk's tuner had
picked n_split=2 at all five shapes, at 0.4-2.5% over the old default, and
three of those were under its 1% adoption bar, so records were adopted at
only two keys. In one process it was faster than the old default at 15 of 16
shape-runs (two runs, M 512 to 16384) by 0.4-2.6%. The one loss, -1.2% at
8192x24576x4096, sat inside that run's 2.1% noise. About 1%, under the
verdict band: it changes no verdict in the table above.

## Norms and softmax: where the time goes

There are three regimes, and each loses for a different reason.

**1-32 rows: a longer instruction stream.** A one-row RMSNorm is 8 KB. From
the device assembly, per wave:

| | instructions | b128 loads | b128 stores | barriers |
|---|---|---|---|---|
| Triton's RMSNorm, 16 warps (the tuned pick) | 221 | 2 | 1 | 2 |
| hk's RMSNorm, folded plan, 4 warps | 846 | 16 | 8 (+32 b16 on the unaligned path) | 3 |

FlyDSL's is the same shape as Triton's. It runs one workgroup a row, each
thread takes 8 contiguous elements in one 128-bit buffer copy, and the
reduction is `shuffle_xor` within the wave plus LDS across waves, with two
barriers. `tools/hk-bench/norm_stages.py` times a one-row kernel stage by
stage. A bare hk tile copy (load one 16x64 tile a warp, store it) takes
3.47 us. Triton's entire RMSNorm takes 2.69 us, and hk's 5.17 us.

**2048 rows: the folded tile path, and a duplicate.** 2048x4096 is 33 MB and
stays in the Infinity Cache across back-to-back calls, so the ceiling is the
cache's, not HBM's. `tools/hk-bench/norm_layout.py` copies the tensor through
each register tile, one process, us:

| | 2D view | folded view (the small-M norms' plan) |
|---|---|---|
| torch's elementwise copy (`mul` by 1) | 18.95 (1.77 TB/s) | -- |
| through an fp32 tile, as shipped | 27.69 | 21.82 (1.54 TB/s) |
| through a bf16 tile, as shipped | 41.45 | 31.47 |
| bf16 tile, each run stored once | 26.82 | 21.12 |
| fp32 tile, each run loaded once | 28.57 | 25.05 |

hk's RMSNorm at this shape took 27.17 us, against its own copy's 21.82. So
about 5 us is the norm's own work (the weight, the reduction, three LDS round
trips) and 3 us is the tile path.

The duplicate is real, and HipKittens' own `quant.cuh` says why: "a 16-bit
row tile holds each row twice, in lanes l and l + 16". The packed fp32 to
16-bit store completes the same 32-byte run in both lanes, and
`store_bounded` then writes it from both halves. Its comment called that
benign; it had not been measured. For a bf16 tile it costs a third of the
copy.

The fix is in HipKittens, behind a macro that defaults off,
`HK_RT_HALF_SPLIT`:
- bit 1: each half loads its own 16 bytes of the run and takes the other 16
  from its partner with `v_permlanex16`;
- bit 2: each half stores only its own 16 bytes.

On hk's own kernels and plans, every variant bit-identical to the shipped
one, us:

| | plan | shipped | split stores | split both |
|---|---|---|---|---|
| rms_norm 1x4096 | w4 tpw1 fold16 | 3.84 | 3.76 | 3.67 |
| rms_norm 32x4096 | w4 tpw1 fold16 | 3.91 | 3.83 | 3.77 |
| rms_norm 2048x4096 | w4 tpw1 fold16 | 27.17 | 25.47 | 24.72 |
| rms_norm 8192x4096 | w16 tpw4 | 193.07 | 194.01 | 197.91 |
| fused_add_rms_norm 2048x4096 | w4 tpw1 fold16 | 46.12 | 42.37 | 40.15 |
| fused_add_rms_norm 8192x4096 | w16 tpw4 | 348.98 | 351.00 | 370.25 |

So the folded one-tile plans now use kernels built with both splits (`_h` in
`norm.KERNELS`), and the unfolded hold keeps the mirrored access. With four
tiles a warp, the exchange's VALU work costs more than the halved traffic
returns.

Softmax did not move with the split (0.99x), because its kernel uses the
unbounded loads and stores, and those have no split. Its tiles are all backed
up into bounds, so the bounded forms read and write the same elements.
Switched to them, at 2048x4096 (folded):
- shipped: 29.99 us;
- bounded: 23.69 us;
- bounded with the store split: 23.22 us.

That is now ahead of tuned Triton's 25.73. At 8192 rows (unfolded) bounded
was 0.94x, so that plan stays as it was. The folded plan takes bounded
accesses with the store split (`_hb`).

**8192 rows: within 3-9%, and a weight read by element.** Here the tensors
pass the Infinity Cache, so the ceiling is HBM's: torch's elementwise copy
moves 0.77 TB/s, and hk's tile copy 0.74. The unfolded plan holds four
16x64 tiles a warp. Its weight is a row vector broadcast down a WMMA-shaped
tile, which loads with 128 `global_load_u16` a lane where the data takes 32
`b128`. A stride-0 global would make the weight a tile load. But the launcher
refuses a row stride below the column count, and the bounded loads' buffer
range is computed from the stride. Both would need checking, so this was
written down rather than tried.

**The gap underneath all three.** hk's register tiles and vectors on RDNA3
have the lane layouts WMMA gives them:
- the 16-bit row tile is the operand layout, each row in both halves;
- the fp32 tile is the accumulator layout, each lane owning alternate
  columns.

A row kernel has no layout in which a lane owns a contiguous run of one row,
which is Triton's blocked layout and what FlyDSL's tiled copies build with
`make_layout`. Everything the norm planner does (fold, hold, the 16-row
tile, the cross-warp and fold reductions through LDS) works around that.
The split recovers the cheap part. A row-contiguous register vector, with its
loads, stores, wave reductions and broadcasts in HipKittens' RDNA3 tree and a
type in the DSL, is the structural fix. Estimate one to two weeks.

## What the fixes moved, and where each wins

The scoreboard just before the fixes (the same harness, the same day):
- rms_norm 2048x4096: 0.0270 ms (0.71x);
- fused add 2048x4096: 0.0469 (0.78x);
- softmax 2048x4096: 0.0298 (0.64x);
- the GEMM: 0.94-0.97x.

After the fixes:
- rms_norm 2048x4096: 0.0248 (0.77x);
- fused add 2048x4096: 0.0401 (0.90x);
- softmax 2048x4096: 0.0231 (0.83x). That is ahead of tuned Triton's 0.0258,
  and the first row-kernel case here where hk beats a tuned Triton;
- the GEMM: 0.95-0.98x.

None of those is a win against the best arm.

Against chapter 18, the GEMM has turned from a 1.03-1.08x win into a
0.95-0.98x loss. Nothing in hk got slower: hk's GEMM went from 82.1-85.7 to
84.8-87.5 TFLOP/s. FlyDSL went from 73.9-81.2 to 89.0-90.5, because it now
runs its own dispatcher. The norms and softmax lose by less than in chapter
18 at every shape but 8192 rows, where they are where they were.

So the honest division, on this card at these shapes:
- **FlyDSL wins the GEMM and the row kernels.** The GEMM is a four-wave tile
  with `sched_group_barrier` interleaving and a K rotation. The row kernels
  use layouts in which a thread owns 8 contiguous elements, reaching
  1.75-1.87 TB/s out of the Infinity Cache. It is 402 lines of GEMM and
  1,784 of norm kernels (its files, helpers included).
- **Triton wins the 8192-row norms and softmax**, closest to HBM. At one
  row it is tied with FlyDSL, and both are about 2x hk.
- **hk wins attention**, 2.73-3.80x the attention torch runs. There is no
  FlyDSL attention on gfx11 to compare, and hk's is the kernel that fills
  that gap. hk's GEMM is second, 5-7% ahead of tuned Triton and 23-33%
  ahead of hipBLASLt, in 124 lines whose LDS waits are derived and whose
  spills fail the build.

hk-dsl beside FlyDSL rather than instead of it is what this data supports.
What hk-dsl would need in order to close the GEMM gap is one op it does not
have; to close the norm gap, one register layout it does not have.

## What is left, ranked

1. `sched_group_barrier` in the dialect and a plan for hk's GEMM schedule:
   worth 5-6% to FlyDSL, against a 2-5% gap. Half a day for the op, one to
   three days for the plan.
2. A streamed-A `dot_tile`, so FlyDSL's four-wave 64x64 tile fits hk's
   schedule without spilling. One to two days, and worth something only
   after item 1.
3. A row-contiguous register layout for memory-bound row kernels: the small-
   and mid-M norm gap. One to two weeks.
4. Bounded accesses and the split for the unweighted folded plans
   (`hk.ops.rmsnorm`, `layernorm`). Same code path as softmax, not measured.
5. A broadcast weight for the unfolded weighted norms (8192 rows and up):
   128 element loads a lane today.

## Reproduction

```
cd hk-dsl
export FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 PYTHONPATH=python
python3 tools/hk-bench/scoreboard.py --only gemm,norm,softmax,attn --rounds 5 --json sb.json
python3 tools/hk-bench/scoreboard.py --only norm,softmax --rounds 9 --calls 50   # small M
python3 tools/hk-bench/scoreboard.py --only gemm --tune        # hk records for new shapes
python3 tools/hk-bench/gemm_vs_flydsl.py                       # FlyDSL's options, hk's tiles
python3 tools/hk-bench/norm_layout.py                          # tile copies, the split
python3 tools/hk-bench/norm_stages.py                          # one row, stage by stage
```
