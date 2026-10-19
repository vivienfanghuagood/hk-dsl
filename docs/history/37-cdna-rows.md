# gfx950's rows: strips on wave64, and what each op still pays

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 29](29-cdna.md) left gfx950's row and pointwise kernels on CDNA
tiles: against torch's eager call, add 0.59-0.99x, the norms 0.61-1.05x,
softmax 0.68-1.04x. torch was the only opponent then. This chapter measures
the ops against every arm the [scoreboard](26-the-scoreboard-retuned.md)
has, at the shapes an LLM runs them. It finds that hk lost by up to 2.5x
past 2048 rows. It ports [chapter 27](27-strips.md)'s register strips to
wave64 and runs every one of these ops on them. Of 132 cases, 99 now tie or
win; the other 33 are 0.90-0.98x, and the sections below say why for each.

## The scoreboard on gfx950

`tools/hk-bench/scoreboard.py` gained three groups:
- `llm-norm`: rms_norm (weighted), rmsnorm and layernorm (unweighted), the
  fused add + RMSNorm in place, softmax;
- `llm-quant`: int8 quantize, per row, absmax / 127;
- `llm-pointwise`: add, mul, exp, silu_and_mul on a gate/up output, and
  NeoX RoPE on (tokens, heads, 128) with each token's angles.

The shapes are rows 1, 32, 2048 and 8192 by hidden 4096, 5120 and 8192 (the
heads are hidden / 128). silu_and_mul runs at intermediate sizes 14336,
17408 and 28672 instead. The method is the scoreboard's:
- one process, the arms interleaved, every arm checked against fp32 first;
- each round queued behind a GPU spin ([chapter 32](32-measuring-the-measurements.md));
- the median of 7 rounds of 50 calls.

The arms are torch eager; Inductor (static shapes: one code object per op
otherwise went dynamic at the second shape and, past Dynamo's recompile
limit, eager); Triton as written and autotuned (`triton_ref.py`, which
records what it searched); and FlyDSL's dispatcher and autotuner where
gfx950 has the op. FlyDSL has the norms and softmax and nothing standalone
for quantize, the elementwise ops, silu_and_mul or RoPE, and those arms say
so. The stack was the `rocm/pytorch` image: torch 2.10.0+rocm7.2.3, Triton
3.6.0, FlyDSL 0.4.0, hipcc 7.2.3.

The rule that rejects impossible rounds needed this card's ceilings
(`_rounds.CARDS`). `--probe-ceilings` measured torch's out-of-place add at
6.9-7.2 TB/s inside the 256 MB cache and 5.8-6.1 TB/s past it. At 7.2 the
rule rejected a fused add's real 8.35 TB/s at 2048x4096. An in-place add at
a 32 MB footprint, inside the eight XCDs' 4 MB L2s, ran at 9.1 TB/s, and
that is the ceiling now.

## Where it started

hk at `35421a7`, its ratio to the fastest other arm (tuned verdicts):

| op | 1-32 rows | 2048 rows | 8192 rows | verdicts |
|---|---|---|---|---|
| rms_norm | 0.86-1.00 | 0.40-0.50 | 0.57-0.64 | 8 loss, 4 tie |
| rmsnorm | 0.98-1.00 (1 row: refused) | 0.46-0.56 | 0.56-0.70 | 6 loss, 3 tie, 3 refused |
| layernorm | 0.90-1.05 (1 row: refused) | 0.51-0.56 | 0.53-0.66 | 7 loss, 2 tie/win, 3 refused |
| fused add + RMSNorm | 0.65-0.99 | 0.42-0.56 | 0.61-0.76 | 11 loss, 1 tie |
| softmax | 0.84-1.00 (1 row: refused) | 0.48-0.57 | 0.54-0.64 | 7 loss, 2 tie, 3 refused |
| quantize | 0.42-1.05 (1 row: refused) | 0.44-0.45 | 0.43-0.61 | 8 loss, 1 win, 3 refused |
| add, mul | 0.97-1.02 | 0.80-0.88 | 0.88-0.92 | 14 loss, 10 tie/win |
| exp | 0.98-1.01 | 0.58-0.72 | 0.75-0.81 | 7 loss, 5 tie |
| silu_mul | 1.05-1.06 (1 row: refused) | 0.71-0.88 | 0.74-0.76 | 6 loss, 3 win, 3 refused |
| RoPE | -- | -- | -- | no token-major op |

At 8192x8192 hk's RMSNorm moved 4.0 TB/s and tuned Triton's 7.0. The tile
kernels that need a whole 16-row tile refused one row. `silu_mul`'s
wrapper wanted contiguous halves, so the scoreboard handed it copies.
`hk.ops.rope` is (batch*heads, seq, D) with angles per position, not an
LLM's (tokens, heads, D).

## Measured before choosing

- **The small rows are the card's floor, not a launch path.** Queued behind
  the spin, every arm at 1 and 32 rows takes 3.8-4.2 us: torch, Triton,
  FlyDSL and hk alike. These kernels move under 2 MB, under 0.3 us at
  these rates. Chapter 32's late-launch penalty does not show, because the
  queue removes it, and the floor is the same for every launcher.
- **The kernel arguments are not late.** On gfx950 a strip kernel's entry is
  two dependent scalar-cache round trips before the first load (load, wait,
  load, wait). gfx11's `touch_kernargs` reads every line in one clause. Built
  for gfx950 and A/B'd in one process against the same kernels without it,
  over 25 kernel x shape pairs, it measured 0.98-1.04x, the median 1.00. It
  was removed again.
- **What is left is bandwidth past 2048 rows**, which the strips were built
  for on gfx11: a lane owning contiguous 16-byte runs of one row, the row
  read once.

## Strips on wave64

`include/hk/cdna.cuh` now has `kittens::rs`, its warp ops and its memory
ops, ported from include/rdna3 with what wave64 changes:
- A chunk is 64 lanes x RUN.
- The row reduction's butterfly is DPP `quad_perm` (xor 1, xor 2),
  `row_half_mirror` and `row_mirror` inside each 16-lane row, then
  `v_permlane16_swap` and `v_permlane32_swap` across rows. Every stage
  combines a value with a partner that combines the same pair, so every
  lane ends with the same bits. gfx942 has no swaps and would take
  ds_bpermute; it is not enabled.
- Stores round bf16 to nearest even (`hk_cvt`; the CDNA trees' convertor
  truncates).
- The bounded accesses use the shim's buffer descriptor.

`Target.strips` (gfx1100, gfx950) replaces the check on include/rdna3's
define. The DSL refuses a wave64 strip that is not whole 64-lane chunks,
and the register estimate divides by the target's wave. The norm kernels'
`cw = CHUNKS * 32 * RUN` became the target's wave; nothing else in
`norm._strip_kernel` changed.

The weighted RMSNorm at 4096 columns, 2 warps of 4 chunks, has 8
`global_load_dwordx4` (row and weight), 4 stores, 4 `v_add_f32_dpp`, the
two swaps and one barrier: 249 instructions a wave, 68 VGPRs, 7 waves a
SIMD. The strip tests pass on gfx950 with the widths scaled by wave/32, and
so do the 252 norm tests (the fused add's residual bit-exact).

**The plan** is 32 elements a lane on wave64 (`_STRIP_PER_LANE_WAVE64`):
four bf16 runs, the warps chosen to split the row's chunks exactly where a
split near the target does. Swept at 8, 16, 32 and 64 a lane over the five
kinds x 4096/5120/8192 columns x 2048/8192 rows (`strip_plan.py`), the rule
is within 5% of the fastest at all 30 points and within 1% at 23. gfx11's
16 a lane (8 for the fused add) was 7-16% behind at 2048x8192; 32 a lane
split by ceiling (5120's ten chunks as three warps of four) was 8-30% behind
at 2048x5120.

**Two fixes the quantize kernel found, which every strip kernel has.**
- The cross-warp combine read its LDS vector through a generic pointer,
  `flat_load_dword` (both counters, the flat path); through an LDS pointer
  it is `ds_read2_b32`.
- The int8 clamp `r < -128 ? -128 : (r > 127 ? 127 : r)` compiled to two
  `v_cmp`, two `v_cndmask` and the VCC hazard's `s_nop` an element; it is
  one `v_med3_f32` now. The quantize kernel went from 462 instructions to
  332.

The host's ROCm 7.1 accepted binding a `const float&` to an LDS-address-space
value; the image's 7.2.3 refused it. The codegen tier runs on the host, so
only a GPU run in the image caught it.

**An occupancy cap past the cache**, as gfx11's ([chapter 27](27-strips.md)).
Tuned Triton's 8192x8192 picks carry a `waves_per_eu`. A strip kernel with
53 KB of LDS reserved holds three workgroups a CU (`_STRIP_RESERVE_AT`).
8192x8192 bf16, us, one process:

| | no cap | 32 KB | 53 KB | 80 KB | tuned Triton |
|---|---|---|---|---|---|
| rmsnorm | 39.19 | 38.79 | 37.83 | 45.07 | 37.67 |
| layernorm | 39.96 | 38.95 | 38.81 | 54.04 | 37.64 |
| rms_norm | 38.75 | 38.52 | 37.77 | 48.80 | 38.19 |
| softmax | 39.91 | 39.37 | 38.88 | 58.94 | 50.41 |
| fused add | 76.95 | 76.83 | 77.53 | 76.00 | -- |

Inside the cache it costs (8192x4096 rmsnorm 20.13 to 22.63 us). So it
applies from the cache's 256 MiB, which 8192x8192's in and out are exactly.

## The other ops on strips

- **quantize** (`quant._strip_quantize_kernel`): one workgroup a row, any
  row count. It holds 16 elements a lane from 4096 rows, where it streams
  HBM and more warps on a row won by 1-5%, and 32 below.
- **The elementwise ops** run over a flat (1, n) view, fp32 between a
  widening load and a store that rounds to nearest even. That is torch's
  arithmetic, and add, mul, sub, max and min are bit for bit torch's. The
  plan is 8 warps of one chunk, 4096 elements a workgroup. A two-input op
  moving 24-48 MB takes eight chunks a warp (about one workgroup a CU at
  2048x4096), which won 1.16-1.19x in four of six runs and lost in two; see
  below.
- **silu_mul** takes the gate/up output's halves as strided views and reads
  them in place, at 512 elements a workgroup. What mattered was the grid
  order. Workgroups ordered rows first put consecutive ones a row apart,
  57 KB at 14336. Rows first, the best of 16 plans took 263.6 us at
  8192x28672 (tuned Triton 250.9) and 123.9 at 8192x14336 (127.5); ordered
  along a row, as the memory lies, the shipped plan takes 229.2 and 116.7.
- **RoPE** is `hk.ops.rotary(x, cos_sin, positions=None)`: vLLM's
  rotary_embedding, NeoX style, token-major, from `cos_sin_cache` rows of
  `[cos | sin]` and optional int32 positions.
  - A head is one strip row: 128 columns, two bf16 a lane.
    `Tile.swap_halves()` is a new strip map that exchanges each row's two
    halves: a register swap for an even number of chunks, `v_permlane32_swap`
    for a row of one chunk. With it,
    `out = x [cos | cos] + swap_halves(x) [-sin | sin]`, both factors built
    once a warp from the token's one row of the table.
  - The first version wrote each head's store before the next head's load.
    `x` and `o` may be one tensor, so the compiler cannot hoist a load
    above a store, and every head waited a memory round trip: 53.4 us at
    8192x64 to tuned Triton's 39.5. With every load issued first it is 39.2
    to 39.0.
  - Four warps a token, or one per eight heads from 64 heads, measured over
    4-16 warps.

## The result

The scoreboard, the norms from one run and quantize and the pointwise ops
from another of the same kernels (see "Two runs shared with other jobs"):

| op | 1-32 rows | 2048 rows | 8192 rows | verdicts |
|---|---|---|---|---|
| rms_norm | 0.99-1.01 | 0.96-1.04 | 0.98-1.01 | 1 loss, 9 tie, 2 win |
| rmsnorm | 0.97-1.00 | 0.94-1.01 | 0.98-1.00 | 4 loss, 8 tie |
| layernorm | 1.00-1.08 | 0.95-1.14 | 0.95-0.99 | 3 loss, 4 tie, 5 win |
| fused add + RMSNorm | 0.99-1.02 | 0.95-1.01 | 0.98-1.00 | 3 loss, 8 tie, 1 win |
| softmax | 0.98-1.00 | 0.95-0.97 | 0.98-1.01 | 4 loss, 8 tie |
| quantize | 1.06-1.09 | 0.98-1.04 | 0.98-1.02 | 2 loss, 2 tie, 8 win |
| add | 0.98-1.04 | 0.90-0.97 | 0.97-1.00 | 5 loss, 6 tie, 1 win |
| mul | 0.98-1.02 | 0.90-0.97 | 0.97-1.00 | 5 loss, 6 tie, 1 win |
| exp | 0.97-1.00 | 0.99-1.05 | 0.99-1.00 | 2 loss, 9 tie, 1 win |
| silu_mul | 1.05-1.08 | 1.00-1.04 | 1.08-1.09 | 2 tie, 10 win |
| RoPE | 1.04-1.09 | 0.94-1.03 | 0.96-0.99 | 4 loss, 1 tie, 7 win |

At 8192x8192, hk's RMSNorm went from 66.3 us to 37.5 (7.2 TB/s) against
tuned Triton's 38.0, and softmax from 61.6 to 39.3 against FlyDSL's 39.7.
silu_mul at 8192x28672 is 228.2 us against tuned Triton's 248.7. The case
by case table is at the end.

What the 33 losses are:
- **Six at the floor** (rmsnorm and softmax 32x4096, mul 1x5120, add
  32x4096, exp 32x5120 and 32x8192: 0.97-0.98x): 3.9 us against torch's or
  FlyDSL's 3.8.
- **Nine 2048-row norms and softmax** (0.94-0.97x), cache-resident 5-20 us
  kernels. The per-lane target that wins there differs by kind (64 a lane
  for softmax, 16 for the weighted norm); one rule for all is within 5%.
  FlyDSL's tuned softmax holds a row in one wave, 128 elements a lane at
  8192 columns, past the strips' 64.
- **Four 8192-row norms** (rmsnorm and LayerNorm 8192x4096, LayerNorm and
  the fused add 8192x8192: 0.95-0.98x), against tuned Triton.
- **add and mul at 2048 rows** (0.90x at 2048x4096, 0.97x at 5120 and 8192):
  see placement, below. At 8192x8192 they are 0.97x.
- **RoPE at 40 and 64 heads** (0.94-0.98x). Two bf16 a lane is a 4-byte
  access. A 16-byte form puts a head on 16 lanes and needs a lane rotation
  inside 16-lane rows (DPP `row_ror:8`) and the angles in the strip's
  run-of-8 pattern, which a strip load of a 128-wide table row does not
  give. Not done.
- **quantize at 2048x8192 and 8192x4096** (0.98x), 0.2-0.3 us behind tuned
  Triton.

## What did not work, or did not hold

- **`touch_kernargs` on gfx950**: no effect, above.
- **The elementwise plan inside the cache moves with where the buffers
  sit.** At 2048x4096, eight warps of eight chunks measured 6.39 us
  against tuned Triton's 7.39 in one run. In another process on another
  GPU it was 7.96-8.34 on three fresh buffer sets against 7.50, and in the
  final run 8.2 against 7.4. Two A/B runs of six plans against the
  other arms (`strip_pointwise_ab.py`) agreed with each other to 1-2%, but
  not with those. Rounds inside one process sit within 0.3%; the spread
  is between allocations. The rule takes the plan that is closest to the
  fastest on average (0.985 over 12 shapes x 2 runs) and keeps the large
  one for the one band it won in most runs. Past 8192 rows the ratios
  hold across runs.
- **Per-row-count plans for the norms.** 16 a lane beat 32 for quantize at
  8192 rows and is used there. For the norms the two were within 1% of each
  other at 8192 rows, so nothing switches.
- **Two runs shared with other jobs.** This node's other users took up to
  seven of its eight GPUs during the work, twice in the middle of a
  scoreboard run.
  The second half of one run read 40 us for a 4 us kernel in every arm.
  `_rounds` cannot see that, because no arm is faster than it can be.
  `tools/hk-bench/watched.sh` now logs the GPU's use and VRAM every 30 s
  beside a run. Of the final run, the norms (no VRAM but this process's)
  are reported, and quantize and the pointwise ops come from the previous
  run, whose kernels are the same.

## Tests

- `tests/hk/ir/test_strip.py`: a wave64 strip's spelling and its chunk
  check, `swap_halves` in the IR and the C++, and its refusals (gfx11, a
  WMMA tile, three chunks).
- `tests/hk/ir/test_strip_plan.py`: the wave64 plan and capacity, and
  gfx950's occupancy cap from 256 MiB.
- `tests/hk/codegen/test_strip_cdna.py`: every shipped strip kernel builds
  for gfx950 without spilling at 6 waves a SIMD or more; the weighted norm's
  ISA (8 b128 loads, 4 stores, 4 `v_add_f32_dpp`, both swaps, no
  ds_bpermute); RoPE's swap.
- `tests/hk/gpu/test_cdna_rows.py`: quantize at 1-2048 rows and ragged
  widths in three dtypes, the binary ops bit for bit torch's, the unary
  ones, silu_mul on halves and in place, rotary with and without positions
  at D 64/128/256 and in place, a zero rotation that is a copy.
  `test_strip.py` runs on gfx950 with its widths scaled by wave/32.
- lit: `swap_halves` round-trips and is refused on a tile.

RDNA output is unchanged. With `tools/cdna-probes/dump_sources.py`, which
now hashes the strip and RoPE tables too, run on `origin/main` and on this
branch: gfx1100's generated source is identical for every existing kernel
(226 new ones), and gfx1201's differs only in the text of refusals. gfx1100
keeps its strip plan, its tile kernels for quantize, the elementwise ops and
silu_mul, and its own occupancy cap.

## Reproduction

```
# one idle MI355X, rocm/pytorch 2.10 image, cp312 hk._mlir over python/hk/_mlir
export FLYDSL_ROOT=/path/to/FlyDSL HK_ARCH=gfx950
python3 tools/hk-bench/scoreboard.py --only llm-norm,llm-quant,llm-pointwise \
    --rounds 7 --calls 50 --json rows.json
python3 tools/hk-bench/scoreboard_summary.py rows.json
python3 tools/hk-bench/scoreboard.py --probe-ceilings --rounds 7 --calls 50
python3 tools/hk-bench/strip_plan.py --kinds rmsw,addrmsw,softmax,rms,layer,quant \
    --cols 4096,5120,8192 --rows 2048,8192 --targets 8,16,32,64
python3 tools/hk-bench/strip_pointwise_ab.py      # elementwise plans against the other arms
python3 tools/hk-bench/strip_pointwise.py --ops silu_mul --runs 8,4
```

## Every case

The tuned verdict: hk's time, the fastest other arm and its time, the ratio
(above 1 is hk faster), a tie within the larger of 2% and the two arms'
noise.

| case | hk, us | fastest other, us | hk vs it |
|---|---|---|---|
| rms_norm 1x4096 | 3.93 | 3.89 torch | 0.99x tie |
| rms_norm 1x5120 | 3.81 | 3.85 flydsl | 1.01x tie |
| rms_norm 1x8192 | 3.83 | 3.83 flydsl | 1.00x tie |
| rms_norm 32x4096 | 3.96 | 3.96 flydsl_tuned | 1.00x tie |
| rms_norm 32x5120 | 3.94 | 3.94 flydsl | 1.00x tie |
| rms_norm 32x8192 | 3.94 | 3.91 flydsl | 0.99x tie |
| rms_norm 2048x4096 | 5.72 | 5.52 triton_tuned | 0.96x loss |
| rms_norm 2048x5120 | 6.92 | 7.18 flydsl | 1.04x win |
| rms_norm 2048x8192 | 10.54 | 10.96 triton_tuned | 1.04x win |
| rms_norm 8192x4096 | 19.99 | 19.59 triton_tuned | 0.98x tie |
| rms_norm 8192x5120 | 25.00 | 24.92 inductor | 1.00x tie |
| rms_norm 8192x8192 | 37.51 | 37.95 triton_tuned | 1.01x tie |
| rmsnorm 1x4096 | 3.80 | 3.81 flydsl_tuned | 1.00x tie |
| rmsnorm 1x5120 | 3.81 | 3.81 flydsl | 1.00x tie |
| rmsnorm 1x8192 | 3.81 | 3.81 flydsl_tuned | 1.00x tie |
| rmsnorm 32x4096 | 3.94 | 3.84 flydsl | 0.97x loss |
| rmsnorm 32x5120 | 3.91 | 3.93 flydsl | 1.00x tie |
| rmsnorm 32x8192 | 3.92 | 3.91 flydsl | 1.00x tie |
| rmsnorm 2048x4096 | 5.71 | 5.39 inductor | 0.94x loss |
| rmsnorm 2048x5120 | 6.35 | 6.42 triton_tuned | 1.01x tie |
| rmsnorm 2048x8192 | 10.91 | 10.21 triton_tuned | 0.94x loss |
| rmsnorm 8192x4096 | 20.06 | 19.62 triton_tuned | 0.98x loss |
| rmsnorm 8192x5120 | 24.91 | 24.48 triton_tuned | 0.98x tie |
| rmsnorm 8192x8192 | 37.89 | 37.77 triton_tuned | 1.00x tie |
| layernorm 1x4096 | 3.92 | 4.21 inductor | 1.08x win |
| layernorm 1x5120 | 3.89 | 3.89 flydsl | 1.00x tie |
| layernorm 1x8192 | 3.92 | 4.21 triton | 1.07x win |
| layernorm 32x4096 | 3.94 | 4.22 triton | 1.07x win |
| layernorm 32x5120 | 3.93 | 3.94 flydsl | 1.00x tie |
| layernorm 32x8192 | 3.94 | 3.94 flydsl | 1.00x tie |
| layernorm 2048x4096 | 5.88 | 6.14 triton_tuned | 1.04x win |
| layernorm 2048x5120 | 6.70 | 7.60 triton_tuned | 1.14x win |
| layernorm 2048x8192 | 11.18 | 10.65 triton_tuned | 0.95x loss |
| layernorm 8192x4096 | 20.68 | 19.70 triton_tuned | 0.95x loss |
| layernorm 8192x5120 | 25.29 | 25.14 triton_tuned | 0.99x tie |
| layernorm 8192x8192 | 38.71 | 37.76 triton_tuned | 0.97x loss |
| fused_add_rms_norm 1x4096 | 3.91 | 3.91 flydsl | 1.00x tie |
| fused_add_rms_norm 1x5120 | 3.90 | 3.93 flydsl | 1.01x tie |
| fused_add_rms_norm 1x8192 | 3.93 | 3.91 flydsl | 0.99x tie |
| fused_add_rms_norm 32x4096 | 3.86 | 3.95 flydsl | 1.02x win |
| fused_add_rms_norm 32x5120 | 3.95 | 3.98 flydsl | 1.01x tie |
| fused_add_rms_norm 32x8192 | 3.98 | 3.99 flydsl | 1.00x tie |
| fused_add_rms_norm 2048x4096 | 7.76 | 7.54 triton_tuned | 0.97x loss |
| fused_add_rms_norm 2048x5120 | 11.85 | 11.95 triton_tuned | 1.01x tie |
| fused_add_rms_norm 2048x8192 | 20.36 | 19.41 triton_tuned | 0.95x loss |
| fused_add_rms_norm 8192x4096 | 38.29 | 38.30 triton_tuned | 1.00x tie |
| fused_add_rms_norm 8192x5120 | 48.55 | 48.82 triton | 1.00x tie |
| fused_add_rms_norm 8192x8192 | 77.33 | 75.76 triton_tuned | 0.98x loss |
| softmax 1x4096 | 3.83 | 3.83 torch | 1.00x tie |
| softmax 1x5120 | 3.90 | 3.87 flydsl_tuned | 0.99x tie |
| softmax 1x8192 | 3.91 | 3.90 flydsl_tuned | 1.00x tie |
| softmax 32x4096 | 3.91 | 3.83 flydsl | 0.98x loss |
| softmax 32x5120 | 3.94 | 3.90 flydsl_tuned | 0.99x tie |
| softmax 32x8192 | 3.98 | 3.95 flydsl_tuned | 0.99x tie |
| softmax 2048x4096 | 6.41 | 6.17 flydsl_tuned | 0.96x loss |
| softmax 2048x5120 | 7.57 | 7.22 flydsl_tuned | 0.95x loss |
| softmax 2048x8192 | 11.02 | 10.73 flydsl_tuned | 0.97x loss |
| softmax 8192x4096 | 20.72 | 20.34 flydsl | 0.98x tie |
| softmax 8192x5120 | 25.54 | 25.64 flydsl | 1.00x tie |
| softmax 8192x8192 | 39.27 | 39.69 flydsl | 1.01x tie |
| quantize 1x4096 | 3.82 | 4.14 inductor | 1.08x win |
| quantize 1x5120 | 3.81 | 4.14 inductor | 1.09x win |
| quantize 1x8192 | 3.82 | 4.15 inductor | 1.09x win |
| quantize 32x4096 | 3.92 | 4.22 triton | 1.08x win |
| quantize 32x5120 | 3.94 | 4.22 inductor | 1.07x win |
| quantize 32x8192 | 3.91 | 4.13 triton | 1.06x win |
| quantize 2048x4096 | 4.77 | 4.98 triton_tuned | 1.04x win |
| quantize 2048x5120 | 5.69 | 5.85 triton_tuned | 1.03x win |
| quantize 2048x8192 | 8.52 | 8.35 triton_tuned | 0.98x loss |
| quantize 8192x4096 | 15.83 | 15.48 triton_tuned | 0.98x loss |
| quantize 8192x5120 | 19.31 | 19.69 triton_tuned | 1.02x tie |
| quantize 8192x8192 | 29.12 | 28.65 triton_tuned | 0.98x tie |
| add 1x4096 | 3.81 | 3.81 torch | 1.00x tie |
| add 1x5120 | 3.91 | 3.86 torch | 0.99x tie |
| add 1x8192 | 3.86 | 3.90 torch | 1.01x tie |
| add 32x4096 | 3.93 | 3.84 torch | 0.98x loss |
| add 32x5120 | 3.92 | 3.95 torch | 1.01x tie |
| add 32x8192 | 3.84 | 3.99 torch | 1.04x win |
| add 2048x4096 | 8.22 | 7.37 triton_tuned | 0.90x loss |
| add 2048x5120 | 10.06 | 9.78 triton_tuned | 0.97x loss |
| add 2048x8192 | 14.87 | 14.39 inductor | 0.97x loss |
| add 8192x4096 | 28.15 | 28.17 triton_tuned | 1.00x tie |
| add 8192x5120 | 34.29 | 33.94 triton_tuned | 0.99x tie |
| add 8192x8192 | 69.90 | 67.50 triton_tuned | 0.97x loss |
| mul 1x4096 | 3.89 | 3.89 torch | 1.00x tie |
| mul 1x5120 | 3.90 | 3.81 torch | 0.98x loss |
| mul 1x8192 | 3.80 | 3.89 torch | 1.02x win |
| mul 32x4096 | 3.80 | 3.85 torch | 1.01x tie |
| mul 32x5120 | 3.91 | 3.94 torch | 1.01x tie |
| mul 32x8192 | 3.83 | 3.85 torch | 1.01x tie |
| mul 2048x4096 | 8.22 | 7.38 triton_tuned | 0.90x loss |
| mul 2048x5120 | 10.07 | 9.77 triton_tuned | 0.97x loss |
| mul 2048x8192 | 14.88 | 14.39 inductor | 0.97x loss |
| mul 8192x4096 | 28.13 | 28.15 triton_tuned | 1.00x tie |
| mul 8192x5120 | 34.31 | 33.98 triton_tuned | 0.99x tie |
| mul 8192x8192 | 69.85 | 67.82 triton_tuned | 0.97x loss |
| exp 1x4096 | 3.91 | 3.91 torch | 1.00x tie |
| exp 1x5120 | 3.89 | 3.81 torch | 0.98x tie |
| exp 1x8192 | 3.92 | 3.85 torch | 0.98x tie |
| exp 32x4096 | 3.83 | 3.83 torch | 1.00x tie |
| exp 32x5120 | 3.95 | 3.82 torch | 0.97x loss |
| exp 32x8192 | 3.93 | 3.85 torch | 0.98x loss |
| exp 2048x4096 | 5.22 | 5.51 triton_tuned | 1.05x win |
| exp 2048x5120 | 6.50 | 6.45 triton_tuned | 0.99x tie |
| exp 2048x8192 | 10.49 | 10.50 triton_tuned | 1.00x tie |
| exp 8192x4096 | 19.52 | 19.49 triton_tuned | 1.00x tie |
| exp 8192x5120 | 23.89 | 23.73 triton_tuned | 0.99x tie |
| exp 8192x8192 | 37.19 | 37.08 triton_tuned | 1.00x tie |
| silu_mul 1x14336 | 3.92 | 4.13 triton_tuned | 1.06x win |
| silu_mul 1x17408 | 3.92 | 4.11 triton_tuned | 1.05x win |
| silu_mul 1x28672 | 3.91 | 4.11 triton_tuned | 1.05x win |
| silu_mul 32x14336 | 3.96 | 4.26 triton_tuned | 1.08x win |
| silu_mul 32x17408 | 3.96 | 4.25 inductor | 1.07x win |
| silu_mul 32x28672 | 4.06 | 4.32 triton_tuned | 1.06x win |
| silu_mul 2048x14336 | 25.23 | 25.20 triton_tuned | 1.00x tie |
| silu_mul 2048x17408 | 30.23 | 30.33 triton_tuned | 1.00x tie |
| silu_mul 2048x28672 | 58.69 | 60.80 triton_tuned | 1.04x win |
| silu_mul 8192x14336 | 117.00 | 127.63 triton_tuned | 1.09x win |
| silu_mul 8192x17408 | 143.62 | 155.71 triton_tuned | 1.08x win |
| silu_mul 8192x28672 | 228.23 | 248.74 triton_tuned | 1.09x win |
| rope 1x32x128 | 3.89 | 4.15 triton | 1.07x win |
| rope 1x40x128 | 3.89 | 4.15 triton | 1.07x win |
| rope 1x64x128 | 3.91 | 4.20 triton | 1.07x win |
| rope 32x32x128 | 3.94 | 4.11 triton | 1.04x win |
| rope 32x40x128 | 3.90 | 4.24 triton_tuned | 1.09x win |
| rope 32x64x128 | 3.91 | 4.23 triton | 1.08x win |
| rope 2048x32x128 | 5.60 | 5.76 triton_tuned | 1.03x win |
| rope 2048x40x128 | 6.45 | 6.16 triton_tuned | 0.96x loss |
| rope 2048x64x128 | 10.71 | 10.10 triton_tuned | 0.94x loss |
| rope 8192x32x128 | 20.31 | 19.83 triton_tuned | 0.98x loss |
| rope 8192x40x128 | 25.14 | 24.18 triton_tuned | 0.96x loss |
| rope 8192x64x128 | 39.20 | 39.00 triton_tuned | 0.99x tie |
