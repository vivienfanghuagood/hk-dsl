# gfx950's leftovers: the weights streamed past the caches, a RoPE inside one DPP row, and plans that hold wherever the buffers sit

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Two gfx950 items were left open. [Chapter 33](33-cdna-gemm.md) got hk's own
GEMM to 0.9x or better of the best of hipBLASLt, FlyDSL's `gemm_a16w16` and
HipKittens at 108 of the 122 Llama-3-8B/Qwen3-8B linear points; the 14
misses were M = 256-512 at wide N (0.81-0.89x) and M = 32-128 where the
weights stream (0.83-0.90x). [Chapter 37](37-cdna-rows.md) got the norms,
elementwise and fused kernels to tie or win 99 of 132 cases against torch,
Inductor, tuned Triton and FlyDSL; the 33 others were 0.90-0.98x.

Now: **122 of 122** GEMM points at 0.9x or better, as shipped, the median of
three processes; **120 of 132** row cases tie or win. Measured on one
MI355X, rocm/pytorch 2.10 (ROCm 7.2.3), one process per run.

## The GEMM: what the misses were short of

At M = 32-64 a linear is a weight stream: 24576 x 4096 bf16 is 201 MB,
read once, and hipBLASLt read it at 5.2-6.1 TB/s where hk's four-warp
kernel reached 4.4-4.9. At M = 256-512 the pingpong's 256 x 256 blocks were
a wave and a fraction of the 256 CUs (2048 x 12288 is 384 blocks). Five
kinds of candidate went into `gemm_cdna.CANDIDATES`:

- **Weights streamed.** Chapter 39's copy findings moved to the GEMM: B
  allocated as `hk.lds.alloc(shape="16x64")` (whole 128-byte lines a copy)
  and copied `nontemporal` (`sc0 nt`), with 128-256-deep K-tiles. A stays
  cached: every block of N reads it again. `dp32x48nt`, two warps of
  32 x 48, puts 256 blocks on N = 12288, one a CU, and streams 5.7 TB/s at
  M = 32 against hipBLASLt's 4.8 (17.7 against 21.0 us). `dp32x96nt`,
  `dp32x64nt`, `dp32nt`, `dp64nt` and the split `dp32x96ntk4` cover the
  other N.
- **Blocks that fill the CUs.** 96 x 256 = 24576 and 224 x 128 = 28672: a
  whole number of workgroups a CU where a power of two leaves a part wave
  (`dp64x96`, `dp128x96`, `dp128x224`, `dp128x64`, `dp128x0`).
- **The wave split** (`ppw`, `ppwd`): the pingpong on the columns whose
  256 x 256 blocks are whole waves of the CUs, another candidate for the
  rest of N: 2048 x 12288 went from 0.915 to 0.983x hipBLASLt.
- **The pingpong's tail.** N = 151936's last 128 columns ran as one
  128-wide block row walking all of K, 41-50 us at K = 4096; split along K
  by M (`tail_for`), 8-20 us.
- **The split-K reduce on strips**: 256 workgroups on a 128 x 4096 output,
  4.3 us where `reduce_kernel`'s 16 x 128 tiles took 9.2 (hipBLASLt 4.4 for
  its own eight shares).

One bug surfaced on the way: include/cdna4's `st_subtile` addresses whole
subtiles of a named layout, so a 32-column window of st_16x64 read its
subtile's first 32 columns whatever its index, and the four-warp kernel
with 32-deep slices gave wrong answers. HKToCpp now refuses a window that is
not whole subtiles of its layout, and the four-warp kernel reads 64 deep
on st_16x64.

## The harness, and the table checked as shipped

Chapter 33's harness took the best of five rounds per arm, across about 20
candidates: a best-of-N that flatters hk. `tools/hk-bench/gemm_cdna_routes.py`
now queues each round behind a GPU spin (chapter 32's queue), rotates which
arm goes first, and records the median of seven rounds beside the best and
the median deviation. `--routed` times `hk.ops.gemm_cdna.matmul` -- the
shipped route, one arm -- against the same baselines.

The full sweep with every candidate (`routes_full3`) gave 119 of the 120 grid
points at 0.9x or better; the miss, the Llama lm_head at M = 64 (0.87x:
dp64nt 347 TFLOP/s against hipBLASLt's 399), sits inside hipBLASLt's own
spread across processes there (357-405). 73 of the 122 routes changed from
chapter 33's table. Routed, in three separate processes, every point's
median is at 0.9x or better:

| N x K | M=1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | 1.36 | 1.06 | 1.00 | 1.19 | 1.57 | 1.23 | 0.93 | 0.92 | 1.10 | 1.26 | 0.97 | 0.96 |
| 6144 x 4096 | 1.11 | 1.15 | 1.03 | 1.10 | 1.26 | 0.93 | 0.97 | 1.00 | 0.93 | 0.95 | 0.99 | 0.95 |
| 12288 x 4096 | 1.42 | 1.21 | 1.21 | 1.20 | 0.92 | 0.97 | 0.98 | 0.93 | 0.94 | 0.99 | 0.94 | 0.95 |
| 14336 x 4096 | 1.16 | 1.04 | 1.03 | 1.02 | 0.91 | 1.00 | 1.17 | 1.20 | 0.97 | 0.96 | 0.99 | 0.95 |
| 24576 x 4096 | 1.31 | 0.99 | 1.20 | 1.05 | 1.00 | 1.16 | 0.96 | 0.92 | 0.91 | 0.95 | 0.94 | 0.92 |
| 28672 x 4096 | 1.02 | 1.08 | 1.01 | 1.12 | 0.95 | 1.10 | 0.95 | 0.95 | 0.98 | 1.00 | 0.93 | 0.93 |
| 4096 x 12288 | 1.11 | 1.10 | 0.95 | 1.26 | 1.06 | 0.98 | 1.00 | 1.03 | 1.11 | 1.14 | 0.97 | 0.96 |
| 4096 x 14336 | 1.06 | 1.03 | 1.06 | 1.14 | 1.00 | 0.96 | 1.13 | 1.05 | 1.10 | 1.12 | 0.96 | 0.95 |
| 128256 x 4096 | 1.03 | 1.10 | 1.17 | 1.01 | 0.96 | 0.94 | 0.94 | 1.00 | 0.98 | 0.98 | 1.00 | 0.99 |
| 151936 x 4096 | 1.03 | 0.99 | 0.99 | 0.96 | 0.91 | 1.05 | 0.96 | 1.11 | 1.11 | 0.98 | 1.00 | 1.01 |

The squares: 4096³ 1450-1461 TFLOP/s (0.967-0.978x hipBLASLt's 1493-1506),
8192³ 1592-1594 (0.966-0.969x, 1642-1650). The tightest point is the
unmerged gate/up at M = 64 (0.911, 0.910, 0.890): 224 blocks of 64 x 64 on
256 CUs, an eighth of the part idle.

## RoPE in one DPP row

The 4-byte rotary kernel gives a lane two bf16 of a head, so a 128-wide head
is 64 lanes and its halves 32 apart. At eight bf16 a lane -- one 16-byte
access -- a head is 16 lanes of a chunk and its halves eight apart, inside
one DPP row. Two pieces in the dialect make that kernel:

- `hk.rotate_half {span}`: NeoX's `[x1 | x2] -> [-x2 | x1]` on each
  span-column segment of a strip. Halves 8 lanes apart are one
  `row_ror:8`, 16 a `v_permlane16_swap`, 32 a `v_permlane32_swap`, whole
  chunks a register swap; the sign is one XOR.
- `hk.load {period}`: strip column c reads column `idx.c + c % period`, so
  one 512-column chunk -- four heads -- gets the token's 64 angles under each
  head. The run-of-8 angle table is a load, not a table.

`hk.ops.rotary` takes the 16-byte form (`rotary16`) for D = 128 and 256 at
two bytes an element: one warp of two chunks inside 64 MB, a token a
workgroup of four or eight warps past it, ten one-warp workgroups a token
at 40 heads (five warps lost: 24.98 against 24.59 us at 8192 tokens).

## Norms and elementwise

The strip plans aimed every norm at 32 elements a lane. The best target
follows the width: 4096-5120 columns want more warps a row (8-16 a lane),
8192 fewer -- a row in one or two waves, as FlyDSL's tuned softmax holds
128 a lane. rms, layer and softmax may now hold 128 a lane on wave64, and
`_STRIP_PER_LANE_AT` picks the target by (kind, columns, rows band) on
gfx950: rmsnorm 2048x8192 10.86 -> 10.01 us, layernorm 2048x8192 11.36 ->
10.47, softmax 8192x4096 21.07 -> 20.38.

add/mul at 2048 rows won or lost by 20% between runs because the 8 x 8
plan moves with where x, y and o sit relative to each other. Carved out of
one allocation at seven relative offsets, add at 2048x4096 took 6.34-7.97 us
on 8 x 8 and 7.06-7.31 on two warps of eight chunks (tuned Triton
7.27-7.50). 24-64 MB now takes 2 x 8, past the cache two warps of one
chunk.

## The scoreboard now

`scoreboard.py --only llm-norm,llm-quant,llm-pointwise`, chapter 37's cases
and arms, 7 rounds of 50 calls, the verdict a tie within the larger of 2%
and the two arms' noise:

| op | 1-32 rows | 2048 rows | 8192 rows | verdicts |
|---|---|---|---|---|
| rms_norm | 0.99-1.02 | 1.01-1.09 | 1.00-1.02 | 9 tie, 3 win |
| rmsnorm | 0.97-1.01 | 1.02-1.04 | 0.99-1.00 | 1 loss, 10 tie, 1 win |
| layernorm | 1.00-1.05 | 1.01-1.16 | 0.98-1.01 | 2 loss, 4 tie, 6 win |
| fused add + RMSNorm | 0.96-1.02 | 0.95-1.00 | 0.99-1.01 | 3 loss, 8 tie, 1 win |
| softmax | 0.99-1.00 | 0.98-0.99 | 0.99-1.01 | 12 tie |
| int8 quantize | 1.03-1.05 | 0.98-1.03 | 0.98-1.02 | 1 loss, 2 tie, 9 win |
| add | 0.97-1.04 | 0.98-1.05 | 1.00 | 1 loss, 9 tie, 2 win |
| mul | 0.97-1.04 | 0.98-1.05 | 0.99-1.00 | 1 loss, 8 tie, 3 win |
| exp | 0.98-1.02 | 1.00-1.05 | 1.00 | 1 loss, 9 tie, 2 win |
| silu_mul | 1.03-1.05 | 1.00-1.01 | 1.08-1.11 | 3 tie, 9 win |
| RoPE | 1.05-1.06 | 0.97-1.06 | 0.96-0.99 | 2 loss, 3 tie, 7 win |

120 of 132 tie or win (chapter 37: 99). The 12 losses, 0.95-0.98x:

- five at the 3.8-4.0 us launch floor (rmsnorm and fused add at 32x4096,
  add 1x8192, mul and exp 1x5120): 3.90 us against torch's or FlyDSL's
  3.79, 0.1 us that is not in the kernel's body;
- fused add + RMSNorm at 2048x4096 and 2048x8192 (0.97, 0.95x: 20.38 us
  against tuned Triton's 19.41) -- no per-lane target 8-64, capped or not,
  passes 20.17;
- layernorm at 8192x4096 and 8192x8192 (0.98x each, 0.2% outside the band);
- quantize at 2048x8192 (0.98x);
- RoPE at 2048x64 (10.39 against 10.09 us) and 8192x40 (25.25 against
  24.19; 24.59 in the plan sweep, every warps/chunks/LDS-reserve plan
  measured).

## What did not work

- **The grid-stride elementwise form** (workgroup g takes blocks g, g + G,
  ..., every load before the first store): 7.03 us at its best on add
  2048x4096, level with the plain 2 x 8 plan's 7.07. Kept for measuring
  (`STRIP_GS_KERNELS`); nothing routes to it.
- **Non-temporal weights on the pingpong** (`ppnt`): at M = 256-512 the
  lm_head's weights are read by several block rows, and `nt` threw that
  reuse away: 0.74 against the plain pingpong's 0.81x at 256 x 151936,
  0.83 against 0.97 at 512.
- **224-wide blocks at M = 32-64**: 0.81x at 64 x 28672; balanced blocks
  help only where the blocks are big enough to stay compute-fed.
- **32 x 96 and wider two-warp blocks at 256-deep K-tiles**: refused by the
  register estimate (584-668 VGPRs); 32 x 48 at 128-deep tiles is what fits.
- **An LDS reserve capping RoPE's workgroups a CU** (20-80 KB, as Triton's
  `waves_per_eu=2`): no plan moved by more than 1%.

## Tests

The dialect: `rotate_half` and `load {period}` round-trip, and the verifier
refuses a span whose halves are not 8, 16, 32 lanes or whole chunks apart,
a tile, `period` with `bounded`, and part of a run (lit). `tests/hk/ir`:
the templates, the plan for rotary16, the subtile-window refusal, the
streaming blocks' copies, the wave cut and the K-split tail. `tests/hk/gpu`:
every candidate at three more shapes (96-wide, 224-wide, a pingpong tail),
the wave split on two and four CUs, RoPE at 8192 x 40 and 4096 x 64; the
gfx950 tier ran 907 passed and 149 skipped (the skips are mostly candidates
that do not take a test shape), after one test that pinned chapter 37's
elementwise plan was moved to the new one.
gfx1100's and gfx1201's generated sources are identical for every existing
kernel (`tools/cdna-probes/dump_sources.py`; 43 new ones, which RDNA
refuses at trace time with their reason).

## Reproduction

```
# one idle MI355X, rocm/pytorch 2.10 image, cp312 hk._mlir over python/hk/_mlir
export FLYDSL_ROOT=/path/to/FlyDSL HK_ARCH=gfx950
python3 tools/hk-bench/gemm_cdna_routes.py --rounds 7 --out routes.json
for i in 1 2 3; do python3 tools/hk-bench/gemm_cdna_routes.py --routed --rounds 7 --json routed_$i.json; done
python3 tools/hk-bench/scoreboard.py --only llm-norm,llm-quant,llm-pointwise \
    --rounds 7 --calls 50 --json rows.json
```
