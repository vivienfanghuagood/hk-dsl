# gfx1201's small-M GEMM: a loss that was half the cache, and schedules shipped for 32-512 rows

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 48](48-gfx1201-ceilings-and-first-scoreboard.md) left one loss on the
gfx1201 GEMM. Against rocBLAS / hipBLASLt free to pick a layout (a weight transposed
once at load), hk lost below 256 rows by up to 1.75x. Its comparison read one weight
back to back. At 4096 x 4096 that weight is 32 MB, and it stays in the R9700's 64 MB
Infinity Cache from the second call on. [Chapter 49](49-gfx1201-attention-fp8-and-flydsl-from-source.md)
had already found the same trap in the fp8 decode rows. This chapter measures the
loss as a model meets it, then removes it.

## Half of it was the cache

`tools/rdna4-probes/small_m_cold.py` times hk's NT GEMM against hipBLASLt in NT,
NN (W stored K x N) and TT (C^T = W A^T) twice: once with one W read back to back
("hot"), and once with copies of W rotated past twice the cache ("cold", read from
GDDR6 the way a model's next layer reads its weight). At 4096 x 4096 with gfx1100's
schedules, as hk ran on gfx1201 until now:

| rows | hk / the library's best layout, hot | cold |
|---|---|---|
| 16 | 0.57x | 0.75x |
| 64 | 0.58x | 0.73x |
| 256 | 0.93x | 0.81x |

Cold, the loss shrinks but stays: 0.73-0.81x. The library's best layout reaches
0.60 TB/s of the card's 0.62. hk's default 128 x 128 tile reaches 0.43-0.46 TB/s,
because 64 rows of 4096 columns make 32 workgroups for 64 CUs.

## Tuned under both compilers

`hk.tune` searches hk's own schedule space for one shape and keeps a winner only past
its acceptance rule, with W rotated past the cache. It ran on Qwen3's four (N, K) at
32, 64, 128, 256 and 512 rows, twice: in rocm/pytorch's ROCm 7.2.3 (clang 22), and in
the R9700 vLLM image (TheRock ROCm 7.14, clang 23).
[Chapter 50](50-gfx1201-in-vllm.md) found that the two compilers spill in opposite
places, so a schedule measured under one proves nothing about the other.

The champions are 32- or 64-row tiles, mostly 64 columns wide, with one N split.
They are 3-40% faster than the default at 32-256 rows and 6-15% at 512. Under clang 22
five shapes at 256 and 512 rows were ties, not wins; under clang 23, one.

`tools/rdna4-probes/smallm_cross.py` then timed each compiler's champion under the
other, in one process, against the default:

- At 13 of the 15 shapes both compilers have a champion for, the two are within 1%
  of each other under either compiler.
- At 256 x 6144 x 4096, clang 22's champion is 5-7% faster under both.
- At 32 x 6144 x 4096, both champions are the decode GEMV (combine depth 2 against
  8); depth 8 is 2.5% faster under clang 22 and equal under clang 23.
- At 512 x 24576 x 4096, clang 23's champion is 7% faster than the default under
  clang 23 and 3% slower under clang 22, so that shape keeps the default.

That leaves 18 schedules, each the faster of the two everywhere it was timed:
1.03-1.40x the default under either compiler (`python/hk/tuned/routes_gemm_gfx1201.json`).
The one at 1.03x is 256 x 4096 x 12288 under clang 22; clang 23 has it at 1.13x.

## Where hk reads them

`hk.tune` writes its winners to this machine's record file. The shipped file has the
same format, for an arch hk ships records for, and
`hk.tune.records.shipped_route` reads it behind the machine's own records.
`hk.ops.gemm.user_route` asks for this machine's record first, then the shipped one.
So `matmul`, `schedule_for`, `hk.templates.gemm`'s plan and the vLLM route's "hk" path
all take a shipped schedule unless the machine tuned the shape itself.

`hk.tune`'s incumbent on an RDNA arch is now the shipped record too, so a later tune
has to beat it. `HK_AUTOTUNE=0` ignores the shipped records as it ignores every tuning
record. gfx950's file of that name is `gemm_cdna`'s own table and is not read here.

## Against the field, cold

`small_m_cold.py` again, now through `hk.ops.gemm.matmul` with the shipped records,
under ROCm 7.2.3, W from GDDR6 (TFLOP/s ratios, 32-512 rows):

| (N, K) | hk / hipBLASLt NT (a linear layer's layout) | hk / the library's best layout |
|---|---|---|
| 6144 x 4096 | 1.28-1.68x | 0.99-1.02x |
| 4096 x 4096 | 1.41-3.21x | 0.97-1.04x |
| 24576 x 4096 | 1.14-1.67x | 0.94-1.04x |
| 4096 x 12288 | 1.27-5.69x | 0.92-1.09x |

In the layout a linear layer has, hk wins every row. Against a library given the
transposed weight, the shapes from 32 to 128 rows tie or win. The losses left are at
256 and 512 rows, 0.92-0.97x, where the library's NN kernel streams at the bandwidth
roof and hk's is a few percent short. Two of those rows are shapes without a shipped
schedule: 512 x 24576 x 4096 and 512 x 4096 x 12288. In the vLLM image, against
`F.linear`, the tuned schedules are 0.99-1.31x (`hk.tune`'s own baseline column).

## In vLLM

`tools/hk-bench/route_tune.py` re-measured the gfx1201 vLLM table in the image with
the shipped schedules (Qwen3.5-9B and Qwen3-1.7B, every M, the engine's own path
against hk's):

| (N, K) | M 33-256 before | after |
|---|---|---|
| qkv 6144 x 4096 | vLLM's path | hk at 33-128 and 12% of 129-256 |
| o 4096 x 4096 | vLLM's path (hk's decode GEMV to 64) | hk |
| gate_up 24576 x 4096 | vLLM's path | hk at 65-128 and 24% of 129-256 |
| down 4096 x 12288 | vLLM's path to 64, hk from 65 | hk |

The table ships with those four (N, K), as before. The rest of a model's shapes
are measured at load, with the default schedule.

Gate C, `python -m hk.integration ab`, ran one process with three arms: stock, the
new table and the old one (`--routes`). Qwen3.5-9B, 1024-token prompts, 32, 64 and
128 requests, 5 rounds, CUDA graphs:

- **Tokens:** both arms' greedy tokens were identical to stock's, 192 of 192 at
  every batch.
- **Request time:** 1.11x, 1.08x and 1.09x stock with the new table, all past the
  band. The old table's was 1.08x, 1.01x and 1.06x, and only the last was past its
  band.
- **Prefill:** 1.12x stock with the new table, 1.09x with the old one (band 2%).
- **Decode step:** noise in all three arms. The band is 14-35% on this shared box,
  well past what a 32-512-row GEMM changes in a step. The medians move the expected
  way: 38.1 ms against stock's 46.0 at 32 requests, 135.6 against 154.8 at 128.
  The evidence for decode is `route_tune`'s per-M measurement inside the engine,
  where every M routed to hk won by at least the table's 3% margin.

## What is left

- At 256 and 512 rows the library's NN kernel, given a transposed weight, is 3-8%
  ahead. An engine does not transpose its weights, so `F.linear` is what hk is
  measured against there.
- Shapes outside the four (N, K), Qwen3.5's own projections among them, are measured
  at load against the default schedule; tuning them takes `python -m hk.tune` on the
  machine.
- Up to 16 rows the decode GEMV already routes by `decode_gemm`'s schedules; no
  shipped record covers it.
