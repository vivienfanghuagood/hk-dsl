# gfx1201's large-M GEMM: winners that change with the compiler

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 51](51-gfx1201-small-m-gemm.md) shipped gfx1201 schedules for 32-512 rows. From
1024 rows hk's bf16 GEMM still ran gfx1100's default tile: 139-144 TFLOP/s, 71-73% of the
WMMA peak, and 6-8% ahead of Triton ([chapter 48](48-gfx1201-ceilings-and-first-scoreboard.md)).
This chapter tunes that range: Qwen3's four (N, K) at 1024, 2048, 4096 and 8192 rows.

## Two compilers, two answers

`hk.tune` ran twice, with W rotated past the Infinity Cache: once in rocm/pytorch's ROCm
7.2.3 (clang 22), once in the R9700 vLLM image (TheRock ROCm 7.14, clang 23).

| compiler | shapes with a winner | gain over the default |
|---|---|---|
| clang 23 (vLLM image) | 11 of 16 | 3.5-6%: a 32-deep dot slice, one N split |
| clang 22 (ROCm 7.2.3) | 3 of 16 | 3-4%; the other 13 within 1-3% (ties) |

`tools/rdna4-probes/smallm_cross.py` timed each compiler's champions under the other, in one
process, against the default:

- **7 shapes** (2048-8192 x 4096 x 4096, 2048-8192 x 6144 x 4096, 1024 x 24576 x 4096):
  clang 23's champion is 4.1-6.1% faster than the default under clang 23 and within 1% of it
  under clang 22.
- **4 shapes** (4096 and 8192 rows of 24576 x 4096 and of 4096 x 12288): clang 23's champion
  is 2-5% *slower* than the default under clang 22. Clang 22's champion there is faster than
  the default under both, 4.1-5.7% under clang 23 and 2.3-3.4% under clang 22.

Chapter 51's small-M champions agreed across compilers to 1%. At these sizes they do not.
`routes_gemm_gfx1201.json` gains 11 schedules: clang 23's at the first seven shapes and clang
22's at the other four. No shape is more than 1% slower than the default under either
compiler.

## Asking another arch's records

While checking them, the IR tier under `HK_ARCH=gfx1201` failed one test:
`hk.schedules.lookup(arch="gfx1100")` returned a gfx1201 schedule, which is illegal on
gfx1100. `hk.ops.gemm.schedule_for` and `user_route` read the records of the *current*
arch whatever arch the caller asked about, and so did `hk.templates.gemm.plan` and
`hk.tune`'s incumbent. They take an `arch` now, and the records they read are that arch's.
Until then no shipped gfx1201 schedule had been illegal on gfx1100, so nothing had shown it.

## In vLLM

From 1024 rows the gfx1201 vLLM table already routes these shapes to hk
([chapter 51](51-gfx1201-small-m-gemm.md)). The exception is o_proj (4096 x 4096) past 4096
rows, which stays on vLLM's path. So the table stays as it is, and the shipped schedules
change only how fast those calls run. A `route_tune.py` re-measurement with them
moved no M above 1024. Below 512 rows it flipped routes in both directions, which the
schedules there (unchanged) cannot explain, so it is not shipped.

Gate C (`python -m hk.integration ab`, Qwen3.5-9B, 5 rounds, CUDA graphs):

| prompt x requests | prefill, hk / stock | request |
|---|---|---|
| 1024 x 1 | 1.094x | noise |
| 1024 x 16 | 1.137x | 1.091x |
| 4096 x 1 | 1.111x | noise |
| 4096 x 16 | 1.125x | 1.098x |

Tokens were identical to stock's, 72 of 72, at every point. Chapter 50's run of the same
configuration, without these schedules, had prefill at 1.113-1.141x. The difference is
inside the spread of two runs in two processes. The linear layers are part of a prefill
step, and these GEMMs got 4-6% faster, so the step should gain a few percent. This box does
not resolve that across processes (the A/B arms are stock against hk, not hk against hk).
The evidence for the gain is the kernel measurement above.

## What is left

At the shipped shapes the bf16 GEMM from 1024 rows now runs 4-6% faster in the vLLM image.
Under ROCm 7.2.3 it is 2-3% faster at four shapes and within 1% at the rest. Larger gains would need more than a schedule: a compiler-keyed table, as
attention has (`_TILING_FROM_CLANG`), would let each compiler run its own champion. At these
shapes that is worth 2-5% more under clang 22, and the RDNA4 line pauses here.
