# gfx1201's ceilings, and where hk stands there out of the box

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 47](47-gfx1201-on-hardware.md) left gfx1201 correct and untuned:
every plan and schedule gfx1100's. Before tuning anything, two questions:
what can a Radeon AI PRO R9700 do, and where does hk already stand against
what a gfx1201 user runs today? Measured on one R9700 of the shared dev box,
the card idle but for the run, rocm/pytorch rocm7.2.3 (torch 2.10.0, hipBLASLt
and rocBLAS of ROCm 7.2.3, triton 3.6.0; FlyDSL is not in that image).

## The ceilings

| | R9700 (gfx1201, 64 CUs) | W7900D (gfx1100, 96 CUs) |
|---|---|---|
| bf16 / f16 WMMA, back to back on every CU | **197.1 / 198.5 TFLOP/s** | 100.5 |
| fp8 x fp8 WMMA | 371.8 TFLOP/s | -- |
| iu8 WMMA | 373.8 TOP/s | 96 |
| 2:4 sparse bf16 (SWMMAC 16x16x32) | 391.5, dense-equivalent | -- |
| an add, cache-resident (best footprint) | 1.81 TB/s at 48 MB (64 MB cache) | 2.43 at 80-96 MB (96 MB) |
| past the Infinity Cache | 0.62 TB/s (datasheet 640 GB/s) | 0.72-0.75 (864) |

`tools/rdna4-probes/wmma_peaks.hip` and `scoreboard.py --probe-ceilings`. A
gfx1201 CU issues twice the WMMA FLOPs of a gfx1100 one, at a higher clock:
197 TFLOP/s is 128 SIMD32s x ~3.0 GHz x 512 FLOP a cycle, above the
datasheet's 191 at the boost clock. The memory side went the other way: a
narrower bus and a smaller cache. So the R9700 is twice the W7900D's compute
over three quarters of its bandwidth, and anything that was balanced on
gfx1100 is further toward memory-bound here.

Until now gfx1201 had no entry in the ceiling tables, and the screens fell
back to gfx1100's: `_rounds.use()` kept the W7900D's 122.6 TFLOP/s, so a
gfx1201 GEMM round faster than 141 TFLOP/s would have been rejected as
"faster than the card can" -- hk's 4096³ runs at 144. The R9700's figures are
now in `_rounds.CARDS`, `hk.check.CARDS`, `hk.profile.analyze.CEILINGS`,
`hk.integration.headroom.PEAKS`, `hk.tune`'s cost cards and stats, and
`hk.target.gfx1201.CEILINGS`. With a cost card, hk.tune has a GEMM space on
gfx1201 (`tests/hk/gpu/test_tune.py` runs there). And `--probe-ceilings`
itself could not run: on gfx1201 under ROCm 7.2.3 `copy_` is a DMA, as on
gfx950, and left no kernel record to time; a `neg` stands in wherever `copy_`
leaves none.

## Out of the box, against the field

The scoreboard's out-of-the-box arms (median of five queued rounds; torch is
hipBLASLt for the GEMM and AOTriton for attention; Triton is the reference
kernel and its autotuned twin over gfx1100's space):

| case | hk | best other | ratio |
|---|---|---|---|
| GEMM 4096x4096x4096 | 0.956 ms, 144 TFLOP/s | Triton 1.031 | **1.08x** |
| GEMM 2048x6144x4096 | 0.740 | Triton tuned 0.791 | 1.07x |
| GEMM 2048x4096x4096 | 0.489 | Triton 0.525 | 1.07x |
| GEMM 2048x24576x4096 | 3.017 | Triton tuned 3.239 | 1.07x |
| GEMM 2048x4096x12288 | 1.535 | Triton tuned 1.634 | 1.06x |
| attention n4096 causal, GQA 4 | 1.690 ms, 81 TFLOP/s | AOTriton 5.623 | **3.33x** |
| attention n4096 full, GQA 4 | 2.738, 100 TFLOP/s | AOTriton 5.129 | **1.87x** |
| softmax 2048x4096 / 8192x4096 | 0.0211 / 0.2196 | Triton tuned 0.0250 / 0.2189 | 1.19x / tie |
| rms_norm 1 / 32 / 2048 / 8192 rows | 2.3 us / 2.6 / 21.2 / 220 | | tie / 1.07x / 1.03x (0.97x tuned Triton) / tie |
| fused add + rms_norm 1 / 2048 rows | 2.2 us / 42.9 | | 1.09x / 1.02x |
| fused add + rms_norm 32 rows | 3.7 us | Inductor 3.3 | **0.88x** |
| fused add + rms_norm 8192 rows | 0.480 ms | Triton tuned 0.446 | **0.93x** |

And the GEMM against both BLAS libraries directly, each in every layout,
since only one of them is free for a linear layer (`F.linear` is NT: A
row-major, the weight N x K):

| M x N x K | hk (NT) | rocBLAS / hipBLASLt, NT | their best layout |
|---|---|---|---|
| 4096 x 4096 x 4096 | 129 TFLOP/s | 92 / 92 | 130 / 131 (TT) |
| 8192 x 8192 x 4096 | 128 | 121 / 121 | 123 / 122 (NN) |
| 4096 x 24576 x 4096 | 127 | 122 / 122 | 123 / 123 (TT) |
| 1024 x 4096 x 4096 | 126 | 105 / 110 | 119 / 119 (TT) |
| 256 x 4096 x 4096 | 109 | 61 / 62 | 124 / 124 (NN) |
| 64 x 4096 x 4096 | 43 | 14 / 15 | 75 / 75 (TT) |
| 16 x 4096 x 4096 | 14 | 8 / 8 | 19 / 19 (TT) |

(A different process from the scoreboard's, so 4096³ reads 129 here, 144
there.) In the layout a linear layer has, hk is 1.05-2.9x either library at
every shape. Where a library may pick a layout -- a weight transposed once
at load -- it ties hk from 1024 rows up and beats it below 256 rows by up to
1.75x. The two libraries are the same here to within 1-4%.

## What this leaves for tuning

Every gfx1201 number above is gfx1100's plan or schedule on a card with twice
the compute a CU and three quarters of the bandwidth. The losses are the
fused add + RMSNorm at 32 and 8192 rows (its gfx1100 plan and LDS reserve),
and the small-M GEMM against a library given its best layout. The GEMM's 144
TFLOP/s is 73% of the WMMA ceiling.
