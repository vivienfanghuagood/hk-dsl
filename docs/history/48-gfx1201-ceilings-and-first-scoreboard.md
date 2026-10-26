# gfx1201's ceilings, and where hk stands there out of the box

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 47](47-gfx1201-on-hardware.md) left gfx1201 correct and untuned:
every plan and schedule gfx1100's. Before tuning anything, two questions:
what can a Radeon AI PRO R9700 do, and where does hk already stand against
what a gfx1201 user runs today? Measured on one R9700 of the shared dev box,
the card idle but for the run, rocm/pytorch rocm7.2.3 (torch 2.10.0, hipBLASLt
and rocBLAS of ROCm 7.2.3, triton 3.6.0), with FlyDSL 0.4.0 and AITER (main,
ffd08f2, built from source) beside it in a directory of our own.

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
"faster than the card can" -- hk's 4096³ runs at 139-144. The R9700's figures are
now in `_rounds.CARDS`, `hk.check.CARDS`, `hk.profile.analyze.CEILINGS`,
`hk.integration.headroom.PEAKS`, `hk.tune`'s cost cards and stats, and
`hk.target.gfx1201.CEILINGS`. With a cost card, hk.tune has a GEMM space on
gfx1201 (`tests/hk/gpu/test_tune.py` runs there). And `--probe-ceilings`
itself could not run: on gfx1201 under ROCm 7.2.3 `copy_` is a DMA, as on
gfx950, and left no kernel record to time; a `neg` stands in wherever `copy_`
leaves none.

## FlyDSL and AITER on gfx1201

What a gfx1201 user can get from AMD's own kernel stacks, found by installing
them and running what they ship:

- **FlyDSL 0.4.0** (the PyPI release, and the version AITER pins) knows
  gfx1201. Its RDNA4 kernels are one GEMM, `kernels/gemm/rdna_f16_gemm.py`
  (bf16/f16 WMMA, one 128x128x32 tile, A and B^T as `F.linear` has them), and
  an fp8 and an int4 GEMM; its norm and softmax kernels take any arch, and
  their autotuners run there. FlyDSL's own tests for those pass on the R9700
  (60 passed, the rest skipped as gfx11-only or multi-GPU). FlyDSL has **no
  attention for gfx12**: `flydsl_flash_attn_func` routes a dense bf16 call to
  `flash_attn_generic`, which stages K and V through
  `llvm.amdgcn.raw.ptr.buffer.load.lds`, a global-to-LDS load gfx12 does not
  have, and LLVM does not fail the build, it aborts the process: `LLVM ERROR:
  Do not know how to expand this operator's operand`
  (`tools/rdna4-probes/fly_probe.py`). Making it run means a register-staged
  load stage in its place -- a rewrite of the kernel's loads, not a patch --
  so the scoreboard does not call it. Nor does FlyDSL tune its gfx12 GEMM:
  `pick_tile` and the autotuner are the gfx11 kernel's.
- **AITER** ships a FlyDSL attention written for gfx1201,
  `aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_func`: BSHD, one head
  count for Q, K and V (no GQA), head_dim at least 64 and a multiple of 32,
  the sequence padded to 128. It is right against fp32 (rel err 4.8e-3 causal,
  hk's 2.5e-3). Its `rms_norm` and `rmsnorm2d_fwd_with_add` build for gfx1201
  and are right. Its bf16 GEMM entry, `tgemm.mm`, has no tuned row for a
  64-CU gfx1201 and falls back to `F.linear` -- hipBLASLt, the torch column
  again. It has no softmax.

The scoreboard has an `aiter` arm now, counted out of the box: AITER's entry
for the operation as an engine calls it, recording what it dispatched to. For
attention the GQA heads are expanded and the layout made BSHD once, outside
the timing, so the arm times the kernel and not those copies -- an engine
would pay them. The FlyDSL arm runs `rdna_f16_gemm` on gfx12.

## Out of the box, against the field

Every arm in one process (median of five queued rounds; torch is hipBLASLt for
the GEMM and AOTriton for attention; Triton is the reference kernel or its
autotuned twin over gfx1100's space, FlyDSL its default or autotuned kernel,
whichever is faster):

| case | hk | torch | Inductor | Triton | FlyDSL | AITER | hk vs best other |
|---|---|---|---|---|---|---|---|
| GEMM 4096x4096x4096 | **0.986 ms** | 1.502 | -- | 1.050 | 1.207 | 1.500 (hipBLASLt) | 1.06x |
| GEMM 2048x6144x4096 | **0.745** | 0.858 | -- | 0.800 | 0.936 | 0.865 | 1.07x |
| GEMM 2048x4096x4096 | **0.513** | 0.594 | -- | 0.531 | 0.645 | 0.598 | 1.03x |
| GEMM 2048x24576x4096 | **3.111** | 3.361 | -- | 3.320 | 3.732 | 3.387 | 1.07x |
| GEMM 2048x4096x12288 | **1.562** | 1.790 | -- | 1.652 | 1.932 | 1.808 | 1.06x |
| attention n4096 causal, GQA 4 | 1.707 ms | 5.825 | -- | -- | -- (aborts) | **1.410** | **0.83x** |
| attention n4096 full, GQA 4 | 2.778 | 5.294 | -- | -- | -- (aborts) | **2.702** | **0.97x** |
| rms_norm 1 row | **2.4 us** | 5.4 | 2.7 | 2.6 | 2.5 | 3.2 | 1.03x |
| rms_norm 32 rows | **2.2** | 5.1 | 2.8 | 2.7 | 2.8 | 8.3 | 1.25x |
| rms_norm 2048 rows | 21.3 | 28.8 | 22.4 | **20.5** | 21.0 | 50.1 | 0.96x |
| rms_norm 8192 rows | 219 | 225 | 222 | 218 | 219 | 218 | tie |
| fused add + rms_norm 1 row | **2.5 us** | 7.2 | 2.7 | 2.9 | 3.2 | 3.6 | 1.07x |
| fused add + rms_norm 32 rows | **2.6** | 6.5 | 3.4 | 4.0 | 3.3 | 14.3 | 1.30x |
| fused add + rms_norm 2048 rows | 43.0 | 61.0 | 56.4 | 43.4 | 43.1 | 79.0 | tie |
| fused add + rms_norm 8192 rows | 482 | 571 | 460 | **446** | 451 | **446** | **0.93x** |
| softmax 2048 rows | 21.3 us | 32.1 | 65.8 | 25.9 | 21.4 | -- | tie |
| softmax 8192 rows | 219 | 221 | 232 | 219 | 221 | -- | tie |

(Norm rows are 4096 wide. "hk vs best other" is against every other arm, the
tuned ones included.) hk's GEMM is the fastest at every shape; FlyDSL's
gfx12 GEMM, untuned, runs 106-114 TFLOP/s, 0.80-0.85x hk's. AITER's
attention beats hk's: causal by 21% (97 TFLOP/s against 80), full by 3%. Its
causal runs at 96% of its full rate, hk's at 81% of its own, which points at
how hk spreads the causal triangle over 64 CUs rather than at its inner loop. AITER's norms
lose to everything below 8192 rows; at 8192 the fused add ties the best,
hk's 0.93x of it. The norm rows also moved between processes: a first run
had fused add + rms_norm at 32 rows 0.88x Inductor, this one 1.30x
-- a ratio of microseconds from one process is a weak number.

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
the compute a CU and three quarters of the bandwidth. The losses: attention
against AITER's gfx1201 FlyDSL kernel (0.83x causal, 0.97x full), the fused
add + RMSNorm at 8192 rows (its gfx1100 plan and LDS reserve), and the small-M
GEMM against a library given its best layout. The GEMM's 139-144 TFLOP/s is
71-73% of the WMMA ceiling.
