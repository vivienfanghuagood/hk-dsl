# Goal: a few kernels that are hot, under-optimised everywhere, and won by hk

Status: proposed 2026-10-24, the milestone after the tuning goal
([tuning-goal.md](tuning-goal.md): T1-T3 merged; T4's first sealed run FAILED
on gfx1100's floor, record in `bench/tune/agent-runs/t4-20261024-a`).
Refocused the same day: the first draft (E1-E5: every model, every kernel
family) spread the work across kernels where the best AMD library is already
near the hardware's limit. This version picks the kernels where hk can win
something that matters, from measurements, and pauses the rest.

## How a kernel is chosen

A kernel is worth hk's effort only if all three hold, each measured, not
assumed:

1. **Hot.** At least ~10% of a request's (or a step's) device time in the
   engine as people serve the model, at the shapes it is served at.
2. **Under-optimised by the best baseline there is**, not just by the one
   the engine ships: hipBLASLt, AITER (CK / asm / Triton), FlyDSL, tuned
   Triton, all run on the model's exact shapes in one process, interleaved.
   The best of them is at most ~60% of the roofline (the matrix-core peak,
   or HBM bandwidth over the bytes the operation must move).
3. **Winnable.** A credible path to >= 1.2x over that best baseline -- the
   hardware shows it is possible (e.g. the same chip doing the same work
   faster at another shape), and hk has, or can get, the instructions it
   needs.

## The opportunity tables (2026-10-24)

Measured in the engines' own runs: vLLM's `deepseek_v41/amd` package (nightly
ROCm 10 image) on MI355X, TP2, dummy weights, batch 8 x 1,024 tokens, with
torch-profiler traces (shapes recorded) and rocprofv3 DRAM counters; and
SGLang-Diffusion on one W7900D. Peaks: MI355X 5.03 PF/s dense FP8 (MXFP8),
2.5 PF/s BF16, 8.0 TB/s HBM; W7900D 122.6 TF/s BF16 WMMA. The MXFP8
baselines: `bench/e2e/deepseek_v41/mxfp8_opp.py` (branch `claude/cdna-fp8`).

### MI355X, DeepSeek-V4.1-Flash, prefill (TTFT; chunked, M = rows a step)

Share = of the step's kernel time without the all-reduce's cross-rank wait
(the all-reduce itself is 15-50% and a communication problem, not a kernel).

| kernel | share (M = 5,120) | engine's kernel | best baseline at the model's shapes | roofline | verdict |
|---|---|---|---|---|---|
| **MoE, a8w4 grouped GEMM** (gate/up + SiLU, down; 384 experts top-6, FP4 weights) | **38%** | AITER `fused_moe`: `mfma_moe1_..._t128x128x256` 653 us, `gemm2_a4w4_..._bm64` 431 us a layer | AITER's (the engine's pick per M); the same kernel family (smaller tiles) streams the full 2.3 + 1.1 GB of weights in 360 / 196 us at M = 640 | HBM: 293 / 141 us (all experts touched at M >= 640) | **45% / 33% of roofline -> K1** |
| **MXFP8 32x32-block GEMM** (attention projections, shared expert, indexer) | **19%** | vLLM `rocm_mxfp8_block32_gemm` (Triton `dot_scaled`) | best of vLLM / FlyDSL gfx950 MX bmm / hipBLASLt MX: **21-42% of peak** at M = 2,048-8,192 (18 shapes) | 5.03 PF/s | **-> K2** (hk today: 6-20%) |
| sparse MLA prefill (head_dim 512, one KV head, window 128 + top-512 compressed keys) | 14% | Triton `_sparse_mla_..._FP8_DSV4`, 470 us a layer | Triton is the only one | ~86 us at BF16 peak (~18% now) | **reserve K4** |
| mHC seam (`mhc_fused_post_pre_delayed_rmsnorm`, 4 streams x 5,120) | 12% | Triton, 148 us a layer | Triton | ~53 us at HBM (420 MB): 35% | **reserve K5** |

### MI355X, DeepSeek-V4.1-Flash, decode (TPOT 6.7 ms at batch 8)

| kernel | share | measured | verdict |
|---|---|---|---|
| `dsv41_mono` K1 + K2 (FlyDSL monokernel: a whole decoder layer in two persistent launches -- attention, MoE, both all-reduces in-kernel) | 55% | K2: 522 MB a launch in 124 us = 4.2 TB/s, 52% of HBM | Hot and below roofline, but not a kernel: matching it means re-building a fused layer with in-kernel cross-rank hand-offs. Not now; revisit with the monokernel's own timeline build (`timeline=True`) to see whether its MoE stage is the part below roofline. |
| AITER MoE at M > 48 (the non-mono path) | 19% at batch 8 | `moe1 t32x128x256` 7.1 TB/s, `gemm2_a4w4 bm32` 6.1 TB/s | at roofline: no |
| MXFP8 GEMM, decode M | 5% | 9.1 TB/s by the counters (LLC hits) | at roofline: no |

### W7900D, Qwen-Image 2.1 and MiniMax-H3 (SGLang-Diffusion)

| kernel | share | best baseline | hk now | verdict |
|---|---|---|---|---|
| **DiT attention** (non-causal, head_dim 128; Qwen B1 H32 N4,096; H3 H56 N14,720) | stock: 31% (Qwen), ~44% of H3's denoise; with the best baseline in: ~13% / ~21%; grows with N^2 for longer clips and larger frames | tuned Triton flash: 48-50% of peak (AOTriton, what torch ships: 17%) | 52-53%, integrated (PR #21: 1.50x Qwen, 1.43x H3 denoise) | **-> K3** (the winnable part is now 50% -> 70%+ of peak) |
| DiT GEMM (Qwen 4096 x {12288, 4096} x {4096, 12288}; H3 14720 x {21504, 28672, 5376} x {5376, 7168}) | 73% of Qwen's device time | FlyDSL 71-73% (hipBLASLt 56-58%, Triton 66-68%) | 72-74%, integrated with per-shape fallback | best baseline above 60%: **paused** |
| H3 VAE decode (MIOpen convolutions) | ~30% of H3's request (109 s of 345 s stock) | not measured | -- | **measure next**: candidate if MIOpen is far from roofline |

## The core kernels

| id | kernel | GPU, model | target |
|---|---|---|---|
| **K1** | MoE a8w4 grouped GEMM for prefill: gate/up with fused SiLU-mul, and down, FP8 activations x MXFP4 weights (`v_mfma_scale_f32_16x16x128_f8f6f4`), AITER's weight shuffle and routing order | MI355X, DeepSeek-V4.1-Flash | >= 1.3x AITER's per layer at M = 2,048-16,384, toward the weight-streaming time the same chip shows at M = 640; then prefill TTFT faster in vLLM, tokens identical |
| **K2** | MXFP8 block-scaled GEMM (E8M0 per 1x32 activations, 32x32 weights) at prefill M | MI355X, DeepSeek-V4.1-Flash | >= 1.2x the best of vLLM / FlyDSL / hipBLASLt on the 6 projection shapes at M >= 2,048 (best today 21-42% of peak; target >= 55%); bit-identical to vLLM's kernel as hk's is now |
| **K3** | DiT attention, non-causal, head_dim 128, BNHD in place | W7900D, Qwen-Image 2.1 / MiniMax-H3 | >= 70% of peak (1.3x tuned Triton) at N = 4,096-60,000; whole-request gain reported for Qwen 1024^2 and H3 clips at served lengths |
| K4 (reserve) | sparse MLA prefill, head_dim 512, MQA, gathered keys | MI355X | taken up when K1 or K2 meets its target |
| K5 (reserve) | mHC seam: the 4-stream fold, Sinkhorn gate and RMSNorm in one pass at HBM speed | MI355X | same |

Each kernel is done when (a) its scoreboard against every baseline above
shows the target, on the model's shapes, interleaved, in one process; (b) the
engine runs it through `hk.integration` without patching the engine's tree;
(c) the end-to-end A/B is outside the measured noise, outputs identical (or
images within the stock path's own run-to-run variation); (d) the steps are
written as a replayable loop ([docs/e2e-loop.md](../e2e-loop.md)) so a
developer's harness can do the next one.

K1 and K2 share the scaled-MFMA path hk already has (FP8 operands, E8M0
scales, bit-identical to vLLM, but at 6-20% of peak: the matrix unit is 16%
busy). Raising that main loop is the first step of both.

## Paused

- gfx950 bf16 GEMM: the pingpong's main loop (the first draft's E5) and more
  dense-GEMM schedule work (tuning-goal K4): DeepSeek-V4.1's GEMMs are not bf16.
- MXFP8 at decode M, MoE at decode M > 48: at roofline.
- The decode monokernel: see above.
- W7900D DiT GEMM: hk is at parity with FlyDSL, the best baseline, at 72-74%
  of peak; it stays integrated (per-shape measured routing) but gets no new
  work.
- MiniMax-H3 as a target of its own (the first draft's E2): it is K3's
  validation scenario; its offload and VAE stages are measured, not
  optimised, until the VAE measurement says otherwise.

## Done before the refocus

- Qwen-Image 2.1, 1024^2, 30 steps, one W7900D: 42.2 s -> 28.2 s (1.50x),
  hk attention + GEMM via `hk.integration sglang-diffusion` (PR #21,
  [chapter 46](../history/46-qwen-image-on-rdna3.md)).
- MiniMax-H3 4 s clip: denoise 1.43x with hk attention, the request 1.15x
  (median of three interleaved runs; the VAE decode stage is noisy).
- DeepSeek-V4.1-Flash: the profiling path in vLLM's AMD package (dummy
  weights, TP2, traces with shapes, DRAM counters), the tables above; an
  MXFP8 GEMM in hk bit-identical to vLLM's (branch `claude/cdna-fp8`).

## Not in scope

- Lossy speedups as hk's result (cache-dit, fewer steps, lower-bit weights
  than the checkpoint ships).
- Communication kernels (the all-reduce is the largest non-kernel item in
  prefill; that belongs to vLLM / RCCL / quick-reduce).
- Training kernels.
