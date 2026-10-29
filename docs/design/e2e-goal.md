# Goal: a few kernels that are hot, under-optimised everywhere, and won by hk

Status: proposed 2026-10-24, the milestone after the tuning goal
([tuning-goal.md](tuning-goal.md): T1-T3 merged; T4's first sealed run FAILED
on gfx1100's floor).
Refocused the same day: the first draft (E1-E5: every model, every kernel
family) spread the work across kernels where the best AMD library is already
near the hardware's limit. This version picks the kernels where hk can win
something that matters, from measurements, and pauses the rest.

**Where it stands (2026-10-26).** All three core kernels and the T-CR testbed are in main. The
measurements, decisions and status notes below are dated records from 2026-10-24/25; for the
current state of each route see [integration](../integration.md#on-by-default-and-turning-it-off)
and [serving readiness](serving-readiness.md).
- **K1** (MoE a8w4): `hk.ops.moe_a8w4` and vLLM's a8w4 route (#34). It runs 1.15-1.18x AITER at
  M >= 8192 and takes only those M (`MIN_M`), so the 1.3x target below was not reached. It is
  deterministic, and its max-rel against exact fp32 is within the K1 criterion below.
- **K2** (MXFP8 GEMM): `hk.ops.mxfp8` with `gemm_cdna`'s MX mode and vLLM's MXFP8 linear route
  (#33), plus the activation's quant fused in (#41). It takes six measured (N, K) at M >= 2048
  and is bit-identical to vLLM's kernel: 1.00-1.48x on the 18 measured points (serving readiness,
  B4), and vLLM's route 1.01-1.58x with the fused quant (#41).
- **End to end**, K1 and K2 together give TP2 TTFT 1.05-1.06x at 8k-16k prompt tokens on two
  MI355X. With real weights, GSM8K and ARC are within stock's run-to-run spread (serving
  readiness, B1/B2).
- **K3** (DiT attention): `hk.ops.attn_k3` (#24), opt-in at first.
- **Default-on (#54):** K1, K2 and K3 are on by default on the GPU each was validated on (K1/K2
  gfx950, K3 gfx1100). `HK_DISABLE=1` turns hk off.
- **T-CR** (ConvRot INT8): int8 on gfx1100 (#31) and the ConvRot linear (#26), opt-in: lossy by
  design. MiniMax-H3's denoise is 1.17-1.22x bf16's.
- **T-SA** ran again from main: the hk arm no longer leaves the DSL.
- Decode on CDNA (the `dsv41_mono` row below) became its own track, the megakernel work
  (serving readiness, A1; [comm.md](comm.md)).

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
baselines: `bench/e2e/deepseek_v41/mxfp8_opp.py`.

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
scales, bit-identical to vLLM). Winnable, measured: gfx950's scaled MFMA runs
at 4.5-4.7 PF/s register-only and hipBLASLt's plain FP8 GEMM at 3.14 PF/s
(8192^3), while every MX GEMM is at 1.25-2.2 PF/s -- the MX penalty is
software. K2 status as of 2026-10-24 (then on a branch; in main since #33 and #41, see above):
a four-wave form (`mx4w`,
accumulators in AGPRs, packed scales) with a one-K-tile loop body (hk-lds-pipeline
now names ring-indexed LDS buffers, so the body no longer has to unroll over the
ring) at 2.43 PF/s at 8192^3; against the best baseline at the 18 prefill points
(FlyDSL at 13 of them) 0.85-1.27x, geomean ~1.06x -- the 1.2x target is not met.
One gfx950 hazard found on the way: a load into scale VGPRs that queued scaled
MFMAs still read corrupts them. A second one suspected at the time, register loads and LDS-DMA
copies retiring out of order on vmcnt, was refuted the same evening by a controlled
micro-benchmark: they retire in issue order.

K3 status as of 2026-10-24 (`hk.ops.attn_k3`, then opt-in with `HK_SDPA_K3=1`; in main since
#24 and on by default on gfx1100 since #54). The kernel is the Sage
experiment's hk-arm structure in bf16: S^T = K Q^T, a lazy running max, fp16 V^T
pre-transposed in the order the P^T operands are cheapest in, the grid's tail
split over KV. It reads SGLang's (B, N, H, D) views in place at any length.
- Kernel: 76-82 TF/s at Qwen / H3 sizes, against 63-69 for hk.ops.attn's kernel
  (1.18-1.21x) and 55-57 for tuned non-causal Triton FA-2 (1.4x).
- Accuracy at Qwen-Image 2.1's real shape (q 4,096, kv 4,141): mean error
  2.95e-5, against a bf16 floor of 2.89e-5 and 4.50e-5 for both hk.ops.attn
  and aotriton.
- End to end (production image, 1024^2 / 30 steps, interleaved): 1 GPU 29.0 ->
  28.4 s (-2.1%); 4 GPUs 9.77 -> 9.55 s (-2.3%).
- Two parts of the done-criteria need restating (decided below).
  - The peak: 70% of the 122.6 TF/s spec is 85% of the ~100 TF/s the card
    attains even in a register-only WMMA loop; K3 is at ~82% of attainable.
    What is left (prologue/epilogue, LDS, exp2) is 2-5% items each.
  - Image identity: hk's accepted path is itself 40.9 dB from stock (1 GPU),
    against stock's 61 dB run-to-run, because 30 steps amplify any numeric
    change. K3 is 36.6 dB (1 GPU) and 37.7 dB (4 GPUs, where hk is 37.1). The
    measurable criterion is per-call accuracy, which K3 meets.

### Decisions (2026-10-24, late)

- **K1 correctness.** Measured against an exact fp32 reference (bf16 activations,
  dequantized FP4 weights, no activation quantization) on sampled tokens, hk's max-rel
  error may exceed AITER's own error on the same tokens by at most 1e-2. Equivalently,
  hk's max-rel against aiter.fused_moe may be up to 3e-2. The original <= 1e-2 against
  AITER cannot be met by anything that does not copy AITER's internal numerics: at most
  M AITER switches to a stage-1 kernel that writes FP8 h itself, and hk and AITER then
  differ by 2.0-2.7e-2, while both sit 4-5% from the exact reference.
- **K3 criterion.** Per-call accuracy, not image identity. K3 meets it (mean error 2.95e-5
  against a bf16 floor of 2.89e-5 at Qwen's real shape).
- **T-CR (ConvRot).** Image drift is not a goal. The project's object is the kernel; no
  further work goes into ConvRot's image quality.
- **Direction on CDNA.** K1 and K2 must turn into a visible end-to-end gain for
  DeepSeek-V4.1-Flash in vLLM (prefill TTFT), not stop at kernel ratios. In order:
  (a) K1 to >= 1.3x;
  (b) K1 and K2 wired into vLLM through `hk.integration` with a TTFT A/B;
  (c) if two free MI355X are not available, a single-GPU form (fewer layers, or TP1 with
  dummy weights) for at least a layer- or model-level end-to-end number.
  W7900D work (K3, T-CR) stays where it is.

### Testbed kernels (2026-10-24)

Two kernels chosen to test the DSL on something unfamiliar, from
the MiniMax-H3 / Qwen-Image DiT on a W7900D, recorded by the same criteria:

| id | kernel | reference / baseline | what the hardware offers |
|---|---|---|---|
| **T-CR** | ConvRot INT8 W8A8 GEMM: the online rotation, per-group scales, INT8 GEMM on `v_wmma_i32_16x16x16_iu8`, as a third `SGLANG_DIFFUSION_CONVROT_INT8_BACKEND` (both existing ones are CUDA-only); H3 quantized online from its bf16 checkpoint (the official repo ships no int8; community ComfyUI ConvRot files exist) | SGLang's two CUDA backends (semantics); on gfx1100, hipBLASLt `torch._int_mm` and the bf16 path | gfx1100's int8 WMMA runs at the fp16 rate (measured 96 vs 101 TOP/s register-only; iu4 193): no compute gain. hipBLASLt int8 62.7 / 75.9 TOP/s against bf16 FlyDSL / hk ~88 TF/s at H3 / Qwen shapes. The gain on offer is bytes: int8 halves the DiT, which H3 streams from host RAM every step -- measured before anything is claimed |
| **T-SA** | SageAttention INT8: Q, K int8 with smooth-K, PV in fp16 (gfx1100 has no fp8) | SageAttention 1.0.6's Triton kernel: its default `num_stages=4` crashes Triton's AMD stream pipeliner on gfx1100; with 1 it runs at 49 TF/s-equivalent (Triton 3.6; 40 in 3.5) at Qwen's shape, max-rel 1.9e-2 vs fp32 -- against hk's fp16 attention at 62-70 TF/s | the same: int8 QK^T is not faster than fp16 on RDNA3, so the kernel can at best save bytes and registers; on par with K3 is the realistic ceiling, and its value here is as a DSL testbed |

Status (2026-10-24, evening; both have since landed, see the top of this page):
- **T-CR**: the first round was a `dsl` failure -- hk's RDNA3 layer was built on
  the false premise that gfx1100 has no int8 WMMA. #31 (in main)
  adds i8 register tiles and the iu8 WMMA (and fixes a latent `st<int8>` swizzle
  bug in HipKittens); the tuned bf16 GEMM schedule on int8 operands is bit-exact
  and 87.7-93.2 TOP/s (hipBLASLt int8 68.7-83.7). The whole ConvRot linear
  (rotation, per-row quant, int8 GEMM, dequant epilogue) is 1.18x hk's bf16 GEMM
  at H3's 14720x21504x5376 and at parity at Qwen's shapes. The `hk` backend is in
  `hk.integration.sglang_diffusion` (#26); H3 end to end, measured after this note: denoise
  1.17-1.22x bf16's.
- **T-SA, the controlled experiment**:
  both arms correct at round 1 and at the same speed in the end -- hk 1.50x the
  reference, Triton 1.48x (both ~1.15x hk's bf16 attention) -- but the hk arm took
  77 minutes against Triton's 29 minutes, and got there only by
  injecting two C++ helpers into hk's generated source, because hk refused int8
  tiles. Verdict for this kernel: the workflow and harness held; the DSL's gap
  was decisive. It is closed in main (#31). The hk arm was rerun twice: on #31's branch
  it reached 1.52x but still wrote two C++ helpers; from main (`3fd779f`) it reached 1.505x with
  no C++ outside the DSL, correct at round 1, in 68 minutes.

## How the kernels are built: one task each (KDA-style)

Each kernel above is a task in a fixed, recorded workflow -- NVlabs' Kernel
Design Agents (github.com/NVlabs/kda), whose method is hardware-agnostic,
ported to AMD and hk -- so that a failure can be attributed to the harness or
to the DSL. A task has a definition (op signature, dtypes and layouts, the
shapes with their weights in the model, a reference, a tolerance or "bitwise",
the baselines to beat), a contract (objective, correctness, target, validation
and evaluation commands, promotion criteria), a draft and a plan written before
any code, and a record of every candidate and every measured number.

**One round** = one candidate: edit -> build (spill gate, ISA dump) ->
validate (bitwise or tolerance against the reference, **and N repeated runs
compared with each other** -- K2's race passed a single check) -> evaluate
(in one process, interleaved against every baseline, weights rotated past the
LLC) -> profile (hk.profile verdict) -> record. Every failed round is
classified: `dsl` (hk cannot express it), `codegen` (hk's passes / C++ /
hipcc produce something wrong or slow), `harness` (the workflow's tools
mislead or miss), `algorithm` (the idea itself), `env` (machine, driver,
cluster).

**Reused parts.** `hk.check` (correctness against a reference),
`hk.tune.measure` (interleaved, LLC-rotated timing), `hk.profile` (counters,
thread trace, ISA mix, verdict, JSON), `hk.schedules` (knobs as data), T4's
harness (sealed shapes, an audit that the submitted kernel actually ran, a
GPU-time ledger).

**The controlled experiment** (answers "harness or DSL?"). T-SA (or T-CR if
SageAttention's Triton kernel does not run on gfx1100): the same task
contract, the same workflow and skills, the same agent (a fresh
session, as T4) and budget (8 h wall, 3 GPU-h on one
W7900D), two arms -- **hk-dsl** and **Triton** (the control language with a
gfx1100 backend). The reference is a black-box oracle in both (its source is
not given; the transcript is audited as T4's is). Compared: rounds to the
first correct candidate, best interleaved speed against the reference at the
contract's shapes, and the distribution of failure classes. Both arms stuck
in the same class -> the workflow or harness; only the hk arm stuck -> the
DSL. A second run with the arms' order swapped on fresh sessions guards
against one lucky run.

K2's race is the first task run through the flow (recorded after the fact:
what was tried, and that the bug was a `codegen` failure in hk-lds-pipeline
found only by a repeat-run check the harness did not have -- a `harness`
gap too).

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
  [chapter 46](../history/46-qwen-image-on-rdna3.md)) -- on upstream SGLang
  main. **In the production image** (sglang-radeon@sha256:e9c8365e, SGLang
  main of 2026-09-22, `sglang serve` with the production flags, timed over
  HTTP): stock 50.25 s -> hk 29.0 s, 1.73x on one card; the weights'
  revisions 790c926 (production) and d26bb61 differ only in README.md. Four
  cards (`--num-gpus 4`, the production deployment): stock 17.75-17.94 s ->
  hk 9.67-9.89 s, **1.81x**.
- MiniMax-H3 4 s clip: denoise 1.43x with hk attention, the request 1.15x
  (median of three interleaved runs; the VAE decode stage is noisy).
- DeepSeek-V4.1-Flash: the profiling path in vLLM's AMD package (dummy
  weights, TP2, traces with shapes, DRAM counters), the tables above; an
  MXFP8 GEMM in hk bit-identical to vLLM's (then a branch; #33 in main).

## Not in scope

- Lossy speedups as hk's result (cache-dit, fewer steps, lower-bit weights
  than the checkpoint ships).
- Communication kernels (the all-reduce is the largest non-kernel item in
  prefill; that belongs to vLLM / RCCL / quick-reduce).
- Training kernels.
