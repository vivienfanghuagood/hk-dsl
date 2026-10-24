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
scales, bit-identical to vLLM). Winnable, measured: gfx950's scaled MFMA runs
at 4.5-4.7 PF/s register-only and hipBLASLt's plain FP8 GEMM at 3.14 PF/s
(8192^3), while every MX GEMM is at 1.25-2.2 PF/s -- the MX penalty is
software. K2 status (branch `claude/cdna-fp8`): a four-wave form (`mx4w`,
accumulators in AGPRs, packed scales, a three-stage ring) at 2.2-2.3 PF/s on
the largest projection, 1.46-1.54x vLLM, ~1.1x FlyDSL; small grids and short
K still lose.

### Testbed kernels (owner, 2026-10-24)

Two kernels the owner chose to test the DSL on something unfamiliar, from
the MiniMax-H3 / Qwen-Image DiT on a W7900D, recorded by the same criteria:

| id | kernel | reference / baseline | what the hardware offers |
|---|---|---|---|
| **T-CR** | ConvRot INT8 W8A8 GEMM: the online rotation, per-group scales, INT8 GEMM on `v_wmma_i32_16x16x16_iu8`, as a third `SGLANG_DIFFUSION_CONVROT_INT8_BACKEND` (both existing ones are CUDA-only); H3's own convrot int8 checkpoint | SGLang's two CUDA backends (semantics); on gfx1100, hipBLASLt `torch._int_mm` and the bf16 path | gfx1100's int8 WMMA runs at the fp16 rate (measured 96 vs 101 TOP/s register-only; iu4 193): no compute gain. hipBLASLt int8 62.7 / 75.9 TOP/s against bf16 FlyDSL / hk ~88 TF/s at H3 / Qwen shapes. The gain on offer is bytes: int8 halves the DiT, which H3 streams from host RAM every step -- measured before anything is claimed |
| **T-SA** | SageAttention INT8: Q, K int8 with smooth-K, PV in fp16 (gfx1100 has no fp8) | SageAttention's Triton kernel, if it runs on gfx1100 (checked first); else the fp16 kernels of K3 | the same: int8 QK^T is not faster than fp16 on RDNA3, so the kernel can at best save bytes and registers; on par with K3 is the realistic ceiling, and its value here is as a DSL testbed |

## How the kernels are built: one task workspace each (KDA-style)

Owner, 2026-10-24: a lead agent iterating by hand cannot tell, when it fails,
whether the harness or the DSL is at fault. Each kernel above is therefore a
task in a fixed, recorded workflow -- NVlabs' Kernel Design Agents
(github.com/NVlabs/kda), whose method is hardware-agnostic, ported to AMD and
hk.

**Layout.** The reusable flow in `~/hk-dx/kda/_flow/` (prompt, record tools,
skills' sources); one workspace a task in `~/hk-dx/kda/<task>/`:

    task.json         definition + workloads + baselines (FlashInfer-Trace-like): op signature,
                      dtypes/layouts, the shapes with their weights in the model, reference
                      callable, tolerance or "bitwise", baselines to beat
    docs/contract.md  objective, correctness, target, allowed approaches, validation command,
                      evaluation command, promotion criteria
    docs/draft.md     baseline + how it is validated, risks, candidate directions ranked by
                      expected value and risk, first steps, the exact commands, the evidence
                      that promotes / revises / rejects a candidate -- before any code
    docs/plan.md      the draft made executable
    candidates.jsonl  one line a candidate: name, parent, change, why, status
                      (proposed/built/correct/promoted/rejected), failure class, evidence paths
    benchmark.csv     every measured number: candidate, shape, arm, us, TF/s, % peak, run id
    profile/          hk.profile / rocprofv3 reports per candidate
    runs/ outputs/    raw logs
    .claude/skills/   the task's skills (never installed globally)

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
GPU ledger), the GPU locks and helpers. New: `kda_record.py` (schema-checked
appends to candidates.jsonl / benchmark.csv), `kda_validate.py` (repeat-run
determinism on top of hk.check).

**Skills (AMD versions of KDA's three).** `hk-kernelwiki`: gfx950 / gfx1100
hard limits (VGPR granule 24 on gfx11, 512 regs with AGPRs at one wave a
SIMD on gfx950, LDS 64 / 160 KB, WMMA / MFMA rates measured), HipKittens
idioms, and the pitfalls already paid for (a VGPR spill corrupts silently;
`s_waitcnt` does not hold a WMMA back; narrowing stores go bytewise; gfx11's
`__syncthreads` flushes L0; register and LDS loads retire out of order on
vmcnt). `hk-profile-report`: reading hk.profile / rocprofv3 output into a
next step (replaces ncu-report-skill). `kda-loop`: draft -> plan -> one
candidate a round -> record (replaces humanize).

**The controlled experiment** (answers "harness or DSL?"). T-SA (or T-CR if
SageAttention's Triton kernel does not run on gfx1100): the same task
contract, the same workflow and skills, the same agent (a fresh
claude-opus-5-5 headless session, as T4) and budget (8 h wall, 3 GPU-h on one
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
  revisions 790c926 (production) and d26bb61 differ only in README.md. The
  four-card number is being measured.
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
