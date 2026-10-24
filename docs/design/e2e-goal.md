# Goal: kernels that win end to end in the models people run now

Status: proposed 2026-10-24, the milestone after the tuning goal
([tuning-goal.md](tuning-goal.md): T1-T3 merged, T4's first sealed run under
way). That milestone made hk's own kernels tunable; this one points them at
what is being served this month, and makes the path from "a kernel is slow
in my model" to "my engine runs hk's kernel and is faster" short enough that
a developer's own harness -- a person or a coding agent -- walks it alone.

## The goal

On each GPU, for the models named below, in the engine people serve them
with:

1. **End to end faster than the stock engine** -- the same engine, version
   and configuration with and without hk's kernels, the output the same
   (greedy tokens identical; images and video within a stated tolerance) --
   by a margin outside the measured noise.
2. **Every win reproducible from a harness**: each kernel reached through the
   documented loop (`headroom` -> template / `new-op` -> `hk.check` /
   `hk.profile` / `hk.tune` -> `ab` -> plugin), with the commands in the
   record, so the next model's kernels can be written the same way without us.

| GPU | models | engine | what "end to end" is |
|---|---|---|---|
| MI355X (gfx950) | **DeepSeek-V4.1-Flash** (40 layers; MQA head_dim 512 with compressed and sparse-indexed KV, a sliding window of 128; 384 routed experts top-6, FP4 experts, FP8 32x32-block weights; mHC, Engram, MTP) | vLLM's `vllm/models/deepseek_v41/amd` (the nightly ROCm 10 image has it), aiter on | output tokens/s and TPOT at serving concurrencies, prefill TTFT |
| W7900D (gfx1100) | **Qwen-Image 2.1** (MMDiT, 20B; 1024^2, 30 steps: 50.6 s on one card today) and **MiniMax-H3** (33B omni-modal DiT, text to video and audio; ~10 min a 4 s clip, the DiT streamed from host RAM) | SGLang-Diffusion (`sglang.multimodal_gen`) with sglang-radeon's RDNA3 shims | wall time a request, at the resolutions and step counts served |

## Where it stands (2026-10-24)

- **gfx950 dense GEMM.** hk.tune reaches 13/25 of the held-out shapes at
  0.95x the best baseline (21/25 with a route to hipBLASLt). Chapter 45: what
  is left is the pingpong's main loop -- 1.56 instructions a matrix
  instruction against hipBLASLt's 0.58 -- and stream-K's share epilogue.
  DeepSeek-V4.1-Flash's GEMMs are mostly not bf16 (FP8 block-scaled
  projections, FP4 experts), so this matters less for G1 than its own kernels.
- **gfx950 kernels that exist**: bf16 GEMM (pingpong, 4-warp ring, decode
  GEMV), attention (head_dim 64/128, paged decode/prefill), 4-bit decode GEMM,
  the row ops, a MoE GEMM that does not build there (T4's harness). Nothing
  yet for FP8 block scales, FP4 experts, head_dim 512, a sparse indexer or
  mHC's Sinkhorn mixing.
- **gfx1100 diffusion.** SGLang-Diffusion on RDNA3 runs attention through
  `torch_sdpa` -- the only backend that works there (aiter is CDNA-only, FA3
  Hopper-only) -- and the H3 family's windowed attention
  (`hybrid_window_attn_h3`, which gathers rows into varlen segments) through
  per-segment SDPA calls with a host sync each (sglang-radeon's
  `varlen_attention.py`). hk's gfx1100 attention (head_dim 64/128, dense
  B,H,N,D, the kernel behind hk's vLLM attention backend) is not wired in. VAE
  convolutions are MIOpen's (`MIOPEN_FIND_MODE=FAST` already removed their
  cold start). Qwen-Image's knobs that help are known (regional compile 6.5%,
  cache-dit 2.9x lossy); none of them is a kernel.
- **Integration.** `hk.integration` has vLLM plugins (attention backend,
  linear, int4, MoE), an SGLang LLM plugin, `headroom` (a step's time by
  kernel family against what hk has), `ab` (verdicts against the measured
  noise) and `new-op`. Nothing for SGLang-Diffusion, nothing for vLLM's
  hardware-isolated `vllm/models/*/amd` model packages.

## Acceptance criteria

| id | criterion | evidence |
|---|---|---|
| E1 | **Qwen-Image 2.1 on a W7900D**: hk's kernels as a SGLang-Diffusion attention backend (and GEMM / norm / modulation ops where `headroom` ranks them) make a 1024^2 30-step request faster than stock (torch_sdpa, hipBLASLt) outside the noise, images within the tolerance the stock path's own run-to-run variation sets. | `ab`-style interleaved runs, images, docs/history |
| E2 | **MiniMax-H3 on a W7900D**: its varlen windowed attention on an hk kernel (segments read on the GPU, no host sync), and the DiT's offload made cheaper where the kernels allow; a 4 s clip faster than stock. | same |
| E3 | **DeepSeek-V4.1-Flash on an MI355X**: `headroom` on the vLLM AMD model (dummy weights where the real ones do not fit the shared box; the layer count cut with `hf_overrides` for per-layer attribution), then hk kernels for the families it ranks -- expected: FP8 32x32-block GEMM, FP4 grouped expert GEMM, head_dim-512 MQA decode over the compressed/sparse KV -- each >= the kernel vLLM runs there, and the engine faster end to end at a stated concurrency, tokens identical. | per-kernel scoreboards, `ab`, docs/history |
| E4 | **The loop, written down**: for each of E1-E3 a walkthrough a developer's harness can replay -- the commands from `headroom` to the plugin, and an `hk.integration` adapter for SGLang-Diffusion and for vLLM's `models/*/amd` packages that loads an hk kernel without patching the engine's tree. | docs, examples, tests |
| E5 | **gfx950 GEMM's main loop**: fewer non-matrix instructions a matrix instruction in the pingpong (the SALU address work, LDS reads per MFMA, hazard nops), measured by `hk.profile` and on the acceptance split. | chapter, scoreboard |

## Order

E1 first: one card, a 33 GB model, an attention path where the stock
backend is the slowest one the card has, and a metric (seconds an image) that
needs no serving setup. E3's per-kernel work starts in parallel from the
model's config and vLLM's AMD package, since its end-to-end runs need
several of the shared MI355X's GPUs. E2 builds on E1's backend. E5 runs
alongside, on GPU time the others leave.

## Not in scope

- Lossy speedups as hk's result (cache-dit, fewer steps, lower-bit weights
  than the checkpoint ships): they are product decisions; hk's numbers are at
  the checkpoint's precision and the request's step count.
- Training kernels.
