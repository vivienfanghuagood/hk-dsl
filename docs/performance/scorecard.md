# The scorecard: the only performance numbers we quote

**On the two scenarios re-measured against stock vLLM in its best configuration, both on Radeon PRO W7900D,
hk-dsl gives long prompts their first token 1.2-1.6x sooner and generates 1.1-1.3x faster, with the same
answers and KV-cache capacity within 2% of stock's.**

A claim from this project quotes this page. Each scenario compares hk with the **best stock configuration**
found for it: the backends, switches and tuning the engine already has, with no code changed. Both run in the
same session on the same GPUs, end to end (serving, scheduling included). Every number is the median that vLLM's
own benchmark, `vllm bench serve`, prints. Ranges, earlier runs and what was discarded:
[the method](method.md#the-scorecard-scenarios). Kernel-level numbers guide the work; they are not claims.

## MiniCPM-V-4.7-35B-A3B, vLLM, 2x Radeon PRO W7900D

**Result:** hk makes the first token of a long prompt 1.2-1.6x sooner and decoding 1.1-1.3x faster.

**Setup:** two Radeon PRO W7900D (gfx1100), tensor parallel 2. vLLM 0.1.dev1+g10cc2f6ae (a ROCm 7 build),
torch 2.13, hk-dsl 0.9.23. `vllm bench serve` with random prompts: three 32k- or 128k-token prompts one at a time
for first token; 2k-token prompts with 256 output tokens at 1 and 8 concurrent requests for time per output token.
One server start per arm; 0 failed requests.

**Best stock:** `--attention-backend ROCM_SEGMENTED_ATTN`, an offline-tuned TunableOp file and CUDA graphs. Both
changes are lossless against plain stock (GSM8K-200 and an image VQA set, paired). **hk:** hk-dsl's default routes
on `TRITON_ATTN`, with the same TunableOp file.

| metric | best stock | hk | change |
|---|---:|---:|---:|
| first token, 128k-token prompt | 32.2 s | 20.3 s | **1.59x** |
| first token, 32k-token prompt | 4.43 s | 3.70 s | **1.20x** |
| time per output token, 8 requests | 30.7 ms | 24.3 ms | **1.27x** |
| time per output token, 1 request | 12.3 ms | 11.1 ms | **1.11x** |

Quality: equal (GSM8K-200 91.0% for stock, 89.5% and 91.5% on hk's two starts, hk-dsl 0.9.20; McNemar p = 1.0).
What hk costs: a cold start about 60 s longer (6.4 min against 5.5 min in this run), and a one-time hk compile
(~38 s) on the first long prompt after an empty hk cache.

**Reproduce:** [the commands](../../scripts/scorecard/README.md#minicpm-v-47-35b-a3b-2x-radeon-pro-w7900-48-gb), or
[`scripts/scorecard/minicpmv-35b.sh`](../../scripts/scorecard/minicpmv-35b.sh), a loop over them -- 2x W7900 48 GB,
about 30 min, plus 45 min of TunableOp tuning once.

## Qwen3.8-27B GPTQ-Int4, vLLM, 1x Radeon PRO W7900D

**Result:** hk makes the first token of a long prompt 1.16x sooner and decoding 1.12-1.13x faster, with a KV
cache 1.7% smaller than stock's.

**Setup:** one Radeon PRO W7900D (gfx1100). `palmfuture/Qwen3.8-27B-GPTQ-Int4`, vLLM 0.33.0, torch 2.13,
ROCm 7.2.3, hk-dsl 0.9.23. `vllm bench serve` with random prompts: three 8k- or 32k-token prompts one at a time for
first token; 2k-token prompts with 256 output tokens at 1 and 8 concurrent requests for time per output token. One
server start per arm; 0 failed requests.

**Best stock:** `--attention-backend ROCM_SEGMENTED_ATTN`, an offline-tuned TunableOp file and CUDA graphs.
**hk:** hk-dsl's default routes on `TRITON_ATTN`, with the same TunableOp file.

| metric | best stock | hk | change |
|---|---:|---:|---:|
| first token, 32k-token prompt | 35.4 s | 30.6 s | **1.16x** |
| first token, 8k-token prompt | 8.24 s | 7.12 s | **1.16x** |
| time per output token, 1 request | 32.7 ms | 28.9 ms | **1.13x** |
| time per output token, 8 requests | 83.4 ms | 74.5 ms | **1.12x** |
| KV cache capacity | 288,085 tokens | 283,306 tokens | 1.7% smaller |

Quality: equal (GSM8K-200 91.0-91.5% for both arms, two runs each; McNemar p = 1.00). The decode gain is hk's int4
linear route (`vllm_gptq`); with it off, hk's decode is level with stock.
What hk costs: a KV cache 1.7% smaller, and a first start after install about 3 min longer while hk compiles
its int4 kernels into its cache (4.1 min against stock's 1.0 min in this run).

**Reproduce:** [the commands](../../scripts/scorecard/README.md#qwen38-27b-gptq-int4-1x-radeon-pro-w7900-48-gb), or
[`scripts/scorecard/qwen3.8-27b-gptq.sh`](../../scripts/scorecard/qwen3.8-27b-gptq.sh), a loop over them -- one
W7900 48 GB, about 20 min, plus 15 min of TunableOp tuning once.

## Not yet re-verified

These were measured before the baseline rules above. Until they are measured again, this page quotes no number
for them.

- DeepSeek-V4.1-Flash on vLLM, 2x Instinct MI355X
- MiniCPM5-2B and MiniCPM-V-4.7-1B on vLLM, Radeon PRO W7900D
- Qwen3.8-27B GPTQ and AWQ (AMD Quark W4A16) on vLLM, Radeon AI PRO R9700
- Wan2.1-T2V-1.3B and Qwen-Image 2.1 on SGLang-Diffusion, W7900D and R9700

## The older numbers

The [performance page](../archive/performance-2026-10.md) and the
[development record](../history/README.md) are kept as a dated record of
kernel-level and engine-level measurements. Some of their baselines were
weaker than the opponents' best configuration, so do not read them as current
claims.
