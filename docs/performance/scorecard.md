# The scorecard: the only performance numbers we quote

**On the one scenario re-measured from a fair baseline so far -- MiniCPM-V-4.7-35B-A3B served by vLLM on two
Radeon PRO W7900D -- hk gives long prompts their first token 1.2-1.5x sooner and generates 1.1-1.3x faster
than stock vLLM in its best configuration, with the same answers.**

A claim from this project quotes this page. Each scenario compares hk with the **best stock configuration**
found for it (backends, switches and tuning the engine already has, no code changed), in the same session, on
the same GPUs; how, and every range and discard rule: [the method](method.md#the-scorecard-scenarios).
Kernel-level numbers guide the work but are not claims.

## MiniCPM-V-4.7-35B-A3B, vLLM, 2x Radeon PRO W7900D

| metric | best stock | hk | hk vs stock |
|---|---|---|---|
| first token, one 32k-token prompt (s) | 4.40 | 3.67 | 1.20x |
| first token, one 128k-token prompt (s) | 35.7 | 23.4 | 1.53x |
| time per output token, 1 request (ms) | 12.2 | 10.9 | 1.12x |
| time per output token, 8 requests (ms) | 32.3 | 24.7 | 1.31x |
| GSM8K-200 (greedy) | 91.0% | 91.5% | equal (p 1.0) |

Tensor parallel 2, CUDA graphs on in both. Best stock: vLLM's RDNA attention backend (`ROCM_SEGMENTED_ATTN`) and
GEMMs tuned offline with TunableOp. hk: hk-dsl 0.9.20 with the same TunableOp file. Where the time goes: hk's
attention for long prompts and its MoE for decoding. Reproduce: [`scripts/scorecard/minicpmv-35b.sh`](../../scripts/scorecard/minicpmv-35b.sh).

## Not yet re-verified

These were measured before the baseline rules above and are being measured again; until then this page quotes
no number for them.

- DeepSeek-V4.1-Flash on vLLM, 2x Instinct MI355X
- MiniCPM5-2B and MiniCPM-V-4.7-1B on vLLM, Radeon PRO W7900D
- Qwen3.8-27B (GPTQ and AWQ int4) on vLLM, Radeon AI PRO R9700
- Wan2.1-T2V-1.3B and Qwen-Image 2.1 on SGLang-Diffusion, W7900D and R9700

## The older numbers

The [performance page](../archive/performance-2026-10.md) and the
[development record](../history/README.md) are kept as a dated record of
kernel-level and engine-level measurements. Some of their baselines were
weaker than the opponents' best configuration, so do not read them as current
claims.
