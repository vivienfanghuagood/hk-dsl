# The scorecard: the only performance numbers we quote

## The rule

A performance claim from this project quotes an end-to-end serving result
from the scorecard, taken as [the method](method.md) says. Kernel-level numbers (one kernel against another kernel)
guide the work, but they are not claims. They depend too much on how well the
opponent was configured.

## Instinct MI355X: DeepSeek-V4.1-Flash on vLLM

vLLM serving DeepSeek-V4.1-Flash with real weights, tensor parallel over two MI355X.

**How it is measured.**

- **One container, fresh servers.** Every arm gets one fresh `vllm serve` per round, with the same engine flags and
  the same recipe environment (`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`,
  `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`). Arm order is reversed every other round. 2 rounds.
- **Fair stock.** The baseline is stock vLLM with AITER's tuned paths switched on through configuration only (no vLLM
  or AITER source change): its tuned MXFP8 GEMM, a MoE table tuned on the model's real routing, and its fused sparse
  MLA. Plain stock (the recipe alone) runs in the same session as a reference.
- **What a cell is.** The mean over rounds, with the range over rounds. A delta smaller than the spread (the larger
  arm's range, in % of the baseline) is shown as `=`. GSM8K: McNemar exact test over every pair of runs; p >= 0.05 is
  `=`.
- **Discarded measurements.** A measurement is dropped if any request failed, if a process outside the container held
  one of the two GPUs during it, or if it stalled (slowest request > 2.4x its lower-quartile request). One was
  dropped: the tuned-stock "mixed" run of round 1 (a stall), so that cell has one round.

**Software.** vLLM 0.32.1rc1.dev36 (`vllm/vllm-openai-rocm` nightly e68efc685, ROCm 10.0, torch 2.12, AITER 0.1.25),
hk-dsl 0.9.16. Engine: TP2, max_model_len 16640, no prefix caching.

**The arms.**

- **tuned stock**: the baseline above.
- **hk + tuned**: hk's default routes on top of the tuned baseline. AITER's fused sparse MLA keeps prefill and decode;
  AITER's tuned MoE table serves the rows hk's MoE route does not take.
- **hk, sparse-MLA route**: hk's default routes including its sparse-MLA prefill. That route runs where vLLM's own
  sparse MLA runs, so this arm leaves `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` unset; decode then uses vLLM's sparse-MLA
  decode kernel, not AITER's.

| metric | tuned stock | hk + tuned | hk, sparse-MLA route | hk + tuned vs tuned stock | sparse-MLA route vs tuned stock |
|---|---|---|---|---|---|
| prefill 8k tokens, tok/s | 69,371 (69,286-69,455) | 72,626 (72,605-72,647) | 74,293 (74,180-74,406) | **+4.7%** | **+7.1%** |
| prefill 8k, TTFT mean (ms) | 883.3 (882.0-884.6) | 843.7 (843.4-844.0) | 824.6 (823.2-826.0) | **-4.5%** | **-6.6%** |
| long (8k in / 128 out, 8 concurrent), req/s | 3.628 (3.622-3.634) | 3.729 (3.724-3.734) | 3.736 (3.721-3.752) | **+2.8%** | **+3.0%** |
| long, TTFT P99 (ms) | 963.4 (954.0-972.8) | 910.4 (897.4-923.4) | 892.6 (885.5-899.7) | **-5.5%** | **-7.4%** |
| mixed (2k in / 256 out, 32 concurrent), req/s | 6.212 (1 round) | 6.274 (6.261-6.287) | 6.138 (6.138-6.138) | +1.0% | -1.2% (worse) |
| mixed, TPOT mean (ms) | 17.75 (1 round) | 17.59 (17.55-17.63) | 18.02 (18.02-18.03) | -0.9% | +1.5% (worse) |
| decode TPOT, 1 concurrent (ms) | 6.566 (6.560-6.572) | 6.552 (6.543-6.562) | 6.533 (6.519-6.548) | = | -0.5% |
| decode TPOT, 4 concurrent (ms) | 7.800 (7.784-7.815) | 7.776 (7.756-7.795) | 7.863 (7.861-7.865) | = | +0.8% (worse) |
| decode TPOT, 8 concurrent (ms) | 9.257 (9.249-9.266) | 9.284 (9.245-9.322) | 9.284 (9.277-9.292) | = | +0.3% (worse) |
| GSM8K (1319, greedy) | 94.69% | 94.62% | 94.20% | = (McNemar p 0.63-1.00) | = (p 0.23-0.47) |

hk + tuned is the configuration we quote: **about +5% on 8k-token prefill throughput and 4-6% lower prefill TTFT
against the AITER-tuned stock, with decode unchanged.** The sparse-MLA route adds about two more points of prefill,
at a cost of up to 1.5% at decode-heavy load (vLLM's sparse-MLA decode kernel against AITER's), an open item.

Against plain stock (the recipe without AITER's tuned paths), hk's routes show larger prefill gains (+15% for
hk + tuned, +18% for the sparse-MLA route). The tuned stock itself is +10% over plain stock, so we do not quote those.

## Radeon (RDNA3 / RDNA4)

End-to-end serving on a Radeon Pro W7900D (gfx1100, RDNA3) and a Radeon AI PRO R9700 (gfx1201, RDNA4), with the
models these GPUs actually serve.

**How it is measured.**

- **Same process, alternating rounds.** Stock and hk run in one process, in alternating rounds (ABBA), at least 3
  rounds. On these GPUs two separate server starts can differ by more than most wins, so arms in separate processes
  are never compared.
  - vLLM: the engines are built in turn inside one process. Where two engines do not fit on the GPU at once, each
    arm's engine is rebuilt every round.
  - SGLang: one server; hk's routes are switched on and off between rounds, and call counters check that each arm ran
    the code it claims.
- **Fair stock.** "Stock" is the engine's best configuration without hk:
  - hipBLASLt with TunableOp tuned on the model's real shapes, including the one-token decode shapes the offline tuner
    skips;
  - the engine's default attention backend.

  For Wan2.1's VAE, stock MIOpen falls back to its naive reference conv3d kernel on both GPUs in every find mode and
  layout tried, so stock there runs PyTorch's own convolution instead (cuDNN/MIOpen off), which is faster.
- **Released wheels only.** hk is installed from PyPI in a clean environment, and the wheel's sha256 is checked
  against the GitHub release asset. The version is given per row.
- **What a cell is.** The median of the per-round medians, with the range over rounds. A ratio whose round ranges
  overlap is shown as `=` (no difference), never as a win. Ratios are stock / hk: above 1 means hk is faster.
- **Foreign processes.** Another user's process inside a measured block discards that round in both arms. A row
  stands only with at least 3 clean rounds left. A row that does not meet these rules is not quotable, and is not shown.

### W7900D (gfx1100)

vLLM rows: vLLM on ROCm 7.14 (TheRock), torch 2.13. SGLang-Diffusion rows: SGLang's Radeon build on ROCm 7.2, torch 2.9.1.

| model | metric | stock | hk | stock / hk |
|---|---|---|---|---|
| MiniCPM5-2B (hk 0.9.14) | TPOT, 1 / 4 / 8 concurrent (ms) | 7.90 / 9.29 / 10.66 | 7.92 / 8.76 / 9.16 | 1.00x (slightly worse) / 1.06x / 1.16x |
| | TTFT 8k / 32k (ms) | 907 / 12,550 | 591 / 4,247 | 1.54x / 2.95x |
| | GSM8K (1319, greedy) | 73.77% | 74.60% | not significant (McNemar p 0.42) |
| MiniCPM-V-4.7 1B, eager (hk 0.9.14) | TPOT, 1 / 4 / 8 concurrent (ms) | 21.5 / 22.3 / 27.2 | 21.3 / 22.2 / 27.3 | 1.01x / = / = |
| | TTFT 8k / 32k (ms) | 656 / 7,451 | 270 / 1,384 | 2.43x / 5.38x |
| | GSM8K | 81.50% | 80.89% | not significant (p 0.42) |
| MiniCPM-V-4.7 1B, eager: the `vllm_norm` route (hk 0.9.19), route off / on | TPOT, 1 / 4 / 8 concurrent (ms) | 22.39 / 22.56 / 21.99 | 16.81 / 18.22 / 17.58 | 1.33x / 1.24x / 1.25x |
| | TTFT 8k (ms) | 358.1 | 334.5 | 1.07x |
| | GSM8K (300, greedy) | 81.67% | 82.33% | not significant (p 0.80) |
| MiniCPM-V-4.7 35B-A3B, 2 GPUs (hk 0.9.14) | TPOT, 1 / 4 / 8 concurrent (ms) | 61.8 / 63.1 / 73.3 | 61.2 / 62.0 / 64.5 | = / = / 1.14x |
| | TTFT 8k / 32k (ms) | 2,277 / 18,070 | 1,575 / 6,957 | 1.45x / 2.60x |
| | GSM8K | 95.22% | 95.83% | not significant (p 0.08); a second accuracy run is planned |
| Wan2.1-T2V-1.3B, 832x480x81, 20 steps (hk 0.9.14) | seconds per video | 496.7 (496.5-497.5) | 187.1 (187.1-187.6) | 2.66x |
| Qwen-Image 2.1, 1024x1024, 30 steps (hk 0.9.14) | seconds per image | 42.0 | 26.2 | 1.60x |

**The `vllm_norm` rows** measure one route, not stock against all of hk. Both columns run hk 0.9.19 with its
default-on routes; the left one turns `vllm_norm` off (`HK_VLLM_NORM=0`). MiniCPM-V-4.7's production engine
arguments run it eager (`enforce_eager`), and in eager mode its Gemma RMSNorm runs as ~8 PyTorch kernels; the route
runs it as one hk kernel. Under torch.compile the route stands aside, so a compiled deployment sees no change.
- **Setup:** one process, the engine rebuilt per block, 6 alternating rounds, every range disjoint. The scorecard row's
  production engine arguments (`enforce_eager`, TRITON_ATTN, 4,096 batched tokens), no TunableOp in either column.
- **Ranges over the 6 rounds:** route off 20.57-23.67 / 21.29-22.74 / 21.12-23.43 ms, route on 16.66-17.05 /
  17.83-18.83 / 17.04-19.79 ms; TTFT 8k 357.2-365.3 against 333.4-335.9 ms.

Quality: Wan2.1, hk's video is 34.6-34.8 dB from stock's, the same as stock against itself across runs (34.6-34.8 dB). Qwen-Image 2.1,
hk's image is 42.8 dB from stock's.

### R9700 (gfx1201)

vLLM rows: vLLM on ROCm 7.14 (TheRock), torch 2.13. SGLang-Diffusion rows: upstream SGLang 0.5.23 on ROCm 7.2.3, torch 2.10.

| model | metric | stock | hk | stock / hk |
|---|---|---|---|---|
| Qwen3.8-27B GPTQ int4 (hk 0.9.16) | TPOT, 4 / 8 concurrent (ms) | 51.5 / 109.1 | 33.8 / 37.2 | 1.52x / 2.93x |
| | TTFT 512 / 2k tokens (ms) | 375 / 1,507 | 284 / 1,165 | 1.32x / 1.29x |
| Qwen3.8-27B AWQ int4 (AMD Quark W4A16) (hk 0.9.16) | TPOT, 4 / 8 concurrent (ms) | 37.8 / 61.9 | 32.4 / 34.4 | 1.17x / 1.80x |
| | TTFT 512 / 2k tokens (ms) | 333 / 1,401 | 274 / 1,122 | 1.22x / 1.25x |
| Wan2.1-T2V-1.3B, 832x480x81, 20 steps (hk 0.9.14) | seconds per video | 283.6 (282.0-295.1) | 178.0 (177.4-190.0) | 1.59x |
| Qwen-Image 2.1, 1024x1024, 30 steps (hk 0.9.20) | seconds per image | 29.1 (28.0-42.2) | 25.5 (24.4-69.9) | = (ranges overlap) |

- **Single-request TPOT on R9700 is not quoted.** It is bimodal per engine build in both arms (about 29 ms or 37 ms,
  depending on the build, not on hk); within a state hk and stock are equal (1.00x).
- **The Qwen-Image 2.1 row is a shared-box run.** Only the measured card was exclusive; the other cards ran other
  users' jobs at about the same load in both arms (31% / 31%). Over 8 alternating rounds hk was faster in 6 (about 25 s
  against 29 s an image), but one slow outlier in each arm (stock 42 s, hk 70 s) makes the round ranges overlap, so no
  ratio is quoted. Single requests on this box swing from 25 to 55 s for stock and hk alike.
- **One GPTQ round was discarded.** Another user's process ran inside its stock block; the row uses the other 5 of 6
  rounds.
- **Quality.**
  - Wan2.1: hk's video is 28.8 dB from stock's. Against an fp32 reference it is 24.7 dB, stock's 26.3 dB. The
    difference comes from hk's attention route (`sgld_attn`); hk's VAE route alone matches stock (26.25 dB).
  - Qwen-Image 2.1: hk's image is 41.4 dB from stock's.

## The older numbers

The [performance page](../archive/performance-2026-10.md) and the
[development record](../history/README.md) are kept as a dated record of
kernel-level and engine-level measurements. Some of their baselines were
weaker than the opponents' best configuration, so do not read them as current
claims.
