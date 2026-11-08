# The scorecard: the only performance numbers we quote

> **In progress.** The Radeon section is complete. The MI355X section still states one result while its
> end-to-end table is re-run; it will then carry the method, the software stack and image, every arm and the
> latest table, as the Radeon section does.

## The rule

A performance claim from this project quotes an end-to-end serving result
from the scorecard, taken as [the method](method.md) says. Kernel-level numbers (one kernel against another kernel)
guide the work, but they are not claims. They depend too much on how well the
opponent was configured.

## Instinct MI355X: DeepSeek-V4.1-Flash on vLLM

- **The setup:** vLLM serving DeepSeek-V4.1-Flash with real weights, tensor
  parallel over two MI355X.
- **The baseline:** stock vLLM with AITER's tuned paths. These are its tuned
  GEMM, a MoE table tuned on real routing, and its fused sparse MLA.
- **What hk adds:** its default routes on top of that baseline. Each arm is
  run at its best configuration.

| | against the AITER-tuned stock |
|---|---|
| 8k-token prefill throughput | **about +5%** |
| decode time per output token | unchanged (within the spread) |

Against a stock vLLM *without* AITER's tuned paths, hk's default routes show a
larger gain. A stock vLLM configured with AITER's tuned paths reaches about
the same gain without hk, so we do not quote it.

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
| MiniCPM-V-4.7 1B (hk 0.9.14) | TPOT, 1 / 4 / 8 concurrent (ms) | 21.5 / 22.3 / 27.2 | 21.3 / 22.2 / 27.3 | 1.01x / = / = |
| | TTFT 8k / 32k (ms) | 656 / 7,451 | 270 / 1,384 | 2.43x / 5.38x |
| | GSM8K | 81.50% | 80.89% | not significant (p 0.42) |
| MiniCPM-V-4.7 35B-A3B, 2 GPUs (hk 0.9.14) | TPOT, 1 / 4 / 8 concurrent (ms) | 61.8 / 63.1 / 73.3 | 61.2 / 62.0 / 64.5 | = / = / 1.14x |
| | TTFT 8k / 32k (ms) | 2,277 / 18,070 | 1,575 / 6,957 | 1.45x / 2.60x |
| | GSM8K | 95.22% | 95.83% | not significant (p 0.08); a second accuracy run is planned |
| Wan2.1-T2V-1.3B, 832x480x81, 20 steps (hk 0.9.14) | seconds per video | 496.7 (496.5-497.5) | 187.1 (187.1-187.6) | 2.66x |
| Qwen-Image 2.1, 1024x1024, 30 steps (hk 0.9.14) | seconds per image | 42.0 | 26.2 | 1.60x |

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
| Qwen-Image 2.1, 1024x1024, 30 steps (hk 0.9.14) | seconds per image | 27.4 (27.3-28.1) | 25.1 (25.0-25.1) | 1.09x |

- **Single-request TPOT on R9700 is not quoted.** It is bimodal per engine build in both arms (about 29 ms or 37 ms,
  depending on the build, not on hk); within a state hk and stock are equal (1.00x).
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
