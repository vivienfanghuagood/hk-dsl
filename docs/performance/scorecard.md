# The scorecard: the only performance numbers we quote

> **Placeholder.** This page will hold the full scorecard: the method, the
> software stack and image, every arm and the latest table. Until then it
> states the one result we quote and where the rest of the numbers stand.

## The rule

A performance claim from this project quotes an end-to-end serving result
from the scorecard. Kernel-level numbers (one kernel against another kernel)
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

**RDNA end-to-end results will come with the RDNA scorecard.** That scorecard
first sets fair baselines: a current AOTriton, hipBLASLt with TunableOp, and
each engine's latest RDNA path. Until it exists, we quote no RDNA numbers.

## The older numbers

The [performance page](../archive/performance-2026-10.md) and the
[development record](../history/README.md) are kept as a dated record of
kernel-level and engine-level measurements. Some of their baselines were
weaker than the opponents' best configuration, so do not read them as current
claims.
