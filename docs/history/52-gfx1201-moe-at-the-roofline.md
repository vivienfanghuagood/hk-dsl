# gfx1201's MoE GEMM at the roofline: hk wins only where vLLM skips the sort

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 50](50-gfx1201-in-vllm.md) left `vllm_moe` off on gfx1201. Its OLMoE decode at 16
requests was 0.90-0.93x stock. An in-engine sweep of the GEMM's schedules found its best
equal to Triton's, 375 against 379 us a call. This chapter asks how much room there is
before tuning anything.

## Both kernels stream at GDDR6's rate

`tools/hk-bench/moe_ab.py` runs both GEMMs of one layer on vLLM's own sort with eight
routings rotated, so the chosen experts' weights are read from GDDR6 each call. It ran
in the R9700 vLLM image. Results are hk's 16-row kernel against Triton's:

| tokens a step | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| OLMoE (64 experts, top 8) | 1.25x | 1.11x | 1.00x | 1.02x | 0.99x | 0.95x | 0.93x | 0.78x |
| Qwen3-30B-A3B (128, top 8) | 1.24x | 1.10x | 1.06x | 1.00x | 0.99x | 0.99x | 1.00x | 0.97x |

From 16 tokens up there is nothing left to take. With the routing uniform, OLMoE at 16
tokens chooses about 56 of its 64 experts, 710 MB of weights for the layer's two GEMMs.
Triton reads them in 1.20 ms, 0.59 TB/s, 96% of the card's 0.62. Qwen3-30B-A3B is the
same: about 82 experts, 778 MB in 1.30 ms. At 64-256 tokens Triton is at 86-96% of that
rate. hk's 16-row blocks fall behind there, because they read an expert's weights once
per block of its tokens. hk on vLLM's 64-row blocks is worse still, 0.27-0.53x at 128
and 256 tokens: a workgroup still holds one 16-row tile of W, and the register budget
leaves it four warps. So a kernel built for those sizes could win at most 4-16%, at
64-256 tokens.

Below 8 tokens, Triton does not stream the few chosen experts at that rate, and hk does.

## In the engine the routing is not uniform

Gate C (`python -m hk.integration ab`, OLMoE, 1024-token prompts, 5 rounds, CUDA
graphs) with hk taking up to 8 tokens a step:

| requests | decode step | verdict |
|---|---|---|
| 1 | 1.12x | win |
| 4 | 0.96x | loss |
| 8 | 0.83x | loss |
| 16 | 1.00x | tie (Triton runs it) |

A model's router puts several of a step's tokens on each expert it chooses, so a
sorted 16-row block is fuller than uniform routing's, and hk's per-block re-read of W
costs more. The win at one request is where vLLM skips the sort: when tokens x top_k x
4 <= experts, it gives each (token, choice) pair a block of its own. For OLMoE that is
up to 2 tokens.

## Two tokens on gfx1201

`vllm_moe.BLOCK16_MAX_TOKENS_AT` gives an arch its own limit, and gfx1201's is 2. Past
it, both the config and the GEMM stay vLLM's. gfx1100's 256 is unchanged.

Gate C at 1, 2 and 4 requests, 1024- and 2048-token prompts, 5 rounds:

| requests | decode step | request time |
|---|---|---|
| 1 | 1.12x / 1.10x | 1.11x / 1.09x |
| 2 | 1.07x / 1.06x | 1.06x / 1.05x |
| 4 | tie / tie | tie / tie |

At every point greedy tokens were identical to stock's (72 of 72), and the routing
counts were equal in every round. `vllm_moe` is now on by default on gfx1201.

## What is left

The MoE GEMM between 64 and 256 tokens a step is the one place a gfx1201 kernel could
still gain, and only 4-16%. It needs blocks that share one read of an expert's weights
across more than 16 tokens. That is a new kernel, not a schedule, and it waits behind
the bf16 GEMM's large-M schedules.
