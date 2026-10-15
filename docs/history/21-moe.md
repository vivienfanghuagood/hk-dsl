# The MoE GEMM, on vLLM's own sorted lists

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

A mixture-of-experts layer is two GEMMs whose rows come from a list.

vLLM's fused MoE first sorts a batch's (token, choice) pairs by expert, with
`moe_align_block_size`. `sorted_ids` holds `token * top_k + choice`, and each
expert's run is padded to a multiple of BLOCK_SIZE_M with the sentinel
`tokens * top_k`. `expert_ids[b]` names block b's expert. Each GEMM is then

    C[sorted_ids[i]] = A[sorted_ids[i] // top_k] @ W[expert_ids[i // BLOCK_M]].T

and the second one multiplies by the routing weight.

[Chapter 19](19-rows-by-index.md) gave the language rows by index. This
chapter writes that GEMM with them, `hk.ops.moe`, and routes vLLM's MoE
layers to it, `hk.integration.vllm_moe`.

## The kernel

At the batches a server decodes, the whole cost is streaming each chosen
expert's weights once. So the kernel is laid out as the decode GEMM of
[chapter 9](09-decode-kernel.md) is:
- W goes straight to registers, on the WMMA's 16-row side;
- each workgroup takes one 16-row block of one expert's W, with its K split
  across the warps and the partial sums meeting through LDS;
- the accumulator is C^T, and its `hk.transpose` is a row-layout tile whose
  rows are tokens, which is what lets the store scatter them.

A's rows come by `gather(..., div=top_k)`. `div` is the one thing this
chapter added to the language: vLLM's list counts (token, choice) pairs, and
the token's row is the pair over top_k. The sentinel lands on row `tokens`,
past A's end, so it reads zeros, and the scatter drops it.

A block past `num_tokens_post_padded` runs its K loop zero times rather than
branching around the kernel. The combine has barriers, which every warp must
reach, and a zero-trip loop reads no weights.

**The naive assignment.** With few tokens against many experts
(`tokens * top_k * 4 <= experts`) vLLM skips the sort. Block b is pair b
alone, of expert `topk_ids[b]`. The same kernel takes that case: its index
list is 0..pairs-1, and block b gathers from position b with the list ending
at b + 1, so its other 15 rows read zeros and store nothing.

**Registers.** Two limits were measured, and `schedule_for` stays inside
both:
- The combine holds every other warp's partial at once: 77, 141 and 269 VGPRs
  at 4, 8 and 16 warps of 32 tokens. The `sched_barrier`s between its loads
  did not batch them, though `decode_gemm`'s identical code is batched. So
  the warp count falls with the block size: 16 warps at 16 rows, 8 at 32, 4
  at 64.
- A 16 x 64 token tile is 32 VGPRs, held twice by the load pipeline, so a
  round loads at most four of them.

Block sizes of 128 have no schedule.

## Against vLLM's Triton kernel

`tools/hk-bench/moe_ab.py`, on vLLM 0.31:
- vLLM picks its config (`try_get_optimal_moe_config`) and its assignment
  (`_prepare_expert_assignment`);
- both kernels run both GEMMs of one layer on the same lists, checked against
  each other first;
- eight routings rotate, so the chosen experts' weights are not left in the
  Infinity Cache by the call before.

µs per layer, both GEMMs; hk16 is hk on a sort redone at 16-row blocks.

| tokens | OLMoE-1B-7B (64 experts, top 8) Triton | hk16 | | Qwen3-30B-A3B (128, top 8) Triton | hk16 | |
|---|---|---|---|---|---|---|
| 1 (naive) | 385.7 | 139.1 | 2.77x | 263.2 | 110.8 | 2.37x |
| 2 (naive) | 607.3 | 260.9 | 2.33x | 379.5 | 199.4 | 1.90x |
| 4 | 792.6 | 432.7 | 1.83x | 548.8 (naive) | 371.6 | 1.48x |
| 8 | 987.1 | 686.0 | 1.44x | 823.0 | 627.3 | 1.31x |
| 16 | 1226.8 | 896.7 | 1.37x | 1182.5 | 1040.0 | 1.14x |
| 32 | 1386.2 | 1020.1 | 1.36x | 1487.2 | 1318.3 | 1.13x |
| 64 | 1454.1 | 1078.1 | 1.35x | 1640.3 | 1516.9 | 1.08x |
| 128 | 1614.5 | 1259.9 | 1.28x | 1883.3 | 1619.4 | 1.16x |
| 256 | 1703.1 | 1697.6 | 1.00x | 1955.1 | 1827.2 | 1.07x |
| 512 | 1748.9 | 2608.9 | 0.67x | 2063.0 | 2378.9 | 0.87x |
| 1024 | 2274.5 | 4349.3 | 0.52x | 2357.5 | 3649.3 | 0.65x |
| 4096 | 6312.0 | 15415.0 | 0.41x | 5289.6 | 10938.4 | 0.48x |

Up to 32 tokens hk streams the chosen experts at 718-785 GB/s. On vLLM's own
64-row blocks it loses at every size it was measured (0.35-0.83x). The
16-row layout re-reads A once per 16 rows of W, which is what a prefill
cannot afford.

## The route

`HK_VLLM_MOE=1` rebinds two names in every vLLM module that imported them:
- **`try_get_optimal_moe_config`** returns vLLM's config with BLOCK_SIZE_M
  16 for an unquantized layer up to `BLOCK16_MAX_TOKENS` (256) tokens. The
  sort and the GEMM both read it, so they change together.
- **`invoke_fused_moe_triton_kernel`** runs hk's GEMM on 16-row blocks it
  takes. Everything else goes back to vLLM's kernel: fp8, int8, int4, a
  biased expert, and every larger batch. The counters say how many did.

vLLM's MoE layer is a custom op that the compiler does not trace into, so
the stand-in runs on every call, captured or not. The compile-cache trap of
[chapter 20](20-vllm-031.md) does not apply.

## End to end

`allenai/OLMoE-1B-7B-0125-Instruct`, bf16, vLLM 0.31.0 under its defaults
(CUDA graphs, prefix caching, chunked prefill). 72 tokens, best of two
rounds, one process per arm. In `hk+moe`, prefill attention is hk's. Decode
attention was Triton's kernel on hk's cache at every batch. That routing was
in force for this run, and [chapter 20](20-vllm-031.md) later reverted it
above 8 requests. The 16- and 32-request rows of `hk+moe` were not measured
again under the reverted boundary, because the pod that held vLLM 0.31 was
evicted first ([chapter 23](23-integration-path.md)). Every arm's greedy tokens are
identical to stock's.

The captured graphs called hk's GEMM 992 times and vLLM's 0 times. The 6528
uncaptured fallbacks are prefill chunks above 256 tokens.

| ctx x batch | stock ms | moe | hk+moe | step: stock ms | moe | hk+moe |
|---|---|---|---|---|---|---|
| 1024 x 1 | 760 | 540 (1.41x) | 396 (1.92x) | 9.93 | 6.83 (1.45x) | 4.84 (2.05x) |
| 1024 x 4 | 1021 | 800 (1.28x) | 648 (1.57x) | 12.14 | 9.01 (1.35x) | 6.95 (1.75x) |
| 1024 x 16 | 2238 | 2022 (1.11x) | 1885 (1.19x) | 22.70 | 19.64 (1.16x) | 18.30 (1.24x) |
| 1024 x 32 | 3368 | 3235 (1.04x) | 2886 (1.17x) | 29.46 | 27.71 (1.06x) | 23.97 (1.23x) |
| 3072 x 1 | 1102 | 885 (1.24x) | 497 (2.22x) | 13.55 | 10.49 (1.29x) | 5.18 (2.62x) |
| 3072 x 4 | 1876 | 1594 (1.18x) | 1127 (1.67x) | 18.62 | 14.64 (1.27x) | 9.15 (2.03x) |
| 3072 x 16 | 5066 | 4831 (1.05x) | 4091 (1.24x) | 38.22 | 34.88 (1.10x) | 31.62 (1.21x) |
| 3072 x 32 | 8676 | 8511 (1.02x) | 6930 (1.25x) | 55.83 | 53.58 (1.04x) | 44.44 (1.26x) |

The MoE patch alone is worth most at the smallest batches, as the kernel
table says it should. hk's attention adds the rest. OLMoE has 16 KV heads
and no GQA, twice Qwen3-8B's KV per token, so the backend choice
(TRITON_ATTN's kernel against stock's ROCM_ATTN fallback) weighs more here.
Prefill is hk's paged prefill, 1.05-1.17x on the prefill phase alone.

## What is left

**Prefill.** Above 256 tokens the layer stays on vLLM's kernel. A tiled MoE
GEMM, staging the gathered rows through LDS as `hk.ops.gemm` stages its
tiles, would be the next kernel. The 64-row numbers above show that the
16-row layout cannot be stretched to it.

**int4 MoE.** vLLM 0.31 already has `moe_gptq_gemm_rdna3`, a HIP W4A16 MoE
kernel for this card. It is the baseline the int4 milestone has to beat.
