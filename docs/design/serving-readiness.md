# Serving readiness after 0.6: what hk still needs before it is on by default in an LLM deployment

Owner question (2026-10-25): what does 0.6 still lack, in base capabilities and in kernel
validation, before it can be used to optimize LLM inference deployments? This page answers it
with evidence and keeps the two halves apart:
- **DSL / runtime capabilities**: what the language, compiler, runtime and packaging can do;
- **kernel validation**: what has been measured about the kernels built on it.

Every item gives:
- **status**, citing code, a test or a measurement;
- **impact** on a deployment;
- **cost** to close it;
- **priority**: P0 blocks a default-on deployment, P1 limits where hk can be used, P2 is later.

Main is `6a9e2c0`: 0.6.0 plus K2's fused quant (#41). Measurements are on an MI355X (gfx950)
under the vLLM nightly image (`nightly-rocm100-e68efc685ab1`, ROCm 10, hipcc clang 23) unless
noted.

**Update, after this page (owner decision 2026-10-26).** B1 (GSM8K and ARC on real weights)
and B2/B3 (a serving A/B on real weights) passed for K1 and K2. A9 and A3 landed in #44. hk is now
**on by default**: every route validated with real weights is on, and ConvRot stays opt-in.
`HK_DISABLE=1` turns all of it off. At startup, the A9 contract check leaves an engine entirely
stock if it fails. See [integration](../integration.md#on-by-default-and-turning-it-off). The
rest of this page is the audit as it was written.

## What 0.6 is today, in one paragraph

On gfx950, hk ships two prefill kernels for DeepSeek-V4.1-Flash behind opt-in vLLM routes.
- **K2, MXFP8 linear** (`HK_VLLM_MXFP8=1`): takes six measured (N, K) at M >= 2048
  (`python/hk/integration/vllm_mxfp8.py: MEASURED, MIN_M`). It is bit-identical to vLLM's kernel.
- **K1, MoE a8w4** (`HK_VLLM_MOE_A8W4=1`): takes M >= 8192
  (`python/hk/integration/vllm_moe_a8w4.py: MIN_M`). It is deterministic, and its max-rel against
  exact fp32 is at or below AITER's own.
- **TP2 TTFT** on two MI355X (all 40 layers, dummy weights,
  `kda/e2e-v41/RESULTS.md`): both together give 1.05-1.06x at 8k-16k prompt tokens, about 1.01x
  at 2k.

Everything else in a DeepSeek-V4.1 deployment stays on vLLM / AITER. That includes decode,
attention (MLA, sparse indexer), communication and every call made under CUDA-graph capture.

## A. DSL / runtime capabilities

| # | Item | Status (evidence) | Impact on a deployment | Cost | Prio |
|---|---|---|---|---|---|
| A1 | **Decode-shaped kernels for DeepSeek-V4.1** | Partly wrong as stated: hk *has* decode kernels, for dense models. `hk.ops.decode_gemm` covers both GPUs, and paged decode attention is `hk.ops.paged` (gfx1100) / `paged_cdna` (gfx950), 0.90-1.05x aiter at 8-128 requests (0.5.0 notes). The vLLM attention backend gave Qwen3-8B 1.13x per request at 16 requests, 1.21x at 64. **For DeepSeek-V4.1 there is nothing.** No MXFP8 route below M=2048, no MoE route below 8192, no MLA. Decode there is dominated by AMD's persistent megakernel `dsv41_mono`, 83% of a 6.94 ms TPOT step (`~/hk-dx/mk/STATUS.md`, 07:55Z). | hk changes nothing in DSV4.1 TPOT. Decode is a megakernel / comm-overlap question (the mk session's phase 0 verdict: "go, narrowly"), not a GEMM route. | High, and research: it is the mk track | P1 (separate track) |
| A2 | **CUDA/HIP-graph capture** | On gfx950, hk launches on torch's current stream, so graphs capture it (0.5.0 notes), and the vLLM ops are declared `graph="safe"`. K1 falls back to AITER whenever `torch.cuda.is_current_stream_capturing()` (`vllm_moe_a8w4.py:49`). In the TP2 run every captured call fell back: `captured: hk 0, fallback 6360` (MoE) and `hk 0, fallback 33390` (MXFP8), `kda/e2e-v41/ab-tp2-L40.log`. vLLM captures only up to 512 tokens and hk's routes start at 2048. On gfx1100 the default launch is the null stream, so it is not captured (0.6.0 known issue). | No effect on the prefill routes, which are never captured. Any decode route (A1) would have to be capture-safe with static shapes, and K1's capture fallback would have to go. | Low for gfx950 prefill (none); medium for gfx1100 (current-stream launch by default); part of A1 for decode | P2 |
| A3 | **JIT, cold start, compile cache** | K1/K2 kernels compile on first use in each TP worker: about 4.2 s a kernel under ROCm 10 (median of 192 builds in a worktree cache, from kernel.cpp-to-.so mtimes). `python -m hk.integration --warm` builds **attention only** (`integration/__main__.py: warm_jobs`, "Attention only, and deliberately"). The cache is content-addressed on disk and shared, but two workers compiling the same kernel both compile it; the loser of the rename race drops its copy (`runtime/compile.py: build`). The key includes hipcc's version, the target's headers and `hk-dialect-N`, so any ROCm or hk upgrade invalidates the cache (by design, `compile.py` docstring). | First prefill after a start or upgrade pays tens of seconds per worker. Autoscaled or preemptible serving sees it on every new replica. | Low-medium: extend `--warm` to the routed K1/K2 kernels (their names are deterministic from MEASURED / MIN_M), and ship or prebuild a cache per image | **P0** |
| A4 | **Quantization formats** | Covered: MXFP8 block (K2), MXFP4 MoE weights in AITER's layout (K1), INT4 AWQ W4A16 (`vllm_int4`, Qwen3-8B-AWQ 1.87x at one request; it keeps a **second copy** of the weights, `vllm_int4.py` docstring), INT8 W8A8 on gfx1100 (ConvRot), per-row int8 quant (`ops/quant.py`). Not covered: FP8 per-tensor / per-channel GEMM routes (`vllm_gemm` routes unquantized bf16 only), GPTQ, an FP8 KV cache (the paged kernels are bf16: `paged_cdna.py` T_K/T_V bf16). | The common FP8 checkpoints (per-tensor / per-channel) and FP8 KV cache get nothing from hk. The AWQ route doubles weight memory. | Medium per format: the fp8 data path is in main (#28), so FP8 GEMM is a schedule plus scales, and FP8 KV is a paged-kernel variant | P1 |
| A5 | **Attention coverage** | Dense SDPA (`hk.ops.attn`, `attn_cdna`, `attn_k3`), paged decode, and paged prefill with a varlen work list (`ops/paged_prefill.py: work_list`). **No MLA or sparse MLA. No sliding window or softcap in hk.ops**: softcap exists only as a bench/dx task (`bench/dx/solutions/t6_attn_softcap`), and the vLLM backend passes `softcap=0` (`vllm_backend.py:325`). | DeepSeek-V4.x attention (MLA plus the sparse indexer) and Gemma / Mistral-style windows stay on vendor kernels. | High (MLA), medium (window / softcap on top of attn_cdna) | P1 |
| A6 | **Communication** | `hk.comm` primitives (scoped tensors, signal / wait, symmetric memory, grid barrier) are PR #38, ready but not merged. An exact-bf16 TP2 all-reduce is on branch `claude/mk-allreduce`. It is graph-capturable and bitwise exact, but "not a win over vLLM's INT4 recipe on its own" (mk STATUS 09:26Z). There is no GEMM + all-reduce fusion in main. | TP all-reduce stays RCCL / vLLM custom AR. At TP2 an M=8192 bf16 AR is ~1.55 ms (mk STATUS 08:03Z), a large share of a prefill layer that hk cannot touch yet. | High: the mk track's phase 2 | P1 (mk track) |
| A7 | **A multi-toolchain CI matrix** | CI builds on ROCm 7.1 (`ci.yml: ROCM_VERSION "7.1"`, IR / codegen tiers). The GPU tier runs in `rocm/pytorch` 7.2.3 (`scripts/ci/run_gfx950.sh: IMAGE`). **The deployment image (vLLM nightly, ROCm 10, clang 23) is not in CI.** Three silent or blocking differences in two days: (1) roc-7.2.3 sinks V_MFMA_SCALE into a divergent `if` (#35); (2) ROCm 10 copies LDS registers before hk's wait, now refused by the wait gate (#39); (3) `gemm_cdna` dp128x224 spills only under ROCm 10 (`test_gemm_cdna[130-1792-768-dp128x224]`, failing in the vLLM image on main). | A kernel green in CI can be refused, or before #39 silently wrong, in the image that serves. | Medium: a gfx950 GPU tier in the vLLM image (already scripted: `hkx950v`) | **P0** |
| A8 | **Hardware coverage** | Validated: gfx950 (MI355X) and gfx1100 (W7900D). **gfx942 (MI300X) and gfx1201 are compile-only** (0.6.0 notes). There is no MI300X here, so nothing about it has been measured or tested on a GPU. | The most common AMD deployment card has no evidence at all. The CDNA3 path (`include/cdna3`, HipKittens) exists but is unmeasured. | Medium once hardware is available; impossible without it | P1 (P0 for an MI300X deployment) |
| A9 | **Packaging and the engine contract** | Wheels: cp310 (manylinux_2_35) and cp312 (manylinux_2_38) only, and torch is deliberately not a dependency (`docs/building.md`). The DSV4.1 routes were measured on one vLLM nightly (`e68efc685`). They patch vLLM internals: `RocmDotScaledMxfp8LinearKernel.apply_weights` / `process_weights_after_loading`, and `aiter.fused_moe.fused_moe` with its signature (`vllm_mxfp8.py`, `vllm_moe_a8w4.py`). A signature drift silently turned stock MoE into a4w4 once (`kda/e2e-v41/RESULTS.md`, fixed by `__wrapped__`). `hk.integration.stack` checks the stack only for vLLM 0.31 on gfx1100 (#17). There is no compatibility matrix. PyPI publishing from CI is not configured (`docs/building.md`, known issue). | A new vLLM nightly can break or silently bypass a route. Python 3.11 / 3.13 users cannot install. | Low-medium: a contract test per patched entry point, run against the target vLLM; a version table; cp311 / cp313 wheels | **P0** (the contract test) / P2 (more wheels) |
| A10 | **Silent-wrong gates** (added here) | Build-time: the spill gate refuses scratch spills, and the wait gate refuses LDS-register copies before hk's wait (#39). cocheck warns on an MFMA under a partial EXEC (#35). Not at build time: run-to-run determinism and numeric checks. Those run in KDA (`kda_validate`) and per-kernel tests, not on the routed shapes of a deployment. | The gates catch the classes found so far. A new compiler or shape can still produce a wrong route that only a numeric check would catch. | Low: a `python -m hk.check`-style self-test of the routed shapes at engine start (bitwise against the fallback on a small batch) | P1 |

## B. Kernel validation

| # | Item | Status (evidence) | Impact | Cost | Prio |
|---|---|---|---|---|---|
| B1 | **Real weights and accuracy** | **Every DSV4.1 run so far used dummy weights** (`load_format="dummy"`, `kda/e2e-v41/v41_ab*.py`). The checkpoint is 510 GB in 48 shards and only its config is here (`~/hk-dx/e3/DeepSeek-V4.1-Flash-1017`, 14 MB). Per-call evidence exists. K2 is bit-identical to vLLM's kernel, so real weights cannot change its outputs. K1 is not bit-identical: max-rel against exact fp32 is 5.3e-3 (TP1) and 4.2e-3 (TP2), against AITER's 5.4e-3 and 6.8e-3. Token comparison is meaningless with stock itself, because AITER's MoE sums top-k with atomics: every arm, stock included, is "not self-consistent". No lm-eval / GSM8K run exists for any DSV4.1 route. | K1 cannot be defaulted on without an accuracy result on the real model. | Medium: download the checkpoint (510 GB) to a node with two free MI355X; GSM8K + an lm-eval subset with K1 on and off; a deterministic stock arm for token comparison (AITER's MoE with a fixed order, or K1 against itself) | **P0** |
| B2 | **Serving benchmarks** | DSV4.1 has only offline, single-request TTFT (`v41_ab_tp2.py`: one prompt, `max_tokens=1`). The dense-model serving numbers (Qwen3-8B on vLLM, per request at 16-128 requests) are from 0.5.0 and on other routes. There is no throughput / latency-percentile / concurrency / soak / memory data for any DSV4.1 route. | TTFT at one request says nothing about p99 under load. That matters with chunked prefill, where M per step is set by the batch budget, not the prompt. | Medium: `vllm bench serve` (random or ShareGPT lengths, several request rates) with routes on and off, plus a one-hour soak recording the route counters and memory | **P0** |
| B3 | **The shape distribution of real traffic** | Routes are keyed on M and (N, K). K2 takes 6 (N, K) at M >= 2048; K1 takes M >= 8192. Under chunked prefill (`enable_chunked_prefill=True` in the run's config) M per forward is at most `max_num_batched_tokens` and mixes prefill and decode. Which M a deployment actually sees, and so how often a route fires, is unmeasured. In the TP2 A/B (all arms and warm-ups), the MoE route took 600 of 8320 uncaptured calls and the MXFP8 route 4728 of 42858. | The end-to-end gain could be near zero, or the boundaries wrong, for a deployment's real budget. | Low: record M from the route counters during B2 (`--shapes` already exists in `v41_ab.py`) | P1 (comes with B2) |
| B4 | **Data behind each "when to enable" boundary** | Data exists for each:<br>- K1: M >= 8192, from 0.90x at 2048, 1.06-1.08x at 5120, 1.15-1.18x at >= 8192 (c32-c34).<br>- K2: M >= 2048 plus the four-warp small-N schedules (c15), 1.00-1.48x on 18 points.<br>- K3: off by default (`HK_SDPA_K3`).<br>- ConvRot: its group sizes only.<br>All were measured at TP2 shapes, mostly kernel-level, with the end-to-end check at three prompt lengths. | Sound for those shapes. Other TP degrees, or ANY mode (`HK_MXFP8_ANY`), have weaker evidence. | Low: re-derive the boundaries from B2 / B3 | P1 |
| B5 | **One numerics policy** | Each kernel has its own criterion:<br>- K2: bit-identical;<br>- K1: max-rel against exact fp32 <= AITER's own + 1e-2, deterministic;<br>- K3: per-call error below the current path's;<br>- ConvRot: per-call oracle; image drift (PSNR 30.4 dB) noted, not gated.<br>Determinism also differs: K1 is deterministic and the AITER baseline is not. | A deployment owner cannot tell what "correct" means across routes, nor whether a route can change outputs. | Low: one page defining the classes (bit-identical, within-baseline, lossy-by-design), each route's class, and its determinism | P1 |
| B6 | **Memory overhead** (added here) | K2 keeps two extra layouts of each weight's block scales, "a few MB a model" (`vllm_mxfp8.py`). K1 reads AITER's weight layout directly. INT4 AWQ keeps a second copy of the weights (A4). | Negligible for DSV4.1 prefill routes; real for the AWQ route. | Low: measure in B2 | P2 |

## Corrections to the starting list

- **"0.6 kernels are almost all prefill; decode is barely covered."** True for DeepSeek-V4.1,
  false for hk in general. Dense-model decode GEMM and paged decode attention exist on both GPUs
  and were measured in vLLM. For DSV4.1, decode is AMD's `dsv41_mono` megakernel, so it belongs
  to the mk track (A1).
- **"K1 falls back under graph capture, and decode runs in graphs."** Both true. But K1 and K2
  are prefill routes (M >= 8192 / 2048) and vLLM captures at most 512 tokens, so capture costs
  them nothing today (A2).
- **"MLA / sparse MLA (K4)."** There is no MLA in main, under any task name. K4 was the
  `gemm_cdna` wide-block / stream-K GEMM (#19).
- **"hk.comm has just started."** Its primitives are PR #38, ready but not merged. The TP2
  all-reduce exists on a branch and does not beat vLLM's INT4 recipe alone (A6).

## Minimum deployable set: MI355X, TP2, DeepSeek-V4.1-Flash prefill, hk on by default

Close these five, in this order, and K1 + K2 can be default-on in that one deployment. Expect
about 1.05-1.06x TTFT at 8k-16k prompts, from the dummy-weight A/B; to be confirmed with real
weights.

1. **B1, real-weight accuracy** (P0). Real checkpoint, GSM8K + an lm-eval subset with K1 / K2
   on and off, and a deterministic token comparison. K2 is bit-identical, so this is K1's gate.
2. **B2 + B3, a serving benchmark with the shape record** (P0). `vllm bench serve` at several
   rates and a one-hour soak, routes on and off: throughput, TTFT / TPOT p50 / p99, memory,
   route counters, and the M histogram. This also sets the boundaries (B4).
3. **A7, a gfx950 GPU tier in the deployment image** (P0). The K1 / K2 / MX tests under the
   vLLM nightly's ROCm 10 in CI, so a compiler difference shows up before it ships.
4. **A9, an engine-contract test** (P0). For each patched vLLM / AITER entry point: the
   signature, that the route is reached (counter > 0), and that the fallback is stock. Run
   against the pinned nightly, and it fails loudly on drift.
5. **A3, AOT warm for the routed kernels** (P0). `--warm` also builds K1 / K2's kernels for
   the measured shapes, so no request pays a JIT compile, and the cache is prebuilt into the
   image.

Not on the list, on purpose: decode (A1), MLA (A5) and fused communication (A6). They are where
the remaining time in a DSV4.1 deployment is, but each is a new kernel or a research track,
not a readiness gap of the two shipped routes. FP8 per-tensor / per-channel and an FP8 KV cache
(A4), and MI300X (A8), decide which *other* deployments hk can serve. They come next.
