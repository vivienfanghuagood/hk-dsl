# An MI355X in the engine: where vLLM's time goes there, and what hk's patches buy

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The question: does any model served by vLLM on gfx950 run 1.05x faster
end to end with hk's patches than the engine as it ships, with aiter on?
Or is there no headroom? The answer is yes, from two patches. hk's attention
backend makes Qwen3-8B 1.13x faster at 16 requests and 1.21x faster at 64.
The 4-bit patch makes Qwen3-8B-AWQ 1.87x faster at one request and 1.35x
faster at 16. Every greedy token matches stock's. The linear patch adds
about 1% on top of the backend, all of it in prefill. The norm and MoE
patches have nothing to give: aiter's kernels already beat or match hk's at
the engine's shapes.

## The card and the engine

**Peaks.** `tools/cdna-probes/peaks.hip` measured them. A register-only
`v_mfma_f32_32x32x16_bf16` loop on every CU sustains 2471-2484 TFLOP/s for
0.1 s at 2381-2396 MHz; the datasheet gives 2516.6 at 2400. A read of four
1 GiB buffers, rotated past the 256 MB Infinity Cache, moves 6.54-6.60 TB/s,
and a copy moves 5.7 TB/s, against the datasheet's 8.0 for HBM3E. Those are
`headroom.PEAKS["gfx950"]`, 2.48e15 and 6.6e12.

**Engine.** The engine is `vllm/vllm-openai-rocm:nightly-rocm100-0cbac6cd`,
the image chapter 39 used: ROCm 10 (HIP 7.15), torch 2.12, vLLM
0.30.1rc1.dev558 and aiter 0.1.23. hk runs there with `hk._mlir` built for
cp312 against the cp310 LLVM install, mounted over the checkout's. In that
image hk's GPU tests for the GEMMs, attention, HipKittens, MoE and rows give
249 passed and 31 skipped. `VLLM_ROCM_USE_AITER` defaults to off. With it on,
aiter replaces:

- the linear layers: `rocm_unquantized_gemm`'s tuned GEMM, which is asm
  `hgemm_bf16`, Triton `gemm_a16w16`, a hipBLASLt solution, or the skinny
  `wvSplitK` at up to 5 tokens;
- the fused add + RMSNorm (`add_rmsnorm_quant_kernel`);
- the MoE layers (CK's `kernel_moe_gemm` with aiter's `opus_moe_sorting`).

Attention stays ROCM_ATTN: Triton's `_fwd_kernel` for prefill and ll4mi for
decode. Qwen3's q/k norm and RoPE are Inductor kernels.

## Headroom

Measured with `headroom --graphs`, aiter on, 2048 tokens of context. The
JSONs were reranked with `--from` after the peaks and the MoE floor (an MoE
model's MLP is its experts, so the linear floor counts only dense layers)
landed. Each family's share is of a step; the percentage in parentheses is
of the roofline.

| model, step | ms | linear | attention | MoE | the rest |
|---|---|---|---|---|---|
| Qwen3-8B decode x1 | 3.67 | 72.7% (86%) | 10.1% (12%) | | 1.04 ms |
| Qwen3-8B decode x16 | 4.97 | 58.2% (80%) | 24.4% (60%) | | 1.07 ms |
| Qwen3-8B decode x64 | 8.46 | 40.9% (68%) | 47.6% (73%) | | 1.15 ms |
| Qwen3-8B prefill 8 x 2048 | 216 | 69.1% (62%) | 18.0% (10%) | | 22.6 ms |
| Qwen3-30B-A3B decode x1 | 4.11 | 21.1% | 13.5% | 49.8% | 1.17 ms |
| Qwen3-30B-A3B decode x16 | 5.60 | 21.0% | 18.5% | 45.4% | 1.20 ms |
| Qwen3-30B-A3B decode x64 | 7.87 | 16.1% | 36.1% | 37.3% | 1.25 ms |

The linear layers run near the roofline at decode. The headroom left is
prefill attention, at 10% of peak, and decode attention at one request,
which is too short to fill the card.

## Kernel for kernel

`tools/hk-bench/engine_kernels_gfx950.py` times hk against what the engine
runs, at its shapes, in one process. aiter's GEMM costs ~60 us of Python per
call at M=16, against a 12 us kernel, so an eager burst times the host. Each
timed burst is therefore queued behind a `torch.cuda._sleep` longer than its
enqueue. `hk.autotune.routes.measure_points` now does the same on gfx950.

**Linear (Qwen3-8B).** Measured on the GEMMs of
[chapter 41](41-cdna-leftovers.md), against aiter's tuned GEMMs. Before
chapter 41, hk lost at every decode shape but two (0.69-0.93x). Now:

- one request: gate_up 1.21x and the LM head 1.04x, but qkv, o and down
  0.79-0.85x;
- 16 requests: the LM head 1.17x, the rest 0.86-1.02x;
- 64 requests: qkv, o and down 1.05-1.09x;
- prefill: qkv and gate_up 1.04-1.05x at 16384 rows, down 1.13x at 2048.

From the kernel times, sending only the winning shapes to hk would save
~7% of a one-request decode step and 1-2% elsewhere.

**Attention.** At 8 x 2048, Triton's prefill kernel takes 0.906 ms. hk's
`attn_cdna` takes 0.364 ms (2.49x; 1.87x with layout copies), HipKittens'
GQA 0.318 ms (2.85x) and aiter's FA 0.319 ms. Chapter 39's paged decode is
1.04-1.27x ROCM_ATTN at 8-128 requests, and its paged prefill 1.10-1.24x
aiter.

**Norms and elementwise ops.** After chapter 41's row work too, aiter's
fused add + RMSNorm is 1.05-1.12x hk's. hk's RMSNorm is 0.74-1.05x
aiter's, and its silu_mul 0.96-1.16x Inductor's.

**MoE (Qwen3-30B-A3B).** aiter's fused_moe beats hk's moe_gemm: 37.7 against
40.2 us at one token, 148 against 222 at 16, 209 against 318 at 64. At 2048
tokens hk's block_m 64 schedule fails to verify on gfx950: the op needs
16x32 fragments where it is given 16x16. vLLM's MoE patch rebinds the Triton
experts, so with aiter's CK path on it routes nothing.

## End to end

`python -m hk.integration ab`: CUDA graphs and torch.compile (`HK_EAGER=0`),
aiter on, block size 32 for every arm, 2048 tokens of context, 160 output
tokens, 3 rounds. Ratios are stock over arm, the median of rounds.

| Qwen3-8B | total x1 | x16 | x64 | prefill x16 | decode x16 | decode x64 |
|---|---|---|---|---|---|---|
| hk (attention backend) | 1.014 tie | **1.126** | **1.209** | 1.147 | 1.037 | 1.150 |
| ROCM_AITER_FA | 0.986 tie | 1.093 | 1.147 | 1.142 | 0.997 | 1.016 |
| hk + gemm, shipped table | 1.013 tie | 1.126 | 1.210 | 1.145 | 1.037 | 1.146 |
| hk + gemm, aiter's points | 0.995 tie | **1.142** | **1.217** | 1.200 | 1.048 | 1.161 |

The hk backend beats both stock and vLLM's own aiter FA backend, by 1.055x
total and 1.13x decode at 64 requests. Its prefill matches aiter FA's, and
its decode is chapter 39's kernel.

The shipped linear table was measured against F.linear, and with it the
gemm arm changes nothing. A second table sends hk only the points where it
beat aiter by 3% or more in the kernel bench above, on main's GEMMs. With
it, prefill rises from 1.15x to 1.20x at 16 and 64 requests, and the total
rises by about 1% over the backend alone. In the same run the backend alone
measured 1.129x and 1.207x. The ~7% that one-request decode promised did
not show (0.989x against the backend's 1.010x, both ties).

| Qwen3-8B-AWQ, fp16 | total x1 | x16 | x64 | decode x1 | x16 | x64 |
|---|---|---|---|---|---|---|
| int4, up to 64 tokens | **1.850** | **1.351** | 0.922 loss | 1.890 | 1.529 | 0.856 |
| int4, gfx950 routing | **1.865** | **1.351** | 1.020 | 1.916 | 1.530 | 1.037 |

The first int4 run lost at 64 requests, though `w4a16_ab.py` measured hk at
1.5-4.0x Triton's `awq_gemm` at every M up to 64. Profiling the captured
step showed why: stock vLLM there does not run `awq_gemm`.
`AutoAWQLinearMethod`'s `x.shape[0] >= 256` sees torch.compile's symbolic
size and takes the dequantize branch at every batch. That means
`awq_dequantize` plus a hipBLASLt GEMM, 7.9 ms of the 13.3 ms step at 64
requests. On gfx950 that path is faster than `awq_gemm` everywhere, and hk's
kernel beats it only at 1-16 tokens, before its schedule changes at 24. On
gfx950 the patch now gives hk at most 16 tokens and falls back to the
dequantize.

The hk backend cannot run beside the int4 arm: it declares bf16 only, and
vLLM serves an AWQ model in fp16.

## What each patch can reach

| patch | upper bound (share) | measured ratio | estimate | verdict |
|---|---|---|---|---|
| attention backend | decode 10-48%, prefill 18% | 2.5-2.9x prefill, 1.04-1.27x decode | measured 1.13-1.21x at 16-64 | **past 1.05x** |
| 4-bit (AWQ) | linear 60-75% | 1.5-4.0x | measured 1.35-1.87x at 1-16 | **past 1.05x** |
| linear (bf16) | 41-73% | 0.79-1.21x against aiter | measured +1% over the backend (prefill 1.04x), a tie at one request | no; needs a decode GEMM that beats aiter at every shape, and the one-request gap explained |
| norms, elementwise, RoPE | ~1 ms of a step | 0.73-1.06x | about 0 | no |
| MoE | 37-53% | 0.66-0.94x against CK | about 0 | no; needs a CDNA MoE GEMM, and block_m 64 to verify |

Still to do:

- the hk backend at one request (0.71-0.88x on long contexts);
- fp16 in the backend, so it can run with AWQ models;
- a gfx950 linear table measured against aiter, over the whole grid rather
  than at the A/B's points (`measure_points` queued, `--rotate-mb 768`),
  and why one-request decode does not get the kernels' 7%.
