# Performance

Every number on this page is copied from the [development record](history/README.md),
with a link to the chapter it comes from; the chapter has the method, the
harness in `tools/hk-bench/` and the caveats. Unless a section says otherwise
the hardware is one W7900D (gfx1100, RDNA3; 864 GB/s of GDDR6). The numbers are
as measured at the time. When the IR moved to MLIR, the kernels that matter
were re-measured against their old selves and came out within noise
([after the MLIR move](#after-the-mlir-move)).

How the numbers were taken matters as much as the numbers: contenders run in
one process, interleaved, with correctness checked before timing, because
cross-process A/B on this chip manufactures 13-16% differences out of nothing
([Phase 5](history/04-phase-5-autotune.md)). Ratios are hk's speed relative to
the baseline named in each table (above 1 is hk faster) unless the table says
it is a time ratio.

- [Summary](#summary)
- [Against Triton and FlyDSL](#against-triton-and-flydsl)
  - [Triton tutorials and the kernels beside a model](#triton-tutorials-and-the-kernels-beside-a-model)
- [GEMM](#gemm)
- [Attention forward](#attention-forward)
- [Norms and elementwise](#norms-and-elementwise)
- [Decode GEMV](#decode-gemv)
- [Paged decode attention](#paged-decode-attention)
- [Paged prefill](#paged-prefill)
- [End to end in vLLM](#end-to-end-in-vllm)
  - [FlyDSL's GEMM on the routes](#flydsls-gemm-on-the-routes)
- [End to end in SGLang](#end-to-end-in-sglang)
- [After the MLIR move](#after-the-mlir-move)
- [Against torch, at full size, after the cross-warp fix](#against-torch-at-full-size-after-the-cross-warp-fix)
- [CDNA, on one MI355X](#cdna-on-one-mi355x) -- a different GPU; kept apart from everything above

## Summary

Grouped by the question each result answers. The last group holds results
that newer ones replaced (an older vLLM, torch 2.10, a kernel since
rewritten); they stay because the chapters cite them.

### End to end, against the stock engine

| kernel | result | against | verdict | source |
|---|---|---|---|---|
| vLLM 0.31 end to end, Qwen3-8B, attention backend / both patches | 1.05-1.48x / 1.20-1.75x per request; 1.01-1.39x TRITON_ATTN | stock vLLM 0.31.0 (ROCM_ATTN), and the TRITON_ATTN backend hk replaces | win at every shape; the gain over TRITON_ATTN is prefill | [vLLM 0.31](history/20-vllm-031.md) |
| vLLM 0.31 end to end, Qwen3-8B, linear routes with FlyDSL's GEMM rows | decode step 1.14x hk+gemm's at 96-128 requests (1.394x / 1.666x stock); prefill 0.9-2.5% faster at 1-4 requests; tie elsewhere. Re-measured after chapter 28, most prefill rows went: decode the same (1.413x / 1.651x stock), prefill 1.5-1.7% faster at 8192-token steps and 1.2% slower at a 1024-token one | the same patches routed by the table without FlyDSL rows; stock vLLM 0.31.0; the table before re-measuring | win where a step's M is one FlyDSL's tiles divide and it measured faster; otherwise unchanged | [chapter 30](history/30-flydsl-routes-gate-contracts.md), [chapter 31](history/31-remeasured.md) |
| vLLM 0.31 end to end, Qwen3-8B-AWQ, 4-bit patch | 1.07-3.80x per request; decode step 1.57-4.46x | stock vLLM 0.31.0, fp16, Triton AWQ | win at every shape | [4-bit weights](history/22-int4.md) |
| vLLM 0.31 end to end, Qwen3-14B-AWQ, 4-bit patch | 1.16-4.43x per request; decode step 2.87-5.52x | stock vLLM 0.31.0, fp16, Triton AWQ | win at every shape | [4-bit weights](history/22-int4.md) |
| vLLM 0.31 end to end, OLMoE-1B-7B, MoE patch / with hk's attention | 1.02-1.41x / 1.17-2.22x per request; decode step to 2.62x | stock vLLM 0.31.0 | win at every shape | [MoE](history/21-moe.md) |
| SGLang 0.5.18 end to end, Qwen3-8B, linear patch | 1.49-1.81x output throughput at concurrency 1-32; time per output token 1.59-1.81x | stock SGLang with sglang-radeon, eager, Triton attention | win at every concurrency | [SGLang](history/24-sglang.md) |
| SGLang 0.5.18 end to end, OLMoE-1B-7B, MoE patch | 1.06-1.32x output throughput at concurrency 1-32 (stock SGLang does not run it) | SGLang with the top-k in torch and its Triton MoE GEMMs | win at every concurrency | [SGLang MoE](history/25-sglang-moe.md) |

### Kernels against Triton and FlyDSL, every arm tuned

| kernel | result | against | verdict | source |
|---|---|---|---|---|
| GEMM, bf16, Qwen3-8B's linear shapes | 87.9-91.3 TFLOP/s | FlyDSL's dispatcher and its autotuner; Triton autotuned over gfx1100's space; hipBLASLt | tie with FlyDSL at all five shapes, 0.99-1.01x (from 0.95-0.98x), once `s_setprio` stopped fencing the scheduler; win over tuned Triton 1.09-1.11x and hipBLASLt 1.28-1.39x | [the GEMM schedule](history/28-gemm-schedule.md) |
| RMSNorm (weighted), fused add + RMSNorm, softmax, on register strips | RMSNorm 1.02-1.12x, softmax 1.02x, fused add 0.83-1.11x across runs | the fastest of Triton and FlyDSL, each autotuned, and Inductor | RMSNorm and softmax win or tie at 1-8192 rows; the fused add ties, with a loss at 1 row in one run of five (the clock) and at 8192 rows in one of two (buffer placement) | [strips](history/27-strips.md) |
| attention forward, n4096 GQA 4, against torch 2.13 | 1.16x causal, 0.99x non-causal | torch 2.13.0+rocm7.2's SDPA, 3.3x faster than 2.10's causal | win causal, tie non-causal; the 2.8-4.0x elsewhere on this page is against an older torch (2.10 where recorded) | [chapter 30](history/30-flydsl-routes-gate-contracts.md) |
| dropout, KV cache write, LayerNorm backward, embedding (`examples/07`-`10`) | dropout 1.02x; KV write 1.18-3.00x; LayerNorm backward 1.12x at 1024 wide, 0.66-0.92x wider; embedding 1.18-1.28x torch at 1-32 tokens, 0.93x Triton at 4096 | the Triton tutorial or a Triton row kernel; torch | win for dropout, the KV write and LayerNorm backward at 1024; loss where a 16-row tile cannot hold a 2048+ wide row, and for the embedding at 4096 tokens | [rows by index](history/19-rows-by-index.md) |

### Kernels against the kernel the engine runs

| kernel | result | against | verdict | source |
|---|---|---|---|---|
| decode GEMV | 1.93-2.03x per step, 1-32 tokens | hipBLASLt | win; upstream vLLM's `wvSplitK` is ahead at one token | [cross-workgroup reduction](history/11-cross-workgroup-reduction.md) |
| decode GEMV, Qwen3-8B's layers, 1-64 tokens | 1.18-2.30x | `F.linear`, torch 2.9.1 (hipBLASLt): what SGLang's linear runs on ROCm | win at every shape | [SGLang](history/24-sglang.md) |
| GEMM on Qwen3-8B's linear layers | 1.09-1.23x at M ≥ 1024 | hipBLASLt (`F.linear`) | win at M ≥ 1024; a comb below it, with losses | [linear layers](history/08-linear-layers.md) |
| W4A16 decode GEMM, fp16, Qwen3-8B's layers | 2.20-2.61x at 1-16 tokens, 1.40-1.84x at 32-64 | vLLM 0.31's Triton AWQ kernel (what AWQ runs on ROCm) | win at every shape | [4-bit weights](history/22-int4.md) |
| W4A16 decode GEMM, against `gptq_gemm_rdna3` | 0.69-0.90x at 1-4 tokens; 1.13-2.25x from 8 | vLLM's HIP W4A16 kernel for gfx1100 | loss to 4 tokens, a skinny GEMV without WMMA's padding; win above | [4-bit weights](history/22-int4.md) |
| MoE GEMM (bf16), OLMoE and Qwen3-30B-A3B shapes | 1.00-2.77x at 1-256 tokens; 0.41-0.87x at 512-4096 | vLLM 0.31's Triton `fused_moe` kernel, its own config | win to 256 tokens, which the route takes; loss above, which stays on Triton | [MoE](history/21-moe.md) |
| MoE layer (bf16) in SGLang, OLMoE and Qwen3-30B-A3B shapes | 1.03-2.51x at 1-256 tokens; 0.61-1.01x at 512-2048 | SGLang's Triton fused-MoE, its default configs | win to 256 tokens, which the route takes; above, it stays on Triton | [SGLang MoE](history/25-sglang-moe.md) |
| paged prefill | 3.6-7.9x | vLLM's Triton `unified_attention` | win | [captured whole](history/17-the-decode-step-captured-whole.md) |
| paged decode attention, vLLM 0.31 | 0.83-0.95x per layer at 16-64 requests | Triton's `unified_attention` on hk's cache, triton-rocm 3.7.1 | loss, kernel for kernel; yet routing 16+ requests to Triton slowed the captured step (4096 x 16: 59.0 -> 74.2 ms), so only up to 8 go there | [vLLM 0.31](history/20-vllm-031.md) |

### Earlier results, since replaced

| kernel | result | against | verdict | source |
|---|---|---|---|---|
| GEMM, bf16 | 75.0 TFLOPs at 4096³, 77.9 at 8192x8192x4096 | handwritten HipKittens C++; rocBLAS | tie with the C++ (1.00x); rocBLAS 0.93x and 0.92x; since overtaken by hk's later schedule with one LDS buffer | [Phase 3](history/02-phase-3-gemm.md) |
| attention forward | 61.0-64.9 TFLOPs | handwritten C++; torch SDPA (aotriton) | tie with the C++, 1.008x causal; 2.9-3.2x torch, 4.0x causal (torch 2.10) | [Phase 4](history/03-phase-4-attention.md) |
| norms, elementwise, RoPE, quantize | 13 of 15 cases faster | `torch.compile` | win; the other two are a tie | [Phases 1 and 2](history/01-phases-1-2.md) |
| RMSNorm, SiLU-mul | slower | Inductor, at Qwen3-8B shapes | loss; the patches were dropped | [strided globals](history/12-strided-globals.md) |
| paged decode attention | 0.66-0.97x per layer up to 8 requests, 1.02-1.04x at 16, 2.6x at 32 | vLLM's Triton attention, captured | loss below 16 requests, where the backend runs Triton's kernel on hk's cache | [captured whole](history/17-the-decode-step-captured-whole.md) |
| vLLM end to end, linear-layer patch | 1.10-1.84x per request | stock vLLM 0.16.1.dev0 | win at every shape, in both modes | [cross-workgroup reduction](history/11-cross-workgroup-reduction.md) |
| vLLM end to end under its defaults, attention backend / both patches | 1.007-1.48x / 1.48-1.85x per request | stock vLLM 0.16.1.dev0, CUDA graphs, prefix caching, chunked prefill | win at every shape | [captured whole](history/17-the-decode-step-captured-whole.md) |
| shipped ops at 7-8B shapes (`examples/validate.py`) | GEMM 1.12x, decode GEMV 3.90-4.38x, attention 2.76x / 3.94x causal, quantize 2.78x; norms 0.88-0.96x | torch eager, PyTorch 2.10 | win, except norms and elementwise, 2-12% behind torch's own fused kernels | [validation](history/16-validation-and-the-cross-warp-race.md) |

## Against Triton and FlyDSL

[The scoreboard, retuned](history/26-the-scoreboard-retuned.md).
`tools/hk-bench/scoreboard.py`: every way to compute each op in one process,
each checked against fp32 before it is timed, each in a profiler session of
its own per round, rounds rotated, best round. Every DSL runs out of the box
and tuned, and records what it searched:
- hk: its default GEMM schedule, and its tuner's record (96 points);
- Triton: the tutorial kernel, and the same kernel autotuned over gfx1100's
  space (120 matmul configs; num_warps x waves_per_eu for the row kernels),
  ranked with the scoreboard's own warm timing;
- FlyDSL 0.4.0.dev970: its dispatcher (`pick_tile` and per-tile options),
  and its autotuner forced to search.

Its flash attention does not run on gfx11, which has no global-to-LDS load.
The rest of the stack was torch 2.10.0+rocm7.2.4 and Triton 3.6.0, which
matters for attention: torch 2.13's SDPA is 3.3x faster (below).
GPU ms. "hk", "Triton" and "FlyDSL" are each the faster of that DSL's two
arms; the verdict is hk's against the fastest other, a tie within the
larger of 2% and the two arms' noise. Rows marked * are best of 9 rounds of
50 calls, the rest best of 5 of 10.

| case | hk | torch | Inductor | Triton | FlyDSL | hk vs best | chapter 18 said |
|---|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.5229 | 1.9682 | -- | 1.6638 | 1.5423 | 1.01x tie | 1.05x |
| gemm 2048x6144x4096 | 1.1285 | 1.5719 | -- | 1.2523 | 1.1325 | 1.00x tie | 1.08x |
| gemm 2048x4096x4096 | 0.7785 | 0.9988 | -- | 0.8615 | 0.7721 | 0.99x tie | 1.08x |
| gemm 2048x24576x4096 | 4.6041 | 5.9432 | -- | 5.0004 | 4.6292 | 1.00x tie | 1.05x |
| gemm 2048x4096x12288 | 2.3099 | 2.9459 | -- | 2.5596 | 2.2950 | 0.99x tie | 1.03x |
| attention n4096 causal, GQA 4 | 3.0094 | 10.8671 | -- | -- | -- | 3.61x win | 3.65x |
| attention n4096, GQA 4 | 4.6924 | 13.2199 | -- | -- | -- | 2.82x win | 2.75x |
| rms_norm 1x4096* | 0.0024 | 0.0042 | 0.0026 | 0.0026 | 0.0025 | 1.04x win | 0.41x |
| rms_norm 32x4096* | 0.0023 | 0.0044 | 0.0027 | 0.0027 | 0.0025 | 1.12x, a tie by noise | 0.47x |
| rms_norm 2048x4096 | 0.0184 | 0.0263 | 0.0197 | 0.0193 | 0.0190 | 1.04x win | 0.70x |
| rms_norm 8192x4096 | 0.1699 | 0.1771 | 0.1775 | 0.1744 | 0.1741 | 1.02x win | 0.92x |
| fused add + rms_norm 1x4096* | 0.0032 | 0.0060 | 0.0026 | 0.0027 | 0.0029 | 0.83x loss (0.83-1.08x over five runs) | 0.37x |
| fused add + rms_norm 32x4096* | 0.0025 | 0.0064 | 0.0029 | 0.0028 | 0.0030 | 1.11x, a tie by noise | 0.51x |
| fused add + rms_norm 2048x4096 | 0.0362 | 0.0488 | 0.0396 | 0.0363 | 0.0358 | 0.99x tie | 0.78x |
| fused add + rms_norm 8192x4096 | 0.3390 | 0.4547 | 0.3708 | 0.3280 | 0.3690 | 0.97x loss (1.02x tie in the other run) | 0.97x |
| softmax 2048x4096 | 0.0187 | 0.0304 | 0.0573 | 0.0255 | 0.0191 | 1.02x tie | 0.63x |
| softmax 8192x4096 | 0.1686 | 0.1755 | 0.2240 | 0.1725 | 0.1725 | 1.02x win | 0.93x |

The GEMM and attention rows are [chapter 28](history/28-gemm-schedule.md)'s:
without the `s_setprio` fence around each WMMA group, hk's schedule gets the
interleave FlyDSL's hints ask for, and the GEMM ties FlyDSL at every shape.
The paragraphs below are chapter 26's account of the gap that closed.

Chapter 18's GEMM win was against FlyDSL's bare 128x128x32 default (1.743 ms
at 4096³) and Triton's seven tutorial configs. FlyDSL's dispatcher runs
4096³ in 1.547 ms. FlyDSL's tile with each option taken away, beside hk's
(`tools/hk-bench/gemm_vs_flydsl.py`), shows where the lead comes from:
`sched_group_barrier` hints are worth 5-6% at every shape, and a per-workgroup
rotation of the K start 5-12%. Bare, its kernel is 0.89-0.97x hk's.
- hk can express the rotation (`gemm_kernel(stagger=)`), and it does not pay
  in hk's schedule: 0.91-1.00x at ten of eleven shapes.
- hk could not express the hints: `hk.arch` had a full `sched_barrier` but
  no instruction-group interleave. Chapter 28 added
  `hk.arch.sched_group_barrier`; with `s_setprio` gone, the hint added
  nothing on top.
- FlyDSL's four-wave 64x64 warp tile spilled in hk's `dot_tile` (281 VGPRs),
  which kept its whole A slice live where FlyDSL streams one fragment.
  Chapter 28's `gemm_kernel(stream_a=True)` streams A: 256 VGPRs and no
  spill under ROCm 7.1, 3 spilled under 7.2.
- FlyDSL's 256x256x32 tile fits hk (252 VGPRs) and is 0.77-0.88x.
- n_split=2, now hk's default, was +0.4-2.6% at 15 of 16 shape-runs.

The norm and softmax rows are [chapter 27](history/27-strips.md)'s: the ops
run on register strips, every RDNA kernel reads its arguments at entry, and
past the Infinity Cache the strip kernels cap their occupancy. The rest of
this list is chapter 26's diagnosis of the tile kernels, which strips
replaced; it stays because the tile kernels still run rows wider than 32768.

The norm losses chapter 26 measured, and why:
- **1-32 rows** is the instruction stream. hk's folded RMSNorm is 846
  instructions a warp, with 16 b128 loads, 8 b128 stores and 3 barriers.
  Triton's, at its tuned 16 warps, is 221 instructions a wave, with 2 loads,
  1 store and 2 barriers. A bare hk tile copy of one row takes 3.47 us, and
  Triton's whole RMSNorm 2.69.
- **2048 rows** (33 MB, Infinity-Cache resident): hk's register tiles have
  WMMA's lane layouts. A 16-bit row tile holds each row twice, in lanes l and
  l + 16, so every load and store moved each 32-byte run from both wave
  halves. HipKittens' `HK_RT_HALF_SPLIT` now moves it once. On the folded
  plans: rms_norm 27.17 to 24.72 us, the fused add 46.12 to 40.15; softmax,
  switched to the bounded accesses, 29.99 to 23.22, ahead of tuned Triton's
  25.73. The remaining gap was the absence of a register layout in which a
  lane owns a contiguous run of one row, which is what Triton's blocked
  layout and FlyDSL's tiled copies give their kernels. Chapter 27's strips
  are that layout.
- **8192 rows** is DRAM-bound, 3-9% behind. The unfolded plan's weight is a row
  vector broadcast down a WMMA-shaped tile, read with 128 element loads a
  lane.

So on this card at these shapes, as of chapters 27 and 28: hk and FlyDSL
tie the GEMM, hk wins the norms and softmax (the fused add ties) and
attention, where neither other DSL has a gfx11 kernel (against torch 2.10's
SDPA; against torch 2.13's, 1.16x causal and a tie non-causal); tuned Triton
and hipBLASLt are behind both on the GEMM.

The performance gate's baseline (`tests/perf/baselines/gfx1100.json`,
[chapter 30](history/30-flydsl-routes-gate-contracts.md), re-measured in
[chapter 31](history/31-remeasured.md)) is the same scoreboard's
out-of-the-box arms, six runs on the vLLM 0.31 venv: torch 2.13.0+rocm7.2,
Triton 3.8.0, FlyDSL 0.4.0, hk with chapters 27 and 28. There FlyDSL's
dispatcher takes 1.00-1.03x hk's time on the GEMM (0.97-0.98x before
chapter 28). On the 2048- and 8192-row norms and softmax each
out-of-the-box arm takes 0.99-3.03x hk's time (Triton's tutorial kernels,
not the tuned ones above; the 0.99x is FlyDSL's 2048-row fused add). Attention changed with torch: 2.13's SDPA runs the causal n4096
case in 3.24-3.27 ms where 2.10's took 10.83, so hk's attention is 1.16x
torch causal and 0.99x non-causal on that torch, against 3.61x and 2.82x in
the table above (chapter 26: 3.80x and 2.73x). The attention rows above are
torch 2.10's.

### Triton tutorials and the kernels beside a model

[Rows by index](history/19-rows-by-index.md): `examples/07`-`10`. Each is
checked against fp32 torch before it is timed (the dropout's mask also
against Triton's, which it equals), all arms in one process, best round. ms.

| kernel | shape | hk | Triton | torch | hk vs Triton |
|---|---|---|---|---|---|
| dropout, p = 0.5 (tutorial 04) | 8192 x 4096 bf16 | 0.415 | 0.424 | 0.278 | 1.02x |
| KV cache write, -1 slots skipped | 16 tokens, 1024 wide | 0.005 | 0.014 | 0.107 | 3.00x |
| | 256 tokens | 0.005 | 0.014 | 0.110 | 2.88x |
| | 4096 tokens | 0.012 | 0.014 | 0.146 | 1.18x |
| LayerNorm backward (tutorial 05), 4096 rows | 1024 wide | 0.040 | 0.044 | 0.070 | 1.12x |
| | 2048 | 0.065 | 0.060 | 0.110 | 0.92x |
| | 4096 | 0.153 | 0.105 | 0.203 | 0.69x |
| | 8192 | 0.534 | 0.354 | 0.564 | 0.66x |
| embedding, 151936 x 4096 bf16, ids rotated past the cache | 1 token | 0.005 | 0.012 | 0.006 | 2.4x |
| | 32 tokens | 0.005 | 0.012 | 0.006 | 2.4x |
| | 4096 tokens | 0.090 | 0.084 | 0.083 | 0.93x |

The dropout is bound by Philox's multiplies in both DSLs; torch's is 1.5x
because it uses all four outputs of a Philox call. The LayerNorm backward
holds a row in registers up to 2048 wide; wider, its 16-row tile reads the
row twice, where Triton's one-row program holds it. The embedding at 4096
tokens is the row layout's 16 random rows per load.

## GEMM

### Against the handwritten C++ and rocBLAS

[Phase 3](history/02-phase-3-gemm.md). ROCm 7.2.4; same process, interleaved,
bit-identity checked before timing (`tools/hk-bench/gemm_ab.py`). `hk.ops.matmul`
and the handwritten `config<128,128,64,16,8,4,8,4>` compile to the same 209
VGPRs, 0 scratch, 0 spill, 7 waves/SIMD.

| shape | variant | ms | TFLOPs | vs C++ |
|---|---|---|---|---|
| 4096x4096x4096 | hk (DSL) | 1.831 | 75.0 | 1.00x |
| | C++ | 1.833 | 75.0 | 1.00x |
| | rocBLAS | 1.975 | 69.6 | 0.93x |
| 8192x8192x4096 | hk (DSL) | 7.055 | 77.9 | 1.00x |
| | C++ | 7.047 | 78.0 | 1.00x |
| | rocBLAS | 7.661 | 71.8 | 0.92x |

A tie with the handwritten kernel, which was the gate. Autotune found one
GEMM key worth changing, 4096x4096x4096, by +1.2%
([Phase 5](history/04-phase-5-autotune.md)).

### Against hipBLASLt, at the shapes Qwen3-8B runs

[The linear layers](history/08-linear-layers.md). One process, variants
interleaved, weights rotated through 3x the 96 MB Infinity Cache, every variant
checked against fp32 before it is timed (`tools/hk-bench/vllm_gemm.py`). The
kernel is the Phase 3 GEMM, unchanged. hk / `F.linear`:

| layer | N x K | M=1024 | M=4096 | M=16384 |
|---|---|---|---|---|
| qkv | 6144 x 4096 | 1.15x | 1.10x | 1.10x |
| o | 4096 x 4096 | 1.18x | 1.09x | 1.09x |
| gate_up | 24576 x 4096 | 1.11x | 1.16x | 1.23x |
| down | 4096 x 12288 | 1.17x | 1.10x | 1.17x |

All 36 layers' GEMMs for one forward step at M=4096: 838.5 ms on hipBLASLt,
744.4 on hk, 1.13x. Below 1024 rows it is a comb, not a curve, and hk loses at
M=192 on every layer and at M=256 on two:

| M | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|
| qkv | 1.38 | 0.88 | 1.13 | 1.43 | 1.20 | 1.17 |
| o | 1.15 | 0.75 | 0.95 | 1.11 | 1.24 | 1.17 |
| gate_up | 1.56 | 0.95 | 1.21 | 1.18 | 1.19 | 1.09 |
| down | 1.16 | 0.71 | 0.96 | 1.15 | 1.33 | 1.18 |

That comb is why the shipped patch routes by a measured table. Re-measured in
a fresh process where the table and the old hand-written rule disagree, in GEMM
milliseconds per step for 36 layers plus the LM head
([routing table](history/14-routing-table.md)):

| M | engine | hand rule | table | table vs rule |
|---|---|---|---|---|
| 96 | 43.57 | 43.57 | 37.57 | 1.16x |
| 128 | 53.80 | 53.80 | 40.10 | 1.34x |
| 208 | 70.14 | 70.14 | 64.26 | 1.09x |
| 256 | 70.97 | 70.97 | 64.53 | 1.10x |
| 320 | 90.02 | 90.02 | 76.05 | 1.18x |
| 448 | 119.87 | 119.87 | 101.36 | 1.18x |
| 576 | 133.74 | 133.74 | 125.04 | 1.07x |
| 784 | 180.39 | 180.39 | 167.86 | 1.07x |
| 1216 | 250.02 | 235.30 | 236.51 | 0.995x |
| 1728 | 342.98 | 329.55 | 336.65 | 0.979x |

Out of sample the table never loses to the engine. The last two rows are what
its 3% bar costs: there it loses to the hand rule by 0.5% and 2.1% of a step.

## Attention forward

The three attention tables below were measured before Phase 6 found and fixed
a biased bf16 conversion; the fix (round instead of truncate) costs +2.1% at
head_dim 128 and +3.4% at 64 on attention, and nothing on the register count
([Phase 6](history/05-phase-6-torch-ops.md)).

### Against the handwritten C++

[Phase 4](history/03-phase-4-attention.md). Same process, interleaved, both
kernels loaded at once (`hk/ops/_attn_ab.py`).

| shape | dsl ms | dsl TF | c++ ms | c++ TF | dsl/c++ |
|---|---|---|---|---|---|
| N=4096 | 7.880 | 61.0 | 7.895 | 60.9 | 1.002x |
| N=8192 | 30.528 | 63.0 | 30.527 | 63.0 | 1.000x |
| N=16384 | 120.334 | 64.0 | 120.300 | 64.0 | 1.000x |
| N=32768 | 475.309 | 64.8 | 475.132 | 64.8 | 1.000x |
| 480p/5s | 1100.228 | 64.9 | 1099.149 | 65.0 | 0.999x |
| causal 8192 | 16.628 | 57.9 | 16.754 | 57.4 | 1.008x |

A tie, ahead under causal. Against torch SDPA on the same chip that is
2.9-3.2x, and 4.0x causal. The generated kernel runs at occupancy 6 against
the handwritten kernel's 5 (224 VGPRs against 247). Autotune moved three
head_dim-128 keys by +1.1% to +2.8% ([Phase 5](history/04-phase-5-autotune.md)).

### Against aotriton, through each entry point

[Phase 6](history/05-phase-6-torch-ops.md). Same process, interleaved,
bit-identity checked before timing (`tools/hk-bench/sdpa_paths.py`); aotriton
is the backend torch's own SDPA picks on this chip. `c` is causal.

| shape | path | ms | TF | host µs | vs pybind |
|---|---|---|---|---|---|
| 1x56x4096x128 | pybind | 7.875 | 61.1 | 3.7 | 1.00x |
| | torch.ops | 7.857 | 61.2 | 3.5 | 1.00x |
| | sdpa drop-in | 7.872 | 61.1 | 9.4 | 1.00x |
| | aotriton | 22.612 | 21.3 | 11.5 | 0.35x |
| 1x56x16384x128 | pybind | 120.204 | 64.0 | 3.8 | 1.00x |
| | torch.ops | 120.449 | 63.9 | 3.1 | 1.00x |
| | sdpa drop-in | 120.466 | 63.9 | 9.5 | 1.00x |
| | aotriton | 368.805 | 20.9 | 11.7 | 0.33x |
| 1x32x8192x128 c | pybind | 9.769 | 56.3 | 3.3 | 1.00x |
| | torch.ops | 9.525 | 57.7 | 2.9 | 1.03x |
| | sdpa drop-in | 9.599 | 57.3 | 9.6 | 1.02x |
| | aotriton | 40.467 | 13.6 | 15.0 | 0.24x |

### Vision towers, on the call vLLM's ViT path makes

[Phase 6](history/05-phase-6-torch-ops.md). `(b, s, h, d)` permuted to
`(b, h, s, d)`, so never contiguous, against aotriton. "vs fp32" is the error
against an fp32 reference. head_dims 72 and 80 have no kernel and fall back to
torch, and those two rows are losses.

| tower | B,H,N,D | path | ms | vs fp32 | speedup |
|---|---|---|---|---|---|
| CLIP ViT-L/14-336 (LLaVA) | 1,16,577,64 | hk (kernel) | 0.0978 | 0.00369 | 1.69x |
| | | torch | 0.1657 | 0.00185 | 1.00x |
| InternViT-300M | 1,16,1025,64 | hk (kernel) | 0.1508 | 0.00243 | 2.49x |
| | | torch | 0.3759 | 0.00105 | 1.00x |
| SigLIP so400m | 1,16,729,72 | hk (back) | 0.1945 | 0.00128 | 0.95x |
| | | torch | 0.1850 | 0.00128 | 1.00x |
| Qwen2.5-VL ViT | 1,16,1024,80 | hk (back) | 0.3406 | 0.00113 | 0.97x |
| | | torch | 0.3295 | 0.00113 | 1.00x |

The error column is from before the bf16 rounding fix; after it, hk's error
against fp32 is `1.00x` torch's at every N in the sweep, not 1.5-2.4x.

## Norms and elementwise

### Against `torch.compile`

[Phases 1 and 2](history/01-phases-1-2.md). Against `torch.compile` on a
W7900D, 13 of 15 cases are faster and the other two (`quantize 16384x4096` and
`layernorm 16384x5120`) land within 1% behind, which is inside the noise band
and so reported as a tie. ms:

| case | hk | compile | eager |
|---|---|---|---|
| rmsnorm 4096x4096 | 0.060 | 0.093 | 0.418 |
| rmsnorm 16384x5120 | 0.486 | 0.775 | 2.823 |
| layernorm 4096x4096 | 0.061 | 0.081 | 0.063 |
| softmax 8192x16384 | 0.707 | 0.874 | 0.738 |
| silu_mul 8192x11008 | 0.761 | 0.763 | 1.218 |
| rope 128x1024x128 | 0.076 | 0.107 | 0.583 |
| quantize 16384x5120 | 0.367 | 0.369 | 6.118 |

### Against what vLLM runs under torch.compile

[Strided globals](history/12-strided-globals.md). In the mode vLLM ships, both
ops come from Inductor, fused with their neighbours. Kernel time on Qwen3-8B
shapes, one process:

| op | shape | Inductor | hk |
|---|---|---|---|
| SiLU-mul | (4096, 2*12288) | 409.5 µs | 429.7 µs (contiguous halves) |
| RMSNorm | (4096, 4096) | 38.8 µs | 55.8 µs (hk unweighted) |
| RMSNorm, q/k heads | (163840, 128) | 42.9 µs | 70.4 µs (hk unweighted) |

A loss: the library's RMSNorm loses by 44-64% before it even has a weight to
multiply, so neither patch was built. This is a different baseline from the
`torch.compile` table above, which times each op standalone.

## Decode GEMV

### Per decode step

[A reduction across workgroups](history/11-cross-workgroup-reduction.md). GEMM
milliseconds per decode step, 36 layers plus the LM head
(`tools/hk-bench/vllm_gemm.py --upstream-skinny`, >= 1.5 GB of distinct weights
per shape, every variant gated against fp32). "upstream wvSplitK" is upstream
vLLM's skinny GEMM, built standalone for gfx1100; it handles one to five
tokens.

| tokens | hipBLASLt | rocBLAS | upstream wvSplitK | hk decode | vs hipBLASLt |
|---|---|---|---|---|---|
| 1 | 39.98 | 37.02 | 19.11 | 19.66 | 2.03x |
| 2 | 39.68 | 37.57 | 19.87 | 19.69 | 2.01x |
| 4 | 39.72 | 39.02 | 20.40 | 19.77 | 2.01x |
| 8 | 39.50 | 40.86 | -- | 19.89 | 1.99x |
| 16 | 41.00 | 41.75 | -- | 20.23 | 2.03x |
| 24 | 39.74 | 43.93 | -- | 20.56 | 1.93x |
| 32 | 40.48 | 46.81 | -- | 20.84 | 1.94x |

At one token upstream's `wvSplitK` is ahead per step. Per matrix it is level
on `qkv`, 1.5-2.6% ahead on `o`, `gate_up` and the LM head, and 6.9% ahead on
`down` (770 GB/s); hk reads `qkv` at 762 GB/s, `o` 724, `gate_up` 800, `down`
720 and the LM head 819, against 676-821 GB/s that a plain read reaches
(`hbm_probe.hip`). At two and four tokens hk is ahead per step. At 33-64
tokens the kernel is register-bound: 28.6 ms against 42.2 installed, 1.47x
([the decode kernel](history/09-decode-kernel.md)).

### Split-K across workgroups: slower everywhere

[A reduction across workgroups](history/11-cross-workgroup-reduction.md).
`decode_gemm(splits=S)` is correct and slower on every shape at every split,
M=1 and M=16 alike, so the GEMV ships unsplit. µs at M=1:

| layer | splits 1 | splits 2 | splits 3 | splits 4 |
|---|---|---|---|---|
| qkv | 66.4 | 76.9 | -- | 81.8 |
| o | 46.5 | 51.4 | -- | 55.0 |
| gate_up | 252.4 | 271.2 | -- | 294.2 |
| down | 139.5 | 144.3 | 153.6 | 146.9 |
| lm_head | 1515.5 | 1663.0 | -- | 1768.4 |

## Paged decode attention

[Phase 7](history/06-phase-7-paged-decode.md). The vLLM attention backend's
decode against vLLM's Triton attention: Qwen3-8B (36 layers, 32 query heads
over 8 KV heads, head_dim 128), bf16, eager, three reps per endpoint; each
point is the slope `(T(72) - T(8)) / 64`, which cancels prefill
(`tools/hk-bench/vllm_sweep.py`). ms per decode step; "ratio" is
`+planner` time over Triton time, so above 1 is hk *slower*.

| ctx | batch | before | split | +planner | triton | ratio |
|---|---|---|---|---|---|---|
| 1024 | 1 | 54.43 | 48.60 | 49.16 | 42.01 | 1.17x |
| 1024 | 4 | 54.60 | 51.66 | 51.10 | 42.83 | 1.19x |
| 1024 | 16 | 57.65 | 57.65 | 54.46 | 46.54 | 1.17x |
| 1024 | 32 | 59.23 | 59.68 | 59.59 | 61.53 | 0.97x |
| 4096 | 1 | 82.68 | 51.10 | 50.67 | 42.97 | 1.18x |
| 4096 | 4 | 83.09 | 61.65 | 53.51 | 45.97 | 1.16x |
| 4096 | 8 | 83.86 | 71.20 | 58.06 | 49.13 | 1.18x |
| 4096 | 16 | 91.74 | 91.29 | 64.54 | 56.78 | 1.14x |

A loss everywhere except ctx 1024 batch 32. Before the KV split the worst case
was 1.92x; the split brought it to 1.19x. Standalone, the kernel reaches 56% of
the card's 864 GB/s at ctx 4096 batch 16 and 76% at ctx 16384 batch 32,
against 1.9% before. The remaining gap is flat, a constant ~7-9 ms per step,
which is the wrapper's small torch ops rather than the kernel.

### Captured whole, layer by layer

[The decode step captured whole](history/17-the-decode-step-captured-whole.md).
The wrapper is gone: a cache write and the decode's own launches, captured in
vLLM's decode graphs. Each arm walks 36 layers' caches per call, so K/V is
read cold as a decode step reads it; a one-layer A/B rereads it from the
Infinity Cache and hid the loss below (`tools/hk-bench/paged_ab.py --layers
36`, graph clock, ms per layer). "hk kernel" is hk's split decode, walk and
merge, without the cache write; "hk layer" is the backend as shipped, which
below sixteen requests runs Triton's kernel on hk's cache.

| ctx | batch | Triton layer | Triton attention | hk kernel | hk layer |
|---|---|---|---|---|---|
| 1024 | 1 | 0.0207 | 0.0174 | 0.0264 | 0.0214 |
| 1024 | 4 | 0.0427 | 0.0392 | 0.0493 | 0.0431 |
| 1024 | 8 | 0.0725 | 0.0687 | 0.0841 | 0.0730 |
| 1024 | 16 | 0.1322 | 0.1284 | 0.1264 | 0.1328 |
| 4096 | 1 | 0.0447 | 0.0412 | 0.0503 | 0.0452 |
| 4096 | 4 | 0.1184 | 0.1151 | 0.1217 | 0.1157 |
| 4096 | 8 | 0.2187 | 0.2149 | 0.2221 | 0.2128 |
| 4096 | 16 | 0.4204 | 0.4166 | 0.4016 | 0.4080 |
| 1024 | 32 | 0.5852 | 0.5832 | 0.2211 | 0.2302 |
| 4096 | 32 | 2.0337 | 2.0301 | 0.7626 | 0.7696 |

The 32-request rows walk 16 layers' caches, not 36, which is still four
times the Infinity Cache at 1024 tokens. Above 16 requests Triton's backend
leaves its 16-segment kernel for the unsplit one, and hk is 2.6x it.

hk's kernel loses up to eight requests and wins from sixteen. The loss is in
every launch: at 1024 tokens and one request the walk is 21.0 us to Triton's
13.5, the merge 8.0 to its reduction's 3.1, and vLLM's C++ cache write
through hk's view 4.1 to the Triton one's 2.6 (`tools/hk-bench/paged_kernels.py`).
The kernel reaches 83-86% of the DRAM bandwidth where there is enough of it
to stream.

## Paged prefill

[The decode step captured whole](history/17-the-decode-step-captured-whole.md).
One launch for every prefill request in a step, K/V through the page table;
against vLLM's `unified_attention` on the same batches (`tools/hk-bench/paged_ab.py
--prefill`, ms):

| batch | Triton | hk | |
|---|---|---|---|
| prompt 2048 | 4.47 | 0.87 | 5.1x |
| prompt 8192 | 82.6 | 10.5 | 7.9x |
| chunk 512 after 4096 | 4.69 | 0.79 | 5.9x |
| 4 x 1024 after 1024 | 12.6 | 2.38 | 5.3x |
| 8 x 3000 + 200 | 10.2 | 2.85 | 3.6x |
| 16 decodes + chunk 512 | 3.03 | 0.72 | 4.2x |

## End to end in vLLM

Qwen3-8B, bf16, one W7900D. The baseline (1.00x) is stock vLLM 0.16.1.dev0
choosing its own backends; arms alternate in separate processes and each takes
its minimum over rounds; every arm produced the same greedy tokens as stock.
Ratios are stock time over patched time, so above 1 is hk faster. "Graph mode"
is vLLM as it ships: torch.compile and CUDA graphs on.

### Under vLLM's defaults, with the decode captured whole

[The decode step captured whole](history/17-the-decode-step-captured-whole.md).
vLLM's defaults: CUDA graphs, prefix caching and chunked prefill on. `hk` is
the attention backend (hk's paged prefill, hk's decode from 16 requests and
Triton's on hk's cache below), `hk+gemm` with the linear-layer patch as well.
72 output tokens, best of two rounds; "step" is one decode step at the full
batch. Greedy tokens identical to stock's.

| ctx x batch | stock ms | hk ms | x | hk+gemm ms | x | step, stock ms | hk | x | hk+gemm | x |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 3206 | 3184 | 1.007x | 1730 | 1.85x | 41.32 | 41.35 | 0.999x | 21.29 | 1.94x |
| 1024 x 4 | 4033 | 3924 | 1.028x | 2384 | 1.69x | 42.31 | 42.46 | 0.996x | 22.16 | 1.91x |
| 1024 x 16 | 7628 | 7160 | 1.065x | 5150 | 1.48x | 46.32 | 46.81 | 0.989x | 26.57 | 1.74x |
| 1024 x 32 | 13059 | 11264 | 1.159x | 8768 | 1.49x | 61.64 | 51.00 | 1.209x | 31.18 | 1.98x |
| 4096 x 1 | 4448 | 3986 | 1.116x | 2458 | 1.81x | 42.21 | 42.24 | 0.999x | 22.16 | 1.91x |
| 4096 x 4 | 9195 | 7274 | 1.264x | 5286 | 1.74x | 45.29 | 44.84 | 1.010x | 24.54 | 1.85x |
| 4096 x 16 | 27923 | 20254 | 1.379x | 16738 | 1.67x | 56.20 | 56.36 | 0.997x | 36.92 | 1.52x |
| 4096 x 32 | 55372 | 37439 | 1.479x | 31789 | 1.74x | 109.38 | 70.56 | 1.550x | 50.32 | 2.17x |

### The linear-layer patch (`HK_VLLM_GEMM=1`)

[A reduction across workgroups](history/11-cross-workgroup-reduction.md). Every
unquantized linear runs on the decode GEMV (M <= 64) or the generated GEMM, and
everything else is stock. Prefix caching and chunked prefill off. Graph mode,
72 output tokens (128 at batch 32 and 64); `total` is the whole batch, `prefill`
the same batch generating one token, `decode` one step at the full batch.

| ctx x batch | total, stock ms | total, hk ms | x | prefill, stock ms | prefill, hk ms | x | decode, stock ms | decode, hk ms | x |
|---|---|---|---|---|---|---|---|---|---|
| 1024x1 | 3211.6 | 1750.2 | 1.84 | 273.9 | 243.0 | 1.13 | 41.37 | 21.23 | 1.95 |
| 1024x4 | 4029.4 | 2489.2 | 1.62 | 1025.1 | 922.7 | 1.11 | 42.30 | 22.05 | 1.92 |
| 1024x16 | 7551.4 | 5600.6 | 1.35 | 4291.5 | 3706.4 | 1.16 | 45.90 | 26.04 | 1.76 |
| 1024x32 | 16346.1 | 12836.7 | 1.27 | 8616.1 | 7403.7 | 1.16 | 61.59 | 42.56 | 1.45 |
| 1024x64 | 28382.1 | 24055.6 | 1.18 | 17341.0 | 14881.4 | 1.17 | 88.22 | 71.72 | 1.23 |
| 4096x1 | 4452.9 | 2930.6 | 1.52 | 1446.3 | 1364.0 | 1.06 | 42.35 | 22.03 | 1.92 |
| 4096x4 | 9056.0 | 7237.1 | 1.25 | 5929.3 | 5439.1 | 1.09 | 44.97 | 24.45 | 1.84 |
| 4096x16 | 27523.8 | 24837.4 | 1.11 | 23860.2 | 21779.3 | 1.10 | 56.11 | 36.38 | 1.54 |

Nothing loses, at any shape, in any phase, in graph or eager mode; graph and
eager agree to within 2.5% for both arms. Batch 1 is a latency number,
1.52-1.84x; the batches are throughput numbers, 1.10-1.35x at 16 requests and
1.18-1.27x at 32-64.

### The attention backend, and both patches together

[The linear layers](history/08-linear-layers.md), graph mode, before the decode
GEMV existed: `gemm` is the prefill GEMM patch alone, `hk` the attention
backend alone, `hk+gemm` both. 72 output tokens; decode measured as the slope
`(T(72) - T(24)) / 48`.

| | stock ms | gemm | hk | hk+gemm |
|---|---|---|---|---|
| total, 1024 ctx x 1 | 3216.2 | 1.007x | 0.864x | 0.870x |
| total, 1024 x 4 | 4035.1 | 1.024x | 0.887x | 0.904x |
| total, 1024 x 16 | 7556.9 | 1.073x | 0.965x | 1.025x |
| total, 4096 x 1 | 4452.6 | 1.016x | 0.973x | 0.993x |
| total, 4096 x 4 | 9063.7 | 1.040x | 1.149x | 1.201x |
| total, 4096 x 16 | 27517.9 | 1.044x | 1.293x | 1.364x |
| prefill, 1024 x 1 | 273.8 | 1.111x | 1.064x | 1.192x |
| prefill, 4096 x 16 | 23829.6 | 1.090x | 1.436x | 1.630x |
| decode, 1024 x 1 | 41.46 | 0.998x | 0.849x | 0.848x |
| decode, 1024 x 16 | 46.02 | 1.003x | 0.834x | 0.845x |
| decode, 4096 x 1 | 42.33 | 0.998x | 0.840x | 0.842x |
| decode, 4096 x 16 | 56.23 | 0.999x | 0.837x | 0.856x |

The attention backend loses wherever the request is mostly decode, and wins
where prefill dominates (its prefill runs on the dense kernel). The eager
version of this measurement is 1.29x on the prefill-heavy shape and 0.87x at
1024 x 1 ([Against the engine](history/07-against-the-engine.md)).

[The decode kernel](history/09-decode-kernel.md), graph mode, after it: `gemm`
is now the prefill route plus the decode GEMV for M <= 64.

| | stock ms | gemm | hk+gemm |
|---|---|---|---|
| total, 1024 ctx x 1 | 3202.4 | 1.717x | 1.382x |
| total, 1024 x 4 | 4035.5 | 1.552x | 1.309x |
| total, 1024 x 16 | 7625.0 | 1.354x | 1.297x |
| total, 4096 x 1 | 4440.6 | 1.457x | 1.436x |
| total, 4096 x 4 | 9059.0 | 1.227x | 1.464x |
| total, 4096 x 16 | 27588.0 | 1.104x | 1.476x |
| decode, 1024 x 1 | 41.24 | 1.808x | 1.402x |
| decode, 1024 x 16 | 46.99 | 1.807x | 1.417x |
| decode, 4096 x 1 | 42.09 | 1.784x | 1.371x |
| decode, 4096 x 16 | 57.18 | 1.580x | 1.281x |
| total, 1024 x 32 (128 tokens) | 16512.7 | 1.267x | 1.370x |
| total, 1024 x 64 (128 tokens) | 28509.3 | 1.162x | 1.362x |
| decode, 1024 x 32 | 62.98 | 1.447x | 1.591x |
| decode, 1024 x 64 | 89.22 | 1.195x | 1.668x |

`gemm` alone is the best patch whenever decode dominates; `hk+gemm` is best on
long-context prefill and on wide decode. At batch 1-16 the attention backend's
decode is slower than stock's, which leaves decode at 0.75-0.81x of what `gemm`
alone gets.

### Routing by the measured table, at decode batch 96 and 128

[The routing table](history/14-routing-table.md). Graph mode, 512-token
prompts, 160 output tokens, one round. `hand rule` is the patch routed by the
old hand-written rule, `table` by the measured table.

| ctx x batch | total, stock ms | hand rule | x | table | x | decode, stock ms | hand rule | x | table | x |
|---|---|---|---|---|---|---|---|---|---|---|
| 512x96 | 27349.9 | 25561.4 | 1.07 | 24523.9 | 1.12 | 97.48 | 97.98 | 0.99 | 91.48 | 1.07 |
| 512x128 | 37156.1 | 34813.6 | 1.07 | 32419.1 | 1.15 | 136.11 | 136.92 | 0.99 | 121.14 | 1.12 |

### FlyDSL's GEMM on the routes

[Chapter 30](history/30-flydsl-routes-gate-contracts.md). vLLM 0.31.0, torch
2.13.0+rocm7.2, FlyDSL 0.4.0, Qwen3-8B, bf16. Prefix caching and chunked
prefill off; three rounds, arms alternating, best of each. `hk+gemm` is the
attention backend and the linear layers routed by the table shipped before
(2026-10-14). `+FlyDSL` uses the same patches with the table re-measured with
FlyDSL installed. Its own runs equal the old table's, but for one tie at
gate_up M = 3; it adds FlyDSL rows where FlyDSL's GEMM, at the tile its
dispatcher picks, beat the table's answer by 2%. Ratios are against stock.
Every arm's greedy tokens matched stock's.

Per call (`route_tune.py`, against the table's own answer at each M, where
one of FlyDSL's tiles divides M):

| M | qkv | o | gate_up | down |
|---|---|---|---|---|
| 64-256, median | 0.84x the time | 0.85x | 0.87x | 0.97x |
| 272-1024, median | 1.19x | 1.14x | 1.33x | 1.22x |
| above 1024, median | 0.98x | 0.98x | 0.99x | 0.975x |

FlyDSL ran at 304 of the table's 740 (shape, M) points and was routed at
105. Between 272 and 1024 rows most Ms leave it only 32- or 64-row tiles,
and it loses.

Eager, 72 output tokens:

| ctx x batch | stock | hk+gemm | x | +FlyDSL | x | prefill, hk+gemm ms | +FlyDSL ms |
|---|---|---|---|---|---|---|---|
| 1024 x 1 | 2011.98 | 1686.58 | 1.193x | 1682.37 | 1.196x | 196.56 | 191.56 |
| 1024 x 4 | 2820.60 | 2392.51 | 1.179x | 2381.10 | 1.185x | 727.52 | 721.03 |
| 1024 x 16 | 7503.73 | 5020.04 | 1.495x | 5022.69 | 1.494x | 2934.70 | 2937.27 |
| 4096 x 1 | 3645.15 | 2343.31 | 1.556x | 2339.26 | 1.558x | 795.19 | 786.91 |
| 4096 x 4 | 7240.82 | 5069.63 | 1.428x | 5055.22 | 1.432x | 3183.78 | 3183.20 |
| 4096 x 16 | 24676.60 | 15732.51 | 1.569x | 15747.82 | 1.567x | 12728.54 | 12745.88 |

Decode steps at batch 1-16 are within 0.7% of each other. Graph mode (vLLM
as it ships), 512-token prompts, 160 output tokens:

| ctx x batch | total, stock ms | hk+gemm | x | +FlyDSL | x | decode, stock ms | hk+gemm | x | +FlyDSL | x |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 x 96 | 22471.29 | 17750.13 | 1.266x | 16732.92 | 1.343x | 71.72 | 58.58 | 1.224x | 51.44 | 1.394x |
| 512 x 128 | 29184.06 | 21325.82 | 1.368x | 20232.26 | 1.442x | 92.32 | 62.95 | 1.467x | 55.42 | 1.666x |

Prefill there is unchanged (8543 against 8549 ms, 11412 against 11447).
FlyDSL pays where a step's M is one of its tiles divides and one it was
measured fastest at:
- decode at 96 and 128 requests, 12% a step, where hk's 128-row GEMM
  leaves the grid underfilled and FlyDSL runs 32- and 64-row tiles;
- pure prefill steps of 1024 and 4096 tokens, 1.0-2.6% of prefill.

Below 64 requests decode stays on hk's GEMV and vLLM's skinny GEMMs. A step
mixing decodes into a prefill has an M no tile divides. FlyDSL ran in
vLLM's captured decode graphs (2,052 captured launches a round).

**Re-measured after chapter 28** ([chapter 31](history/31-remeasured.md)).
hk's GEMM gained 3% on vLLM's op and FlyDSL's dispatcher did not, so the
table shipped now routes FlyDSL at 41 points instead of 105. Per call,
against the table's own answer:

| M | qkv | o | gate_up | down |
|---|---|---|---|---|
| 64-256, median | 0.84x the time | 0.87x | 0.88x | 1.00x |
| 272-1024, median | 1.22x | 1.17x | 1.42x | 1.28x |
| above 1024, median | 1.02x | 1.005x | 1.02x | 0.997x |

The decode rows route 96 and 128 requests as before. End to end, the
shipped table against the one above, on the same tree, three rounds,
greedy tokens matching stock's:

| ctx x batch | prefill, previous table ms | shipped | total, previous ms | shipped | decode |
|---|---|---|---|---|---|
| 1024 x 1, eager | 188.65 | 190.94 (+1.2%) | 1678.40 | 1680.82 | within 0.7% |
| 1024 x 4, eager | 710.85 | 708.58 (-0.3%) | 2373.18 | 2364.15 | |
| 4096 x 1, eager | 776.68 | 773.35 (-0.4%) | 2322.72 | 2321.58 | |
| 4096 x 4, eager | 3117.40 | 3063.70 (-1.7%) | 4990.75 | 4955.49 | |
| 512 x 96, graph | 8367.64 | 8238.37 (-1.5%) | 16544.46 | 16523.99 | 51.18 / 51.66 ms |
| 512 x 128, graph | 11214.53 | 11041.54 (-1.5%) | 20016.65 | 19982.60 | 56.08 / 56.16 ms |

The 1.5-1.7% is a row the new table added, gate_up's 8192-token steps on
FlyDSL's 256x256 tile. The 1.2% loss is the rows it dropped at M = 1024,
which measured under the 2% bar per call. Against stock, decode at 96 and
128 requests is 1.413x and 1.651x (chapter 30: 1.394x and 1.666x).

### The vision tower

[Phase 6](history/05-phase-6-torch-ops.md). Inside vLLM 0.16.1.dev0 (ROCm
7.2.1) serving `OpenGVLab/InternVL2-2B`, the SDPA patch reached the kernel: 120
kernel calls, 0 fallbacks over five requests. An end-to-end time cannot resolve
it: the spread is +-25% on a ~250 ms request of which the vision tower is
~4 ms, so the tower's own number is the
[vision-tower table](#vision-towers-on-the-call-vllms-vit-path-makes) above.

## End to end in SGLang

[SGLang](history/24-sglang.md). SGLang 0.5.18 with AMD-AIM/sglang-radeon
0.1.5 and its prebuilt gfx1100 kernel, torch 2.9.1 on ROCm 7.2.1, Qwen3-8B in
bf16. Eager with Triton attention, as sglang-radeon runs RDNA3, because HIP
graph capture hangs there. `HK_SGLANG_LINEAR=1` routes every unquantized
linear by the table measured against SGLang's F.linear.

Served (`launch_server`, then `bench_serving`: random ids, 512 tokens in,
128 out, each server warmed first):

| concurrency | output tok/s, stock | hk | x | TTFT ms, stock | hk | TPOT ms, stock | hk | x |
|---|---|---|---|---|---|---|---|---|
| 1 | 22.75 | 41.27 | 1.81 | 93.5 | 51.9 | 43.53 | 24.00 | 1.81 |
| 8 | 159.52 | 270.65 | 1.70 | 534.6 | 431.7 | 46.08 | 26.14 | 1.76 |
| 32 | 456.02 | 678.08 | 1.49 | 606.2 | 489.7 | 54.41 | 34.26 | 1.59 |

In one process (`bench_one_batch`, 512 tokens in, 64 out; stock's warm runs,
since its first paid Triton's JIT):

| batch | prefill, stock ms | hk | x | decode step, stock ms | hk | x |
|---|---|---|---|---|---|---|
| 1 | 147.3-148.0 | 110.2-110.3 | 1.34 | 44.24-44.28 | 25.06-25.64 | 1.75 |
| 8 | 911.0-912.5 | 744.4-745.2 | 1.22 | 46.57-46.83 | 26.57-26.58 | 1.76 |
| 32 | 3969-4117 | 3114-3120 | 1.30 | 54.81-55.06 | 34.51-34.63 | 1.59 |

Routed by vLLM's table instead, batch 1 stays on F.linear (44.25 ms). hk's
linear and F.linear are equally far from fp64. In a greedy check, two of three
prompts matched stock over 48 tokens and the third split at a near-tie.

### MoE: OLMoE-1B-7B

[SGLang MoE](history/25-sglang-moe.md). Stock SGLang does not run it on
gfx1100, because sgl_kernel's `topk_softmax` fails. So the baseline is
`HK_SGLANG_MOE=topk`, which computes the top-k in torch and keeps SGLang's
Triton expert GEMMs. The patched arm is `HK_SGLANG_MOE=1`, with hk's GEMMs up
to 256 tokens a step. Both were measured after hk's launch got leaner, with
warm caches.

| concurrency | output tok/s, Triton GEMMs | hk | x | mean TTFT ms, Triton | hk | mean TPOT ms, Triton | hk | x |
|---|---|---|---|---|---|---|---|---|
| 1 | 61.6 | 65.5 | 1.06 | 36.4 | 34.5 | 16.07 | 15.10 | 1.06 |
| 8 | 345.6 | 457.2 | 1.32 | 142.3 | 132.2 | 22.15 | 16.55 | 1.34 |
| 32 | 836.2 | 1057.6 | 1.26 | 328.4 | 311.1 | 35.82 | 27.92 | 1.28 |

These are mean TTFTs. The medians at concurrency 8 differ by 2.4x only
because the two servers batched their prefills differently, and prefill
runs Triton in both. Greedy output was identical over SGLang's three
correctness prompts.

## After the MLIR move

[The IR as an MLIR dialect](history/15-mlir-dialect.md), Gate B: old and new
kernels in one process, interleaved for 7 rounds, the old as control.

| kernel | shape | new / old |
|---|---|---|
| GEMM | 4096³ | 1.002x (71.6 TF) |
| attention d128 | N=4096, non-causal / causal | 1.001x / 0.993x |
| decode GEMV | M=1 N=4096, M=1 N=12288, M=4 N=4096 | 1.010x / 0.999x / 1.000x |
| rmsnorm | 8192x4096, 8192x8192, 32768x1024 | 0.998x / 1.002x / 1.002x |

Every gap is smaller than that row's own round-to-round spread. Gate C, the
GEMM patch in vLLM 0.16.1 serving Qwen3-8B (eager, context 1024, 72 output
tokens), from the tree before the move and after, each against stock:

| | before | after |
|---|---|---|
| total, batch 1 / 16 | 1.820x / 1.338x | 1.824x / 1.340x |
| prefill | 1.125x / 1.160x | 1.125x / 1.160x |
| decode per step | 1.928x / 1.735x | 1.933x / 1.746x |

## Against torch, at full size, after the cross-warp fix

[Typical kernels checked at full size](history/16-validation-and-the-cross-warp-race.md):
`examples/validate.py` on the W7900D with PyTorch 2.10 (ROCm 7.2), every
check passing, hk against torch's eager call, interleaved in one process.

| case | shape | hk vs torch |
|---|---|---|
| add / exp | 8192x4096 | 0.98x / 0.92x |
| rmsnorm / layernorm / softmax | 8192x4096 | 0.96x / 0.96x / 0.88x |
| silu_mul | 8192x4096 | 1.57x |
| rope (NeoX) | 32 heads x 2048 x 128 | 9.19x |
| quantize int8 | 4096x4096 | 2.78x |
| gemm | 4096³ / M=2048 N=12288 K=4096 | 1.12x / 1.12x |
| decode GEMV | M=1, N=4096 / N=12288, K=4096 | 4.38x / 3.90x |
| attention | 1x32x4096x128, non-causal / causal | 2.76x / 3.94x |

The norms trailed torch's own fused kernels by 4-12% on the tile kernels of
the time; on chapter 27's strips the scoreboard's torch column has hk ahead
at every norm and softmax shape (softmax 8192x4096: 0.1686 against 0.1755
ms). Phase 2's "13 of 15 faster" was against `torch.compile`, a different
baseline. Before the fix the
norm and quantize rows were wrong at this size, not slow. The drain the fix
added costs nothing measurable: 1.002x for rmsnorm, 1.010x for softmax, after
against before.

## CDNA, on one MI355X

Everything above is gfx1100. This section is gfx950 (CDNA4, Instinct MI355X),
measured on one GPU of a shared 8-GPU node, inside `rocm/pytorch` (ROCm 7.2.3,
PyTorch 2.10). Contenders were interleaved in one process, best of 5, and each
check is `max|got - want| / max|want|` against an fp32 torch reference. gfx942
(MI300X) is compile-only: there is no part here. Source:
[CDNA3 and CDNA4](history/29-cdna.md).

### The first pass: correct, not yet tuned

The elementwise and row kernels below ran with tiles planned for RDNA3.

| kernel | shape | check | resources (gate) | hk vs baseline |
|---|---|---|---|---|
| `examples/01_vector_add.py` | 8192² bf16 | exact | 25 VGPR, 8 waves | 0.61x `torch.add` |
| `examples/02_rmsnorm.py` | 8192x4096 | 2.9e-3 | 28 VGPR, 8 waves | 0.58x `F.rms_norm`; `hk.ops.norm.rmsnorm` 0.72x |
| `examples/03_softmax.py` | 8192x4096 | 1.95e-3 | 37 VGPR, 8 waves | 0.51x `torch.softmax`; `hk.ops.norm.softmax` 0.76x |
| MFMA GEMM, registers only (`tools/cdna-probes/mma_reg.py`) | up to 2048x2048x1024 | ≤ 6.3e-3 | 22 VGPR + 16 AGPR, 8 waves | -- |
| GEMM through LDS (`tools/cdna-probes/gemm_lds.py`) | 4096³ | ≤ 5.7e-3, identical in 3 runs | 20 VGPR + 16 AGPR, 8 waves | 540 TFLOP/s, 0.35x hipBLASLt (1530) |

### CDNA tiles for the elementwise and row kernels

[CDNA3 and CDNA4, part 3](history/29-cdna.md#cdna-tiles-for-the-elementwise-and-row-kernels).
`hk.ops` on the MI355X after the CDNA tiling (16x32 fragments, wider blocks),
bf16, against torch 2.10's eager call, best of 5 x 50 launches, interleaved in
one process, each checked first.

| op | 64x4096 | 2048x4096 | 8192x4096 | 16384x8192 |
|---|---|---|---|---|
| add | 0.59x | 0.95x | 0.90x | 0.99x |
| exp | 0.72x | 0.78x | 0.87x | 0.76x |
| rmsnorm (no weight) | 0.94x | 0.71x | 0.93x | 1.05x |
| rms_norm (weight) | 0.73x | 0.61x | 0.95x | 0.98x |
| layernorm | 0.99x | 0.89x | 0.98x | 0.98x |
| softmax | 0.68x | 0.87x | 1.04x | 0.91x |
| fused add + RMSNorm (in place) | 1.27x | 1.06x | 1.44x | 1.25x |
| silu_mul | 1.48x | 1.82x | 1.76x | 1.28x |

Before the CDNA tiling, at 8192x4096: add 0.58x (8192x8192), rmsnorm 0.70x,
weighted RMSNorm 0.72x, layernorm 0.76x, softmax 0.80x, fused add + RMSNorm
1.19x. At 16384x8192, where the row is folded, gfx1100's plan (8 warps a
tile each) gave rmsnorm 0.89x, layernorm 0.82x and softmax 0.77x; the CDNA
plan (2 warps, 4 tiles each) is the table's. Short tensors (2048 rows) still
lose, the weighted RMSNorm most (0.61x); at 64 rows the losses are fixed
launch latency.

### GEMM: hk's own

[gfx950's GEMM in hk](history/33-cdna-gemm.md). `hk.ops.gemm.matmul` on gfx950
routes every shape to one of hk's own kernels: HipKittens' CDNA4 schedule
written in hk (`hk.ops.gemm_cdna`: copies into LDS left in flight, the
pingpong), a four-warp kernel (`hk.ops.gemm_cdna_dp`), either with two-launch
split-K, or the decode GEMV. bf16, C = A·Bᵀ, eager, every arm
checked against fp32 and interleaved in one process, best of 5 rounds:

| M = N = K | hk | HipKittens | hipBLASLt | FlyDSL `gemm_a16w16` | hk / best |
|---|---|---|---|---|---|
| 4096 | 1453-1465 | 1435-1445 | 1474-1487 | 1428-1445 | 0.98-0.99x |
| 8192 | 1566-1568 | 1571-1575 | 1613-1616 | 1582-1591 | 0.97x |

At Llama-3-8B's and Qwen3-8B's linear shapes (N x K of qkv, o, gate_up, down
and lm_head, M from 1 to 8192, 122 points), hk's routed kernel is at 0.9x
or better of the best of hipBLASLt, FlyDSL and HipKittens at 108 points.
- M ≥ 1024 runs at 0.91-1.15x.
- M = 1-4 runs at 0.89-1.85x.
- 13 of the 14 misses are at M = 32-512 (0.81-0.90): lm_head at M = 256
  (0.81), N = 24576-28672 at M = 32-64 (0.83-0.88), down_proj at M = 128
  (0.84), N = 6144-12288 at M = 256-512 (0.86-0.88). The 14th is 4 x 24576
  (0.89), within the run-to-run spread.

The full grid and what limits it are in
[the chapter](history/33-cdna-gemm.md#routed). Before this, `examples/04_gemm.py`
(64x64 blocks, no pipelining) ran at 631 TFLOP/s at 4096³, 0.41x hipBLASLt.

Where the time goes at 8192³ (hand-patched builds in one process): without
its copies the kernel runs 15% faster, and with C's store switched off it is
level with hipBLASLt. The store's cost follows its bytes, and the pattern is a
fifth of it.

### HipKittens' CDNA4 GEMM and GQA forward, as hk ops

[CDNA3 and CDNA4, part 5](history/29-cdna.md#part-5-hipkittens-cdna4-kernels-behind-hks-engine-layer);
method and every number in [design/cdna-hipkittens-ops.md](design/cdna-hipkittens-ops.md).
`hk.ops.hipkittens` builds HipKittens' kernels verbatim through hk's cache and
gate, and checks each build on the device first. Every arm is checked against
fp32 first, then interleaved in one process, best of 7 rounds.

GEMM, bf16, C = A·Bᵀ, TFLOP/s eager. HipKittens needs M and N to be multiples
of 256.

| M=N=K | HipKittens | hipBLASLt | rocBLAS | FlyDSL `gemm_a16w16` | HK / hipBLASLt |
|---|---|---|---|---|---|
| 1024 | 90 | 169 | 157 | 38 | 0.53x |
| 2048 | 420 | 667 | 655 | 302 | 0.63x |
| 4096 | 1432 | 1484 | 1502 | 1447 | 0.96x |
| 8192 | 1539 | 1612 | 1614 | 1574 | 0.95x |

At Llama-3-8B and Qwen3-8B's linear shapes, HipKittens trails hipBLASLt:
- 0.18-0.93x at M = 256-1024, where a 256x256 tile per workgroup is too few
  workgroups;
- 0.91-0.96x at M ≥ 4096.

It wins 4-27% only where hipBLASLt's heuristic picks a slow kernel. For
example, at o_proj M = 2048 it runs 886 TFLOP/s against hipBLASLt's 708. The
measured gfx950 route table (`routes_vllm_linear_gfx950.json`) sends those
points, 37 of 206 and all at M ≥ 1792, to HipKittens. hk's own GEMM (above)
now runs level with it at large M and ahead of it below M = 2048.

GQA forward, bf16, TFLOP/s:

| B, H, H_KV, N, D | causal | HipKittens | torch SDPA (aotriton flash) | FlyDSL flash | HK / SDPA | HK / FlyDSL |
|---|---|---|---|---|---|---|
| 16, 64, 8, 1024, 128 | no / yes | 966 / 663 | 570 / 253 | 1010 / 702 | 1.69x / 2.62x | 0.96x / 0.94x |
| 16, 64, 8, 4096, 128 | no / yes | 1242 / 1092 | 690 / 432 | 1254 / 1120 | 1.80x / 2.53x | 0.99x / 0.98x |
| 16, 64, 8, 8192, 128 | no / yes | 1283 / 1192 | 772 / 497 | 1277 / 1212 | 1.66x / 2.40x | 1.00x / 0.98x |
| 1, 32, 8, 4096, 128 | no / yes | 1149 / 896 | 659 / 349 | 1145 / 974 | 1.74x / 2.57x | 1.00x / 0.92x |
| 16, 64, 8, 4096, 64 | no / yes | 933 / 796 | 726 / 539 | 929 / 855 | 1.28x / 1.48x | 1.00x / 0.93x |

aiter is not installable here: PyPI's `aiter` is an unrelated package, and
`amd-aiter` is not published.
