# Performance

Every number on this page is copied from the [development record](history/README.md),
with a link to the chapter it comes from; the chapter has the method, the
harness in `tools/hk-bench/` and the caveats. Unless a section says otherwise
the hardware is one W7900D (gfx1100, RDNA3; 864 GB/s of HBM). The numbers are
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
- [After the MLIR move](#after-the-mlir-move)
- [Against torch, at full size, after the cross-warp fix](#against-torch-at-full-size-after-the-cross-warp-fix)

## Summary

| kernel | result | against | verdict | source |
|---|---|---|---|---|
| GEMM, bf16, Qwen3-8B's linear shapes | 82.1-85.7 TFLOP/s | the faster of Triton (autotuned) and FlyDSL; hipBLASLt | win, 1.03-1.08x; hipBLASLt 1.19-1.34x | [the scoreboard](history/18-the-scoreboard.md) |
| GEMM, bf16 | 75.0 TFLOPs at 4096³, 77.9 at 8192x8192x4096 | handwritten HipKittens C++; rocBLAS | tie with the C++ (1.00x); rocBLAS 0.93x and 0.92x; since overtaken by one LDS buffer | [Phase 3](history/02-phase-3-gemm.md) |
| RMSNorm (weighted), fused add + RMSNorm, softmax | 0.37-0.97x | the faster of Triton, FlyDSL, Inductor | loss: fixed latency at 1-32 rows, 3-8% at 8192 | [the scoreboard](history/18-the-scoreboard.md) |
| dropout, KV cache write, LayerNorm backward, embedding (`examples/07`-`10`) | dropout 1.02x; KV write 1.18-3.00x; LayerNorm backward 1.12x at 1024 wide, 0.66-0.92x wider; embedding 1.18-1.28x torch at 1-32 tokens, 0.93x Triton at 4096 | the Triton tutorial or a Triton row kernel; torch | win for dropout, the KV write and LayerNorm backward at 1024; loss where a 16-row tile cannot hold a 2048+ wide row, and for the embedding at 4096 tokens | [rows by index](history/19-rows-by-index.md) |
| GEMM on Qwen3-8B's linear layers | 1.09-1.23x at M ≥ 1024 | hipBLASLt (`F.linear`) | win at M ≥ 1024; a comb below it, with losses | [linear layers](history/08-linear-layers.md) |
| attention forward | 61.0-64.9 TFLOPs | handwritten C++; torch SDPA (aotriton) | tie with the C++, 1.008x causal; 2.9-3.2x torch, 4.0x causal | [Phase 4](history/03-phase-4-attention.md) |
| norms, elementwise, RoPE, quantize | 13 of 15 cases faster | `torch.compile` | win; the other two are a tie | [Phases 1 and 2](history/01-phases-1-2.md) |
| RMSNorm, SiLU-mul | slower | Inductor, at Qwen3-8B shapes | loss; the patches were dropped | [strided globals](history/12-strided-globals.md) |
| decode GEMV | 1.93-2.03x per step, 1-32 tokens | hipBLASLt | win; upstream vLLM's `wvSplitK` is ahead at one token | [cross-workgroup reduction](history/11-cross-workgroup-reduction.md) |
| paged decode attention | 0.66-0.97x per layer up to 8 requests, 1.02-1.04x at 16, 2.6x at 32 | vLLM's Triton attention, captured | loss below 16 requests, where the backend runs Triton's kernel on hk's cache | [captured whole](history/17-the-decode-step-captured-whole.md) |
| paged prefill | 3.6-7.9x | vLLM's Triton `unified_attention` | win | [captured whole](history/17-the-decode-step-captured-whole.md) |
| vLLM end to end, linear-layer patch | 1.10-1.84x per request | stock vLLM 0.16.1.dev0 | win at every shape, in both modes | [cross-workgroup reduction](history/11-cross-workgroup-reduction.md) |
| vLLM end to end under its defaults, attention backend / both patches | 1.007-1.48x / 1.48-1.85x per request | stock vLLM 0.16.1.dev0, CUDA graphs, prefix caching, chunked prefill | win at every shape | [captured whole](history/17-the-decode-step-captured-whole.md) |
| vLLM 0.31 end to end, Qwen3-8B, attention backend / both patches | 1.05-1.48x / 1.20-1.75x per request; 1.01-1.39x TRITON_ATTN | stock vLLM 0.31.0 (ROCM_ATTN), and the TRITON_ATTN backend hk replaces | win at every shape; the gain over TRITON_ATTN is prefill | [vLLM 0.31](history/20-vllm-031.md) |
| paged decode attention, vLLM 0.31 | 0.83-0.95x per layer at 16-64 requests | Triton's `unified_attention` on hk's cache, triton-rocm 3.7.1 | loss, kernel for kernel; yet routing 16+ requests to Triton slowed the captured step (4096 x 16: 59.0 -> 74.2 ms), so only up to 8 go there | [vLLM 0.31](history/20-vllm-031.md) |
| MoE GEMM (bf16), OLMoE and Qwen3-30B-A3B shapes | 1.07-2.77x at 1-256 tokens; 0.41-0.87x at 512-4096 | vLLM 0.31's Triton `fused_moe` kernel, its own config | win to 256 tokens, which the route takes; loss above, which stays on Triton | [MoE](history/21-moe.md) |
| vLLM 0.31 end to end, OLMoE-1B-7B, MoE patch / with hk's attention | 1.02-1.41x / 1.17-2.22x per request; decode step to 2.62x | stock vLLM 0.31.0 | win at every shape | [MoE](history/21-moe.md) |
| W4A16 decode GEMM, fp16, Qwen3-8B's layers | 2.20-2.61x at 1-16 tokens, 1.40-1.84x at 32-64 | vLLM 0.31's Triton AWQ kernel (what AWQ runs on ROCm) | win at every shape | [4-bit weights](history/22-int4.md) |
| W4A16 decode GEMM, against `gptq_gemm_rdna3` | 0.69-0.90x at 1-4 tokens; 1.13-2.25x from 8 | vLLM's HIP W4A16 kernel for gfx1100 | loss to 4 tokens, a skinny GEMV without WMMA's padding; win above | [4-bit weights](history/22-int4.md) |
| vLLM 0.31 end to end, Qwen3-8B-AWQ, 4-bit patch | 1.07-3.80x per request; decode step 1.57-4.46x | stock vLLM 0.31.0, fp16, Triton AWQ | win at every shape | [4-bit weights](history/22-int4.md) |
| vLLM 0.31 end to end, Qwen3-14B-AWQ, 4-bit patch | 1.16-4.43x per request; decode step 2.87-5.52x | stock vLLM 0.31.0, fp16, Triton AWQ | win at every shape | [4-bit weights](history/22-int4.md) |
| shipped ops at 7-8B shapes (`examples/validate.py`) | GEMM 1.12x, decode GEMV 3.90-4.38x, attention 2.76x / 3.94x causal, quantize 2.78x; norms 0.88-0.96x | torch eager, PyTorch 2.10 | win, except norms and elementwise, 2-12% behind torch's own fused kernels | [validation](history/16-validation-and-the-cross-warp-race.md) |

## Against Triton and FlyDSL

[The scoreboard](history/18-the-scoreboard.md). `tools/hk-bench/scoreboard.py`:
every way to compute each op, in one process, each checked against fp32
before it is timed, GPU time from the profiler's kernel records by kernel
name. Triton is the tutorial kernel with its own autotune (`triton_ref.py`);
FlyDSL is 0.4.0.dev970's own kernel (its flash attention does not run on
gfx11, which has no global-to-LDS load). GPU ms; the last column is hk against
the fastest other arm.

| case | hk | torch | Inductor | Triton | FlyDSL | hk vs best |
|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.612 | 1.999 | -- | 1.687 | 1.743 | 1.05x |
| gemm 2048x6144x4096 | 1.202 | 1.613 | -- | 1.300 | 1.306 | 1.08x |
| gemm 2048x4096x4096 | 0.837 | 1.043 | -- | 0.903 | 0.930 | 1.08x |
| gemm 2048x24576x4096 | 4.847 | 5.944 | -- | 5.090 | 5.075 | 1.05x |
| gemm 2048x4096x12288 | 2.490 | 2.958 | -- | 2.645 | 2.571 | 1.03x |
| attention n4096 causal, GQA 4 | 2.994 | 10.924 | -- | -- | -- | 3.65x |
| attention n4096, GQA 4 | 4.823 | 13.240 | -- | -- | -- | 2.75x |
| rms_norm 1x4096 | 0.0057 | 0.0044 | 0.0025 | 0.0026 | 0.0023 | 0.41x |
| rms_norm 32x4096 | 0.0048 | 0.0044 | 0.0026 | 0.0026 | 0.0023 | 0.47x |
| rms_norm 2048x4096 | 0.0274 | 0.0264 | 0.0196 | 0.0193 | 0.0191 | 0.70x |
| rms_norm 8192x4096 | 0.1949 | 0.1812 | 0.1810 | 0.1807 | 0.1791 | 0.92x |
| fused add + rms_norm 1x4096 | 0.0074 | 0.0080 | 0.0027 | 0.0031 | 0.0032 | 0.37x |
| fused add + rms_norm 32x4096 | 0.0051 | 0.0063 | 0.0026 | 0.0026 | 0.0028 | 0.51x |
| fused add + rms_norm 2048x4096 | 0.0505 | 0.0537 | 0.0436 | 0.0438 | 0.0394 | 0.78x |
| fused add + rms_norm 8192x4096 | 0.3560 | 0.4621 | 0.3664 | 0.3460 | 0.3831 | 0.97x |
| softmax 2048x4096 | 0.0305 | 0.0313 | 0.0582 | 0.0261 | 0.0192 | 0.63x |
| softmax 8192x4096 | 0.1893 | 0.1775 | 0.2271 | 0.1781 | 0.1757 | 0.93x |

The losses, and why:
- **1-32 rows** is fixed latency: hk's 16x64 fp32 tile per warp is a
  longer instruction stream than Triton's one-dimensional block for the
  same 8 KB.
- **Mid-size softmax** pays two LDS round trips with barriers per
  reduction, twice.
- **8192 rows** is bandwidth, 3-8% behind.

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
The kernel reaches 83-86% of HBM where there is enough of it to stream.

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

### The vision tower

[Phase 6](history/05-phase-6-torch-ops.md). Inside vLLM 0.16.1.dev0 (ROCm
7.2.1) serving `OpenGVLab/InternVL2-2B`, the SDPA patch reached the kernel: 120
kernel calls, 0 fallbacks over five requests. An end-to-end time cannot resolve
it: the spread is +-25% on a ~250 ms request of which the vision tower is
~4 ms, so the tower's own number is the
[vision-tower table](#vision-towers-on-the-call-vllms-vit-path-makes) above.

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

The norms trail torch's own fused kernels by 4-12%; Phase 2's "13 of 15
faster" was against `torch.compile`, a different baseline. Before the fix the
norm and quantize rows were wrong at this size, not slow. The drain the fix
added costs nothing measurable: 1.002x for rmsnorm, 1.010x for softmax, after
against before.
