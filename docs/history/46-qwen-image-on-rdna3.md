# Qwen-Image 2.1 on a W7900D: hk's attention and GEMM inside SGLang-Diffusion, 1.50x a 1024^2 image

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The end-to-end goal ([design/e2e-goal.md](../design/e2e-goal.md), E1) starts
with the model that most needs it: a text-to-image DiT on RDNA3, where
SGLang-Diffusion has exactly one attention backend that runs (`torch_sdpa`:
aiter is CDNA's, FlashAttention 3 Hopper's). This chapter measures Qwen-Image
2.1 there, puts hk's attention and then hk's GEMM behind SGLang-Diffusion
without touching SGLang's tree, and checks the image.

The answer: a 1024 x 1024, 30-step image takes **42.2 s stock and 28.2 s with
both patches, 1.50x** (attention alone 34.8 s, 1.21x; both with the
copies still in, 29.1 s), on one W7900D. hk's
attention runs a DiT call in 4.4 ms where torch's (AOTriton's `attn_fwd`)
takes 13.6 ms, and hk's GEMM is 1.27-1.28x hipBLASLt at all four DiT
projection shapes. The image is visually the same, and closer to an fp32-
softmax reference than the stock image is.

## The engine, and why it is SGLang's main

pod `hfang-hk-vllm`, W7900D GPU 0, ROCm 7.2.1, torch 2.9.1, Qwen-Image 2.1
from the Hub (33 GB: transformer 14.2, text encoder 17.5, VAE 1.4).

- **SGLang 0.5.18 has no native Qwen-Image 2.1 pipeline**: it logs "no
  native sglang support" and falls back to diffusers, at **78.9 / 80.3 /
  80.4 s** an image. That is not the path anyone serves.
- **SGLang main (79701a2, 2026-10-24) has the native pipeline, and it does
  not run on ROCm.** The DiT asks `current_platform.is_cuda()`, which is
  true on ROCm, and then calls fused Triton/CUDA kernels (`qknorm_complex_rope`
  among them) whose validators require `torch.version.hip is None`; every
  request fails with "invalid input for qknorm_complex_rope". A post-import
  hook that sets the module's `_is_cuda` to False on HIP
  (`bench/e2e/qwen_image/rocm_fix_sitecustomize.py`) sends it down the torch
  paths. Every arm below has it; it is a fix SGLang needs upstream, not part
  of hk's result.
- Stock, so fixed: **42.20 / 42.24 / 42.26 s** (warm-up 44.9 s), 26.7 GB peak.

## Where the time goes

rocprofv3 `--kernel-trace --stats` over a warm-up and one image (two
images), kernels sorted into families (`bench/e2e/qwen_image/families.py`):

| family | stock | hk attention |
|---|---|---|
| GEMM (one hipBLASLt kernel, `MT96x96x32`, 10,554 calls) | 49.3 s (59%) | 49.4 s (73%) |
| attention | 26.0 s (31%), AOTriton `attn_fwd`, 13.6 ms a call | 8.4 s (12%), `attn_fwd_d128`, 4.4 ms a call |
| copies, casts, cat | 2.6 s | 4.8 s |
| elementwise (modulation, RMSNorm's pieces, SiLU-mul, residuals) | 4.0 s | 4.1 s |
| everything else | 1.5 s | 1.5 s |
| total | 83.5 s | 68.1 s |

The DiT is 32 layers of 32 heads x 128 over 4,096 image tokens (patch 1,
64-channel latents), plus a text prefix of a few dozen tokens attended once
a request. Its attention is one call a layer a step: B=1, H=32, N_q=4,096,
N_kv=4,096 + the prefix, non-causal -- the shape hk's gfx1100 kernel was
written for. Its projections are four GEMMs at M=4,096: QKV and gate/up
(4096x12288x4096), out (4096x4096x4096) and down (4096x4096x12288).

## The attention patch

`hk.integration.sglang_diffusion` (new): SGLang-Diffusion's `SDPAImpl.forward`
calls `torch.nn.functional.scaled_dot_product_attention` at call time, so
the patch is `hk.ops.sdpa.patch()` -- the drop-in that takes bf16, head_dim
64/128, unmasked calls and gives torch everything else. The hard part is
reaching the process that runs the DiT: SGLang-Diffusion spawns workers and,
unlike SGLang's LLM engine, loads no plugins there. `python -m
hk.integration sglang-diffusion install` writes a one-line `.pth` and a boot
module into the environment's site-packages; with `HK_SGLD` set, the boot
hooks the import of the SDPA backend module and patches in whichever process
imports it. Without it, the boot is one environment lookup.

The first run found a bug in the drop-in itself: after `patch()`, its
fallback called `F.scaled_dot_product_attention` -- which was now the
drop-in -- and the first masked call (the text prefix's causal attention)
recursed until Python stopped it. The fallback now calls the function
`patch()` saved, and a GPU test covers a patched call that falls back.

Of the 4,008 attention calls in four images, 3,840 took hk's kernel (one per
layer per step); 168 went to torch: the masked text prefix, prefix segments
shorter than the gfx1100 kernel's 192-row Q tile, and the VAE's single-head
attention at head_dim 1,152. **34.69 / 34.83 / 34.90 s, 1.21x.** The copies
grew by 2.2 s over two images: hk's kernel takes (B, H, N, D) contiguous and
SGLang hands it transposed views, so q, k, v and the output are copied.

## The GEMM patch

The DiT's projections all go through `UnquantizedLinearMethod.apply` in
`sglang.multimodal_gen.runtime.layers.linear` (F.linear on ROCm without
aiter). Interleaved on GPU 0 (`bench/e2e/qwen_image/gemm_ab.py`, `GemmBench`,
3 sets of 7 rounds, weights rotated past the Infinity Cache):

| M x N x K | hipBLASLt | hk.ops.matmul | |
|---|---|---|---|
| 4096 x 12288 x 4096 | 5,778 us (71 TF/s) | 4,567 us (90) | 1.265x |
| 4096 x 4096 x 4096 | 2,002 us (69) | 1,561 us (88) | 1.282x |
| 4096 x 4096 x 12288 | 5,971 us (69) | 4,671 us (88) | 1.278x |
| 4096 x 24576 x 4096 | 12,319 us (67) | 9,666 us (85) | 1.274x |

So `apply_linear()` routes a projection to `hk.ops.matmul` when it is bf16,
without bias and at least 1,024 rows (`MIN_M`; nothing below was measured),
and leaves the rest to the engine. 26,883 calls took hk's GEMM, 896 the
engine's (the text prefix and the timestep/modulation projections).
**29.06 / 29.16 / 29.25 s, 1.45x stock.**

## Attention in place

SGLang hands the drop-in `.transpose(1, 2)` views of (B, N, H, D) memory,
and the kernel took contiguous (B, H, N, D): q, k, v and the output were
copied every call (+2.2 s over two images). `hk.ops.attn.attn_kernel` gains
`bnhd=True`: the operands are passed as (B, 1, N, H x D) and a head's rows are
read at a column offset of head x head_dim. Five address expressions and the
grid change; the loops are the same instructions (227 VGPRs, no scratch,
occupancy 6, as before), and the default kernels' C++ is byte-identical to
what it was. The drop-in takes that path whenever every operand is such a
view and hands the output back as one. **28.12 / 28.23 / 28.31 s** (from
29.1), and the image is bit-identical to the copying path's.

## Is the GEMM at its ceiling?

`python -m hk.tune` on the four DiT shapes, 120 GPU-s each: no winner. The
shipped route is the best of 215 legal schedules at every shape (the closest
challenger, `rdna:ns1` at 4096x24576x4096, a tie at +3.1% against a 3.6%
bar), at 1.255-1.270x hipBLASLt in the tuner's own interleaved rounds. What
is left for the GEMM is the kernel, not the schedule: 85-90 TF/s of the
W7900D's 122.6.

## The image

Every arm is deterministic: two images of the same arm are identical to the
byte. Against stock the patched image differs at texture level -- MAE 0.97
(attention) and 1.10 (both), PSNR 42.1 and 40.0 dB, 0.7% and 1.3% of pixels
off by more than 8 -- and side by side the signage, the cat's fur and the
book spines are the same. To say which is closer to the math, a fourth arm
ran stock with torch's math SDPA (fp32 softmax, 127.2 s an image):

| against the math-SDPA image | MAE | PSNR | pixels > 8 |
|---|---|---|---|
| stock (AOTriton flash) | 1.83 | 35.4 dB | 3.7% |
| hk attention | 1.61 | 36.7 dB | 2.9% |
| hk attention + GEMM | 1.37 | 38.4 dB | 1.9% |

The patched images sit inside the stock path's own distance from the
reference, and closer to it. (The in-place attention's image is the same
bytes as the copying path's.)

## What is left

At 28.2 s, by the stock profile's proportions and the speedups: GEMM is
about two thirds of an image (hk's 85-90 TF/s against the W7900D's 122.6 TF/s
bf16 peak, at the best schedule hk.tune finds), then attention (4.4 ms a
call, ~4.2 s an image) and ~2 s of elementwise work that SGLang fuses on CUDA
(RMSNorm + complex RoPE, LayerNorm + modulation, SiLU-mul) and runs as
separate torch kernels here. The next steps that follow: hk's rowwise
templates for the norm/RoPE/modulation fusions the CUDA path has, and the
same patches on MiniMax-H3 (E2), whose DiT is twice the size and streamed
from host RAM.

## How it was run

```bash
# pod hfang-hk-vllm: /workspace/venv-sgld = /workspace/venv-sgl (torch 2.9.1, ROCm 7.2.1,
# sglang-radeon 0.1.5) with SGLang main 79701a2 installed editable, cache-dit 1.2.3,
# diffusers 0.42.0, imageio-ffmpeg, addict
python -m hk.integration sglang-diffusion install          # once, with hk on PYTHONPATH
P=<Qwen/Qwen-Image-2.1 snapshot>
env HIP_VISIBLE_DEVICES=0 PYTHONPATH=<rocm fix>:<hk>/python [HK_SGLD=1] \
    python bench/e2e/qwen_image/qi_bench.py --model $P --out OUT --runs 3
# 1024x1024, 30 steps, seed 42, MIOPEN_FIND_MODE=FAST, SGLANG_USE_AITER=0;
# a warm-up request, then three timed ones; wall time of DiffGenerator.generate
```

Results: `bench/e2e/qwen_image/*.json` (base-sdpa: SGLang 0.5.18's diffusers
fallback; native-sdpa: stock; native-hk: attention only, at 82f2472, when
`HK_SGLD=1` meant the attention patch alone; native-hk2: both, attention copying; native-hk3: both, attention in place; native-math:
torch's math SDPA).
