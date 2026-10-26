# gfx1201 in vLLM: a nightly toolchain, two compilers that spill in opposite places, and the linear layers on hk

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Chapters [47](47-gfx1201-on-hardware.md)-[49](49-gfx1201-attention-fp8-and-flydsl-from-source.md)
ran hk on gfx1201 in rocm/pytorch's ROCm 7.2.3 image. A Radeon AI PRO R9700 user
serving a model runs something else: `ghcr.io/inferstation/vllm-rocm-r9700-main`,
the vLLM build for that card. This chapter is hk inside it, and the first engine
route validated on gfx1201.

## The image

| | rocm/pytorch (chapters 47-49) | the R9700 vLLM image |
|---|---|---|
| base | Ubuntu 24.04, glibc 2.39 | Debian 12, glibc 2.36 |
| ROCm | 7.2.3 | 7.14 nightly (TheRock, as pip packages) |
| compiler | hipcc 7.2.3 | AMD clang 23.0.0git |
| torch | 2.10.0 | 2.13.0a0+rocm7.14 |
| vLLM | -- | 0.1.dev1+ga50259bb3 |

Four things broke, none of them in a kernel:

- **hk's MLIR bindings would not import**: built under glibc 2.39, they want
  `GLIBC_2.38`. Rebuilt in the image with its own cmake and gcc 12, they do.
  (A release wheel is built on glibc 2.35 since #66, and takes the image as
  it is.)
- **Every op registered as a torch op failed to compile.** TheRock's torch is
  hipified the newer way: `c10/hip/HIPGuard.h` declares `c10::cuda::CUDAGuard`
  and no `c10::hip::HIPGuard`. The generated shim now picks the spelling by
  `C10_HIP_API`, which only the older headers define (#79).
- **The decode GEMM's four-tile kernel spilled**, and the spill gate refused
  it rather than run it wrong. The two compilers disagree about where: clang 23
  spills it at combine depth 4 (264 VGPRs, 21 spilled), hipcc 7.2.3 at depth 0
  (9 spilled), and depth 2 builds under both (`tools/rdna4-probes/dec_t4.py`).
  `decode_gemm.SCHEDULES_AT["gfx1201"]` holds depth 2. Any schedule hk ships for
  gfx1201 has to build under both compilers, not under the one CI has.
- **vLLM did not see the GPU** when the run's `PYTHONPATH` replaced the image's:
  the image puts `amd_smi` on it, and vLLM's ROCm platform detection needs it.
  Newer vLLM also registers its cache ops from `_C_stable_libtorch`, which that
  platform module imports, so a test calling `_custom_ops` bare now imports it.

With those, hk's whole GPU tier passes in the image on the R9700, every group.

## The linear layers

`vllm_gemm` routes vLLM's unquantized linear layers by a measured table: per
(N, K) and M, the engine's GEMM, hk's, or hk's decode GEMV, whichever won by
3%. gfx1201 had no table, so the route did nothing (the first run patched
nothing and said so). `tools/hk-bench/route_tune.py`, in the image, measured
Qwen3.5-9B's and Qwen3-1.7B's layers against its `rocm_unquantized_gemm`: at
4096 wide hk's GEMM takes gate_up from 400 rows, down from 464, qkv from 848;
the decode GEMV takes some M under 32; the 248K-vocabulary lm_head stays the
engine's. At 2048 wide (1.7B) the two trade the lead M by M.

Gate C, `python -m hk.integration ab`, stock vLLM against the route, three
rounds alternating, vLLM as it ships (CUDA graphs and torch.compile,
`HK_EAGER=0`):

| | Qwen3-1.7B | Qwen3.5-9B |
|---|---|---|
| greedy tokens, first request, 4 shapes | 72/72 identical | 72/72 identical |
| routing counts, round to round | equal | equal |
| prefill, ctx 1024/4096 x batch 1/16 | tie | **1.12-1.16x** |
| total | tie, 1.02x at 4096x16 | 1.01-1.10x |
| decode | tie, 1.06x at 4096x16 | tie, 1.05x at 1024x16 |

Qwen3.5-9B is a hybrid: three of every four layers are linear attention, whose
projections (`in_proj_qkvz`, 12288x4096) the table did not have; the route
measured them at load on the first round (50 s, kept in this machine's
records), and the counts match from round to round regardless. Eager mode told
a different story for the small model: decode at batch 1 0.77x, 3.5 ms a step
more over Qwen3-1.7B's 112 linear calls -- the dispatch's cost, presumably, not
measured apart; under graphs it is gone.

On that evidence `vllm_gemm` is on by default on gfx1201 too, with the table
in `hk/tuned/routes_vllm_linear_gfx1201.json`. It ships the four (N, K) that
Qwen3-8B shares with Qwen3.5-9B -- gfx1100's table already holds them -- and
not the rest: a shape no shipped table held is one the T4 tuning goal's
held-out set may draw (`bench/tune/shapes.py` excludes every shipped (N, K)),
and adding one would redraw gfx1100's and gfx950's sets. The rest are measured
at load, as the 9B's linear-attention projections were. Gate C again with
exactly that -- the shipped table and an empty records directory, a fresh
install: the first two engines measured the missing shapes at load (60 s, then
7), tokens identical, counts equal round to round, prefill 1.11-1.14x, total
1.01-1.10x.
