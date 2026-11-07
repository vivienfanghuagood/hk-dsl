# Toolchains: which hipcc works where

hk-dsl compiles every kernel with the hipcc on your `PATH` (or `HK_HIPCC`), so the AMD clang behind
that hipcc decides what builds. `hipcc --version` prints it (`AMD clang version 22.0.0git ...`). This
page says which clang each GPU is built and run with, how hk checks it, and what is known to break,
for the 0.8.0 to 0.9.14 wheels. Which GPUs are supported at all is
[supported hardware](supported-hardware.md).

## The matrix

"Runs" means hk's GPU tests (or the stated measurements) run on that GPU with that compiler; "builds"
means kernels are compiled and checked (spills, waits, MFMA hazards) without a GPU.

| hipcc's clang | ROCm, and where you get it | gfx1100 (W7900 / W7900D) | gfx1201 (R9700) | gfx950 (MI355X) | gfx942 (MI300X) | how hk checks it |
|---|---|---|---|---|---|---|
| 20 | 7.1, AMD's apt repository (`hip-dev`, `rocm-device-libs`) | builds | builds | builds | builds | CI's no-GPU tiers on every pull request: the codegen tier for gfx1100 (10 shards) and gfx1201 (4 shards), and the CDNA build gate for gfx950 and gfx942 ([testing](testing.md#ci)) |
| 22 | 7.2.2-7.2.4, `rocm/pytorch:rocm7.2.x` | runs | runs | runs | -- | CI's GPU tier: a W7900D (7.2.4), an R9700 and an MI355X (7.2.3) |
| 23 | 10.0, vLLM's `nightly-rocm100` image; TheRock 7.14 (the Radeon vLLM serving images and the R9700 vLLM image) | runs, in the Radeon serving images | runs, in the R9700 vLLM image | runs | -- | gfx950: CI's `gfx950-vllm` job. RDNA: the [scorecard](performance/scorecard.md) rows run in those images |
| 24 | 10.1, `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` (TheRock) | builds (the codegen tier, by hand) | not checked | runs: the quick start, example 01 (`--build` and run) and `hk.check` (by hand) | -- | not in CI |

- **gfx942 is compile-only** with every compiler: no MI300X was available, so nothing has run on one
  ([supported hardware](supported-hardware.md)).
- **Some choices follow the compiler.** On gfx1201 the causal attention tiling is picked by the clang
  major version (`hk.ops.attn._TILING_FROM_CLANG`): the tiling that was fastest under clang 22 is
  not the fastest under clang 23. And a decode GEMM schedule that builds under one compiler can spill under
  the other, so the shipped gfx1201 schedules build under both.
- **TheRock's layout** puts hipcc in `/usr/local/bin` and ROCm under `site-packages/_rocm_sdk_devel`,
  with none of its LLVM tools on `PATH`. hk finds them beside hipcc's clang
  ([the runtime](runtime.md#environment-variables)).

## Known issues

None open. If you see one of these, your hk-dsl is older than 0.9.0: upgrade (`pip install -U hk-dsl`).

| symptom | compiler, GPU | handled since |
|---|---|---|
| Every compile fails, `--build` included: `ResourceError: the build emitted no kernel-resource-usage remarks` (clang 24 prints the resource remarks in a new shape). | clang 24, every GPU | 0.9.0: both shapes are read |
| The fp16/bf16 RMSNorm and LayerNorm of `hk.ops`, its rope, examples 02 and 03, and four decode-attention schedules fail with `ResourceError` (a spill). | clang 24, gfx1100 | 0.9.0 |
| The four-tile decode GEMM spills, so tuning at load fails and `vllm_gemm` routes nothing in the Radeon vLLM serving images. | clang 23, gfx1100 | 0.9.0 |
| Some attention scores come out wrong: LLVM pads too few wait states after an MFMA whose result is read behind a branch (llvm/llvm-project#225989, not in ROCm 7.2.3 to 10.1). | clang 22-24, gfx950 | 0.9.0: hk pads what the compiler missed (`HK_MFMA_GATE=fix`, [the runtime](runtime.md#the-resource-gates)) |

If a kernel fails with a compiler that is not in the matrix, or with one that is, please report it
with what `hipcc --version` prints.
