# hk-dsl — tile-level AMD GPU kernels in Python, on HipKittens

hk-dsl is a Python DSL for AMD GPU kernels written at the level of
[HipKittens](https://github.com/HazyResearch/HipKittens)' tiles: register and
shared-memory tiles, matrix-core MMAs, explicit waits. A kernel is traced into
an MLIR dialect whose passes check it, translated to HipKittens C++, compiled
by the stock hipcc and called from torch. It ships with patches that put its
kernels into vLLM and SGLang, shape by shape, wherever they measured faster
than the engine's own. It is for people who write or tune kernels for AMD
GPUs, and for people serving models on them who want faster kernels without
writing any. Why it exists, and the silent failures it refuses to compile,
are in [the documentation](docs/README.md#why-this-exists).

## Where it runs

| target | GPUs | status |
|---|---|---|
| gfx1100, RDNA3 | Radeon PRO W7900 / W7900D, RX 7900 | primary: every kernel and every engine patch, measured end to end |
| gfx950, CDNA4 | Instinct MI355X | validated: every kernel the GPU tier runs passes; GEMM, attention and paged attention are CDNA kernels of hk's own; measured end to end in vLLM ([details](docs/design/cdna.md)) |
| gfx1201, RDNA4 | Radeon RX 9070 | compile-only: every `hk.ops` kernel compiles and passes the gates; not run on hardware |
| gfx942, CDNA3 | Instinct MI300X | compile-only: none was available to run on |

## Results at a glance

Ratios are hk's speed over the named baseline (above 1 is hk faster). Each
line links to its table in [performance](docs/performance.md), with the
method, the software stack of every measurement, and every loss.

**RDNA3, one Radeon PRO W7900D (gfx1100)**, against the stock engine, then kernel for kernel:

- **vLLM 0.31, Qwen3-8B**: attention backend and linear layers, 1.20-1.75x
  per request at 1-32 requests ([A1](docs/performance.md#vllm-031-on-gfx1100)).
- **vLLM 0.31, Qwen2.5-7B-Instruct**, in no shipped table: routes measured at
  its first start, 1.17-1.32x at 16-128 requests and 1.04-1.15x at 1-4
  ([A1](docs/performance.md#a-model-no-table-had-measured)).
- **vLLM 0.31, 4-bit and MoE**: Qwen3-14B-AWQ up to 4.43x and Qwen3-8B-AWQ up
  to 3.80x stock's Triton AWQ path; OLMoE-1B-7B 1.17-2.22x with the MoE patch
  and hk's attention ([A1](docs/performance.md#4-bit-and-moe-models)).
- **SGLang 0.5.18 with sglang-radeon, Qwen3-8B**: 1.49-1.81x the output
  throughput at concurrency 1-32 ([A1](docs/performance.md#sglang-0518-on-gfx1100)).
- The Qwen2.5-7B ratios and the later vLLM tables ran with Triton 3.8.0 over
  vLLM 0.31's triton-rocm 3.7.1; a re-validation on the matched stack is pending
  ([A1](docs/performance.md#vllm-031-on-gfx1100)).
- **Against tuned Triton, FlyDSL, hipBLASLt and torch**: the bf16 GEMM ties
  FlyDSL (0.99-1.01x), 1.09-1.11x tuned Triton, 1.28-1.39x hipBLASLt; RMSNorm
  and softmax 1.02-1.12x the fastest other; attention 1.24x causal and 1.05x
  non-causal torch 2.13's SDPA ([A2](docs/performance.md#a2-kernels-against-the-tuned-field)).
- **Against what the engine calls**: paged prefill 3.6-7.9x, the 4-bit GEMM
  1.40-2.61x, the decode GEMV about 2x hipBLASLt; paged decode 0.86-1.12x
  triton-rocm 3.7.1's kernel, a loss at most batches
  ([A3](docs/performance.md#a3-kernels-inside-the-engines), [A4](docs/performance.md#a4-losses-and-limits)).

**CDNA4, one Instinct MI355X (gfx950)**, against vLLM nightly with aiter on:

- **Qwen3-8B**: the attention backend, 1.13x per request at 16 requests and
  1.21x at 64; 1.055x vLLM's own aiter FA backend at 64
  ([B1](docs/performance.md#b1-end-to-end-in-vllm-aiter-on)).
- **Qwen3-8B-AWQ**: the 4-bit patch, 1.87x at one request and 1.35x at 16.
  hk's bf16 linear layers add about 1%, its norm and MoE kernels nothing
  ([B4](docs/performance.md#b4-losses-and-no-headroom-findings)).
- **Kernels**: the GEMM at 0.9x or better of the best of hipBLASLt, FlyDSL and
  HipKittens at all 122 Llama-3-8B/Qwen3-8B linear shapes; attention
  0.98-1.08x the best of HipKittens and FlyDSL; norms and elementwise tie or
  win 126-131 of 132 cases over three runs ([B2](docs/performance.md#b2-kernels-against-the-tuned-field));
  paged decode 0.90-1.05x aiter and 1.04-1.27x ROCM_ATTN at 8-128 requests,
  0.71-0.88x at one or two long ones ([B3](docs/performance.md#b3-kernels-inside-the-engine)).

## Install

**From PyPI**, for CPython 3.10 (glibc 2.35+) or 3.12 (glibc 2.38+) on Linux
x86_64. The release is 0.4.0; it carries the HipKittens headers, so at run
time only hipcc is needed, and torch (not a dependency) to launch:

```bash
pip install hk-dsl
```

**From source**, with the HipKittens submodule (other interpreters, memory
limits and the scripts' variables: [getting started](docs/getting-started.md#install),
[building](docs/building.md)):

```bash
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
```

## A first kernel

```python
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    t = hk.RT(hk.bf16, 16, 64)
    o[idx] = a[idx].load(t) + b[idx].load(t)

print(add.trace().dump())     # the hk MLIR module, after the passes -- needs nothing
print(add.source())           # the HipKittens C++ it translates to -- needs nothing
print(add.build().report())   # hipcc + the resource gate -- no GPU
add(a, b, out)                # launch -- an AMD GPU and torch
```

Each step needs more than the one before it. `build()` needs no GPU on
purpose: generated code that compiles but spills, the failure most likely to
bite, is refused on a laptop. The walk-through with tensors and a check
against torch: [getting started](docs/getting-started.md#your-first-kernel);
more kernels: [examples](examples/README.md).

## In vLLM and SGLang

Install hk-dsl in the engine's environment and turn patches on with
environment variables. Each patch reports what it took, and every call it
does not serve stays on the engine's kernel.

```bash
HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B                  # linear layers
HK_PATCH_VLLM=1 HK_VLLM_BACKEND=1 vllm serve Qwen/Qwen3-8B --attention-backend TRITON_ATTN
HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --attention-backend triton --disable-cuda-graph
```

`HK_VLLM_MOE`, `HK_VLLM_INT4` (AWQ) and `HK_SGLANG_MOE` turn on the others.
For a model no shipped table covers, the linear patch measures its shapes at
the first start (within 60 s), or ahead of time:

```bash
python -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct          # measure the routes once
python -m hk.integration ab --model Qwen/Qwen3-8B --arms stock,hk+gemm  # against stock: win, loss, tie or noise
```

Every patch and switch: [integration](docs/integration.md).

## Documentation

- [Documentation index](docs/README.md): why hk-dsl exists, the traps it removes, where it stops
- [Getting started](docs/getting-started.md): install, a first kernel, tests and examples
- [The language](docs/language.md), [the IR](docs/ir.md), [the runtime](docs/runtime.md)
- [Integration](docs/integration.md): the vLLM and SGLang patches, routing, CUDA graphs
- [Performance](docs/performance.md): every result, level by level, RDNA and CDNA apart
- [Building](docs/building.md) and [testing](docs/testing.md): the toolchain, wheels, CI, the performance gate
- [For agents](docs/agents.md): find a gap, add a kernel, prove it helped
- [Development record](docs/history/README.md): how it got here, chapter by chapter, negative results included

## License

MIT; see [LICENSE](LICENSE). hk-dsl began as part of a fork of HipKittens by
HazyResearch, and it generates C++ against the HipKittens headers, which remain
HipKittens', under its MIT License. See [NOTICE](NOTICE).
