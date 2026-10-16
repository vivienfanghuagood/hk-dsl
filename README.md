# hk-dsl — tile-level AMD GPU kernels in Python, on HipKittens

hk-dsl is a Python DSL for AMD GPU kernels at the level of
[HipKittens](https://github.com/HazyResearch/HipKittens)' tiles (register and
shared-memory tiles, matrix-core MMAs, explicit waits), together with the
patches that put those kernels into vLLM and SGLang wherever they measure
faster than the engine's own. A kernel is traced into an MLIR dialect,
checked, translated to HipKittens C++, compiled by the stock hipcc and called
from torch. The primary target is RDNA3 (Radeon PRO W7900, RX 7900); CDNA4
(Instinct MI355X) runs part of it; RDNA4 and CDNA3 compile
([where it runs](#where-it-runs)).

## Why it exists

Fast tile kernels for AMD GPUs can be written: HipKittens' C++ reaches the
hardware's peak. Three things stood between that and a faster model.

- **The serving engines do not use them.** vLLM and SGLang are tuned for
  Instinct, and on a Radeon they run slow paths. On a W7900D, SGLang's decode
  GEMMs use 38-44% of the memory bandwidth, and the kernel vLLM runs for 4-bit
  weights is 1.4-2.6x slower than hk's. hk-dsl ships its kernels with
  patches that put them into the engines, and measures each patch end to end.
  Qwen3-8B is 1.20-1.75x faster per request in vLLM 0.31 and has 1.49-1.81x
  the output throughput in SGLang; Qwen3-14B-AWQ is up to 4.43x faster.
- **The kernels fail silently.** At the tile level, the compiler can hoist an
  MMA above the wait that guards it, a register spill reorders against
  hand-placed waits, an operand in the wrong layout takes eight times the
  instructions, and on CDNA4 an asynchronous copy into shared memory races the
  barrier after it. None of these produces a diagnostic, and the first three
  cost days each to find. In hk-dsl each is either impossible to write or
  refused before the kernel runs ([the traps](#the-traps-it-removes)).
- **Writing and tuning them is slow.** HipKittens' RDNA3 flash-attention
  forward is 1108 lines of C++ with 18 `#define` knobs, and retuning it means
  editing macros and rebuilding. In hk-dsl the knobs are Python parameters
  that a tuner searches. One 59-line kernel is RMSNorm, LayerNorm and
  softmax, weighted and with a fused add, and the forms measured win or tie
  against tuned Triton and FlyDSL. A kernel becomes a torch op, then an
  engine patch, in a few lines.

hk-dsl does not replace Triton or FlyDSL. Its patches pick, shape by shape,
the kernel that measured fastest: hk's, the engine's own, HipKittens' or
FlyDSL's. In vLLM, FlyDSL's GEMM serves the decode steps at
96 and 128 requests, which it makes 12% shorter.

A complete kernel, and the call that compiles and launches it:

```python
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    t = hk.RT(hk.bf16, 16, 64)
    o[idx] = a[idx].load(t) + b[idx].load(t)

add(a, b, out)          # traces, verifies, translates, compiles (once), launches
```

## The traps it removes

None of these produces a diagnostic when it happens:

- **`s_waitcnt` carries no register dependence**, so the compiler will hoist a
  WMMA above the wait that was supposed to guard it. The translation binds the
  fragment with a volatile asm after every wait ([IR](docs/ir.md#translation)).
  CDNA4 is no different: on an MI355X, every LDS GEMM built without the bind
  returned NaN.
- **A VGPR spill is silent corruption, not slowness**, because the library
  hand-places `s_waitcnt` and scratch traffic reorders against it. `k.build()`
  refuses to return a spilling kernel ([runtime](docs/runtime.md#the-resource-gates)).
- **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
  degrades to eight times the instructions. The passes flag it while tracing,
  not at benchmark time ([IR](docs/ir.md#passes)).
- **On CDNA4, a copy from global memory into LDS is still in flight at the
  barrier after it.** It counts on `vmcnt`, which nothing waits for before the
  barrier, so the first LDS GEMM on an MI355X gave a different wrong answer on
  each run. The translation follows every such copy with its wait
  ([chapter 29](docs/history/29-cdna.md#the-race)).

`python -m hk.runtime.cocheck` runs the same resource and wait gates on any
code object, hk's or not ([runtime](docs/runtime.md#any-code-object-cocheck)).
The longer argument is in [why this exists](docs/README.md#why-this-exists).

## Where it runs

| target | GPUs | status |
|---|---|---|
| gfx1100, RDNA3 | Radeon PRO W7900 / W7900D, RX 7900 | primary: every kernel and every patch; every number below unless marked |
| gfx1201, RDNA4 | Radeon RX 9070 | every `hk.ops` kernel compiles and passes the gates; not run on hardware |
| gfx950, CDNA4 | Instinct MI355X | the elementwise, norm, fused, quantize and rows-by-index kernels run and pass; GEMM and GQA forward through HipKittens' own CDNA4 kernels, as hk ops |
| gfx942, CDNA3 | Instinct MI300X | compiles; none was available to run on |

On one MI355X, against torch: hk's fused kernels are ahead (fused add +
RMSNorm 1.06-1.44x, SiLU-mul 1.28-1.82x, RoPE 8.85x, int8 quantize 2.21x); its
plain elementwise and norm kernels are mostly behind (0.59-1.05x); its own
GEMM reaches 0.41x hipBLASLt. HipKittens' GQA forward, wrapped, runs
1.28-2.62x torch SDPA, level with FlyDSL. HipKittens' GEMM is routed to the
37 of 206 linear shapes where it beats hipBLASLt. Status:
[design/cdna.md](docs/design/cdna.md); record: [chapter 29](docs/history/29-cdna.md);
numbers: [performance](docs/performance.md#cdna-on-one-mi355x).

## Install

**From PyPI**, for CPython 3.10 (glibc 2.35+) or 3.12 (glibc 2.38+) on Linux
x86_64:

```bash
pip install hk-dsl
```

A wheel is built per Python interpreter, because it carries the MLIR bindings
for one CPython. It includes the HipKittens headers, so at run time only hipcc
is needed. The same wheels are attached to each
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases). For
another interpreter, build one with `scripts/build_wheel.sh`
([building](docs/building.md#wheels)).

The release on PyPI is 0.3.1. Since then, main has added CDNA, register strips,
the GEMM schedule, SGLang's MoE patch and plugin, and FlyDSL on the routes;
until the next release, those need a source build.

**From source.** The backend is the `third_party/HipKittens` submodule, so
clone with `--recursive`:

```bash
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
```

torch is not a dependency: tracing and compiling do not need it, launching
does. Details, memory limits and the variables the scripts take:
[getting started](docs/getting-started.md#install).

## First kernel

With `add` above, each step needs more than the one before it:

```python
print(add.trace().dump())     # the hk MLIR module, after the passes -- needs nothing
print(add.source())           # the HipKittens C++ it translates to -- needs nothing
print(add.build().report())   # hipcc + the resource gate -- no GPU
add(a, b, out)                # launch -- an AMD GPU and torch
```

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop. The full
walk-through, with tensors and a check against torch, is
[getting started](docs/getting-started.md#your-first-kernel).

## Results

On one W7900D (gfx1100). Each number is from the development record, linked.
[Performance](docs/performance.md) has the full tables, the methods, the
losses, and the older results (vLLM 0.16, the first scoreboard) that newer
ones replaced here.

**End to end**, against the stock engine:

| model, engine | patch | result | against | source |
|---|---|---|---|---|
| Qwen3-8B, vLLM 0.31 | linear layers and attention | 1.20-1.75x per request; attention alone 1.05-1.48x | stock vLLM 0.31.0 | [vLLM 0.31](docs/history/20-vllm-031.md) |
| Qwen3-8B, vLLM 0.31 | the same, with FlyDSL's GEMM on the routes | decode step 12% shorter at 96-128 requests, the request 5-6%; unchanged elsewhere | the same patches without FlyDSL | [chapter 30](docs/history/30-flydsl-routes-gate-contracts.md) |
| Qwen3-8B-AWQ / Qwen3-14B-AWQ, vLLM 0.31 | 4-bit linear layers | 1.07-3.80x / 1.16-4.43x per request | stock vLLM 0.31.0, fp16 | [4-bit weights](docs/history/22-int4.md) |
| OLMoE-1B-7B, vLLM 0.31 | MoE / MoE and attention | 1.02-1.41x / 1.17-2.22x per request | stock vLLM 0.31.0 | [MoE](docs/history/21-moe.md) |
| Qwen3-8B, SGLang 0.5.18 | linear layers (`HK_SGLANG_LINEAR=1`) | 1.49-1.81x output throughput at concurrency 1-32 | stock SGLang with sglang-radeon, eager | [SGLang](docs/history/24-sglang.md) |
| OLMoE-1B-7B, SGLang 0.5.18 | MoE (`HK_SGLANG_MOE=1`) | runs at all (stock fails in sgl_kernel's top-k); 1.06-1.32x output throughput | SGLang with the top-k in torch and its Triton MoE GEMMs | [SGLang MoE](docs/history/25-sglang-moe.md) |

**Kernels against Triton and FlyDSL**, each tuned for gfx1100, in one process
([the scoreboard](docs/history/26-the-scoreboard-retuned.md)):

| kernel | result | against | source |
|---|---|---|---|
| GEMM, bf16, Qwen3-8B's linear shapes | 87.9-91.3 TFLOP/s: a tie with FlyDSL, 0.99-1.01x; 1.09-1.11x tuned Triton; 1.28-1.39x hipBLASLt | FlyDSL's dispatcher and autotuner; Triton autotuned; hipBLASLt | [the GEMM schedule](docs/history/28-gemm-schedule.md) |
| RMSNorm, softmax, on register strips | 1.02-1.12x and 1.02x: a win or a tie at 1-8192 rows | the fastest of Triton and FlyDSL, each autotuned, and Inductor | [strips](docs/history/27-strips.md) |
| fused add + RMSNorm, on register strips | a tie, 0.83-1.11x across runs | the same | [strips](docs/history/27-strips.md) |
| attention forward, n4096, GQA 4 | 1.16x causal, 0.99x non-causal; a tie with HipKittens' C++ | torch 2.13's SDPA (3.3x faster than 2.10's, against which hk is 2.7-3.8x); FlyDSL has no attention on gfx11 | [chapter 30](docs/history/30-flydsl-routes-gate-contracts.md), [Phase 4](docs/history/03-phase-4-attention.md) |
| dropout, KV cache write, LayerNorm backward | 1.02x; 1.18-3.00x; 1.12x at 1024 wide, *slower* (0.66-0.92x) at 2048-8192 | the Triton tutorial or a Triton row kernel | [rows by index](docs/history/19-rows-by-index.md) |

**Kernels inside the engines**, against the kernel the engine runs:

| kernel | result | against | source |
|---|---|---|---|
| decode GEMV, 1-32 tokens | 1.93-2.03x per step; upstream vLLM's `wvSplitK` is ahead at one token | hipBLASLt | [cross-workgroup reduction](docs/history/11-cross-workgroup-reduction.md) |
| W4A16 decode GEMM (AWQ), fp16 | 1.40-2.61x; *slower* (0.69-0.90x) than `gptq_gemm_rdna3` at 1-4 tokens | vLLM 0.31's Triton AWQ kernel | [4-bit weights](docs/history/22-int4.md) |
| MoE GEMM, bf16 | 1.00-2.77x at 1-256 tokens; *slower* above, where the layer stays on Triton | vLLM 0.31's Triton `fused_moe` | [MoE](docs/history/21-moe.md) |
| paged prefill, cached prefixes and chunks included | 3.6-7.9x | vLLM's Triton `unified_attention` | [captured whole](docs/history/17-the-decode-step-captured-whole.md) |
| paged decode attention | *slower*: 0.83-0.95x per layer at 16-64 requests; up to 8 requests the backend runs Triton's kernel on hk's cache | vLLM 0.31's Triton attention, captured | [vLLM 0.31](docs/history/20-vllm-031.md) |

## Repository layout

```
hk/dsl/         the language: @kernel / @func, values and operators, the AST
                rewrite, control flow, hk.arch / lds / math / group / collective
hk/types.py     dtypes and the tile / vector / tensor types a kernel is written in
hk/compiler/    the MLIR context, the pass pipeline, the translation entry point
hk/_mlir/       the built bindings (hk dialect + MLIR core), not checked in
hk/target/      gfx1100 / gfx1201 / gfx942 / gfx950 facts, as data
hk/codegen/     the module boundary around a translated kernel (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                the code-object checker (cocheck.py), and the torch.ops
                registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time
hk/tuned/       the schedules that search picked, and the measured routing
                tables, as shipped data
hk/ops/         kernels written in the DSL, the SDPA drop-in (sdpa.py), and
                HipKittens' CDNA4 kernels wrapped as hk ops (hipkittens.py)
hk/integration/ patching a running vLLM or SGLang; engine.py is what every
                patch shares (op, routes, counters, CUDA-graph contract)
mlir/           the hk dialect (ODS), its passes, the C++ translation,
                hk-opt / hk-translate, lit tests
include/        hk's own headers: pyutils/hk_bind.cuh (the fast pybind entry)
                and hk/cdna.cuh (what CDNA needs beside HipKittens)
third_party/HipKittens/
                the backend: kittens.cuh and include/rdna3, rdna4, cdna3,
                cdna4; also the C++ kernels hk is measured against
scripts/        building MLIR, the dialect and the wheels; the GPU suite and the
                performance gate
tools/hk-bench/ the benchmarks, the scoreboard, the performance gate and the
                engine harnesses behind every number here
examples/       complete kernels, each checked against torch and timed
docs/           the documentation, and the development record (docs/history)
```

The `hk/` paths are under `python/`; the tests are in `tests/hk/`.

## Documentation

- [Documentation index](docs/README.md), and why this exists
- [Getting started](docs/getting-started.md) — install, a first kernel, running the tests and examples
- [The language](docs/language.md) — parameters, types, operators, memory, control flow, limits
- [The IR](docs/ir.md) — the hk dialect, its passes, the C++ translation, `hk-opt` / `hk-translate`
- [The runtime](docs/runtime.md) — the cache, the resource gates, warming, autotune, `torch.ops`
- [Integration](docs/integration.md) — vLLM and SGLang: the patches, their switches, routing, CUDA graphs
- [Performance](docs/performance.md) — every result, with its hardware, baseline and source
- [Building](docs/building.md) — MLIR, the dialect, the wheels
- [Testing](docs/testing.md) — the test tiers, CI and the performance gate
- [CDNA3/4](docs/design/cdna.md) — what runs on gfx942 / gfx950, and what does not yet
- [Development record](docs/history/README.md) — how it got here, chapter by chapter, with the negative results

## Examples

Each runs on the GPU (checks against torch, then times), and with
`--dump ir|cpp` or `--build` on a machine without one. See
[examples/README.md](examples/README.md).

| file | what it is |
|---|---|
| [`examples/01_vector_add.py`](examples/01_vector_add.py) | elementwise add, the smallest complete kernel |
| [`examples/02_rmsnorm.py`](examples/02_rmsnorm.py) | RMSNorm: runtime loop, row reduction, cross-warp combine, broadcast |
| [`examples/03_softmax.py`](examples/03_softmax.py) | row softmax: max, exp, sum, normalise |
| [`examples/04_gemm.py`](examples/04_gemm.py) | C = A·Bᵀ with LDS staging, WMMA and a derived LDS wait |
| [`examples/05_attention.py`](examples/05_attention.py) | flash-attention forward with an online softmax |
| [`examples/06_control_flow.py`](examples/06_control_flow.py) | `for` / `if` / `while` on device values, carried scalars, the generated `scf` and C++ |
| [`examples/07_dropout.py`](examples/07_dropout.py) | Triton's dropout tutorial: `hk.rand`, the same draws as `tl.rand`, the same elements kept |
| [`examples/08_embedding.py`](examples/08_embedding.py) | an embedding lookup: rows by index with `gather` |
| [`examples/09_kv_scatter.py`](examples/09_kv_scatter.py) | a KV cache write: rows to slots with `scatter`, padding slots dropped |
| [`examples/10_layernorm_backward.py`](examples/10_layernorm_backward.py) | Triton's LayerNorm backward: row and column reductions, partial sums without a lock |
| [`examples/11_vllm_integration.py`](examples/11_vllm_integration.py) | your kernel in vLLM: register, check, patch, and an end-to-end A/B in one command |
| [`examples/validate.py`](examples/validate.py) | the shipped `hk.ops` kernels at typical shapes against torch: correctness and speed |

## License

MIT; see [LICENSE](LICENSE). hk-dsl began as part of a fork of HipKittens by
HazyResearch, and it generates C++ against the HipKittens headers, which remain
HipKittens', under its MIT License. See [NOTICE](NOTICE).
