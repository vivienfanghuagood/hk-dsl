# hk-dsl — HipKittens kernels in Python, over an MLIR dialect

Write Radeon kernels in Python, get HipKittens C++, get a `.so`, call it from
torch. The point is not convenience: it is that three classes of *silently
wrong* kernel on RDNA3 stop being expressible.

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

## The three traps

Each of these cost days to find by hand, and none of them produces a
diagnostic:

- **`s_waitcnt` carries no register dependence**, so the compiler will hoist a
  WMMA above the wait that was supposed to guard it. The translation binds the
  fragment with a volatile asm after every wait ([IR](docs/ir.md#translation)).
- **A VGPR spill is silent corruption, not slowness**, because the library
  hand-places `s_waitcnt` and scratch traffic reorders against it. `k.build()`
  refuses to return a spilling kernel ([runtime](docs/runtime.md#the-resource-gates)).
- **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
  degrades to eight times the instructions. The passes flag it while tracing,
  not at benchmark time ([IR](docs/ir.md#passes)).

The whole argument is in [why this exists](docs/README.md#why-this-exists).

## Install

**From a wheel.** One per Python interpreter, because it carries the MLIR
bindings for one CPython; it includes the HipKittens headers, so at run time
only hipcc is needed. Build it with `scripts/build_wheel.sh`
([building](docs/building.md#wheels)), then:

```bash
pip install dist/hk_dsl-0.2.0-cp310-cp310-linux_x86_64.whl
```

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
add(a, b, out)                # launch -- a Radeon and torch
```

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop. The full
walk-through, with tensors and a check against torch, is
[getting started](docs/getting-started.md#your-first-kernel).

## Results

On one W7900D (gfx1100). Each number is from the development record, linked;
[performance](docs/performance.md) has the full tables, the methods, and the
losses.

| kernel | result | against | source |
|---|---|---|---|
| GEMM, bf16, Qwen3-8B's linear shapes | 82.1–85.7 TFLOP/s: 1.03–1.08x | the faster of Triton (autotuned) and FlyDSL; hipBLASLt 1.19–1.34x | [the scoreboard](docs/history/18-the-scoreboard.md) |
| RMSNorm, fused add + RMSNorm, softmax | 0.37–0.97x, *slower* | the faster of Triton, FlyDSL and Inductor | [the scoreboard](docs/history/18-the-scoreboard.md) |
| dropout, KV cache write, LayerNorm backward | 1.02x; 1.18–3.00x; 1.12x at 1024 wide, 0.66–0.92x (*slower*) at 2048–8192 | the Triton tutorial or a Triton row kernel | [rows by index](docs/history/19-rows-by-index.md) |
| attention forward | 61.0–64.9 TFLOPs, 0.999–1.002x (1.008x causal); 2.9–3.2x torch SDPA, 4.0x causal | the handwritten C++; torch SDPA | [Phase 4](docs/history/03-phase-4-attention.md) |
| norms, softmax, RoPE, SiLU-mul, quantize | 13 of 15 cases faster, the other two a tie | `torch.compile` | [Phases 1 and 2](docs/history/01-phases-1-2.md) |
| decode GEMV, 1–32 tokens | 1.93–2.03x per step | hipBLASLt | [cross-workgroup reduction](docs/history/11-cross-workgroup-reduction.md) |
| paged decode attention | 1.02–1.04x per layer at 16 requests, 2.6x at 32; *slower* (0.66–0.97x) up to 8, where the backend runs Triton's kernel instead | vLLM's Triton attention, captured | [captured whole](docs/history/17-the-decode-step-captured-whole.md) |
| paged prefill, cached prefixes and chunks included | 3.6–7.9x | vLLM's Triton `unified_attention` | [captured whole](docs/history/17-the-decode-step-captured-whole.md) |
| Qwen3-8B in vLLM, linear-layer patch | 1.52–1.84x at batch 1, 1.10–1.35x at batch 16 | stock vLLM 0.16.1.dev0 | [cross-workgroup reduction](docs/history/11-cross-workgroup-reduction.md) |
| Qwen3-8B in vLLM under its defaults, both patches | 1.48–1.85x per request at every shape; attention alone 1.007–1.48x | stock vLLM 0.16.1.dev0, CUDA graphs, prefix caching, chunked prefill | [captured whole](docs/history/17-the-decode-step-captured-whole.md) |
| shipped ops at 7-8B shapes ([`validate.py`](examples/validate.py)) | GEMM 1.12x, decode GEMV 3.90–4.38x, attention 2.76x (3.94x causal), quantize 2.78x; norms 0.88–0.96x | torch eager, PyTorch 2.10 | [validation](docs/history/16-validation-and-the-cross-warp-race.md) |

## Repository layout

```
hk/dsl/         the language: @kernel / @func, values and operators, the AST
                rewrite, control flow, hk.arch / lds / math / group / collective
hk/types.py     dtypes and the tile / vector / tensor types a kernel is written in
hk/compiler/    the MLIR context, the pass pipeline, the translation entry point
hk/_mlir/       the built bindings (hk dialect + MLIR core), not checked in
hk/target/      gfx1100 / gfx1201 facts, as data
hk/codegen/     the module boundary around a translated kernel (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                and the torch.ops registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time
hk/tuned/       the schedules that search picked, and the measured routing
                tables, as shipped data
hk/ops/         kernels written in the DSL, plus the SDPA drop-in (sdpa.py)
hk/integration/ patching a running vLLM or SGLang; engine.py is what every
                patch shares (op, routes, counters, CUDA-graph contract)
mlir/           the hk dialect (ODS), its passes, the C++ translation,
                hk-opt / hk-translate, lit tests
include/        hk's own headers (pyutils/hk_bind.cuh, the fast pybind entry)
third_party/HipKittens/
                the backend: kittens.cuh and include/rdna3, include/rdna4;
                also the C++ kernels hk is measured against (kernels/rdna3)
scripts/        building MLIR, the dialect and the wheels
tools/hk-bench/ the benchmarks and engine harnesses behind every number here
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
- [Testing](docs/testing.md) — the four tiers and the gates
- [Development record](docs/history/README.md) — how it got here, phase by phase, with the negative results

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
| [`examples/validate.py`](examples/validate.py) | the shipped `hk.ops` kernels at typical shapes against torch: correctness and speed |

## License

MIT; see [LICENSE](LICENSE). hk-dsl began as part of a fork of HipKittens by
HazyResearch, and it generates C++ against the HipKittens headers, which remain
HipKittens', under its MIT License. See [NOTICE](NOTICE).
