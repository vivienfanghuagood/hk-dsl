# Getting started

From nothing to a kernel that is right and at least as fast as
`torch.compile`, then to one you wrote yourself.

- [Install](#install)
- [Your first kernel: a template](#your-first-kernel-a-template)
- [Check it](#check-it)
- [Writing the kernel yourself](#writing-the-kernel-yourself)
- [The four entry points](#the-four-entry-points)
- [Timing a kernel, and CUDA graphs](#timing-a-kernel-and-cuda-graphs)
- [Running the tests](#running-the-tests)
- [Running the examples](#running-the-examples)
- [Your model in vLLM](#your-model-in-vllm)
- [Where next](#where-next)

## Install

You need Linux x86_64 and ROCm's hipcc to compile; a Radeon (gfx1100, RDNA3)
or an Instinct MI355X (gfx950, CDNA4; see [CDNA](design/cdna.md) for what runs
there) and torch to launch. torch is not a dependency of the package — install the
ROCm build you already use.

**From PyPI**, for CPython 3.10 (glibc 2.35+) or 3.12 (glibc 2.38+):

```bash
pip install hk-dsl
```

A wheel is built per Python interpreter, because it carries the MLIR bindings
for one CPython. The same wheels are attached to each
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases). For
another interpreter, build one ([building](building.md#wheels)).

It carries the HipKittens headers, the bindings and the shipped tuning
records, so nothing else is needed at run time but hipcc.

The release on PyPI is 0.5.0. It has the templates (`hk.templates.elementwise`,
`rowwise` and `gemm`), `hk.check`, the API reference in [docs/api](api/README.md),
gfx950's register strips and the same `hk.ops` arguments on both GPUs, which
0.4.0 lacked.

**From source.** The backend is the `third_party/HipKittens` submodule, so
clone with `--recursive` (or run `git submodule update --init`). The dialect
needs MLIR, built once:

```bash
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
scripts/build_wheel.sh      # optional: a wheel for this interpreter, headers included
```

Keep `JOBS` × ~2 GB under the memory limit (a 48 Gi pod is
`JOBS=6 LINK_JOBS=1`). Then either install the wheel, or use the checkout in
place by putting `python/` on `PYTHONPATH` (or `pip install -e .`). The
scripts' variables are in [building](building.md#mlir-and-the-dialect).

## Your first kernel: a template

Most kernels a model needs are the math of one element (an activation, a
gated product, a quantizer) or of one row (a norm, a softmax, a loss), or a
GEMM with something done to its output. For those, write only that math and
let a template build the kernel:

```python
# first.py
import torch
import torch.nn.functional as F
import hk
from hk import templates as T

gelu_and_mul = T.elementwise(lambda g, u: hk.math.gelu(g) * u)

@T.rowwise
def rms_norm(x, w, *, EPS=1e-6):
    return x * (T.row_mean(x * x) + EPS).rsqrt() * w

if __name__ == "__main__":
    gate_up = torch.randn(4096, 2 * 14336, device="cuda", dtype=torch.bfloat16)
    g, u = gate_up[:, :14336], gate_up[:, 14336:]        # the halves, as views
    y = gelu_and_mul(g, u)
    torch.testing.assert_close(y, F.gelu(g, approximate="tanh") * u, rtol=2e-2, atol=2e-2)

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
    torch.testing.assert_close(rms_norm(x, w), ref.to(torch.bfloat16), rtol=2e-2, atol=2e-2)
    print(gelu_and_mul.plan(g, u))
    print(rms_norm.plan(x, w))
```

```bash
python3 first.py
```

A hook gets fp32 register strips of a row (the load widens bf16, the store
rounds back to nearest even, as torch does); `T.row_mean`, `T.row_sum`,
`T.row_max` and `T.row_logsumexp` reduce the whole row. The template picks
the layout, the warps, the grid order and the launch for the GPU it runs
on, from the plans hk's tuned `hk.ops` kernels measured, and handles ragged
shapes and strided views; `plan()` prints what it chose. Measured against
the tuned kernel for the same job, the templates are 0.98-1.03x on an MI355X
and 0.92-1.02x on a W7900D, and they beat `torch.compile` at large shapes
([measured](templates/elementwise-rowwise.md#measured)).

A GEMM with a fused epilogue -- a bias, an activation, a residual -- is
`hk.templates.gemm(a, b, epilogue=fn, **operands)`, the tuned GEMM of each
GPU with your function traced into its store ([GEMM template](templates/gemm.md)).

The [elementwise and rowwise guide](templates/elementwise-rowwise.md) has
softmax, the fused add + RMSNorm, LayerNorm, int8 quantization, a
cross-entropy over 151936 columns and an MoE router, each a few lines.

## Check it

`python -m hk.check` builds a kernel, checks it against a torch reference
and times it against torch eager and `torch.compile` in one process:

```bash
python -m hk.check first.py:rms_norm \
    --ref "lambda x, w: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w" \
    --inputs "bf16[4096,4096], bf16[4096]"
```

It prints the registers, occupancy and spills of every kernel it launched,
the error against the reference (computed in fp32) next to torch's own, the
times with their noise, a verdict per baseline (`win`, `tie`, `loss`) and
hints. Exit status 0 is pass, 1 a wrong answer, 2 an error; `--json out.json`
writes it all for a script or an agent. A template's reference is a
`lambda` (its arguments have no names to read); a raw kernel's can be an
expression over its parameter names. Everything else:
[hk.check](check.md).

## Writing the kernel yourself

When no template fits, write the kernel. A kernel is a Python function over
tensors, traced into the hk dialect; this one adds two bf16 tensors:

```python
# add.py
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    t = hk.RT(hk.bf16, 16, 64)
    o[idx] = a[idx].load(t) + b[idx].load(t)
```

```python
import torch
from add import add

print(add.trace().dump())       # the hk MLIR module -- no compiler, no GPU
print(add.source())             # the HipKittens C++ it translates to
print(add.build().report())     # hipcc and the resource gate -- no GPU

a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = torch.empty_like(a)
add(a, b, out)                  # launches, on the GPU
torch.testing.assert_close(out, a + b, rtol=3e-2, atol=1e-2)
```

Keep the kernel in a file: a kernel body has to be source `inspect` can
read, so a body typed into `python3 -c` traces but cannot have runtime loops
([limits](language.md#limits)). One block of the grid adds one 16x64 tile; a
2-D tensor is `(rows, cols)`, and the grid lambda reads its extents.

**This is the shape of a kernel, not the fast way to stream memory.** A
16x64 register tile is the matrix core's layout: one warp a tile, each lane
holding pieces of several rows. For an elementwise or row-wise kernel it
leaves bandwidth unused. The same tiling of a GELU-and-mul is 0.78x
`torch.compile` on an MI355X and 0.83x on a W7900D (geomean over eight LLM
shapes; 0.54x at 4096x14336 on the MI355X), where the template above is
1.11x and 1.03x ([the DX suite](../bench/dx/FINDINGS.md),
[the templates](templates/elementwise-rowwise.md#measured)). Write a
streaming kernel on **register strips**, each lane owning 16-byte runs of
a row:

```python
# gelu_mul.py
import hk

@hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
def gelu_mul(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *, CW=512):
    S = hk.RS(hk.fp32, 1, CW, 8)                  # a register strip: one row, 8 bf16 (16 B) a lane
    per_row = hk.cdiv(o.cols, CW)
    row = hk.arch.block_idx.x // per_row
    c0 = (hk.arch.block_idx.x - row * per_row) * CW
    x = a.at[row, c0].load(S, bounded=True)       # bf16 -> fp32 on the load; past the end reads 0
    x.gelu(into=x)                                # hk.math.gelu: the tanh form, fp32 only
    x *= b.at[row, c0].load(S, bounded=True)
    o.at[row, c0].store(x, bounded=True)          # fp32 -> bf16 on the store
```

It ties `torch.compile` at 4096x14336 on an MI355X and is 1.02x it on a
W7900D ([hk.check](check.md#the-loop)). A strip's columns are whole chunks
of the wave's lanes times the run: 32 x 8 on gfx1100, 64 x 8 on gfx950, so
`CW=512` fits both ([register strips](language.md#register-strips)). Keep
register tiles (`hk.RT`), staged through LDS (`hk.ST`), for what feeds the
matrix core: GEMM and attention, where the examples and the tuned
`hk.ops` kernels are the models.

`add.build().report()` prints one line per kernel instantiation. Built for
gfx1100 (ROCm 7.2.4) while writing this page, it read:

```
add_kernel(globals): vgpr=76->96 occ=16 lds=0 scratch=0 spill=0 hdrm=20
```

A kernel that spilled would not get this far: `build()` raises
`hk.ResourceError` instead of returning it
([why](runtime.md#the-resource-gates)).

## The four entry points

Four entry points, in increasing order of what they need:

| call | needs | does |
|---|---|---|
| `k.trace()` | nothing | runs the body once into an hk MLIR module and runs the passes |
| `k.source()` | nothing | the HipKittens C++ translated from that module |
| `k.build()` | hipcc | compiles and **gates on the resource remarks** |
| `k(*tensors)` | an AMD GPU + torch | launches |

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop.

Each takes the kernel's constexprs as keyword arguments, `k.trace(D=64)` or
`k(x, o, EPS=eps)`, and caches per set of them; the first call to `k(...)`
does all of trace, source and build (once) and later calls are a dict lookup
([what a launch does](runtime.md#what-a-launch-does)).

## Timing a kernel, and CUDA graphs

Time kernels in one process, interleaved round by round against their
baselines, as `hk.check` does: across processes an A/B of the same kernel
has shown 13-16% differences out of nothing
([how numbers are taken](performance.md#how-the-numbers-are-taken)).

**On gfx1100, `k(...)` launches on the null stream.** RDNA's launcher takes
no stream (CDNA's takes torch's current one). Eager back-to-back calls on
torch's default stream are timed correctly, but a CUDA graph does not
capture the kernel -- a replay runs nothing, and times nothing -- and work
on another stream is not ordered against it. To capture an hk kernel on
gfx1100, or to run it on a stream of your own, launch it on torch's stream:
set `HK_LAUNCH=codeobj` before the first call (every `@hk.kernel` and
template then launches its code object on torch's current stream), or call
it through `hk.torch_op(k)` ([the stream](runtime.md#what-a-launch-does)).
The serving engines' patches already do.

## Running the tests

```bash
ninja -C $HK_MLIR_BUILD check-hk                     # lit: the dialect, passes, translation
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
python3 -m pytest tests/hk/gpu -q                    # an AMD GPU and torch
```

A tier that cannot run on this machine is skipped with a reason. The suite
builds for gfx1100 unless `HK_ARCH` says otherwise, so on another GPU run it
with `HK_ARCH` set to that GPU's arch (`HK_ARCH=gfx950` on an MI355X). The
tiers and what each catches are in [testing](testing.md).

## Running the examples

[`examples/`](../examples/README.md) has eleven kernels written from scratch,
from the smallest complete one to flash attention, Triton's tutorials and
the row kernels a serving engine runs; a kernel taken into vLLM end to end
(`11_vllm_integration.py`); two files of template kernels
(`12_templates_rowwise.py`, `13_templates_gemm.py`); and `validate.py`,
which checks the shipped `hk.ops` kernels against torch. Each kernel example
checks itself against torch and then times itself:

```bash
python3 examples/12_templates_rowwise.py       # on the GPU: check against torch, then time
```

On a machine without the GPU, the same scripts show the IR, the C++ or the
compiled resources:

```bash
python3 examples/01_vector_add.py --dump ir    # the hk MLIR module -- no GPU, no compiler
python3 examples/01_vector_add.py --dump cpp   # the HipKittens C++ it translates to
python3 examples/01_vector_add.py --build      # compile with hipcc, print the resources -- no GPU
```

## Your model in vLLM

On a Radeon (gfx1100), with vLLM installed, hk's linear-layer patch runs a
model's unquantized bf16 linear layers on hk's decode GEMV and GEMM wherever
they measured faster than vLLM's own kernels:

```bash
HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen2.5-7B-Instruct
```

The routes shipped in the wheel were measured on Qwen3-8B. For any other
model, the first start measures the shapes the shipped table lacks, after
the weights load and before vLLM sizes its cache. It spends at most
`HK_AUTOTUNE_BUDGET` seconds on that (60 by default), decode-sized batches
first. The results are kept, so later starts measure nothing. The `[hk]`
line in the server's log says what was measured and what stayed on vLLM's
path. To pay that cost ahead of time instead:

```bash
python -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct   # add --tp N for tensor parallel
```

To see what it bought on your machine, against vLLM as it ships, in one
command (greedy outputs are compared too):

```bash
python -m hk.integration ab --model Qwen/Qwen2.5-7B-Instruct --arms stock,gemm
```

What it measured on a model no table had, and what it costs, is in
[chapter 35](history/35-any-model.md). The patches, and what each one
replaces, are in [integration](integration.md).

## Where next

* [The templates](templates/elementwise-rowwise.md) and the
  [GEMM template](templates/gemm.md): every hook, operand and plan override.
* [hk.check](check.md): the loop, the options, the JSON report.
* [The language](language.md): parameters, tiles and strips, operators,
  memory, control flow, and what the language refuses.
* [The API reference](api/README.md): every name, and
  [hk.math](api/hk.math.md)'s functions.
* [The runtime](runtime.md): the cache, the gates, the stream, autotune,
  `torch.ops`.
* [Performance](performance.md): what each kernel and patch measured, level
  by level for RDNA3 and CDNA4, the losses included.
