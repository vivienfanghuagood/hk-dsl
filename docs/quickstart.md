# Quickstart

Two ways in. Pick the one you came for:

- [Make vLLM or SGLang faster on your Radeon](#make-vllm-or-sglang-faster):
  install hk next to the engine. No code.
- [Write your first kernel](#your-first-kernel-a-template): a template, then
  `hk.check`, then a kernel written by hand.

## Make vLLM or SGLang faster

In the Python environment that runs your engine, on a Radeon W7900 / RX 7900
(gfx1100) or R9700 / RX 9070 (gfx1201):

```bash
pip install hk-dsl
python3 -m hk.integration --status      # the routes hk will apply on this GPU
```

Then start the engine as you always do. hk's plugin loads by itself and prints
one `[hk] ... routes on: ...` line per process; `HK_DISABLE=1` gives you the
stock engine back.

```bash
vllm serve Qwen/Qwen3-8B
python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend triton --disable-cuda-graph
```

Which models and GPUs, how to check it worked, and troubleshooting:
[hk in vLLM](vllm.md), [hk in SGLang](sglang.md) (SGLang is gfx1100 only for
now).

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
shapes and strided views; `plan()` prints what it chose. A template runs
the plan of hk's tuned kernel for the same job; `hk.check`, below, times it
against `torch.compile` on your GPU.

A GEMM with a fused epilogue -- a bias, an activation, a residual -- is
`hk.templates.gemm(a, b, epilogue=fn, **operands)`, the tuned GEMM of each
GPU with your function traced into its store ([GEMM template](templates/gemm.md)).

The [elementwise and rowwise guide](templates/elementwise-rowwise.md) has
softmax, the fused add + RMSNorm, LayerNorm, int8 quantization, a
cross-entropy over 151936 columns and an MoE router, each a few lines.

## Check it

`python3 -m hk.check` builds a kernel, checks it against a torch reference
and times it against torch eager and `torch.compile` in one process:

```bash
python3 -m hk.check first.py:rms_norm \
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
leaves bandwidth unused: the same tiling of a GELU-and-mul is slower than
`torch.compile` on both GPUs. Write a streaming kernel on
**register strips**, each lane owning 16-byte runs of a row:

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

`hk.check` times it against `torch.compile` on your GPU
([hk.check](check.md#the-loop)). A strip's columns are whole chunks
of the wave's lanes times the run: 32 x 8 on gfx1100, 64 x 8 on gfx950, so
`CW=512` fits both ([register strips](language.md#register-strips)). Keep
register tiles (`hk.RT`), staged through LDS (`hk.ST`), for what feeds the
matrix core: GEMM and attention, where the examples and the tuned
`hk.ops` kernels are the models.

One kernel per dtype (or per `warps`, or per anything a constexpr cannot
carry) is a factory: put `@hk.specialize` on a function that defines the
kernel and returns it, and it builds each variant once and names it after its
arguments -- no `lru_cache`, no name string, no `body.__name__`
([kernels made by a factory](language.md#kernels-made-by-a-factory)):

```python
@hk.specialize
def gelu_mul_for(*, dtype=hk.bf16):
    @hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
    def gelu_mul(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *, CW=512):
        ...                                       # the body above
    return gelu_mul

gelu_mul_for(dtype=hk.fp16)                       # the kernel `gelu_mul_for__dtypefp16`
```

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
([how numbers are taken](performance/method.md)).

**On gfx1100, `k(...)` launches on the null stream.** RDNA's launcher takes
no stream (CDNA's takes torch's current one). Eager back-to-back calls on
torch's default stream are timed correctly, but a CUDA graph does not
capture the kernel -- a replay runs nothing, and times nothing -- and work
on another stream is not ordered against it; a call under a non-default
torch stream warns once (`HK_STRICT_STREAM=1` makes it an error). To capture
an hk kernel on gfx1100, or to run it on a stream of your own, launch it on
torch's stream: set `HK_LAUNCH=codeobj` before the first call (every
`@hk.kernel` and template then launches its code object on torch's current
stream), or call it through `hk.torch_op(k)`
([the stream](runtime.md#what-a-launch-does)). `hk.torch_op` makes the kernel a
`torch.library` custom op that `torch.compile` traces without a graph break and
a CUDA graph captures: [use a kernel as a PyTorch op](torch-ops.md).
The serving engines' patches already do.

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
* [hk in vLLM](vllm.md) and [hk in SGLang](sglang.md): the engines.
* [The scorecard](performance/scorecard.md): the end-to-end results the
  project quotes.
* [Examples](examples.md): fifteen examples, from a vector add to flash
  attention, two template files and `validate.py`.
