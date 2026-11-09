# Use a kernel as a PyTorch op

`hk.torch_op(kernel)` registers one of your kernels as a `torch.library` custom op,
`torch.ops.<namespace>.<name>`. As an op, the kernel runs on torch's current stream,
`torch.compile` traces it without a graph break, and a CUDA graph captures it. Those are the
properties a serving engine or a compiled model needs. The serving engines' routes call hk's
kernels this way.

- [The recipe](#the-recipe)
- [What works](#what-works)
- [What it does not do](#what-it-does-not-do)
- [The two registrations](#the-two-registrations)
- [How it is tested](#how-it-is-tested)

## The recipe

A kernel from the [quickstart](quickstart.md#writing-the-kernel-yourself), in a file of its own:

```python
# gelu_mul.py
import hk

@hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
def gelu_mul(a: hk.Tensor[hk.T], b: hk.Tensor[hk.T], o: hk.Tensor[hk.T], *, CW=512):   # hk.T: the call's dtype
    S = hk.RS(hk.fp32, 1, CW, 8)
    per_row = hk.cdiv(o.cols, CW)
    row = hk.arch.block_idx.x // per_row
    c0 = (hk.arch.block_idx.x - row * per_row) * CW
    x = a.at[row, c0].load(S, bounded=True)
    x.gelu(into=x)
    x *= b.at[row, c0].load(S, bounded=True)
    o.at[row, c0].store(x, bounded=True)
```

```python
import torch
import hk
from gelu_mul import gelu_mul

op = hk.torch_op(gelu_mul.variant(T=hk.bf16), ns="mylib", name="gelu_mul")   # the bf16 kernel: torch.ops.mylib.gelu_mul
print(op.default._schema)  # mylib::gelu_mul(Tensor a, Tensor b, Tensor(a!) o) -> ()

def mlp_act(gate, up):
    out = torch.empty_like(gate)
    torch.ops.mylib.gelu_mul(gate, up, out)
    return out

g = torch.randn(4096, 14336, device="cuda", dtype=torch.bfloat16)
u = torch.randn_like(g)
y = mlp_act(g, u)                                   # eager
y = torch.compile(mlp_act, fullgraph=True)(g, u)    # compiled, no graph break
```

Call `hk.torch_op` once per process, at import time of your module, before anything compiles or
captures. It is memoised on `(ns, name)`, because torch refuses to register an op twice.
Building the op needs hipcc but no GPU, and it goes through hk's
[resource gates](runtime.md#the-resource-gates) like any build.

## What works

| use | status | notes |
|---|---|---|
| eager calls | yes | runs on torch's current stream, so it is ordered with your other work on that stream |
| `torch.compile(fullgraph=True)` | yes | no graph break; the op has a Meta (fake) implementation |
| `torch.compile(dynamic=True)` | yes | the kernel reads its extents at launch; one compiled graph serves every shape |
| `torch.compile(mode="reduce-overhead")` | yes | Inductor's CUDA graphs capture and replay it |
| `torch.cuda.CUDAGraph` / `torch.cuda.graph` | yes | a replay reads the input tensors as they are at replay time; warm up once before capture |
| `torch.library.opcheck` | passes | schema, fake implementation, autograd and AOT-dispatch registration checks |
| vLLM / SGLang graphs | yes | the routes of [the serving engines](integration.md) use this path |

**The op takes its outputs as arguments.** Every global tensor the kernel stores into becomes a
`Tensor(a!)` out-parameter, and the op returns nothing. The IR knows where a kernel writes, but
not what shape a result should have, and an engine usually has the buffer already. Allocate the
output (`torch.empty_like`) and pass it in, as above. `torch.compile` handles the mutation
through functionalization.

**The checks are torch's.** A call with a wrong dtype, a tensor on the CPU, or a non-contiguous
tensor that is not a slice of its last axis raises a `RuntimeError` naming the argument. A strided
view is never read wrong. Shapes are your kernel's contract: an `hk.ops` kernel with a fixed
layout checks it at its entry point, and your own kernel should do the same before calling the
op if its loads are not bounded.

## What it does not do

- **No Inductor lowering.** The op is an opaque call in a compiled graph. Inductor does not fuse
  it with its neighbours or generate code for it. The ops around it are compiled as usual.
- **No autograd formula.** The op has no backward. If you need one, register it with
  `torch.library.register_autograd`, typically with a second hk kernel for the gradient.
- **Templates are not ops.** `hk.templates` kernels (`T.elementwise`, `T.rowwise`,
  `T.gemm`) are Python callables that pick a plan per call. To use one inside a compiled graph,
  wrap it in your own `torch.library.custom_op` with a fake implementation.
- **One device per call.** Every tensor must be on the same GPU. A module is loaded per device
  on first use.

## The two registrations

By default the op is registered over a device-only code object (`hk.runtime.codeobj`). hipcc
compiles only the kernel, and the launch is done in Python through the HIP runtime torch already
loaded. That takes about a second a kernel.

`HK_TORCH_OP=library`, set before `hk.torch_op` is called, builds a libtorch registration
instead. That one compiles torch's headers, about 15 s a kernel, and its implementation is C++
behind the dispatcher. Both have the same schema, the same checks and the same behaviour in every
row of the table above. [The runtime](runtime.md#torch-ops-and-the-sdpa-drop-in) has the
details and compile times.

On gfx1100, calling a kernel directly (`k(...)`) launches on the null stream, which a CUDA graph
does not capture. Going through `hk.torch_op` launches on torch's stream on every GPU
([timing and CUDA graphs](quickstart.md#timing-a-kernel-and-cuda-graphs)).

## How it is tested

`tests/hk/gpu/test_torch_op_recipe.py` runs this page's recipe on every GPU of the GPU tier
(gfx1100, gfx1201, gfx950), for both registrations:
- eager results against a float reference;
- `torch.library.opcheck`;
- `torch._dynamo.explain` (one graph, no break);
- `torch.compile` with `fullgraph`, `dynamic` and `reduce-overhead` against eager over several
  shapes;
- a CUDA graph captured and replayed on new inputs.

`tests/hk/gpu/test_torch_op.py` covers the stream, the out-parameter and the refusals.
