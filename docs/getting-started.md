# Getting started

From nothing to a kernel you have traced, read, compiled and launched.

- [Install](#install)
- [Your first kernel](#your-first-kernel)
- [The four entry points](#the-four-entry-points)
- [Running the tests](#running-the-tests)
- [Running the examples](#running-the-examples)
- [Where next](#where-next)

## Install

You need Linux x86_64 and ROCm's hipcc to compile; a Radeon (gfx1100, RDNA3)
and torch to launch. torch is not a dependency of the package — install the
ROCm build you already use.

**From a wheel.** A wheel is built per Python interpreter, because it carries
the MLIR bindings for one CPython. CPython 3.10 and 3.12 wheels are attached
to each [GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases);
for another interpreter, build one ([building](building.md#wheels)). Then:

```bash
pip install hk_dsl-0.3.0-cp310-cp310-linux_x86_64.whl
```

It carries the HipKittens headers, the bindings and the shipped tuning
records, so nothing else is needed at run time but hipcc.

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

## Your first kernel

```python
import torch
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    t = hk.RT(hk.bf16, 16, 64)
    o[idx] = a[idx].load(t) + b[idx].load(t)

print(add.trace().dump())       # the hk MLIR module -- no compiler, no GPU
print(add.source())             # the HipKittens C++ it translates to
print(add.build().report())     # hipcc and the resource gate -- no GPU

a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = torch.empty_like(a)
add(a, b, out)                  # launches, on a Radeon
torch.testing.assert_close(out, a + b, rtol=3e-2, atol=1e-2)
```

Save it as a file and run it: a kernel body has to be source `inspect` can
read, so a body typed into `python3 -c` traces but cannot have runtime loops
([limits](language.md#limits)). One block of the grid adds one 16x64 tile; a
2-D tensor is `(rows, cols)`, and the grid lambda reads its extents. The
tolerance is the one the GPU tests use for bf16.

`add.build().report()` prints one line per kernel instantiation. Built for
gfx1100 while writing this page, it read:

```
add_kernel(globals): vgpr=71->72 occ=16 lds=0 scratch=0 spill=0 hdrm=25
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
| `k(*tensors)` | a Radeon + torch | launches |

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop.

Each takes the kernel's constexprs as keyword arguments, `k.trace(D=64)` or
`k(x, o, EPS=eps)`, and caches per set of them; the first call to `k(...)`
does all of trace, source and build (once) and later calls are a dict lookup
([what a launch does](runtime.md#what-a-launch-does)).

## Running the tests

```bash
ninja -C $HK_MLIR_BUILD check-hk                     # lit: the dialect, passes, translation
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
python3 -m pytest tests/hk/gpu -q                    # a Radeon and torch
```

A tier that cannot run on this machine is skipped with a reason. The tiers and
what each catches are in [testing](testing.md).

## Running the examples

[`examples/`](../examples/README.md) has six kernels, from the smallest
complete one to flash attention, and `validate.py`, which checks the shipped
`hk.ops` kernels against torch. Each kernel example checks itself against
torch and then times itself:

```bash
python3 examples/01_vector_add.py              # on the GPU: check against torch, then time
```

On a machine without the GPU, the same scripts show the IR, the C++ or the
compiled resources:

```bash
python3 examples/01_vector_add.py --dump ir    # the hk MLIR module -- no GPU, no compiler
python3 examples/01_vector_add.py --dump cpp   # the HipKittens C++ it translates to
python3 examples/01_vector_add.py --build      # compile with hipcc, print the resources -- no GPU
```

## Where next

* [The language](language.md): parameters, types, operators, memory, control
  flow, and what the language refuses.
* [The IR](ir.md): what `dump()` shows, the passes, the translation.
* [The runtime](runtime.md): the cache, the gates, autotune, `torch.ops`.
* [Integration](integration.md): running the kernels inside vLLM or SGLang.
