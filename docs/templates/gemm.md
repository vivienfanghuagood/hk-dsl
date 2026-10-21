# `hk.templates.gemm`: a tuned GEMM whose store is yours

```python
import hk

def bias_gelu(acc, ctx):                      # acc: an fp32 tile of x @ w.T
    acc += ctx.bias                           # ctx.bias: b, one value per column
    return hk.templates.gemm.gelu_tanh(acc)   # GELU(tanh), applied in registers

y = hk.templates.gemm(x, w, epilogue=bias_gelu, bias=b)
# == F.gelu(F.linear(x, w, b), approximate="tanh"), one kernel
```

`C = epilogue(A @ Bᵀ accumulated in fp32, ctx)`, with A `(M, K)` and B
`(N, K)` bf16 -- B as `F.linear` and `hk.ops.matmul` take it -- and C
`(M, N)` in `out_dtype`. The GEMM is the one `hk.ops.matmul` runs for the
shape, kernel for kernel and schedule for schedule; the epilogue is a traced
function that replaces its store. So a linear layer's bias, activation,
residual add or output scale costs a few percent of the GEMM instead of a
second pass over C, on a Radeon (gfx1100) and an Instinct (gfx950) alike.

- [The call](#the-call)
- [Writing an epilogue](#writing-an-epilogue)
- [Ready-made epilogues](#ready-made-epilogues)
- [How the plan is chosen](#how-the-plan-is-chosen)
- [What it costs: measured](#what-it-costs-measured)
- [Limits](#limits)

## The call

```python
hk.templates.gemm(a, b, *, epilogue=None, out=None, out_dtype=None, **operands) -> C
hk.templates.gemm(*, epilogue=..., out_dtype=..., **operands)  # bound: call it as f(a, b, ...)
```

| argument | |
|---|---|
| `a` | `(M, K)` bf16; any row stride |
| `b` | `(N, K)` bf16 |
| `epilogue` | `fn(acc)` or `fn(acc, ctx)`, below; `None` is the plain GEMM (and with bf16 C, *is* `hk.ops.matmul`) |
| `out` | C to write, `(M, N)` with unit column stride (a column slice of a wider tensor is fine); may be the residual itself |
| `out_dtype` | C's dtype when `out` is not given: `torch.bfloat16` (default), `torch.float16`, `torch.float32`; on gfx950 `torch.int8`, `torch.float8_e4m3fn`, `torch.float8_e5m2` |
| `**operands` | what the epilogue reads as `ctx.<name>`, by keyword |

An operand's kind is its shape against C, as torch broadcasting reads it:

| operand | kind | in the epilogue |
|---|---|---|
| `(N,)` or `(1, N)` tensor | row vector: one value per column of C -- a bias, a per-channel scale | a vector in `acc`'s layout; `acc + v` adds it to every row |
| `(M, 1)` tensor | column vector: one value per row -- a per-token scale | a vector; `acc * v` scales each row |
| `(M, N)` tensor | a tile of C's shape -- a residual, a gate | a tile like `acc` |
| a Python `int`/`float` | a constant, compiled in | the number |

Tensors may be bf16, fp16, fp32 (or int8); they are converted to fp32 as they
load. A Python number is a constexpr: a value that changes from call to call
compiles a kernel per value, so pass it as a `(N,)` tensor instead.

The first call for a signature (the epilogue, the shapes, dtypes and strides,
the operand kinds) plans, traces and compiles; later ones are a dictionary
lookup and the launches, so the template's host cost is below
`hk.ops.matmul`'s own (whose `route()` runs every call).

Without a GPU, `hk.templates.gemm.plan(m, n, k, arch=...)` lists the
launches, and `hk.templates.gemm.kernels(m, n, k, epilogue=..., **operands)`
returns them as `(step, kernel, constexprs)` for `kernel.trace().dump()`,
`kernel.source(**constexprs)` and `kernel.build(**constexprs)` -- operands
may be `torch.empty(..., device="meta")`.
[`examples/13_templates_gemm.py`](../../examples/13_templates_gemm.py)
`--dump cpp` / `--build --arch gfx950` uses it.

## Writing an epilogue

```python
def epilogue(acc, ctx):     # or epilogue(acc), if it reads nothing
    ...
    return tile             # what to store; or None, having written acc in place

def epilogue(acc, bias, residual, *, alpha=1.0):   # the same, operands by name
    return (acc + bias) * alpha + residual
```

The parameters after `acc` are bound by name: `ctx` is the context, any
other name the operand the call passed by that name (as
`hk.templates.rowwise`'s hooks take theirs), and a keyword-only parameter a
number with a default that the call may override (`alpha=0.5`).

It is traced like an `@hk.func` body ([the language](../language.md)), once
per kernel and per accumulator tile the kernel stores:

- `acc` is a register tile of C in the accumulator's own layout, fp32. Its
  shape is the kernel's business (a 32x32 WMMA tile on gfx1100, a 64x64
  quadrant of the 256x256 pingpong on gfx950, a 1 x 512 strip in a split-K
  reduction, ...): write the epilogue elementwise, and it is right on all of
  them.
- `ctx.<name>` is an operand, loaded for this tile and converted to fp32 --
  where the epilogue reads it, so one it does not read costs nothing (a
  kernel may load it earlier, to hide the latency: the decode GEMV before its
  K loop, a tile operand a band of rows ahead).
- `ctx.row`, `ctx.col` are C's coordinates of the tile's first element
  (device `Int`s; with them, `acc.right_fill_(1000 - ctx.col, 0.0)` zeroes
  C's columns from 1000 on). `ctx.m` is C's row count; `ctx.tile` the tile's
  type.
- The math is the language's: `+ - * /` between tiles, vectors and numbers,
  `.exp()`, `.exp2()`, `.log()`, `.relu()`, `.silu()`, `.gelu()`,
  `.rsqrt()`, `.maximum()`, `.minimum()`, the in-place forms, `.to()`. A
  Python `if` on a number is a trace-time branch; `if` on device values
  compiles.
- It returns a tile of `acc`'s shape and layout. Store rounding and
  conversion to `out_dtype` are the template's (an int8 C rounds to nearest
  even and saturates).

**`init_from(name)`.** `@hk.templates.gemm.init_from("residual")` on an
epilogue makes the accumulator *start* as the `(M, N)` operand `residual`
instead of zero -- BLAS's `beta * C` with beta 1: `C = fn(residual + A @ Bᵀ,
ctx)`. A kernel that can (gfx1100's, gfx950's pingpong and four-warp blocks)
loads it into the accumulators before its K loop, behind the first tiles'
copies, which on an MI355X costs a third of loading it at the store; one that
cannot (a split-K share, the decode GEMV) adds it at the store. The result
is the same either way, rounded once.

**`chain(f, g, ...)`** is one epilogue applying `f`, then `g`, ... to what the
last returned, all reading the same `ctx`. An `init_from` epilogue can only
come first.

## Ready-made epilogues

Attributes of `hk.templates.gemm`; each names the operands it reads.

| epilogue | computes | operands |
|---|---|---|
| `bias` | `acc + bias` | `bias` (N,) |
| `gelu`, `silu`, `relu` | the activation | -- |
| `bias_gelu`, `bias_silu`, `bias_relu` | `act(acc + bias)` | `bias` |
| `residual` | `acc + residual`, by `init_from` | `residual` (M, N) |
| `bias_residual` | `acc + bias + residual`, by `init_from` | `bias`, `residual` |
| `scale` | `acc * scale` -- with an int8 or fp8 `out_dtype`, a quantizer | `scale`: a number, (N,) or (M, 1) |
| `gelu_tanh(x)` | a helper for your own: GELU(tanh) of a tile, in place | -- |

GELU is the tanh approximation, `F.gelu(x, approximate="tanh")`; there is no
erf GELU in the language yet. `gelu_tanh` computes it as `x * sigmoid(2u)`,
with `sigmoid(2u) = rsqrt(1 + 2^(x (a x^2 + b)))^2`: one `exp2` and one
`rsqrt` an element, where `Tile.gelu()` clamps, exponentiates and divides
correctly rounded. Both agree with torch to a few fp32 ulps; at the end of a
GEMM, where the epilogue's arithmetic overlaps nothing, `gelu_tanh` is half
the cost (on an MI355X 3.6% of a 4096 x 24576 x 4096 GEMM against 9%). Use it
in your own epilogues.

## How the plan is chosen

The template does not choose kernels; it takes `hk.ops.matmul`'s choice and
changes what each launch stores.

**gfx1100** (`hk.ops.gemm`): one launch of the WMMA kernel at
`schedule_for(M, N, K)`'s schedule -- a tuned record from
`python/hk/tuned/gemm_bf16.json` where one exists for the shape's bucket,
else the 128x128x64 default. The epilogue runs on each warp's 32x32
accumulator where it was stored. N must be a multiple of the schedule's
block (128), K of its K-tile (64). The one place the template leaves
`hk.ops.matmul`'s choice is a decode batch, M <= 32: `hk.ops.matmul` runs
the WMMA kernel's 16-row `THIN` schedule there, the template
`hk.ops.decode_gemm` (the GEMV hk's vLLM routes already use) with the
epilogue -- 1.6-2.7x faster on a W7900D at Qwen3-8B's layers. At 33-64 rows
the GEMV's four token tiles leave no registers for an epilogue, so those
stay on the WMMA kernel.

**gfx950** (`hk.ops.gemm_cdna`): `route(M, N, K)` -- the measured table in
`python/hk/tuned/routes_gemm_gfx950.json`, else its rule -- names a
candidate, and the plan follows `gemm_cdna.run` launch for launch:

| candidate | launches | the epilogue runs |
|---|---|---|
| `pp`, `pp128`, `ppnt` (the 256x256 eight-warp pingpong) | one | in the store of each 64x64 quadrant (the first two inside the pingpong, as the plain kernel stores them) |
| `dp*` (four-warp blocks, weight-streaming `nt` ones included) | one | in each warp's store |
| `decode` (the GEMV; M <= 64) | one | on warp 0's combined 16·tokens x 16 tile, after the warps' partial sums meet; its operands are loaded before the K loop |
| `*k2`, `*k4`, `*k8` (split-K) | the GEMM, storing fp32 shares to a scratch; then a reduction | **only in the reduction**, once, on the sum of the shares |
| `ppw`, `ppwd` (whole waves of 256-blocks, the rest of N another candidate's) | one per part, on a column slice of C | in each part's own way, with `ctx.col` still C's column |
| `pp` at an N that 256 does not divide | the 256-wide blocks, then the last 128 columns by `tail_for` (often split) | likewise |

Split-K is where an epilogue can go wrong: run on each share, a bias would
be added `splits` times and a GELU taken of partial sums. The template never
does that -- the share kernels take no epilogue (`gemm_kernel(splits>1,
epilogue=...)` refuses one) and the reduction that adds the shares in order
runs it on the total. That is the plan for every epilogue, linear or not: a
linear one (a scale) would gain nothing from running on the shares, since
the reduction pass is there anyway. The decode GEMV's split (workgroups
meeting through a counter) likewise runs it in the workgroup that arrives
last, on the sum. `plan()` shows which launch carries it:

```
>>> for s in hk.templates.gemm.plan(128, 4096, 4096, arch="gfx950"): print(s)
dp [dp64k4] cols 0:4096 -> scratch
reduce [dp64k4] cols 0:4096 -> out +epilogue
```

The reduction runs on strips, a row of C a grid row (never the flat view
the plain reduction uses, so that the epilogue's coordinates are C's).

Small M follows the route as well: on gfx950 M <= 4 goes to the decode GEMV
and 16-64 to the weight-streaming or split four-warp blocks, all with the
epilogue; on gfx1100 M <= 32 goes to the decode GEMV, above.

## What it costs: measured

Measured interleaved in one process (`tools/hk-bench/gemm_epilogue.py
--graphs`), at Qwen3-8B's and Llama-3-8B's linear layers (qkv and o_proj are
the same shape in both), every layer with both epilogues -- bias +
GELU(tanh), an up-projection's, and a residual add, o_proj's and down's --
whichever it would really carry. Medians of rounds of bursts; *device* is a CUDA graph of
the calls replayed (the GPU's time, as a serving engine's captured step sees
it), *host* back-to-back eager calls (what a Python caller pays). Columns:
`hk.ops.matmul` alone (the GEMM the template is), the template, the unfused
sequence (`hk.ops.matmul`, then the epilogue in torch), `torch.compile` of
`F.linear` + the epilogue (default mode), and hipBLASLt (`F.linear`) + the
epilogue in torch, eager.

### MI355X (gfx950)

**Device time.** Against plain `hk.ops.matmul`: 45 of 48 within 5% (0.97-1.05x).
The misses are Llama-3-8B's gate_up at M = 1024, 1.09x with either epilogue,
and o_proj at 4096 with bias + GELU, 1.051x: a GEMM of 1.75 waves of 256x256
workgroups pays the epilogue's arithmetic once a wave with nothing to overlap
it (every other route candidate is slower with the epilogue still,
measured), and a 4096-wide output is little work beside it. Faster than the
unfused sequence at all 48 (1.04-1.45x); than `torch.compile` at 46, level
at two (down at M = 4096 with bias + GELU, 0.99x and 1.00x, where the plain
`hk.ops.matmul` is 3-4% behind hipBLASLt); than hipBLASLt + the epilogue at
46, 0.99x at two (qkv and Qwen's gate_up at M = 1024, residual).

| layer (N x K) | M | epilogue | hk.ops.matmul | template | vs hk | unfused hk + torch | torch.compile | hipBLASLt + torch | template vs compile | vs hipBLASLt+ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-8b o_proj (4096 x 4096) | 16 | bias_gelu | 8.9 | 9.1 | 1.020 | 14.8 | 10.8 | 11.5 | 1.20x | 1.27x |
| qwen3-8b o_proj (4096 x 4096) | 128 | bias_gelu | 14.6 | 14.5 | 0.993 | 20.6 | 21.8 | 22.8 | 1.50x | 1.57x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | bias_gelu | 36.3 | 37.3 | 1.028 | 51.4 | 47.4 | 47.3 | 1.27x | 1.27x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | bias_gelu | 94.2 | 99.1 | 1.051 | 128.3 | 111.2 | 110.5 | 1.12x | 1.12x |
| qwen3-8b down (4096 x 12288) | 16 | bias_gelu | 19.3 | 19.3 | 1.000 | 24.9 | 23.3 | 23.8 | 1.21x | 1.23x |
| qwen3-8b down (4096 x 12288) | 128 | bias_gelu | 27.6 | 27.7 | 1.003 | 33.9 | 34.3 | 34.7 | 1.24x | 1.25x |
| qwen3-8b down (4096 x 12288) | 1024 | bias_gelu | 89.2 | 88.7 | 0.994 | 104.6 | 106.0 | 106.8 | 1.19x | 1.20x |
| qwen3-8b down (4096 x 12288) | 4096 | bias_gelu | 259.3 | 266.2 | 1.027 | 294.7 | 267.2 | 268.0 | 1.00x | 1.01x |
| llama3-8b down (4096 x 14336) | 16 | bias_gelu | 21.7 | 21.4 | 0.989 | 27.1 | 27.9 | 28.1 | 1.30x | 1.31x |
| llama3-8b down (4096 x 14336) | 128 | bias_gelu | 29.9 | 29.9 | 0.998 | 36.0 | 36.7 | 37.1 | 1.23x | 1.24x |
| llama3-8b down (4096 x 14336) | 1024 | bias_gelu | 100.3 | 99.3 | 0.990 | 114.9 | 115.7 | 116.2 | 1.17x | 1.17x |
| llama3-8b down (4096 x 14336) | 4096 | bias_gelu | 304.9 | 311.5 | 1.022 | 350.7 | 308.7 | 315.0 | 0.99x | 1.01x |
| qwen3-8b qkv (6144 x 4096) | 16 | bias_gelu | 12.2 | 12.3 | 1.014 | 17.9 | 14.9 | 15.4 | 1.20x | 1.24x |
| qwen3-8b qkv (6144 x 4096) | 128 | bias_gelu | 18.2 | 18.2 | 1.000 | 24.8 | 23.0 | 23.7 | 1.26x | 1.31x |
| qwen3-8b qkv (6144 x 4096) | 1024 | bias_gelu | 56.9 | 57.2 | 1.005 | 77.1 | 58.1 | 58.0 | 1.02x | 1.01x |
| qwen3-8b qkv (6144 x 4096) | 4096 | bias_gelu | 149.7 | 155.0 | 1.036 | 194.7 | 179.3 | 174.4 | 1.16x | 1.13x |
| qwen3-8b gate_up (24576 x 4096) | 16 | bias_gelu | 37.0 | 36.7 | 0.991 | 41.9 | 42.3 | 43.8 | 1.15x | 1.20x |
| qwen3-8b gate_up (24576 x 4096) | 128 | bias_gelu | 35.4 | 35.2 | 0.992 | 46.1 | 59.4 | 59.3 | 1.69x | 1.69x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | bias_gelu | 167.5 | 175.1 | 1.045 | 218.2 | 180.0 | 182.9 | 1.03x | 1.04x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | bias_gelu | 567.7 | 595.0 | 1.048 | 760.5 | 634.9 | 631.9 | 1.07x | 1.06x |
| llama3-8b gate_up (28672 x 4096) | 16 | bias_gelu | 35.2 | 34.5 | 0.981 | 40.4 | 48.4 | 48.5 | 1.40x | 1.40x |
| llama3-8b gate_up (28672 x 4096) | 128 | bias_gelu | 42.1 | 40.6 | 0.965 | 54.0 | 62.3 | 61.8 | 1.54x | 1.52x |
| llama3-8b gate_up (28672 x 4096) | 1024 | bias_gelu | 181.3 | 197.6 | 1.089 | 242.6 | 220.0 | 219.3 | 1.11x | 1.11x |
| llama3-8b gate_up (28672 x 4096) | 4096 | bias_gelu | 667.1 | 697.6 | 1.046 | 882.4 | 735.1 | 738.1 | 1.05x | 1.06x |
| qwen3-8b o_proj (4096 x 4096) | 16 | residual | 8.9 | 9.1 | 1.025 | 10.8 | 10.9 | 10.7 | 1.20x | 1.18x |
| qwen3-8b o_proj (4096 x 4096) | 128 | residual | 14.7 | 14.6 | 0.992 | 16.4 | 22.3 | 22.1 | 1.53x | 1.52x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | residual | 38.1 | 39.5 | 1.037 | 43.6 | 50.0 | 48.3 | 1.27x | 1.22x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | residual | 94.2 | 98.4 | 1.044 | 110.0 | 106.6 | 107.6 | 1.08x | 1.09x |
| qwen3-8b down (4096 x 12288) | 16 | residual | 19.1 | 19.2 | 1.003 | 20.9 | 23.0 | 22.9 | 1.20x | 1.20x |
| qwen3-8b down (4096 x 12288) | 128 | residual | 27.7 | 27.9 | 1.009 | 29.6 | 34.8 | 34.0 | 1.25x | 1.22x |
| qwen3-8b down (4096 x 12288) | 1024 | residual | 89.1 | 90.3 | 1.013 | 94.2 | 104.5 | 104.5 | 1.16x | 1.16x |
| qwen3-8b down (4096 x 12288) | 4096 | residual | 259.0 | 264.5 | 1.021 | 283.1 | 267.3 | 267.3 | 1.01x | 1.01x |
| llama3-8b down (4096 x 14336) | 16 | residual | 21.6 | 21.4 | 0.991 | 23.1 | 27.6 | 27.0 | 1.29x | 1.26x |
| llama3-8b down (4096 x 14336) | 128 | residual | 29.9 | 30.1 | 1.006 | 31.9 | 37.7 | 36.6 | 1.25x | 1.22x |
| llama3-8b down (4096 x 14336) | 1024 | residual | 100.2 | 100.3 | 1.001 | 104.5 | 114.4 | 114.3 | 1.14x | 1.14x |
| llama3-8b down (4096 x 14336) | 4096 | residual | 302.6 | 315.6 | 1.043 | 328.8 | 318.3 | 317.6 | 1.01x | 1.01x |
| qwen3-8b qkv (6144 x 4096) | 16 | residual | 12.1 | 12.3 | 1.015 | 13.9 | 14.8 | 14.6 | 1.21x | 1.19x |
| qwen3-8b qkv (6144 x 4096) | 128 | residual | 18.2 | 18.3 | 1.004 | 20.3 | 22.7 | 23.2 | 1.24x | 1.27x |
| qwen3-8b qkv (6144 x 4096) | 1024 | residual | 56.0 | 57.6 | 1.028 | 62.7 | 59.7 | 57.1 | 1.04x | 0.99x |
| qwen3-8b qkv (6144 x 4096) | 4096 | residual | 149.4 | 155.1 | 1.039 | 171.5 | 173.4 | 172.3 | 1.12x | 1.11x |
| qwen3-8b gate_up (24576 x 4096) | 16 | residual | 35.8 | 36.0 | 1.006 | 36.8 | 42.9 | 42.8 | 1.19x | 1.19x |
| qwen3-8b gate_up (24576 x 4096) | 128 | residual | 35.7 | 35.3 | 0.988 | 38.3 | 58.5 | 57.3 | 1.66x | 1.62x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | residual | 174.4 | 178.7 | 1.024 | 199.7 | 182.4 | 176.2 | 1.02x | 0.99x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | residual | 571.8 | 597.9 | 1.046 | 669.6 | 650.8 | 620.9 | 1.09x | 1.04x |
| llama3-8b gate_up (28672 x 4096) | 16 | residual | 35.0 | 34.1 | 0.973 | 36.0 | 48.5 | 47.6 | 1.42x | 1.40x |
| llama3-8b gate_up (28672 x 4096) | 128 | residual | 41.2 | 40.4 | 0.981 | 44.0 | 62.6 | 61.2 | 1.55x | 1.51x |
| llama3-8b gate_up (28672 x 4096) | 1024 | residual | 183.5 | 200.0 | 1.090 | 222.8 | 219.7 | 215.2 | 1.10x | 1.08x |
| llama3-8b gate_up (28672 x 4096) | 4096 | residual | 670.2 | 703.0 | 1.049 | 781.4 | 760.4 | 726.1 | 1.08x | 1.03x |

<details><summary>Host time (eager calls) on the MI355X</summary>

At M <= 128 the template's host path is shorter than `hk.ops.matmul`'s (whose
`route()` runs every call): 0.7-0.97x it, and it is the fastest of all five
at every M <= 128 shape.

| layer (N x K) | M | epilogue | hk.ops.matmul | template | vs hk | unfused hk + torch | torch.compile | hipBLASLt + torch | template vs compile | vs hipBLASLt+ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-8b o_proj (4096 x 4096) | 16 | bias_gelu | 20.0 | 14.3 | 0.715 | 26.5 | 34.2 | 20.4 | 2.39x | 1.43x |
| qwen3-8b o_proj (4096 x 4096) | 128 | bias_gelu | 20.2 | 15.7 | 0.778 | 27.6 | 34.2 | 26.8 | 2.18x | 1.71x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | bias_gelu | 40.3 | 40.1 | 0.994 | 54.6 | 52.5 | 51.4 | 1.31x | 1.28x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | bias_gelu | 95.8 | 99.2 | 1.036 | 129.6 | 113.7 | 113.0 | 1.15x | 1.14x |
| qwen3-8b down (4096 x 12288) | 16 | bias_gelu | 21.4 | 19.7 | 0.918 | 27.5 | 35.8 | 27.9 | 1.82x | 1.42x |
| qwen3-8b down (4096 x 12288) | 128 | bias_gelu | 30.0 | 28.7 | 0.957 | 35.1 | 41.6 | 40.0 | 1.45x | 1.39x |
| qwen3-8b down (4096 x 12288) | 1024 | bias_gelu | 91.5 | 89.4 | 0.977 | 106.1 | 110.1 | 111.4 | 1.23x | 1.25x |
| qwen3-8b down (4096 x 12288) | 4096 | bias_gelu | 260.9 | 266.8 | 1.023 | 295.4 | 271.6 | 271.8 | 1.02x | 1.02x |
| llama3-8b down (4096 x 14336) | 16 | bias_gelu | 23.6 | 22.2 | 0.938 | 28.8 | 39.2 | 32.0 | 1.77x | 1.44x |
| llama3-8b down (4096 x 14336) | 128 | bias_gelu | 31.6 | 30.8 | 0.976 | 37.7 | 42.0 | 41.6 | 1.36x | 1.35x |
| llama3-8b down (4096 x 14336) | 1024 | bias_gelu | 102.3 | 100.7 | 0.984 | 116.4 | 120.0 | 121.4 | 1.19x | 1.21x |
| llama3-8b down (4096 x 14336) | 4096 | bias_gelu | 303.1 | 313.3 | 1.034 | 352.0 | 314.0 | 315.9 | 1.00x | 1.01x |
| qwen3-8b qkv (6144 x 4096) | 16 | bias_gelu | 19.8 | 14.0 | 0.709 | 26.3 | 34.7 | 20.1 | 2.47x | 1.43x |
| qwen3-8b qkv (6144 x 4096) | 128 | bias_gelu | 20.9 | 18.8 | 0.900 | 27.4 | 34.5 | 27.5 | 1.83x | 1.46x |
| qwen3-8b qkv (6144 x 4096) | 1024 | bias_gelu | 60.1 | 58.6 | 0.974 | 79.3 | 62.4 | 62.3 | 1.07x | 1.06x |
| qwen3-8b qkv (6144 x 4096) | 4096 | bias_gelu | 151.1 | 155.8 | 1.031 | 195.9 | 179.6 | 180.0 | 1.15x | 1.16x |
| qwen3-8b gate_up (24576 x 4096) | 16 | bias_gelu | 37.6 | 36.4 | 0.967 | 42.4 | 62.2 | 47.2 | 1.71x | 1.30x |
| qwen3-8b gate_up (24576 x 4096) | 128 | bias_gelu | 36.4 | 35.4 | 0.971 | 47.2 | 63.0 | 61.9 | 1.78x | 1.75x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | bias_gelu | 166.1 | 174.2 | 1.049 | 219.4 | 182.4 | 184.9 | 1.05x | 1.06x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | bias_gelu | 572.9 | 592.3 | 1.034 | 753.8 | 640.5 | 635.0 | 1.08x | 1.07x |
| llama3-8b gate_up (28672 x 4096) | 16 | bias_gelu | 35.9 | 34.8 | 0.970 | 40.6 | 50.6 | 51.7 | 1.45x | 1.49x |
| llama3-8b gate_up (28672 x 4096) | 128 | bias_gelu | 41.6 | 40.4 | 0.971 | 53.1 | 65.3 | 63.6 | 1.62x | 1.57x |
| llama3-8b gate_up (28672 x 4096) | 1024 | bias_gelu | 185.7 | 203.2 | 1.094 | 250.0 | 223.4 | 221.0 | 1.10x | 1.09x |
| llama3-8b gate_up (28672 x 4096) | 4096 | bias_gelu | 667.9 | 696.6 | 1.043 | 882.4 | 737.0 | 738.8 | 1.06x | 1.06x |
| qwen3-8b o_proj (4096 x 4096) | 16 | residual | 19.6 | 14.1 | 0.718 | 22.3 | 29.6 | 19.3 | 2.11x | 1.37x |
| qwen3-8b o_proj (4096 x 4096) | 128 | residual | 19.7 | 15.5 | 0.788 | 23.0 | 30.8 | 25.1 | 1.98x | 1.62x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | residual | 40.9 | 41.1 | 1.004 | 45.1 | 53.8 | 52.9 | 1.31x | 1.29x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | residual | 96.3 | 98.3 | 1.021 | 111.5 | 111.7 | 113.0 | 1.14x | 1.15x |
| qwen3-8b down (4096 x 12288) | 16 | residual | 21.0 | 19.6 | 0.933 | 22.9 | 30.8 | 26.0 | 1.57x | 1.33x |
| qwen3-8b down (4096 x 12288) | 128 | residual | 29.4 | 28.9 | 0.982 | 30.6 | 39.4 | 37.8 | 1.36x | 1.31x |
| qwen3-8b down (4096 x 12288) | 1024 | residual | 91.7 | 90.7 | 0.989 | 95.6 | 109.0 | 107.1 | 1.20x | 1.18x |
| qwen3-8b down (4096 x 12288) | 4096 | residual | 261.0 | 266.0 | 1.019 | 288.9 | 270.0 | 271.2 | 1.02x | 1.02x |
| llama3-8b down (4096 x 14336) | 16 | residual | 23.3 | 21.8 | 0.937 | 24.7 | 36.3 | 30.7 | 1.66x | 1.41x |
| llama3-8b down (4096 x 14336) | 128 | residual | 31.5 | 30.9 | 0.980 | 32.8 | 41.8 | 39.9 | 1.35x | 1.29x |
| llama3-8b down (4096 x 14336) | 1024 | residual | 102.0 | 101.6 | 0.997 | 106.3 | 119.3 | 119.2 | 1.17x | 1.17x |
| llama3-8b down (4096 x 14336) | 4096 | residual | 310.1 | 322.9 | 1.041 | 338.0 | 316.0 | 319.3 | 0.98x | 0.99x |
| qwen3-8b qkv (6144 x 4096) | 16 | residual | 19.7 | 14.4 | 0.733 | 23.3 | 30.2 | 18.6 | 2.09x | 1.29x |
| qwen3-8b qkv (6144 x 4096) | 128 | residual | 20.2 | 18.8 | 0.929 | 22.7 | 31.3 | 26.2 | 1.66x | 1.39x |
| qwen3-8b qkv (6144 x 4096) | 1024 | residual | 59.4 | 60.1 | 1.011 | 65.5 | 64.4 | 62.2 | 1.07x | 1.04x |
| qwen3-8b qkv (6144 x 4096) | 4096 | residual | 151.1 | 156.1 | 1.033 | 172.4 | 175.2 | 176.5 | 1.12x | 1.13x |
| qwen3-8b gate_up (24576 x 4096) | 16 | residual | 36.4 | 36.3 | 0.998 | 37.3 | 45.5 | 43.6 | 1.25x | 1.20x |
| qwen3-8b gate_up (24576 x 4096) | 128 | residual | 36.5 | 35.5 | 0.972 | 38.7 | 62.7 | 60.3 | 1.77x | 1.70x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | residual | 165.9 | 177.0 | 1.067 | 200.6 | 188.9 | 185.1 | 1.07x | 1.05x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | residual | 578.3 | 607.2 | 1.050 | 674.1 | 657.9 | 635.2 | 1.08x | 1.05x |
| llama3-8b gate_up (28672 x 4096) | 16 | residual | 35.8 | 34.2 | 0.956 | 35.9 | 50.9 | 48.2 | 1.49x | 1.41x |
| llama3-8b gate_up (28672 x 4096) | 128 | residual | 42.3 | 40.8 | 0.965 | 44.8 | 65.3 | 62.9 | 1.60x | 1.54x |
| llama3-8b gate_up (28672 x 4096) | 1024 | residual | 184.2 | 199.4 | 1.083 | 222.4 | 222.2 | 218.9 | 1.11x | 1.10x |
| llama3-8b gate_up (28672 x 4096) | 4096 | residual | 680.5 | 705.9 | 1.037 | 787.4 | 760.0 | 734.5 | 1.08x | 1.04x |

</details>

### W7900D (gfx1100)

Host time (eager calls: hk's gfx1100 entry points launch on the null
stream, which a CUDA graph cannot capture, so there is no replay of them;
at these sizes the host's share is small but for M = 16). `*`: at M = 16 the plain kernel is
`hk.ops.decode_gemm` -- the GEMV the template runs there -- not
`hk.ops.matmul`'s 16-row WMMA kernel, which is 1.2-1.8x slower still (its
times are in the result files). The torch columns are hipBLASLt on this
part, which hk's GEMM beats by 1.3x (docs/history), hence the margins.

All 48 within 5% of the plain kernel (0.89-1.03x); faster than the unfused
sequence, `torch.compile` and hipBLASLt + the epilogue at all 48 (geomean
1.69x over `torch.compile`).

| layer (N x K) | M | epilogue | hk.ops.matmul | template | vs hk | unfused hk + torch | torch.compile | hipBLASLt + torch | template vs compile | vs hipBLASLt+ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-8b o_proj (4096 x 4096) | 16 | bias_gelu | 25.4* | 24.7 | 0.971 | 53.6 | 97.5 | 97.6 | 3.95x | 3.96x |
| qwen3-8b o_proj (4096 x 4096) | 128 | bias_gelu | 98.0 | 99.7 | 1.016 | 118.0 | 134.7 | 139.9 | 1.35x | 1.40x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | bias_gelu | 436.4 | 444.0 | 1.017 | 477.3 | 640.4 | 636.8 | 1.44x | 1.43x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | bias_gelu | 1538.1 | 1566.6 | 1.019 | 1687.3 | 2075.2 | 2069.3 | 1.32x | 1.32x |
| qwen3-8b down (4096 x 12288) | 16 | bias_gelu | 58.8* | 56.4 | 0.958 | 129.1 | 236.5 | 238.3 | 4.20x | 4.23x |
| qwen3-8b down (4096 x 12288) | 128 | bias_gelu | 248.4 | 249.0 | 1.002 | 266.8 | 337.7 | 339.8 | 1.36x | 1.36x |
| qwen3-8b down (4096 x 12288) | 1024 | bias_gelu | 1293.0 | 1321.0 | 1.022 | 1354.5 | 1800.2 | 1798.6 | 1.36x | 1.36x |
| qwen3-8b down (4096 x 12288) | 4096 | bias_gelu | 4739.6 | 4804.8 | 1.014 | 4892.4 | 6312.6 | 6276.5 | 1.31x | 1.31x |
| llama3-8b down (4096 x 14336) | 16 | bias_gelu | 158.0* | 156.9 | 0.993 | 194.4 | 285.8 | 285.6 | 1.82x | 1.82x |
| llama3-8b down (4096 x 14336) | 128 | bias_gelu | 305.5 | 305.7 | 1.001 | 327.6 | 403.2 | 406.7 | 1.32x | 1.33x |
| llama3-8b down (4096 x 14336) | 1024 | bias_gelu | 1510.3 | 1524.5 | 1.009 | 1545.8 | 2110.5 | 2107.7 | 1.38x | 1.38x |
| llama3-8b down (4096 x 14336) | 4096 | bias_gelu | 5653.9 | 5706.4 | 1.009 | 5787.4 | 7623.3 | 7575.6 | 1.34x | 1.33x |
| qwen3-8b qkv (6144 x 4096) | 16 | bias_gelu | 32.5* | 31.7 | 0.975 | 58.4 | 136.8 | 132.7 | 4.32x | 4.19x |
| qwen3-8b qkv (6144 x 4096) | 128 | bias_gelu | 99.7 | 99.8 | 1.001 | 132.8 | 186.7 | 193.1 | 1.87x | 1.94x |
| qwen3-8b qkv (6144 x 4096) | 1024 | bias_gelu | 582.0 | 590.3 | 1.014 | 638.3 | 852.6 | 858.0 | 1.44x | 1.45x |
| qwen3-8b qkv (6144 x 4096) | 4096 | bias_gelu | 2242.3 | 2289.5 | 1.021 | 2497.1 | 3091.7 | 3086.0 | 1.35x | 1.35x |
| qwen3-8b gate_up (24576 x 4096) | 16 | bias_gelu | 260.7* | 260.8 | 1.000 | 439.1 | 585.0 | 577.1 | 2.24x | 2.21x |
| qwen3-8b gate_up (24576 x 4096) | 128 | bias_gelu | 538.4 | 538.6 | 1.000 | 596.9 | 732.3 | 726.7 | 1.36x | 1.35x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | bias_gelu | 2330.6 | 2376.5 | 1.020 | 2566.0 | 3167.7 | 3180.0 | 1.33x | 1.34x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | bias_gelu | 9431.2 | 9421.5 | 0.999 | 10364.5 | 13340.7 | 13207.8 | 1.42x | 1.40x |
| llama3-8b gate_up (28672 x 4096) | 16 | bias_gelu | 302.2* | 301.7 | 0.998 | 505.7 | 602.5 | 610.4 | 2.00x | 2.02x |
| llama3-8b gate_up (28672 x 4096) | 128 | bias_gelu | 587.9 | 592.1 | 1.007 | 646.2 | 904.9 | 895.5 | 1.53x | 1.51x |
| llama3-8b gate_up (28672 x 4096) | 1024 | bias_gelu | 2714.6 | 2755.0 | 1.015 | 3010.5 | 3726.4 | 3715.6 | 1.35x | 1.35x |
| llama3-8b gate_up (28672 x 4096) | 4096 | bias_gelu | 11004.7 | 11022.2 | 1.002 | 12142.5 | 15645.3 | 15490.7 | 1.42x | 1.41x |
| qwen3-8b o_proj (4096 x 4096) | 16 | residual | 25.4* | 24.9 | 0.981 | 45.7 | 100.0 | 96.9 | 4.02x | 3.90x |
| qwen3-8b o_proj (4096 x 4096) | 128 | residual | 103.8 | 104.1 | 1.003 | 111.9 | 147.9 | 148.4 | 1.42x | 1.43x |
| qwen3-8b o_proj (4096 x 4096) | 1024 | residual | 436.0 | 444.4 | 1.019 | 450.7 | 627.6 | 621.5 | 1.41x | 1.40x |
| qwen3-8b o_proj (4096 x 4096) | 4096 | residual | 1560.8 | 1596.7 | 1.023 | 1674.2 | 2102.4 | 2120.0 | 1.32x | 1.33x |
| qwen3-8b down (4096 x 12288) | 16 | residual | 59.2* | 56.2 | 0.949 | 123.2 | 243.5 | 237.2 | 4.33x | 4.22x |
| qwen3-8b down (4096 x 12288) | 128 | residual | 246.4 | 241.8 | 0.981 | 256.0 | 370.0 | 369.1 | 1.53x | 1.53x |
| qwen3-8b down (4096 x 12288) | 1024 | residual | 1306.1 | 1286.9 | 0.985 | 1331.3 | 1817.7 | 1817.4 | 1.41x | 1.41x |
| qwen3-8b down (4096 x 12288) | 4096 | residual | 4765.2 | 4793.6 | 1.006 | 5015.2 | 6141.4 | 6232.6 | 1.28x | 1.30x |
| llama3-8b down (4096 x 14336) | 16 | residual | 158.1* | 157.7 | 0.997 | 188.1 | 289.3 | 285.1 | 1.84x | 1.81x |
| llama3-8b down (4096 x 14336) | 128 | residual | 304.5 | 296.7 | 0.975 | 318.3 | 408.8 | 408.9 | 1.38x | 1.38x |
| llama3-8b down (4096 x 14336) | 1024 | residual | 1505.0 | 1485.5 | 0.987 | 1529.2 | 2120.9 | 2117.9 | 1.43x | 1.43x |
| llama3-8b down (4096 x 14336) | 4096 | residual | 5675.3 | 5594.2 | 0.986 | 5795.8 | 7629.2 | 7574.8 | 1.36x | 1.35x |
| qwen3-8b qkv (6144 x 4096) | 16 | residual | 32.6* | 32.9 | 1.011 | 49.2 | 138.6 | 132.2 | 4.21x | 4.01x |
| qwen3-8b qkv (6144 x 4096) | 128 | residual | 99.9 | 99.1 | 0.992 | 116.0 | 188.7 | 194.2 | 1.90x | 1.96x |
| qwen3-8b qkv (6144 x 4096) | 1024 | residual | 589.9 | 599.9 | 1.017 | 606.2 | 851.2 | 846.3 | 1.42x | 1.41x |
| qwen3-8b qkv (6144 x 4096) | 4096 | residual | 2279.4 | 2356.2 | 1.034 | 2521.5 | 3112.1 | 3190.0 | 1.32x | 1.35x |
| qwen3-8b gate_up (24576 x 4096) | 16 | residual | 260.4* | 261.6 | 1.005 | 421.4 | 590.8 | 578.6 | 2.26x | 2.21x |
| qwen3-8b gate_up (24576 x 4096) | 128 | residual | 536.5 | 474.9 | 0.885 | 581.8 | 745.2 | 726.6 | 1.57x | 1.53x |
| qwen3-8b gate_up (24576 x 4096) | 1024 | residual | 2334.7 | 2386.7 | 1.022 | 2555.4 | 3207.3 | 3253.0 | 1.34x | 1.36x |
| qwen3-8b gate_up (24576 x 4096) | 4096 | residual | 9595.7 | 9883.5 | 1.030 | 10297.8 | 13374.0 | 13280.5 | 1.35x | 1.34x |
| llama3-8b gate_up (28672 x 4096) | 16 | residual | 302.9* | 303.0 | 1.001 | 487.5 | 601.9 | 616.7 | 1.99x | 2.04x |
| llama3-8b gate_up (28672 x 4096) | 128 | residual | 593.2 | 530.2 | 0.894 | 628.8 | 904.1 | 901.4 | 1.71x | 1.70x |
| llama3-8b gate_up (28672 x 4096) | 1024 | residual | 2712.1 | 2726.0 | 1.005 | 2977.0 | 3731.3 | 3789.1 | 1.37x | 1.39x |
| llama3-8b gate_up (28672 x 4096) | 4096 | residual | 11023.0 | 11043.5 | 1.002 | 11866.6 | 15775.5 | 15747.5 | 1.43x | 1.43x |



## Limits

- A and B are bf16, B is `(N, K)`; C bf16, fp16 or fp32, and on gfx950 also
  int8 and fp8 (`torch.float8_e4m3fn`, `float8_e5m2`) -- include/rdna3's
  tile store is 16- and 32-bit only. An fp8 C saturates at +-448 (e4m3),
  where torch's own cast gives NaN.
- N a multiple of 128 and K of 64; M anything.
- A column-vector operand is read by an unbounded vector load, so when M is
  not a multiple of 256 the template pads a copy of it (one small extra
  launch a call).
- An epilogue too big to fit beside a kernel's registers (the build refuses
  any spill) is not an error: the template warns, stores the fp32 sum and
  runs the epilogue in a pass of its own (the split-K reduction, of one
  share) -- right, at the cost of a second pass over C.
- The epilogue sees a tile, not a row: a reduction across C's columns (a
  row-wise absmax for per-token quantization) is not an epilogue -- every
  tile holds part of a row.
- GELU is the tanh form only.
- The template is as fast as the GEMM it takes: on gfx950 a shape outside
  `routes_gemm_gfx950.json` gets `route()`'s rule, which is not always the
  best kernel (2048 x 16384 x 4096 runs the four-warp `dp128` at 0.76x
  hipBLASLt), and the template inherits it. Tuning the route
  (`tools/hk-bench/gemm_cdna_routes.py`) helps both.
- On gfx950, an epilogue's arithmetic at the end of the pingpong overlaps
  nothing (every wave of the CU stores at once), so its cost is its
  instruction count: GELU(tanh) is ~3.5% of a large GEMM, and 5-8% at
  M = 1024 on the widest layers, where the grid is 1.5-2 waves of workgroups
  and the epilogue is paid per wave.
