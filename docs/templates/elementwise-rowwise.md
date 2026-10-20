# Elementwise and rowwise templates

`hk.templates.elementwise` and `hk.templates.rowwise` build a complete,
tuned kernel from the part that is yours: the math of one element, or of
one row. The template owns the layout, the per-GPU plan, ragged shapes,
strided views and the launch; you write a few lines of hook and call the
result with torch tensors.

```python
import hk
from hk import templates as T

gelu_and_mul = T.elementwise(lambda g, u: hk.math.gelu(g) * u)
y = gelu_and_mul(gate_up[:, :d], gate_up[:, d:])     # the halves, read in place

@T.rowwise
def rms_norm(x, w, *, EPS=1e-6):
    return x * (T.row_mean(x * x) + EPS).rsqrt() * w

y = rms_norm(x, weight)                               # weight: one value per column
```

[`examples/12_templates_rowwise.py`](../../examples/12_templates_rowwise.py)
runs five of them against torch. The rest of this page is what a hook may
do, what the templates decide for you, how to see and override that, and
what they measured.

- [elementwise](#elementwise)
- [rowwise](#rowwise)
- [operands](#operands)
- [the plan, and overriding it](#the-plan-and-overriding-it)
- [without a GPU](#without-a-gpu)
- [measured](#measured)
- [limits](#limits)

## elementwise

```python
k = T.elementwise(fn, inputs=None, outputs=None, *, plan=None, arch=None, name=None)
out = k(*tensors, out=None, **constexprs)
```

`fn` takes one tile per operand and returns a tile, or a tuple of tiles for
several outputs. The tiles are **fp32 register strips** of a stretch of one
row: loads widen from the tensor's dtype, stores round to nearest even into
the output's, so `lambda a, b: a + b` is torch's bf16 add bit for bit.
Everything in [the language](../language.md#values-and-operators) that maps
elements works: `+ - * /`, `hk.math.exp / log / gelu / silu / rsqrt /
relu / abs`, `hk.max(a, b)`, `t.exp()`, a number on either side (`1.0 - p`,
`127.0 / t`), `abs(t)`, `t.step()`.

```python
silu_mul = T.elementwise(lambda g, u: hk.math.silu(g) * u)
bias_gelu = T.elementwise(lambda x, b: hk.math.gelu(x + b))          # b: (cols,)
scaled = T.elementwise(lambda x, s, *, A=1.0: x * s * A)             # s: (rows, 1)
split = T.elementwise(lambda x: (x * 2.0, x + 1.0), outputs=(hk.fp32, hk.i8))
```

- **constexprs**: the hook's parameters with defaults (`A=1.0`) are baked
  in; a call may override them (`scaled(x, s, A=0.5)`), which builds that
  variant once.
- **outputs**: the first full operand's dtype by default; a dtype, a tuple
  of dtypes, or an int (`outputs=2`, two of the default dtype). An int8
  output saturates and rounds to nearest even.
- **out=**: a tensor (or a tuple) to write into. Exactly in place
  (`out=x`) is safe; an `out` that partially overlaps an input is handled by
  copying the inputs first.

A hook that calls `t.sum(axis=1)` is refused while tracing: on a strip that
reduces one warp's share of a row, which is a wrong answer. Reductions are
`rowwise`'s.

## rowwise

```python
k = T.rowwise(fn, inputs=None, outputs=None, *, like=None, plan=None, arch=None, name=None)
outs = k(*tensors, out=None, **constexprs)
```

`fn` is the math of one row. A full or per-column operand arrives as an
fp32 strip of the row (the warp's share of it); inside the hook,

| call | gives |
|---|---|
| `T.row_sum(t)`, `T.row_max(t)`, `T.row_min(t)` | the whole row's sum / max / min |
| `T.row_mean(t)` | `row_sum(t) / row_length()` |
| `T.row_logsumexp(t)` | `log(sum(exp(t)))`, max-shifted, online (one read of a streamed row) |
| `T.pick(t, col)` | operand `t`'s element at column `col` of this row (a gather; `col` clamped into the row) |
| `T.row_length()` | the row's length, a device int |

reduce **the whole row** -- across the lanes and warps that hold it, a
ragged tail masked with the reduction's identity -- and give a column
vector: one value for the row, which broadcasts against strips on either
side (`x - m`, `m - x`, `x / s`, `s * x`). A division by a row value is a
multiplication by its reciprocal (one division a row).

The hook returns a tile (a full output) or a column vector (a per-row
output: a scale, a loss), or a tuple; per-row outputs are declared:

```python
@T.rowwise(outputs=2)                                   # y and the new residual
def add_rms_norm(x, r, w, *, EPS=1e-6):
    h = x + r
    return h * (T.row_mean(h * h) + EPS).rsqrt() * w, h

y, r = add_rms_norm(x, r, w, out=(x, r))                # vLLM's, in place

@T.rowwise
def softmax(x):
    e = (x - T.row_max(x)).exp()
    return e / T.row_sum(e)

@T.rowwise
def layernorm(x, *, EPS=1e-5):
    xc = x - T.row_mean(x)
    return xc * (T.row_mean(xc * xc) + EPS).rsqrt()

@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32)))         # per-token int8
def quantize(x):
    amax = T.row_max(abs(x)).maximum(1e-12)
    return x * (127.0 / amax), amax / 127.0

@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32), hk.bf16))  # add + RMSNorm + int8
def add_rms_quant(x, r, w, *, EPS=1e-6):
    h = x + r
    y = h * (T.row_mean(h * h) + EPS).rsqrt() * w
    amax = T.row_max(abs(y)).maximum(1e-10)
    return y * (127.0 / amax), amax / 127.0, h

@T.rowwise(outputs=T.per_row(hk.fp32))                  # cross-entropy, any vocabulary
def cross_entropy(logits, label):                       # label: (rows,) int64 or int32
    loss = T.row_logsumexp(logits) - T.pick(logits, label)
    return loss * (label != -100).to_int()              # ignore_index -> 0
```

A per-row int32 or int64 operand (a label) arrives as a device int; an int64
tensor is read in place, as its low 32 bits (every vocabulary index, and
-100). A per-row float operand arrives as a column vector.

As in `elementwise`, `t.sum(axis=1)` inside a hook is refused with the
spelling that works.

## operands

Each operand is one of four kinds, inferred from the tensors by torch's
broadcasting against the largest operand, or declared with `inputs=`:

| kind | shape | declare | in the hook |
|---|---|---|---|
| full | the problem's `(..., cols)` | a dtype, `hk.bf16` | a strip |
| per column | `(cols,)` | `T.per_col(hk.bf16)` | a strip (the row's stretch of it) |
| per row | `(...,)` or `(..., 1)` | `T.per_row(hk.fp32)` | elementwise: a strip; rowwise: a column vector (an int: a scalar) |
| one value | `()` or `(1,)` on the device | `T.scalar(hk.fp32)` | as per row |

A full operand may be any view whose last axis is contiguous and whose
leading axes are one row stride apart: a slice of the last axis (one half
of a fused gate/up output), a batched `(B, S, D)` tensor, a 3-D slice. Rows
and columns need not divide anything: a ragged stretch is read and written
with bounded accesses. Declared operands are checked (`inputs=(hk.bf16,
T.per_col(hk.bf16))`); bf16, fp16 and fp32 go in, and int8 and int32 come
out of an elementwise kernel too.

## the plan, and overriding it

`k.plan(*tensors)` says what a call runs:

```
>>> gelu_and_mul.plan(x[:, :14336], x[:, 14336:])
rows 4096x14336 on gfx950: 1 warp(s) x 1 chunk(s) of 64x8, grid 114688, exact
>>> rms_norm.plan(x, w)
8192x8192 like rmsw on gfx950: one workgroup a row, 4 warp(s) x 4 chunk(s) of 64x8 (32 a lane), held, read once, exact, 53 KB LDS reserved
>>> cross_entropy.plan(logits, labels)
64x151936 like softmax on gfx950: one workgroup a row, 8 warp(s) x 4 chunk(s) of 64x8 (32 a lane), 10 steps a pass, bounded
```

A **chunk** is the wave's lanes (64 on gfx950, 32 on gfx1100) times the
**run**, the contiguous elements one lane reads: 16 bytes where the row's
length keeps the runs aligned (8 bf16, 4 fp32), halved until it does. The
plans are hk.ops' measured ones, reached through hk.ops' own functions
rather than copied:

- **elementwise, operands that are views of rows** (the halves of a
  gate/up output, a broadcast operand): `hk.ops.fused.silu_strip_plan`,
  silu_mul's -- consecutive workgroups walk along a row; on gfx950 one warp
  of one chunk a workgroup, on gfx1100 eight, with 48 KB of LDS reserved
  (two workgroups a WGP) once a call moves more than the 96 MB Infinity
  Cache.
- **elementwise, all operands contiguous** on gfx950: one flat row,
  `hk.ops.elementwise.flat_strip_plan` by the bytes moved.
- **rowwise, a row that fits in registers**: one workgroup a row, the row
  read once, with `hk.ops.norm.strip_plan_for` (and
  `hk.ops.quant.strip_plan_for` for a quantization): the warps, the chunks
  a warp holds, gfx950's per-lane targets by width and row count, and the
  occupancy cap past the cache. Which hk.ops kernel's plan (`like`) follows
  from the hook: two full inputs and two full outputs plan like the fused
  add; an int8 output like quantize; a `row_max` or `row_logsumexp` like
  softmax; a per-column operand like the weighted RMSNorm; two row sums
  like LayerNorm; else like RMSNorm. `like="softmax"` names it.
- **rowwise, a row too wide to hold** (past 16 warps x 64 elements a lane:
  65536 columns on gfx950, 32768 on gfx1100), or a ragged row wider than
  8192 that would hold more than 32 a lane: **multi-pass**. The row
  streams through a runtime loop in steps of warps x chunks; each pass
  folds one reduction the hook needs (a dry trace of the hook finds which),
  and a last pass writes the full outputs. `row_logsumexp` is online, so a
  cross-entropy reads its logits once and a softmax written as
  `(x - T.row_logsumexp(x)).exp()` twice.
- A build the resource gate refuses (it would spill) steps down by itself:
  the same row over twice the warps, then the stream.

Override any of it: `k.with_plan(warps=4, chunks=2)` or `T.rowwise(...,
plan={"warps": 4, "chunks": 2})`; elementwise takes `warps`, `chunks`,
`run`, `flat`, `reserve`; rowwise `warps`, `chunks`, `run`, `reserve`,
`steps` (a stream of that many steps) and `stream=True`.

## without a GPU

`hk.templates.meta(shape, dtype, stride=None)` stands in for a tensor, and
`k.kernel(*metas)` returns the `hk.Kernel` and its constexprs:

```python
B = hk.bf16
kern, consts = gelu_and_mul.kernel(T.meta((4096, 14336), B, stride=(28672, 1)),
                                   T.meta((4096, 14336), B, stride=(28672, 1)))
print(kern.source(**consts))        # the HipKittens C++
kern.build(**consts)                # hipcc and the resource gates, no GPU
```

`T.elementwise(..., arch="gfx1100")` plans and builds for a GPU that is not
in the machine.

## measured

<!-- MEASURED -->

## limits

- The templates run on register strips, which gfx1100 and gfx950 have;
  gfx942 and gfx1201 refuse them while planning.
- A hook is traced, not rewritten: a Python `if` on a device value inside
  it raises. Arithmetic selects (`(label != -100).to_int()`, `t.step()`)
  do what a branch would.
- Elementwise runs of a ragged flat tensor whose length is odd are one
  element (aligned accesses); a tensor past 2 GiB must divide the plan
  exactly.
- `pick` reads one element per row; it is a gather, not a lookup table.
- A rowwise hook's per-row outputs are fp32 (or any dtype `store_scalar`
  takes); its full outputs any dtype.
- The rows of a rowwise call are one workgroup each: a decode step's
  handful of rows of 4096 is a handful of workgroups, at the launch floor
  like every other arm.
