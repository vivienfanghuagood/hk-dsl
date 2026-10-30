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
do, what the templates decide for you, and how to see and override that. A
dated appendix keeps what they measured when they landed.

- [elementwise](#elementwise)
- [rowwise](#rowwise)
- [operands](#operands)
- [the plan, and overriding it](#the-plan-and-overriding-it)
- [without a GPU](#without-a-gpu)
- [hk.ops on the same plans](#hkops-on-the-same-plans)
- [limits](#limits)
- [appendix: measured at hk 0.4.0](#appendix-measured-at-hk-040)

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

def router(k):                                          # MoE routing: softmax, top-k, renormalise
    @T.rowwise(outputs=(T.per_row(hk.i32, k), T.per_row(hk.fp32, k)))
    def route(logits, col):                             # col: arange(E), fp32, made once
        p = (logits - T.row_logsumexp(logits)).exp()
        ids, ws = [], []
        for _ in range(k):                              # a Python loop: unrolled while tracing
            m = T.row_max(p)
            i = T.row_min(col + (m - p).step() * 1e9)   # the first column holding the max
            ids.append(i)
            ws.append(m)
            p = p - (1.0 - (col - i).abs().step()) * 2.0
        s = ws[0]
        for w in ws[1:]:
            s = s + w
        return ids, [w / s for w in ws]
    return route
```

A per-row int32 or int64 operand (a label) arrives as a device int; an int64
tensor is read in place, as its low 32 bits (every vocabulary index, and
-100). A per-row float operand arrives as a column vector. A per-row output
of `k` values a row, `T.per_row(dtype, k)`, is a `(rows, k)` tensor, the hook
returning a list of `k` row values for it.

An output computed before the hook's first row reduction, and not written
after it, is stored where that reduction begins (the fused add's `x + r`):
held to the end it would stay live across the reductions, registers an
in-place kernel gives back.

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

## hk.ops on the same plans

The ops the templates reproduce take the same arguments on both GPUs now
(the cross-architecture audit in docs/dx/api-consistency.md listed where they
did not):

| op | before, on gfx1100 | now, on both |
|---|---|---|
| `silu_mul` | 16x64 tiles: contiguous `a`, `b`, at least 16x64 | strips: the halves of a `(M, 2I)` gate/up output as views, any shape; `out=` checked |
| `add` `sub` `mul` `maximum` `minimum` `exp` `relu` `gelu` `neg` `abs` | the shape had to tile (`100x1000` refused) | strips over a flat view, any size below 2 GiB (2048x4096 add: 23.6 us, the tile kernel 38.9, torch 23.6) |
| `quantize` | at least 16x64 | strips, any row count (an odd row into int8 keeps the tile plan on gfx1100: a one-byte run) |
| `rotary` | `NotImplementedError` ("use rope", other arguments) | vLLM's `cos_sin` rows and optional positions, in place; on gfx1100 a head's halves are two strips, D = 64 x 1-8 |

Still different, and why: the norms hold a row in registers up to 64
values a lane on gfx1100 (32768 columns) and up to 128 on gfx950, and past
that `hk.ops` falls back to its tile kernels (at least 16x64); the rowwise
template streams such a row instead, on both. `gelu` is fp32-only in both
`hk.ops` tile and strip tables (HipKittens specialises it for float).

## limits

- The templates run on register strips, which gfx1100, gfx1201 and gfx950
  have; gfx942 has none and refuses them while planning.
- A hook is traced, not rewritten: a Python `if` on a device value inside
  it raises. Arithmetic selects (`(label != -100).to_int()`, `t.step()`)
  do what a branch would.
- A row whose length is odd is read in runs of one element (aligned
  accesses), and on gfx1100 such a row cannot go into an int8 output (a
  one-byte run, which its strip store does not move): the template says so.
  A tensor past 2 GiB must divide the plan exactly.
- On gfx1100 a template kernel is launched on the null stream (RDNA's
  launcher takes none), so a CUDA graph does not capture it; on gfx950 it
  runs on torch's current stream and is captured.
- Python overhead: a call builds a key of its tensors' shapes, strides and
  dtypes and checks `out=` against the inputs. At the 1-32 row launch floor
  that is 0.3-1 us more than an `hk.ops` call in eager mode (the kernels'
  own times are equal; see the [appendix](#appendix-measured-at-hk-040)).
- `pick` reads one element per row; it is a gather, not a lookup table.
- A rowwise hook's per-row outputs are fp32 (or any dtype `store_scalar`
  takes); its full outputs any dtype.
- The rows of a rowwise call are one workgroup each: a decode step's
  handful of rows of 4096 is a handful of workgroups, at the launch floor
  like every other arm.

## Appendix: measured at hk 0.4.0

> **A dated record, not current claims.** These measurements were taken when
> the templates landed (October 2026: hk 0.4.0, torch 2.10, Triton 3.6) and
> have not been re-run against current baselines. The only performance
> numbers this project quotes are on the [scorecard](../performance/scorecard.md).
> `python3 -m hk.check` times a template against `torch.compile` on your GPU.

Every number below was taken in one process per GPU, the arms interleaved
(rounds rotate the order, each round queued behind a GPU spin, the median
of 11 rounds of 30 calls; `tools/hk-bench/templates_table.py`); the hook is
the one on this page, `hk.ops` is the tuned kernel
for the same job, `torch.compile` is the default mode, `dynamic=False`, one
compiled function per shape (and Dynamo's recompile limit raised: past it a
compiled function silently falls back to eager). Ratios are `other / this`:
above 1 the template is faster.

- **MI355X (gfx950)**, rocm/pytorch 2.10.0+rocm7.2.3, Triton 3.6.0, hk 0.4.0
  at this branch: **device time**, each arm's calls replayed from a CUDA
  graph (the host's launch cost removed, as a serving engine's captured step
  sees it).
- **W7900D (gfx1100)**, torch 2.10.0+rocm7.2.4, Triton 3.6.0: **eager**
  back-to-back calls, host overhead included. RDNA's launcher takes no
  stream, so a CUDA graph does not capture an hk kernel there (a replay
  times nothing); at 1-32 rows these numbers are the launch path, not the
  kernel.

Summary (rows 1-32 "small", 2048-8192 "large"; widths 4096/5120/8192, the
gate/up ops at intermediate 4096 and 14336 on the halves of a `(M, 2I)`
output):

| op (user lines) | MI355X 1-32 rows: x hk.ops / x torch.compile | MI355X 2048-8192 rows | W7900D 1-32 rows | W7900D 2048-8192 rows |
|---|---|---|---|---|
| silu_mul on the halves (1) | 0.99-1.00 / 1.06-1.11 | 1.00-1.02 / 1.00-1.07 | 1.00-1.02 / 3.6-4.2 | 1.00-1.01 / 1.00-1.19 |
| gelu_tanh_and_mul on the halves (1) | -- / 0.99-1.04 | -- / 1.01-1.05 | -- / 3.3-4.1 | -- / 1.01-1.31 |
| rms_norm (3) | 1.00-1.03 / 1.01-1.09 | 0.98-1.00 / 1.01-1.19 | 1.00 / 3.7-4.1 | 1.00-1.01 / 1.04-1.12 |
| fused_add_rms_norm, in place (4) | 0.99-1.00 / 0.99-1.14 | 1.00 / 1.06-1.57 | 0.92-0.99 / 4.1-4.4 | 0.93-1.02 / 1.01-2.61 |
| softmax (4) | 1.00 / 1.46-2.28 | 1.00 / 2.75-3.19 | 1.00 / 3.6-4.1 | 0.99-1.00 / 1.26-3.16 |
| int8 per-token quantize (4) | 0.99 / 0.99-1.23 | 0.99-1.00 / 1.06-1.38 | 0.99-1.00 / 4.1-4.8 | 0.94-1.00 / 1.01-2.12 |

On the W7900D the 1-32 row columns are eager calls: torch.compile's guards
cost it 15-20 us there, which is why it reads 3-4x; the kernels themselves
are at the launch floor.

The DX suite ([bench/dx](../../bench/dx/README.md), W1's harness, the
template solutions in `bench/dx/solutions/*/template.py`), gfx950, device
time, geomean over the task's shapes (min-max):

| task (template lines) | MI355X x torch.compile | MI355X x best baseline | W7900D (`--quick`, 3 shapes) x torch.compile | W7900D x best baseline |
|---|---|---|---|---|
| T1 gelu_tanh_and_mul (7) | 1.11 (1.03-1.63) | 1.00 | 1.03 (0.99-1.10) | 0.98 |
| T2 add + RMSNorm + int8 (17) | 1.45 (1.05-1.83) | 1.21; 1.48 x hk.ops (two kernels) | 1.17 (1.04-1.30) | 1.10; 1.67 x hk.ops |
| T3 cross-entropy, V up to 151936 (10) | 2.50 (1.24-3.43) | 1.76 | 1.58 (1.14-2.88) | 1.22 |
| T8 router softmax + top-k (28) | 9.06 (6.90-13.55) | 1.11 | 7.37 (6.09-9.86) | 1.34 |

The best baseline is the fastest of torch eager, torch.compile, Triton and
the hk.ops kernels the task has; `docs_only` solutions are contenders, not
baselines (T3's docs-only kernel, 116 lines, splits a row over workgroups at
64 rows and is 1.6x the template there; at 512 rows and up they tie). On the
W7900D the harness launches hk through code objects on torch's stream so
that CUDA graphs capture it (bench/dx/README.md).

What misses the targets (>= 0.95x hk.ops, >= 1.0x torch.compile at large
shapes), and why:

- **Host overhead at 1-32 rows, eager.** A template call builds a key of
  its tensors' shapes, strides and dtypes and checks `out=` against the
  inputs: 0.3-1 us more Python than an `hk.ops` call. On the MI355X that
  makes an eager in-place fused add 0.86-0.89x hk.ops at 1-32 rows (8.2
  against 7.1 us) where the kernels tie (2.16 against 2.14 us, graphed); on
  the W7900D, where launches are cheaper, 0.92-0.99x. Under a CUDA graph it
  is gone.
- **The launch floor, MI355X.** gelu at 1x4096, the fused add at 1-32 rows
  and quantize at 1x4096 are 0.99x torch.compile: 2.10-2.16 us both ways.
- **The fused add at 2048x8192 on the W7900D, 0.93x hk.ops** (79.2 against
  73.4 us; 2.61x torch.compile): the same plan (16 warps of two chunks, 48 KB
  reserved), the residual stored before the reduction as hk.ops stores it,
  the weight loaded first. Not found; 2048x4096 and 2048x5120 tie.
- **int8 quantize at 2048 rows on the W7900D, 0.94-0.98x hk.ops** (16.3
  against 15.4 us at 4096 columns), hk.ops' strip quantize at the same plan:
  the hook's `x * (127 / amax)` is a new tile where hk.ops rescales in place.
- **T1 against Triton at the cache-resident sizes**, MI355X: 0.95-0.99x at
  128-4096 tokens of 9216-21504 (3.42 against 3.25 us at 128x14336, 14.75
  against 14.08 at 1024x14336); 1.03-1.63x torch.compile everywhere, and
  1.06x Triton's 74.0 us at 4096x14336. W7900D: 0.99x torch.compile at
  4096x14336 (515 against 508 us).
- **T8, the router, against Triton from 4096 tokens**: 0.56-0.92x on the
  MI355X (17.4 against 9.8 us at 16384x128; 0.81x at 4096x8): each of the k
  rounds is two row reductions (the max, then the first column holding it),
  where Triton's argmax is one. 8.5-13.6x torch.compile there, and 1.8-1.9x
  Triton at 1-256 tokens of 128 experts; on the W7900D 1.05-2.05x Triton at
  its three `--quick` shapes.
- **Strips only.** gfx942 and gfx1201 have none, and the templates refuse
  them. *(Correction, 2026-10-30: gfx1201 has strips, and runs the
  templates, since the RDNA4 port (#71); only gfx942 has none.)*

<details><summary>Every case, gfx950 (device time, us)</summary>

| op | shape | template us | hk.ops us | torch.compile us | x hk.ops | x torch.compile |
|---|---|---:|---:|---:|---:|---:|
| silu_mul (halves) | 1x2*4096 | 1.95 | 1.94 | 2.17 | 0.99 | 1.11 |
| silu_mul (halves) | 32x2*4096 | 2.12 | 2.11 | 2.27 | 1.00 | 1.07 |
| silu_mul (halves) | 2048x2*4096 | 8.56 | 8.57 | 9.15 | 1.00 | 1.07 |
| silu_mul (halves) | 8192x2*4096 | 29.00 | 29.48 | 29.76 | 1.02 | 1.03 |
| silu_mul (halves) | 1x2*14336 | 2.08 | 2.08 | 2.25 | 1.00 | 1.08 |
| silu_mul (halves) | 32x2*14336 | 2.17 | 2.18 | 2.30 | 1.00 | 1.06 |
| silu_mul (halves) | 2048x2*14336 | 25.50 | 25.58 | 26.68 | 1.00 | 1.05 |
| silu_mul (halves) | 8192x2*14336 | 117.29 | 117.16 | 116.95 | 1.00 | 1.00 |
| gelu_tanh_and_mul (halves) | 1x2*4096 | 2.12 |  | 2.10 |  | 0.99 |
| gelu_tanh_and_mul (halves) | 32x2*4096 | 2.19 |  | 2.24 |  | 1.02 |
| gelu_tanh_and_mul (halves) | 2048x2*4096 | 9.81 |  | 9.95 |  | 1.01 |
| gelu_tanh_and_mul (halves) | 8192x2*4096 | 30.07 |  | 31.34 |  | 1.04 |
| gelu_tanh_and_mul (halves) | 1x2*14336 | 2.13 |  | 2.21 |  | 1.04 |
| gelu_tanh_and_mul (halves) | 32x2*14336 | 2.30 |  | 2.37 |  | 1.03 |
| gelu_tanh_and_mul (halves) | 2048x2*14336 | 26.65 |  | 28.01 |  | 1.05 |
| gelu_tanh_and_mul (halves) | 8192x2*14336 | 117.89 |  | 120.04 |  | 1.02 |
| rms_norm | 1x4096 | 2.01 | 2.02 | 2.10 | 1.00 | 1.04 |
| rms_norm | 32x4096 | 2.21 | 2.21 | 2.23 | 1.00 | 1.01 |
| rms_norm | 2048x4096 | 5.23 | 5.24 | 5.61 | 1.00 | 1.07 |
| rms_norm | 8192x4096 | 19.78 | 19.79 | 20.43 | 1.00 | 1.03 |
| rms_norm | 1x5120 | 2.05 | 2.06 | 2.11 | 1.00 | 1.03 |
| rms_norm | 32x5120 | 2.26 | 2.25 | 2.47 | 1.00 | 1.09 |
| rms_norm | 2048x5120 | 6.44 | 6.44 | 7.66 | 1.00 | 1.19 |
| rms_norm | 8192x5120 | 24.91 | 24.92 | 25.26 | 1.00 | 1.01 |
| rms_norm | 1x8192 | 2.05 | 2.11 | 2.11 | 1.03 | 1.03 |
| rms_norm | 32x8192 | 2.30 | 2.30 | 2.41 | 1.00 | 1.05 |
| rms_norm | 2048x8192 | 10.58 | 10.37 | 12.06 | 0.98 | 1.14 |
| rms_norm | 8192x8192 | 38.24 | 38.24 | 39.96 | 1.00 | 1.04 |
| fused_add_rms_norm | 1x4096 | 2.16 | 2.14 | 2.14 | 0.99 | 0.99 |
| fused_add_rms_norm | 32x4096 | 2.31 | 2.31 | 2.33 | 1.00 | 1.01 |
| fused_add_rms_norm | 2048x4096 | 7.75 | 7.75 | 12.14 | 1.00 | 1.57 |
| fused_add_rms_norm | 8192x4096 | 38.99 | 39.08 | 41.28 | 1.00 | 1.06 |
| fused_add_rms_norm | 1x5120 | 2.26 | 2.27 | 2.41 | 1.00 | 1.07 |
| fused_add_rms_norm | 32x5120 | 2.46 | 2.47 | 2.80 | 1.00 | 1.14 |
| fused_add_rms_norm | 2048x5120 | 11.99 | 12.03 | 15.13 | 1.00 | 1.26 |
| fused_add_rms_norm | 8192x5120 | 48.96 | 48.94 | 60.47 | 1.00 | 1.24 |
| fused_add_rms_norm | 1x8192 | 2.31 | 2.29 | 2.36 | 0.99 | 1.02 |
| fused_add_rms_norm | 32x8192 | 2.52 | 2.53 | 2.73 | 1.00 | 1.08 |
| fused_add_rms_norm | 2048x8192 | 19.61 | 19.66 | 24.07 | 1.00 | 1.23 |
| fused_add_rms_norm | 8192x8192 | 77.32 | 77.46 | 103.52 | 1.00 | 1.34 |
| softmax | 1x4096 | 2.57 | 2.57 | 3.76 | 1.00 | 1.46 |
| softmax | 32x4096 | 2.71 | 2.71 | 4.22 | 1.00 | 1.56 |
| softmax | 2048x4096 | 6.28 | 6.25 | 18.13 | 1.00 | 2.89 |
| softmax | 8192x4096 | 20.46 | 20.46 | 60.20 | 1.00 | 2.94 |
| softmax | 1x5120 | 2.47 | 2.48 | 4.72 | 1.00 | 1.91 |
| softmax | 32x5120 | 2.68 | 2.68 | 5.07 | 1.00 | 1.89 |
| softmax | 2048x5120 | 7.14 | 7.14 | 22.79 | 1.00 | 3.19 |
| softmax | 8192x5120 | 25.40 | 25.39 | 80.33 | 1.00 | 3.16 |
| softmax | 1x8192 | 2.45 | 2.45 | 5.59 | 1.00 | 2.28 |
| softmax | 32x8192 | 2.66 | 2.66 | 6.07 | 1.00 | 2.28 |
| softmax | 2048x8192 | 11.25 | 11.25 | 30.91 | 1.00 | 2.75 |
| softmax | 8192x8192 | 38.09 | 38.13 | 110.36 | 1.00 | 2.90 |
| int8 quantize | 1x4096 | 2.16 | 2.14 | 2.14 | 0.99 | 0.99 |
| int8 quantize | 32x4096 | 2.43 | 2.41 | 2.64 | 0.99 | 1.09 |
| int8 quantize | 2048x4096 | 4.85 | 4.82 | 6.56 | 0.99 | 1.35 |
| int8 quantize | 8192x4096 | 15.78 | 15.65 | 19.78 | 0.99 | 1.25 |
| int8 quantize | 1x5120 | 2.17 | 2.15 | 2.57 | 0.99 | 1.18 |
| int8 quantize | 32x5120 | 2.58 | 2.56 | 2.97 | 0.99 | 1.15 |
| int8 quantize | 2048x5120 | 5.81 | 5.76 | 8.04 | 0.99 | 1.38 |
| int8 quantize | 8192x5120 | 19.48 | 19.39 | 24.19 | 1.00 | 1.24 |
| int8 quantize | 1x8192 | 2.20 | 2.18 | 2.60 | 0.99 | 1.18 |
| int8 quantize | 32x8192 | 2.48 | 2.46 | 3.06 | 0.99 | 1.23 |
| int8 quantize | 2048x8192 | 8.55 | 8.49 | 10.76 | 0.99 | 1.26 |
| int8 quantize | 8192x8192 | 30.34 | 30.29 | 32.27 | 1.00 | 1.06 |

</details>

<details><summary>Every case, gfx1100 (eager, us)</summary>

| op | shape | template us | hk.ops us | torch.compile us | x hk.ops | x torch.compile |
|---|---|---:|---:|---:|---:|---:|
| silu_mul (halves) | 1x2*4096 | 3.92 | 3.96 | 16.35 | 1.01 | 4.17 |
| silu_mul (halves) | 32x2*4096 | 4.03 | 4.06 | 16.18 | 1.01 | 4.01 |
| silu_mul (halves) | 2048x2*4096 | 24.09 | 24.37 | 28.55 | 1.01 | 1.19 |
| silu_mul (halves) | 8192x2*4096 | 273.95 | 273.97 | 275.15 | 1.00 | 1.00 |
| silu_mul (halves) | 1x2*14336 | 3.93 | 4.01 | 15.81 | 1.02 | 4.02 |
| silu_mul (halves) | 32x2*14336 | 4.50 | 4.51 | 16.05 | 1.00 | 3.57 |
| silu_mul (halves) | 2048x2*14336 | 241.39 | 241.39 | 244.12 | 1.00 | 1.01 |
| silu_mul (halves) | 8192x2*14336 | 947.25 | 947.25 | 955.92 | 1.00 | 1.01 |
| gelu_tanh_and_mul (halves) | 1x2*4096 | 3.99 |  | 16.20 |  | 4.06 |
| gelu_tanh_and_mul (halves) | 32x2*4096 | 4.22 |  | 16.35 |  | 3.87 |
| gelu_tanh_and_mul (halves) | 2048x2*4096 | 26.34 |  | 34.62 |  | 1.31 |
| gelu_tanh_and_mul (halves) | 8192x2*4096 | 274.07 |  | 277.57 |  | 1.01 |
| gelu_tanh_and_mul (halves) | 1x2*14336 | 4.00 |  | 16.25 |  | 4.06 |
| gelu_tanh_and_mul (halves) | 32x2*14336 | 4.94 |  | 16.16 |  | 3.27 |
| gelu_tanh_and_mul (halves) | 2048x2*14336 | 241.41 |  | 244.17 |  | 1.01 |
| gelu_tanh_and_mul (halves) | 8192x2*14336 | 947.49 |  | 955.20 |  | 1.01 |
| rms_norm | 1x4096 | 3.96 | 3.95 | 15.94 | 1.00 | 4.03 |
| rms_norm | 32x4096 | 4.10 | 4.10 | 15.33 | 1.00 | 3.74 |
| rms_norm | 2048x4096 | 20.54 | 20.46 | 22.09 | 1.00 | 1.08 |
| rms_norm | 8192x4096 | 179.62 | 179.59 | 187.80 | 1.00 | 1.05 |
| rms_norm | 1x5120 | 4.04 | 4.03 | 16.10 | 1.00 | 3.99 |
| rms_norm | 32x5120 | 4.14 | 4.13 | 16.92 | 1.00 | 4.09 |
| rms_norm | 2048x5120 | 24.97 | 24.89 | 26.73 | 1.00 | 1.07 |
| rms_norm | 8192x5120 | 224.68 | 224.52 | 233.82 | 1.00 | 1.04 |
| rms_norm | 1x8192 | 4.16 | 4.15 | 16.39 | 1.00 | 3.94 |
| rms_norm | 32x8192 | 4.37 | 4.37 | 16.41 | 1.00 | 3.76 |
| rms_norm | 2048x8192 | 38.39 | 38.93 | 42.96 | 1.01 | 1.12 |
| rms_norm | 8192x8192 | 357.16 | 357.01 | 376.89 | 1.00 | 1.06 |
| fused_add_rms_norm | 1x4096 | 4.46 | 4.10 | 18.97 | 0.92 | 4.25 |
| fused_add_rms_norm | 32x4096 | 4.43 | 4.25 | 19.40 | 0.96 | 4.38 |
| fused_add_rms_norm | 2048x4096 | 36.99 | 37.58 | 47.45 | 1.02 | 1.28 |
| fused_add_rms_norm | 8192x4096 | 345.31 | 345.95 | 372.31 | 1.00 | 1.08 |
| fused_add_rms_norm | 1x5120 | 4.60 | 4.48 | 19.53 | 0.97 | 4.25 |
| fused_add_rms_norm | 32x5120 | 4.69 | 4.60 | 19.37 | 0.98 | 4.13 |
| fused_add_rms_norm | 2048x5120 | 50.13 | 49.81 | 51.06 | 0.99 | 1.02 |
| fused_add_rms_norm | 8192x5120 | 445.64 | 446.18 | 476.00 | 1.00 | 1.07 |
| fused_add_rms_norm | 1x8192 | 4.42 | 4.31 | 19.16 | 0.98 | 4.33 |
| fused_add_rms_norm | 32x8192 | 4.64 | 4.58 | 19.25 | 0.99 | 4.15 |
| fused_add_rms_norm | 2048x8192 | 79.29 | 73.40 | 207.07 | 0.93 | 2.61 |
| fused_add_rms_norm | 8192x8192 | 737.29 | 737.27 | 743.14 | 1.00 | 1.01 |
| softmax | 1x4096 | 4.16 | 4.16 | 17.19 | 1.00 | 4.13 |
| softmax | 32x4096 | 4.31 | 4.31 | 17.23 | 1.00 | 4.00 |
| softmax | 2048x4096 | 20.96 | 20.90 | 65.45 | 1.00 | 3.12 |
| softmax | 8192x4096 | 180.31 | 180.29 | 230.87 | 1.00 | 1.28 |
| softmax | 1x5120 | 4.42 | 4.40 | 17.20 | 1.00 | 3.89 |
| softmax | 32x5120 | 4.45 | 4.44 | 17.16 | 1.00 | 3.86 |
| softmax | 2048x5120 | 25.70 | 25.57 | 79.17 | 0.99 | 3.08 |
| softmax | 8192x5120 | 223.85 | 224.22 | 317.62 | 1.00 | 1.42 |
| softmax | 1x8192 | 4.53 | 4.51 | 16.89 | 1.00 | 3.73 |
| softmax | 32x8192 | 4.76 | 4.75 | 17.35 | 1.00 | 3.64 |
| softmax | 2048x8192 | 39.32 | 39.18 | 124.18 | 1.00 | 3.16 |
| softmax | 8192x8192 | 350.23 | 351.55 | 442.45 | 1.00 | 1.26 |
| int8 quantize | 1x4096 | 4.05 | 4.03 | 19.23 | 1.00 | 4.75 |
| int8 quantize | 32x4096 | 4.10 | 4.08 | 18.75 | 1.00 | 4.57 |
| int8 quantize | 2048x4096 | 16.26 | 15.35 | 19.20 | 0.94 | 1.18 |
| int8 quantize | 8192x4096 | 51.07 | 49.22 | 108.06 | 0.96 | 2.12 |
| int8 quantize | 1x5120 | 4.11 | 4.08 | 17.67 | 0.99 | 4.30 |
| int8 quantize | 32x5120 | 4.22 | 4.18 | 18.95 | 0.99 | 4.49 |
| int8 quantize | 2048x5120 | 19.60 | 19.14 | 26.52 | 0.98 | 1.35 |
| int8 quantize | 8192x5120 | 177.43 | 177.33 | 179.66 | 1.00 | 1.01 |
| int8 quantize | 1x8192 | 4.32 | 4.29 | 17.72 | 0.99 | 4.10 |
| int8 quantize | 32x8192 | 4.43 | 4.40 | 18.28 | 0.99 | 4.13 |
| int8 quantize | 2048x8192 | 31.51 | 30.13 | 35.02 | 0.96 | 1.11 |
| int8 quantize | 8192x8192 | 278.16 | 278.21 | 288.10 | 1.00 | 1.04 |

</details>
