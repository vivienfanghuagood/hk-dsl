# The language

A kernel is a Python function, traced once per set of constexprs into the
[hk dialect](ir.md), translated to HipKittens C++ and compiled by hipcc. This
page is the language a kernel body is written in. For what happens to it after
tracing, see [the IR](ir.md) and [the runtime](runtime.md); for complete
programs, see [`examples/`](../examples/README.md).

- [Kernels and parameters](#kernels-and-parameters)
- [Types](#types)
- [Values and operators](#values-and-operators)
- [Memory](#memory)
- [Control flow](#control-flow)
- [Namespaces](#namespaces)
- [`@hk.func`](#hkfunc)
- [Errors and locations](#errors-and-locations)
- [Limits](#limits)

## Kernels and parameters

A kernel is a Python function. Positional parameters are tensors
(`hk.Tensor[dtype]`, 4-D `(batch, depth, rows, cols)`). Constexprs are the
keyword-only parameters, or any parameter annotated `hk.Constexpr[int]` with a
default. They are baked into the source and are ordinary Python values in the
body, so `range(D // 16)` unrolls and `if KIND == "softmax"` picks a branch at
trace time.

```python
T = hk.RT(hk.fp32, 16, 64)

@hk.kernel(warps=4, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), p.x.batch))
def row_sumsq(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
    row = hk.arch.block_idx.y * 16
    acc = hk.zeros(hk.col_vec(T))
    for cb in range(hk.arch.warp_id(), hk.cdiv(x.cols, 64), 4):
        v = x.at[row, cb * 64].load(T)       # global -> registers, bf16 -> fp32
        acc += (v * v).sum(axis=1)           # one row_sum, accumulated in place
    o.at[row, hk.arch.warp_id()] = acc
```

`@hk.kernel` takes `warps`, `grid`, and optionally `arch`, `name`,
`max_vgprs` and `min_occupancy` (`python/hk/dsl/kernel.py`). The grid is a
function of the parameters: `p.<tensor>.<batch|depth|rows|cols>` gives an
extent and `p.<CONSTEXPR>` gives a constexpr value, and it must return a
3-tuple. A kernel that names no `arch` is built for `hk.target.current_arch()`
(see [runtime](runtime.md#which-gpu)). `max_vgprs` and `min_occupancy` are
extra build gates on top of the unconditional no-spill rule
([resource gates](runtime.md#the-resource-gates)).

A constexpr does not have to be keyword-only. A positional one is a constexpr
when it is annotated so, and it still needs a default, because every
constexpr is baked into the source:

```python
@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def scale_pow2(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], N: hk.Constexpr[int] = 1):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    v = x[idx].load(hk.RT(hk.fp32, 16, 64))
    for _ in range(N):                       # N is a Python int: unrolled
        v = v + v
    o[idx] = v.to(hk.bf16)
```

Constexpr values are ints or floats. Specialising on a constexpr produces a
different kernel, and a different compilation. `k.specialize(D=64)` is
explicit about that; calling with a different value does the same thing
implicitly — `k(x, o, EPS=eps)` passes the constexpr to the call itself, uses
the caches directly and makes no object at all. `specialize()` is for handing
a configured kernel to something that will call it many times.

A tensor with fewer than four dimensions fills the trailing ones, so a 2-D
tensor is `(rows, cols)` (`include/pyutils/hk_bind.cuh`).

## Types

* **Types are values you build**: `hk.RT(dtype, rows, cols, layout)` a register
  tile, `hk.RS(dtype, rows, cols, run=8)` a register strip, `hk.ST` an LDS
  tile, `hk.col_vec(t)` / `hk.row_vec(t)` the vectors a reduction of `t`
  gives, `hk.SV` an LDS vector.

A tile's lane layout is the one WMMA gives it: a 16-bit row tile holds each
row twice, an fp32 tile gives a lane alternate columns. A kernel that only
streams rows wants a lane to own contiguous runs of a row instead, and that
is a strip (`!hk.rt<f32, 1x1024, strip<8>>`, `kittens::rs`): lane l owns, of
every row, the runs of `run` columns starting at `j*32*run + l*run`, so a
bf16 run of 8 is one 16-byte access and nothing is held twice. `cols` is a
whole number of `32*run`-column chunks. A strip loads and stores in elements
(`x.at[r, c]`; `bounded=True` reads zero past the tensor's last row and
column and writes nothing there), converts its element type on the way,
maps, reduces along rows (a butterfly that leaves the answer in every lane)
and along columns, broadcasts its `col_vec` and `row_vec`, and fills by
column (`right_fill_`, `left_fill_`); its `row_vec` has the strip's own
column layout, so a weight vector loads with the accesses of the rows it
scales, and its `col_vec` is what `hk.collective.cross_warp` combines. It is
never an mma operand or an LDS tile, and it is gfx11-only.

```python
S = hk.RS(hk.fp32, 1, 512)                  # one row, 512 columns a warp
v = x.at[row, warp * 512].load(S)           # 2 x 16 bytes a lane of bf16
sq = cross_warp((v * v).sum(axis=1), "add", WARPS)
v *= (sq / x.cols + eps).rsqrt()
o.at[row, warp * 512].store(v)
```

`hk.ops.norm`'s RMSNorm, fused add + RMSNorm, LayerNorm and softmax run on
strips wherever a row fits one ([chapter 27](history/27-strips.md)).

The dtypes are `hk.bf16`, `hk.fp16`, `hk.fp32`, `hk.i8` and `hk.i32`, plus
`hk.fp8e4m3` and `hk.fp8e5m2`, which are accepted only where the target has an
fp8 WMMA — gfx1201, not gfx1100 (see
[the gfx1201 chapter](history/13-gfx1201.md)).

Two spellings are deliberate, both kept on purpose when the language moved to
MLIR ([history](history/15-mlir-dialect.md)):
- a vector type is named by the tile it reduces, `hk.col_vec(T)`, not
  `hk.ColVec[fp32, 16]`, because the dialect ties a vector's layout to its
  tile;
- a load names the register tile it makes, `x.at[r, c].load(T)`, not
  `x.tile(16, 64)[r, c].load(fp32)`, because `x.tile[r, c]` already means a
  tile-unit coordinate.

## Values and operators

* **Device values are objects with operators.** `a + b`, `t * v` (a vector
  broadcasts along the axis it came off; `v[:, None]` / `v[None, :]` spell
  that axis out and check it), `t.exp()`, `t.to(hk.bf16)`,
  `t.sum(axis=1)`, `acc += a @ b.T` (one WMMA into `acc`; the variant follows
  from the transposes), `-t`, `hk.min(a, b)`; scalars are `Int`/`Bool` with
  Python arithmetic and comparisons.
* **Tiles are buffers.** An operation either makes a new tile or writes an
  existing one — `+=`, `t.exp_()`, `into=` — which is how include/rdna3 spells
  everything and what keeps a loop's registers the same registers.

## Memory

* **Memory is indexed.** `x[coord]`, `x.at[r, c]` (element coordinate) and
  `x.tile[r, c]` are views; `.load(type)`, `.store(value)`, `x[idx] = value`,
  `.load_scalar()`. LDS tiles come from `hk.lds.alloc`, and `s[i]`,
  `s.subtile(...)`, `s.load(type, wait=False)`, `s.granules()` follow.

A global carries one runtime row stride, so a slice of the last axis — the
two halves of a fused `(M, 2I)` output, say — is accepted without a copy:
contiguous along it, rows evenly strided, outer axes compact over the rows.
Anything else is refused with a message saying why
([strided globals](history/12-strided-globals.md)). Register loads and stores
can be bounded by rows, `x[idx].load(ty, bounded=True)`, which makes a tile
that runs past the last row read zeros and drop the stores past it, with no
branch ([row-predicated loads and stores](history/10-row-predicated.md)).

LDS offsets and every `lgkmcnt` are derived by the passes, which also prove
that every asynchronously loaded tile is waited for before it is read: see
[`hk-lds-alloc` and `hk-lds-pipeline`](ir.md#passes).
[`examples/04_gemm.py`](../examples/04_gemm.py) stages through LDS with a
derived wait.

## Rows by index, atomics, random numbers

What a mixture-of-experts GEMM, an embedding, a KV cache write, a split-K
reduction and a dropout need, and a tile load cannot say:

```python
t = table.gather(T, ids, at, col=c)   # tile row r <- table row ids[at + r]
cache.scatter(t, slots, at, col=c)    # tile row r -> cache row slots[at + r]
out.at[r, c].atomic_add(acc)          # an fp32 tile, element by element
old = cnt.at[0, e].atomic_add_scalar(1)   # one int32 add per warp (also _max_)
u = hk.rand(T32, seed, offset, row_stride)   # what triton.language.rand draws
keep = (u - p).step()                 # 1 where u > p, else 0
```

`ids` is an int32 global. Gather and scatter move row-layout tiles, a lane's
row being its own; an index outside the tensor -- `-1` for a CUDA graph's
padding, `numel` for `moe_align_block_size`'s -- or a position at or past
`count=` reads zeros or writes nothing, with no branch, because the row is a
buffer offset and an out-of-range one points past the resource. Two tile rows
scattered to one row race.

`atomic_add` takes fp32 tiles only: a 16-bit row tile is replicated across the
wave halves, and adding it would count every element twice. The scalar
atomics are issued by lane 0 and return the old value to every lane; a
workgroup of W warps that calls one performs W of them.

`hk.rand(ty, seed, offset, row_stride)` fills an fp32 tile: element (r, c) is
Philox4x32-10 at `offset + r * row_stride + c`, bit for bit Triton's draw for
that seed and offset, so a kernel can be checked against its Triton original
exactly. The seed may be a device scalar (`x.at[0].load_scalar()`): a
non-tensor argument is a constexpr, built in, and a seed that changes every
step would compile every step. `step()` turns a comparison into a factor;
[`examples/07_dropout.py`](../examples/07_dropout.py) is a dropout with it,
and [chapter 19](history/19-rows-by-index.md) records what each of these
measured.

## Control flow

* **Control flow is Python.** The body is rewritten at the AST level, as CuTe
  DSL does: a `for` over a device bound, an `if` on a device condition and a
  `while` on a device test become `scf.for` / `scf.if` / `scf.while`. A scalar
  the loop or branch rebinds is carried through it (iter_args, branch
  results); rebinding a *tile* inside a runtime loop is refused, with the
  in-place spelling in the message. `hk.range(n, unroll=4)` asks for a runtime
  loop with a pragma, `hk.range_constexpr` insists on unrolling, and
  `hk.scope()` / `hk.if_(c, opaque=True)` keep their register-allocator meaning.

A `for` over a Python bound — a constexpr, `range(D // 16)` — is unrolled at
trace time, as in `scale_pow2` above. [`examples/06_control_flow.py`](../examples/06_control_flow.py)
shows `for`, `if` and `while` on device values, carried scalars, and the
`scf` and C++ they become. What control flow cannot do is under
[limits](#limits).

## Namespaces

* **The rest has namespaces**: `hk.arch` (block and warp ids, `barrier`,
  `sched_barrier`, `sched_group_barrier`, `setprio`, `vm_wait`,
  `wg_arrive`), `hk.lds` (`alloc`, `reserve`, `wait_for`), `hk.math` (elementwise functions, and `add`/`mul`/... with
  `into=`), `hk.group(n)` (workgroup loads, stores and the staging pipeline),
  `hk.collective` (`cross_warp`, `fold_rows`).

`hk.arch.sched_group_barrier(classes, size, sync_id=0)` asks the compiler's
scheduler for `size` instructions of `classes` (`"mma"`, `"vmem_read"`,
`"ds_read"`, `"valu"`, ... or a mask) at this point, after the earlier
groups with the same `sync_id`: a run of them is an interleave, "one staging
load, then two WMMAs". It moves only what the scheduler can see, and an
inline-asm instruction is in no class; include/rdna3's LDS fragment reads
and writes are asm, so in hk's GEMM it can place the staging loads and the
WMMAs, not the LDS traffic. `s_setprio` ends a scheduling region, so a
group does not reach across one. `hk.arch.sched_barrier(allow)` is a fence
that lets the classes in `allow` cross. `hk.lds.reserve(nbytes)` takes LDS
and uses none of it, an occupancy cap: 128 KB a WGP over the bytes a
workgroup holds is how many it runs ([chapter 27](history/27-strips.md),
[chapter 28](history/28-gemm-schedule.md)).

`hk.arch.wg_arrive(counter, expected)` counts this workgroup's arrival at
`counter` (an element view of an int32 global) and returns how many arrived
before it; the last to arrive resets the counter, and it contains barriers, so
every warp must reach it (`python/hk/dsl/arch.py`). Why it is a counter rather
than fp32 atomics is in [the chapter that added it](history/11-cross-workgroup-reduction.md),
where it was introduced as `hk.wg_arrive(counter, idx, expected)`.

## `@hk.func`

`@hk.func` marks a helper that is traced inline, with its control flow
rewritten too. Calling one outside a kernel that is being traced raises.

## Errors and locations

A device value used where Python needs a number — a tile extent, a list index,
`range()` outside a traced function — raises at that line, saying which of
those it was.

The passes' errors come out of `k.trace()` as `hk.VerifyError`, ending
`at file:line in function` — the kernel author's line
([passes](ir.md#passes)).

An op that the library does not implement for the requested dtype is the same
kind of late failure, so it is caught the same way: `gelu` on a bf16 tile
raises while tracing rather than producing `undefined hidden symbol` at the end
of a 30-second compile.

## Limits

**What the language does not do.** Each of these raises, with the
spelling that works:

* A loop or branch body that `return`s, `break`s or `continue`s is left as
  Python, so its condition has to be known at trace time. `while ... else`
  is not rewritten.
* A workgroup operation (`barrier`, group loads and stores, staging) cannot
  sit inside a runtime branch or a runtime `while`: a warp that leaves early
  leaves the rest waiting. A `for` over a bound every warp agrees on is fine.
* A tile rebound inside a runtime loop is refused; it is updated in place
  (`+=`, `into=`), so that every iteration writes the same registers.
* A kernel body must be source `inspect` can read. Code built with `exec`
  or a lambda traces, but its loops cannot be runtime loops.
* Pass errors name values by their base name (`lds`), not the C++'s
  deduplicated one (`lds1`), so "add lds to that lds_wait_for" can be
  ambiguous with several LDS tiles in flight. The old front end had the same
  limit.
* Nothing is eliminated before the passes, so an unused allocation still
  counts in `lds_bytes`. The old front end behaved the same way.
* MLIR is built from source (23.1.3). Wheels are Linux x86_64 only, one
  CPython each; gfx1201 is still compile-only.

This list is from [the MLIR chapter](history/15-mlir-dialect.md), which ends
with "For whoever continues": the next pieces of work, in order of value.
