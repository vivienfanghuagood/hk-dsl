# The language

A kernel is a Python function, traced once per set of constexprs into the
[hk dialect](ir.md), translated to HipKittens C++ and compiled by hipcc. This
page is the language a kernel body is written in. For what happens to it after
tracing, see [the IR](ir.md) and [the runtime](runtime.md); for complete
programs, see [`examples/`](../examples/README.md).

Before writing a kernel, see whether a template does the job:
[`hk.templates.elementwise` / `rowwise`](templates/elementwise-rowwise.md) for
the math of an element or a row, [`hk.templates.gemm`](templates/gemm.md) for
a GEMM with a fused epilogue. They reach the tuned kernels' speed in a few
lines. This page is for the kernels they do not cover. Every name it uses is
in [the API reference](api/README.md).

- [Kernels and parameters](#kernels-and-parameters)
- [Types](#types)
- [Values and operators](#values-and-operators)
- [Memory](#memory)
- [Rows by index, atomics, random numbers](#rows-by-index-atomics-random-numbers)
- [Control flow](#control-flow)
- [Between workgroups, XCDs and GPUs: `hk.comm` (gfx950)](#between-workgroups-xcds-and-gpus-hkcomm-gfx950)
- [Which XCD runs a workgroup: placement](#which-xcd-runs-a-workgroup-placement)
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
`max_vgprs`, `min_occupancy`, `defines`, `resident` and `placement`
(`python/hk/dsl/kernel.py`; the last two are
[below](#between-workgroups-xcds-and-gpus-hkcomm-gfx950)).
`defines` is a dict emitted as `#define KEY value` lines ahead of the
generated C++, for the knobs of the headers it includes. The one a kernel
most often needs is `defines={"HK_BF16_ROUND": 2}` on gfx1100: there a
conversion to bf16 *in registers* (`t.to(hk.bf16)`, `copy()`) rounds ties
away from zero, HipKittens' cheaper default, while every store that narrows
rounds to nearest even as torch does, and gfx950 rounds to even
throughout. A kernel that converts in registers and must match torch bit
for bit -- a residual `x + r` of two bf16 values, which is a tie about one
time in ten -- sets it ([chapter 38](history/38-rdna-losses.md#bf16-stores-round-as-torch-rounds)). The grid is a
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

* **Types are values you build**: `hk.RT(dtype, rows, cols, layout, frag=None)`
  a register tile, `hk.RS(dtype, rows, cols, run=8)` a register strip, `hk.ST`
  an LDS tile, `hk.col_vec(t)` / `hk.row_vec(t)` the vectors a reduction of
  `t` gives, `hk.SV` an LDS vector. A tile's `frag` is the matrix-core
  fragment it is laid out in, where that is not 16x16; left as `None`, the
  `hk-fragments` pass sets it on the operand tiles of a target that needs one
  (gfx950's bf16 operand is 16x32) ([IR](ir.md#passes)). On gfx950 an
  accumulator may be `frag=(32, 32)` (the 32x32x16 MFMA), and an operand may
  name its lane run, `frag=(16, 32, 4)`: a lane holding two runs of 4 rows
  rather than one of 8, which is what a 32x32 accumulator's lane holds (see
  [below](#an-accumulator-read-as-an-operand-gfx950)).

### Register strips

A tile's lane layout is the one WMMA gives it: a 16-bit row tile holds each
row twice, and an fp32 tile gives a lane alternate columns. A kernel that only
streams rows wants each lane to own contiguous runs of a row instead, and the
difference is large: a GELU-and-mul on 16x64 tiles is 0.54x `torch.compile`
at 4096x14336 on an MI355X, on strips 1.03x
([the DX suite](../bench/dx/FINDINGS.md)). Elementwise and row-wise kernels
belong on strips, and the [templates](templates/elementwise-rowwise.md) put
them there for you. That is
a strip (`!hk.rt<f32, 1x1024, strip<8>>`, `kittens::rs`):
- **Layout.** Lane l owns, of every row, the runs of `run` columns starting
  at `j*wave*run + l*run`: a chunk is the wave's lanes, 32 on gfx11 and 64 on
  gfx950. A bf16 run of 8 is one 16-byte access, and nothing is held twice.
  `cols` is a whole number of `wave*run`-column chunks.
- **Memory.** A strip loads and stores in elements (`x.at[r, c]`), and
  converts its element type on the way. `bounded=True` reads zero past the
  tensor's last row and column, and writes nothing there.
- **Operations.** It maps, reduces along rows (a butterfly that leaves the
  answer in every lane; on gfx950 in an order you can state, below) and
  along columns, broadcasts its `col_vec` and
  `row_vec`, and fills by column (`right_fill_`, `left_fill_`). On gfx950,
  `swap_halves()` exchanges each row's two halves (column c takes column
  `(c + cols/2) % cols`, RoPE's rotate_half without the sign): a register
  swap for an even number of chunks, one `v_permlane32_swap` a dword for a
  row of one chunk.
- **Vectors.** Its `row_vec` has the strip's own column layout, so a weight
  vector loads with the accesses of the rows it scales. Its `col_vec` is
  what `hk.collective.cross_warp` combines.
- **Limits.** It is never an MMA operand or an LDS tile. It exists where
  `Target.strips` says: gfx1100 (include/rdna3), gfx1201 (include/rdna4)
  and gfx950 (hk's wave64 port in `include/hk/cdna.cuh`); gfx942 refuses
  it while tracing. The gfx950 port arrived in 0.5.0; the 0.4.0 wheel refuses strips
  there ([what each version has](api/README.md)). Its elements are bf16,
  fp16 or fp32, and on gfx950 also i32
  ([below](#elementwise-c-over-strips-and-i32-strips-gfx950)).

```python
# WARPS and EPS are constexprs; row and warp come from hk.arch
S = hk.RS(hk.fp32, 1, 512)                  # one row, 512 columns a warp
v = x.at[row, warp * 512].load(S)           # 16 bytes a lane of bf16, twice on gfx11
sq = hk.collective.cross_warp((v * v).sum(axis=1), "add", WARPS)
v *= (sq / x.cols + EPS).rsqrt()
o.at[row, warp * 512].store(v)
```

**A stated row-reduction order (gfx950).** A result that must match another
kernel's bit for bit -- a norm's sum of squares against vLLM's or aiter's --
needs the order of the additions, not only their set.
`t.sum(axis=1, lanes="seq" | "tree", xor=(32, 16, 8, 4, 2, 1))`, and the same
for `max`, `min` and `prod`, states it:
- a lane's own values are combined left to right (`"seq"`) or as a balanced
  tree (`"tree"`), then the lanes over the given xor offsets in that order,
  each of 1, 2, 4, 8, 16 and 32 once;
- the strip's values are summed as they are: `(x * y).sum(...)` adds the
  rounded products, and the compiler does not fuse them into the sum (#57);
- `x.sum(axis=1, lanes="seq", xor=..., times=y)` is the fused form:
  `s = fma(x0, y0, r(x1 * y1))`, then `s = fma(xi, yi, s)`, then the xor
  sums. That is what a compiled `s += a * b` loop computes, and what
  dsv41_mono's and aiter's norms do for their sum of squares (`times=x`).

Without the arguments nothing changes (a tree, then 1 .. 32). They are
refused on wave32 targets, on columns and on WMMA tiles.

```python
sq = v.sum(axis=1, lanes="seq", xor=(32, 16, 8, 4, 2, 1), times=v)   # aiter's order
```

`hk.ops.norm`'s RMSNorm, fused add + RMSNorm, LayerNorm and softmax run on
strips wherever a row fits one ([chapter 27](history/27-strips.md)). On
gfx950 so do quantize, the elementwise ops, `silu_mul` (which then takes
the two halves of a gate/up output as strided views) and `hk.ops.rotary`,
vLLM's rotary_embedding in NeoX style, token-major, from a `cos_sin_cache`
of `[cos | sin]` rows and optional int32 positions
([chapter 37](history/37-cdna-rows.md)).

### Elementwise C++ over strips, and i32 strips (gfx950)

`hk.map_cpp(expr, *args, dtype=None)` evaluates one C++ expression for each
element of a strip. It is for what the operators cannot say and a kernel must
reproduce bit for bit from another implementation: bit manipulation, a fused
multiply-add, one particular instruction (`__builtin_amdgcn_*`). Triton's
`inline_asm_elementwise` is the same escape, for the same reason. New in
0.8.0 (#53).
- **Arguments.** `x0`, `x1`, ... in `expr` are `args` in order. A strip
  binds its element (r, c); an `hk.Int` (a `load_scalar()`, an index) or a
  Python int binds its value, the same for every element. Each has its C++
  type: `float` for an fp32 strip, `int` for an i32 strip, an `hk.Int` or a
  Python int. A Python float is refused: write it in the expression (`0.5f`).
  At least one argument is a strip, and every strip has one shape and run,
  which are the result's.
- **Types.** Strip arguments and the result are fp32 or i32: convert a bf16
  or fp16 strip with `.to(hk.fp32)` first. The result is the expression's
  value cast to `dtype`, which defaults to the strips' one dtype. Strips of
  fp32 and i32 together must say `dtype=`, because taking the first strip's
  would silently make an fp32 result an integer.
- **The expression** is one C++ expression, with no `;` and no braces
  (`ValueError`). It is emitted as a lambda,
  `kittens::map_cpp(dst, [&](const auto x0, ...) { return (expr); }, args...)`,
  and compiled as written: hk does not parse it, so a mistake in it is a
  hipcc error. Floating-point contraction is off inside it, so
  `x0 * x1 + x0` is two roundings and a fused multiply-add is spelled
  `__builtin_fmaf(x0, x1, x0)`. Elsewhere HIP compiles with contraction on,
  and whether a product and a sum fused changed with the code around them;
  `__fmul_rn` / `__fadd_rn` are no barrier on ROCm.
- **Where.** Strips on gfx950 only. A wave32 target (gfx1100, gfx1201) refuses
  `hk.map_cpp` while tracing (`TypeError`), and a register tile in its
  place fails verification.

**i32 strips.** On gfx950 a strip may hold i32, `hk.RS(hk.i32, rows, cols,
run)`, for `hk.map_cpp`'s integer work: index arithmetic, exponent bits,
scale bytes. It loads from and stores to i32 tensors like any strip,
`bounded=True` included. `.to(hk.fp32)` and `.to(hk.i32)` convert in
registers, and an fp32 strip stored into an i32 tensor converts on the way.
fp32 to i32, either way, truncates toward zero (2.5 to 2, -1.7 to -1).
Operators and reductions have no i32 implementation (`a + 1` on an i32 strip
is a `TypeError`), so integer arithmetic is written in `hk.map_cpp`. There
are no other integer strips (an i8 strip is refused). i32 strips are tested
on gfx950 only.

```python
S = hk.RS(hk.fp32, 1, 512)    # one row: 64 lanes x a run of 8
I = hk.RS(hk.i32, 1, 512)     # the same layout, i32

@hk.kernel(warps=1, grid=lambda p: (p.x.rows, 1, 1))
def exp_ldexp(x: hk.Tensor[hk.fp32], n: hk.Tensor[hk.i32],
              e: hk.Tensor[hk.i32], o: hk.Tensor[hk.fp32]):
    row = hk.arch.block_idx.x
    v = x.at[row, 0].load(S)
    k = n.at[row, 0].load(I)                  # an i32 strip
    # the biased exponent of v: bits, which no operator says
    e.at[row, 0].store(hk.map_cpp("(__builtin_bit_cast(int, x0) >> 23) & 0xFF", v, dtype=hk.i32))
    # v * 2^k: fp32 and i32 strips together, so the result's dtype is said
    o.at[row, 0].store(hk.map_cpp("__builtin_ldexpf(x0, x1)", v, k, dtype=hk.fp32))
```

On an MI355X, for `x` of shape (64, 512) and `n` in [-8, 8], `e` equals
`(x.view(torch.int32) >> 23) & 0xFF` and `o` equals `torch.ldexp(x, n.float())`,
bit for bit. `tests/hk/gpu/test_strip.py` has an `hk.Int` argument and the
contraction check.

### Dtypes, and two deliberate spellings

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
  `load_scalar(invariant=True)` says the kernel never writes the element (a
  page table, a length): on gfx950 it is a scalar-cache `s_load`, on
  lgkmcnt, where a global load's wait would sit behind every copy in
  vmcnt's queue.

A global carries one runtime row stride, so a slice of the last axis — the
two halves of a fused `(M, 2I)` output, say — is accepted without a copy:
contiguous along it, rows evenly strided, outer axes compact over the rows.
Anything else is refused with a message saying why
([strided globals](history/12-strided-globals.md)). Register loads and stores
can be bounded by rows, `x[idx].load(ty, bounded=True)` and
`o[idx].store(t, bounded=True)`, which makes a tile that runs past the last
row read zeros and drop the stores past it, with no branch
([row-predicated loads and stores](history/10-row-predicated.md)); a strip
is bounded by rows and columns alike (`x.at[r, c].load(S, bounded=True)`).
A workgroup's synchronous copy, `g.load(tile, view)` / `g.store(view, tile)`,
is bounded by rows without asking on gfx1100 and on gfx950: rows past the
tensor's last read zero and are not written. On gfx950 that holds from 0.7.0
(#49); 0.6.0's read up to a tile's rows past the tensor there and its
`g.store` overwrote them. gfx942's copies are still unbounded.

LDS offsets and every `lgkmcnt` are derived by the passes, which also prove
that every asynchronously loaded tile is waited for before it is read: see
[`hk-lds-alloc` and `hk-lds-pipeline`](ir.md#passes). An `hk.lds.alloc`
inside a runtime `if` counts toward the launch's LDS (the two arms start at
the same offset, the larger continues); one inside a runtime `for` or `while`
is refused, so allocate it before the loop. Before 0.7.0 both lay past the
LDS the launch requested: reads returned 0, writes were dropped (#56).
[`examples/04_gemm.py`](../examples/04_gemm.py) stages through LDS with a
derived wait.

**Copies left in flight (gfx950).** `g.load(tile, view, wait=False)` is
gfx950's DMA, `buffer_load_dwordx4 ... lds`: sixteen bytes a lane straight
into LDS, counted on vmcnt and returned in flight. `g.wait_for(*tiles)` is the
`s_waitcnt vmcnt(n)` that retires this warp's share of the copies into those
tiles, with `n` derived from the order the copies were issued, as `lgkmcnt` is
for LDS reads. The other warps' shares are visible after a barrier:

```python
g.load(As[1], a.at[m0, k1], wait=False)   # next tile, in flight
...                                        # math on As[0]
g.wait_for(As[1])                          # vmcnt, derived
hk.arch.barrier()                          # every warp's share has landed
x = As[1].subtile(64, 64, wr, 0).load(T, wait=False)
```

`g.load(..., nontemporal=True)` copies with `sc0 nt`, for data read once, and
`hk.lds.alloc(..., shape="16x64" | "32x32")` lays an allocation out as that
include/cdna4 st_shape rather than the one its reads' fragment picks (16x32
row operands, and 32x16 stride-4 col operands, read out of those too): a copy
then reads each row in 128- or 64-byte runs instead of 64 or 32, and its
lines once. The paged decode's copies went from 6.15 to 6.9 TB/s with both
([chapter 39](history/39-cdna-paged.md)); non-temporal in the shorter runs
fetched each line twice.

The tile is a whole allocation or a buffer of a stack picked by a Python int,
so `hk-lds-pipeline` knows which buffer each copy writes and checks the two
hazards a copy has across warps: a read of the buffer before every warp's
share has landed, and a copy into it before every warp's reads of it have.
Both are a count of barriers between the two events. In exchange, a read of
such a buffer may stay in flight across a barrier, which the LDS rules refuse
for a buffer picked at run time.

**Two halves a barrier apart.** `with hk.arch.pingpong(behind):` makes the
warps where `behind` holds wait at one extra `s_barrier` on entry, and the
others at one on exit. Between the two, every barrier pairs a program point in
one half with the next in the other: a schedule that alternates memory and
math between barriers runs the halves out of phase, so with eight warps the
two waves on a SIMD take turns at the matrix core. `behind` comes from
`warp_id()`, both ends are in the kernel's own body, and inside, the pass
counts two barriers where one would do outside. gfx950's GEMM
(`hk.ops.gemm_cdna`, [chapter 33](history/33-cdna-gemm.md)) is built from
these.

### An accumulator read as an operand (gfx950)

Attention's P comes out of an MFMA as an accumulator and P·V wants it as an
operand. On gfx950 that needs no data movement if the scores are kept
transposed (Sᵀ = K·Qᵀ, Oᵀ += Vᵀ·Pᵀ) and P is read in place:

```python
T_S = hk.RT(hk.fp32, 64, 32, "col", frag=(32, 32))       # Sᵀ, an accumulator
T_P = hk.RT(hk.bf16, 64, 32, "col", frag=(16, 32, 4))    # Pᵀ, an operand
S = hk.zeros(T_S)
S += k_tile @ q_tile.T
S.subtile(32, 1).exp2_()               # rows 32-63: the same registers
grow = m_new.any_above(m, 8.0)         # one int a wave: did any lane's max jump
S.as_operand(hk.bf16, into=P)          # converts, no register moves
hk.arch.pin(P, m)                      # P and m stay in this phase
O += v_tile.T @ P                      # V read with P's lane run
```

- `Tile.as_operand(dtype, into=)` reads a col-layout accumulator (32x32 or
  16x16) as the stride-4 operand fragment of the same rows, converting fp32
  to bf16 or fp16 on the way. The reduction index then runs down the rows in
  the accumulator's order; `hk-fragments` gives the MMA's other operand the
  same stride, so the product is unchanged.
- `Tile.subtile(rows, index)` is rows `[index*rows, (index+1)*rows)` of a
  register tile as a tile: an alias, so a write into it writes the tile.
  `index` is a Python int.
- `Vec.any_above(other, threshold)` is 1 in every lane if any lane's entry
  exceeds `other`'s by more than `threshold`: a wave vote, usable as a
  branch condition or a factor.
- `hk.arch.pin(*values)` holds register values to the program point: their
  definitions stay above it and their uses below. LLVM sinks a value to its
  use; in a schedule of phases a barrier apart that moves work into the other
  phase, which `sched_barrier` (a fence for the scheduler only) does not stop.

`hk.ops.attn_cdna` is built from these ([chapter 34](history/34-cdna-attention.md)).

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

## Between workgroups, XCDs and GPUs: `hk.comm` (gfx950)

A persistent kernel, a task queue, a hand-off of a tile from one XCD or GPU
to another: on gfx950 a plain store stays in the writer's XCD's L2, and a
plain re-read on the reader's side can be stale even after the flag it
waited for arrived. `hk.comm` lowers each hand-off to what a memory-model
litmus suite found correct ([the design and the measurements](design/comm.md)):
- **The scope belongs to the tensor.** `hk.Tensor[dtype, "device"]` (between
  the XCDs of one GPU) or `hk.Tensor[dtype, "system"]` (between GPUs): every
  load, store and scalar atomic through it carries the scope's cache-policy
  bits, since both sides of a hand-off need them.
- **Signals.** `hk.comm.signal(view, value)` drains, barriers and stores one
  scoped flag; `hk.comm.wait(view, value, timeout_s=, error=)` polls it, and
  a timeout sets the `error` element (or traps) instead of hanging.
  `hk.comm.arrive` / `hk.comm.barrier` are an epoch'd grid barrier,
  `hk.comm.xcd_barrier` one with one global atomic per XCD a round.
- **Work queues.** `hk.comm.claim(view)` takes the next task (one scoped
  atomic add, no drain); every scalar fetch-op (`atomic_min_scalar`,
  `_or_`, `_and_`, `_xor_`, `_exch_` beside `_add_` and `_max_`) works
  plain or scoped.
- **Mailboxes.** `hk.comm.post(view, value, tag)` / `hk.comm.take(view, tag)`
  send one int32 and its tag in one 8-byte scoped store: a 1.51 us round
  trip between two MI355X. A mailbox orders nothing else.
- **Residency.** A grid that waits on its own workgroups must all be on the
  GPU at once: `hk.kernel(..., resident=True)` refuses a launch that cannot
  be, with `ResourceError`, instead of deadlocking
  (`Kernel.resident_capacity()`).
- **Between GPUs.** `hk.runtime.SymmetricBuffer` is one buffer per rank
  with every peer's mapped; while one is open, a launch refuses it bound to
  an unscoped parameter.

```python
@hk.kernel(warps=1, grid=lambda p: (64, 1, 1), resident=True)
def drain(q: hk.Tensor[hk.i32, "device"], done: hk.Tensor[hk.i32], *, N: int = 1):
    t = hk.comm.claim(q.at[0])               # the next task, once each
    while t < N:
        done.at[t].atomic_add_scalar(1)
        t = hk.comm.claim(q.at[0])
```

Every name is in [the API reference](api/hk.comm.md);
[`examples/14_task_graph.py`](../examples/14_task_graph.py) is a persistent
task graph with dependencies, and `hk.arch.xcd_id()` says which XCD ran each
task. What hk.comm still lacks against the NVIDIA megakernel designs is its
[roadmap](design/comm.md#roadmap).

## Which XCD runs a workgroup: placement

A gfx950 deals workgroup `w` to the XCD of `w % 8`, and each XCD has its own
4 MB L2. Workgroups that read the same bytes -- attention's K and V of one KV
head, a GEMM's panels -- fetch them once per XCD that reads them, and
numbered the wrong way each L2 holds many unrelated streams at once.
`hk.kernel(placement=...)` renumbers the grid at kernel entry; what each
workgroup computes is unchanged, so outputs are bitwise the same:
- `placement=hk.Placement.cluster()`: one sharing cluster an XCD at a time,
  the key found by the analysis (`hk.placement.analyze`), or named with
  `key=(hk.Digit.of("z"), hk.Digit.of("y"))` where it is read from memory;
- `hk.Placement.chunked(chunk, tail)`: `block_idx.x` alone, in rounds of
  `chunk` consecutive ids an XCD (a GEMM's split-K `y` untouched);
- `placement="auto"`: the cluster placement where the policy
  (`hk.placement.choose`) takes it -- not for order-dependent or resident
  kernels, from five KV streams an XCD, not for non-uniform work under twelve
  rounds an XCD. `k.placement_choice(*args)` says what it decided and why.

On a part with one L2 (gfx1100, `hk.arch.xcd_count() == 1`) every placement
is the identity. `python -m hk.check FILE.py:NAME --placement --inputs ...`
checks that a launch covers its grid exactly once
([check](check.md#which-xcd-runs-which-workgroup---placement)). gfx950's
attention takes the cluster placement by measurement: 1.05-1.18x at 128 KV
heads, 8K-128K tokens (#55). The design and its evidence are in
[workgroup placement](design/placement-pass.md);
[`examples/15_placement.py`](../examples/15_placement.py) shows it on a kernel
of your own. New in 0.8.0 (#63, #65).

## Namespaces

* **The rest has namespaces**:
  - `hk.arch`: block, warp and lane ids, `xcd_id()` and `xcd_count()`
    (gfx950's XCDs), `target()` (the target being traced
    for, for a body whose tiling depends on it), `barrier(drain=False)`,
    `pingpong(behind)`, `sched_barrier`, `sched_group_barrier` and
    `SCHED_CLASSES`, `setprio`, `pin`, `vm_wait`, `wg_arrive`;
  - `hk.lds`: `alloc`, `shared_vec` and `alloc_vec`, `reserve`, `wait_for`;
  - `hk.math`: elementwise functions -- `exp`, `exp2`, `log`, `log2`, `abs`,
    `relu`, `gelu` (the tanh form), `dgelu`, `sqrt`, `rsqrt`, `silu`, and
    `add`/`sub`/`mul`/`div`/`maximum`/`minimum` with `into=`. The
    transcendental ones are fp32 only, and there is no `tanh`, `sigmoid` or
    `erf`: compose them. Each one's signature and dtypes are in
    [the catalogue](api/hk.math.md);
  - `hk.group(n)`: workgroup loads (`wait=False` on gfx950) and their
    `wait_for`, stores, and the staging pipeline;
  - `hk.collective`: `cross_warp`, `fold_rows`;
  - `hk.comm` (gfx950): scoped hand-offs between workgroups, XCDs and GPUs
    ([above](#between-workgroups-xcds-and-gpus-hkcomm-gfx950)).

**Scheduling hints.** `hk.arch.sched_group_barrier(classes, size,
sync_id=0)` asks the compiler's scheduler for `size` instructions of
`classes` (`"mma"`, `"vmem_read"`, `"ds_read"`, `"valu"`, ... or a mask; the
names are `hk.arch.SCHED_CLASSES`) at this point, after the earlier groups
with the same `sync_id`. A run of them is an interleave: "one staging load,
then two WMMAs". It moves only what the scheduler can see, and an inline-asm
instruction is in no class. include/rdna3's LDS fragment reads and writes are
asm, so in hk's GEMM it can place the staging loads and the WMMAs, not the
LDS traffic. `s_setprio` ends a scheduling region, so a group does not reach
across one. `hk.arch.sched_barrier(allow)` is a fence that lets the classes
in `allow` cross ([chapter 28](history/28-gemm-schedule.md)).

**Occupancy caps.** `hk.lds.reserve(nbytes)` allocates LDS the kernel never
uses, to cap how many workgroups share a WGP: on gfx1100 a WGP holds
128 KB ÷ (the kernel's LDS + `nbytes`) workgroups
([chapter 27](history/27-strips.md)).

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
* A kernel body must be source `inspect` can read. Code built with `exec`,
  typed into `python -c`, written as a lambda, or loaded from a stale
  `__pycache__` compiled at another path traces, but its `for`, `if` and
  `while` are not rewritten, so its loops cannot be runtime loops. An error
  about control flow on a device value then says that the source could not
  be read, and why.
* Pass errors name values by their base name (`lds`), not the C++'s
  deduplicated one (`lds1`), so "add lds to that lds_wait_for" can be
  ambiguous with several LDS tiles in flight. The old front end had the same
  limit.
* Nothing is eliminated before the passes, so an unused allocation still
  counts in `lds_bytes`. The old front end behaved the same way.
* MLIR is built from source (23.1.3). Wheels are Linux x86_64 only, one
  CPython each.
* gfx942 is compile-only, and has no register strips (so no templates).
  gfx1201 runs the GPU tier on an R9700 with gfx1100's plans and schedules,
  none of them tuned there yet ([chapter 47](history/47-gfx1201-on-hardware.md)). On gfx950 strips build, and its GEMM and attention are
  kernels of hk's own (`hk.ops.gemm_cdna`, `hk.ops.attn_cdna`,
  [chapters 33](history/33-cdna-gemm.md) and [34](history/34-cdna-attention.md)),
  with HipKittens' CDNA4 kernels also wrapped as hk ops
  ([CDNA3/4](design/cdna.md)).

This list is from [the MLIR chapter](history/15-mlir-dialect.md), which ends
with "For whoever continues": the next pieces of work, in order of value.
