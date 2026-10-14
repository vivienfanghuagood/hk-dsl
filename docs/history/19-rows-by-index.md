# Rows by index, atomics and random numbers: four kernels from the Triton side

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The scoreboard of the [previous chapter](18-the-scoreboard.md) compared
kernels both languages could write. This chapter is about the kernels hk
could not write at all. Its tiles came from coordinates, so no row could come
from a list. That rules out every mixture-of-experts GEMM, every embedding
and every cache write. A partial sum could not leave its workgroup except
through `wg_arrive`'s counter, and there was nothing to draw a random number
with. Triton's tutorials use all three.

## The ops

Five, each a call into the HipKittens layer underneath
(`include/rdna3/ops/warp/memory/tile/indexed.cuh`,
`include/rdna3/ops/warp/register/tile/random.cuh`):

```python
t = table.gather(T, ids, at, col=c)   # tile row r <- table row ids[at + r]
cache.scatter(t, slots, at, col=c)    # tile row r -> cache row slots[at + r]
out.at[r, c].atomic_add(acc)          # fp32, element by element
old = cnt.at[0, e].atomic_add_scalar(1)   # one per warp; also _max_
u = hk.rand(T32, seed, offset, row_stride)
keep = (u - p).step()                 # 1 where u > p
```

**Gather and scatter.** These build one buffer resource over the whole global
and give each lane its own row as the voffset. An index outside the rows is
how vLLM marks a padding token (-1) and how `moe_align_block_size` pads its
lists (numel). Such an index becomes an offset past the resource, so its load
reads zeros and its store is dropped, with no branch. Each path keeps the
load and store fast paths:
- 16-bit tiles move in b128 runs;
- fp32 tiles of a 16-bit global widen in registers, as `load_bounded` does;
- the store packs through `v_permlanex16`.

**Atomics.** `atomic_add` takes fp32 tiles only. A 16-bit row tile is
replicated across the wave halves, so adding it would count every element
twice. The scalar atomic is issued by lane 0 and broadcast with
`readfirstlane`; issued by every lane, a counter would count lanes rather
than warps.

**Random numbers.** `hk.rand` is Philox4x32-10 with Triton's constants and
round order, followed by Triton's `uint_to_uniform_float`. Element (r, c) is
the draw at `offset + r * row_stride + c`, so a kernel can be checked against
its Triton original bit for bit. `test_indexed.py` compares the two exactly.

## Dropout, and a product that skipped its rounding

`examples/07_dropout.py` is Triton's low-memory dropout tutorial: keep where
`rand > p`, scale by `1 / (1 - p)`. The seed is a one-element int32 tensor
read with `load_scalar`, because a non-tensor argument is a constexpr, baked
into the build. A seed that changes every step would otherwise compile every
step.

At 256 x 4096 hk kept exactly the elements Triton kept. At 8192 x 4096 it
kept one element differently, out of 33.5 million. The draws themselves still
matched everywhere. The mask is `x * step(r - p)`, and `r` is
`float(s) * 2^-31`. HIP compiles with `-ffp-contract=fast-honor-pragmas`, so
clang contracted the product and the subtraction into a single FMA. The
product then skipped its own rounding. A draw whose exact value lay just
above 0.5 but rounded onto it kept in hk and dropped in Triton.

`uniform_from` now pins the product with an empty `asm("" : "+v"(u))`. That
holds under `-ffast-math` too, where `#pragma clang fp contract(off)` is
ignored. The masks are identical at every size tried.

One difference stays, and it is the sign of zero. A dropped negative input
is `x * 0 = -0.0` here, where Triton's `tl.where` writes `+0.0`. The two are
equal as numbers, so `torch.equal` agrees, but they differ as bits.

| 8192 x 4096 bf16, p = 0.5 | time | |
|---|---|---|
| hk | 0.415 ms | 1.02x Triton |
| Triton tutorial | 0.424 ms | |
| torch `F.dropout` | 0.278 ms | 1.52x Triton |

Both DSL kernels run at 0.3 TB/s, far under the 864 GB/s of HBM. They are
bound by Philox's multiplies, a quarter-rate `v_mul_hi_u32` four to a round,
ten rounds per element. torch draws four numbers from each Philox call and
uses all of them; `tl.rand`, and so `hk.rand`, use one. A four-at-a-time draw
would match `tl.rand4x` rather than `tl.rand` and is not written.

## An embedding, and a benchmark that timed a cache

`examples/08_embedding.py` gathers rows of a 151936 x 4096 table, Qwen3's
vocabulary at an 8B hidden size. The first run put hk at 0.48x torch at 4096
tokens.

Every timed call reused the same ids, though. Those rows plus the output
come to 67 MB, which fits in the 96 MB Infinity Cache, and torch's 2 TB/s
was above what HBM can deliver. Serving sees new ids every step. With eight
id sets rotated, which is more than three times the cache, the gap is 0.84x.

What was left was latency. Each gathered row is a random address, and a
16-row tile has 16 of them in flight per wave. Tiles of 32 rows took it to
0.92x. Wider tiles and more warps per workgroup were measured and did worse
(16 x 256: 0.65x; 32 x 64 on 2 warps: 0.82x).

| 151936 x 4096 bf16 | 1 token | 32 tokens | 4096 tokens |
|---|---|---|---|
| hk, x `F.embedding` | 1.18x | 1.28x | 0.92x |
| Triton row kernel, x `F.embedding` | 0.47x | 0.55x | 1.00x |

What remains at 4096 tokens is the row layout. One b128 load touches 16 rows
16 bytes at a time, and the two wave halves load the same addresses. A copy
needs no WMMA layout, so a layout of whole-row runs would close the gap. It
is not written.

## A KV cache write

`examples/09_kv_scatter.py` writes each new token's K row into its slot of a
524288-slot cache, with every seventh slot set to -1 as graph padding. The
reference is a Triton kernel with one program per token.

| 1024-wide rows | 16 tokens | 256 tokens | 4096 tokens |
|---|---|---|---|
| hk, x Triton | 3.00x | 2.88x | 1.18x |
| torch `index_copy_`, x Triton | 0.13x | 0.13x | 0.09x |

The torch arm has to mask out the -1 slots first, so it pays for kernels the
other two do not need.

## LayerNorm backward

`examples/10_layernorm_backward.py` is the backward pass of tutorial 05.
`dx` needs two row reductions, `c1 = mean(xhat * w * dy)` and
`c2 = mean(w * dy)`, and then the row again. `dw` and `db` are sums over
every row, which no workgroup holds. The tutorial adds each row's share into
GROUP_SIZE_M buffers under a spin lock, then sums those in a second kernel.

Here a workgroup sums its own 16 rows in registers with `sum(axis=0)`, a
column reduction, and writes them to a row of the partials that belongs to
it alone. That needs no lock and no atomic. A second kernel, 4 warps to a
64-column strip, sums the partials' rows.

The first version read each row block twice, once per pass:

| 4096 rows | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| read twice, x Triton | 0.73x | - | 0.67x | 0.55x |
| held where it fits, x Triton | **1.12x** | 0.92x | 0.69x | 0.66x |
| torch `native_layer_norm_backward`, x Triton | 0.64x | 0.55x | 0.52x | 0.63x |

Timed separately, the dx kernel was all of it (145 µs of a 160 µs step at
4096). Holding a warp's x-hat and dy tiles across the two passes removes the
second read. A 16 x 64 fp32 pair costs 64 VGPRs, and two pairs with their
temporaries come to about 208. Sixteen warps afford that, at 256 VGPRs a
lane. Thirty-two warps cap a lane at 192, and the build gate refused the
spill. So a held row is at most 2048 wide.

That is where hk's tiles cost it against Triton. A tile has at least 16
rows, and 16 rows of 4096 in fp32, two tensors, are more registers than a WGP
can give them. Triton's program is one row, and it holds one row of 8192.
Past 4096 the second read comes from HBM, which is why 8192 grows faster
than its bytes. The lever is a row split across workgroups, with c1 and c2
reduced through global memory. It is not written.

## Left for later

**Moved to the MoE and int4 work.** These are in the plan's list for this
milestone, and each belongs where it is used:
- the grouped GEMM is the MoE GEMM;
- the persistent matmul is that GEMM's schedule;
- int4/int8 unpack will be a dequantising load in the W4A16 GEMM, not a
  bit-op layer.
