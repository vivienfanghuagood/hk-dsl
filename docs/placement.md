# Workgroup placement: which XCD runs which workgroup

On an MI355X (gfx950), workgroup `w` of a launch runs on XCD `w % 8`, and each of the eight XCDs
has its own 4 MB L2. The order in which a kernel numbers its work therefore decides two things:

- how many times bytes that several workgroups share are fetched (once per XCD that reads them);
- how many unrelated streams each L2 holds at once.

`placement=` on `@hk.kernel` changes that order and nothing else. Each workgroup computes exactly
what it did before, so outputs are bitwise the same. Only the time changes, in either direction.

The design and how it works are in [design/placement-pass.md](design/placement-pass.md). The API
is [hk.placement](api/hk.placement.md). A runnable example is
[`examples/15_placement.py`](../examples/15_placement.py): an attention-shaped kernel placed by
the analysis, checked bitwise against the unplaced one, and timed.

- [When to use it](#when-to-use-it)
- [How to use it](#how-to-use-it)
- [What it measured](#what-it-measured)
- [Limits, and turning it off](#limits-and-turning-it-off)

## When to use it

The kernel must run on a part with more than one L2 (gfx950 or gfx942) and meet all of these:

- **Workgroups read the same input bytes.** For example: attention's workgroups of one (batch, KV
  head) read one K/V; a GEMM's blocks of one row read one A panel; a MoE block's column blocks read
  one x tile.
- **Under the hardware's order, an XCD would hold many such groups at once.** It paid off from
  five per XCD. For attention that means 40 or more KV heads; with up to two per XCD (16 KV
  heads, or GQA with 8 KV heads) it is within 2%.
- **The shared bytes are a large part of the traffic.** The reads stream through the L2 rather
  than sitting in registers or LDS.

It is not for:

- kernels whose workgroups wait for each other (`hk.comm.wait`, grid barriers, `resident=True`);
- kernels that add into floating-point outputs atomically. Reordering changes the summation order
  there, so "bitwise the same" no longer holds;
- gfx1100 / gfx1201, which have one L2. There every placement is the identity.

## How to use it

### `placement="auto"`: let the analysis decide, per launch

```python
@hk.kernel(arch="gfx950", warps=8, placement="auto",
           grid=lambda p: (p.q.depth, hk.cdiv(p.q.rows, 256), p.q.batch))
def attn(q: hk.Tensor[hk.bf16], k: hk.Tensor[hk.bf16], v: hk.Tensor[hk.bf16],
         o: hk.Tensor[hk.bf16]):
    x, qb, b = hk.arch.block_idx.x, hk.arch.block_idx.y, hk.arch.block_idx.z
    head_kv = x % k.depth                       # the read the analysis keys on
    ...

attn.placement_choice(q, k, v, o)   # Choice(placed=True, reason='16 streams an XCD', ...)
attn(q, k, v, o)                    # launches the clustered kernel where the policy takes it
```

How "auto" works:

1. **The analysis.** It reads the traced kernel. For each global read it finds the grid digits
   the address depends on, weights them by bytes, and takes the dominant shared read's digits as
   the cluster key. On the kernel above the key is `(z, x % k.depth)`, i.e. (batch, KV head), with
   no annotation.
2. **The policy, at each launch geometry.** It launches the clustered kernel only if all of these
   hold:
   - there are five or more streams per XCD under the default order;
   - for work that is not uniform (a loop whose trip count depends on the grid, e.g. causal), each
     XCD gets at least twelve rounds of workgroups;
   - nothing in the kernel waits, arrives or adds atomically.
3. **The choice.** `Kernel.placement_choice(*tensors)` says what it picks and why.

A kernel registered as a torch op (`hk.torch_op`) is launched by torch, not through
`Kernel.__call__`. There "auto" cannot choose, so the op picks the variant itself:

```python
k = attn.placed() if attn.placement_choice(q, k_, v, o).placed else attn
```

`hk.ops.attn` does this through `attn_cdna.mapping_for`.

### A fixed placement

```python
# One cluster per XCD at a time, the key from the analysis (or given: a key read through an index
# in memory -- a block table, a MoE sort -- is one the analysis cannot see).
@hk.kernel(..., placement=hk.Placement.cluster(key=(hk.Digit.of("x", "mod", "k.depth"),)))

# A GEMM's 1D grid of blocks: chunks of 32 consecutive ids per XCD, the tail in round-robin
# order (HipKittens' chiplet_transform_chunked; what hk's GEMMs use, before their wgm swizzle).
@hk.kernel(..., placement=hk.Placement.chunked(32, "stay"),
           grid=lambda p: (hk.cdiv(p.a.rows, 256) * (p.b.rows // 256), splits, 1))
```

| placement | what it installs |
|---|---|
| `None` (default) | the hardware's order |
| `"auto"` | `Placement.cluster()` per launch where the policy takes it, else the default |
| `Placement.cluster(key=None)` | flattens the grid. XCD `d` takes one contiguous range of the cluster-major units (the first `total % xcds` XCDs one unit more, a bijection for any total), so it walks one cluster before the next. The key's digits are outermost; the other digits keep their order. |
| `Placement.chunked(chunk, tail)` | renumbers `block_idx.x` alone, in place: rounds of `xcds * chunk` consecutive ids per XCD. `chunk="even"` is an eighth of the axis. `chunk=0` is contiguous ranges. `tail`: past the last whole round, `"stay"` (round-robin order) or `"spread"` (contiguous ranges) |
| `Kernel.placed(placement=None)` | a sibling kernel with that placement (default `cluster()`), its name suffixed (`_x8cl0p`) |

### Persistent kernels

A persistent kernel loops over units that are not workgroups, so a grid placement does not apply.
It uses the same function directly:

```python
t = hk.placement.renumber(wg + i * G, blocks, chunk=32, tail="stay")   # the i-th virtual block
first, count = hk.placement.xcd_range(wg % hk.arch.xcd_count(), units)  # this XCD's units
```

### Checking it

`hk.check --placement` evaluates, for every workgroup id of a launch, the decode the kernel's own
emitted ops compute, and checks that it covers the grid exactly once. A raw kernel needs no GPU
and no torch:

```bash
python -m hk.check my_kernel.py:kernel --placement --inputs "i32[2,1,1,45]"
# case 0: kernel: a bijection over 270 workgroups, grid (45, 3, 2), order (z, y, x), 8 XCDs
```

## What it measured

All measurements are on an MI355X. Old and new were run in one process, interleaved, with
`rocm-smi` snapshots of the GPU taken around each run. Every output was bitwise identical to the
default order's. Raw data: `<workspace>/research/2511.02132/runs/` (PRs #55, #63, #65).

**Attention** (`attn_cdna`, MHA unless noted, D 128, B 1; clustered / default):

| KV heads | N, causal | clustered / default | what hk.ops.attn routes to |
|---|---|---|---|
| 128 | 8K-128K, non-causal | 1.117-1.175 | clustered |
| 128 | 8K-128K, causal | 1.047-1.149 | clustered |
| 64 | 8K-128K, non-causal | 1.022-1.078 | clustered |
| 64 | 16K-128K, causal | 1.033-1.080 | clustered |
| 64 | 8K, causal | 0.975-0.981 | **default** (too few rounds) |
| 40 | 8K-128K, non-causal | 1.000-1.045 | clustered |
| 40 | 16K-128K, causal | 1.015-1.051 | clustered from 12 rounds (N 16K at B 1: default) |
| 40 | 8K, causal | **0.90** | **default** |
| 16, 24, 32 | 8K-128K | 0.98-1.03 | default (2-4 streams per XCD) |
| GQA 64/8, 32/8; 8/1 x B8 | 8K-32K | 0.993-1.007 | default (one KV head per XCD already) |

Where routed, no measured point is slower than the default (74 points, worst 1.000).

The gain is clock, not bandwidth. At 128 heads and 32K, the bytes past the L2s drop from 149 GB to
9.7 GB and the clock rises from 1.43 to 1.64 GHz, while the matrix core is 74% busy in both.

**Kernels moved onto placement from hand-written renumbering** (new / old):

| kernel | points | new / old |
|---|---|---|
| bf16 GEMM (`gemm_cdna`, `gemm_cdna_dp`; split-K, ragged M) | 11 | 0.997-1.007 (the same order) |
| K2, MXFP8 GEMM (`mxfp8` as vLLM routes it) | the 18 DeepSeek-V4.1 points | 0.992-1.011 (the same order) |
| K1, MoE a8w4, M where the old remap applied (stage-1 grid a multiple of 8) | M 8192, 11000, 12411, 16384 | 1.0005-1.0012 |
| **K1, M where the old remap fell back to the hardware's order** | M 9500, 10000, 14000, 15001 | **1.19-1.24** |

The K1 gain does not show under vLLM's default chunked prefill. Its 8192-token chunks give the
route M = 8192, where the old remap already applied. It shows with a larger
`max_num_batched_tokens` (the ragged last chunk), or with prefill that is not chunked.

## Limits, and turning it off

- **The analysis sees addresses built from the block ids by `+ - * // %` with constants and tensor
  extents.** A key that goes through memory (`load_scalar`, gathers, a block table) is flagged as
  a heuristic. For those, give the key: `Placement.cluster(key=...)`.
- **Contiguous ranges balance unit counts, not work.** Where units differ in cost (causal
  attention, MoE experts of different sizes) and each XCD has few rounds, the clustered order can
  end on long units (−10% at 40 KV heads, 8K, causal). "auto" avoids that case. A fixed placement
  does not.
- **"auto" chooses only through `Kernel.__call__`.** Torch ops choose the variant themselves (see
  above).
- **The thresholds (5 streams, 12 rounds) were measured on attention.** On other kernels they are
  a starting point, not a measurement.
- **The XCD count** comes from the KFD topology's `num_xcc`: 8 on an MI355X, 4 on an MI350P or in
  a DPX partition, 6 on an MI300A. `HK_XCDS` overrides it, and `hk.arch.xcd_count()` reads it in a
  kernel.

Turning it off:

| scope | how |
|---|---|
| one kernel | `placement=None`, or launch the unplaced kernel instead of `.placed()` |
| attention | `attn_cdna.kernel_for(head_dim, causal)` without `mapping="spatial"` is the default-order kernel. `hk.ops.attn` picks per call, through `attn_cdna.mapping_for`. |
| the whole process | `HK_XCDS=1`: every placement is the identity, including the GEMMs' and K1/K2's XCD order that predates this pass (they then run in the hardware's order) |

Every placement keeps outputs bitwise identical, so turning it off changes time only.
