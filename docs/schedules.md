# Schedules: inspect, constrain and pass a GEMM schedule

`hk.schedules` is the schedule of every GEMM hk runs, as data: which kernel
family, and every knob of it. You can ask what hk runs for a shape, whether
another schedule can run it (and why not, before anything compiles), walk
the legal ones near it, run one, store it in a tuning record, and hand it to
`hk.templates.gemm`. This page is the guide; the design and the measurements
behind the legality check are in [design/schedules.md](design/schedules.md),
every knob's default and values in the [API reference](api/hk.schedules.md).

## What hk runs for a shape

```python
import hk.schedules as S

s = S.lookup(2048, 16384, 4096, arch="gfx950")   # hk.ops.matmul's schedule
print(s)                    # dp(stages=4): a Dp with the knobs off its defaults
s.knobs()                   # every knob
s.steps(2048, 16384, 4096)  # the launches: [dp cols 0:16384 -> out +epilogue]
k = s.kernel()              # the hk.Kernel; k.source() is its C++, k.build() compiles it
```

`lookup(m, n, k, arch, op="matmul")` is what `hk.ops.matmul` launches:
on gfx950 the route table's candidate (`hk.ops.gemm_cdna.route`), on
gfx1100 `hk.ops.gemm.schedule_for` (the tuned records included).
`op="template"` is what `hk.templates.gemm` plans, which on gfx1100 sends
M <= 32 to the decode GEMV. The schedule's kernels are the very kernels
those paths build: same names, byte-identical C++ (a test holds it).

A shape whose kernels are several launches has a schedule that says so:

```python
c = S.lookup(300, 151936, 4096, arch="gfx950")   # Qwen3's lm_head at M = 300
# cols(head=pp(), tail=dp(stages=4, xcd_chunk=0), cut='wave'):
# the pingpong on the columns whose 256x256 blocks fill whole waves of the
# 256 CUs, the four-warp kernel on the rest
[str(st) for st in c.steps(300, 151936, 4096)]
# ['pp cols 0:131072 -> out +epilogue', 'dp cols 131072:151936 -> out +epilogue']
S.Dp(splits=4).steps(1024, 4096, 4096)
# split-K: the GEMM into an fp32 scratch, then the reduction (which carries an epilogue)
```

## Is it legal, and why not

```python
S.Dp(block_n=96).check(4096, 4096, 4096, arch="gfx950")
# ['N=4096 is not a multiple of block_n=96']
S.Rdna(block_m=256, k_step=32).check(4096, 4096, 4096, arch="gfx1100")
# ['registers: up to 317 VGPRs predicted, over the 256 a wave has: it would
#   spill, which the build refuses -- a smaller block, n_split or k_step, or stream_a']
S.PingPong().check(4096, 4096, 4096, arch="gfx1100")
# ['pp: pp builds for gfx950, not gfx1100']
```

`check(m, n, k, arch)` is `[]` or the reasons, and costs microseconds: no
trace, no compile. It covers the arch, the dtype (bf16), the knobs'
arithmetic (warp grids, whole fragments and copies, the families' own
constraints), LDS as `hk-lds-alloc` will lay it out, the registers the build
will need (a model fitted to thousands of builds), and the shape (K and N
against `k_step`, `splits`, `block_n`; M against the decode GEMV's tiles).
`check_kernel(arch)` is the shape-free half -- whether `k.build()` will pass
-- and `predict(arch)` the numbers behind it. `require(...)` raises
`hk.schedules.Illegal` (a `ValueError`, `.reasons`) instead; `launch()` and
`hk.templates.gemm(schedule=)` call it.

What it promises: a schedule it accepts traces and builds -- on 1,500
sampled points the model had not seen, 1 of 823 accepted did not (0.12%).
What it does not: that the schedule is fast. It refuses about one in ten of
the points that would have built (the register model is an upper bound), so
a point it refuses may be worth a `k.build()` if it is close to the limit
(`predict()` says how close). Its occupancy is the upper bound's, so a
`min_occupancy` gate is judged pessimistically too: the default `Rdna()`
predicts 239 VGPRs (6 waves) and builds to 211 (7).

## The space around a schedule

```python
for t in S.space(2048, 16384, 4096, arch="gfx950"):            # every family's seed space
    ...
for t in S.space(2048, 16384, 4096, arch="gfx950", around=s):  # s and its legal neighbours
    ...
S.Dp.space(2048, 16384, 4096, arch="gfx950",
           axes={"block_m": 64, "stages": [2, 3, 4, 5], "wgm": "all"})
```

Every space yields legal schedules only, each kernel once. The default is
the product of every knob's seed values, the rest at the family's starting
point for the shape -- a few dozen to a few hundred points a family (dp at
2048x16384x4096: 496; rdna at 4096^3: 72). `around=s, radius=r` is every
schedule within `r` knob changes of `s`, each a step along its knob's value
list (radius 1 around hk's own choice at 2048x16384x4096: 17 points; radius
2: 133). `axes` pins a knob (`{knob: value}`), gives its values
(`{knob: [...]}`) or opens its whole value list (`{knob: "all"}`); with
`around`, it names the knobs that may move. `t.replace(knob=value)` makes one
by hand.

## Running one

```python
y = s.launch(a, b)                        # C = A @ B.T, every launch of the call
y = hk.templates.gemm(a, b, schedule=s)   # the same
y = hk.templates.gemm(a, b, epilogue=hk.templates.gemm.bias_gelu, bias=bias, schedule=s)
hk.templates.gemm.plan(m, n, k, schedule=s)      # the launches, with the one carrying the epilogue
hk.templates.gemm.kernels(m, n, k, schedule=s, epilogue=..., bias=meta_tensor)  # no GPU
```

`b` is (N, K), as everywhere in hk. A schedule given to the template is
checked first; an illegal one raises with the reasons.

## Records

```python
rec = s.to_record()          # {"family": "dp", "stages": 4}: the knobs off the defaults
S.from_record(rec) == s
S.from_record(["dp", {"stages": 4}])          # a route table's candidate form
S.Rdna.from_record(record["schedule"])        # a gemm_bf16.json record's schedule
```

Records are JSON. `full=True` writes every knob. A record naming a knob the
family does not have raises, or with `strict=False` drops it.

- **gfx1100 tuning records** (`python/hk/tuned/gemm_bf16.json`, and
  `$HK_TUNE_DIR`): a record's `schedule` is an `Rdna` record without the
  family; `hk.ops.gemm.schedule_for` reads it by bucketed M and exact N, K.
  Every knob that changes the C++ can be recorded.
- **gfx950 route table** (`python/hk/tuned/routes_gemm_gfx950.json`): routes
  name candidates. The names hk.ops defines (`pp`, `dp128`, `ppw`, ...) keep
  their meaning -- some depend on the shape, which `S.from_candidate(name, m,
  n, k)` spells out. A table may define names of its own in its
  `candidates` as `[family, {knobs}]`; those are taken exactly, and
  `hk.ops.matmul` and `hk.templates.gemm` route to them.

## The knobs

The full list, with defaults, the values a search takes and the seed space,
is generated into the [API reference](api/hk.schedules.md#the-families).
What each family is and what its knobs trade:

### gfx1100: `Rdna` -- the WMMA GEMM (`hk.ops.gemm`)

A `block_m x block_n` block of C per workgroup of `warps` waves in a
`warp_rows x (warps / warp_rows)` grid; K in `k_step` tiles staged through
LDS. The knobs are the C++'s `#define`s: the tile (`block_m`, `block_n`,
`k_step`, `dot_slice`), the warp grid (`warps`, `warp_rows`), how a warp tile
is cut into accumulators (`n_split`), the L2 swizzle (`wgm`), one or two LDS
buffers and where the staging drain lands (`lds_buffers`, `write_pos`), and
the schedule hints of [chapter 28](history/28-gemm-schedule.md): `setprio`
(off -- it was a fence, 3.5-4.3% slower), `sched` (sched_group_barrier
interleaving), `stream_a` (FlyDSL's one-fragment-at-a-time A), `stagger`
(FlyDSL's K rotation, a loss here). The default is one LDS buffer, 128x128x64,
eight waves as 4x2 (three workgroups a WGP).

### gfx950: `PingPong` -- HipKittens' 8-wave schedule (`hk.ops.gemm_cdna`)

The 256x256 block in four quadrants, eight waves as two halves a barrier
apart so every SIMD has one wave at the matrix core and one at memory
([chapter 33](history/33-cdna-gemm.md)). Knobs: the block (`block_m`,
`block_n` in 128/256), `k_step`, the warp grid (`warps_m x warps_n` = 8), the
L2/XCD placement (`wgm`, `xcd_chunk`), `setprio`, `early_store` (the first
two quadrants stored while the last are computed), `late_b` (the read that
makes HipKittens' own waits right), `transposed_c` (kept as a measured loss),
`splits` (split-K), and the weight-streaming copies of
[chapter 41](history/41-cdna-leftovers.md): `lds_shape="16x64"` (whole-line
copies) and `nt` (nontemporal, past the caches).

### gfx950: `Dp` -- the four-warp ring (`hk.ops.gemm_cdna_dp`)

Smaller blocks for shapes the pingpong has too few of
([chapter 33](history/33-cdna-gemm.md#the-shapes-256x256-has-too-few-blocks-for)):
`block_m x block_n` over a `warps_m x warps_n` grid, copies `stages - 1`
K-tiles ahead in a ring of LDS buffers, one barrier a tile; `pipelined` reads
tile t+1's operands during tile t; `splits` cuts K across workgroups; the
CU-filling blocks (96, 224 columns) and the streamed weights (`lds_shape`,
`nt`, deep `k_step`) of [chapter 41](history/41-cdna-leftovers.md).

### Both: `Decode` -- the decode GEMV (`hk.ops.decode_gemm`)

For M <= 64: W straight to registers, one 16-row block of W a workgroup, its
K split across `warps` waves and, with `splits`, across workgroups.
`tokens` 16-row tiles of X, `slices` 16-deep K slices a load, the combine's
batch (`combine_depth`) and the workgroups a WGP/CU holds (`resident`)
([chapter 9](history/09-decode-kernel.md)).

### `Columns`

Two schedules on column ranges of C: `head` on the first `cut` columns,
`tail` on the rest; `cut` is a count, `"block"` or `"wave"`.

## Attention

The attention kernels follow the same protocol, over a problem that is a
sequence length, a head dim and causal rather than M x N x K:

```python
s = S.lookup_attention(n, 128, causal=True, arch="gfx1100")   # what hk.attention runs
s.check(n, 128, "gfx1100", causal=True)                         # [] or reasons
for t in S.space_attention(n, 128, "gfx1100", causal=True, around=s): ...
t.kernel(128, causal=True).source()
out = t.launch(q, k, v, causal=True)                            # (B, H, N, D) bf16
```

`AttnRdna` (gfx1100, `hk.ops.attn`; [chapter 3](history/03-phase-4-attention.md)):
the query block (`q_block` x `warps` rows; only 16-row blocks trace), the KV
step (`kv_block`, 32 or 64; 64 at head_dim 128 spills), how many 16x16 LDS
reads of K and V^T go in flight before their waits (`qk_tiles`, `pv_tiles`:
what its tuner searches and the shipped `attn_fwd_d128*.json` records hold),
and `setprio`. `AttnCdna` (gfx950, `hk.ops.attn_cdna`;
[chapter 34](history/34-cdna-attention.md)): the tile is HipKittens'; the knobs
are the schedule's choices around it -- the lazy rescale's headroom (`lazy`),
the MFMA/VALU interleave (`sched`), `setprio`, and under causal the longest
blocks first (`lpt`) and skipping the math past a warp's queries (`skip`).
Every point of both spaces was built at head_dim 64 and 128, causal or not:
`check` accepted 508, all of which built, and refused 9 that would have.

## Adding a knob

A knob is one field on the family's class and the parameter of the kernel
factory it is passed to:

```python
stagger: int = knob(0, values=(0, 1, 2), doc="workgroup w starts K at tile w*stagger")
```

`values` is what a search may set it to, `seed` the values the default space
takes, `doc` the line the API reference shows. If it has a constraint, add it
to the family's `_check_struct`; if it changes the live registers, to
`hk/schedules/_regs.py`'s features, then re-sample and refit:

```bash
python3 tools/hk-bench/schedules_legality.py sample --out s.json --n 300 --seeds
python3 tools/hk-bench/schedules_legality.py fit s.json --write
python3 tools/hk-bench/schedules_legality.py sample --out held_out.json --n 300 --seed 7
python3 tools/hk-bench/schedules_legality.py eval held_out.json
```

(hipcc only, no GPU; a few minutes on a many-core box.)
