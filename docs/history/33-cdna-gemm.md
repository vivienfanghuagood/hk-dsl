# gfx950's GEMM in hk: the DMA left in flight, two halves a barrier apart, and where the time goes

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 29](29-cdna.md) ended with hk's own GEMM on the MI355X at 631 TFLOP/s
at 4096³, 0.41x hipBLASLt, and with HipKittens' CDNA4 GEMM wrapped as an hk op
because a native one "is HipKittens-level scheduling work". This chapter does
that work. HipKittens' GEMM is now written in hk. Three language features
made it expressible, and the pass that checks them found a read HipKittens'
hand-counted waits do not cover. The kernel runs level with HipKittens at the
large squares. Two more hk kernels and a measured route table cover the
smaller M of Llama-3-8B's and Qwen3-8B's linear layers. The last section says
where the time still goes and what is below the bar.

Hardware and software throughout: one MI355X (gfx950) of the shared node, GPU
5, idle at the start of every run, in `rocm/pytorch:rocm7.2.3 … py3.12 torch
2.10` (HIP 7.2.53211, hipBLASLt from that image), FlyDSL 0.4.0 (PyPI) with its
checkout's `kernels/` at `88f1969`, HipKittens at the pinned submodule. Every
arm is checked against fp32 first, then timed interleaved in one process,
alternate rounds reversed, one pass streaming 3x the 256 MB Infinity Cache;
numbers are the best round, and spreads are recorded in the tables' JSON.

## What hk could not say

HipKittens' `kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp`, piece
by piece, against hk at `be52996`:

| piece | hk before | now |
|---|---|---|
| 256x256x64 block, eight warps as 2x4, four 64x32 accumulators a warp | expressible | the same |
| 16x16x32 MFMA on 16x32 operand fragments, `st_16x32` swizzled LDS | expressible (chapter 29) | the same |
| global → LDS with `buffer_load_dwordx4 … lds`, **left in flight**, `s_waitcnt vmcnt(N)` by hand | no: `g.load` was issued and waited (`vmcnt(0)`, `lgkmcnt(0)`) | `g.load(..., wait=False)` and `g.wait_for(tiles)`, the vmcnt derived |
| **two halves a barrier apart**: `if (warp_row == 1) s_barrier` at entry, `if (warp_row == 0)` at exit | no: `hk-verify` refuses a barrier in a branch | `with hk.arch.pingpong(cond):` |
| LDS reads **in flight across a barrier** | no: the pass refused every read in flight at a barrier | allowed for a buffer the pass can name; the overwrite is checked instead |
| `s_setprio`, `sched_barrier`, XCD remap and the WGM swizzle | expressible | the same |
| precomputed swizzled offsets, M0 written by inline asm | -- | the compiler hoists the offsets; hk passes the LDS pointer and lets it set M0 |

## The language

**Copies left in flight.** `g.load(tile, view, wait=False)` is
`hk.group_load {async}`. On gfx950 it translates to the shim's
`group_load_async<NW>`, which issues sixteen bytes a lane with the
`st_shape`'s swizzle applied on the global side, as include/cdna4 does. The
buffer descriptor is the tile's own and ends at the tensor's last row, so a
partial last tile reads zeros, and nothing wraps past 4 GiB. `g.wait_for(*tiles)`
is `hk.group_wait_for`, an `s_waitcnt vmcnt(n)`. `hk-lds-pipeline` derives `n`
from issue order, as it derives `lgkmcnt` for LDS reads. A tile is a whole
allocation or a stack's buffer picked by a Python int; a runtime index is
refused for an async copy.

**The cross-warp checks.** The same naming lets the pass check the two hazards
a copy has between warps. Both reduce to how many barriers separate two
events in program order:
- a read of a buffer must come after this warp's copy into it is retired, and
  after a barrier, so that every other warp's share has landed;
- a copy into a buffer must come after every warp's reads of it have landed,
  a barrier ago.

Inside a pingpong half the workgroup is a barrier behind, so the same
barrier is one program point later in half the warps, and two barriers are
needed when both events are inside it. Counted as instances: warp P's event
after `b_P` barriers, warp Q's after `b_Q`, safe iff `b_Q ≥ b_P + 1`. In the
pingpong the leading half has executed one more barrier at the same program
point, which gives the 2. This is also what lets a read of a named buffer stay
in flight across a barrier: the hazard the old rule guarded against is now
checked where it happens, at the write. A read of a buffer picked at run time
keeps the old rule, and RDNA output is unchanged.

**Two halves.** `with hk.arch.pingpong(warp_id // 4 == 1):` emits the two
conditional barriers. `hk-verify` keeps both ends in the kernel's own body,
and refuses a condition that comes from `lane_id`.

**A row limit.** `g.load(..., wait=False, rows=n)` ends the copy's descriptor
`n` rows from the tile's first; past it the copy reads zeros. It was added
for the K padding below, which first moved the copy's row instead and cost
7%.

Tests: lit `Transforms/lds-dma.mlir` (the counts, every hazard, the
pingpong's second barrier, a loop's steady state), `Target/dma.mlir`, and the
verifiers; `tests/hk/ir/test_cdna_gemm.py`; the codegen tier holds the
derived copy count to the `buffer_load … lds` count in the disassembly.

## The read HipKittens' waits do not cover

Writing the schedule in hk, the pass refused it on the first iteration:

```
load_shared reads Bs[2] while this warp's copy into it is in flight.
```

HipKittens waits for all but three copies after the prologue and at the end
of each iteration (`vmcnt(6)`). The copy it leaves outstanding is the one
into `Bs[1][0]`. The fourth memory phase of the next iteration reads that
buffer before its own wait. Nothing orders the read after the copy except
the time four phases take. `gemm_kernel(hipkittens_waits=True)` reproduces
HipKittens' waits, and the IR tier holds the refusal.

What it is not: the reason ROCm 7.1 miscompiles HipKittens' GEMM (chapter 29).
hk's kernel is correct under ROCm 7.1 at every shape tried, deterministic
over three runs. With HipKittens' waits patched back into hk's C++ by hand it
is still correct under 7.1 and 7.2.3, so the window does not open at these
shapes.

hk's schedule reads that tile one phase later, in the fifth memory phase
together with A's half (`late_b`). The fourth phase's wait already covers it
there, and HipKittens' own counts then pass the check unchanged. The first
fix, waiting for the copy where it was read (`vmcnt(4)` at the end of each
iteration), left fewer copies in flight and was 0.3-0.6% slower.

## The kernel

`hk.ops.gemm_cdna` is the schedule in hk, and `hk.ops.gemm.matmul`'s gfx950
form. Its loop matches HipKittens' instruction for instruction: 128 MFMAs, 48
`ds_read_b128`, 16 copies, 16 barriers and 10 waits per two K-tiles. It
takes any M, K in 64-wide tiles (an odd count is padded with one tile copied
with `rows=0`) and N in 256-wide blocks. An N that 256 does not divide, like
lm_head's 151936, runs its last 128 columns as a second, 128-wide launch into
a column slice of the output, because a 128-wide block runs at two thirds of
a 256-wide one.

| M = N = K | hk | HipKittens | hipBLASLt | FlyDSL | hk / best |
|---|---|---|---|---|---|
| 4096 | 1453-1465 | 1435-1445 | 1474-1487 | 1428-1445 | 0.98-0.99 |
| 8192 | 1566-1568 | 1571-1575 | 1613-1616 | 1582-1591 | 0.97 |

Two measured knobs, kept as options:
- `s_setprio` around the math is worth 10% here (1308 without it at 4096³),
  the opposite of gfx1100, where removing it was the gain (chapter 28).
- `transposed_c`, accumulating Cᵀ for 8-byte stores (a quarter of the
  instructions), is 2-6% slower: the swapped MFMA operands cost 1.2% in the
  loop, and the store costs what its bytes cost, not its instructions.

## Where the time goes

The tool for this was a hand-patched build of the generated C++, timed
against the unpatched one in one process (numerics ignored where a patch
breaks them):

| patch, at 8192³ | TFLOP/s | what it says |
|---|---|---|
| none | 1523-1546 | |
| no copies in the loop | 1769 | the global → LDS path costs 15% |
| C's store behind a never-true branch (accumulators kept live) | 1608-1621 | the store costs 5-6%, and without it hk is level with hipBLASLt |
| one of the four quadrants stored | 1590 | the store's cost follows its bytes |
| C stored as perfectly coalesced 16-byte rows (wrong values) | 1545 | the pattern is a fifth of it |
| a third of the LDS reads dropped | 1550 | LDS bandwidth is not the limit |

Dropping the store outright made the compiler delete the MFMAs it fed: 2433
TFLOP/s, meaningless, and why the store is kept behind a branch.

So the gap at 8192³ is the epilogue. A workgroup with 128 KB of LDS fills a
CU, so every CU writes its tile at the same moment and nothing overlaps it.
Storing the first two quadrants as soon as they are final (`early_store`)
takes back 0.4%. hipBLASLt's kernel for these shapes is persistent with
Stream-K (`PersistentLoopStart`, `SK_Store` in its code object), four waves
of 128x128. Its loop issues sixteen copies a wave per K-tile, one between
every two MFMAs, with up to eighteen in flight (`vmcnt(18)`, about 72 KB a
CU). hk's pingpong keeps 32-48 KB in flight.

Overlapping one tile's store with the next tile's loads in a persistent hk
kernel runs into vmcnt. On gfx9 stores count on vmcnt too, and LLVM treats
loads and stores as completing out of order. A derived count therefore
cannot include the stores issued after the copy it waits for, which means
every copy wait after the store waits for the store as well. This is
recorded as the next thing to solve for large M.

## The shapes 256x256 has too few blocks for

At M = 1024, N = 4096 a 256x256 block is 64 workgroups for 256 CUs.
`hk.ops.gemm_cdna_dp` is a second kernel built from the same features: four
warps in 2x2, blocks of 64-128 by 128, a ring of S LDS buffers with the
copies S-1 K-tiles ahead, and one barrier a K-tile. That barrier is both the
read-after-copy edge for tile t and the copy-after-read edge for the buffer
the next copy overwrites, and the pass checks both. Measured and kept or
dropped:
- **XCD-aware numbering** (`xcd_chunk=0`, an eighth of the grid per XCD):
  below 8 x 32 blocks HipKittens' fixed chunk renumbers nothing, and the
  blocks sharing a B tile land on different XCDs' L2s. +16% at 256x4096x4096,
  +45% at 256x6144x4096.
- **Register pipelining** (tile t+1's operands read under tile t's MFMAs):
  no change, kept as the default because it costs nothing.
- **Deeper rings** (6 stages): no change. Without its copies this kernel runs
  twice as fast at 256-1024 rows, so the copy path is its limit too, not the
  latency the ring hides.
- **Split-K through `hk.arch.wg_arrive`**, as hk's decode GEMV does: 2-8x
  slower than not splitting. Measured, not explained; the likely cause is
  the device-scope fences around `wg_arrive`, on a part whose eight XCDs
  each have their own L2.
- **Split-K as two launches** (fp32 shares to a scratch, then
  `reduce_kernel`): 35 → 122 TFLOP/s at 32x4096x14336 (1.06x hipBLASLt),
  268 → 557 at 256x4096x14336 (1.05x). The pingpong takes the same splits:
  445 → 1162 at 1024x4096x14336 (1.12x), 886 → 1341 at 2048x4096x14336
  (1.10x).

The decode GEMV (`hk.ops.decode_gemm`, W straight to registers, chapter 9)
did not build on gfx950 because include/cdna4 has no fp32 register ↔ LDS
tile copy, which its cross-warp combine needs. The shim now has one. Its 48
GPU tests now run and pass on gfx950. Its gfx1100 schedules were swept on
gfx950 (slices, warps, residency) and none moved it by more than 10%.

## Routed

`gemm_cdna.CANDIDATES` names hk's kernels: the pingpong at 256 and 128 rows
with and without splits, the four-warp kernel at 128x128, 64x128 and 64x64
with and without splits, and the decode GEMV.
`tools/hk-bench/gemm_cdna_routes.py` times every candidate that takes a shape
against hipBLASLt, FlyDSL and HipKittens, and writes
`hk/tuned/routes_gemm_gfx950.json`. `matmul` takes the table's candidate at
the nearest measured M for that N x K, and a rule for unmeasured shapes.
Every kernel it routes to is hk's own.

The table below is hk's routed kernel over the best of hipBLASLt, FlyDSL and
HipKittens, eager. 108 of 122 points are at or above 0.9.

Bold is below 0.9 (a bold 0.90 rounds up from below).

| N x K | M=1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | 1.85 | 1.35 | 1.16 | 0.97 | 1.05 | 1.10 | 0.92 | **0.88** | 1.15 | 1.06 | 0.98 | 0.96 |
| 6144 x 4096 | 1.13 | 1.11 | 0.95 | 0.96 | 1.02 | 0.93 | 0.92 | **0.86** | 0.93 | 0.95 | 0.94 | 0.96 |
| 12288 x 4096 | 1.14 | 1.00 | 1.03 | 0.91 | 0.94 | 0.90 | **0.87** | 0.94 | 0.94 | 0.92 | 0.97 | 0.96 |
| 14336 x 4096 | 1.01 | 0.96 | 1.01 | 0.95 | 0.92 | 0.97 | 1.13 | 1.12 | 1.00 | 1.01 | 0.99 | 0.96 |
| 24576 x 4096 | 1.06 | **0.89** | 0.96 | **0.83** | **0.88** | 0.94 | 0.95 | **0.89** | 0.91 | 0.99 | 0.94 | 0.92 |
| 28672 x 4096 | 1.07 | 1.02 | 0.96 | 1.02 | **0.86** | 0.92 | **0.87** | 0.96 | 1.01 | 1.01 | 0.94 | 0.94 |
| 4096 x 12288 | 1.20 | 1.20 | 0.92 | 1.06 | **0.90** | **0.84** | 0.94 | 1.06 | 1.13 | 1.11 | 0.97 | 0.98 |
| 4096 x 14336 | 1.11 | 1.12 | 0.97 | 1.14 | 0.95 | **0.84** | 1.06 | 1.07 | 1.12 | 1.11 | 0.95 | 0.94 |
| 128256 x 4096 | 1.06 | 0.92 | 0.91 | **0.90** | 0.92 | 0.90 | 0.91 | 1.02 | 0.99 | 0.97 | 0.99 | 0.98 |
| 151936 x 4096 | 1.00 | 0.93 | 0.90 | 0.91 | 0.93 | 0.91 | **0.81** | 0.98 | 1.04 | 0.94 | 0.98 | 0.96 |

The kernel each point routes to (`pp` the pingpong, `pp128` its 128-row block, `dp128`/`dp64`/`dp64x64` the four-warp kernel's 128x128, 64x128 and 64x64 blocks, `kN` N splits of K, `decode` the GEMV):

| N x K | 1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | decode | decode | decode | dp64x64k2 | dp64k4 | dp64k4 | dp128k4 | dp128k2 | dp128 | ppk2 | pp | pp |
| 6144 x 4096 | decode | decode | dp64x64k2 | dp64k4 | dp64k4 | dp128k4 | dp128k2 | ppk4 | ppk2 | pp | pp | pp |
| 12288 x 4096 | decode | decode | dp64x64 | dp64x64 | dp64x64 | ppk4 | ppk4 | ppk2 | pp | pp | pp | pp |
| 14336 x 4096 | decode | decode | dp64x64 | dp64x64 | dp64x64 | ppk4 | dp128 | ppk2 | pp | pp | pp | pp |
| 24576 x 4096 | decode | decode | dp64x64 | dp64x64 | dp64 | dp128 | pp128 | pp | pp | pp | pp | pp |
| 28672 x 4096 | decode | decode | dp64 | dp64x64 | dp64 | dp128 | pp128 | pp | pp | pp | pp | pp |
| 4096 x 12288 | decode | decode | dp64x64k4 | dp64k8 | dp64x64k4 | dp64k4 | dp128k4 | dp128k2 | ppk4 | ppk2 | pp | pp |
| 4096 x 14336 | decode | decode | dp64k8 | dp64x64k4 | dp64x64k4 | dp64k4 | dp128k4 | dp128k2 | ppk4 | ppk2 | pp | pp |
| 128256 x 4096 | decode | decode | dp64 | dp64x64 | dp64 | dp128 | pp | pp | pp | pp | pp | pp |
| 151936 x 4096 | decode | decode | dp64 | dp64 | dp64 | dp128 | pp | pp | pp | pp | pp | pp |

How the grid got there: a first table with the pingpong, the four-warp
kernel and the GEMV had 76 points at 0.9. Split-K as two launches for the
four-warp kernel took it to 92. Split-K for the pingpong too (`ppk2`,
`ppk4`) took down_proj at M = 1024-2048 from 0.76-0.84 to 1.11-1.13, and
1024 x 6144 to 0.93: 96. A 64x64 block and its splits (twice the
workgroups at small M), and a 128-row pingpong block, took it to 108.

Below 0.9, and why:
- **M = 256-512 with N = 6144-151936** (0.81-0.89): between the two kernels.
  The pingpong has too few blocks, the four-warp kernel's copy path tops out
  near 900-1000 TFLOP/s, and splitting a 4096-deep K costs its scratch. The
  worst is lm_head at M = 256 (0.81).
- **M = 32-128 with N = 24576/28672, or down_proj** (0.83-0.90): weight
  streaming at 3.3-4.4 TB/s against hipBLASLt's 4-5. A kernel that keeps W
  out of LDS, as the GEMV does, with X shared through it, is the next
  candidate.
- **4 x 24576 and 32 x 128256** (0.89, 0.90): within the run-to-run spread.

Two caveats. First, these are eager calls: hk's pybind launch costs 5.6 µs on
the host against `F.linear`'s 12 µs, which favours hk at the smallest M. hk's
launch is on the null stream, so a CUDA graph cannot capture it. The
code-object launcher (`hk.runtime.codeobj`) launches on torch's stream, and
the router does not use it yet. Second, single-point ratios move 1-3% between
processes; the JSON keeps every round.

## For attention next

The two features attention needs from this chapter are in place:
- copies left in flight with derived vmcnts and checked hazards;
- the pingpong.

HipKittens' GQA forward uses both. What still stops a native CDNA attention
is chapter 29's list:
- an accumulator cast into an operand, a relayout across lanes on gfx950
  that hk-fragments refuses;
- 16-column operand tiles;
- `transpose_sep` and `make_causal_t`.

The epilogue finding carries over too: attention's output store is per
query block, and the same store-at-once burst will show at large batch.

## Tests and CI

- lit: 12 of 12.
- IR tier: 657 passed, 2 skipped.
- Codegen tier: 1038 passed.
- gfx950 GPU tier: 608 passed, 170 skipped, 0 failed (main: 471 / 216). The
  new `tests/hk/gpu/test_gemm_cdna.py` checks every candidate against fp32
  at six shapes chosen for their edges; 31 of the skips are a candidate and
  a shape it does not take.
- `tests/hk/gpu/test_gemm.py` (23) and `test_decode_gemm.py` (48) run
  natively on gfx950 where they skipped.
- RDNA and existing CDNA output: every example and `hk.ops` instantiation is
  byte-identical to `be52996` on gfx1100, gfx1201, gfx942 and gfx950
  (`tools/cdna-probes/dump_sources.py`, which now records a refused kernel
  and goes on; on gfx1201 it used to stop at the first).

## Reproduce

```bash
# container, one MI355X; FlyDSL wheel on PYTHONPATH, FLYDSL_ROOT its checkout
python3 tools/hk-bench/gemm_cdna_routes.py --out python/hk/tuned/routes_gemm_gfx950.json
python3 tools/hk-bench/hipkittens_cdna.py gemm --hk --flydsl --no-squares --lm-head --m 1,16,64,256,1024,4096
HK_ARCH=gfx950 python3 -m pytest tests/hk/gpu/test_gemm.py tests/hk/gpu/test_decode_gemm.py
# host
PYTHONPATH=python python3 -m pytest -m 'not gpu' tests/hk/ir/test_cdna_gemm.py tests/hk/codegen/test_cdna_gemm_builds.py
```
