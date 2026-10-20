# CDNA3 and CDNA4: wave64 targets, and a race the IR caught on the first GEMM

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Everything before this chapter ran on gfx1100 and gfx1201. This chapter adds
gfx942 (CDNA3, MI300X) and gfx950 (CDNA4, MI355X) as targets. The design
assessment is in [design/cdna.md](../design/cdna.md); this chapter records
what was merged and what it measured. The hardware was one MI355X of a shared
8-GPU node, used one GPU at a time inside `rocm/pytorch` (ROCm 7.2.3,
PyTorch 2.10). No gfx942 part was available, so everything said about gfx942
is compile-only.

## Part 1: the targets, and what wave64 changes

The work started from a question: is hk RDNA3-only? Mostly it is not. The
RDNA3 parts turned out to be a short list:
- the target table;
- four places that multiplied warps by 32;
- the C++ spelling of the two tile types;
- about 25 library functions that only the fork's `include/rdna3` defines;
- what hk assumes three of them do.

### What was merged

- **Two targets as data.** `python/hk/target/gfx942.py` and `gfx950.py`
  (wave 64, 8 waves a SIMD, 512 registers a lane, granule 8, the LDS figures,
  the MFMA shapes, DMA present), with the C++ twin in `Target.cpp`. The
  register model comes from LLVM's own occupancy remark, read by a clobber
  sweep (`tools/cdna-probes/granule_sweep.py`). It has not been timed, so
  both targets are `verified=False`.
- **One register file.** On gfx90a and later, the AGPRs that hold MFMA
  accumulators come out of the same 512 registers as the VGPRs. The gate now
  charges `align4(VGPRs) + AGPRs` on such targets (`unified_agprs`). Read on
  VGPRs alone, the probe GEMM's 22 VGPRs + 16 AGPRs would claim headroom it
  does not have.
- **wave64.** The thread count, `__launch_bounds__` and every
  `kittens::group<N>` now use the target's wave. Before, a gfx950 kernel
  would have launched half its lanes and still compiled.
- **The fragment in the tile type.** gfx950's bf16 MFMA reads 16x32
  operands, which `include/cdna4` spells `rt<T, R, C, layout, rt_16x32>`. The
  translator gives mma operand tiles that shape, and the shared tiles they are
  read from the matching `st_16x32`.
- **Two waits the CDNA library leaves to the caller.** `include/cdna4`'s
  `load(rt, st)` returns with the `ds_read` in flight, so `load_shared(wait=True)`
  emits the wait and the bind. Its group load returns with the DMA in
  flight, so `group.load` is followed by `vm_wait<0>`.
- **A shim, `include/hk/cdna.cuh`.** It is included after `kittens.cuh` on
  CDNA only, and the HipKittens submodule is untouched. It defines
  `warpid_uniform`, `lds_wait` (which refuses a count above 15, because
  gfx9's lgkmcnt has 4 bits), `vm_wait`, `lds_bind`, `lds_wait_for`, and
  `rsqrt`, `silu` and `step`.

### The race

The first GEMM through LDS on gfx950 compiled cleanly, with no scratch. It
then gave a different wrong answer on each of three runs: max error 13.8,
93.4 and 1.0, against a 1e-2 tolerance. The ISA showed a
`buffer_load_dwordx4 … lds` and then an `s_barrier`, with no
`s_waitcnt vmcnt` between them. Three facts combine here:
- the DMA counts on vmcnt, and cdna4's group load returns with it still in
  flight;
- gfx950 has a back-off barrier, so LLVM puts no wait before `s_barrier`;
- the `ds_read`s after the barrier are inline asm, which LLVM's wait
  insertion cannot see.

One `vm_wait<0>()` emitted after the group load made the GEMM correct and
deterministic. This is the RDNA3 argument again, in a new place: the
library's contract differs from what the IR op promises, nothing fails at
compile time, and the IR is the one place that can keep the contract for
every kernel.

### What it measured

[CDNA, on one MI355X](../performance.md#the-first-cdna-pass) has the
numbers. In short:
- Vector add, RMSNorm and softmax are correct, at 0.51-0.61x torch with
  tiles planned for RDNA3.
- A register-only MFMA GEMM and a GEMM through LDS are correct. The LDS GEMM
  runs at 540 TFLOP/s at 4096³, 0.35x hipBLASLt; it has no pipelining.
- Of the 410 kernel instantiations hk ships, 196 compiled for gfx950 and
  195 for gfx942 with no hand edits. Almost all the failures were missing
  library functions. Bounded load and store were the biggest of them.

## Part 2: the library, and three things it found

Most of what still failed was the library: `load_bounded` and `int8` alone
were 200 of the 214 failures on gfx950. The fix kept HipKittens untouched.
The register-tile memory ops became hk's own, in `include/hk/cdna.cuh`.
They are written once against both CDNA lane layouts, through one coordinate
function. A base tile is R x C, and a lane holds K runs of S contiguous
elements, G apart: include/cdna3's 16x16 tile with S = 4, or include/cdna4's
shape from the tile type.

- **A strided gl.** On wave64 targets the globals are
  `kittens::strided_gl<T>`. This is upstream's gl plus `row_stride_internal`,
  the field the fork's rdna3 gl has and the code-object layout table names.
  Group (DMA) loads and vector loads take it unchanged, because they reach a
  global through `stride<>()` and `operator[]`. A slice of the last axis now
  reads in place, and the code-object launcher builds on CDNA.
- **One buffer descriptor per tile.** It is based at the tile and sized by
  the tile's rows that are inside the tensor and before `limit`. Rows past
  the end read zero, and stores to them are dropped, with no branch. That is
  all of `load_bounded` and `store_bounded`. The unbounded load and store on
  a strided gl take the same path, as include/rdna3's do. Because it is sized
  by the tile, the descriptor cannot wrap. include/cdna4's is sized by the
  whole tensor, in 32 bits, and reads zeros past 4 GiB.
- **Everything else the translation calls.** int8 as a storage type,
  `store_scalar`, `load_scalar`, `wg_arrive`, the two atomics, rows by index
  and Philox, plus GELU on CDNA3.

Three things turned up on the way, each of them silent in its own way.

1. **A waterfall loop.** On gfx942 the first build of a weighted RMSNorm
   wrapped every `buffer_load` in a `v_readfirstlane` / `s_and_saveexec`
   loop. LLVM could not prove the descriptor wave-uniform, so it looped over
   each distinct value. The result is correct but serialized. A
   `readfirstlane` on each word of the descriptor states what is true: one
   tile, the same descriptor in every lane.
2. **Truncating bf16.** On the MI355X, fused add + RMSNorm failed its
   bit-exact check on the residual it writes back. Both CDNA trees' scalar
   `convertor<bf16, float>` truncates, which biases every value toward zero
   by up to one ulp. include/cdna3's packed conversion truncates too.
   include/cdna4's packed one, `v_cvt_pk_bf16_f32`, rounds. hk's stores now
   round to nearest even, through `__float2bfloat16` (one
   `v_cvt_pk_bf16_f32` on gfx950). On CDNA3, so does the fp32-to-bf16 tile
   copy that hk's casts emit.
3. **Vector loads from another GPU.** include/cdna3's register-vector loads
   are ThunderKittens' wave32 code. They index a register array by lane,
   which puts it in scratch: 72 B a lane for the weighted norms and 648 B for
   LayerNorm backward. The gate refused all 163 of those kernels. Their ortho
   mapping is also not cdna3's 16-lane layout. hk's versions load lane by
   lane.

**An arch check.** A test that pinned `arch="gfx1100"` crashed the process on
the MI355X, and another hung for 400 s. `runtime.arch.check_device` now
compares the kernel's arch with what the runtime reports for the device, not
with `HK_ARCH`, and raises `ArchMismatchError` before anything loads. The GPU
tests that pinned gfx1100 now leave the arch to `HK_ARCH`. conftest pins it
to gfx1100, so an RDNA run is unchanged.

**Result.** 493 of 506 kernel instantiations build for gfx950 and 493 for
gfx942, up from 196/410 and 195/410. This uses the device scaffold, with the
spill gate (`tools/cdna-probes/coverage.py --ops`). What is left:
- the two GEMMs' unwaited LDS reads, which wait for a per-target count;
- attention, paged and decode, whose 16-column operand tiles, transposes and
  LDS granules have no CDNA form yet.

On the MI355X, these GPU-tier files pass: elementwise, bounded, strided,
while, indexed (gather, scatter, atomics), load_scalar, phase2 (int8 and
`store_scalar` among them), the weighted norms and the target RMSNorm, 207
tests in all. `test_torch_op` passes too, now that the code-object launcher
builds.

## Part 3: the fragment in the type, CDNA's own waits, and the bind measured

### The fragment becomes a fact of the IR

On gfx950 an mma operand is a 16x32 (row) or 32x16 (col) fragment.
include/cdna4 lays a register tile out fragment by fragment, so the fragment
decides three things: the lane layout, the C++ type, and how many
`ds_read`s it takes to load the tile. Part 1's translator picked the
fragment by a heuristic, after tracing, where nothing else could see it. So
the LDS pipeline could not count a load's reads, and `load(wait=False)` was
refused on CDNA.

The fragment is now a parameter of the type:
`!hk.rt<bf16, 32x32, row, frag 16x32>`. It is printed only when set, so
every RDNA type and IR dump is unchanged. A new pass, `hk-fragments`,
assigns it after `hk-verify`. It gives the fragment to:
- every tile type an mma reads as A or B;
- every type tied to one of those by a cast, a transpose, a broadcast or a
  vector retype.

The accumulators stay 16x16. The pass also turns two silent compile errors
into IR errors with locations:
- *A type that would need two fragments.* This is attention's case: an
  accumulator cast into a bf16 operand, which on gfx950 is a relayout across
  lanes, not a cast.
- *An operand tile that is not a whole number of fragments*, such as a
  16-wide one.

Both used to be `static_assert`s deep in include/cdna4. Every attention, paged
and decode kernel now stops at trace time with that message.

### Counted waits on CDNA

`hk-lds-pipeline` now derives a CDNA load's `ds_read` count from the type:
- include/cdna4: one `ds_read_b128` per 16x32 row fragment, one `ds_read_b64`
  per 16x16, two `ds_read_b64_tr_b16` per 32x16 col fragment, and one per
  16x16;
- include/cdna3: one `ds_read_b64` per 16x16 of a row tile. A col tile is read
  with ordinary loads the compiler waits for, so it is refused for
  `wait=False`.

The count must be exact or low, never high. A load counted high makes every
wait issued while it is in flight too shallow.
`tests/hk/codegen/test_cdna_lds_counts.py` builds one asynchronous load of
each kind and holds the derived count to the `ds_read`s in the disassembly.

Two hardware facts about gfx9 shaped the rest:
1. **lgkmcnt has 4 bits.** The targets now carry `lgkmcnt_max` (63 on
   gfx11/12, 15 on gfx9). A derived wait above it is emitted as the cap. That
   is exact: issue stalls at the cap, so a wave can never have more than 15
   LDS ops outstanding.
2. **Two contracts the pass assumes that include/cdna4 does not keep.** Its
   vectorised LDS store returns with inline-asm `ds_write`s in flight, and
   its DMA counts on vmcnt only. The pass takes both to retire every LDS op,
   as include/rdna3's do. A barrier after either would therefore let another
   warp read stale LDS: the chapter 16 race in a new place. On CDNA the
   translation now follows each of them with `lgkmcnt(0)`.

`examples/04_gemm.py`, with its unwaited LDS reads, now builds for gfx950. It
runs and passes its check on the MI355X, as do 16 of the other examples.

### Does the compiler hoist an MFMA above the wait?

This was the design doc's first open question. On gfx1100 the answer was
yes, and it is why hk binds a tile's registers after every LDS wait.

`tools/cdna-probes/bind_hazard.py` builds an LDS GEMM step loop at seven
register-pressure points: the reduction depth of a step from 32 to 128, and
2 to 8 B tiles in flight. Each is built twice, as hk generates it and with
every bind removed, then disassembled.

| BK x B tiles | VGPR+AGPR, bound / unbound | MFMAs reading unfilled registers, bound / unbound | MI355X, max rel err over 3 runs, bound / unbound |
|---|---|---|---|
| 32 x 2 | 26+16 / 22+20 | 0 / 4 | 3.1e-3 / NaN |
| 64 x 2 | 50+20 / 42+26 | 0 / 8 | 2.4e-3 / NaN |
| 64 x 4 | 80+36 / 54+38 | 0 / 24 | 2.1e-3 / NaN |
| 128 x 2 | 90+24 / 66+20 | 0 / 28 | 2.9e-3 / NaN |
| 128 x 4 | 140+48 / 92+46 | 0 / 60 | 2.5e-3 / NaN |
| 64 x 8 | 140+68 / 82+70 | 0 / 58 | 2.0e-3 / NaN |
| 128 x 8 | 216+92 / 104+76 | 0 / 120 | 2.3e-3 / NaN |

The answer is yes, at every point. Without the bind, LLVM puts MFMAs between
the inline-asm `ds_read`s and their wait, and every unbound build returns NaN.
The unbound builds use fewer VGPRs only because they reuse registers that are
not ready yet. `tests/hk/codegen/test_cdna_bind.py` holds this result.

### hk's two gates, for any code object

The scan that found those MFMAs needs no hk source. It became
`hk.runtime.cocheck`, which runs two gates on any AMDGPU code object:
- **Resources**, read from the metadata note. Scratch or a spill fails;
  registers (the AGPRs included on gfx90a+) and occupancy are reported.
- **Waits**, read from the disassembly. An instruction that reads a register
  an LDS or vector-memory load is still writing, before an `s_waitcnt`
  retires it, fails.

The waits gate walks the code linearly, models the counters as in-order
queues, walks each loop again from its back edge, and empties the queues at
`s_endpgm`. A memory op that takes its address from the load just issued
reads the old value by issue order, so it is a warning, not a failure.

Results:
- **This machine's compile cache** (1527 gfx950 and 1979 gfx942 builds): it
  flags exactly the unbound probes, plus one warning in hk's own output.
  include/cdna4's two-instruction col-layout LDS read has plain `=v` outputs.
  LLVM may therefore give the first `ds_read`'s destination the register that
  holds the second's address. That is correct only because the second
  instruction issues before the first returns.
- **HipKittens' CDNA4 kernels**: its GEMMs, GQA forward (causal and not) and
  softmax pass both gates.
- **Triton 3.6**: its matmul and softmax on gfx950 pass too.

```bash
python -m hk.runtime.cocheck kernel.co ...      # --bundle for a host .so
```

### CDNA tiles for the elementwise and row kernels

The shipped elementwise, norm, fused and quant kernels ran with gfx1100's
tiling. That meant a 16x64 block in 16x16 fragments, which on wave64 is a
4-element run per lane: an 8-byte load for bf16. A sweep of tile shape x
fragment x warps for bf16 add favoured two changes:
- the 16x32 fragment, which makes each run one 16-byte `buffer_load_dwordx4`;
- a 16x256 block.

Together they took bf16 add from 0.58x to 0.92x torch at 8192x8192.

To make this per target without touching RDNA output:
- `hk.arch.target()` lets a kernel body ask what it is traced for.
- Targets say whether their tiles take a fragment shape at all (`tile_shapes`:
  gfx950 only).
- The elementwise wrappers pick the widest block that tiles the shape, and
  the resource gate picks the width. bf16 relu at 256 columns spills SGPRs
  under ROCm 7.2.3's compiler and not under 7.1's, so a refused build steps
  down once.
- The row kernels take the 16x32 fragment for their fp32 tile, except where
  a row is folded onto the tile's rows.

The numbers are in [performance](../performance.md#the-first-cdna-pass). In
short, at 8192x4096 the norms went from 0.70-0.80x torch to 0.93-1.04x,
fused add + RMSNorm from 1.19x to 1.44x, and `silu_mul` is 1.76x.

The norm planner (warps, tiles a warp) was swept on the MI355X. Its choices
were within 3% of the best almost everywhere, except for wide folded rows.
There, gfx1100's one warp a block lost to a quarter of the warps holding
four tiles each: a 16x64 fp32 tile is 16 VGPRs a lane on wave64, not 32, so
the live tiles that cost gfx1100 its occupancy are cheap. With that rule, at
16384x8192 rmsnorm went from 0.89x to 1.05x, layernorm from 0.82x to 0.98x
and softmax from 0.77x to 0.91x. Short tensors (2048 rows) still trail
torch, the weighted RMSNorm most (0.61x).

## Part 4: the register model timed, and validate.py on the MI355X

The gfx950 register model came from LLVM's occupancy remark: 512 registers
a lane, VGPRs and AGPRs together, granule 8, 8 waves a SIMD. It stayed
unverified until it was timed, as gfx1100's granule had been.

`tools/cdna-probes/occupancy_timed.py` does the timing. Each kernel pins N
VGPRs (and M AGPRs) with an asm clobber, then runs a latency-bound chain of
dependent loads. 65536 waves go through, and the total time steps at the
model's boundaries:

| VGPRs | 32-64 | 65-72 | 73-80 | 81-96 | 97-128 | 129-168 | 169-256 |
|---|---|---|---|---|---|---|---|
| model, waves/SIMD | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
| timed, 8 x t(32)/t | 8.00 | 7.10 | 6.47 | 5.50 | 4.46 | 3.46 | 2.33 |

The AGPR cases step too: 64 VGPRs + 1 AGPR at 7, + 64 at 4, and 128 + 64
at 2.

The first version of the probe timed flat. Each lane chased its own random
entry, so every step was a 64-address gather, limited by the cache's address
rate rather than by resident waves. One contiguous 256-byte read a step
fixed it. gfx950 is now `verified=True`, so a register-tile estimate past
the file is an error there, not a warning. gfx942 stays unverified, since
there is no MI300X here.

`examples/validate.py` now runs on the MI355X and passes every case hk has a
native CDNA path for. It skips GEMM, decode GEMV and attention there, and
says so; on RDNA it still runs them. Against torch eager: rope 8.85x, int8
quantize 2.21x, `silu_mul` 1.77x, softmax 1.06x, layernorm 0.98x, rmsnorm
0.94x, add 0.93x, exp 0.87x.

## Part 5: HipKittens' CDNA4 kernels behind hk's engine layer

A native hk GEMM or attention on CDNA that matches HipKittens is
HipKittens-level scheduling work. HipKittens already ships both for
gfx950, so they were wrapped instead. `hk.ops.hipkittens` builds them
through hk's compile cache and spill gate:
- `kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp`;
- the four `attn/gqa{,_causal}/kernel{,_d64}.cpp`.

Each source is upstream's, verbatim, with hk's entry point appended. That
entry point launches on torch's stream rather than the null stream, so a
CUDA graph can capture it. The ops are `gemm_bf16` and `gqa_fwd`, plus the
`engine.Op`s `hk_cdna::gemm_bf16` and `hk_cdna::gqa_fwd`. They refuse, with
the reason, a shape the kernel cannot take; nothing is padded. vLLM's linear
patch gains a `hipkittens` path. The full write-up is
[design/cdna-hipkittens-ops.md](../design/cdna-hipkittens-ops.md).

### The toolchain decides whether the kernels are right

This was the most important finding, and neither of hk's gates sees it.
- **The GEMM under ROCm 7.1's hipcc** builds cleanly and passes the spill
  gate, then gets rows 128-191 of every 256-row tile wrong (relative error
  0.1-0.86). Upstream's own `dispatch_micro`, built unchanged, does the same.
  `hk.runtime.cocheck` passes that build too: no MFMA reads an unfilled
  register, and the waits are intact. The difference between the two
  toolchains' ISA is scalar and 64-bit address arithmetic.
- **The GEMM under ROCm 7.2.3** is correct.
- **The attention kernels under ROCm 7.2.3** do not compile. clang 22 finds
  include/cdna4's three textually identical `all` concepts (rt, art, rv)
  ambiguous.
- **The attention kernels under ROCm 7.1** compile and are correct.

So the first launch of every build on each device runs it on seeded inputs
against fp32 torch. A wrong build raises `Miscompiled`, which names the
hipcc. The engine ops and vLLM's linear then take torch, with one warning.
This is the same argument as the spill gate: a kernel that cannot be
trusted is refused, not returned. Here the evidence has to come from the
device, because no static check sees it.

### What it measured (MI355X, HipKittens unmodified)

- **GQA forward is the win.** It runs 1.28-2.62x torch SDPA, which picks
  aotriton's flash kernel for every configuration measured: 1.66-1.84x
  non-causal and 2.17-2.62x causal at d=128. FlyDSL's gfx950 flash attention
  is level with it: HipKittens is 0.92-1.00x FlyDSL.
- **The GEMM mostly is not.** hipBLASLt is ahead almost everywhere: by 4-9%
  at M ≥ 4096, and by much more below M = 2048. There, a 256x256 tile per
  workgroup leaves most of the 256 CUs idle.
- **Where the GEMM wins, it is hipBLASLt's heuristic.** HipKittens wins 4-27%
  only where hipBLASLt picks a slow kernel. At M = 2048, N = K = 4096,
  hipBLASLt runs 708 TFLOP/s, rocBLAS 965 and HipKittens 886.
- **The route table records this.** The measured gfx950 table,
  `routes_vllm_linear_gfx950.json` (from `route_tune.py --paths hipkittens`),
  sends 37 of 206 (N, K, M) points to HipKittens, all at M ≥ 1792. A re-run
  agreed on 271 of the 272 lookups. This is the engine layer doing its job:
  it routes a shape to whichever kernel measured fastest, including the
  vendor library, and records the evidence.
- **aiter could not be measured.** PyPI's `aiter` is an unrelated package,
  and `amd-aiter` is not published.

### Regressions

None for RDNA. Every example and `hk.ops` instantiation, in every scaffold
plus the IR dump, is byte-identical to main for gfx1100 and gfx1201: 509
kernels after the rebase onto `5d7dd28`
(`tools/cdna-probes/dump_sources.py`, run on both checkouts and diffed). `tests/hk/ir`:
520 passed, and `tests/hk/codegen` 1010. `tests/hk/conftest.py` now pins `HK_ARCH=gfx1100` unless one is
set, because an MI355X is now recognised rather than falling back to
gfx1100. The GPU tier skips itself when the GPU is not the pinned arch.
