# gfx1201 on hardware: an accumulator that was not gfx11's, a tree brought up to RDNA3, and the waits gfx12 spells differently

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Until this chapter gfx1201 was a target hk compiled for and had never run
([chapter 13](13-gfx1201.md)): HipKittens' `include/rdna4` was written without
a gfx12 part, every claim about where a value lives inferred from builtin
signatures. A box with eight Radeon AI PRO R9700 (gfx1201, Navi48, 64 CUs,
32 GB GDDR6, ROCm 7.2.3) is now available, shared with other users. This
chapter takes the port from compile-only to the GPU tier running there.

The answer: the operand layout had been inferred right and the accumulator's
wrong, `include/rdna4` was five weeks and 32 commits behind `include/rdna3`,
and hk itself assumed gfx1100's count of LDS reads per tile -- which on gfx1201
makes every derived wait too shallow. With those fixed, HipKittens' rdna4 unit
tests pass in every section on the R9700 and hk's GPU tier runs there:
914 passed, 195 skipped, 0 failed, every group of `scripts/gpu_suite.sh` green, attention's exact-answer tests among them. Nothing is tuned for gfx1201 yet: every plan
and schedule is gfx1100's, and no engine route is on.

## The box

amd-G7466-M6: 2x EPYC 9334, 503 GB, eight R9700 (PCI 03/23/43/63/83/a3/c3/e3),
ROCm 7.2.3 on the host, `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0`
for everything that runs (the gfx950 CI's image; torch 2.10.0 sees gfx1201).
A dozen people use it, most of them as `amd`, and one user's service holds
GPU 0. None of them takes our locks, so every GPU run here goes through
`scripts/ci/gfx1201/hkr9`: our lock of the card, then a check that no process
of anyone's holds it, then a container that sees that one card with core dumps
off (the root partition is over 90% full). The dialect builds against the
`ci-mlir-23.1.3` release's MLIR as on the gfx950 runner.

## The fragment layout, measured

`tools/rdna-probes/wmma_layout.hip`'s `PROBE_GFX12_W32` variant, written for
this day and never run, was the first thing run:

- **A and B operands** (v8f16/v8bf16, 4 VGPRs): lane l feeds row (or column)
  l%16, nothing replicated. That is what `include/rdna4` assumed.
- **The f32 accumulator** (v8f32): lane l register r holds row **r + 8·(l/16)**,
  column l%16 -- the upper half holds rows 8..15. `include/rdna4` had carried
  gfx11's over, row 2r + l/16, the halves interleaving even and odd rows. Every
  mma unit test failed, 224 of 224; the conversions tests, which round-trip
  through the library's own layout, all passed and said nothing.

So on gfx12 every fragment has one shape: a lane owns a line (l%16) and a
contiguous run of 8 along the other axis, the upper half 8 further on.
`WMMA_ELEMENT_STRIDE` is 1 and `WMMA_HALF_SHIFT` 8 for every type, and two
things follow. An accumulator and the operand of the same layout hold the same
elements in the same lanes, so `copy()` from fp32 to bf16 -- attention's P, a
GEMM's epilogue -- is pointwise; on gfx11 it crossed the wave halves in both
directions. And a row-layout tile's lane holds a contiguous run of its row
whatever the types, so every global<->register transfer is one 8-, 16- or
32-byte run a lane (`detail::run` in `global_to_register.cuh`): a bf16 global
into an fp32 tile is one `buffer_load_b128` a base tile, where gfx11 read
twice what it kept, and an fp32 tile into a bf16 global one store, where
gfx11 had to assemble each run across the halves (`pack_interleaved_run`).

The int8 WMMA is the gfx12 form, `v_wmma_i32_16x16x16_iu8_w32_gfx12`: 8 bytes
a lane, the fp8 shape. All four fp8 pairings and int8 pass on the part.

## Five weeks of RDNA3, merged

`include/rdna4` was forked on 18 September; `include/rdna3` has since gained
the register strips, int8 WMMA, indexed rows and atomics, bounded loads and
stores, `load_w4`, granule addressing, Philox, `lds_bind`, RNE stores and more,
and hk generates calls to all of it: the codegen tier for gfx1201 failed 927
tests, every sampled one on an API the tree did not have. The merge was
three-way (`git merge-file`, base the fork point), each conflict resolved by
re-deriving the gfx11 code for the gfx12 layout rather than taking either side:
the transfers above; granules a fragment one `ds_load_b128` a lane, the wave
half folded into SB/32 addresses; `load_w4` dequantizing a lane's own word of
codes with nothing crossing the halves; the masks' half term
`half_shift * half`.

Three things the merge brought in were wrong for gfx12 and compiled anyway:

- **`vm_wait` came back as `s_waitcnt vmcnt(N)`**, gfx11's spelling, which the
  gfx12 assembler still accepts (the legacy encoding). Now `s_wait_loadcnt`.
- **A register array indexed by lane went to LDS.** The rdna4 `rv<float, align>`
  loads indexed a lane's registers by a position that depends on the lane, and
  LLVM's promote-alloca moved the array into *static LDS* -- 64 KB of it at
  S=2, over the limit with the test harness's 32 KB of dynamic LDS, and the
  dispatch failed with `HSA_STATUS_ERROR_INVALID_ALLOCATION`. rdna3's later
  `rv_align_held` / `rv_align_entry` loops have constant indices.
- **A waited load did not bind its tile.** `load<wait=true>()` issues its
  inline-asm `ds_load_b128`s and then a bare `s_wait_dscnt 0`; nothing ties the
  tile's consumers to the wait, and `mma_gate.hip`'s mma_AtBt came back as
  garbage on the R9700 -- LLVM had scheduled the WMMAs above the wait. It now
  ends in `lds_wait_for<0>(dst)`, the wait and the bind. `include/rdna3` has
  the same bare wait, latent there; hk's build-time wait gate refuses a kernel
  that hits it.

And one test was wrong in both trees: `warp/shared/vec/conversions` took
`auto ref = subvec_inplace(...)`, a copy in private memory, whose address the
LDS accessors then read as an LDS offset.

On the R9700, built and run one leaf section at a time so that a fault cannot
hide what follows it: every section passes, mma 229 of 229 with fp8 and the
new int8 section. On a W7900D, `tests/unit/rdna3` passes as before (1664, the
five new int8 tests included). HipKittens #2.

## The target, measured

- **VGPR granule 24**, from LLVM's Occupancy remark over a sweep of 1..256:
  216 VGPRs buy 7 waves a SIMD and 217 buy 6; 240 buy 6 and 241 buy 5.
- **LDS: 128 KB a WGP**, as on gfx1100 (the target had said 64, a guess). One
  wave a workgroup, a workgroup's LDS swept from 8 to 64 KB, the peak number
  resident: 64 at 48-64 KB, 96 at 40, 128 at 32, 160 at 24 -- 4 MB over the
  part's 32 WGPs every time.
- `i8` in the mma table (2 VGPRs an operand), `strips=True`, and
  `verified=True`: budget violations are errors again.

## What hk assumed about gfx1100

**The count of LDS reads a tile load issues.** hk-lds-pipeline derives each
`lds_wait_for` from how many `ds_read`s the loads it waits past issue, and
`hk.dsl.values.lds_loads` said two a 16-bit base tile on every wave32 target:
gfx1100's, where a lane holds the mirrored 32-byte row. On gfx1201 a lane
holds 16 bytes, one read. A count that is high makes every wait too shallow:
`lds_wait_for<4>` meant to leave the second half of B in flight leaves A in
flight too. The count is now the target's: reads = max(1, operand VGPRs x 4 /
16), two for gfx1100's bf16, one for its int8 and for everything on gfx1201.
The GEMM example's waits step down 2 then 0 there, 4 then 0 on gfx1100.

**The wait gate's reading of gfx12 waits.** `hk.runtime.cocheck`, which every
build runs to refuse a read of a register an LDS load is still filling, read
gfx12's `s_wait_dscnt 0x1` by its leading digit -- a full drain -- and did not
read the combined `s_wait_loadcnt_dscnt 0x302` at all. It now reads hex
immediates and the combined forms (loadcnt or storecnt in bits 13:8, dscnt in
5:0). That is what found the waited-load hazard above.

**gfx1100-only tables.** The strip plans (`STRIP_ARCHS`, silu_mul's plan, the
rowwise templates' stream and fold, the norms' LDS reserve, moved to the
R9700's 64 MB Infinity Cache), the GEMM and attention schedule families'
`arches`, and the register model the schedules' legality check uses. The model
was fitted on gfx1201's own builds (`tools/hk-bench/schedules_legality.py`,
1,941 RDNA GEMM and 1,249 decode GEMV schedules under hipcc 7.2.3): no false
accepts, but a loose envelope -- it refuses 603 RDNA points that built -- and
for the decode GEMV infeasible as a hard one (the shipped 4-token schedule
builds at 255 of 256 VGPRs beside points with its features that spill), so the
fit now has a soft stage that gives up as few spills as it can to the build's
spill gate (three here). The same schedules use fewer VGPRs on gfx1201 than on
gfx1100 (the default GEMM 165 against 218), as halved operands should.

## Where it stands

On the R9700 box, the R1 tree (rocm/pytorch rocm7.2.3, hipcc 7.2.3):

| tier | gfx1100 | gfx1201 |
|---|---|---|
| lit (`check-hk`) | 14 passed | -- |
| IR (`tests/hk/ir`) | 1363 passed | 1363 passed |
| codegen (`tests/hk/codegen`) | 1120 passed, 4 skipped, 1 xfail | 1109 passed, 15 skipped, 1 xfail |
| GPU (`gpu_suite.sh`) | (the W7900D: unchanged) | 914 passed, 195 skipped |
| HipKittens unit tests | 1664 passed (W7900D) | every section, 0 failed |

The one codegen failure in both columns, gfx950's 3-warp fused-add strip at
occupancy 5 where the test wants 6, is hipcc 7.2.3's (CI's 7.1 passes it) and
has nothing to do with RDNA. The GPU tier's skips, by reason: gfx950's own
tests (the scaled MFMA, LDS DMA, `hk.comm`, the CDNA ops, gfx950's routes);
data measured on gfx1100 (the routing tables, the route tuner's candidates,
hk.tune's GEMM spaces); `hk.ops.attn_k3`, whose C++ shim is written for
gfx11's fragments; FlyDSL and vLLM, which the image does not have; and the
attention cases with a window, which hk does not take on any arch. Two strip
tests had gated on `include/rdna3` and run on gfx1201 now.

Not done: anything about speed. Every plan, schedule and routing table is
gfx1100's, the ceilings (WMMA rate, LDS and GDDR6 bandwidth) are unmeasured, and
no vLLM or SGLang route is on for gfx1201 -- they stay off until validated
there. The GPU tier runs on the dev box as gpu.yml's `gfx1201` job,
continue-on-error until it is green on main.
