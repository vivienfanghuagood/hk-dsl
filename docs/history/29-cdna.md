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

[CDNA, on one MI355X](../performance.md#cdna-on-one-mi355x) has the
numbers. In short:
- Vector add, RMSNorm and softmax are correct, at 0.51-0.61x torch with
  tiles planned for RDNA3.
- A register-only MFMA GEMM and a GEMM through LDS are correct. The LDS GEMM
  runs at 540 TFLOP/s at 4096³, 0.35x hipBLASLt; it has no pipelining.
- Of the 410 kernel instantiations hk ships, 196 compiled for gfx950 and
  195 for gfx942 with no hand edits. Almost all the failures were missing
  library functions. Bounded load and store were the biggest of them.

### Regressions

None for RDNA. Every example and `hk.ops` instantiation, in every scaffold
plus the IR dump, is byte-identical to main for gfx1100 and gfx1201: 509
kernels after the rebase onto `992a180`
(`tools/cdna-probes/dump_sources.py`, run on both checkouts and diffed). `tests/hk/ir`:
516 passed. `tests/hk/conftest.py` now pins `HK_ARCH=gfx1100` unless one is
set, because an MI355X is now recognised rather than falling back to
gfx1100. The GPU tier skips itself when the GPU is not the pinned arch.
