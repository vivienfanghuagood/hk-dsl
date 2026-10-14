# Phase 4: the generated attention kernel matches the handwritten one

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Phase 4 is done: the generated attention kernel matches the handwritten one,
and beats it under causal.** The gate was 64-65 TFLOPs plus elementwise
agreement with `F.scaled_dot_product_attention` across every H3 shape, a
non-dividing N, B=2, GQA and causal. Same process, interleaved, both kernels
loaded at once (`hk/ops/_attn_ab.py`):

```
shape           dsl ms   dsl TF   c++ ms   c++ TF  dsl/c++
N=4096           7.880     61.0    7.895     60.9   1.002x
N=8192          30.528     63.0   30.527     63.0   1.000x
N=16384        120.334     64.0  120.300     64.0   1.000x
N=32768        475.309     64.8  475.132     64.8   1.000x
480p/5s       1100.228     64.9 1099.149     65.0   0.999x
causal 8192     16.628     57.9   16.754     57.4   1.008x
```

Against torch SDPA on the same chip that is 2.9-3.2x, and 4.0x causal. 31
numerical cases pass (`tests/hk/gpu/test_attn.py`). The generated kernel also
runs at **occupancy 6 against the handwritten kernel's 5** -- 224 VGPRs versus
247 -- which is the one place the IR is structurally ahead rather than level.

Three findings came out of getting there, and all three now live in the library
or the IR rather than in a kernel:

* **`warpid()` costs a divergent branch.** It is `threadIdx.x >> 5`, which the
  compiler keeps in a VGPR because it cannot see that a wave is 32 consecutive
  threads. Anything derived from it inherits the VGPR, so a comparison on it
  becomes `v_cmpx` with exec save and restore rather than one `s_cbranch`, and
  every value the branch needs is pinned in a vector register across it. The
  causal kernel spilled 90 VGPRs that way and none with
  `kittens::warpid_uniform()`.

* **An index inside a tile mask costs one register per element.** Written the
  obvious way -- ask `rt_base_coord` for (row, col) and evaluate the predicate
  -- a mask holds 8 loop-invariant registers, which LLVM hoists into the
  preheader and spills. Writing it as `dbase >= thr` with a scalar threshold is
  *worse*: LLVM reassociates straight back. The fix is to fold the index into
  the lane term so the per-element part is an integer literal after unrolling,
  which is one register for the whole tile. `detail::diag_fill` and
  `detail::axis_fill` in `conversions.cuh` now do this for all eight fills.

* **Branches in the hot region are a register-allocation lever, and the trade
  is asymmetric.** A mask that fires on one block in the whole loop (the
  backed-up last KV block) is worth a uniform branch: 224 VGPRs and no spill
  with it, 116 B/lane without. A mask that fires on nearly every block (causal)
  is worth no branch at all: 226 and none without it, 156 B/lane with it.

The last 2% was one thing the IR could not say. The handwritten kernel skips
the online-softmax rescale when the running max did not grow -- 64 `v_mul_f32`
per KV block at head_dim 128, against that block's 32 WMMAs -- and that needs a
predicate that reduces *out of* the register file and into control flow, which
no elementwise op can produce. `hk.s_any_ne` is that op, and it closed the gap
exactly. Causal then went ahead on longest-processing-time-first dispatch: the
work per workgroup is a ramp, so reversing the q index starts the long ones
while there is still short work to fill the slots they free.

---

← [Phase 3](02-phase-3-gemm.md) · [Development record](README.md) · [Phase 5](04-phase-5-autotune.md) →
