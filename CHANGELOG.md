# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## Unreleased

What is on main after 0.8.0. `pip install hk-dsl` (0.8.0) does not have it;
the API reference marks such names *New after 0.8.0*.

**gfx950 paged prefill: long KV walks cut over the cached prefix.** A
short chunk on a long prefix, or a batch on few heads, is a few query tiles
of 256 rows walking the whole context, with most of the machine idle: 4096
cached + 512 new tokens on 32/8 heads is 64 workgroups for 256 CUs.
- When the tiles times the heads come short of 256 workgroups,
  `hk.ops.paged_prefill.split_plan` cuts each tile's walk over the prefix
  every one of its queries sees, at least 16 64-key blocks a piece, the last
  piece taking the rest with the diagonal. The cut tiles write normalised
  partials and (m, l) (`attn_cdna.attn_kernel(kv_split=True)`,
  `split_kernels_for`), and `attn_cdna.prefill_merge_kernel` weighs them.
- Its entries go longest first across requests. On one or two kv heads the
  vLLM backend takes that order even with nothing to cut: 1.2-1.4x on a batch
  of prompts there. On more kv heads that order cost 2-4% of L2 reuse, and
  the backend keeps the single-tile work list unless the plan cuts.
- The vLLM backend plans it in the metadata builder, from
  `seq_lens_cpu_upper_bound` (exact for prefill rows); without it the
  prefill is the single launch as before.
- Measured on the attention task's prefill grid (MI355X; 32/8, 64/8, 32/4 and
  8/1 heads; 13 batches from fresh prompts to 65536 + 2048 chunks), same
  process, against AITER's Gluon unified attention, FlyDSL's paged varlen and
  AITER's varlen:
  - hk was 0 WIN, 18 tie, 34 LOSS; it is 18 / 16 / 18;
  - geometric mean 0.93x -> 1.20x;
  - 1.29x main on average, up to 6.9x: 32768 + 512 on 8/1 heads takes 101 µs,
    where it took 697 (AITER Gluon 602);
  - no batch slower than main beyond 1.1%.
- Cut tiles' outputs change by the order partials are folded in (within the
  fp32 check; deterministic). Uncut ones are bitwise as before.

**gfx950 causal attention walks query blocks in pairs.** A causal workgroup of
`hk.ops.attn_cdna` (what `hk.attention` and the SDPA drop-in run on gfx950)
now walks query block nqb - 1 - y and then block y, so every workgroup does
the same work. Before, longest-first order inside each batch still ended
grids with few workgroups a batch on a round of long tiles.
- Outputs are bitwise the same (all 42 causal points of the attention grid,
  head_dim 128, 8/1 to 64/8 heads, N 1k-32k).
- Same process, Q/K/V rotated, against the single-block walk:
  - 8 query / 1 KV heads (a 70B-class model at TP8): 1.26-1.30x at N up to 4k;
  - 16 / 2 heads: 1.10-1.12x at 1k-2k;
  - 40 / 8 heads: 1.01-1.03x;
  - the rest within ±1.4%.
- At 8/1 and 16/2 heads, short N, hk now ties AITER's asm `fmha_v3`, where it
  had been at 0.74-0.92x.
- `attn_cdna.kernel_for(..., pairs=False)` and the `pairs` knob of
  `hk.schedules.AttnCdna` give back the single-block walk. The spatial
  numbering and the paged prefill do not change.

**K2 below its MIN_M** (gfx950, `vllm_mxfp8.RANGES`). DeepSeek-V4.1's MXFP8 linears at decode
and mid-size M were measured against vLLM's kernel, FlyDSL's, AITER Triton's and hipBLASLt's, route
level with each side's activation quant, weights rotated, two sweeps: hk now takes engram wkv
(12800 x 6144) from M 160, wo_b (5120 x 4096) from 416, the shared expert's w2 (5120 x 1152) from
512, wq_b (16384 x 1280) from 576, the shared expert's w1|w3 (2304 x 5120) from 960 and fused
wq_a|wkv (1792 x 5120) from 1216, in the ranges where it won by >= 3% in every round of both
sweeps -- 1.04-1.61x the best of the four -- on the four-warp kernel at 128 x 128 or the
eight-warp one. Bitwise vLLM's output at every point. At M <= 128, and for the indexer's wq_b
below its MIN_M, no hk kernel wins; those calls stay with vLLM.

**Host time per call** (#91, cross-arch). An engine route's Python no longer
re-decides per call what only the call's shapes decide, and code-object
kernels launch from a per-shape memo without the torch dispatcher
(`hk.runtime.hostpath`, [integration](docs/integration.md#host-time-per-call)).
Same kernels, same arguments, bitwise the same outputs; `HK_HOST_FASTPATH=0`
restores every call's previous path.
- Host µs per call, before -> now: `vllm_gemm` decode 38.5 -> 14.0 and
  prefill 46.5 -> 17.6 (W7900D); `vllm_int4` 33.6 -> 12.5 (W7900D) and
  19.9 -> 9.4 (MI355X); K3 attention 82.5 -> 49.2; K1 167.1 -> 122.6, and a
  MoE call K1 leaves to AITER 252.6 -> 208.6 (AITER's own: 202.7).
- Eager decode of Qwen3-8B-AWQ: 1.27x tokens/s on MI355X, 1.29-1.37x on
  W7900D, at 1 and 16 requests; DeepSeek-V4.1-Flash TP2 1.04-1.07x. Prefill
  and GPU-bound decode (Qwen3-8B bf16) unchanged.
- Code-object launches in C (#92) where hk._mlir has `_hkLaunch` (built with the bindings): 2.9 µs a
  launch on the W7900D and 4.6 on the MI355X, under the pybind entry point's 3.4 / 5.3 (the
  Python launcher: 5.4 / 6.6).
- New: `hk.runtime.hostpath` (`ENABLED`, `set_enabled`),
  `hk.runtime.codeobj.direct`, `hk.runtime.torch_ext.launcher`,
  `CodeObjectKernel.launcher` (`native=`, #92), `hk.ops.w4a16.plan`. They are runtime and
  integration names, which the API reference does not cover, so they carry
  no mark there; [the runtime](docs/runtime.md) and
  [integration](docs/integration.md#host-time-per-call) describe them.

**DeepSeek-V4.1 routes, retuned from the MI355X scoreboard** (#84, gfx950, real weights, TP2; every
point against the best of AITER's tuned FlyDSL, FlyDSL, AITER Triton, hipBLASLt and vLLM's kernel).
- K1 (`vllm_moe_a8w4`) takes M >= 2560, not 8192 (`HK_MOE_A8W4_MIN_M`), with #85's kernels: on
  the real router's choices 1.03-1.08x AITER at 2560, 1.08-1.20x at 3072-4096 and 1.27-1.38x
  from 4608 to 8192, layers 20 and 30. With uniform routing, which DSV4.1's router does not
  produce, AITER is 3-11% faster from 2304 to 3584; from 4096 hk wins there too (the reasons
  are at `MIN_M`).
- K2 (`vllm_mxfp8`) takes the indexer's wq_b (N 4096, K 1280) from M 2825, 1.08-1.20x, and
  engram's wkv (12800 x 6144) from M 1024 instead of 2048, 1.29-1.79x; per (N, K) in
  `MIN_M_NK`. Bit-identical to vLLM's kernel at every new point, ragged M included.

**fp8 GEMM on gfx1201** (#76, RDNA4). `hk.ops.scaled_mm(a, b, scale_a, scale_b,
*, bias=None, out=None)` is `torch._scaled_mm`'s product, bf16
`(A @ B.T) * scale_a * scale_b (+ bias)` from OCP fp8 operands with fp32 accumulation, in
`hk.ops.matmul`'s (N, K) layout, with per-tensor or per-token x per-channel scales. It runs on
gfx1201's fp8 WMMA: `hk.ops.gemm`'s kernel takes `op_dtype=hk.fp8e4m3` / `hk.fp8e5m2` there,
and gfx1100 (no fp8 WMMA) and gfx950 (whose fp8 GEMMs are gemm_cdna's) refuse it. On a
Radeon AI PRO R9700 against `torch._scaled_mm` (hipBLASLt), in one process: 240 vs 236 TFLOP/s
at 4096³ per-tensor; per-token x per-channel 1.28-1.99x at M 128-4096 (N 4096 / 14336); a tie
at M 1-64 with K 4096, and 0.89x at M 1-64 with K 14336 (decode shapes, not tuned yet).

**K1 MoE a8w4 at prefill M: 1.34-1.49x AITER** (gfx950). On DeepSeek-V4.1-Flash's TP2 shapes with
real weights and real routing, against `aiter.fused_moe`'s tuned FlyDSL kernels (MI355X scoreboard
harness, four runs): 1.42-1.49x at M 8192 and 1.34-1.36x at 16384, from 1.35-1.37x / 1.22-1.24x
before; 1.14-1.47x at M 4096-7707, and 1-4% less time than before at M 2560-3584. Outputs
are bit for bit those of before; decode M unchanged.
The route's thresholds are not touched.
- `store_scatter` stores 16x16 fp32 fragments as bf16 in 16-byte runs (`v_permlane16_swap` pairs
  lanes l and l+16): half the store instructions. Stage 2's scattered top-k rows were a third of
  its time.
- `Global.scatter(..., staged=N)` reads its row ids from an LDS copy made by
  `g.gather_prep(ids, at, N, raw=True)` (both new after 0.8.0). A global id load in an epilogue is a vmcnt
  wait, and on gfx950 that retires, in order, every load and store issued before it: in the
  resident stage 2, the next column tile's weight prefetch and the epilogue's own stores. Stage 2
  now stages its block's ids, and loads its rows' routing weights once, not once a column tile.
- Both stages issue the next token fragment's LDS read before the current fragment's MFMAs
  (`stage1_sorted(xpipe=True)`, the default).
- `load_fp4_frags` builds its per-expert buffer resource with a 32-bit compare and a select: the
  64-bit bounds check was two scalar branches at every K step.

**Attention on gfx1201** (#77, RDNA4). `hk.attention` (and the SDPA drop-in) runs its own
tiling on gfx1201 from 256 query rows: 16 warps (a 256-row Q tile), a head-first grid
that starts the longest rows of every head first, and, causal, warps that skip the KV
blocks they precede. At n 4096, GQA 4, head_dim 128 it ties AITER's gfx1201 FlyDSL kernel
(1.40 vs 1.41 ms causal, 2.65 vs 2.69 full; gfx1100's tiling took 1.71 / 2.78). The causal
KV block depends on the compiler: 64 rows under ROCm 7.2.3's clang 22, 32 under the
R9700 vLLM image's clang 23, where 64 is 16-20% slower. A shorter Q keeps gfx1100's
192-row tiling, so what `hk.attention` accepts is the same on every RDNA arch.
`attn_kernel(warp_skip=True)` without `causal` raises.

**Fixes**
- **gfx950: `hk.ops.gemm.matmul` raised `ResourceError` in the vLLM image** (#84) at M 192-512 for
  N 28672 / 14336, K 4096: its table routes there to `dp128x224`, which ROCm 10's hipcc builds
  with a 13-VGPR spill, and the spill gate refuses it. The route now records a refused candidate
  (once a process, with a line on stderr) and takes the next-fastest one the table measured at
  that point (`pp128`, `ppk2`), else the rule's next; a refused kernel is never run.
- **A kernel registered as a torch op through libtorch's headers did not
  compile against TheRock's torch 2.13** (#79), the torch of the Radeon AI PRO
  R9700 vLLM image (ROCm 7.14), whose ROCm headers are hipified the newer way
  (`c10::cuda::CUDAGuard`, no `c10::hip::HIPGuard`): the C++ shim
  `hk.codegen.scaffold` writes for `hk.runtime.torch_ext` failed with
  `no type named 'HIPGuard' in namespace 'c10::hip'`. It now picks the names by
  `C10_HIP_API`, which only the older headers define; with those (rocm/pytorch's
  torch 2.10, the vLLM 0.31 venv's 2.13, the gfx950 images) it builds the same
  code as before.
- **The LDS wait check refused a correct kernel** (#87). `hk.runtime.cocheck`'s scan walked the code in
  address order and carried a block's in-flight LDS loads past an unconditional `s_branch` into a block
  that code never reaches. It now follows branches: a forward branch carries its states to its target.

**CI and tests**
- A fourth GPU-tier job, `gfx1201`, on a shared RDNA4 box (Radeon AI PRO R9700), with an
  ephemeral runner registered for this repository. It is continue-on-error and not a required
  check until the RDNA4 port's tier is green (#67,
  [testing](docs/testing.md#the-gfx1201-runner)).
- The `gfx1201` job is no longer continue-on-error, now that the RDNA4 port's GPU tier is green
  there: a red gfx1201 fails the run (#97). Whether it is one of main's required checks is a
  branch-protection setting.

## 0.8.0

hk-dsl 0.8.0: workgroup placement as a compiler pass on gfx950; `hk.map_cpp`,
i32 strips, and the strip ops and element exchange a megakernel's roles need; gfx1201 (RDNA4) on
hardware; and reproducible wheels, one `manylinux_2_35` wheel per interpreter.
hk stays on by default in vLLM and SGLang for the routes validated on your GPU:
`HK_DISABLE=1` turns every route off, `HK_<ROUTE>=0` one of them
([integration](docs/integration.md#on-by-default-and-turning-it-off)).

On PyPI: `pip install hk-dsl`, for CPython 3.10 and 3.12 on Linux x86_64 with
glibc 2.35 or newer (Ubuntu 22.04 and later, and vLLM's ROCm images): one
`manylinux_2_35` wheel per interpreter. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed. The API reference marks what 0.8.0 adds *New after
0.7.0*.

    pip install hk_dsl-0.8.0-cp312-cp312-manylinux_2_35_x86_64.whl

**Workgroup placement as a pass** (#63, gfx950; design in
[docs/design/placement-pass.md](docs/design/placement-pass.md), #62).
Which XCD runs which workgroup, as one mechanism instead of hand-written
remaps:
- `hk.kernel(placement=None | "auto" | hk.Placement)`. `hk.Placement.cluster()`
  runs one sharing cluster an XCD at a time, its key found by the analysis
  (`hk.placement.analyze`: each global read's sharing key over the grid axes)
  or named with `hk.Digit`s; `hk.Placement.chunked(chunk, tail)` renumbers
  `block_idx.x` alone. `"auto"` launches the cluster placement where
  `hk.placement.choose` takes it: not for order-dependent or resident
  kernels, from five KV streams an XCD, and not for non-uniform work under
  twelve rounds an XCD (thresholds measured in #55). `Kernel.placed()` and
  `Kernel.placement_choice()` give the placed kernel and the decision.
- Outputs are bitwise the unplaced kernel's. `attn_cdna`'s `mapping="spatial"`
  (#55) is now `hk.Placement.cluster()`, and `gemm_cdna` / `gemm_cdna_dp`
  take `Placement.chunked` instead of their inline remaps: 11 GEMM shapes
  bitwise, new/old 0.997-1.007; 15 attention points bitwise, 0.993-1.003 of
  #55's hand-written numbering.
- `hk.arch.xcd_count()`: KFD's `num_xcc` (MI350P / DPX: 4, MI300A: 6), the
  arch's default otherwise, `HK_XCDS` to override. With 1 (gfx1100) every
  placement is the identity.
- `python -m hk.check FILE.py:NAME --placement --inputs ...` checks that a
  launch's placement covers its grid exactly once, evaluating the decode the
  kernel's own emitted ops compute ([check](docs/check.md#which-xcd-runs-which-workgroup---placement)).
- **M2** (#65): K2 (`mxfp8`), K1 (`moe_a8w4`) and the stream-K plan take it too, so hk.ops has no
  hand-written XCD remap left (and no `NUM_XCDS`); persistent kernels use
  `hk.placement.renumber` / `hk.placement.xcd_range`. K2 at the 18 DSV4.1 points bitwise,
  0.992-1.011; K1 bitwise, and **1.19-1.24x faster** at M where its old remap fell back to the
  hardware's order (stage-1 grid not a multiple of 8: M 9500, 10000, 14000, 15001), within 0.1%
  where it applied. `HK_XCDS=1` turns every placement off.
- A user guide, [docs/placement.md](docs/placement.md) (when to use it, examples, the measured
  tables, limits, turning it off); the design document rewritten as built; an
  [hk.placement](docs/api/hk.placement.md) API page. `hk.Digit.of`'s radix may be spelled
  `"tensor.dim"` (`"k.depth"`).

**`hk.map_cpp` and i32 strips** (#53, gfx950;
[the language](docs/language.md#elementwise-c-over-strips-and-i32-strips-gfx950)).
- `hk.map_cpp(expr, *args, dtype=None)` evaluates one C++ expression for each
  element of its strip arguments (`x0`, `x1`, ... bound to the elements, or
  to an `hk.Int` / Python int), into an fp32 or i32 strip. It is for what a
  kernel must reproduce bit for bit and the operators cannot say: bit
  manipulation, a fused multiply-add, one particular instruction. The
  expression compiles with floating-point contraction off (`a * b + c` is two
  roundings; an fma is `__builtin_fmaf`). Strips of fp32 and i32 together
  must say `dtype=`; wave32 targets refuse it.
- Strips may hold i32: they load, store, convert with `.to` and are
  `hk.map_cpp`'s arguments and results. Operators and reductions on them are
  refused, so integer arithmetic is written in `hk.map_cpp`.

**Strip ops for megakernel roles** (#72, gfx950), which DeepSeek-V4.1's K1
written in hk uses (bitwise to the hand-written K1 over ~496k layer calls a
rank):
- `hk.iota(ty)`: an i32 strip whose element (r, c) is c.
- `Global.load_elems(idx)` / `store_elems(val, idx)`: element gather and
  scatter at flat indices; an index below zero or past the tensor reads 0 and
  writes nothing.
- `Tile.group_sum` / `group_max` / `group_min(cols, lanes=, xor=)`:
  reductions over groups of lanes, in a stated order like
  `sum(axis=1, lanes=, xor=)`.
- `Tile.lane_gather(lanes)` (`ds_bpermute` of each element), and
  `hk.mfma_strip("f32_16x16x4", a, b, acc)`, one raw MFMA on strip registers
  for reproducing another kernel's MFMA bit for bit.

**Element exchange for megakernel hand-offs** (#73, gfx950), also from K1 in
hk:
- `Global.post_elems(val, idx, tag, runs=, bf16=)` / `take_elems(idx, tag,
  runs=, bf16=, timeout_s=)`: tagged mailbox pairs, each value and its tag in
  one 8-byte scoped store, as `hk.comm.post` / `take` per element. `runs=True`
  moves a lane's run as 16-byte pairs of pairs; `bf16=True` packs two bf16 a
  pair (values must already be bf16; refused without `runs`). A wave with no
  live pair polls nothing, and a timeout traps.
- `SharedTile.load_elems` / `store_elems`: an fp32 LDS tile addressed element
  by element, with `ds` operations (not FLAT, which would also count in vmcnt).
- Scoped `load_elems` / `store_elems`; `store_elems(runs=True)`, one store a
  run, fp32 to e4m3 packed four at a time; scoped register-tile loads take
  `bounded=n`.
- `hk.uniform(cond)` / `hk.if_(..., uniform=True)`: a branch declared
  workgroup-uniform may hold barriers, which is a persistent kernel's role
  dispatch; a divergent enclosing branch is still refused.

**gfx1201 (RDNA4) on hardware** (#70, #71). gfx1201 goes from compile-only to
hk's GPU tier passing on a Radeon AI PRO R9700 (914 passed, 0 failed;
[chapter 47](docs/history/47-gfx1201-on-hardware.md)):
- HipKittens moves to `9d182f93`: `include/rdna4` with gfx12's accumulator
  layout as measured (every MMA unit test had failed on the old one) and
  rdna3's work since the fork re-derived for gfx12. gfx1100 and gfx950 are
  unaffected: `include/rdna3` and the CDNA trees are byte-identical.
- The LDS reads a tile load issues come from the target: one on gfx1201, where
  hk assumed gfx1100's two. hk-lds-pipeline derives every LDS wait from that
  count, and a count too high makes every wait too shallow.
- The wait gate reads gfx12's waits (hex immediates, `s_wait_loadcnt_dscnt`,
  `s_wait_storecnt_dscnt`), the target is measured (VGPR granule 24, LDS pool
  128 KB, i8 WMMA, strips), and the schedules' register model is fitted on
  gfx1201's own builds.
- Its ceilings are measured on the R9700 (#74): bf16 / f16 WMMA 197 TFLOP/s,
  fp8 372, iu8 374 TOP/s; an add at 1.81 TB/s inside the 64 MB Infinity Cache,
  0.62 TB/s past it. `hk.check`, `hk.profile`, `hk.tune` and the scoreboard
  use them (they fell back to gfx1100's, which would have called hk's own 4096³
  GEMM faster than the card can), and `hk.tune` has a GEMM space there.
- The scoreboard on gfx1201, out of the box (gfx1100's plans), every arm in one
  process, with FlyDSL and AITER arms (#74,
  [chapter 48](docs/history/48-gfx1201-ceilings-and-first-scoreboard.md),
  [performance](docs/performance.md#part-c-rdna4-gfx1201-one-radeon-ai-pro-r9700)):
  hk's GEMM is the fastest at the five shapes (1.03-1.07x the best other arm).
  It loses attention to AITER's (0.83x causal, 0.97x full, at 4096 tokens,
  GQA 4) and fused add + rms_norm at 8192 rows (0.93x).
- No engine route is validated on gfx1201, so none is on by default there, and
  nothing is tuned there yet (its tables are gfx1100's).

**K1 (MoE a8w4) on gfx950, measured for this release** on the release's code
(after #85): real DeepSeek-V4.1-Flash weights, two layers, real, gate-only and
uniform routing, weights rotated past the caches, interleaved rounds in one
process; hk deterministic and within AITER's own error against an exact fp32
MoE everywhere.

| M | hk / AITER (tuned FlyDSL) | K1 now / K1 at #65 | K1 now / K1 before #65 |
|---|---|---|---|
| 64 | 0.77-0.84x | 1.26-1.45x | 1.12-1.20x |
| 128-512 | 0.74-0.85x | 1.05-1.30x | 1.04-1.31x |
| 2048 | 0.97-0.99x | 1.08-1.15x | |
| 4096 | 1.13-1.20x | 1.10-1.12x | |
| 8192 (routed) | 1.27-1.35x | 1.09-1.12x | 1.08-1.12x |
| 16384 (routed) | 1.16-1.23x | 1.04-1.06x | |

(The "K1 at #65" figures at M 2048, 4096 and 16384 compare each commit in its
own process, each normalised by its own in-process AITER; the other columns are
one process. The two methods agree within about 3%.)

- vLLM's K1 route takes M >= 8192, where hk wins at 12 of 12 points (before
  #85: 1.11-1.23x). Below that the route leaves the MoE to AITER, so decode in
  production is AITER's either way.
- At decode M (64-512) hk's K1 is still slower than AITER's tuned FlyDSL
  kernels, 0.74-0.85x, up from 0.53-0.78x at #65.

**Build gates**
- **The spill gate passes a scratch frame the code never touches** (#68). A
  kernel with ScratchSize > 0 and no VGPR spill builds when its code object has
  no `scratch_*` instruction, no `flat_*` memory instruction, no flat scratch
  and no private segment buffer (`cocheck.unused_scratch_frame`); the build
  records `scratch_note = "reserved-unused scratch frame"` and says so on
  stderr and in `build.log`. Found on K1's indexer build under ROCm 10 (36 B
  reserved, never used). A VGPR spill, any scratch access, and a code object
  that cannot be read are still refused.

**Fixes**
- **K1 MoE a8w4 at decode M ran slower after #65** (#85; M 64: 0.82-0.89x of
  K1 before #65, found by the MI355X scoreboard). The sorted kernels' grids are
  sized for the most sorted blocks a launch can have, and #65's
  `Placement.chunked(0, "spread")` dealt contiguous ranges of the whole grid, so
  the live blocks (235 of 390 at M 64) landed on the first five XCDs. They now
  renumber in the body over the live blocks only
  (`hk.placement.renumber(..., live=)`, new). Outputs are bitwise the same, and
  every M got faster (numbers in *K1 on gfx950* below).
- **gfx950: fp32 register tiles through LDS were FLAT loads and stores**
  (#59). `store(st<float>, rt<float>)` / `load(rt<float>, st<float>)` addressed
  the tile through a generic pointer, so every access was FLAT: slower to
  issue than `ds_read` / `ds_write`, and counted in vmcnt as well as lgkmcnt,
  so each wait for an LDS value also waited for every global load in flight.
  They are `ds_read` / `ds_write` now. Results are unchanged; affected are the
  cross-warp combines of fp32 partials (`hk.ops.decode_gemm`, `w4a16`, `moe`)
  and user kernels that move fp32 tiles through LDS on gfx950.

**Wheels** (#66)
- cp312 is one `manylinux_2_35` wheel, built on glibc 2.35 in vLLM's ROCm
  image, like cp310. Through 0.6.0 cp312 was built on Ubuntu 24.04 and needed
  glibc 2.38, so `pip install hk-dsl` found no cp312 wheel in vLLM's ROCm
  images; 0.7.0 added a second, `manylinux_2_35` cp312 wheel. (0.7.0's PyPI
  description still says CPython 3.12 needs glibc 2.38: that page cannot be
  edited after upload, and 0.8.0's states the wheels above.)
- `scripts/check_wheel_glibc.py` fails a wheel that needs a newer glibc (or
  GLIBCXX / CXXABI, with auditwheel) than its manylinux tag; CI's IR leg
  builds the cp310 wheel and runs it.
- Reproducible wheels: no build path in the binaries (`-ffile-prefix-map`,
  RUNPATH `$ORIGIN` at link time), entry times from `SOURCE_DATE_EPOCH`
  (the last commit's), fixed modes; the same commit builds byte-identical
  wheels ([building](docs/building.md#reproducible-wheels)).

**Docs**
- The API reference marks names against the 0.7.0 wheel (it marked them
  against 0.6.0, so `placement=` and `hk.arch.xcd_count` read as if 0.7.0 had
  them), and documents `hk.Placement`, `hk.Digit` and
  `hk.runtime.SymmetricBuffer`.
- [The language](docs/language.md) gains `hk.comm`, workgroup placement, the
  stated strip-reduction order (#52, #57), the bounded group copies on gfx950
  (#49) and LDS allocated in a runtime branch (#56).
- The docs site builds with `mkdocs build --strict` in CI: every page is in
  the nav (hk.profile, schedules, hk.tune and the end-to-end loop were not),
  heading ids follow GitHub's (so `#which-xcd-runs-which-workgroup---placement`
  resolves on both), and the links #54 broke (`#turning-them-on`) point at
  `#on-by-default-and-turning-it-off`.
- The install pages state 0.7.0's three wheels and that CPython 3.12 needs
  glibc 2.35, not 2.38; [the runtime](docs/runtime.md#which-gpu) has the arch
  detection order 0.7.0 introduced.
- [Releasing](docs/releasing.md): the release steps, and the documentation
  sync that must pass before a tag.
- Examples [14](examples/README.md) (an hk.comm task graph, gfx950) and
  [15](examples/README.md) (workgroup placement).
- `hk.map_cpp` and i32 strips in the language and the API reference (#81).
- No internal address, host name or user path in the repository (#83): docs,
  bench records and scripts use placeholders and environment variables, and
  `scripts/check_public_hygiene.py` (generic rules, an explicit allowlist)
  runs in CI's IR leg and before every tag.
- [docs/design/e2e-goal.md](docs/design/e2e-goal.md), the end-to-end goal
  refocused on three core kernels, and [docs/design/kda.md](docs/design/kda.md)
  (#20); bench/tune's record of T4 run 20261024-a (#22).

**CI and tests**
- Several gfx950 runners in parallel, each with its own GPUs and state, and
  GPU locks shared with manual measurements (#60,
  [testing](docs/testing.md#several-gfx950-runners)).
- A cancelled GPU job no longer leaves its container on the GPU (#69,
  [testing](docs/testing.md#the-gpu-tier)).
- Tests of gfx1100's measured data are pinned to gfx1100
  (`@pytest.mark.arch`, #64).
- The fenced paged-attention checks (`test_fenced.py`) build their reference
  on the host from one readback of each tensor, read twice and compared, and a
  relative-error failure prints both maxima. The one intermittent paged failure
  seen on gfx950, `relative error 9.73e+29`, matches an all-zero reference
  (0.97265625 / 1e-30), not a wrong output: the paged decode did not fail in
  thousands of repetitions with poisoned allocations and polluted LDS (#89;
  tests only, no kernel change).

**Compatibility**
- No public DSL name or parameter was removed or renamed since 0.7.0 (checked
  name by name against the 0.7.0 wheel). The modules' `NUM_XCDS` constants in
  `hk.ops.attn_cdna`, `gemm_cdna`, `gemm_cdna_dp` and `moe_a8w4` are gone
  (#65): the XCD count is `hk.arch.xcd_count()`.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-6`. 0.7.0's cached builds are rebuilt once.
- The spill gate can build a kernel that 0.7.0 refused: one whose only
  scratch is a reserved frame its code never touches (#68).
- gfx950's attention, GEMMs, K1 and K2 now number their workgroups through
  the placement pass: outputs bitwise as before. `HK_XCDS=1` turns every
  placement off.
- cp312 has one wheel, `manylinux_2_35`; 0.7.0's cp312 `manylinux_2_38` wheel
  has no 0.8.0 counterpart, and pip on a newer glibc takes the 2_35 one.

**Known issues**
- A GPU-tier CI run on MI355X (gfx950) once reported a mismatch in the
  fenced-allocator test of the paged-decode split path
  (`tests/hk/gpu/test_fenced.py::test_paged_decode_and_prefill`, fused decode,
  head_dim 64, 32 splits). It did not recur in later runs or in several
  thousand targeted repetitions with poisoned allocations and polluted LDS; the
  reported error value matches an all-zero host-side reference rather than a
  wrong kernel output. The paged decode is used only by the opt-in vLLM
  attention backend (`HK_VLLM_BACKEND=1`, off by default) and is not reached by
  any default-on route. The test has since been hardened (#89, tests only): it
  builds its reference on the host from one stable readback of each tensor, so
  a recurrence names its cause.
- gfx942: the synchronous group copies are still unbounded (#49): on gfx942 a
  `g.load` / `g.store` at a partial last tile reads and writes past the
  tensor. There is no gfx942 here to test the bounded form on.
- The gfx1100 defaults were not run end to end. Each gfx1100 route's evidence
  is its earlier opt-in run with real models; the default-on configuration
  itself was checked by unit tests, the GPU tier and the engine contract.
- The vLLM decoder attention backend stays off by default: it replaces ROCm's
  default decoder attention for every model and raises rather than falls back
  on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and other head
  sizes; validated on Qwen3-8B only, and on gfx950 slower at one request on
  long contexts (0.71-0.88x).
- SGLang-Diffusion installs made before 0.7.0 stay opt-in until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  spills and the spill gate refuses it (rocm/pytorch 7.2.3 builds it). There,
  `hk.ops.gemm.matmul`'s gfx950 route picks it for N 28672 / 14336, K 4096 at
  M 192-512 and raises `ResourceError` instead of falling back. vLLM's linear
  route (`HK_VLLM_GEMM`) is off on gfx950, so no default route reaches it.
- gfx1201 runs hk's GPU tier on a dev box, not in CI's required tiers; no
  route is validated or on by default there and nothing is tuned. gfx942 is
  compile-only.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

## 0.7.0

hk-dsl 0.7.0: hk is on by default in vLLM and SGLang, for the routes validated
with real weights on your GPU; `HK_DISABLE=1` turns it off. Also `hk.comm` for
hand-offs between gfx950's XCDs and between GPUs, and fixes for two silent
errors that 0.6.0's DSL has in user kernels: on gfx950 a synchronous `g.store`
at a partial last tile overwrites the memory after the tensor, and LDS
allocated inside a runtime branch reads back as 0. No shipped op or route was
affected by either.

On PyPI: `pip install hk-dsl` (CPython 3.10 or 3.12 on glibc 2.35+, Linux
x86_64). Three wheels: cp310 `manylinux_2_35`, and cp312 both
`manylinux_2_35` and `manylinux_2_38`; pip picks the newest tag the system's
glibc takes. The cp312 `manylinux_2_35` wheel was added on release day, for
glibc 2.35 systems such as vLLM's ROCm images (Ubuntu 22.04), where the
`manylinux_2_38` one does not install. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed (and a ROCm build of torch to launch).

    pip install hk_dsl-0.7.0-cp312-cp312-manylinux_2_35_x86_64.whl

(This paragraph first said "3.12 on glibc 2.38+", as do 0.7.0's PyPI page and
the GitHub release notes; the `manylinux_2_35` cp312 wheel makes 2.35 enough.)

**hk is on by default** (#54). With hk-dsl installed, its vLLM and SGLang
plugins (and the SGLang-Diffusion boot) apply every route that was validated
with real weights on your GPU: outputs bit-identical to the engine's, or error
within the engine's own noise. Before, every route needed its own `HK_*=1`.
Turning hk off is now for comparison experiments:

```bash
HK_DISABLE=1 vllm serve <model>        # stock: nothing patched, in every engine
HK_VLLM_MXFP8=0 vllm serve <model>     # one route off (HK_<ROUTE>=0)
python3 -m hk.integration --status     # every route on or off, and which rule decided
```

- **On by default** (the evidence for each route is in
  [integration](docs/integration.md#on-by-default-and-turning-it-off)):

  | GPU | engine | routes (switch) |
  |---|---|---|
  | gfx950 (MI355X) | vLLM | K2 MXFP8 linear for DeepSeek-V4.1 at M >= 2048 (`HK_VLLM_MXFP8`); K1 MoE a8w4 for DeepSeek-V4.1 at M >= 8192 (`HK_VLLM_MOE_A8W4`); AWQ 4-bit (`HK_VLLM_INT4`) |
  | gfx1100 (W7900D) | vLLM | unquantized linear (`HK_VLLM_GEMM`); bf16 MoE (`HK_VLLM_MOE`); AWQ 4-bit (`HK_VLLM_INT4`); the encoder SDPA (`HK_VLLM_SDPA`) |
  | gfx1100 | SGLang | linear (`HK_SGLANG_LINEAR`); MoE and the gfx1100 top-k fix (`HK_SGLANG_MOE`) |
  | gfx1100 | SGLang-Diffusion | DiT attention (`HK_SGLD_ATTN`); DiT linear (`HK_SGLD_LINEAR`) |
  | gfx1100 | `hk.ops.sdpa` | K3 DiT attention (`HK_SDPA_K3`) |

  On a GPU a route was not validated on, it stays off: the gfx1100 routes on
  gfx950 (vLLM's linear, MoE and encoder SDPA among them), and every route on
  gfx942 and gfx1201.
- **Stays opt-in:**
  - **ConvRot INT8** (`HK_SGLD_CONVROT=1`, or `HK_SGLD=attn,linear,convrot`):
    it is lossy by design (PSNR 30.4 dB from bf16), a model-level choice and
    not a toolchain default.
  - **The vLLM decoder attention backend** (`HK_VLLM_BACKEND=1`): see the
    known issues.
  - **`HK_MXFP8_ANY`**, K2 at shapes it did not measure (no real-weight run).
- **Turning routes off.** The first rule that applies decides:
  1. `HK_DISABLE=1` turns every route off, in every engine. Nothing beats it.
  2. An engine's switch at `0` (`HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0`,
     `HK_SGLD=0`) turns off that engine's routes.
  3. A route's own switch decides that route: `HK_<ROUTE>=0` off, `=1` on (also
     a default-off route, or a GPU it was not validated on).
  4. The legacy switches still mean what they meant: `HK_SGLD=attn,linear,convrot`
     turns on exactly those parts, `HK_SGLD=1` attention and linear,
     `HK_PATCH_VLLM=1` also the encoder SDPA; `HK_SGLANG_MOE=topk` and
     `HK_VLLM_BACKEND=<slot>` work as before.
  5. Otherwise the default applies.

  `0`, `off`, `no` and `false` are off; an empty value counts as unset.
  `python3 -m hk.integration --status` prints every route, whether it is on,
  and which rule decided.
- **The engine contract is checked at startup** (`hk.integration.contract`).
  Before it patches anything, a plugin checks every route it is about to apply
  against the installed vLLM / AITER / SGLang: each patched or fallback entry
  point exists, with the parameters the route uses. If any check fails, for
  example after a version change, the plugin patches **nothing** in that
  engine, prints one warning line, and the engine runs fully stock:

  ```
  [hk] WARNING: vllm: engine contract FAILED -- hk patched NOTHING in this engine, it runs stock (...)
  ```

  It is all or nothing per engine, never half-patched: a route that raises
  while it is applied undoes the ones applied before it, with the same line.
- **The first JIT compile says so.** A route's kernels compile on first use,
  seconds each, in every worker. If they were not pre-built, the first
  compile in each process prints one line that suggests
  `python3 -m hk.integration warm-routes` (and `--warm` for attention), run
  with the server's `HK_CACHE_DIR` before the server starts.
- **GPU detection reads the KFD topology.** The vLLM ROCm image sets
  `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`. Without `HK_ARCH`, hk took the
  list's first entry, so it read an MI355X as gfx90a, built for gfx1100, and
  with default-on it would have left every gfx950 route off.
  `hk.runtime.arch.detect()` now goes `HK_ARCH`, then a variable naming exactly
  one arch, then the KFD topology in sysfs (no HIP context), then torch and
  rocminfo, and uses a list's first entry only as the last resort.
- **Comparison scripts must say stock.** Unset no longer means stock, so an
  A/B's stock arm sets `HK_DISABLE=1`. `python3 -m hk.integration ab` /
  `headroom` and `tools/hk-bench/` do this themselves.
- **SGLang-Diffusion:** an install made before 0.7.0 keeps its old boot, which
  needs `HK_SGLD=1`. Run `python3 -m hk.integration sglang-diffusion install`
  again for the defaults; `... sglang-diffusion status` says when the boot is
  the old one.
- **Fixed:** with several routes on in one vLLM (int4 and MXFP8 on gfx950), its
  compile cache root was nested again in every worker
  (`hk-int4/hk-mxfp8/hk-int4/...`). It is now one directory named for all the
  routes (`hk-int4+mxfp8`), the same in every process.
- `hk.ops.sdpa.reset_stats()` also resets K3's counters.

**Fixes**
- **gfx950: synchronous `g.load` / `g.store` went out of bounds at a partial
  last tile** (#49). This affects 0.6.0's DSL. In 0.6.0 on gfx950, in a
  user-written kernel whose synchronous `g.load` covers a partial last tile
  (the tile's rows run past the tensor's last row):
  - the load reads up to a tile's rows past the tensor: an intermittent memory
    fault (10 of 10 fenced cases faulted, at M = 1, 63, 65, 130 and 1000);
  - its `g.store` **silently overwrites the memory after the tensor**: with
    slack after the output, a probe found 63 of 63 rows past it overwritten,
    with no error.

  No shipped `hk.ops` op and no vLLM / SGLang route was affected, in 0.6.0 or
  in this release: all run clean under an end-fenced allocator on gfx950 and
  gfx1100, and no shipped kernel issues a `g.store`. The fix: on CDNA, the
  synchronous `g.load` stops at the tensor's last row (rows past it read zero),
  and `g.store` drops the rows past it (`group_load_sync`, `group_store_bounded`
  in `include/hk/cdna.cuh`). `hk.ops.mxfp8.gemm`, the one shipped CDNA kernel
  with a synchronous group copy, is bit-identical and within 0.6% of its speed.
  gfx942 still uses the unbounded copies (no hardware to test). RDNA3 already
  bounded them. `tests/hk/gpu/test_fenced.py` now runs every `hk.ops` family at
  ragged shapes under the end-fenced allocator, in CI on gfx950 and gfx1100.
- **LDS allocated inside a runtime branch was not part of the launch's LDS**
  (#56). The `hk-lds-alloc` pass sized a kernel's launch from the
  allocations in the kernel's own block only, while the generated C++ bumps
  its allocator along the path the kernel takes. So a tile that a kernel
  allocated with `hk.lds.alloc` inside a runtime `if` (a branch on a device
  value) lay past the end of the LDS the launch requested: its reads returned
  0 and its writes were dropped, with no error. An allocation inside a runtime
  loop was past the end the same way. Found porting DeepSeek-V4.1's K1 to hk,
  where a reduction tile allocated in a role branch read back as zeros (87.5%
  of q wrong). Every released version through 0.6.0 has the pass in that form,
  so a user kernel that allocates LDS in a runtime branch or loop is affected
  there. No shipped kernel is: a scan of every kernel the IR and codegen tiers
  translate (both arches) and the gfx950 GPU suite builds found no LDS
  allocation outside a kernel's own block, in 0.6.0 or in this release. The
  pass now walks every region: the two arms of an `if` start at the same
  offset and the larger one continues, and an allocation inside a runtime
  `for` / `while` is refused, with a message to allocate it before the loop.
  `tests/hk/gpu/test_lds_branch.py` reads such a tile back across lanes (all
  zeros before the fix, exact after).
- **`mxfp8.quant` read up to 63 rows past the activation** (#47). At
  M % 64 != 0, K2's fused quant (#41, after 0.6.0) copied whole 64-row tiles of
  x. When x ended an allocator segment, that read took a GPU memory fault in
  vLLM with `HK_VLLM_MXFP8` on: the first three real-weight DeepSeek-V4.1 runs
  with K1 and K2 on faulted, and since it depends on the allocator's layout,
  later runs of the same arm completed. It only ever affected main after
  0.6.0; the released 0.6.0 does
  not contain `mxfp8.quant`. The last block's copy now stops at row M, with the
  output unchanged: 48 of 48 ragged-M cases are bitwise equal to vLLM's quant +
  GEMM. The end-fenced allocator test (`test_fenced.py`, which #49 extended)
  came with it.

- **A stated strip-reduction order was not the documented one for products**
  (#57). With #52's `lanes=` / `xor=`, `(x * x).sum(axis=1, lanes="seq",
  xor=...)` compiled to fused multiply-adds: HIP's FMA contraction fused each
  product into the running sum across hk's inlined ops, so the sum was not the
  documented one of rounded products (4288 of 32768 lanes differed from it on
  MI355X). Only an ordered reduction of products was affected, and no `hk.ops`
  kernel uses one. #52 came after 0.6.0, so 0.6.0 is not affected. The ordered
  reduction now keeps each summand's rounding (no fma reaches in), and
  `sum(..., times=y)` is the fused form, stated.

- **Tests: `test_sdpa` compares against fp32 attention on torch's math
  backend** (#61), test-only: no kernel or library change. Its intermittent
  head_dim-64 failure on a shared MI355X was torch's reference, not hk: in the
  vLLM ROCm 10 image, torch's fused bf16 SDPA (AOTriton 0.13.50) gave wrong,
  nondeterministic tiles under same-GPU load, while hk's kernel gave identical
  bits on every call.

**New in the language: hk.comm (gfx950)** -- hand-offs between the XCDs of
one GPU and between GPUs, each lowered to what a memory-model litmus suite
(20000 rounds a combination, across XCDs and across two MI355X) found correct.
The memory model is in [docs/design/comm.md](docs/design/comm.md).
- **Scoped tensors, signal / wait, symmetric memory** (#38):
  - `hk.Tensor[dtype, "device" | "system"]`: every load, store and scalar
    atomic through it carries the scope's cache-policy bits (on gfx950 both
    sides of a hand-off need them, so the scope belongs to the tensor);
  - `hk.comm.signal` (drain, barrier, one scoped flag store) and
    `hk.comm.wait` (a scoped poll with a timeout that sets an error element,
    or traps, instead of hanging);
  - `hk.comm.arrive` / `hk.comm.barrier`: an epoch'd grid barrier;
  - `hk.kernel(..., resident=True)` and `Kernel.resident_capacity()`: a grid
    that cannot all be resident raises `ResourceError` instead of deadlocking;
  - `hk.runtime.SymmetricBuffer`: one buffer per rank with every peer's mapped;
    while one is open, a launch refuses it bound to an unscoped parameter.
- **Every scalar fetch-op, and scoped col_vecs** (#45): `View.atomic_min_scalar`,
  `atomic_or_scalar`, `atomic_and_scalar`, `atomic_xor_scalar` and
  `atomic_exch_scalar` beside `atomic_add_scalar` / `atomic_max_scalar`, each
  returning the old value, plain or at a scoped tensor's scope; a row tile's 32-bit `col_vec` loaded from or
  stored to a scoped tensor (a per-row scale next to its tile).
- **Work queues and per-XCD aggregation** (#48): `hk.comm.claim` (a work
  queue's take: one scoped atomic add, no drain) for persistent kernels and
  task graphs; `hk.arch.xcd_id()`, read from the hardware; and
  `hk.comm.xcd_barrier`, a grid barrier with one global atomic per XCD a round
  (2.53 us against 3.71 us flat at 256 workgroups). The tile-signal pattern
  through a double-buffered symmetric buffer is tested across two GPUs.
- **Tagged mailboxes** (#50): `hk.comm.post` / `hk.comm.take` send a one-int32
  message and its tag as one 8-byte scoped store and one poll, no drain: a
  1.51 us round trip between two MI355X, against 3.40 us for a tile with
  signal / wait. A mailbox orders nothing else.

**New in the language: strips**
- **A stated row-reduction order** (#52, #57, gfx950): `t.sum(axis=1,
  lanes="seq" | "tree", xor=(32, 16, 8, 4, 2, 1))`, and the same for `max`,
  `min` and `prod`, for a result that must match another kernel's bit for bit.
  The guaranteed order:
  - a lane's own values are combined left to right (`"seq"`) or as a balanced
    tree (`"tree"`), then the lanes over the given xor offsets in that order,
    each of 1, 2, 4, 8, 16 and 32 once;
  - the strip's values are summed as they are: `(x * y).sum(...)` adds the
    rounded products, and the compiler does not fuse them into the sum;
  - `x.sum(axis=1, lanes="seq", xor=..., times=y)` is the fused form, spelled
    with fma: `s = fma(x0, y0, r(x1 * y1))`, then `s = fma(xi, yi, s)`, then the
    xor sums. That is what a compiled `s += a * b` loop computes, and it is
    dsv41_mono's and aiter's sum of squares in their norms (`times=x`).

  Without the arguments nothing changes (a tree, then 1 .. 32). Refused on
  wave32 targets, on columns and on WMMA tiles.

**Kernels and serving**
- **gfx950 attention: a spatial workgroup numbering, routed by measurement**
  (#55). `attn_cdna` (dense flash attention and paged prefill) gains
  `mapping="spatial"`: each XCD takes a contiguous range of (batch, KV head)
  clusters, so the workgroups resident on an XCD read one K/V stream. Outputs
  are bitwise the default numbering's. `hk.ops.attn` and the SDPA ops pick it
  when an XCD would otherwise hold five or more KV heads' streams, except for
  causal grids with fewer than twelve rounds of workgroups an XCD. Against the
  default, MHA at D 128 and 8K-128K tokens: 1.05-1.18x at 128 KV heads, up to
  1.08x at 64 and 1.05x at 40 (at 128 heads, 32K, the L2 hit rate goes from
  0.47 to 0.96). The routed choice was never below the default at the 74 timed
  points. The paged prefill accepts the option but is not
  routed (unmeasured).
- **K2's activation quant and its scale layout in one launch** (#41):
  `hk.ops.mxfp8.quant(x, packed=True)`, and `hk.mx_quant_store` writes the
  packed scale layout. The vLLM route makes one opaque call
  (`hk_vllm::mxfp8_linear`) instead of vLLM's quant, hk's scale packing and the
  GEMM: 1.01-1.58x per call at the 12 DeepSeek-V4.1 TP2 shapes, bit-identical.
- **Serving readiness** (#43, #44): the engine-contract test, a gfx950 GPU tier
  in the vLLM deployment image (ROCm 10), and `python3 -m hk.integration
  warm-routes --tp 2`, which builds the DeepSeek-V4.1 routes' kernels before
  the engine starts (31.2 s on an empty cache, 0.9 s warm).
  [docs/design/serving-readiness.md](docs/design/serving-readiness.md) lists what
  0.6 lacked for a deployment; [docs/design/comm.md](docs/design/comm.md) has
  hk.comm's roadmap (#51); [docs/design/placement-pass.md](docs/design/placement-pass.md)
  designs workgroup placement as a compiler pass (#62, design only).
- CI: pushes run only on main, and a PR's superseded runs are cancelled (#42);
  GPU-tier runs write no core dumps (#46).

**Compatibility**
- **Unset now means on.** With hk-dsl installed, vLLM and SGLang (and
  SGLang-Diffusion, after `install`) run with hk's validated routes unless
  `HK_DISABLE=1` or a route's switch is `0`. Every existing `HK_*=1` still
  works.
- No public DSL name or parameter was removed or renamed since 0.6.0 (checked
  name by name against the 0.6.0 wheel). Changed signatures only gained
  parameters: the strip reductions' `lanes=` and `xor=` (and `sum`'s
  `times=`), and `hk.kernel`'s `resident=`. One integration op was
  replaced: vLLM's MXFP8 route calls `hk_vllm::mxfp8_linear`, which quantizes
  inside, instead of `hk_vllm::mxfp8_block32` (#41).
- On gfx950, a synchronous `g.load` now reads zero for a tile's rows past the
  tensor, and `g.store` no longer writes them. In-bounds rows are unchanged.
- An LDS allocation inside a runtime loop is now refused at build time (it
  was silently past the launch's LDS); allocate it before the loop. One inside
  a runtime branch now counts toward the launch's LDS, so such a kernel asks
  for more LDS than 0.6.0's did.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-5`. 0.6.0's cached builds are rebuilt once.

**Known issues**
- **gfx942: the synchronous group copies are still unbounded** (#49). There
  is no gfx942 here to test the bounded form on, so on gfx942 a `g.load` /
  `g.store` at a partial last tile still reads and writes past the tensor.
- **The gfx1100 defaults were not run end to end.** Each
  gfx1100 route's evidence is its earlier opt-in run with real models. The
  default-on configuration itself was checked by the unit tests, the gfx1100
  GPU tier and the engine contract in each engine's environment, not by a
  model run on a W7900D with nothing set. The real-weight default-on smoke ran
  on MI355X (DeepSeek-V4.1-Flash, TP2).
- **The vLLM decoder attention backend stays off by default.** It replaces
  ROCm's default decoder attention for every model, and it raises rather than
  falls back on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and
  other head sizes. It was validated on Qwen3-8B only, and on gfx950 it is
  slower at one request on long contexts (0.71-0.88x).
- **SGLang-Diffusion installs made before 0.7.0 stay opt-in** until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  at 130x1792x768 spills and is refused. rocm/pytorch 7.2.3 builds it.
- gfx1201 and gfx942 are compile-only, and no route is on by default there.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

See docs/api (marks "New after 0.6.0"), [integration](docs/integration.md),
[docs/design/comm.md](docs/design/comm.md) and
[docs/testing.md](docs/testing.md).
