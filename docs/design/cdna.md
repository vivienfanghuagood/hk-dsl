# hk-dsl on CDNA3 / CDNA4 (gfx942 / gfx950): assessment and prototype

> Hardware: one Instinct MI355X (gfx950) of a shared 8-GPU node; ROCm 7.1 hipcc on the
> host, `rocm/pytorch:rocm7.2.3 … py3.12 torch 2.10` for anything that launches. gfx942
> (MI300X) is compile-only: no part here. Sections 1-5 are the assessment as written on
> branch `cdna` (based on `fdb87f1`); their file:line references are to `fdb87f1` for
> hk-dsl and to the submodule at `cbd4effc` for HipKittens. The history is
> [chapter 29](../history/29-cdna.md).

## Status

What has landed since the assessment, in the order of section 6:

| step | what | where |
|---|---|---|
| targets, wave64, fragment, two waits | gfx942/gfx950 as data; AGPRs charged in one file; `rt_16x32` operands; `lds_wait_for` after CDNA LDS reads, `vm_wait<0>` after the DMA | `python/hk/target/gfx9{42,50}.py`, `HKToCpp.cpp` |
| 1. library | `include/hk/cdna.cuh`: a strided gl (`kittens::strided_gl<T>`, the globals on wave64 targets); register-tile load/store through one descriptor per tile, which is all of `load_bounded`/`store_bounded`; int8; `store_scalar`, `load_scalar`, `wg_arrive`; `atomic_add`, `atomic_scalar`; `load_gather`, `store_scatter`; `philox_uniform`; GELU on CDNA3; RNE float->bf16 on every hk store and CDNA3 cast; lane-direct vector loads on CDNA3 | the shim; HipKittens untouched |
| 5. row stride | the strided gl above; the code-object launcher builds on CDNA | `cdna.cuh`, `HKToCpp.cpp` |
| 6. arch check | `runtime.arch.check_device` before any load: `ArchMismatchError` instead of a crash | `python/hk/runtime/arch.py` |
| 7. fragment in the type | `!hk.rt<..., frag 16x32>`; `hk-fragments` assigns it to operand types and refuses a type tied to two fragments (attention's accumulator->operand cast) | `HKTypes.td`, `Fragments.cpp` |
| 2. counted waits | per-target `ds_read` counts from the type, held to the ISA by a test; `lgkmcnt_max` 15 on gfx9; `wait=False` on CDNA; `lgkmcnt(0)` after include/cdna4's vectorised LDS store and DMA, which the pass assumes retire | `LdsPipeline.cpp`, `HKToCpp.cpp` |
| 3. bind hazard | measured: without the bind every gfx950 build hoists MFMAs above the wait and returns NaN; with it, correct | `tools/cdna-probes/bind_hazard.py` |
| code-object gate | `hk.runtime.cocheck`: resources and wait hazards from any code object (hk, HipKittens, Triton, FlyDSL) | `python/hk/runtime/cocheck.py` |
| CDNA tiles | 16x32 fragments and wider blocks for the elementwise and row kernels; the gate picks the width; four tiles a warp for wide folded norm rows | `hk.ops.elementwise`, `norm`, `fused`, `quant` |
| 4. HipKittens' CDNA4 kernels | `hk.ops.hipkittens`: the GEMM and GQA forward, verbatim, as `engine.Op`s, checked on the device first (ROCm 7.1 miscompiles the GEMM, 7.2.3 cannot compile the GQA); GQA 1.28-2.62x SDPA; the GEMM routed only where it beats hipBLASLt, by a measured gfx950 table ([details](cdna-hipkittens-ops.md)) | `python/hk/ops/hipkittens.py`, `hk/tuned/routes_vllm_linear_gfx950.json` |
| 8. occupancy timed | the gfx950 register model's steps measured on the part, at the model's boundaries; gfx950 `verified=True` | `tools/cdna-probes/occupancy_timed.py` |
| 9. hk's own GEMM | `g.load(wait=False)` (the DMA left in flight, `rows=` limit) and `g.wait_for` with derived vmcnts; `hk.arch.pingpong`; cross-warp copy hazards checked by barrier count, so named reads may cross a barrier. HipKittens' CDNA4 GEMM written with them (`hk.ops.gemm_cdna`), level with HipKittens; a four-warp kernel, two-launch split-K for both, and the decode GEMV on gfx950, routed by a measured table (108 of 122 Llama/Qwen linear points at 0.9x the best baseline or better): `hk.ops.gemm.matmul` has a gfx950 form ([chapter 33](../history/33-cdna-gemm.md)) | `group.py`, `arch.py`, `LdsPipeline.cpp`, `hk/cdna.cuh`, `hk.ops.gemm_cdna{,_dp}`, `hk/tuned/routes_gemm_gfx950.json` |
| 10. torch's stream | the CDNA pybind entry point takes the stream and `Kernel.__call__` passes torch's current one, so every hk op on gfx950 is ordered against torch's work and CUDA graphs capture it (0.2 µs a call); before, it launched on the null stream and a capture came out empty | `scaffold.takes_stream`, `pyutils/hk_bind.cuh`, `tests/hk/gpu/test_stream_cdna.py` |
| 11. hk's own attention | 32x32 accumulators, `Tile.as_operand` (the stride-4 operand fragment, `rt_16x32_4`), `Tile.subtile`, `Vec.any_above`, `hk.arch.pin`, LDS shapes per allocation. HipKittens' CDNA4 GQA forward written with them (`hk.ops.attn_cdna`), bf16, D 64/128, causal or not, any N: 0.98-1.08x the stronger of HipKittens and FlyDSL at the ten measured points, 1.30-3.13x SDPA; `hk.attention` and `hk.ops.sdpa` run it on gfx950 ([chapter 34](../history/34-cdna-attention.md)) | `values.py`, `arch.py`, `Fragments.cpp`, `HKToCpp.cpp`, `hk/cdna.cuh`, `hk.ops.attn_cdna`, `hk.ops.attn` |
| 12. strips on wave64 | `kittens::rs` ported to wave64 in the shim (a DPP `quad_perm`/`row_half_mirror`/`row_mirror` and `v_permlane16/32_swap` butterfly, LDS-pointer combines), `Target.strips`; on gfx950 the norms, softmax, quantize (any row count), the elementwise ops (bit for bit torch's), silu_mul on a gate/up output's halves in place and `hk.ops.rotary` (vLLM's NeoX RoPE, `Tile.swap_halves`) run on strips, planned at 32 elements a lane, capped at three workgroups a CU from the 256 MiB cache. Against the fastest of torch, Inductor, Triton and FlyDSL each tuned, at rows 1-8192 x hidden 4096-8192: 99 of 132 cases tie or win, the rest 0.90-0.98x ([chapter 37](../history/37-cdna-rows.md)) | `hk/cdna.cuh`, `norm.py`, `quant.py`, `elementwise.py`, `fused.py`, `values.py`, `HKOps.td` |
| 13. paged attention | `load_scalar(invariant=True)` (an s_load through the scalar cache), `hk.lds.alloc(shape=)` (an allocation laid out as a named st_shape its reads also take), `g.load(nontemporal=True)`, the 32x16 stride-4 col read counted, 16-column column reductions by permlane swaps. The split-KV decode (`hk.ops.paged_cdna`: a warp a split, Pᵀ read in place, pages by DMA in whole-line runs, a merge a strip a query row) and the paged prefill (`attn_kernel(paged=True)`); `hk.ops.paged` and `hk.ops.paged_prefill` route to them. Qwen3-8B against what vLLM runs: decode 0.90-1.05x aiter's gluon kernel and 1.04-1.27x ROCM_ATTN at 8-128 requests, 0.71-0.88x at one or two past 2048 tokens; prefill 1.10-1.24x aiter at batches. `load_w4` and fp16 MFMA into fp32 for the 4-bit GEMM; hk._mlir linked -Bsymbolic so it runs in a vLLM process ([chapter 39](../history/39-cdna-paged.md)) | `values.py`, `lds.py`, `group.py`, `HKOps.td`, `HKToCpp.cpp`, `LdsPipeline.cpp`, `hk/cdna.cuh`, `hk.ops.paged_cdna`, `hk.ops.attn_cdna`, `mlir/python/CMakeLists.txt` |
| 14. the leftovers | `hk.rotate_half {span}` (a DPP `row_ror:8` / `v_permlane16_swap` / `v_permlane32_swap` on a strip's segments) and `hk.load {period}` make a 16-byte RoPE (`rotary16`); per-lane norm targets by width and rows on gfx950; placement-proof elementwise plans; GEMM candidates streaming the weights (st_16x64, `sc0 nt`), blocks that fill the 256 CUs, the pingpong split at whole waves and its tail split along K, the split-K reduce on strips; HKToCpp refuses a subtile window narrower than its layout's subtile. GEMM at 0.9x or better at all 122 linear points, the rows 120 of 132 tie-or-win ([chapter 41](../history/41-cdna-leftovers.md)) | `values.py`, `HKOps.td`, `HKOps.cpp`, `HKToCpp.cpp`, `hk/cdna.cuh`, `hk.ops.fused`, `norm`, `elementwise`, `gemm_cdna{,_dp}`, `hk/tuned/routes_gemm_gfx950.json` |

Builds now (`tools/cdna-probes/coverage.py --ops`, device scaffold, spill gate): **496/506
on gfx950** (the attention example's two kernels since chapter 39) **and 494/506 on gfx942**,
from 196/410 and 195/410. What fails names its reason at trace time: the RDNA attention
kernels and paged attention (16-column operand tiles, an accumulator cast into an operand,
`lds_granules`), and the RDNA form of `hk.ops.gemm` (32x16 operand tiles, register
staging). gfx950 runs none of the RDNA forms: `hk.ops.gemm.matmul` routes to hk's gfx950
GEMMs, `hk.attention` to `hk.ops.attn_cdna`, `hk.ops.paged` to `hk.ops.paged_cdna` and
`hk.ops.paged_prefill` to `attn_cdna`'s paged form, and the decode GEMV builds there since
the shim gained fp32 register-LDS tile copies, the 4-bit GEMM since it gained `load_w4`.
On the MI355X (ROCm 7.2.3) the GPU tier is 784 passed, 66 skipped, 0 failed with them
(from 729/121 at `6a97779`, 658/127 at `35421a7`, 615/170 at `535be1c`, 350/190 at
`3b6c6ce` and 471/216 at `be52996`). The 66 are not missing paths: schedules a shape
does not fit, FlyDSL's RDNA3 GEMM, HipKittens' GQA under ROCm 7.2.3's hipcc, gfx1100's
route candidates, and vLLM not installed; the rule that reported a missing CDNA path as
a skip is gone ([testing](../testing.md#running-each-tier)). `examples/validate.py`
passes every case on gfx950, attention included.

## Summary of the assessment (at `fdb87f1`)

Sections 1-5 as written before the work above; the status table says what has changed.

* **hk's IR and Python frontend are not RDNA3-bound.** The RDNA3 parts are a short list:
  the target table, four `* 32`s, the C++ spelling of two tile types, about 25 library
  functions that only the fork's `include/rdna3` defines, and the semantics hk assumes for
  three of them (an LDS load that waits, a group load that waits, a 6-bit `lgkmcnt`).
* **On an MI355X, hk kernels compile for gfx950, run, and give correct results:** vector
  add (exact), RMSNorm, softmax, a register-only MFMA GEMM and a GEMM through LDS
  (`v_mfma_f32_16x16x32_bf16`, global→LDS DMA). Of the 410 instantiations hk ships
  (examples plus `hk.ops`), 196 build for gfx950 and 195 for gfx942 with no hand edits.
  Almost every failure is a missing library function (§1.4). The rest are loud on
  purpose: two `wait=False` refusals and two fragment-shape `static_assert`s in attention.
  gfx1100 and gfx1201 output is byte-identical to `fdb87f1`.
* **On CDNA4, hk's safety argument found a real silent bug.** HipKittens' CDNA4 group
  load is an asynchronous `buffer_load … lds`, and nothing orders it against the barrier
  that follows. The first CDNA4 LDS GEMM therefore gave a *different wrong answer on each
  run* (max error 13.8, 93.4 and 1.0 against a 1e-2 tolerance). Adding one `s_waitcnt
  vmcnt(0)` in the translation made it correct and deterministic. The other hazard hk
  encodes, an inline-asm `ds_read` that LLVM considers ready immediately, is present
  unchanged in `include/cdna4`.
* **Effort:** elementwise and fused ops work now. Norms, reductions and quantization are
  about 1–2 weeks (bounded loads are cheap on CDNA, via the buffer descriptor). A native
  GEMM, attention, MoE or decode path is weeks each. Wrapping HipKittens' own CDNA4
  GEMM/attention behind `hk.ops` and the engine layer is days.

---

## 1. What in hk-dsl is RDNA3-specific

### 1.1 Facts carried as data (already per target)

| Fact | gfx1100 (`python/hk/target/gfx1100.py`, `mlir/lib/Target/Target.cpp:24`) | consequence |
|---|---|---|
| wave | 32 (`gfx1100.py:15`) | thread counts, lane math in the library |
| VGPR file / granule | 1536 per lane, granule **24** (`:17-23`), measured | occupancy, headroom |
| LDS | 64 KB per workgroup, **128 KB WGP pool** shared by 4 SIMD32 (`:26-35`) | occupancy by LDS |
| matrix unit | WMMA 16x16x16; a bf16 operand is **mirrored** across wave halves, 8 VGPRs (`:37-46`) | an operand tile costs as much as an fp32 one |
| global→LDS DMA | **absent** (`:48-52`) | every staging path is a register buffer |
| fast LDS reads | `row` only (`:57-61`) | the `slow-smem-layout` warning |
| waits | `LDS_WAIT_NEEDS_FRAGMENT_BIND` (`:72-78`) | the bind after every wait |

These are already data, which is why gfx942/gfx950 could be added as two table entries
(§4.2). Two facts have no field yet: a register file shared with AGPRs, and which
HipKittens headers the target needs.

### 1.2 wave32 baked in outside the table

* `HKToCpp.cpp:194`: `threads() = warps * 32`. This gives `__launch_bounds__` and
  `block()`. On wave64 the kernel launches half its lanes and computes half its answer
  (it still compiles).
* `HKToCpp.cpp:956, 962, 1103`: `kittens::group<threads / 32>`.
* `python/hk/dsl/kernel.py:130` (`TracedKernel.threads`), `python/hk/dsl/group.py:32`
  (`Group.threads`, which feeds `GroupLoadOp`'s and the stage buffers' thread counts).

### 1.3 Tile types as C++ spells them

* `HKToCpp.cpp:53-55` writes `rt<T, R, C, layout>`. This matches `include/rdna3` and
  `include/cdna3` (`cdna3/types/register/rt.cuh:54`). `include/cdna4` adds a fifth
  parameter, the **MFMA fragment shape** (`cdna4/types/register/rt.cuh:55`, default
  `rt_16x16`; shapes in `rt_shape.cuh:30-37`). Its 16x16x32 MFMA takes `rt_16x32`
  row-layout or `rt_32x16` col-layout operands (`cdna4/.../tile/mma.cuh:161-244,
  323-401`), and copies require equal shapes (`conversions.cuh:140-162`).
* Shared tiles: hk writes `st<T, R, C>`. cdna3 has a defaulted layout
  (`cdna3/types/shared/st.cuh:51`). cdna4 *requires* an `st_shape`
  (`cdna4/types/shared/st.cuh:49`), which must match the register tile read out of it
  (`st_16x32` for `rt_16x32`, `cdna4/.../tile/assembly/shared_to_register.cuh:57`).
* `python/hk/types.py:111-116`: every extent is a multiple of 16. This still holds on
  CDNA; 32-multiple extents are needed only for gfx950 operands.

### 1.4 The library surface: what only `include/rdna3` has

The translation calls about 60 `kittens::` names. All of them exist in the fork's
`include/rdna3`; these do not exist in upstream's `include/cdna3`/`cdna4` (the fork's
CDNA trees are upstream `de0cddbd` plus five MXFP8/overflow files):

| Missing on CDNA | used by | on CDNA, needs |
|---|---|---|
| `warpid_uniform`, `lds_wait`, `lds_bind`/`lds_wait_for`, `vm_wait`, `rsqrt`, `silu`, `step` | every multi-warp kernel, norms | **done in a shim** (`include/hk/cdna.cuh`) |
| `load_bounded`, `store_bounded` (row-predicated) | 96 norm kernels, several examples (quant's first error is `int8`) | rewrite: cheap via the buffer descriptor (§3) |
| `load_scalar`, `store_scalar`, `wg_arrive`, `atomic_add`, `atomic_scalar` | control flow, cross-workgroup reductions | rewrite (scalar/atomic, layout-light) |
| `load_gather`, `store_scatter`, `load_w4`, `philox_uniform` | embedding, KV scatter, MoE, int4, dropout | rewrite against CDNA lane layouts |
| `lds_granules`, `lds_read_frag`, `lds_write_frag` | attention, decode GEMM | rewrite (wave64 address factoring) |
| `group::load_global_to_register_buffer` / `store_register_buffer_to_shared` | `hk.ops.gemm` staging | **replace** with the DMA on CDNA |
| `transpose_sep`, `make_causal_t` (cdna4) | attention | port |
| an `int8` type, `gl::row_stride` | quant; strided tensors and the code-object launcher | small type additions |

### 1.5 The hazard encodings, and whether they hold on CDNA

1. **A wait carries no register dependence.** `rdna3/.../memory/util/util.cuh:237-295`
   gives the bind after `s_waitcnt`, emitted only by `LdsWaitForOp`
   (`HKToCpp.cpp:910-917`). **This holds on CDNA4 unchanged:** cdna4's `load(rt, st)` is
   `asm volatile("ds_read_b128 %0, …" : "=v"(…))` with the waits commented out
   (`cdna4/.../tile/shared_to_register.cuh:78-100`, col layout `247-271`), and the
   shipped GEMM relies on a bare `asm volatile("s_waitcnt lgkmcnt(0)")` before
   `mma_ABt` (`kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp:146-148`).
   Whether the scheduler actually hoists an MFMA there on gfx950 has not been measured
   (§3, *validate first*). *Since measured (status, step 3): it does, and every LDS GEMM
   built without the bind returns NaN.*
2. **The LDS-load contract differs.** hk assumes `load(rt, st)` waits and `load<false>`
   does not (`rdna3/.../tile/shared_to_register.cuh:81, 141`; `HKToCpp.cpp:876`). cdna3
   and cdna4 `load(rt, st)` never wait and take no `<false>`. Unchanged, a `wait=True`
   hk load on CDNA would read registers before they land.
3. **The derived waits are RDNA3 arithmetic.** `lds_loads()` counts two `ds_read_b128`
   per 16x16 base tile (`python/hk/dsl/values.py:1131-1135`), which is rdna3's count.
   cdna4 issues one `ds_read_b128` per 16x32 fragment or a `ds_read_b64` per 16x16 one.
   gfx9's `lgkmcnt` is **4 bits** (0–15), against gfx11's 6, so a derived count above 15
   would wrap into an under-wait.
4. **The group load's contract differs.** hk's `group.load` is "issued and waited"
   (`dsl/group.py:44-48`), which is true on RDNA3 (a register round-trip). cdna3's waits
   (`cdna3/.../tile/global_to_shared.cuh:53-55`). **cdna4's does not**: it is
   `llvm_amdgcn_raw_buffer_load_lds` (`cdna4/.../tile/global_to_shared.cuh:61, 97`),
   counted on vmcnt and returned in flight. gfx90a+ has a back-off barrier, so LLVM
   inserts no wait before `s_barrier`. Measured consequence in §4.
5. **The spill gate reads VGPRs only** (`python/hk/runtime/resources.py:31-39`). On
   gfx90a+, AGPRs come out of the same 512-entry file, and the MFMA accumulators land
   there. The probe GEMM is 22 VGPRs + 16 AGPRs; read on VGPRs alone, that misstates
   occupancy and headroom.

### 1.6 The measured layer

`hk.integration` (engine.Op, RouteTable, register.op/patch) is arch-agnostic code over
**gfx1100 measurements**: the shipped route tables and `hk/tuned/*.json` are keyed on
gfx1100. On gfx950, `RouteTable` finds no table and routes nothing to hk
(`tests/hk/ir/test_vllm_gemm.py:31` asserts the gfx1100 one exists). That is the safe
default and the right one until there are MI355X measurements. *Since measured (status,
step 4): `routes_vllm_linear_gfx950.json` sends 37 of 206 points to HipKittens' CDNA4
GEMM, and none to `hk.ops.gemm`.*

---

## 2. What HipKittens offers on CDNA, and how hk ops map onto it

### 2.1 The CDNA trees

| | gfx942 (`include/cdna3`) | gfx950 (`include/cdna4`) | gfx1100 (`include/rdna3`) |
|---|---|---|---|
| wave | 64 (`common/util.cuh:54`) | 64 (`common/util.cuh:32`) | 32 (`common/util.cuh:80`) |
| LDS cap | 65536 (`util.cuh:127-129`) | 160000 (`util.cuh:110`); device reports 163840/block | 65536 |
| bf16 matrix op | MFMA 16x16x16 (`mfma_f32_16x16x16bf16_1k`) | MFMA 16x16x32, 32x32x16; fp8/f6/f4 `mfma_scale` 16x16x128, 32x32x64 (`ops/warp/register/tile/mma.cuh:13-124`) | WMMA 16x16x16, mirrored |
| rt type | `rt<T,R,C,layout>` | `rt<T,R,C,layout,shape>` | `rt<T,R,C,layout>` |
| st type | `st<T,R,C,layout=row>` | `st<T,R,C,shape>` (required) | `st<T,R,C>` |
| global→reg | buffer loads with an SRD | buffer loads with an SRD (`tile/global_to_register.cuh:39-41`) | fork's `load`/`load_bounded` |
| global→LDS | `buffer_load_dword … lds`, waited | `buffer_load_dwordx4 … lds`, **async** | none (register buffer) |
| LDS→reg | asm `ds_read_b64`, no wait | asm `ds_read_b128` / `ds_read_b64_tr_b16`, no wait | asm, `load<wait>` |
| shipped kernels | `kernels/cdna3/{gemm, layernorm}` | `kernels/cdna4/{gemm/{bf16fp32,fp8fp32,mxfp8}, attn/gqa{,_causal}{,_backwards}, layernorm, rotary, softmax}` | the fork's `kernels/rdna3` |

Register model, from `tools/cdna-probes/granule_sweep.py` (each kernel pins one `v`/`a`
register, and LLVM's Occupancy remark is read back; gfx942 and gfx950 are identical):
512 registers per lane shared by VGPRs and AGPRs, allocation granule **8**, AGPRs start
at `align4(VGPRs)`, at most **8 waves/SIMD**. 64 VGPRs gives 8 waves, 65 gives 7, and
64 + 1 AGPR also gives 7. This is the compiler's model and has not been timed on the part
the way gfx1100's 24 was. *Since timed on the MI355X (status, step 8).*

### 2.2 hk ops onto it (gfx950)

| hk ops | lowering on gfx950 | status here |
|---|---|---|
| `block_idx`, `warp_id`, `lane_id`, `coord`, `extent`, scalar arith, `for`/`while`/`if` | unchanged; `warpid_uniform` = readfirstlane(tid/64) | **works** |
| maps: `add sub mul div max min exp exp2 log log2 abs relu gelu dgelu copy cast` | unchanged (`cdna4/.../tile/maps.cuh`, `vec/maps.cuh`) | **works** |
| `rsqrt silu step` | shim | **works** |
| row/col `max min sum prod`, `add_row … div_col`, `broadcast_*`, rv types | unchanged (`tile/reductions.cuh`); `rt::col_vec` naming identical | **works** |
| tile `load`/`store` global (tile or element coords) | unchanged spelling → `buffer_load_dwordx4` | **works** |
| `load(bounded)`, `store(bounded)`, rows-by-index, scalar, atomics, Philox, w4 | new: narrow the SRD `num_records` for bounded rows; the rest are new code | missing |
| LDS vectors (`sv`) load/store, `cross_warp` | unchanged; plain C++, compiler-tracked | **works** |
| LDS tiles: `alloc`, `subtile`, `load_shared(wait=True)`, `store_shared` | `st_shape` derived from the consumer; load + `lds_wait_for<0>` | **works** (GEMM probe) |
| `load_shared(wait=False)`, `lds.wait_for(n)` | needs per-target ds_read counts and a 15 cap | refused at trace time |
| `granules`, `load_frag`/`store_frag` | new | missing |
| `mma_AB/ABt/AtB/AtBt` (bf16/fp16) | operand fragment `rt_16x32`/`rt_32x16` chosen by the translator; 16x16 fp32 col accumulator | **works** (ABt measured) |
| `group.load` | cdna4 DMA + `vm_wait<0>` | **works** (GEMM probe) |
| `group.stage`/`commit` (register staging) | replace by DMA + counted `vm_wait` | missing (by design) |
| `barrier`, `sched_barrier`, `setprio` | unchanged builtins | **works** |
| masks `left_fill … tril`, `make_causal` | present in cdna4; `make_causal_t` missing | partial |
| `transpose`, `transpose_sep` | cdna4 transposes swap shapes; `transpose_sep` missing | partial |

gfx942 is closer to hk than gfx950 is: its fragment is 16x16x16 and its tile spellings
are hk's, so only wave64, the shim and the wait semantics differ.

---

## 3. Path, estimates, risks, what to validate first

### 3.1 Per op family

Coverage today is `tools/cdna-probes/coverage.py --ops`, which builds each instantiation
through hk's own build and spill gate. gfx950: 196/410; first errors are `load_bounded`
104, `int8` 96, `load_scalar` 5, `lds_granules` 4, attention's 16x32-fragment
`static_assert` 2, `wait=False` refusals 2, `load_gather` 1. gfx942: 195/410, the same
list plus `gelu`, which `include/cdna3` lacks.

| family | gfx950 builds today | what it takes | estimate | risk |
|---|---|---|---|---|
| elementwise / fused (`hk.ops.elementwise`, `fused`) | 31/31, 18/18; GPU tier `test_elementwise` 44/44 pass | retune tiles/warps for wave64 and HBM3E (examples run at 0.5–0.6x torch, planned for RDNA3) | 2–3 d | low |
| norms / softmax (`hk.ops.norm`) | 144/240; the other 96 are bounded variants | `load_bounded`/`store_bounded` by narrowing the buffer descriptor (OOB rows read 0 and drop stores in hardware), re-plan TPW/warps for 64 lanes and 160 KB LDS, tune | 1–1.5 wk | low–med |
| quantization (`hk.ops.quant`) | 0/96 | the above, plus an `int8` type and `store_scalar` | +2–3 d | low |
| cross-workgroup reductions, control flow | examples fail on `load_scalar`/`wg_arrive`/atomics | scalar and atomic ops (layout-light) | 3–4 d | med (memory ordering) |
| GEMM (`hk.ops.gemm`) | 0/1 (register staging); the probe LDS GEMM is correct, 540 TFLOP/s vs hipBLASLt's 1530 at 4096³ | **(a)** wrap `kernels/cdna4/gemm/bf16fp32` as an `hk.ops` kernel behind `engine.Op`: 2–3 d. **(b)** native: per-target `lds_loads` and the 4-bit cap (2–3 d), DMA-pipelined staging with counted `vm_wait` (1 wk), 32x32x16 fragments, 256x256 tiles, XCD swizzle (1–2 wk) | (a) days, (b) 3–4 wk | med: perf parity is HipKittens-level work |
| attention / SDPA / paged prefill | 0/4 (first error: a 16-column operand tile, which the 16x32 fragment cannot hold) | fragment-aware tiling, transposes, granules, wave64 online softmax; or wrap `kernels/cdna4/attn/gqa*` first | wrap 3–5 d; native 3–4 wk | high (native) |
| MoE (`hk.ops.moe`, SGLang/vLLM MoE) | — | gather/scatter rows + grouped GEMM, after GEMM; watch the 4 GiB descriptor limit | 1–2 wk after GEMM | high |
| decode GEMV (`decode_gemm`), paged decode | 0/3, 0/2 | granules, cross-workgroup combine, wave64 split planning; DMA helps a bandwidth-bound kernel | 1–2 wk each | med |

### 3.2 Risks

1. **Silent races from asm-based library ops.** Measured once already (§4.3). Every
   CDNA4 helper that returns with LDS or vmem traffic in flight needs its hk contract
   written down and enforced in the translation, as for the group load.
2. **4-bit `lgkmcnt`.** A derived wait above 15 wraps. The shim `static_assert`s it, and
   the pass has to know.
3. **32-bit buffer sizes.** cdna4's register loads size the descriptor as
   `uint32_t buffer_size = batch*depth*rows*cols*sizeof(U)` from the tile's origin
   (`cdna4/.../tile/global_to_register.cuh:39, 199`). A tensor over 4 GiB, such as stacked
   MoE expert weights, wraps the count, and the hardware returns zeros past it. This is
   silent, and upstream's. *Since step 1, hk's register-tile loads and stores no longer use
   it: their descriptor is sized by the tile's own rows. The group (DMA) load sizes by
   `rows * row_stride` of the tile and is safe too; rows-by-index (`load_gather`) spans the
   whole tensor and caps at 3.75 GiB, as on RDNA3.*
4. **The register model is compiler-modelled, not timed**, so the new targets are
   `verified=False`. Also, LLVM may park VGPRs in AGPRs (`v_accvgpr_write`) instead of
   scratch. That is not a correctness issue, but the gate does not see it. *Since step
   8, gfx950's model is timed on the MI355X and the target is `verified=True`; gfx942's
   is still modelled.*
5. **The fragment shape is a translator heuristic.** A tile type used both as an
   operand and as a 16x16 tile becomes a library `static_assert` (loud, not silent). The
   proper fix is shape as a field of `!hk.rt`. *Fixed (step 7): the fragment is in the
   type, and `hk-fragments` refuses a type tied to two fragments.*
6. **An arch mismatch at launch segfaults** (`tests/hk/gpu/test_bounded.py` pins
   gfx1100 and crashed the process on gfx950) instead of raising. *Fixed (step 6).*
7. **Strided tensors.** The CDNA `gl` has no row stride. The shim accepts contiguous
   rows and refuses the rest. The code-object launcher (`scaffold.device`,
   `scaffold.py:79-80`) names `row_stride_internal` and does not build on CDNA, which
   blocks `test_torch_op` (0/8 here). *Fixed (step 5): `kittens::strided_gl`;
   `test_torch_op` passes on the MI355X.*
8. **Truncating bf16 conversions in the CDNA trees.** Found in step 1: both trees'
   scalar `convertor<bf16, float>` truncates, and include/cdna3's packed one too. hk's
   stores and CDNA3 casts round to nearest even. One library path hk does not wrap still
   truncates: include/cdna4's ortho and naive register-vector stores to a bf16 global (its
   align store uses the packed, rounding conversion).
9. **Upstream churn.** cdna4's fast GEMM path uses `G::load(…, swizzled_offsets, srd,
   base, lds_base)`, an API still moving upstream.

### 3.3 What to validate first

1. **The bind hazard on gfx950.** Build hk's CDNA LDS kernels with the bind removed,
   sweep register pressure, and diff the ISA and results, as was done for gfx1100. If
   MFMAs do hoist, HipKittens' own CDNA4 kernels are exposed too, and hk's IR is the
   systematic fix.
2. **Bounded load/store via the descriptor**, then the GPU tier for `norm` and `quant` on
   the MI355X: the biggest coverage step for the least code.
3. **Per-target `lds_loads` and the 15 cap** in `hk-lds-pipeline`, then re-enable
   `wait=False` on CDNA.
4. **Time the occupancy model** (granule 8, unified file) on the part and flip
   `verified`.
5. **HipKittens CDNA4 GEMM/attention behind `engine.Op`**, with a RouteTable measured
   in vLLM on MI355X.

---

## 4. Prototype

### 4.1 Results on one MI355X

One GPU of an eight-GPU node, passed alone to the container (`/dev/kfd` and its render node). Each check is
`max|got - want| / max|want|` against an fp32 torch reference. Times are best-of-5, in
one process, with contenders interleaved.

| kernel | check | resources (gate) | time |
|---|---|---|---|
| `examples/01_vector_add.py` 8192² bf16 | exact | 25 VGPR → 32, 8 waves | 0.114 ms, 0.61x `torch.add` |
| `examples/02_rmsnorm.py` 8192x4096 | 2.9e-3 | 28 VGPR, 8 waves | 0.044 ms, 0.58x `F.rms_norm`; `hk.ops.norm.rmsnorm` ran at 0.72x (timed only) |
| `examples/03_softmax.py` 8192x4096 | 1.95e-3 | 37 VGPR, 8 waves | 0.51x `torch.softmax`; `hk.ops.norm.softmax` 0.76x (timed only) |
| `tools/cdna-probes/mma_reg.py` (registers only), 3 shapes up to 2048x2048x1024 | ≤ 6.3e-3 | 22 VGPR + 16 AGPR → 40, 8 waves | — |
| `tools/cdna-probes/gemm_lds.py` (LDS + DMA), 3 shapes up to 4096³ | ≤ 5.7e-3, identical in 3 runs | 20 VGPR + 16 AGPR → 40, 8 waves | 0.254 ms = 540 TFLOP/s; hipBLASLt 1530 (0.35x) |

ISA of `mma_reg` per K step: 4 `v_mfma_f32_16x16x32_bf16` and 4 `buffer_load_dwordx4`.
`gemm_lds`: `buffer_load_dwordx4 … lds`, `s_waitcnt vmcnt(0)`, `s_barrier`,
`ds_read_b128`, `s_waitcnt lgkmcnt(0)`, then the MFMAs, which read operands from VGPRs
and accumulate in `a[0:15]`.

GPU tier (`tests/hk/gpu`, `HK_ARCH=gfx950`, a subset):

| file | result | what failed |
|---|---|---|
| `test_elementwise.py` | 44/44 pass | — |
| `test_phase2.py` | 79/113 pass | `int8` type, `store_scalar` |
| `test_rms_norm_weighted.py` | 0/14 | `load_bounded`/`store_bounded` |
| `test_torch_op.py` | 0/8 | code-object layout table (`row_stride_internal`) |
| `test_bounded.py` | segfault | the test pins `arch="gfx1100"` and launches it on gfx950 (risk 6) |
| `test_target_rmsnorm.py` | timed out at 400 s | not investigated |

Nothing returned a wrong answer except the GEMM in §4.3, before its fix.

### 4.2 Exactly what changed (branch `cdna`; HipKittens untouched)

* **Targets as data.** `python/hk/target/gfx950.py` and `gfx942.py` were added: wave 64,
  8 waves/SIMD, 512 registers, granule 8, the LDS figures, MFMA shapes, DMA present, fast
  layouts (`row`,`col` on gfx950 via `ds_read_b64_tr_b16`; `row` on gfx942),
  `verified=False`. Two new fields in `base.py`: `unified_agprs` (with `charged_vgprs`)
  and `extra_includes`. `tile_vgprs` is scaled by fragment size (identical for the
  16x16x16 targets). Aliases `CDNA3`/`CDNA4`. The C++ twin is in
  `mlir/lib/Target/Target.cpp`, and `test_mlir_dialect` holds the two equal for all four
  targets.
* **wave64.** `TracedKernel.threads` and `Group.threads` use the target's wave.
  `HKToCpp.cpp`: `threads()` and the three `group<N>` spellings divide by `tgt->wave`.
  `hk-verify` refuses more than 1024 threads.
* **Tile spelling.** `HKToCpp.cpp` `findMmaOperands`/`tileCpp`: where the target's
  fragment is not 16x16x16, mma operand tiles get `rt_16x32` (row) or `rt_32x16` (col).
  The shared allocation they are read from gets the matching `st_` shape, and every other
  shared tile gets `st_16x16`.
* **The two waits CDNA's library does not do.** On wave64 targets, `load_shared` emits
  `kittens::load(d, s)` and, for `wait=True`, `kittens::lds_wait_for<0>(d)`.
  `group.load` is followed by `kittens::vm_wait<0>()`. `load_shared(wait=False)` is
  refused at trace time with the reason.
* **The shim** `include/hk/cdna.cuh`, included after `kittens.cuh` on CDNA only, defines
  `warpid_uniform`, `lds_wait` (`static_assert` N ≤ 15), `vm_wait`, `lds_bind` (also for
  sub-128-bit tiles), `lds_wait_for`, `rsqrt`/`silu`/`step`, and a 6-argument `make_gl`
  that accepts contiguous rows and throws on a real stride.
* **The gate.** `resources.py` parses `AGPRs` and charges `align4(VGPRs) + AGPRs` on
  unified targets.
* **Tests.** `tests/hk/conftest.py` pins `HK_ARCH=gfx1100` unless one is set, because
  an MI355X box is now recognised rather than falling back to gfx1100. The GPU tier skips
  itself when the GPU is not the pinned arch. New: `tests/hk/ir/test_cdna.py` (13) and
  `tests/hk/codegen/test_cdna_builds.py` (10: 5 kernels × gfx950/gfx942).
* **Probes.** `tools/cdna-probes/`: `granule_sweep.py`, `coverage.py`, `mma_reg.py`,
  `gemm_lds.py`.

Regressions: for every example and `hk.ops` instantiation (413), every scaffold
(pybind, bare, device, torch) and the IR dump are **byte-identical** to `fdb87f1` for
gfx1100 and gfx1201. `tests/hk/ir`: 513 passed (500 + 13), 2 skipped. `tests/hk/codegen`:
808 passed (798 + 10).

### 4.3 The bug the GEMM probe found

`gemm_lds` is `examples/04_gemm.py` with every LDS read waited. Its first gfx950 build
compiled cleanly with no scratch, and gave a different wrong answer on each of three runs
(max error 13.8, 93.4 and 1.0, tolerance 1e-2). The ISA had `buffer_load_dwordx4 … lds`
and then `s_barrier` with no `s_waitcnt vmcnt` between them:

* the DMA counts on vmcnt and cdna4's load returns with it in flight;
* gfx950 has a back-off barrier, so LLVM adds no wait before `s_barrier`;
* the `ds_read`s after the barrier are inline asm, which `SIInsertWaitcnts` cannot see.

One `vm_wait<0>()` emitted after the group load made it correct and deterministic. This
is the RDNA3 argument in a new place: the library's contract differs from what the IR
op promises, nothing fails at compile time, and the IR is the one place where the
contract can be kept for every kernel.

### 4.4 Reproduce

```bash
# host, cp310 hk._mlir (build dir of your own, not the shared one)
HK_LLVM_ROOT=/dev/shm/hk-llvm HK_MLIR_BUILD=/dev/shm/$USER-hk-build-cp310 \
  PYTHON=python3 scripts/build_hk_mlir.sh
HK_ARCH=gfx950 python3 examples/02_rmsnorm.py --build        # compile + gate, no GPU
HK_ARCH=gfx950 python3 tools/cdna-probes/coverage.py --ops   # 494/506 now; 196/410 at the assessment
python3 tools/cdna-probes/granule_sweep.py gfx950
# a cp312 hk._mlir for rocm/pytorch: same cmake against the cp310 MLIR install (its C++
# is interpreter-agnostic; the core Python extension builds from the installed sources),
# installed to a separate prefix and bind-mounted over python/hk/_mlir in the container.
HK_ARCH=gfx950 python3 tools/cdna-probes/gemm_lds.py          # in the container, one GPU
```

---

## 5. Where hk-dsl complements FlyDSL on CDNA

What FlyDSL is (local checkout `88f1969`): an MLIR-native stack whose `fly` dialect makes
layouts first-class (`!fly.layout`, `!fly.coord_tensor`; composition, product, divide,
partition) and lowers directly to ROCDL. It ships a deep CDNA4 catalog
(`kernels/attention/flash_attn_gfx950.py`, `kernels/gemm/gemm_a16w16_gfx950.py`, fp8,
fp4 and MX GEMMs, MoE, MLA and paged decode). It is **not CDNA-only any more**: it has
RDNA3/RDNA4 MMA atoms (`lib/Dialect/FlyROCDL/GFX11/MmaAtom.cpp`) and RDNA3 GEMMs
(`kernels/gemm/rdna3_f16_gemm.py`). A split of "FlyDSL for CDNA, hk for RDNA" is
therefore neither accurate nor the useful framing.

Where FlyDSL is stronger, plainly: arbitrary layouts and swizzles (hk has fixed tile
types), a measured CDNA4 kernel catalog, no 5–40 s hipcc build per kernel, and
fp8/fp4/MX as first-class types.

Where hk adds something on CDNA:

1. **A lowering target that is HipKittens.** hk emits readable C++ against a library
   with hand-tuned MI355X/MI300X GEMM and attention kernels (§2.1), so those kernels can
   become `hk.ops` entries without being re-derived. FlyDSL's path does not go through
   that library.
2. **Correctness gates as a product.** These are the spill gate (FlyDSL's Python compile
   path has no scratch/spill check: nothing in `python/flydsl` matches it), the
   AGPR-aware occupancy charge, trace-time refusals with reasons, and contracts the IR
   keeps over an asm-based library (§4.3). FlyDSL's loads are compiler-visible LLVM ops,
   so hazard (1) of §1.5 matters less to its generated code. It does expose raw
   `rocdl.s_waitcnt` (`python/flydsl/expr/rocdl/universal.py:40`), which carries the
   same "no dependence" semantics wherever it is paired with asm. The gate could become
   a tool-agnostic check over any code object's `.amdhsa` notes (VGPR, AGPR, scratch)
   and certify FlyDSL, Triton and HipKittens artifacts alike.
3. **The engine layer.** `engine.Op`, `RouteTable` and `register.op`/`patch`, plus the
   torch-op scaffold (`TORCH_LIBRARY_FRAGMENT`, a Meta impl, out-parameter ops that CUDA
   graphs can capture), are measured inside vLLM and SGLang. FlyDSL's checkout has
   per-kernel interfaces (e.g. `kernels/attention/flash_attn_interface.py`) but no
   routing-table layer. An `Op` is a callable, so the same table can route a shape to a
   FlyDSL kernel, a HipKittens kernel or an hk kernel by measurement.

---

## 6. Recommended next steps, in order

1. ~~`load_bounded`/`store_bounded` via the buffer descriptor, `int8`, `store_scalar`; then
   the norm and quant GPU tier on the MI355X.~~ Done (see Status).
2. ~~`hk-lds-pipeline`: per-target ds_read counts and the 4-bit cap; re-enable `wait=False`
   on CDNA.~~ Done.
3. ~~The bind-hazard probe on gfx950 (§3.3.1).~~ Done: the bind is load-bearing.
4. ~~HipKittens' CDNA4 GEMM and GQA attention as `hk.ops` behind `engine.Op`, with a
   RouteTable measured in vLLM on MI355X against hipBLASLt/aiter.~~ Done, measured against
   hipBLASLt, rocBLAS, SDPA and FlyDSL (aiter not installable here); the table is measured
   against `F.linear` (no vLLM in the container), and vLLM end to end on MI355X is next.
5. ~~A row stride for the CDNA `gl` (on the fork's branch, not upstream), which also
   unblocks the code-object launcher.~~ Done in hk's shim instead (`strided_gl`), with
   HipKittens untouched.
6. ~~An arch check at first launch (raise, don't segfault).~~ Done.
7. ~~MFMA fragment shape as a field of `!hk.rt` rather than a translator heuristic;~~
   done. ~~32x32x16 fragments and a CDNA path for attention.~~ Done: 32x32 accumulators,
   the stride-4 operand read off one, and `hk.ops.attn_cdna`
   ([chapter 34](../history/34-cdna-attention.md)). ~~Paged prefill and split-KV decode on
   gfx950.~~ Done ([chapter 39](../history/39-cdna-paged.md)). Next: fp8 `mfma_scale`
   fragments.
8. ~~Time the unified-file occupancy on the part and set `verified=True`.~~ Done for gfx950.
