# HipKittens' CDNA4 GEMM and GQA attention as hk ops, measured on an MI355X

> Written on branch `cdna-hkops` (from `cdna` at `3e2f70a`); merged with chapter 29, part 5. Hardware: one Instinct MI355X (gfx950,
> idle at the start of every session). Everything that launches
> ran in `rocm/pytorch:rocm7.2.3 … py3.12 torch 2.10`; ROCm 7.1's hipcc was mounted
> read-only from the host where a build needed it. HipKittens is the pinned submodule
> (`49fa4a38`), unmodified. Raw numbers: `tools/hk-bench/hipkittens_cdna.py` (its JSON
> output) and the `points` of `python/hk/tuned/routes_vllm_linear_gfx950.json`.

## Summary

* **`hk.ops.hipkittens`** builds HipKittens' own CDNA4 kernels through hk's compile cache
  and spill gate. The kernels are `kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp`
  and the four `attn/gqa{,_causal}/kernel{,_d64}.cpp`. hk appends its own launch to the
  verbatim source, on torch's stream. The ops are `gemm_bf16` and `gqa_fwd`, plus the
  `engine.Op`s `hk_cdna::gemm_bf16` and `hk_cdna::gqa_fwd` (fake impls, `graph="safe"`,
  torch fallback). Shapes a kernel cannot take are refused with the reason. Any arch but
  gfx950 is refused.
* **Whether the kernels are right depends on the hipcc, and the spill gate cannot see it.**
  * The GEMM is correct built by ROCm 7.2.3. Built by ROCm 7.1 it builds clean, passes the
    gate, and is wrong (rel err 0.1–0.86): rows 128–191 of every 256-row tile, at every
    shape. Upstream's own `dispatch_micro`, built unchanged, does the same.
  * The attention kernels do not compile with ROCm 7.2.3's clang 22. They compile and are
    correct with 7.1.

  So every build runs once on the device against fp32 torch before first use. A build
  that fails that check raises `Miscompiled`, and the engine ops take torch instead.
* **GEMM: hipBLASLt is faster almost everywhere.** HipKittens is 0.91–0.96x at M ≥ 4096
  and 0.2–0.9x below M = 2048. It wins by 4–27% only at particular Ms where hipBLASLt's
  heuristic picks a slow kernel, for example o_proj at M = 2048 (886 vs 708 TFLOP/s).
* **GQA forward: 1.28–2.62x torch SDPA** (aotriton flash, which SDPA picks for every
  config here). That is 1.66–1.84x non-causal and 2.17–2.62x causal at d=128. FlyDSL's
  gfx950 flash attention is level with it: HipKittens is 0.92–1.00x FlyDSL.
* **The gfx950 route table** (`routes_vllm_linear_gfx950.json`, written by
  `route_tune.py --paths hipkittens`) sends 37 of 206 measured (N, K, M) points to
  HipKittens, all at M ≥ 1792, by 3–27%. A second run agreed on 271 of the 272
  lookups over the 34-M grid.

## 1. What was built

| file | what |
|---|---|
| `python/hk/ops/hipkittens.py` | sources, builds, launch, shape gates, on-device check, `engine.Op`s |
| `python/hk/integration/vllm_gemm.py` | a fourth path, `"hipkittens"`, gated on what the GEMM takes |
| `tools/hk-bench/route_tune.py` | `--paths hipkittens`, `--rotate-mb`, several `--model`s |
| `tools/hk-bench/hipkittens_cdna.py` | the measurements below |
| `python/hk/tuned/routes_vllm_linear_gfx950.json` | the table, written by route_tune |
| `tests/hk/{ir,codegen,gpu}/test_hipkittens*.py` | the three tiers |

**Source and launch.** The generated source has four parts:
1. `#define tk_kernel hk_hipkittens_upstream_module`, so upstream's own
   `PYBIND11_MODULE(tk_kernel, …)` lands under a name nothing imports;
2. the `ATTN_*` defines (attention only);
3. the upstream file, verbatim;
4. hk's entry point, `PYBIND11_MODULE(HK_MODULE_NAME, …)`.

The entry point takes raw pointers and `torch._C._cuda_getCurrentRawStream`. Upstream's
launch uses the null stream, which a CUDA graph cannot capture. hk's sets the 160 KB
dynamic-LDS attribute once per device, and checks `hipGetLastError`. The tensor checks
(dtype, contiguity, device, 16-byte alignment) run in Python. The glue lives in the
Python file, not in `include/`, so the cache key covers it through the source text, and
adding it did not change the include-tree fingerprint every other kernel is keyed on.
Flags are `common.mk`'s plus, for attention, its Makefile's `-DHIP_ENABLE_WARP_SYNC_BUILTINS
-ffast-math`. Each build takes about 5 s.

**Constraints**, read off the kernels (`gemm_unsupported` / `gqa_unsupported` give the
reason):

| kernel | takes | because |
|---|---|---|
| GEMM, C = A·Bᵀ, A (M,K), B (N,K) | M, N multiples of 256 | grid = (M/256)(N/256); the C store (col-layout accumulator) is a plain pointer write, so a partial tile writes past C |
| | K multiple of 128 | main loop takes two 64-wide K steps a trip, epilogue the last two |
| | M·K, N·K < 2³⁰ | `make_srsrc(base, M * row_stride_bytes)` in `int` |
| GQA fwd, q (B,N,H,D), k/v (B,N,H_KV,D) | D ∈ {64, 128} | the four shipped kernels |
| | N multiple of 256, N_q = N_kv | 8 waves × 32 query rows a workgroup, unpredicated; no KV cache |
| | H multiple of H_KV | head h reads KV head h / (H/H_KV) |
| | B·N·H·D < 2³¹ | q/o descriptors are a uint32 byte count from an `int` product |
| | scale 1/√D, causal top-left | compiled in (`TEMPERATURE_SCALE`) |

Nothing is padded. A ragged M cannot be had for free here: the A loads are bounded by the
buffer descriptor, but the C store is not.

**One GQA build per (H, H_KV, N, D, causal).** `ATTN_B` is only upstream `grid()`'s z.
The launch computes the grid from the batch it is handed, so every batch size shares one
build. Loaded modules stay for the process. The cache makes the build a load in later
processes.

**Resources** (hk's gate, gfx950 charge align4(VGPR)+AGPR): GEMM 210 VGPRs, 0 AGPRs,
2 waves/SIMD, under both toolchains. Attention d128 224–238 VGPRs, 2 waves; d64 157–161,
3 waves. No scratch anywhere.

**Host cost a call** (2000 calls of a 256×256×128 GEMM, which leaves the host the
bottleneck):

| call | µs |
|---|---|
| `gemm_bf16(…, out=)` | 6.4 |
| allocating `out` | 7.6 |
| through `torch.ops.hk_cdna.gemm_bf16` | 12.0 |
| `F.linear` | 12.1 |

## 2. The toolchain matrix

| | ROCm 7.1 hipcc (clang 20) | ROCm 7.2.3 hipcc (clang 22) |
|---|---|---|
| GEMM | builds, gate passes, **wrong**: rel err 0.1–0.86, rows 128–191 of every 256-row tile, non-deterministic at 256×256×128 | **correct**, ≤ 6.6e-3, bit-identical on repeat |
| GQA (all 4 files) | **correct**, ≤ 5.4e-3 | **does not compile**: `call to 'zero' is ambiguous` (also `ones`, `copy`, `mul`, `sub`) |

What was established:

* **The GEMM error is upstream's under clang 20, not the wrapper's.** Upstream's file was
  built unchanged with each hipcc and run through its own `dispatch_micro`. It shows the
  same rows wrong with 7.1 and none with 7.2.3.
* **The bad rows are the `C_accum[1][*]` of the four `warp_row == 0` waves.** Those are the
  waves that run one barrier phase ahead: `if (warp_row == 1) s_barrier` at entry,
  `if (warp_row == 0)` at exit.
* **The two builds issue the same memory traffic in the same order.** The device ISA has
  the same skeleton of DMA, `s_waitcnt vmcnt/lgkmcnt`, `s_barrier`, `ds_read` and MFMA
  (256 MFMAs, 96 `ds_read`s, identical order). No MFMA reads a register an outstanding
  `ds_read` may still be writing; that was checked by tracking `lgkmcnt` in both. So
  this is not the bind hazard of cdna.md §1.5. The differences are all scalar and 64-bit
  address arithmetic (`v_lshl_add_u64` 70 vs 36, `s_mul_hi_i32` only in 7.1).
* **The root cause was not isolated further.**
* **The attention failure is clang 22's.** include/cdna4's `ducks::rt::all`,
  `ducks::art::all` and `ducks::rv::all` are textually identical concepts in three
  namespaces, and clang 22 finds an `rv` satisfying all three. Upstream's file fails the
  same way with or without `-ffast-math`, at -O0, and under `-std=c++2b`.

What hk does about it:
* The first launch of each build on each device runs it on seeded inputs against fp32
  torch. The GEMM check is 512³ (the loop, the epilogue, both wave rows, four tiles). The
  attention check is the compiled shape at B=1, compared on the first, a middle and the
  last 32-row query block of every head. The tolerance is 2e-2; the correct builds
  measure 5e-3 and the miscompiled GEMM 0.48.
* A failure raises `Miscompiled`, naming the hipcc. A build that does not compile raises
  `Unavailable`. Both are remembered, so a failing hipcc runs once. The engine ops and
  vLLM's linear take torch instead, with one warning per reason.
* The check synchronizes, so a first launch under CUDA-graph capture raises
  `GraphCaptureError` ("run this shape once before capturing it") instead of freezing
  anything.

Consequences:
* `HK_HIPCC` chooses the toolchain. The measurements below ran the GEMM under 7.2.3 and
  attention under 7.1, in separate processes.
* The CI codegen tier runs ROCm 7.1, so there the GEMM builds and passes the gate, and
  would be wrong on a GPU. That is the case the on-device check exists for.

## 3. Correctness

Each check is max|got − want| / max|want| against fp32 torch on the GPU.

* GEMM (7.2.3): 256×256×128, 512×1024×256, 4096³, 1024×6144×4096, 2048×4096×12288,
  2048×4096×14336, 512×14336×4096, 8192³, 256×28672×4096, 768×4096×8320: 5.2e-3 to
  6.6e-3, bit-identical on repeat. Over the 69 benchmark shapes the worst is 6.4e-3;
  hipBLASLt's is 3.4e-3. A guard region after C is untouched.
* GQA (7.1), o against fp32 (full reference) and the LSE in absolute terms:
  B,N,H,H_KV,D = (2,256,8,2,128), (2,1024,64,8,128), (1,2048,32,8,128), (2,1024,64,8,64),
  (1,4096,32,8,128), (3,512,16,16,128), causal and not. o: 2.6e-3 to 4.9e-3. LSE:
  1.5e-3 to 5.4e-3. Deterministic. Over the 12 benchmark configs (sampled rows) the worst
  is 5.4e-3, the same as FlyDSL's; SDPA's is 3.3e-3.

## 4. GEMM vs hipBLASLt

Method (`hipkittens_cdna.py gemm --flydsl --graphs`):
* Every arm is checked against fp32 first, then interleaved in one process: 7 rounds,
  alternate rounds reversed.
* A round is one pass over enough weight copies to stream 768 MB (3× the 256 MB Infinity
  Cache).
* Best round, TFLOP/s = 2MNK/t. Eager, and CUDA-graph replay. The two agree within 2%
  except where the host bounds a small call (hipBLASLt at 1024³: 169 eager, 270 graph).

The arms:
* **hipBLASLt** is `F.linear` (torch's default, `_BlasBackend.Cublaslt`).
* **rocBLAS** is `F.linear` under `preferred_blas_library("cublas")`.
* **matmul NN** is `torch.matmul(A, B)` with B stored (K, N).
* **FlyDSL** is `gemm_a16w16`, layout "nt", from the PyPI 0.4.0 wheel plus the checkout's
  `kernels/` (88f1969).

**Squares** (TFLOP/s, eager / graph):

| M=N=K | HipKittens | hipBLASLt | rocBLAS | matmul NN | FlyDSL | HK / hipBLASLt |
|---|---|---|---|---|---|---|
| 1024 | 90 / 91 | 169 / 270 | 157 / 267 | 170 / 200 | 38 / 94 | 0.53 |
| 2048 | 420 / 420 | 667 / 689 | 655 / 677 | 653 / 673 | 302 / 422 | 0.63 |
| 3072 | 887 / 887 | 1171 / 1189 | 1167 / 1194 | 911 / 920 | 951 / 970 | 0.76 |
| 4096 | 1432 / 1428 | 1484 / 1489 | 1502 / 1496 | 1372 / 1367 | 1447 / 1479 | 0.96 |
| 6144 | 1273 / 1273 | 1406 / 1409 | 1396 / 1388 | 1359 / 1344 | 1289 / 1305 | 0.91 |
| 8192 | 1539 / 1539 | 1612 / 1617 | 1614 / 1618 | 1470 / 1476 | 1574 / 1594 | 0.95 |

**LLM shapes**, the (N, K) of Llama-3-8B and Qwen3-8B, merged and unmerged. Each cell is
hipBLASLt time / HipKittens time, eager (> 1: HipKittens faster). A dash marks an M the
kernel does not take.

| N × K | M=64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| 4096 × 4096 | — | — | 0.27 | 0.37 | 0.64 | **1.25** | 0.96 | 0.95 |
| 6144 × 4096 | — | — | 0.36 | 0.51 | 0.74 | 0.92 | 0.93 | 0.94 |
| 12288 × 4096 | — | — | 0.50 | 0.72 | 0.91 | 0.93 | 0.96 | 0.95 |
| 14336 × 4096 | — | — | 0.67 | 0.93 | 1.02 | 1.02 | 0.99 | 0.94 |
| 24576 × 4096 | — | — | 0.78 | 0.86 | 0.89 | 0.95 | 0.93 | 0.91 |
| 28672 × 4096 | — | — | 0.75 | 0.94 | 0.99 | 1.00 | 0.93 | 0.93 |
| 4096 × 12288 | — | — | 0.18 | 0.35 | 0.45 | 0.78 | 0.96 | 0.96 |
| 4096 × 14336 | — | — | 0.20 | 0.34 | 0.43 | 0.77 | 0.95 | 0.95 |

TFLOP/s at the same points (HipKittens / hipBLASLt / FlyDSL, eager):

| N × K | M=256 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 4096 × 4096 | 111 / 411 / 110 | 461 / 722 / 450 | 886 / 708 / 880 | 1432 / 1484 / 1447 | 1461 / 1542 / 1492 |
| 6144 × 4096 | 166 / 460 / 163 | 670 / 906 / 660 | 1183 / 1290 / 1208 | 1245 / 1343 / 1269 | 1464 / 1552 / 1511 |
| 14336 × 4096 | 375 / 559 / 352 | 1310 / 1281 / 1267 | 1332 / 1307 / 1341 | 1348 / 1367 / 1397 | 1467 / 1553 / 1508 |
| 28672 × 4096 | 691 / 919 / 607 | 1286 / 1296 / 1250 | 1309 / 1310 / 1285 | 1440 / 1554 / 1465 | 1452 / 1557 / 1470 |
| 4096 × 14336 | 108 / 552 / 110 | 455 / 1047 / 455 | 903 / 1174 / 906 | 1556 / 1633 / 1560 | 1543 / 1632 / 1546 |

Reading:
* **Small M is a tile-count problem.** A 256×256 output tile per workgroup is 16
  workgroups for 256 CUs at M=256, N=4096. FlyDSL's default 256×256 tile tracks
  HipKittens point for point at small M for the same reason.
* **At large M, hipBLASLt is 4–9% ahead.**
* **The wins are hipBLASLt heuristic teeth, not a regime.** At M = 2048, N = K = 4096,
  hipBLASLt runs 708 TFLOP/s, rocBLAS 965 and HipKittens 886.
* **matmul NN** (B stored K×N) is 5–10% slower than `F.linear`'s NT at large M on
  hipBLASLt.

## 5. GQA forward vs torch SDPA

Method (`hipkittens_cdna.py gqa --flydsl`, attention built by ROCm 7.1):
* Every arm is checked first: three 32-row query blocks of the first and last batch,
  every head, against fp32.
* 7 interleaved rounds of 2–20 calls; best round. FLOPs 4·B·N²·H·D, halved for causal.

The arms:
* **SDPA** is `F.scaled_dot_product_attention(enable_gqa=True)` on contiguous (B,H,N,D).
  `torch._fused_sdp_choice` picks `FLASH_ATTENTION` (aotriton) for all 12 configs, and
  forcing flash measures the same.
* **SDPA's memory-efficient backend** has no GQA kernel ("No available kernel"; it wants
  equal head counts). The math backend was not timed: it materializes the N² scores.
* **SDPA on BNHD views** is SDPA on `transpose(1, 2)` views of hk's layout, with no copy.
  It is within 2% of contiguous.
* **FlyDSL** is `flydsl_flash_attn_func` (BSHD).

| B, H, H_KV, N, D | causal | HipKittens | SDPA (flash) | SDPA on BNHD views | FlyDSL | HK / SDPA | HK / FlyDSL |
|---|---|---|---|---|---|---|---|
| 16, 64, 8, 1024, 128 | no | 966 | 570 | 566 | 1010 | 1.69 | 0.96 |
| 16, 64, 8, 1024, 128 | yes | 663 | 253 | 253 | 702 | 2.62 | 0.94 |
| 16, 64, 8, 2048, 128 | no | 1157 | 630 | 628 | 1175 | 1.84 | 0.98 |
| 16, 64, 8, 2048, 128 | yes | 900 | 414 | 413 | 945 | 2.17 | 0.95 |
| 16, 64, 8, 4096, 128 | no | 1242 | 690 | 689 | 1254 | 1.80 | 0.99 |
| 16, 64, 8, 4096, 128 | yes | 1092 | 432 | 432 | 1120 | 2.53 | 0.98 |
| 16, 64, 8, 8192, 128 | no | 1283 | 772 | 767 | 1277 | 1.66 | 1.00 |
| 16, 64, 8, 8192, 128 | yes | 1192 | 497 | 496 | 1212 | 2.40 | 0.98 |
| 1, 32, 8, 4096, 128 | no | 1149 | 659 | 646 | 1145 | 1.74 | 1.00 |
| 1, 32, 8, 4096, 128 | yes | 896 | 349 | 346 | 974 | 2.57 | 0.92 |
| 16, 64, 8, 4096, 64 | no | 933 | 726 | 716 | 929 | 1.28 | 1.00 |
| 16, 64, 8, 4096, 64 | yes | 796 | 539 | 527 | 855 | 1.48 | 0.93 |

What could not be measured, and why:
* **aiter** is not installable here. PyPI's `aiter` is an unrelated 2019 package, and
  `amd-aiter` is not published. AMD's aiter builds from source with its CK submodule,
  which was not attempted.
* **No attention backward and no paged or varlen variant.** HipKittens ships
  `gqa_backwards`, which was not wrapped.

## 6. The route table

`route_tune.py --paths hipkittens --rotate-mb 768 --rounds 5`, run on Qwen3-8B and
Llama-3-8B shapes (config.json written from the published architectures; there is no HF
cache here), at M = 256, 512, …, 8192, 12288 and 16384 (the GEMM takes multiples of 256
only). The engine is `F.linear`, since vLLM is not in the container. An M takes
HipKittens only where it beat `F.linear` by 3%. The table keeps every timing it decided
from.

| layer | N × K | M routed to HipKittens (hipBLASLt time / HipKittens time) |
|---|---|---|
| qkv | 6144 × 4096 | 3840 (1.06), 4352 (1.12), 4864 (1.18), 7424 (1.14), 7680 (1.14), 12288 (1.15) |
| o | 4096 × 4096 | 1792 (1.07), 2048 (1.27), 3328 (1.03), 3840 (1.16), 6400 (1.04), 6656 (1.04), 7424 (1.15), 7680 (1.14) |
| gate_up (Qwen3) | 24576 × 4096 | 3072 (1.11), 3840 (1.10), 4352 (1.08), 4864 (1.09), 5632 (1.12), 6912 (1.13), 7680 (1.09) |
| down (Qwen3) | 4096 × 12288 | 3328 (1.04), 3840 (1.22), 6656 (1.07), 7424 (1.12), 7680 (1.17) |
| gate_up (Llama) | 28672 × 4096 | 3840 (1.11), 4352 (1.09), 4864 (1.11), 5376 (1.12), 5888 (1.16), 7680 (1.11) |
| down (Llama) | 4096 × 14336 | 3328 (1.09), 3840 (1.22), 6656 (1.08), 7424 (1.15), 7680 (1.20) |
| lm_head | 151936 / 128256 × 4096 | none (151936 is not a multiple of 256; 128256 lost at M=256) |

How `vllm_gemm` reads it:
* Every other M takes `F.linear`, including every M that is not a multiple of 256 and
  every M below 256.
* `vllm_gemm` knows the path: it is in `PATCH.paths`, `route_for` checks
  `hipkittens.gemm_fits`, and `linear` checks the tensors and falls back on
  `Unavailable`.
* The comb is hipBLASLt's. Its kernel choice changes between neighbouring multiples of
  256, and powers of two (4096, 8192, 16384) are where it is best.
* A re-run of the tuner agreed on 271 of the 272 lookups over the 34-M grid. The flip is
  o at M=3328, a 1.03 margin.
* The table is only as good as the hipBLASLt it was measured against (torch 2.10, HIP
  7.2.53211, no TunableOp).

## 7. Tests

| tier | file | result |
|---|---|---|
| IR (no torch, no hipcc) | `tests/hk/ir/test_hipkittens.py`, 37 | pass. Full tier at the merge: 553 passed, 2 skipped |
| codegen (hipcc) | `tests/hk/codegen/test_hipkittens_builds.py`, 2 builds | pass, 14 s cold on ROCm 7.1. The GQA build xfails, with the reason, under a clang-22 hipcc |
| GPU (gfx950) | `tests/hk/gpu/test_hipkittens.py`, 31 | ROCm 7.2.3: 21 pass, 10 skip (GQA does not build). ROCm 7.1: 14 pass, 17 skip (the GEMM fails its check) |

The IR tier covers:
* every shape rule;
* the refusal of gfx1100, gfx1201 and gfx942;
* that the source is upstream verbatim, with hk's entry point and stream;
* one GQA source per shape, none per batch;
* the gfx950 table's arch, engine and paths, and that no M below 256 or ragged M takes
  HipKittens;
* that the gfx1100 table is not loaded as gfx950.

The GPU tier covers:
* correctness against fp32;
* no write past C;
* the refusal of strided and misaligned tensors;
* the ops' fallbacks and counters;
* `torch.compile(fullgraph=True)` and CUDA-graph replay of both ops;
* a forced check failure (`Miscompiled`, then fallback with one warning);
* a first launch under capture refused;
* `hk_vllm::linear` on the gfx950 table.

`tests/hk/gpu/test_vllm_gemm.py` on gfx950 had 4 failures. All four assert gfx1100's
`hk`/`hk_decode` routes, and they failed the same way with no gfx950 table. They now skip
where the arch's table sends nothing to `hk.ops.gemm`, as gfx950's does;
`test_hipkittens.py` covers what it sends to HipKittens.

gfx1100 and gfx1201 output is byte-identical: every scaffold of all 509 example and
`hk.ops` instantiations matches the `cdna` dump.

## 8. Risks and what is left

* **Toolchain sensitivity is the main risk.** Either kernel can change under a hipcc
  bump. The on-device check catches a wrong answer at its probe shapes, not a race that
  appears only at other shapes or under load. The GEMM failure was the same at every
  shape tried.
* **Upstream's waits are hand-counted for one schedule.** `s_waitcnt vmcnt(N)` against the
  DMA is counted by hand, and `asm volatile` `ds_read`s carry no dependence. A compiler
  that schedules differently can break them silently. The ISA comparison above found the
  waits intact; the 7.1 GEMM error lies elsewhere. *Since then
  ([chapter 33](../history/33-cdna-gemm.md)):* written in hk, the same schedule's waits
  are derived and checked, and the check found one read they do not cover. The fourth
  memory phase reads `Bs[1][0]` with the copy into it still outstanding after the
  `vmcnt(6)`, right by timing. Patched back into hk's kernel the read is still right
  under 7.1, so it is not the miscompile. hk's GEMM, which reads that tile a phase
  later, is correct under ROCm 7.1 and 7.2.3, and runs level with this one.
* **The descriptor sizes are 32-bit**, which caps operand size (the table in §1).
* **To route ragged Ms**, the GEMM needs a bounded C store (row-predicated, or through
  the buffer descriptor). Padding A costs a copy that loses most of the 6–27% margins.
* **The kernel sources are not in the wheel.** `$HK_HIPKITTENS_KERNELS` points at a
  checkout's `kernels/cdna4`.
* **Next steps**:
  * the vLLM end-to-end measurement on MI355X, where the route table matters only at
    prefill Ms of 1792 and above;
  * attention under vLLM, which needs a KV-cache/varlen variant HipKittens does not ship
    for CDNA4;
  * hipBLASLt with TunableOp as the baseline.

## Reproduce

```bash
# host, ROCm 7.1 + py3.10: the IR and codegen tiers
PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider -m 'not gpu' tests/hk/ir -n 16
PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider tests/hk/codegen/test_hipkittens_builds.py
# in rocm/pytorch:rocm7.2.3 (cp312 hk._mlir mounted, /opt/rocm-7.1.0 mounted read-only), one MI355X
HK_ARCH=gfx950 python3 -m pytest tests/hk/gpu/test_hipkittens.py
HK_ARCH=gfx950 HK_HIPCC=/opt/rocm-7.1.0/bin/hipcc python3 -m pytest tests/hk/gpu/test_hipkittens.py
PYTHONPATH=<flydsl wheel> python3 tools/hk-bench/hipkittens_cdna.py gemm --flydsl --graphs --out gemm.json
HK_HIPCC=/opt/rocm-7.1.0/bin/hipcc PYTHONPATH=<flydsl wheel> \
  python3 tools/hk-bench/hipkittens_cdna.py gqa --flydsl --out gqa.json
python3 tools/hk-bench/route_tune.py --paths hipkittens --rotate-mb 768 --rounds 5 \
  --model qwen3-8b/config.json,llama-3-8b/config.json --name Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B \
  --m $(seq -s, 256 256 8192),12288,16384 --out python/hk/tuned/routes_vllm_linear_gfx950.json
```
