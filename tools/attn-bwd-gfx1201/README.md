# Flash attention backward on gfx1201 (RDNA4) and gfx1100 (RDNA3)

`hk.ops.attn_bwd.attention(q, k, v, causal=..., scale=...)`: flash attention for training, an autograd
function whose forward is hk's attention kernel with a log-sum-exp output and whose backward is three
DSL kernels (dQ, dK, dV) and a small Delta kernel. bf16, BSHD (B, S, H, D), MHA or GQA, causal (top-left,
equal lengths) or not, head_dim 64 or 128, S >= 192. gfx1201 and, with the schedule changes under
*gfx1100* below, gfx1100.

## Results

Radeon AI PRO R9700 (gfx1201), torch 2.10.0 + ROCm 7.2.4 (`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`).
Baseline: torch `F.scaled_dot_product_attention`, which on gfx1201 dispatches AOTriton (`attn_fwd`,
`bwd_preprocess`, `bwd_kernel_dk_dv`, `bwd_kernel_dq`; checked with the profiler). Transformer Engine's
fused attention is not built for gfx1201, so SDPA is the fastest backend a trainer has there. One
process, both sides interleaved, inputs as views of one fused QKV tensor (`bench.py`):

| shape | B S H/H_kv D mask | SDPA fwd+bwd ms | hk fwd+bwd ms | x | hk fwd ms | hk bwd ms (TF) |
|---|---|---|---|---|---|---|
| P1 | 2 2048 16/16 128 causal | 4.807 | 2.270 | 2.12 | 0.429 | 1.841 (47) |
| P2 | 1 2048 16/16 128 causal | 2.538 | 1.205 | 2.11 | 0.255 | 0.949 (45) |
| NC | 2 2048 16/16 128 full | 6.578 | 3.718 | 1.77 | 0.744 | 2.974 (58) |
| G4 | 2 2048 16/4 128 causal | 5.047 | 2.523 | 2.00 | 0.454 | 2.069 (42) |
| L1B | 1 4096 32/8 64 causal | 8.456 | 4.829 | 1.75 | 0.853 | 3.976 (43) |
| L8B | 1 4096 32/8 128 causal | 19.625 | 8.466 | 2.32 | 1.452 | 7.013 (49) |

P1 is the attention of a 1.28B Llama-style model (hidden 2048, 16 heads, sequence 2048, micro-batch 2).
TF counts 4 B H S^2 D for the forward and 2.5x that for the backward (the standard five matmuls), halved
under causal. The bf16 WMMA ceiling measured on this card is 197 TF.

Per kernel (P1 / NC / L1B; TF counts each kernel's own matmuls):

| kernel | P1 ms (TF) | NC ms (TF) | L1B ms (TF) |
|---|---|---|---|
| forward with LSE | 0.367 (94) | 0.718 (96) | 0.824 (83) |
| Delta | 0.027 | 0.028 | 0.034 |
| dQ (3 matmuls) | 0.650 (79) | 1.058 (97) | 1.286 (80) |
| dK (3 matmuls) | 0.648 (80) | 1.064 (97) | 1.485 (69) |
| dV (2 matmuls) | 0.433 (79) | 0.711 (97) | 1.067 (64) |

Each kernel runs at 40-50% of the ceiling; the backward as a whole counts 5 matmuls of work and does 8
(S is rebuilt in all three kernels, dP in two), which is why it reads 45-58 TF.

## Derivation

FA-2's backward, in the orientations hk's forward (`hk.ops.attn`) already uses, so no layout is new:

- The forward writes one more thing, each query row's base-2 log-sum-exp (`attn_kernel(..., lse=True)`),
  fp32; the backward rebuilds P = exp2(S scale log2(e) - LSE) from it instead of redoing the online softmax.
- Delta = rowsum(dO * O), once a row, by its own kernel.
- dQ: a workgroup owns 12 warps x 16 query rows and walks the KV blocks, K and V staged through LDS with
  prefetch. The fp32 accumulator is a col tile, so S^T = K Q^T and dP^T = V dO^T are [kv, q] tiles: LSE and
  Delta are per-query row vectors on the lane axis, as the forward's running max is, and
  dS^T = P^T (dP^T - Delta) is the B operand of dQ^T += K^T dS^T, with K^T staged transposed in LDS by
  register transposes.
- dK and dV: the same algebra with the roles swapped. A workgroup owns 12 x 16 KV rows, each warp's K
  (and V, for dK) in registers, and walks the Q blocks; S = Q K^T and dP = dO V^T are [q, kv] tiles, so P
  and dS are B operands of dV^T += dO^T P and dK^T += Q^T dS, with Q^T or dO^T staged transposed.
- GQA is addressing: the dK/dV workgroup of a KV head loops over its query heads and accumulates, so the
  group sum happens in registers, with no atomics and no reduction pass.
- BSHD in place: a (B, S, H, D) tensor whose H and D are contiguous is the (B, 1, S, H D) global the
  kernels address with one row stride, so q, k, v as slices of one projection output are read where they
  are. Other layouts (Megatron's SBHD) are copied once.

## Budget

VGPRs per lane (no kernel spills; hk rejects a spilling build):

| kernel | d128 | d64 |
|---|---|---|
| forward with LSE (causal / full) | 192 / 158 | 108 / 106 |
| dQ | 225 / 223 | 149 / 147 |
| dK | 236 / 233 | 160 / 157 |
| dV | 168 / 165 | 123 / 121 |
| Delta | 84 | 44 |

## Levers measured and not taken

- **One fused dK/dV kernel**, AOTriton's shape. Both accumulators are 128 VGPRs at d128, so K and V had to
  live in LDS, the Q staging could not be prefetched, and its LDS held a CU to one workgroup: dK + dV
  2.39 ms at P1 against 1.10 ms for the two kernels, which rebuild S once more.
- **Transposing P and dS in registers** (no LDS round trip for the transposed operand): slower.
- **Tile sizes.** Warps 8/12/16 x a 32- or 64-row walked block, per kernel, at P1, NC and L1B: the landscape
  is flat, the shipped 12 warps x 32 within ~5% of the best everywhere; 64-row blocks spill at d128.
- **Delta in torch**: 0.36 ms at P1, against 0.03 ms for the kernel.

## Attribution, and why the backward is not pushed further (for now)

Each stage deleted in turn (`ablate=` on `dq_kernel` / `dkv_kernel`; timing only, the results are wrong). Same shape as
P1 (B2 S2048 16x128 causal), in-place BSHD, one process, interleaved, R9700. The table gives the share of each kernel's
time that disappears. A deletion also removes whatever fed only that stage, so the matmul rows overlap with the others.

| stage deleted | dQ (681 us, 76 TF) | dK (660 us, 78 TF) | dV (450 us, 76 TF) |
|---|---|---|---|
| global staging of K/V (dQ) or Q/dO (dK, dV) | 7% | 5% | 4% |
| the in-kernel register transposes (K^T, Q^T, dO^T) | 13% | 13% | 18% |
| softmax: scale, mask, LSE, exp2 | 7% | 8% | 11% |
| re-reading operand fragments from LDS | -- | 5% | 5% |
| each one of the three (dV: two) matmul groups | ~25% | ~25% | ~33% |

- **What is left:** with staging, transposes, softmax and LDS re-reads all deleted, only the WMMAs and their format
  conversions remain. They run at 116 TF (dK, 443 us) and 120 TF (dV, 286 us), about 61% of the measured 197 TF bf16
  peak. That is the ceiling of this design. It is not LDS-bound.
- **Where the gap is:** the kernels run at 76-78 TF today, and the rest of the gap is the 8 matmuls a backward of this
  shape does where the standard count is 5.
- **What is left to win, per fwd+bwd call:**
  - Moving the transposes out of the kernels: the forward writes K^T and Q^T, the Delta kernel writes dO^T, and the
    backward stages them like any other tile. About 0.28 ms saved against about 0.08 ms of extra writes, net 0.2 ms.
  - A dK/dV fusion that fits the registers, perhaps with head_dim split across warps: about 0.3 ms. Unproven, and the
    fused kernel tried earlier ran at 29 TF.
  - In a 1.28B Megatron FP8 step (64 calls, about 1.3 s) that is roughly 1-2.5%. It was measured and set aside
    (2026-11-19).
- **Ruled out:** dQ by atomics inside the dK kernel. A warp here owns 16 KV rows, so it would add about 2 GB of fp32
  atomics per call, about 4x the operand traffic. A workgroup-level reduction would need about 192 KB of LDS, and the
  dK kernel has no VGPRs left for K^T.

## gfx1100 (W7900D)

**What changes and why.** gfx1100 holds a 16-bit WMMA operand twice, mirrored across the wave's two halves. So each
warp's two register-resident operands cost 128 VGPRs at head_dim 128: Q and dO in dQ, K and V in dK. With them, the
dQ and dK kernels spilled 154 and 180 VGPRs.

At head_dim 128 on gfx1100 (`_SCHEDULES`), each kernel:
- keeps its second operand (dO, V) in an LDS slot of the warp's own (`do_in_lds`, `v_in_lds`);
- copies each K/V (or Q/dO) block into LDS when it is needed, instead of holding the next one in registers
  (`prefetch=False`);
- runs 8 warps, because the slots take 32 KB of LDS.

The result is 239-253 VGPRs and no scratch. dV and head_dim 64 fit as they were.

**Results.** Radeon PRO W7900D (gfx1100), the same image. SDPA dispatches AOTriton here too (`attn_fwd`,
`bwd_kernel_dk_dv`, `bwd_kernel_dq`; checked with the profiler). `bench.py`, one process, interleaved:

| shape | B S H/H_kv D mask | SDPA fwd+bwd ms | hk fwd+bwd ms | x | hk fwd ms | hk bwd ms (TF) |
|---|---|---|---|---|---|---|
| P1 | 2 2048 16/16 128 causal | 12.061 | 3.735 | 3.23 | 0.811 | 2.925 (29) |
| P2 | 1 2048 16/16 128 causal | 6.921 | 2.213 | 3.13 | 0.497 | 1.716 (25) |
| NC | 2 2048 16/16 128 full | 18.183 | 6.013 | 3.02 | 1.261 | 4.752 (36) |
| G4 | 2 2048 16/4 128 causal | 11.983 | 4.183 | 2.86 | 0.816 | 3.367 (26) |
| L1B | 1 4096 32/8 64 causal | 17.601 | 8.378 | 2.10 | 1.660 | 6.718 (26) |
| L8B | 1 4096 32/8 128 causal | 42.874 | 13.938 | 3.08 | 2.732 | 11.206 (31) |

**In training.** A 1.28B Llama-style model in BF16 (torch modules, TunableOp-tuned GEMMs in both arms, real text,
micro-batch 2 x 4, sequence 2048) on one W7900D. 60 steps an arm, in the order SDPA, hk, hk, SDPA:
- 6143 / 6155 tok/s with SDPA, against 7609 / 7616 with hk, which is **1.24x**.
- The loss at step 60 is 7.342 with SDPA and 7.340 with hk.
- Memory is the same.
- Loss over 200 steps, 3 seeds paired: hk minus SDPA was -0.0033 (sd 0.0092) over steps 151-200, signs mixed. A
  placebo arm (SDPA with hipBLASLt's default GEMM algorithms) moved it +0.026.

## What this is not

- No attention mask, no dropout, no bottom-right causal, no Sq != Sk, no variable-length batches.
- No fp8: bf16 in and out, fp32 LSE and Delta.
- Deterministic (no atomics), and not the fastest possible: the 8-vs-5 matmul overhead is the known gap.

## Reproduce

    python -m pytest tests/hk/gpu/test_attn_bwd.py          # o, dq, dk, dv vs float64 autograd (gfx1201, gfx1100)
    python tools/attn-bwd-gfx1201/bench.py                  # the table above

`HK_ARCH=gfx1201` if the arch is not detected.
