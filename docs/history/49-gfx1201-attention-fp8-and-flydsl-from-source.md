# gfx1201's first tuning: attention against AITER's FlyDSL kernel, an fp8 GEMM, and FlyDSL from source

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 48](48-gfx1201-ceilings-and-first-scoreboard.md) put FlyDSL and AITER
in the gfx1201 scoreboard and found one loss that mattered: AITER ships a
FlyDSL attention written for gfx1201, and it beat hk's -- causal 1.41 ms
against 1.71, 21%. This chapter closes that, adds what gfx1201 has and
gfx1100 does not (an fp8 WMMA), and builds FlyDSL from source to put the RDNA4
kernels its release does not have in the table too.

## FlyDSL, the release and the source

FlyDSL 0.4.0 is the newest wheel on PyPI (v0.4.1 is a git tag with no wheel),
and AITER pins it; that is what chapter 48 ran. Main has one RDNA4 addition
past it, gfx120's integer WMMA and an INT4 GEMM (FlyDSL #1266), which needs
the dialect rebuilt. So FlyDSL main (888d7d8, 0.4.1.dev988) was built the way
its README gives: `scripts/build_llvm.sh` (the `amd-minimal` profile: MLIR,
X86 and AMDGPU), `scripts/build.sh`, `pip install -e .`, in the rocm7.2.3
container with a venv of its own. Two things stood in the way, neither
FlyDSL's: a git worktree's `.git` names a host path the container does not
have, so `build.sh`'s submodule step failed (a plain clone instead), and
`patchelf` was missing (pip has it). FlyDSL's own RDNA4 tests on the R9700:
100 passed, 78 skipped (gfx11-only, multi-GPU), none failed -- the RDNA GEMMs,
the INT4 GEMM, the WMMA atom tests, rmsnorm, softmax, layernorm. AITER's
gfx1201 attention runs on the source build too.

What FlyDSL has for RDNA4, then: a bf16/f16 GEMM (one 128x128x32 tile, no
tuner), an fp8 GEMM built for decode (32-row tiles, no LDS, B preshuffled),
the INT4 GEMM, and its arch-free norms and softmax. Not attention: its dense
entry point goes to `flash_attn_generic`, whose global-to-LDS load gfx12 does
not have, and LLVM aborts the process on it. The RDNA4 attention written in
FlyDSL is AITER's.

## Attention

hk's attention on gfx1201 ran gfx1100's tiling: 12 warps (a 192-row Q tile),
32-row KV blocks, the grid numbered Q block fastest. A sweep against AITER's
kernel in one process (`tools/rdna4-probes/attn_sweep.py`), n4096, 32 query
over 8 KV heads, d128, ms:

| | gfx1100's tiling | 16 warps | + heads_first | + warp_skip, KV 64 | AITER |
|---|---|---|---|---|---|
| causal n1024 | 0.183 | -- | 0.122 | 0.108 | 0.106 |
| causal n2048 | 0.531 | 0.484 | 0.411 | 0.370 | 0.372 |
| causal n4096 | 1.802 | 1.665 | 1.581 | 1.461 | 1.433 |
| causal n8192 | 6.323 | 6.079 | 5.921 | 5.585 | 5.875 |
| full n4096 | 2.968 | 2.829 | 2.835 | -- | 2.793 |

Three changes, each its own knob on `attn_kernel`:

- **Sixteen warps.** A 256-row Q tile: on a part with twice gfx1100's WMMA
  rate a CU and two thirds the CUs, more work a workgroup is the larger step.
- **`heads_first`.** gfx1100's kernel reverses the Q blocks within a head so
  that the long causal ones start first, but its grid numbers Q blocks
  fastest, so the launch order interleaves every head's long blocks with the
  short ones before them. AITER numbers heads fastest and reverses the Q
  index, so all heads' longest blocks launch before anyone's shorter ones:
  longest-processing-time first across the whole grid. So does hk now.
- **`warp_skip`.** A 256-row tile's last KV blocks are fully masked for its
  first warps; the mask fills them with -inf and the warps did the math
  anyway. Now a warp whose rows all precede a block skips its math there (a
  wave-uniform branch; the staging around it stays workgroup-wide). With it a
  64-row KV block pays for causal; for full attention the 64-row block costs
  15%, so full keeps 32.

`_TILING["gfx1201"]` holds them for d128. A Q shorter than 256 rows keeps
gfx1100's tiling, so what `attention` accepts does not move; d64 and gfx1100
are untouched (gfx1100 may well want `heads_first` too -- not measured).

## An fp8 GEMM

gfx1201 has `v_wmma_f32_16x16x16_fp8_fp8`, twice the bf16 rate (372 TFLOP/s
measured). hk.ops.gemm's kernel took `op_dtype=fp8e4m3` with one change -- the
LDS size in bytes, not elements -- and was right on its first run (rel err
3e-3, the bf16 store's): the include tree chapter 47 measured, fp8 fragments
included, did the rest. A sweep of 52 schedules
against hipBLASLt (`tools/rdna4-probes/fp8_sweep.py`) picked twice DEFAULT's
rows: 256x128x64, 48 KB of LDS, 240 TFLOP/s at 4096³ (hipBLASLt 236).

`hk.ops.scaled_mm(a, b, scale_a, scale_b, bias=None)` is torch._scaled_mm's
product in matmul's layout: per-tensor, or a scale a token and a channel,
applied in the epilogue. Up to 128 rows it takes DEFAULT's 128-row block, and
up to 32 rows hk.ops.decode_gemm's kernel, which now takes fp8 too: W
straight into registers, no LDS (`tools/rdna4-probes/fp8_decode.py`; at
32x14336x4096 48 us against the tiled kernel's 89).

## The scoreboard, every arm in one process

FlyDSL from source (0.4.1.dev988) and AITER (ffd08f2) beside hk; median of
five queued rounds; Triton and FlyDSL are the faster of their default and
tuned arms; "hk vs best" counts every other arm.

| case | hk | torch | Triton | FlyDSL | AITER | hk vs best |
|---|---|---|---|---|---|---|
| GEMM 4096³, ms | **0.979** | 1.488 | 1.049 | 1.194 | 1.493 | 1.07x |
| GEMM, four 2048-row shapes | **best** | | | 0.77-0.83x hk | = torch | 1.06-1.08x |
| attention n4096 causal, ms | **1.404** | 5.807 | -- | aborts | 1.409 | tie |
| attention n4096 full, ms | **2.650** | 5.293 | -- | aborts | 2.693 | tie (1.02x) |
| rms_norm 1 / 32 / 2048 / 8192 rows | | | | | | tie / 1.33x / **0.96x** / tie |
| fused add + rms_norm 1 / 32 / 2048 rows | | | | | | 1.06x / 1.30x / tie |
| fused add + rms_norm 8192 rows, us | 481 | 572 | 455 | 450 | **446** | **0.93x** |
| softmax 2048 / 8192 rows | | | | | -- | tie / tie |
| fp8 32x4096x4096, us | **28.3** | 44.0 | -- | 48.2 | 46.7 | 1.56x |
| fp8 32x14336x4096, us | 57.7 | 145 | -- | **47.9** | 52.6 | **0.83x** |
| fp8 128 rows (N 4096 / 14336) | **best** | | | | | 1.07x / 1.15x |
| fp8 2048 rows (N 4096 / 14336) | **best** | | | 0.08x hk | | 1.48x / 1.37x |
| int8 2048 / 4096 rows (N, K 4096) | 0.339 / 0.700 ms | **0.325 / 0.634** | -- | (gfx11 only) | 0.786 / 1.578 | **0.96x / 0.91x** |
| int4 2048 / 4096 rows, ms | -- | -- | -- | 0.169 / 0.402 | -- | (FlyDSL alone) |

The fp8 rows are W8A8 with a scale a token and a channel, which is the
rowwise path in hipBLASLt and a slow one: hk is 1.4x it from 2048 rows.
FlyDSL's fp8 kernel is built for 32-row decode and wins the wide decode row;
it is not meant for prefill (0.08x hk at 2048 rows).

## What is left

- **fp8 decode on a wide layer**: FlyDSL's LDS-free 32-row kernel, 48 us
  against hk's 58 at 32x14336x4096.
- **int8**: hk's int8 GEMM, gfx1100's schedule on the iu8 WMMA, is
  0.91-0.96x torch._int_mm. Nothing has been tuned for it.
- **int4**: hk has no iu4 WMMA path; FlyDSL main does.
- **The norms**: fused add + RMSNorm at 8192 rows 0.93x and rms_norm at 2048
  rows 0.96x. In a probe of their own, the LDS reserve made at most 2% of
  difference at 8192 rows and hk tied tuned Triton there (448 vs 447 us), so
  the scoreboard's loss is in part where the row runs; not settled.
