### moe-w4 at Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 1 | 42.60 | 87.56 (vLLM Triton (searched config)) | **2.06x** | 42.60 | 87.56 | 123.50 |
| E256 top8 2048x512 | 2 | 71.22 | 145.78 (vLLM Triton (searched config)) | **2.05x** | 71.22 | 145.78 | 264.00 |
| E256 top8 2048x512 | 4 | 130.82 | 276.90 (vLLM Triton (searched config)) | **2.12x** | 130.82 | 276.90 | 589.10 |
| E256 top8 2048x512 | 8 | 235.00 | 486.92 (vLLM Triton (searched config)) | **2.07x** | 235.00 | 486.92 | 956.76 |
| E256 top8 2048x512 | 16 | 355.52 | 707.24 (vLLM Triton (searched config)) | **1.99x** | 355.52 | 707.24 | 1387.20 |
| E256 top8 2048x512 | 32 | 593.04 | 1187.40 (vLLM Triton (searched config)) | **2.00x** | 593.04 | 1187.40 | 2357.65 |
| E256 top8 2048x512 | 64 | 788.50 | 1508.38 (vLLM Triton (searched config)) | **1.91x** | 788.50 | 1508.38 | 2676.83 |
| E256 top8 2048x512 | 256 | 930.78 | 1720.61 (vLLM Triton (searched config)) | **1.85x** | 930.78 | 1720.61 | 3087.13 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 16, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 32, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL
