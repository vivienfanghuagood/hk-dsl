### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x256 | 1 | 43.12 | 74.30 (vLLM Triton (vLLM's config)) | **1.72x** | 43.12 | 74.55 | 74.30 |
| E256 top8 2048x256 | 2 | 73.48 | 132.09 (vLLM Triton (searched config)) | **1.80x** | 73.48 | 132.09 | 136.05 |
| E256 top8 2048x256 | 4 | 140.49 | 216.50 (vLLM Triton (searched config)) | **1.54x** | 140.49 | 216.50 | 270.25 |
| E256 top8 2048x256 | 8 | 253.25 | 342.60 (vLLM Triton (searched config)) | **1.35x** | 253.25 | 342.60 | 419.39 |
| E256 top8 2048x256 | 16 | 432.86 | 513.48 (vLLM Triton (searched config)) | **1.19x** | 432.86 | 513.48 | 581.98 |
| E256 top8 2048x256 | 32 | 699.04 | 745.78 (vLLM Triton (searched config)) | **1.07x** | 699.04 | 745.78 | 821.54 |
| E256 top8 2048x256 | 64 | 929.67 | 998.53 (vLLM Triton (searched config)) | **1.07x** | 929.67 | 998.53 | 1039.75 |
| E256 top8 2048x256 | 256 | 1129.03 | 1167.13 (vLLM Triton (searched config)) | **1.03x** | 1129.03 | 1167.13 | 1384.79 |
| E256 top8 2048x256 | 1024 | 1696.45 | 1325.65 (vLLM Triton (searched config)) | **0.78x** | 1696.45 | 1325.65 | 1605.67 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x256 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=2: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=4: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=8: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=16, E256 top8 2048x256 M=32, E256 top8 2048x256 M=64: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=256, E256 top8 2048x256 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=2, E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16, E256 top8 2048x256 M=32: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=64: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=256: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}
  - E256 top8 2048x256 M=1024: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

