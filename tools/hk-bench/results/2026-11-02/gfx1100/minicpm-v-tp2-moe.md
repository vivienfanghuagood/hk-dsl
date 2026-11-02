### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x256 | 1 | 31.84 | 44.00 (vLLM Triton (vLLM's config)) | **1.38x** | 31.84 | 44.10 | 44.00 |
| E256 top8 2048x256 | 2 | 58.86 | 92.14 (vLLM Triton (searched config)) | **1.56x** | 58.86 | 92.14 | 94.28 |
| E256 top8 2048x256 | 4 | 143.64 | 221.10 (vLLM Triton (searched config)) | **1.54x** | 143.64 | 221.10 | 278.46 |
| E256 top8 2048x256 | 8 | 252.14 | 342.08 (vLLM Triton (searched config)) | **1.36x** | 252.14 | 342.08 | 411.70 |
| E256 top8 2048x256 | 16 | 443.78 | 532.50 (vLLM Triton (searched config)) | **1.20x** | 443.78 | 532.50 | 602.28 |
| E256 top8 2048x256 | 32 | 695.60 | 746.88 (vLLM Triton (searched config)) | **1.07x** | 695.60 | 746.88 | 820.64 |
| E256 top8 2048x256 | 64 | 923.68 | 1001.66 (vLLM Triton (searched config)) | **1.08x** | 923.68 | 1001.66 | 1022.98 |
| E256 top8 2048x256 | 256 | 1128.42 | 1174.90 (vLLM Triton (searched config)) | **1.04x** | 1128.42 | 1174.90 | 1386.38 |
| E256 top8 2048x256 | 1024 | 1695.87 | 1320.27 (vLLM Triton (searched config)) | **0.78x** | 1695.87 | 1320.27 | 1609.97 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x256 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=2: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=4, E256 top8 2048x256 M=8: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=16, E256 top8 2048x256 M=64, E256 top8 2048x256 M=256, E256 top8 2048x256 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=2, E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16, E256 top8 2048x256 M=32: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=64: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=256: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}
  - E256 top8 2048x256 M=1024: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL
