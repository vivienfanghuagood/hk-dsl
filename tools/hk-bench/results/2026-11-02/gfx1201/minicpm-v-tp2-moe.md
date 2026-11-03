### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x256 | 1 | 48.82 | 50.67 (vLLM Triton (searched config)) | **1.04x** | 48.82 | 50.67 | 51.85 |
| E256 top8 2048x256 | 2 | 89.19 | 99.26 (vLLM Triton (searched config)) | **1.11x** | 89.19 | 99.26 | 109.01 |
| E256 top8 2048x256 | 4 | 163.82 | 174.15 (vLLM Triton (searched config)) | **1.06x** | 163.82 | 174.15 | 194.28 |
| E256 top8 2048x256 | 8 | 318.37 | 326.30 (vLLM Triton (searched config)) | **1.02x** | 318.37 | 326.30 | 348.70 |
| E256 top8 2048x256 | 16 | 566.55 | 539.23 (vLLM Triton (searched config)) | **0.95x** | 566.55 | 539.23 | 558.89 |
| E256 top8 2048x256 | 32 | 1007.28 | 910.04 (vLLM Triton (searched config)) | **0.90x** | 1007.28 | 910.04 | 921.30 |
| E256 top8 2048x256 | 64 | 1235.15 | 1135.97 (vLLM Triton (searched config)) | **0.92x** | 1235.15 | 1135.97 | 1143.29 |
| E256 top8 2048x256 | 256 | 1575.90 | 1372.99 (vLLM Triton (searched config)) | **0.87x** | 1575.90 | 1372.99 | 1474.43 |
| E256 top8 2048x256 | 1024 | 2222.25 | 1474.39 (vLLM Triton (searched config)) | **0.66x** | 2222.25 | 1474.39 | 1554.15 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x256 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=2, E256 top8 2048x256 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=4: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=8: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=16: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=64, E256 top8 2048x256 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=2, E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16, E256 top8 2048x256 M=32: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=64: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=256: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}
  - E256 top8 2048x256 M=1024: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

