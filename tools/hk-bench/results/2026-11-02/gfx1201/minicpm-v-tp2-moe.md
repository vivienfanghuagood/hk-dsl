### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x256 | 1 | 48.68 | 50.58 (vLLM Triton (searched config)) | **1.04x** | 48.68 | 50.58 | 52.02 |
| E256 top8 2048x256 | 2 | 89.15 | 99.41 (vLLM Triton (searched config)) | **1.11x** | 89.15 | 99.41 | 108.86 |
| E256 top8 2048x256 | 4 | 163.81 | 174.10 (vLLM Triton (searched config)) | **1.06x** | 163.81 | 174.10 | 194.40 |
| E256 top8 2048x256 | 8 | 318.32 | 327.84 (vLLM Triton (searched config)) | **1.03x** | 318.32 | 327.84 | 349.23 |
| E256 top8 2048x256 | 16 | 567.68 | 540.32 (vLLM Triton (searched config)) | **0.95x** | 567.68 | 540.32 | 559.88 |
| E256 top8 2048x256 | 32 | 1006.55 | 911.57 (vLLM Triton (searched config)) | **0.91x** | 1006.55 | 911.57 | 918.79 |
| E256 top8 2048x256 | 64 | 1232.70 | 1134.57 (vLLM Triton (searched config)) | **0.92x** | 1232.70 | 1134.57 | 1141.17 |
| E256 top8 2048x256 | 256 | 1568.06 | 1372.62 (vLLM Triton (searched config)) | **0.88x** | 1568.06 | 1372.62 | 1465.98 |
| E256 top8 2048x256 | 1024 | 2186.56 | 1471.02 (vLLM Triton (searched config)) | **0.67x** | 2186.56 | 1471.02 | 1555.52 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=2, E256 top8 2048x256 M=16: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=4: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=8: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=64, E256 top8 2048x256 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=2, E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16, E256 top8 2048x256 M=32: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=64: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=256: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}
  - E256 top8 2048x256 M=1024: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

