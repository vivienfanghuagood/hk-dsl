### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x256 | 1 | 33.74 | 31.44 (vLLM Triton (vLLM's config)) | **0.93x** | 33.74 | 31.54 | 31.44 |
| E256 top8 2048x256 | 2 | 91.52 | 101.00 (vLLM Triton (vLLM's config)) | **1.10x** | 91.52 | 108.46 | 101.00 |
| E256 top8 2048x256 | 4 | 169.85 | 181.09 (vLLM Triton (searched config)) | **1.07x** | 169.85 | 181.09 | 201.59 |
| E256 top8 2048x256 | 8 | 310.39 | 311.47 (vLLM Triton (searched config)) | **1.00x** | 310.39 | 311.47 | 328.11 |
| E256 top8 2048x256 | 16 | 559.24 | 533.33 (vLLM Triton (searched config)) | **0.95x** | 559.24 | 533.33 | 561.82 |
| E256 top8 2048x256 | 32 | 960.59 | 866.79 (vLLM Triton (searched config)) | **0.90x** | 960.59 | 866.79 | 887.55 |
| E256 top8 2048x256 | 64 | 1245.51 | 1156.29 (vLLM Triton (searched config)) | **0.93x** | 1245.51 | 1156.29 | 1187.13 |
| E256 top8 2048x256 | 256 | 1574.69 | 1362.50 (vLLM Triton (searched config)) | **0.86x** | 1574.69 | 1362.50 | 1433.66 |
| E256 top8 2048x256 | 1024 | 2184.60 | 1472.70 (vLLM Triton (searched config)) | **0.67x** | 2184.60 | 1472.70 | 1554.06 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x256 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=2, E256 top8 2048x256 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=64: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x256 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x256 M=1, E256 top8 2048x256 M=2, E256 top8 2048x256 M=4, E256 top8 2048x256 M=8, E256 top8 2048x256 M=16, E256 top8 2048x256 M=32: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=64: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 4, 'num_stages': 2}
  - E256 top8 2048x256 M=256: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}
  - E256 top8 2048x256 M=1024: fused_moe Triton kernel, config {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL
