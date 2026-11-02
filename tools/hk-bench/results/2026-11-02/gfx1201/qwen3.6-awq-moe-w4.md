### moe-w4 at Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 1 | 34.92 | 69.28 (vLLM Triton (searched config)) | **1.98x** | 34.92 | 69.28 | 106.42 |
| E256 top8 2048x512 | 2 | 54.02 | 127.32 (vLLM Triton (searched config)) | **2.36x** | 54.02 | 127.32 | 230.27 |
| E256 top8 2048x512 | 4 | 112.82 | 212.29 (vLLM Triton (searched config)) | **1.88x** | 112.82 | 212.29 | 497.43 |
| E256 top8 2048x512 | 8 | 197.75 | 366.51 (vLLM Triton (searched config)) | **1.85x** | 197.75 | 366.51 | 865.38 |
| E256 top8 2048x512 | 16 | 334.99 | 614.06 (vLLM Triton (searched config)) | **1.83x** | 334.99 | 614.06 | 1367.68 |
| E256 top8 2048x512 | 32 | 561.64 | 981.45 (vLLM Triton (searched config)) | **1.75x** | 561.64 | 981.45 | 1851.85 |
| E256 top8 2048x512 | 64 | 758.30 | 1367.44 (vLLM Triton (searched config)) | **1.80x** | 758.30 | 1367.44 | 2957.08 |
| E256 top8 2048x512 | 256 | 950.15 | 1543.00 (vLLM Triton (searched config)) | **1.62x** | 950.15 | 1543.00 | 3329.17 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=2: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16, E256 top8 2048x512 M=32, E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 16, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 32, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL
