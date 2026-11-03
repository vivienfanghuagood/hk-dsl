### moe-w4 at Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 1 | 37.03 | 64.97 (vLLM Triton (searched config)) | **1.75x** | 37.03 | 64.97 | 93.21 |
| E256 top8 2048x512 | 2 | 62.81 | 118.42 (vLLM Triton (searched config)) | **1.89x** | 62.81 | 118.42 | 240.29 |
| E256 top8 2048x512 | 4 | 107.17 | 207.24 (vLLM Triton (searched config)) | **1.93x** | 107.17 | 207.24 | 488.08 |
| E256 top8 2048x512 | 8 | 196.91 | 380.57 (vLLM Triton (searched config)) | **1.93x** | 196.91 | 380.57 | 878.78 |
| E256 top8 2048x512 | 16 | 340.86 | 654.31 (vLLM Triton (searched config)) | **1.92x** | 340.86 | 654.31 | 1398.92 |
| E256 top8 2048x512 | 32 | 563.43 | 1047.34 (vLLM Triton (searched config)) | **1.86x** | 563.43 | 1047.34 | 1989.74 |
| E256 top8 2048x512 | 64 | 755.05 | 1465.97 (vLLM Triton (searched config)) | **1.94x** | 755.05 | 1465.97 | 2980.04 |
| E256 top8 2048x512 | 256 | 958.72 | 1690.89 (vLLM Triton (searched config)) | **1.76x** | 958.72 | 1690.89 | 3386.96 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=32, E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=16, E256 top8 2048x512 M=64: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 16, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 32, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

