### moe-w4 at Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 1 | 43.66 | 93.07 (vLLM Triton (searched config)) | **2.13x** | 43.66 | 93.07 | 123.03 |
| E256 top8 2048x512 | 2 | 70.36 | 155.05 (vLLM Triton (searched config)) | **2.20x** | 70.36 | 155.05 | 291.83 |
| E256 top8 2048x512 | 4 | 122.34 | 267.71 (vLLM Triton (searched config)) | **2.19x** | 122.34 | 267.71 | 651.03 |
| E256 top8 2048x512 | 8 | 217.46 | 451.25 (vLLM Triton (searched config)) | **2.08x** | 217.46 | 451.25 | 992.18 |
| E256 top8 2048x512 | 16 | 366.27 | 733.91 (vLLM Triton (searched config)) | **2.00x** | 366.27 | 733.91 | 1472.24 |
| E256 top8 2048x512 | 32 | 589.65 | 1160.01 (vLLM Triton (searched config)) | **1.97x** | 589.65 | 1160.01 | 2342.50 |
| E256 top8 2048x512 | 64 | 769.69 | 1521.15 (vLLM Triton (searched config)) | **1.98x** | 769.69 | 1521.15 | 2694.21 |
| E256 top8 2048x512 | 256 | 1011.73 | 1764.20 (vLLM Triton (searched config)) | **1.74x** | 1011.73 | 1764.20 | 3264.01 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=32: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=2: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16, E256 top8 2048x512 M=64: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 16, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 32, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

