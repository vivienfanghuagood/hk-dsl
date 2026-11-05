### moe-w4 at cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.9.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 64 | 725.94 | 1304.54 (vLLM Triton (searched config)) | **1.80x** | 725.94 | 1304.54 | 2797.10 |
| E256 top8 2048x512 | 256 | 941.25 | 1622.69 (vLLM Triton (searched config)) | **1.72x** | 941.25 | 1622.69 | 3298.53 |
| E256 top8 2048x512 | 512 | 1216.95 | 1629.96 (vLLM Triton (searched config)) | **1.34x** | 1216.95 | 1629.96 | 3367.09 |
| E256 top8 2048x512 | 1024 | 1758.32 | 2084.01 (vLLM Triton (searched config)) | **1.19x** | 1758.32 | 2084.01 | 3492.63 |
| E256 top8 2048x512 | 2048 | 3185.53 | 3467.42 (vLLM Triton (searched config)) | **1.09x** | 3185.53 | 3467.42 | 4241.58 |
| E256 top8 2048x512 | 4096 | 6088.13 | 4250.37 (vLLM Triton (searched config)) | **0.70x** | 6088.13 | 4250.37 | 6953.61 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=512, E256 top8 2048x512 M=2048: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=1024: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=4096: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config): moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
