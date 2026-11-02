### moe-w4 at Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 1 | 36.80 | 64.87 (vLLM Triton (searched config)) | **1.76x** | 36.80 | 64.87 | 92.87 |
| E256 top8 2048x512 | 2 | 62.13 | 116.75 (vLLM Triton (searched config)) | **1.88x** | 62.13 | 116.75 | 237.77 |
| E256 top8 2048x512 | 4 | 106.88 | 203.55 (vLLM Triton (searched config)) | **1.91x** | 106.88 | 203.55 | 486.38 |
| E256 top8 2048x512 | 8 | 195.88 | 371.23 (vLLM Triton (searched config)) | **1.90x** | 195.88 | 371.23 | 874.60 |
| E256 top8 2048x512 | 16 | 339.43 | 624.25 (vLLM Triton (searched config)) | **1.84x** | 339.43 | 624.25 | 1382.36 |
| E256 top8 2048x512 | 32 | 559.87 | 983.91 (vLLM Triton (searched config)) | **1.76x** | 559.87 | 983.91 | 1850.37 |
| E256 top8 2048x512 | 64 | 749.90 | 1373.18 (vLLM Triton (searched config)) | **1.83x** | 749.90 | 1373.18 | 2940.38 |
| E256 top8 2048x512 | 256 | 957.62 | 1581.28 (vLLM Triton (searched config)) | **1.65x** | 957.62 | 1581.28 | 3328.77 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=1: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16, E256 top8 2048x512 M=32, E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config), by row:
  - E256 top8 2048x512 M=1, E256 top8 2048x512 M=2, E256 top8 2048x512 M=4, E256 top8 2048x512 M=8, E256 top8 2048x512 M=16: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 16, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=32: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 32, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=64, E256 top8 2048x512 M=256: moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}

- AITER: n/a: aiter's fused_moe kernels are CDNA-only
- FlyDSL: n/a: no RDNA MoE kernel in FlyDSL

