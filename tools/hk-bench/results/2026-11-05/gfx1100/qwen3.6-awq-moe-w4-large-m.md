### moe-w4 at cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.9.1+gitff65f5b, HIP 7.2.53211-e1a6bc5663, vLLM 0.19.2rc1.dev145+g2f314bc5e.d20260424, Triton 3.5.1, FlyDSL 0.4.0, AITER installed, hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk | vLLM Triton (searched config) | vLLM Triton (vLLM's config) |
|---|---:|---:|---|---:|---:|---:|---:|
| E256 top8 2048x512 | 64 | 784.26 | 2216.95 (vLLM Triton (searched config)) | **2.83x** | 784.26 | 2216.95 | 8062.16 |
| E256 top8 2048x512 | 256 | 939.52 | 2543.63 (vLLM Triton (searched config)) | **2.71x** | 939.52 | 2543.63 | 9057.61 |
| E256 top8 2048x512 | 512 | 1300.13 | 2583.14 (vLLM Triton (searched config)) | **1.99x** | 1300.13 | 2583.14 | 9241.24 |
| E256 top8 2048x512 | 1024 | 2155.94 | 3228.30 (vLLM Triton (searched config)) | **1.50x** | 2155.94 | 3228.30 | 9273.33 |
| E256 top8 2048x512 | 2048 | 3751.45 | 4710.52 (vLLM Triton (searched config)) | **1.26x** | 3751.45 | 4710.52 | 12721.19 |
| E256 top8 2048x512 | 4096 | 6987.95 | 7996.16 (vLLM Triton (searched config)) | **1.14x** | 6987.95 | 7996.16 | 21408.12 |

**Setup of each arm.**
- hk: hk.ops.moe.moe_gemm_w4, BLOCK_M 16, the schedule its vLLM route picks
- vLLM Triton (searched config), by row:
  - E256 top8 2048x512 M=64: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=256, E256 top8 2048x512 M=512: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
  - E256 top8 2048x512 M=1024, E256 top8 2048x512 M=2048, E256 top8 2048x512 M=4096: best of vLLM's config and 108 searched here (0 refused by the compiler or the LDS): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2, 'SPLIT_K': 1}
- vLLM Triton (vLLM's config): moe_wna16 Triton kernel, config {'BLOCK_SIZE_M': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 1}
