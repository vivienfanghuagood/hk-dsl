### linear-w4 at Qwen/Qwen3-8B-AWQ (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 26.36 | 56.17 (vLLM awq_gemm (Triton)) | **2.13x** | 26.36 | 102.87 | 56.17 |
| qkv | 2 | 26.80 | 55.99 (vLLM awq_gemm (Triton)) | **2.09x** | 26.80 | 102.69 | 55.99 |
| qkv | 4 | 26.89 | 56.42 (vLLM awq_gemm (Triton)) | **2.10x** | 26.89 | 103.19 | 56.42 |
| qkv | 8 | 27.19 | 57.17 (vLLM awq_gemm (Triton)) | **2.10x** | 27.19 | 103.07 | 57.17 |
| qkv | 16 | 27.54 | 58.34 (vLLM awq_gemm (Triton)) | **2.12x** | 27.54 | 103.21 | 58.34 |
| qkv | 32 | 33.98 | 64.09 (vLLM awq_gemm (Triton)) | **1.89x** | 33.98 | 103.92 | 64.09 |
| qkv | 64 | 63.29 | 104.36 (vLLM AWQ dequantize + matmul) | **1.65x** | 63.29 | 104.36 | 117.94 |
| qkv | 256 | -- | 187.17 (vLLM AWQ dequantize + matmul) | -- | -- | 187.17 | 462.66 |
| o | 1 | 18.74 | 40.80 (vLLM awq_gemm (Triton)) | **2.18x** | 18.74 | 74.25 | 40.80 |
| o | 2 | 19.07 | 40.44 (vLLM awq_gemm (Triton)) | **2.12x** | 19.07 | 74.17 | 40.44 |
| o | 4 | 19.09 | 40.64 (vLLM awq_gemm (Triton)) | **2.13x** | 19.09 | 74.41 | 40.64 |
| o | 8 | 19.29 | 41.10 (vLLM awq_gemm (Triton)) | **2.13x** | 19.29 | 74.56 | 41.10 |
| o | 16 | 19.36 | 53.31 (vLLM awq_gemm (Triton)) | **2.75x** | 19.36 | 75.29 | 53.31 |
| o | 32 | 23.99 | 46.32 (vLLM awq_gemm (Triton)) | **1.93x** | 23.99 | 78.44 | 46.32 |
| o | 64 | 44.22 | 91.32 (vLLM awq_gemm (Triton)) | **2.06x** | 44.22 | 94.01 | 91.32 |
| o | 256 | -- | 302.56 (vLLM AWQ dequantize + matmul) | -- | -- | 302.56 | 313.39 |
| gate_up | 1 | 91.12 | 193.83 (vLLM awq_gemm (Triton)) | **2.13x** | 91.12 | 789.42 | 193.83 |
| gate_up | 2 | 92.17 | 194.51 (vLLM awq_gemm (Triton)) | **2.11x** | 92.17 | 790.58 | 194.51 |
| gate_up | 4 | 92.99 | 194.88 (vLLM awq_gemm (Triton)) | **2.10x** | 92.99 | 792.39 | 194.88 |
| gate_up | 8 | 94.08 | 197.46 (vLLM awq_gemm (Triton)) | **2.10x** | 94.08 | 790.70 | 197.46 |
| gate_up | 16 | 95.24 | 204.77 (vLLM awq_gemm (Triton)) | **2.15x** | 95.24 | 790.99 | 204.77 |
| gate_up | 32 | 118.88 | 221.11 (vLLM awq_gemm (Triton)) | **1.86x** | 118.88 | 792.33 | 221.11 |
| gate_up | 64 | 224.97 | 439.66 (vLLM awq_gemm (Triton)) | **1.95x** | 224.97 | 810.28 | 439.66 |
| gate_up | 256 | -- | 883.28 (vLLM AWQ dequantize + matmul) | -- | -- | 883.28 | 1985.85 |
| down | 1 | 52.77 | 106.07 (vLLM awq_gemm (Triton)) | **2.01x** | 52.77 | 411.09 | 106.07 |
| down | 2 | 52.38 | 104.51 (vLLM awq_gemm (Triton)) | **2.00x** | 52.38 | 409.19 | 104.51 |
| down | 4 | 53.27 | 105.10 (vLLM awq_gemm (Triton)) | **1.97x** | 53.27 | 408.88 | 105.10 |
| down | 8 | 53.47 | 105.99 (vLLM awq_gemm (Triton)) | **1.98x** | 53.47 | 410.74 | 105.99 |
| down | 16 | 53.20 | 118.31 (vLLM awq_gemm (Triton)) | **2.22x** | 53.20 | 412.58 | 118.31 |
| down | 32 | 65.89 | 112.83 (vLLM awq_gemm (Triton)) | **1.71x** | 65.89 | 415.38 | 112.83 |
| down | 64 | 116.67 | 222.78 (vLLM awq_gemm (Triton)) | **1.91x** | 116.67 | 417.49 | 222.78 |
| down | 256 | -- | 845.88 (vLLM awq_gemm (Triton)) | -- | -- | 1454.48 | 845.88 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

