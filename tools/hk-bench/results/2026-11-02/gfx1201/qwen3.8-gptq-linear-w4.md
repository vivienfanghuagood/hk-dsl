### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 78.47 | 83.17 (vLLM gptq_gemm (exllama)) | **1.06x** | 78.47 | 83.17 | error |
| qkv | 2 | 78.29 | 89.48 (vLLM gptq_gemm (exllama)) | **1.14x** | 78.29 | 89.48 | error |
| qkv | 4 | 78.86 | 102.07 (vLLM gptq_gemm (exllama)) | **1.29x** | 78.86 | 102.07 | error |
| qkv | 8 | 79.19 | 118.98 (vLLM gptq_gemm (exllama)) | **1.50x** | 79.19 | 118.98 | error |
| qkv | 16 | 81.45 | 155.17 (vLLM gptq_gemm (exllama)) | **1.91x** | 81.45 | 155.17 | error |
| qkv | 32 | 103.03 | 304.83 (vLLM gptq_gemm (exllama)) | **2.96x** | 103.03 | 304.83 | error |
| qkv | 64 | 172.02 | 596.14 (vLLM gptq_gemm (exllama)) | **3.46x** | 172.02 | 596.14 | error |
| o | 1 | 35.39 | 68.41 (vLLM gptq_gemm (exllama)) | **1.93x** | 35.39 | 68.41 | error |
| o | 2 | 35.51 | 73.56 (vLLM gptq_gemm (exllama)) | **2.07x** | 35.51 | 73.56 | error |
| o | 4 | 36.06 | 81.61 (vLLM gptq_gemm (exllama)) | **2.26x** | 36.06 | 81.61 | error |
| o | 8 | 36.75 | 92.73 (vLLM gptq_gemm (exllama)) | **2.52x** | 36.75 | 92.73 | error |
| o | 16 | 37.73 | 81.34 (vLLM gptq_gemm (exllama)) | **2.16x** | 37.73 | 81.34 | error |
| o | 32 | 46.20 | 136.60 (vLLM gptq_gemm (exllama)) | **2.96x** | 46.20 | 136.60 | error |
| o | 64 | 73.64 | 147.55 (vLLM gptq_gemm (exllama)) | **2.00x** | 73.64 | 147.55 | error |
| qkvz | 1 | 87.95 | 89.73 (vLLM gptq_gemm (exllama)) | **1.02x** | 87.95 | 89.73 | error |
| qkvz | 2 | 88.10 | 94.74 (vLLM gptq_gemm (exllama)) | **1.07x** | 88.10 | 94.74 | error |
| qkvz | 4 | 88.85 | 104.23 (vLLM gptq_gemm (exllama)) | **1.17x** | 88.85 | 104.23 | error |
| qkvz | 8 | 89.66 | 131.72 (vLLM gptq_gemm (exllama)) | **1.47x** | 89.66 | 131.72 | error |
| qkvz | 16 | 92.30 | 174.00 (vLLM gptq_gemm (exllama)) | **1.89x** | 92.30 | 174.00 | error |
| qkvz | 32 | 115.32 | 349.16 (vLLM gptq_gemm (exllama)) | **3.03x** | 115.32 | 349.16 | error |
| qkvz | 64 | 190.46 | 671.62 (vLLM gptq_gemm (exllama)) | **3.53x** | 190.46 | 671.62 | error |
| ba | 1 | 5.60 | 17.18 (vLLM gptq_gemm (exllama)) | **3.07x** | 5.60 | 17.18 | error |
| ba | 2 | 5.63 | 18.92 (vLLM gptq_gemm (exllama)) | **3.36x** | 5.63 | 18.92 | error |
| ba | 4 | 5.66 | 21.72 (vLLM gptq_gemm (exllama)) | **3.83x** | 5.66 | 21.72 | error |
| ba | 8 | 5.70 | 26.63 (vLLM gptq_gemm (exllama)) | **4.67x** | 5.70 | 26.63 | error |
| ba | 16 | 5.75 | 26.90 (vLLM gptq_gemm (exllama)) | **4.68x** | 5.75 | 26.90 | error |
| ba | 32 | 6.33 | 27.51 (vLLM gptq_gemm (exllama)) | **4.34x** | 6.33 | 27.51 | error |
| ba | 64 | 13.13 | 34.97 (vLLM gptq_gemm (exllama)) | **2.66x** | 13.13 | 34.97 | error |
| out | 1 | 35.24 | 67.17 (vLLM gptq_gemm (exllama)) | **1.91x** | 35.24 | 67.17 | error |
| out | 2 | 35.62 | 72.20 (vLLM gptq_gemm (exllama)) | **2.03x** | 35.62 | 72.20 | error |
| out | 4 | 35.95 | 79.99 (vLLM gptq_gemm (exllama)) | **2.23x** | 35.95 | 79.99 | error |
| out | 8 | 36.55 | 91.20 (vLLM gptq_gemm (exllama)) | **2.50x** | 36.55 | 91.20 | error |
| out | 16 | 37.65 | 80.84 (vLLM gptq_gemm (exllama)) | **2.15x** | 37.65 | 80.84 | error |
| out | 32 | 45.54 | 136.62 (vLLM gptq_gemm (exllama)) | **3.00x** | 45.54 | 136.62 | error |
| out | 64 | 74.72 | 148.00 (vLLM gptq_gemm (exllama)) | **1.98x** | 74.72 | 148.00 | error |
| gate_up | 1 | 181.46 | 175.12 (vLLM gptq_gemm (exllama)) | **0.96x** | 181.46 | 175.12 | error |
| gate_up | 2 | 181.95 | 180.18 (vLLM gptq_gemm (exllama)) | **0.99x** | 181.95 | 180.18 | error |
| gate_up | 4 | 183.54 | 193.91 (vLLM gptq_gemm (exllama)) | **1.06x** | 183.54 | 193.91 | error |
| gate_up | 8 | 185.81 | 229.23 (vLLM gptq_gemm (exllama)) | **1.23x** | 185.81 | 229.23 | error |
| gate_up | 16 | 190.94 | 361.35 (vLLM gptq_gemm (exllama)) | **1.89x** | 190.94 | 361.35 | error |
| gate_up | 32 | 234.54 | 725.36 (vLLM gptq_gemm (exllama)) | **3.09x** | 234.54 | 725.36 | error |
| gate_up | 64 | 395.92 | 1437.78 (vLLM gptq_gemm (exllama)) | **3.63x** | 395.92 | 1437.78 | error |
| down | 1 | 97.18 | 112.71 (vLLM gptq_gemm (exllama)) | **1.16x** | 97.18 | 112.71 | error |
| down | 2 | 97.27 | 121.88 (vLLM gptq_gemm (exllama)) | **1.25x** | 97.27 | 121.88 | error |
| down | 4 | 98.40 | 140.35 (vLLM gptq_gemm (exllama)) | **1.43x** | 98.40 | 140.35 | error |
| down | 8 | 98.29 | 171.91 (vLLM gptq_gemm (exllama)) | **1.75x** | 98.29 | 171.91 | error |
| down | 16 | 100.99 | 197.81 (vLLM gptq_gemm (exllama)) | **1.96x** | 100.99 | 197.81 | error |
| down | 32 | 127.00 | 374.82 (vLLM gptq_gemm (exllama)) | **2.95x** | 127.00 | 374.82 | error |
| down | 64 | 190.17 | 719.43 (vLLM gptq_gemm (exllama)) | **3.78x** | 190.17 | 719.43 | error |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul (GPTQ repacked)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
