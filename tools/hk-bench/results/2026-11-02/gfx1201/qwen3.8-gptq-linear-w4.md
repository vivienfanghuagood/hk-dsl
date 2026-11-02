### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 71.14 | 83.27 (vLLM gptq_gemm (exllama)) | **1.17x** | 71.14 | 83.27 | error |
| qkv | 2 | 70.73 | 89.17 (vLLM gptq_gemm (exllama)) | **1.26x** | 70.73 | 89.17 | error |
| qkv | 3 | 71.13 | 95.49 (vLLM gptq_gemm (exllama)) | **1.34x** | 71.13 | 95.49 | error |
| qkv | 4 | 71.14 | 101.47 (vLLM gptq_gemm (exllama)) | **1.43x** | 71.14 | 101.47 | error |
| qkv | 8 | 72.01 | 119.30 (vLLM gptq_gemm (exllama)) | **1.66x** | 72.01 | 119.30 | error |
| qkv | 16 | 73.28 | 154.50 (vLLM gptq_gemm (exllama)) | **2.11x** | 73.28 | 154.50 | error |
| qkv | 32 | 89.29 | 300.55 (vLLM gptq_gemm (exllama)) | **3.37x** | 89.29 | 300.55 | error |
| qkv | 64 | 176.34 | 587.28 (vLLM gptq_gemm (exllama)) | **3.33x** | 176.34 | 587.28 | error |
| o | 1 | 31.80 | 68.41 (vLLM gptq_gemm (exllama)) | **2.15x** | 31.80 | 68.41 | error |
| o | 2 | 31.95 | 73.51 (vLLM gptq_gemm (exllama)) | **2.30x** | 31.95 | 73.51 | error |
| o | 3 | 32.17 | 77.67 (vLLM gptq_gemm (exllama)) | **2.41x** | 32.17 | 77.67 | error |
| o | 4 | 32.28 | 81.56 (vLLM gptq_gemm (exllama)) | **2.53x** | 32.28 | 81.56 | error |
| o | 8 | 32.62 | 92.66 (vLLM gptq_gemm (exllama)) | **2.84x** | 32.62 | 92.66 | error |
| o | 16 | 34.08 | 80.81 (vLLM gptq_gemm (exllama)) | **2.37x** | 34.08 | 80.81 | error |
| o | 32 | 39.48 | 137.00 (vLLM gptq_gemm (exllama)) | **3.47x** | 39.48 | 137.00 | error |
| o | 64 | 72.36 | 147.33 (vLLM gptq_gemm (exllama)) | **2.04x** | 72.36 | 147.33 | error |
| qkvz | 1 | 79.68 | 89.33 (vLLM gptq_gemm (exllama)) | **1.12x** | 79.68 | 89.33 | error |
| qkvz | 2 | 79.64 | 94.48 (vLLM gptq_gemm (exllama)) | **1.19x** | 79.64 | 94.48 | error |
| qkvz | 3 | 79.76 | 99.06 (vLLM gptq_gemm (exllama)) | **1.24x** | 79.76 | 99.06 | error |
| qkvz | 4 | 80.15 | 104.11 (vLLM gptq_gemm (exllama)) | **1.30x** | 80.15 | 104.11 | error |
| qkvz | 8 | 81.28 | 131.68 (vLLM gptq_gemm (exllama)) | **1.62x** | 81.28 | 131.68 | error |
| qkvz | 16 | 82.84 | 175.39 (vLLM gptq_gemm (exllama)) | **2.12x** | 82.84 | 175.39 | error |
| qkvz | 32 | 97.65 | 346.16 (vLLM gptq_gemm (exllama)) | **3.54x** | 97.65 | 346.16 | error |
| qkvz | 64 | 191.26 | 672.15 (vLLM gptq_gemm (exllama)) | **3.51x** | 191.26 | 672.15 | error |
| out | 1 | 31.65 | 67.01 (vLLM gptq_gemm (exllama)) | **2.12x** | 31.65 | 67.01 | error |
| out | 2 | 31.87 | 71.90 (vLLM gptq_gemm (exllama)) | **2.26x** | 31.87 | 71.90 | error |
| out | 3 | 32.02 | 76.33 (vLLM gptq_gemm (exllama)) | **2.38x** | 32.02 | 76.33 | error |
| out | 4 | 32.10 | 80.04 (vLLM gptq_gemm (exllama)) | **2.49x** | 32.10 | 80.04 | error |
| out | 8 | 32.42 | 91.01 (vLLM gptq_gemm (exllama)) | **2.81x** | 32.42 | 91.01 | error |
| out | 16 | 34.23 | 81.11 (vLLM gptq_gemm (exllama)) | **2.37x** | 34.23 | 81.11 | error |
| out | 32 | 38.92 | 136.38 (vLLM gptq_gemm (exllama)) | **3.50x** | 38.92 | 136.38 | error |
| out | 64 | 74.17 | 147.25 (vLLM gptq_gemm (exllama)) | **1.99x** | 74.17 | 147.25 | error |
| gate_up | 1 | 163.97 | 174.93 (vLLM gptq_gemm (exllama)) | **1.07x** | 163.97 | 174.93 | error |
| gate_up | 2 | 164.72 | 179.86 (vLLM gptq_gemm (exllama)) | **1.09x** | 164.72 | 179.86 | error |
| gate_up | 3 | 165.31 | 185.72 (vLLM gptq_gemm (exllama)) | **1.12x** | 165.31 | 185.72 | error |
| gate_up | 4 | 165.88 | 193.71 (vLLM gptq_gemm (exllama)) | **1.17x** | 165.88 | 193.71 | error |
| gate_up | 8 | 168.28 | 229.71 (vLLM gptq_gemm (exllama)) | **1.36x** | 168.28 | 229.71 | error |
| gate_up | 16 | 171.88 | 363.91 (vLLM gptq_gemm (exllama)) | **2.12x** | 171.88 | 363.91 | error |
| gate_up | 32 | 201.75 | 741.72 (vLLM gptq_gemm (exllama)) | **3.68x** | 201.75 | 741.72 | error |
| gate_up | 64 | 405.82 | 1437.24 (vLLM gptq_gemm (exllama)) | **3.54x** | 405.82 | 1437.24 | error |
| down | 1 | 83.30 | 112.96 (vLLM gptq_gemm (exllama)) | **1.36x** | 83.30 | 112.96 | error |
| down | 2 | 83.53 | 120.98 (vLLM gptq_gemm (exllama)) | **1.45x** | 83.53 | 120.98 | error |
| down | 3 | 83.71 | 130.85 (vLLM gptq_gemm (exllama)) | **1.56x** | 83.71 | 130.85 | error |
| down | 4 | 83.90 | 140.11 (vLLM gptq_gemm (exllama)) | **1.67x** | 83.90 | 140.11 | error |
| down | 8 | 84.28 | 171.68 (vLLM gptq_gemm (exllama)) | **2.04x** | 84.28 | 171.68 | error |
| down | 16 | 89.10 | 197.18 (vLLM gptq_gemm (exllama)) | **2.21x** | 89.10 | 197.18 | error |
| down | 32 | 105.79 | 365.36 (vLLM gptq_gemm (exllama)) | **3.45x** | 105.79 | 365.36 | error |
| down | 64 | 198.15 | 719.31 (vLLM gptq_gemm (exllama)) | **3.63x** | 198.15 | 719.31 | error |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel

- skipped: ba: left unquantized by the checkpoint

