### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 70.40 | 82.91 (vLLM gptq_gemm (exllama)) | **1.18x** | 70.40 | 82.91 | error |
| qkv | 2 | 70.63 | 89.19 (vLLM gptq_gemm (exllama)) | **1.26x** | 70.63 | 89.19 | error |
| qkv | 3 | 71.07 | 95.62 (vLLM gptq_gemm (exllama)) | **1.34x** | 71.07 | 95.62 | error |
| qkv | 4 | 71.23 | 101.67 (vLLM gptq_gemm (exllama)) | **1.43x** | 71.23 | 101.67 | error |
| qkv | 8 | 72.48 | 123.85 (vLLM gptq_gemm (exllama)) | **1.71x** | 72.48 | 123.85 | error |
| qkv | 16 | 74.03 | 170.28 (vLLM gptq_gemm (exllama)) | **2.30x** | 74.03 | 170.28 | error |
| qkv | 32 | 92.56 | 325.92 (vLLM gptq_gemm (exllama)) | **3.52x** | 92.56 | 325.92 | error |
| qkv | 64 | 176.93 | 588.70 (vLLM gptq_gemm (exllama)) | **3.33x** | 176.93 | 588.70 | error |
| o | 1 | 31.80 | 68.35 (vLLM gptq_gemm (exllama)) | **2.15x** | 31.80 | 68.35 | error |
| o | 2 | 31.95 | 73.59 (vLLM gptq_gemm (exllama)) | **2.30x** | 31.95 | 73.59 | error |
| o | 3 | 32.19 | 77.84 (vLLM gptq_gemm (exllama)) | **2.42x** | 32.19 | 77.84 | error |
| o | 4 | 32.31 | 81.76 (vLLM gptq_gemm (exllama)) | **2.53x** | 32.31 | 81.76 | error |
| o | 8 | 32.65 | 92.54 (vLLM gptq_gemm (exllama)) | **2.83x** | 32.65 | 92.54 | error |
| o | 16 | 34.69 | 85.92 (vLLM gptq_gemm (exllama)) | **2.48x** | 34.69 | 85.92 | error |
| o | 32 | 41.17 | 147.85 (vLLM gptq_gemm (exllama)) | **3.59x** | 41.17 | 147.85 | error |
| o | 64 | 71.94 | 148.69 (vLLM gptq_gemm (exllama)) | **2.07x** | 71.94 | 148.69 | error |
| qkvz | 1 | 79.46 | 89.36 (vLLM gptq_gemm (exllama)) | **1.12x** | 79.46 | 89.36 | error |
| qkvz | 2 | 79.88 | 94.61 (vLLM gptq_gemm (exllama)) | **1.18x** | 79.88 | 94.61 | error |
| qkvz | 3 | 80.04 | 99.28 (vLLM gptq_gemm (exllama)) | **1.24x** | 80.04 | 99.28 | error |
| qkvz | 4 | 80.51 | 105.83 (vLLM gptq_gemm (exllama)) | **1.31x** | 80.51 | 105.83 | error |
| qkvz | 8 | 81.88 | 134.56 (vLLM gptq_gemm (exllama)) | **1.64x** | 81.88 | 134.56 | error |
| qkvz | 16 | 83.57 | 193.60 (vLLM gptq_gemm (exllama)) | **2.32x** | 83.57 | 193.60 | error |
| qkvz | 32 | 102.44 | 369.45 (vLLM gptq_gemm (exllama)) | **3.61x** | 102.44 | 369.45 | error |
| qkvz | 64 | 193.75 | 678.40 (vLLM gptq_gemm (exllama)) | **3.50x** | 193.75 | 678.40 | error |
| out | 1 | 31.74 | 67.49 (vLLM gptq_gemm (exllama)) | **2.13x** | 31.74 | 67.49 | error |
| out | 2 | 31.95 | 72.50 (vLLM gptq_gemm (exllama)) | **2.27x** | 31.95 | 72.50 | error |
| out | 3 | 32.11 | 76.63 (vLLM gptq_gemm (exllama)) | **2.39x** | 32.11 | 76.63 | error |
| out | 4 | 32.21 | 80.75 (vLLM gptq_gemm (exllama)) | **2.51x** | 32.21 | 80.75 | error |
| out | 8 | 32.53 | 92.04 (vLLM gptq_gemm (exllama)) | **2.83x** | 32.53 | 92.04 | error |
| out | 16 | 34.83 | 86.13 (vLLM gptq_gemm (exllama)) | **2.47x** | 34.83 | 86.13 | error |
| out | 32 | 40.78 | 148.95 (vLLM gptq_gemm (exllama)) | **3.65x** | 40.78 | 148.95 | error |
| out | 64 | 70.90 | 148.99 (vLLM gptq_gemm (exllama)) | **2.10x** | 70.90 | 148.99 | error |
| gate_up | 1 | 164.13 | 174.98 (vLLM gptq_gemm (exllama)) | **1.07x** | 164.13 | 174.98 | error |
| gate_up | 2 | 165.09 | 180.37 (vLLM gptq_gemm (exllama)) | **1.09x** | 165.09 | 180.37 | error |
| gate_up | 3 | 165.84 | 187.63 (vLLM gptq_gemm (exllama)) | **1.13x** | 165.84 | 187.63 | error |
| gate_up | 4 | 166.72 | 197.21 (vLLM gptq_gemm (exllama)) | **1.18x** | 166.72 | 197.21 | error |
| gate_up | 8 | 169.31 | 231.26 (vLLM gptq_gemm (exllama)) | **1.37x** | 169.31 | 231.26 | error |
| gate_up | 16 | 173.21 | 395.39 (vLLM gptq_gemm (exllama)) | **2.28x** | 173.21 | 395.39 | error |
| gate_up | 32 | 212.26 | 789.30 (vLLM gptq_gemm (exllama)) | **3.72x** | 212.26 | 789.30 | error |
| gate_up | 64 | 415.97 | 1436.21 (vLLM gptq_gemm (exllama)) | **3.45x** | 415.97 | 1436.21 | error |
| down | 1 | 83.46 | 112.98 (vLLM gptq_gemm (exllama)) | **1.35x** | 83.46 | 112.98 | error |
| down | 2 | 83.62 | 123.28 (vLLM gptq_gemm (exllama)) | **1.47x** | 83.62 | 123.28 | error |
| down | 3 | 83.83 | 131.31 (vLLM gptq_gemm (exllama)) | **1.57x** | 83.83 | 131.31 | error |
| down | 4 | 84.23 | 143.29 (vLLM gptq_gemm (exllama)) | **1.70x** | 84.23 | 143.29 | error |
| down | 8 | 85.07 | 178.74 (vLLM gptq_gemm (exllama)) | **2.10x** | 85.07 | 178.74 | error |
| down | 16 | 91.18 | 215.16 (vLLM gptq_gemm (exllama)) | **2.36x** | 91.18 | 215.16 | error |
| down | 32 | 110.29 | 395.42 (vLLM gptq_gemm (exllama)) | **3.58x** | 110.29 | 395.42 | error |
| down | 64 | 203.23 | 718.73 (vLLM gptq_gemm (exllama)) | **3.54x** | 203.23 | 718.73 | error |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel

- skipped: ba: left unquantized by the checkpoint

