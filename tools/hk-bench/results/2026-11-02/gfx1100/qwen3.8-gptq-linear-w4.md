### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 77.48 | 61.74 (vLLM gptq_gemm_rdna3) | **0.80x** | 77.48 | 80.64 | 61.74 † |
| qkv | 2 | 77.41 | 63.13 (vLLM gptq_gemm_rdna3) | **0.82x** | 77.41 | 86.84 | 63.13 † |
| qkv | 4 | 77.51 | 65.11 (vLLM gptq_gemm_rdna3) | **0.84x** | 77.51 | 101.17 | 65.11 † |
| qkv | 8 | 78.25 | 86.28 (vLLM gptq_gemm_rdna3) | **1.10x** | 78.25 | 92.06 | 86.28 † |
| qkv | 16 | 89.08 | 141.29 (vLLM gptq_gemm_rdna3) | **1.59x** | 89.08 | 148.16 | 141.29 † |
| qkv | 32 | 121.10 | 256.49 (vLLM gptq_gemm_rdna3) | **2.12x** | 121.10 | 268.93 | 256.49 † |
| qkv | 64 | 226.34 | 250.14 (vLLM gptq_gemm_rdna3) | **1.10x** | 226.34 | 542.67 | 250.14 |
| o | 1 | 36.88 | 31.48 (vLLM gptq_gemm_rdna3) | **0.85x** | 36.88 | 58.10 | 31.48 † |
| o | 2 | 36.92 | 32.80 (vLLM gptq_gemm_rdna3) | **0.89x** | 36.92 | 61.78 | 32.80 † |
| o | 4 | 37.05 | 34.79 (vLLM gptq_gemm_rdna3) | **0.94x** | 37.05 | 66.48 | 34.79 † |
| o | 8 | 37.46 | 47.50 (vLLM gptq_gemm_rdna3) | **1.27x** | 37.46 | 69.50 | 47.50 † |
| o | 16 | 41.36 | 68.26 (vLLM gptq_gemm_rdna3) | **1.65x** | 41.36 | 80.64 | 68.26 † |
| o | 32 | 55.11 | 119.81 (vLLM gptq_gemm_rdna3) | **2.17x** | 55.11 | 128.47 | 119.81 † |
| o | 64 | 94.99 | 135.59 (vLLM gptq_gemm_rdna3) | **1.43x** | 94.99 | 159.94 | 135.59 |
| qkvz | 1 | 87.67 | 72.55 (vLLM gptq_gemm_rdna3) | **0.83x** | 87.67 | 83.13 | 72.55 † |
| qkvz | 2 | 87.69 | 79.23 (vLLM gptq_gemm_rdna3) | **0.90x** | 87.69 | 90.86 | 79.23 † |
| qkvz | 4 | 87.82 | 89.58 (vLLM gptq_gemm_rdna3) | **1.02x** | 87.82 | 106.17 | 89.58 † |
| qkvz | 8 | 88.94 | 95.42 (vLLM gptq_gemm_rdna3) | **1.07x** | 88.94 | 102.78 | 95.42 † |
| qkvz | 16 | 100.70 | 159.65 (vLLM gptq_gemm_rdna3) | **1.58x** | 100.70 | 165.06 | 159.65 † |
| qkvz | 32 | 136.65 | 285.12 (vLLM gptq_gemm_rdna3) | **2.09x** | 136.65 | 302.22 | 285.12 † |
| qkvz | 64 | 258.78 | 286.31 (vLLM gptq_gemm_rdna3) | **1.11x** | 258.78 | 689.76 | 286.31 |
| ba | 1 | 7.88 | 15.79 (vLLM gptq_gemm_rdna3) | **2.00x** | 7.88 | 20.55 | 15.79 † |
| ba | 2 | 7.88 | 18.80 (vLLM gptq_gemm_rdna3) | **2.39x** | 7.88 | 23.09 | 18.80 |
| ba | 4 | 7.89 | 23.60 (vLLM gptq_gemm_rdna3) | **2.99x** | 7.89 | 27.35 | 23.60 † |
| ba | 8 | 7.91 | 34.39 (vLLM gptq_gemm_rdna3) | **4.34x** | 7.91 | 35.32 | 34.39 † |
| ba | 16 | 7.97 | 34.48 (vLLM gptq_gemm_rdna3) | **4.33x** | 7.97 | 36.25 | 34.48 † |
| ba | 32 | 9.60 | 35.01 (vLLM gptq_gemm_rdna3) | **3.65x** | 9.60 | 38.72 | 35.01 † |
| ba | 64 | 15.78 | 45.09 (vLLM gptq_gemm (exllama)) | **2.86x** | 15.78 | 45.09 | 62.53 |
| out | 1 | 36.88 | 31.58 (vLLM gptq_gemm_rdna3) | **0.86x** | 36.88 | 58.09 | 31.58 † |
| out | 2 | 36.94 | 32.78 (vLLM gptq_gemm_rdna3) | **0.89x** | 36.94 | 61.63 | 32.78 † |
| out | 4 | 37.09 | 34.82 (vLLM gptq_gemm_rdna3) | **0.94x** | 37.09 | 66.40 | 34.82 † |
| out | 8 | 37.55 | 47.53 (vLLM gptq_gemm_rdna3) | **1.27x** | 37.55 | 70.17 | 47.53 † |
| out | 16 | 41.57 | 68.28 (vLLM gptq_gemm_rdna3) | **1.64x** | 41.57 | 80.74 | 68.28 † |
| out | 32 | 53.95 | 119.92 (vLLM gptq_gemm_rdna3) | **2.22x** | 53.95 | 128.12 | 119.92 † |
| out | 64 | 94.60 | 135.68 (vLLM gptq_gemm_rdna3) | **1.43x** | 94.60 | 159.49 | 135.68 |
| gate_up | 1 | 181.70 | 161.05 (vLLM gptq_gemm_rdna3) | **0.89x** | 181.70 | 163.18 | 161.05 † |
| gate_up | 2 | 183.75 | 161.07 (vLLM gptq_gemm_rdna3) | **0.88x** | 183.75 | 170.70 | 161.07 † |
| gate_up | 4 | 173.26 | 152.31 (vLLM gptq_gemm_rdna3) | **0.88x** | 173.26 | 170.26 | 152.31 † |
| gate_up | 8 | 181.00 | 186.88 (vLLM gptq_gemm_rdna3) | **1.03x** | 181.00 | 189.63 | 186.88 † |
| gate_up | 16 | 205.84 | 311.85 (vLLM gptq_gemm_rdna3) | **1.51x** | 205.84 | 329.79 | 311.85 † |
| gate_up | 32 | 283.58 | 600.53 (vLLM gptq_gemm_rdna3) | **2.12x** | 283.58 | 655.36 | 600.53 † |
| gate_up | 64 | 544.54 | 560.87 (vLLM gptq_gemm_rdna3) | **1.03x** | 544.54 | 1472.94 | 560.87 |
| down | 1 | 93.03 | 92.80 (vLLM gptq_gemm_rdna3) | **1.00x** | 93.03 | 150.97 | 92.80 † |
| down | 2 | 93.83 | 96.42 (vLLM gptq_gemm_rdna3) | **1.03x** | 93.83 | 159.71 | 96.42 † |
| down | 4 | 95.90 | 110.89 (vLLM gptq_gemm_rdna3) | **1.16x** | 95.90 | 172.49 | 110.89 † |
| down | 8 | 104.99 | 103.19 (vLLM gptq_gemm_rdna3) | **0.98x** | 104.99 | 159.98 | 103.19 † |
| down | 16 | 130.77 | 170.23 (vLLM gptq_gemm_rdna3) | **1.30x** | 130.77 | 176.54 | 170.23 † |
| down | 32 | 174.66 | 305.03 (vLLM gptq_gemm_rdna3) | **1.75x** | 174.66 | 322.29 | 305.03 † |
| down | 64 | 294.26 | 346.74 (vLLM gptq_gemm_rdna3) | **1.18x** | 294.26 | 663.16 † | 346.74 |

† relative error above 0.02 against the reference (looser than one rounding of a 16-bit output with fp32 accumulation); still below 0.1, so timed. The JSON has each arm's error.

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul (GPTQ repacked)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
- vLLM gptq_gemm_rdna3: vllm._custom_ops.gptq_gemm_rdna3 (vLLM's RDNA3 GPTQ kernel, what its GPTQ layers run on gfx11)
