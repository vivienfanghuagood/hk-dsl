### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL n/a (ModuleNotFoundError: No module named 'flydsl'), AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.9.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 | vLLM rdna_hybrid_w4a16 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| qkv | 1 | 70.37 | 70.06 (vLLM rdna_hybrid_w4a16) | **1.00x** | 70.37 | 82.84 | error | 70.06 |
| qkv | 2 | 70.55 | 70.19 (vLLM rdna_hybrid_w4a16) | **0.99x** | 70.55 | 88.89 | error | 70.19 |
| qkv | 3 | 70.79 | 70.72 (vLLM rdna_hybrid_w4a16) | **1.00x** | 70.79 | 95.31 | error | 70.72 |
| qkv | 4 | 71.00 | 71.50 (vLLM rdna_hybrid_w4a16) | **1.01x** | 71.00 | 101.33 | error | 71.50 |
| qkv | 8 | 72.12 | 121.75 (vLLM gptq_gemm (exllama)) | **1.69x** | 72.12 | 121.75 | error | 258.51 |
| qkv | 16 | 74.04 | 171.81 (vLLM gptq_gemm (exllama)) | **2.32x** | 74.04 | 171.81 | error | 284.53 |
| qkv | 32 | 88.03 | 304.28 (vLLM gptq_gemm (exllama)) | **3.46x** | 88.03 | 304.28 | error | 517.89 |
| qkv | 64 | 170.61 | 294.79 (vLLM rdna_hybrid_w4a16) | **1.73x** | 170.61 | 589.14 | error | 294.79 |
| o | 1 | 31.77 | 31.28 (vLLM rdna_hybrid_w4a16) | **0.98x** | 31.77 | 67.93 | error | 31.28 |
| o | 2 | 31.84 | 33.40 (vLLM rdna_hybrid_w4a16) | **1.05x** | 31.84 | 72.85 | error | 33.40 |
| o | 3 | 31.99 | 34.72 (vLLM rdna_hybrid_w4a16) | **1.08x** | 31.99 | 77.09 | error | 34.72 |
| o | 4 | 32.25 | 35.29 (vLLM rdna_hybrid_w4a16) | **1.09x** | 32.25 | 81.07 | error | 35.29 |
| o | 8 | 32.68 | 92.45 (vLLM gptq_gemm (exllama)) | **2.83x** | 32.68 | 92.45 | error | 129.83 |
| o | 16 | 34.72 | 86.06 (vLLM gptq_gemm (exllama)) | **2.48x** | 34.72 | 86.06 | error | 143.27 |
| o | 32 | 39.92 | 144.18 (vLLM gptq_gemm (exllama)) | **3.61x** | 39.92 | 144.18 | error | 301.54 |
| o | 64 | 72.21 | 133.44 (vLLM rdna_hybrid_w4a16) | **1.85x** | 72.21 | 148.76 | error | 133.44 |
| qkvz | 1 | 79.43 | 79.40 (vLLM rdna_hybrid_w4a16) | **1.00x** | 79.43 | 89.55 | error | 79.40 |
| qkvz | 2 | 79.65 | 79.80 (vLLM rdna_hybrid_w4a16) | **1.00x** | 79.65 | 94.72 | error | 79.80 |
| qkvz | 3 | 80.22 | 80.64 (vLLM rdna_hybrid_w4a16) | **1.00x** | 80.22 | 100.32 | error | 80.64 |
| qkvz | 4 | 80.40 | 81.48 (vLLM rdna_hybrid_w4a16) | **1.01x** | 80.40 | 106.00 | error | 81.48 |
| qkvz | 8 | 82.03 | 136.21 (vLLM gptq_gemm (exllama)) | **1.66x** | 82.03 | 136.21 | error | 292.62 |
| qkvz | 16 | 83.70 | 195.61 (vLLM gptq_gemm (exllama)) | **2.34x** | 83.70 | 195.61 | error | 314.25 |
| qkvz | 32 | 103.33 | 379.30 (vLLM gptq_gemm (exllama)) | **3.67x** | 103.33 | 379.30 | error | 611.13 |
| qkvz | 64 | 199.54 | 325.00 (vLLM rdna_hybrid_w4a16) | **1.63x** | 199.54 | 673.81 | error | 325.00 |
| out | 1 | 31.66 | 31.36 (vLLM rdna_hybrid_w4a16) | **0.99x** | 31.66 | 67.30 | error | 31.36 |
| out | 2 | 31.82 | 33.55 (vLLM rdna_hybrid_w4a16) | **1.05x** | 31.82 | 72.04 | error | 33.55 |
| out | 3 | 32.03 | 34.79 (vLLM rdna_hybrid_w4a16) | **1.09x** | 32.03 | 76.39 | error | 34.79 |
| out | 4 | 32.23 | 35.33 (vLLM rdna_hybrid_w4a16) | **1.10x** | 32.23 | 80.42 | error | 35.33 |
| out | 8 | 32.83 | 92.69 (vLLM gptq_gemm (exllama)) | **2.82x** | 32.83 | 92.69 | error | 132.09 |
| out | 16 | 34.76 | 87.22 (vLLM gptq_gemm (exllama)) | **2.51x** | 34.76 | 87.22 | error | 144.02 |
| out | 32 | 40.88 | 145.65 (vLLM gptq_gemm (exllama)) | **3.56x** | 40.88 | 145.65 | error | 303.04 |
| out | 64 | 73.72 | 134.37 (vLLM rdna_hybrid_w4a16) | **1.82x** | 73.72 | 148.97 | error | 134.37 |
| gate_up | 1 | 163.97 | 164.38 (vLLM rdna_hybrid_w4a16) | **1.00x** | 163.97 | 174.61 | error | 164.38 |
| gate_up | 2 | 164.92 | 166.13 (vLLM rdna_hybrid_w4a16) | **1.01x** | 164.92 | 180.28 | error | 166.13 |
| gate_up | 3 | 165.74 | 167.05 (vLLM rdna_hybrid_w4a16) | **1.01x** | 165.74 | 186.78 | error | 167.05 |
| gate_up | 4 | 166.68 | 169.19 (vLLM rdna_hybrid_w4a16) | **1.01x** | 166.68 | 196.22 | error | 169.19 |
| gate_up | 8 | 169.38 | 231.44 (vLLM gptq_gemm (exllama)) | **1.37x** | 169.38 | 231.44 | error | 661.45 |
| gate_up | 16 | 173.07 | 399.63 (vLLM gptq_gemm (exllama)) | **2.31x** | 173.07 | 399.63 | error | 691.95 |
| gate_up | 32 | 214.71 | 787.29 (vLLM gptq_gemm (exllama)) | **3.67x** | 214.71 | 787.29 | error | 1284.58 |
| gate_up | 64 | 400.24 | 690.75 (vLLM rdna_hybrid_w4a16) | **1.73x** | 400.24 | 1440.26 | error | 690.75 |
| down | 1 | 83.30 | 83.12 (vLLM rdna_hybrid_w4a16) | **1.00x** | 83.30 | 112.83 | error | 83.12 |
| down | 2 | 83.40 | 85.26 (vLLM rdna_hybrid_w4a16) | **1.02x** | 83.40 | 121.90 | error | 85.26 |
| down | 3 | 84.19 | 138.77 (vLLM gptq_gemm (exllama)) | **1.65x** | 84.19 | 138.77 | error | 354.53 |
| down | 4 | 84.71 | 149.73 (vLLM gptq_gemm (exllama)) | **1.77x** | 84.71 | 149.73 | error | 354.70 |
| down | 8 | 85.25 | 182.97 (vLLM gptq_gemm (exllama)) | **2.15x** | 85.25 | 182.97 | error | 368.16 |
| down | 16 | 90.90 | 215.73 (vLLM gptq_gemm (exllama)) | **2.37x** | 90.90 | 215.73 | error | 392.56 |
| down | 32 | 107.61 | 387.68 (vLLM gptq_gemm (exllama)) | **3.60x** | 107.61 | 387.68 | error | 793.01 |
| down | 64 | 209.76 | 423.54 (vLLM rdna_hybrid_w4a16) | **2.02x** | 209.76 | 722.05 | error | 423.54 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
- vLLM rdna_hybrid_w4a16: torch.ops.vllm.rdna_hybrid_w4a16_apply (RDNAHybridW4A16LinearKernel: its HIP skinny GEMM at small batches, Triton above; what vLLM's GPTQ layers run on gfx1201), on pack_skinny_int4's layout, symmetric (no zero points)

- skipped: ba: left unquantized by the checkpoint
