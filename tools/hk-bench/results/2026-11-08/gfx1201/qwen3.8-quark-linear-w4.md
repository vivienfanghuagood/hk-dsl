### linear-w4 at amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL n/a (ModuleNotFoundError: No module named 'flydsl'), AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.9.14. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk exllama (in place) | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 | vLLM rdna_hybrid_w4a16 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 63.63 | 61.65 (vLLM rdna_hybrid_w4a16) | **0.97x** | 63.63 | 63.95 | 75.93 | error | 61.65 |
| qkv | 2 | 64.22 | 62.64 (vLLM rdna_hybrid_w4a16) | **0.97x** | 64.25 | 64.22 | 82.42 | error | 62.64 |
| qkv | 3 | 64.50 | 63.29 (vLLM rdna_hybrid_w4a16) | **0.98x** | 65.03 | 64.50 | 89.25 | error | 63.29 |
| qkv | 4 | 64.57 | 63.97 (vLLM rdna_hybrid_w4a16) | **0.99x** | 66.25 | 64.57 | 95.43 | error | 63.97 |
| qkv | 5 | 64.74 | 65.20 (vLLM rdna_hybrid_w4a16) | **1.01x** | 65.70 | 64.74 | 102.02 | error | 65.20 |
| qkv | 6 | 65.03 | 108.53 (vLLM gptq_gemm (exllama)) | **1.67x** | 65.88 | 65.03 | 108.53 | error | 136.44 |
| qkv | 8 | 65.97 | 102.09 (vLLM gptq_gemm (exllama)) | **1.55x** | 66.94 | 65.97 | 102.09 | error | 147.92 |
| qkv | 16 | 67.09 | 149.70 (vLLM rdna_hybrid_w4a16) | **2.23x** | 68.21 | 67.09 | 159.52 | error | 149.70 |
| qkv | 32 | 86.74 | 284.04 (vLLM rdna_hybrid_w4a16) | **3.27x** | 87.88 | 86.74 | 296.42 | error | 284.04 |
| qkv | 64 | 148.86 | 173.41 (vLLM rdna_hybrid_w4a16) | **1.17x** | 148.86 | 170.28 | 581.77 | error | 173.41 |
| qkv | 128 | 228.10 | 261.85 (vLLM rdna_hybrid_w4a16) | **1.15x** | 228.10 | -- | 599.46 | error | 261.85 |
| qkv | 256 | 377.97 | 507.23 (vLLM rdna_hybrid_w4a16) | **1.34x** | 377.97 | -- | 661.87 | error | 507.23 |
| qkv | 512 | 761.48 | 980.31 (vLLM rdna_hybrid_w4a16) | **1.29x** | 761.48 | -- | 991.18 | error | 980.31 |
| qkv | 1024 | 1492.93 | 1722.23 (vLLM gptq_gemm (exllama)) | **1.15x** | 1492.93 | -- | 1722.23 | error | 1727.23 |
| qkv | 2048 | 2932.84 | 3075.26 (vLLM gptq_gemm (exllama)) | **1.05x** | 2932.84 | -- | 3075.26 | error | 3372.51 |
| qkv | 4096 | 5715.16 | 5469.73 (vLLM gptq_gemm (exllama)) | **0.96x** | 5715.16 | -- | 5469.73 | error | 6630.78 |
| o | 1 | 29.26 | 28.42 (vLLM rdna_hybrid_w4a16) | **0.97x** | 30.22 | 29.26 | 65.64 | error | 28.42 |
| o | 2 | 29.35 | 30.54 (vLLM rdna_hybrid_w4a16) | **1.04x** | 30.82 | 29.35 | 70.78 | error | 30.54 |
| o | 3 | 29.49 | 31.96 (vLLM rdna_hybrid_w4a16) | **1.08x** | 31.30 | 29.49 | 75.47 | error | 31.96 |
| o | 4 | 29.56 | 32.92 (vLLM rdna_hybrid_w4a16) | **1.11x** | 32.93 | 29.56 | 79.31 | error | 32.92 |
| o | 5 | 29.64 | 34.62 (vLLM rdna_hybrid_w4a16) | **1.17x** | 34.01 | 29.64 | 82.98 | error | 34.62 |
| o | 6 | 29.73 | 63.80 (vLLM rdna_hybrid_w4a16) | **2.15x** | 34.08 | 29.73 | 86.37 | error | 63.80 |
| o | 8 | 30.02 | 65.87 (vLLM rdna_hybrid_w4a16) | **2.19x** | 33.99 | 30.02 | 87.91 | error | 65.87 |
| o | 16 | 30.88 | 72.24 (vLLM rdna_hybrid_w4a16) | **2.34x** | 34.73 | 30.88 | 81.52 | error | 72.24 |
| o | 32 | 40.89 | 135.05 (vLLM rdna_hybrid_w4a16) | **3.30x** | 42.59 | 40.89 | 147.40 | error | 135.05 |
| o | 64 | 71.34 | 84.11 (vLLM rdna_hybrid_w4a16) | **1.18x** | 71.34 | 71.56 | 141.31 | error | 84.11 |
| o | 128 | 119.21 | 144.29 (vLLM rdna_hybrid_w4a16) | **1.21x** | 119.21 | -- | 176.09 | error | 144.29 |
| o | 256 | 199.20 | 245.26 (vLLM rdna_hybrid_w4a16) | **1.23x** | 199.20 | -- | 254.90 | error | 245.26 |
| o | 512 | 352.79 | 450.33 (vLLM rdna_hybrid_w4a16) | **1.28x** | 352.79 | -- | 450.64 | error | 450.33 |
| o | 1024 | 681.84 | 736.13 (vLLM gptq_gemm (exllama)) | **1.08x** | 681.84 | -- | 736.13 | error | 812.01 |
| o | 2048 | 1319.57 | 1265.92 (vLLM gptq_gemm (exllama)) | **0.96x** | 1319.57 | -- | 1265.92 | error | 1575.03 |
| o | 4096 | 2634.81 | 2277.11 (vLLM gptq_gemm (exllama)) | **0.86x** | 2634.81 | -- | 2277.11 | error | 3071.05 |
| qkvz | 1 | 72.03 | 69.76 (vLLM rdna_hybrid_w4a16) | **0.97x** | 72.03 | 72.42 | 82.11 | error | 69.76 |
| qkvz | 2 | 73.04 | 71.16 (vLLM rdna_hybrid_w4a16) | **0.97x** | 73.11 | 73.04 | 88.86 | error | 71.16 |
| qkvz | 3 | 73.36 | 72.11 (vLLM rdna_hybrid_w4a16) | **0.98x** | 74.06 | 73.36 | 94.45 | error | 72.11 |
| qkvz | 4 | 73.72 | 73.62 (vLLM rdna_hybrid_w4a16) | **1.00x** | 75.81 | 73.72 | 100.72 | error | 73.62 |
| qkvz | 5 | 73.91 | 76.03 (vLLM rdna_hybrid_w4a16) | **1.03x** | 74.75 | 73.91 | 106.94 | error | 76.03 |
| qkvz | 6 | 74.42 | 118.28 (vLLM gptq_gemm (exllama)) | **1.59x** | 75.34 | 74.42 | 118.28 | error | 164.69 |
| qkvz | 8 | 74.98 | 117.61 (vLLM gptq_gemm (exllama)) | **1.57x** | 75.87 | 74.98 | 117.61 | error | 168.49 |
| qkvz | 16 | 76.80 | 180.12 (vLLM rdna_hybrid_w4a16) | **2.35x** | 77.63 | 76.80 | 195.37 | error | 180.12 |
| qkvz | 32 | 102.49 | 346.36 (vLLM rdna_hybrid_w4a16) | **3.38x** | 102.49 | 102.76 | 379.98 | error | 346.36 |
| qkvz | 64 | 171.94 | 200.91 (vLLM rdna_hybrid_w4a16) | **1.17x** | 171.94 | 200.00 | 664.86 | error | 200.91 |
| qkvz | 128 | 265.63 | 305.45 (vLLM rdna_hybrid_w4a16) | **1.15x** | 265.63 | -- | 681.83 | error | 305.45 |
| qkvz | 256 | 455.65 | 590.24 (vLLM rdna_hybrid_w4a16) | **1.29x** | 455.65 | -- | 742.99 | error | 590.24 |
| qkvz | 512 | 894.89 | 1153.05 (vLLM gptq_gemm (exllama)) | **1.29x** | 894.89 | -- | 1153.05 | error | 1170.87 |
| qkvz | 1024 | 1736.47 | 2008.37 (vLLM gptq_gemm (exllama)) | **1.16x** | 1736.47 | -- | 2008.37 | error | 2079.06 |
| qkvz | 2048 | 3161.23 | 3458.10 (vLLM gptq_gemm (exllama)) | **1.09x** | 3161.23 | -- | 3458.10 | error | 3906.82 |
| qkvz | 4096 | 6113.02 | 6215.92 (vLLM gptq_gemm (exllama)) | **1.02x** | 6113.02 | -- | 6215.92 | error | 7564.69 |
| ba | 1 | 4.47 | 4.19 (vLLM rdna_hybrid_w4a16) | **0.94x** | 4.49 | 4.47 | 16.98 | error | 4.19 |
| ba | 2 | 4.47 | 5.06 (vLLM rdna_hybrid_w4a16) | **1.13x** | 4.86 | 4.47 | 18.64 | error | 5.06 |
| ba | 3 | 4.47 | 5.08 (vLLM rdna_hybrid_w4a16) | **1.14x** | 5.17 | 4.47 | 20.09 | error | 5.08 |
| ba | 4 | 4.48 | 5.52 (vLLM rdna_hybrid_w4a16) | **1.23x** | 5.59 | 4.48 | 21.41 | error | 5.52 |
| ba | 5 | 4.48 | 6.46 (vLLM rdna_hybrid_w4a16) | **1.44x** | 5.21 | 4.48 | 22.68 | error | 6.46 |
| ba | 6 | 4.46 | 19.35 (vLLM rdna_hybrid_w4a16) | **4.34x** | 5.17 | 4.46 | 23.91 | error | 19.35 |
| ba | 8 | 4.46 | 19.32 (vLLM rdna_hybrid_w4a16) | **4.33x** | 5.17 | 4.46 | 26.30 | error | 19.32 |
| ba | 16 | 4.49 | 19.28 (vLLM rdna_hybrid_w4a16) | **4.30x** | 5.17 | 4.49 | 26.64 | error | 19.28 |
| ba | 32 | 6.58 | 19.75 (vLLM rdna_hybrid_w4a16) | **3.00x** | 6.91 | 6.58 | 27.36 | error | 19.75 |
| ba | 64 | 10.34 | 25.60 (vLLM rdna_hybrid_w4a16) | **2.48x** | 11.93 | 10.34 | 34.50 | error | 25.60 |
| ba | 128 | -- | 34.38 (vLLM gptq_gemm (exllama)) | -- | -- | -- | 34.38 | error | 43.21 |
| ba | 256 | -- | 33.14 (vLLM gptq_gemm (exllama)) | -- | -- | -- | 33.14 | error | 49.55 |
| ba | 512 | -- | 32.67 (vLLM gptq_gemm (exllama)) | -- | -- | -- | 32.67 | error | 50.71 |
| ba | 1024 | -- | 35.56 (vLLM gptq_gemm (exllama)) | -- | -- | -- | 35.56 | error | 49.77 |
| ba | 2048 | -- | 48.94 (vLLM gptq_gemm (exllama)) | -- | -- | -- | 48.94 | error | 53.25 |
| ba | 4096 | -- | 78.01 (vLLM rdna_hybrid_w4a16) | -- | -- | -- | 86.38 | error | 78.01 |
| out | 1 | 29.27 | 28.42 (vLLM rdna_hybrid_w4a16) | **0.97x** | 30.28 | 29.27 | 65.74 | error | 28.42 |
| out | 2 | 29.34 | 30.53 (vLLM rdna_hybrid_w4a16) | **1.04x** | 30.75 | 29.34 | 70.39 | error | 30.53 |
| out | 3 | 29.43 | 31.91 (vLLM rdna_hybrid_w4a16) | **1.08x** | 31.23 | 29.43 | 74.61 | error | 31.91 |
| out | 4 | 29.59 | 33.02 (vLLM rdna_hybrid_w4a16) | **1.12x** | 32.67 | 29.59 | 78.65 | error | 33.02 |
| out | 5 | 29.62 | 34.52 (vLLM rdna_hybrid_w4a16) | **1.17x** | 33.95 | 29.62 | 81.86 | error | 34.52 |
| out | 6 | 29.93 | 66.29 (vLLM rdna_hybrid_w4a16) | **2.21x** | 33.96 | 29.93 | 85.81 | error | 66.29 |
| out | 8 | 30.25 | 66.97 (vLLM rdna_hybrid_w4a16) | **2.21x** | 34.05 | 30.25 | 83.82 | error | 66.97 |
| out | 16 | 31.38 | 73.25 (vLLM rdna_hybrid_w4a16) | **2.33x** | 35.54 | 31.38 | 87.91 | error | 73.25 |
| out | 32 | 41.37 | 135.89 (vLLM rdna_hybrid_w4a16) | **3.28x** | 42.89 | 41.37 | 149.08 | error | 135.89 |
| out | 64 | 73.69 | 82.99 (vLLM rdna_hybrid_w4a16) | **1.13x** | 73.69 | 76.09 | 143.20 | error | 82.99 |
| out | 128 | 121.31 | 142.02 (vLLM rdna_hybrid_w4a16) | **1.17x** | 121.31 | -- | 178.37 | error | 142.02 |
| out | 256 | 197.70 | 240.90 (vLLM rdna_hybrid_w4a16) | **1.22x** | 197.70 | -- | 261.56 | error | 240.90 |
| out | 512 | 349.07 | 451.63 (vLLM rdna_hybrid_w4a16) | **1.29x** | 349.07 | -- | 455.83 | error | 451.63 |
| out | 1024 | 656.86 | 744.08 (vLLM gptq_gemm (exllama)) | **1.13x** | 656.86 | -- | 744.08 | error | 811.18 |
| out | 2048 | 1189.68 | 1189.85 (vLLM gptq_gemm (exllama)) | **1.00x** | 1189.68 | -- | 1189.85 | error | 1458.00 |
| out | 4096 | 2329.22 | 2278.28 (vLLM gptq_gemm (exllama)) | **0.98x** | 2329.22 | -- | 2278.28 | error | 2828.60 |
| gate_up | 1 | 149.39 | 144.76 (vLLM rdna_hybrid_w4a16) | **0.97x** | 149.39 | 149.81 | 158.11 | error | 144.76 |
| gate_up | 2 | 150.89 | 147.01 (vLLM rdna_hybrid_w4a16) | **0.97x** | 150.89 | 150.99 | 164.96 | error | 147.01 |
| gate_up | 3 | 151.83 | 148.68 (vLLM rdna_hybrid_w4a16) | **0.98x** | 152.49 | 151.83 | 171.53 | error | 148.68 |
| gate_up | 4 | 152.59 | 151.30 (vLLM rdna_hybrid_w4a16) | **0.99x** | 154.98 | 152.59 | 183.04 | error | 151.30 |
| gate_up | 5 | 153.17 | 165.68 (vLLM rdna_hybrid_w4a16) | **1.08x** | 154.27 | 153.17 | 191.68 | error | 165.68 |
| gate_up | 6 | 154.03 | 205.87 (vLLM gptq_gemm (exllama)) | **1.34x** | 154.88 | 154.03 | 205.87 | error | 347.75 |
| gate_up | 8 | 155.31 | 215.76 (vLLM gptq_gemm (exllama)) | **1.39x** | 155.95 | 155.31 | 215.76 | error | 359.67 |
| gate_up | 16 | 158.64 | 376.02 (vLLM rdna_hybrid_w4a16) | **2.37x** | 159.76 | 158.64 | 401.87 | error | 376.02 |
| gate_up | 32 | 214.57 | 741.55 (vLLM rdna_hybrid_w4a16) | **3.46x** | 214.57 | 215.47 | 788.51 | error | 741.55 |
| gate_up | 64 | 373.33 | 393.40 (vLLM rdna_hybrid_w4a16) | **1.05x** | 373.33 | 454.42 | 1423.15 | error | 393.40 |
| gate_up | 128 | 454.12 | 592.57 (vLLM rdna_hybrid_w4a16) | **1.30x** | 454.12 | -- | 1460.57 | error | 592.57 |
| gate_up | 256 | 873.04 | 996.76 (vLLM rdna_hybrid_w4a16) | **1.14x** | 873.04 | -- | 1637.22 | error | 996.76 |
| gate_up | 512 | 1674.82 | 2012.21 (vLLM rdna_hybrid_w4a16) | **1.20x** | 1674.82 | -- | 2440.99 | error | 2012.21 |
| gate_up | 1024 | 3378.14 | 3853.56 (vLLM rdna_hybrid_w4a16) | **1.14x** | 3378.14 | -- | 3965.25 | error | 3853.56 |
| gate_up | 2048 | 6728.56 | 6992.59 (vLLM gptq_gemm (exllama)) | **1.04x** | 6728.56 | -- | 6992.59 | error | 7705.17 |
| gate_up | 4096 | 13422.88 | 13316.10 (vLLM gptq_gemm (exllama)) | **0.99x** | 13422.88 | -- | 13316.10 | error | 15581.25 |
| down | 1 | 76.39 | 75.68 (vLLM rdna_hybrid_w4a16) | **0.99x** | 76.39 | 76.41 | 110.56 | error | 75.68 |
| down | 2 | 76.80 | 77.54 (vLLM rdna_hybrid_w4a16) | **1.01x** | 77.11 | 76.80 | 122.19 | error | 77.54 |
| down | 3 | 77.36 | 141.11 (vLLM gptq_gemm (exllama)) | **1.82x** | 77.80 | 77.36 | 141.11 | error | 177.62 |
| down | 4 | 78.22 | 153.60 (vLLM gptq_gemm (exllama)) | **1.96x** | 81.58 | 78.22 | 153.60 | error | 180.78 |
| down | 5 | 78.64 | 162.17 (vLLM gptq_gemm (exllama)) | **2.06x** | 79.04 | 78.64 | 162.17 | error | 181.12 |
| down | 6 | 79.20 | 171.87 (vLLM gptq_gemm (exllama)) | **2.17x** | 79.20 | 79.22 | 171.87 | error | 183.35 |
| down | 8 | 79.55 | 175.19 (vLLM gptq_gemm (exllama)) | **2.20x** | 79.56 | 79.55 | 175.19 | error | 187.35 |
| down | 16 | 79.21 | 202.37 (vLLM rdna_hybrid_w4a16) | **2.56x** | 81.16 | 79.21 | 220.71 | error | 202.37 |
| down | 32 | 107.28 | 384.28 (vLLM rdna_hybrid_w4a16) | **3.58x** | 107.28 | 109.45 | 398.36 | error | 384.28 |
| down | 64 | 183.41 | 219.63 (vLLM rdna_hybrid_w4a16) | **1.20x** | 183.41 | 212.46 | 712.60 | error | 219.63 |
| down | 128 | 341.78 | 496.49 (vLLM rdna_hybrid_w4a16) | **1.45x** | 341.78 | -- | 723.94 | error | 496.49 |
| down | 256 | 553.87 | 686.56 (vLLM rdna_hybrid_w4a16) | **1.24x** | 553.87 | -- | 859.96 | error | 686.56 |
| down | 512 | 1012.31 | 1144.29 (vLLM rdna_hybrid_w4a16) | **1.13x** | 1012.31 | -- | 1382.22 | error | 1144.29 |
| down | 1024 | 1737.16 | 2128.93 (vLLM rdna_hybrid_w4a16) | **1.23x** | 1737.16 | -- | 2149.94 | error | 2128.93 |
| down | 2048 | 3324.11 | 3341.14 (vLLM gptq_gemm (exllama)) | **1.00x** | 3324.11 | -- | 3341.14 | error | 4335.28 |
| down | 4096 | 6593.18 | 6499.72 (vLLM gptq_gemm (exllama)) | **0.99x** | 6593.18 | -- | 6499.72 | error | 14112.26 |

**Setup of each arm.**
- hk exllama (in place): hk.ops.w4a16.exllama_matmul (<= 64 tokens) / exllama_large_matmul on RDNAHybridW4A16LinearKernel's own tensors, as vllm_gptq / vllm_quark run them on gfx1201
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
- vLLM rdna_hybrid_w4a16: torch.ops.vllm.rdna_hybrid_w4a16_apply (RDNAHybridW4A16LinearKernel: its HIP skinny GEMM at small batches, Triton above; what vLLM's GPTQ layers run on gfx1201), on pack_skinny_int4's layout, symmetric (no zero points)
