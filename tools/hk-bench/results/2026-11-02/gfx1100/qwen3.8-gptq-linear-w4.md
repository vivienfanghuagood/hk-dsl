### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 67.41 | 61.86 (vLLM gptq_gemm_rdna3) | **0.92x** | 67.41 | 81.00 | 61.86 † |
| qkv | 2 | 67.51 | 63.35 (vLLM gptq_gemm_rdna3) | **0.94x** | 67.51 | 87.05 | 63.35 † |
| qkv | 3 | 67.51 | 70.23 (vLLM gptq_gemm_rdna3) | **1.04x** | 67.51 | 93.63 | 70.23 † |
| qkv | 4 | 67.69 | 65.47 (vLLM gptq_gemm_rdna3) | **0.97x** | 67.69 | 101.80 | 65.47 † |
| qkv | 8 | 68.10 | 86.66 (vLLM gptq_gemm_rdna3) | **1.27x** | 68.10 | 92.59 | 86.66 † |
| qkv | 16 | 69.98 | 144.92 (vLLM gptq_gemm_rdna3) | **2.07x** | 69.98 | 150.19 | 144.92 † |
| qkv | 32 | 102.89 | 256.54 (vLLM gptq_gemm_rdna3) | **2.49x** | 102.89 | 268.42 | 256.54 † |
| qkv | 64 | 183.81 | 251.18 (vLLM gptq_gemm_rdna3) | **1.37x** | 183.81 | 541.07 | 251.18 |
| o | 1 | 33.18 | 31.61 (vLLM gptq_gemm_rdna3) | **0.95x** | 33.18 | 58.12 | 31.61 † |
| o | 2 | 33.28 | 32.96 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.28 | 61.72 | 32.96 † |
| o | 3 | 33.33 | 33.03 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.33 | 64.54 | 33.03 † |
| o | 4 | 33.34 | 34.97 (vLLM gptq_gemm_rdna3) | **1.05x** | 33.34 | 66.52 | 34.97 † |
| o | 8 | 33.67 | 47.69 (vLLM gptq_gemm_rdna3) | **1.42x** | 33.67 | 70.11 | 47.69 † |
| o | 16 | 34.17 | 68.56 (vLLM gptq_gemm_rdna3) | **2.01x** | 34.17 | 81.17 | 68.56 † |
| o | 32 | 47.01 | 120.10 (vLLM gptq_gemm_rdna3) | **2.56x** | 47.01 | 128.13 | 120.10 † |
| o | 64 | 84.17 | 135.97 (vLLM gptq_gemm_rdna3) | **1.61x** | 84.17 | 159.66 | 135.97 |
| qkvz | 1 | 78.28 | 72.78 (vLLM gptq_gemm_rdna3) | **0.93x** | 78.28 | 83.15 | 72.78 † |
| qkvz | 2 | 78.41 | 79.47 (vLLM gptq_gemm_rdna3) | **1.01x** | 78.41 | 91.23 | 79.47 † |
| qkvz | 3 | 78.54 | 79.59 (vLLM gptq_gemm_rdna3) | **1.01x** | 78.54 | 98.52 | 79.59 † |
| qkvz | 4 | 78.62 | 89.90 (vLLM gptq_gemm_rdna3) | **1.14x** | 78.62 | 106.43 | 89.90 † |
| qkvz | 8 | 79.08 | 94.82 (vLLM gptq_gemm_rdna3) | **1.20x** | 79.08 | 102.99 | 94.82 † |
| qkvz | 16 | 80.23 | 162.49 (vLLM gptq_gemm_rdna3) | **2.02x** | 80.23 | 166.01 | 162.49 † |
| qkvz | 32 | 117.02 | 288.84 (vLLM gptq_gemm_rdna3) | **2.47x** | 117.02 | 306.90 | 288.84 † |
| qkvz | 64 | 220.25 | 286.20 (vLLM gptq_gemm_rdna3) | **1.30x** | 220.25 | 690.91 | 286.20 |
| out | 1 | 33.22 | 31.63 (vLLM gptq_gemm_rdna3) | **0.95x** | 33.22 | 58.18 | 31.63 † |
| out | 2 | 33.25 | 32.80 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.25 | 61.76 | 32.80 † |
| out | 3 | 33.33 | 32.96 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.33 | 64.39 | 32.96 † |
| out | 4 | 33.39 | 34.86 (vLLM gptq_gemm_rdna3) | **1.04x** | 33.39 | 66.42 | 34.86 † |
| out | 8 | 33.57 | 47.64 (vLLM gptq_gemm_rdna3) | **1.42x** | 33.57 | 70.02 | 47.64 † |
| out | 16 | 34.15 | 68.25 (vLLM gptq_gemm_rdna3) | **2.00x** | 34.15 | 81.10 | 68.25 † |
| out | 32 | 46.86 | 120.07 (vLLM gptq_gemm_rdna3) | **2.56x** | 46.86 | 128.76 | 120.07 † |
| out | 64 | 82.03 | 135.60 (vLLM gptq_gemm_rdna3) | **1.65x** | 82.03 | 159.43 | 135.60 |
| gate_up | 1 | 161.21 | 161.06 (vLLM gptq_gemm_rdna3) | **1.00x** | 161.21 | 163.38 | 161.06 † |
| gate_up | 2 | 154.53 | 144.83 (vLLM gptq_gemm_rdna3) | **0.94x** | 154.53 | 151.53 | 144.83 † |
| gate_up | 3 | 155.38 | 152.75 (vLLM gptq_gemm_rdna3) | **0.98x** | 155.38 | 160.79 | 152.75 † |
| gate_up | 4 | 154.86 | 152.03 (vLLM gptq_gemm_rdna3) | **0.98x** | 154.86 | 170.57 | 152.03 † |
| gate_up | 8 | 158.89 | 186.77 (vLLM gptq_gemm_rdna3) | **1.18x** | 158.89 | 188.69 | 186.77 † |
| gate_up | 16 | 165.41 | 323.56 (vLLM gptq_gemm_rdna3) | **1.96x** | 165.41 | 337.13 | 323.56 † |
| gate_up | 32 | 240.06 | 612.78 (vLLM gptq_gemm_rdna3) | **2.55x** | 240.06 | 647.50 | 612.78 † |
| gate_up | 64 | 465.22 | 562.84 (vLLM gptq_gemm_rdna3) | **1.21x** | 465.22 | 1481.94 | 562.84 |
| down | 1 | 84.22 | 92.81 (vLLM gptq_gemm_rdna3) | **1.10x** | 84.22 | 150.30 | 92.81 † |
| down | 2 | 84.33 | 96.44 (vLLM gptq_gemm_rdna3) | **1.14x** | 84.33 | 159.72 | 96.44 † |
| down | 3 | 84.41 | 89.97 (vLLM gptq_gemm_rdna3) | **1.07x** | 84.41 | 166.27 | 89.97 † |
| down | 4 | 84.37 | 111.43 (vLLM gptq_gemm_rdna3) | **1.32x** | 84.37 | 172.42 | 111.43 † |
| down | 8 | 84.86 | 103.57 (vLLM gptq_gemm_rdna3) | **1.22x** | 84.86 | 159.40 | 103.57 † |
| down | 16 | 87.82 | 172.38 (vLLM gptq_gemm_rdna3) | **1.96x** | 87.82 | 178.05 | 172.38 † |
| down | 32 | 127.58 | 307.47 (vLLM gptq_gemm_rdna3) | **2.41x** | 127.58 | 323.63 | 307.47 † |
| down | 64 | 240.24 | 348.20 (vLLM gptq_gemm_rdna3) | **1.45x** | 240.24 | 662.89 † | 348.20 |

† relative error above 0.02 against the reference (looser than one rounding of a 16-bit output with fp32 accumulation); still below 0.1, so timed. The JSON has each arm's error.

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
- vLLM gptq_gemm_rdna3: vllm._custom_ops.gptq_gemm_rdna3 (vLLM's RDNA3 GPTQ kernel, what its GPTQ layers run on gfx11)

- skipped: ba: left unquantized by the checkpoint

