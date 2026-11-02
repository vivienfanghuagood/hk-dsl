### linear-w4 at palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM gptq_gemm (exllama) | vLLM gptq_gemm_rdna3 |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 67.33 | 61.89 (vLLM gptq_gemm_rdna3) | **0.92x** | 67.33 | 80.75 | 61.89 † |
| qkv | 2 | 67.42 | 63.29 (vLLM gptq_gemm_rdna3) | **0.94x** | 67.42 | 87.41 | 63.29 † |
| qkv | 3 | 67.45 | 70.19 (vLLM gptq_gemm_rdna3) | **1.04x** | 67.45 | 94.12 | 70.19 † |
| qkv | 4 | 67.56 | 65.23 (vLLM gptq_gemm_rdna3) | **0.96x** | 67.56 | 101.51 | 65.23 † |
| qkv | 8 | 68.44 | 87.20 (vLLM gptq_gemm_rdna3) | **1.27x** | 68.44 | 92.65 | 87.20 † |
| qkv | 16 | 71.65 | 145.66 (vLLM gptq_gemm_rdna3) | **2.03x** | 71.65 | 153.04 | 145.66 † |
| qkv | 32 | 103.24 | 258.51 (vLLM gptq_gemm_rdna3) | **2.50x** | 103.24 | 269.01 | 258.51 † |
| qkv | 64 | 184.41 | 252.48 (vLLM gptq_gemm_rdna3) | **1.37x** | 184.41 | 541.18 | 252.48 |
| o | 1 | 33.18 | 31.65 (vLLM gptq_gemm_rdna3) | **0.95x** | 33.18 | 58.15 | 31.65 † |
| o | 2 | 33.23 | 32.94 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.23 | 61.79 | 32.94 † |
| o | 3 | 33.24 | 33.01 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.24 | 64.34 | 33.01 † |
| o | 4 | 33.36 | 34.97 (vLLM gptq_gemm_rdna3) | **1.05x** | 33.36 | 66.47 | 34.97 † |
| o | 8 | 33.59 | 47.63 (vLLM gptq_gemm_rdna3) | **1.42x** | 33.59 | 70.21 | 47.63 † |
| o | 16 | 34.09 | 68.26 (vLLM gptq_gemm_rdna3) | **2.00x** | 34.09 | 80.57 | 68.26 † |
| o | 32 | 46.97 | 119.69 (vLLM gptq_gemm_rdna3) | **2.55x** | 46.97 | 127.85 | 119.69 † |
| o | 64 | 84.19 | 136.02 (vLLM gptq_gemm_rdna3) | **1.62x** | 84.19 | 159.91 | 136.02 |
| qkvz | 1 | 78.17 | 72.41 (vLLM gptq_gemm_rdna3) | **0.93x** | 78.17 | 83.21 | 72.41 † |
| qkvz | 2 | 78.21 | 79.12 (vLLM gptq_gemm_rdna3) | **1.01x** | 78.21 | 90.89 | 79.12 † |
| qkvz | 3 | 78.43 | 79.28 (vLLM gptq_gemm_rdna3) | **1.01x** | 78.43 | 98.13 | 79.28 † |
| qkvz | 4 | 78.52 | 89.82 (vLLM gptq_gemm_rdna3) | **1.14x** | 78.52 | 106.63 | 89.82 † |
| qkvz | 8 | 79.66 | 95.28 (vLLM gptq_gemm_rdna3) | **1.20x** | 79.66 | 102.51 | 95.28 † |
| qkvz | 16 | 82.85 | 166.81 (vLLM gptq_gemm_rdna3) | **2.01x** | 82.85 | 171.34 | 166.81 † |
| qkvz | 32 | 121.35 | 297.72 (vLLM gptq_gemm_rdna3) | **2.45x** | 121.35 | 314.57 | 297.72 † |
| qkvz | 64 | 216.54 | 285.48 (vLLM gptq_gemm_rdna3) | **1.32x** | 216.54 | 692.53 | 285.48 |
| out | 1 | 33.20 | 31.54 (vLLM gptq_gemm_rdna3) | **0.95x** | 33.20 | 57.91 | 31.54 † |
| out | 2 | 33.27 | 32.68 (vLLM gptq_gemm_rdna3) | **0.98x** | 33.27 | 61.69 | 32.68 † |
| out | 3 | 33.34 | 32.86 (vLLM gptq_gemm_rdna3) | **0.99x** | 33.34 | 64.36 | 32.86 † |
| out | 4 | 33.43 | 34.76 (vLLM gptq_gemm_rdna3) | **1.04x** | 33.43 | 66.43 | 34.76 † |
| out | 8 | 33.62 | 47.45 (vLLM gptq_gemm_rdna3) | **1.41x** | 33.62 | 69.84 | 47.45 † |
| out | 16 | 34.19 | 68.02 (vLLM gptq_gemm_rdna3) | **1.99x** | 34.19 | 80.42 | 68.02 † |
| out | 32 | 46.91 | 119.84 (vLLM gptq_gemm_rdna3) | **2.55x** | 46.91 | 128.22 | 119.84 † |
| out | 64 | 82.20 | 136.29 (vLLM gptq_gemm_rdna3) | **1.66x** | 82.20 | 159.78 | 136.29 |
| gate_up | 1 | 153.64 | 140.68 (vLLM gptq_gemm_rdna3) | **0.92x** | 153.64 | 143.55 | 140.68 † |
| gate_up | 2 | 154.04 | 144.24 (vLLM gptq_gemm_rdna3) | **0.94x** | 154.04 | 150.94 | 144.24 † |
| gate_up | 3 | 162.46 | 153.86 (vLLM gptq_gemm_rdna3) | **0.95x** | 162.46 | 161.06 | 153.86 † |
| gate_up | 4 | 162.62 | 152.50 (vLLM gptq_gemm_rdna3) | **0.94x** | 162.62 | 170.88 | 152.50 † |
| gate_up | 8 | 175.15 | 200.13 (vLLM gptq_gemm_rdna3) | **1.14x** | 175.15 | 202.51 | 200.13 † |
| gate_up | 16 | 185.93 | 368.00 (vLLM gptq_gemm_rdna3) | **1.98x** | 185.93 | 383.98 | 368.00 † |
| gate_up | 32 | 264.61 | 661.84 (vLLM gptq_gemm_rdna3) | **2.50x** | 264.61 | 710.45 | 661.84 † |
| gate_up | 64 | 456.51 | 566.09 (vLLM gptq_gemm_rdna3) | **1.24x** | 456.51 | 1478.23 | 566.09 |
| down | 1 | 83.95 | 92.51 (vLLM gptq_gemm_rdna3) | **1.10x** | 83.95 | 149.41 | 92.51 † |
| down | 2 | 84.07 | 96.43 (vLLM gptq_gemm_rdna3) | **1.15x** | 84.07 | 159.44 | 96.43 † |
| down | 3 | 84.07 | 89.85 (vLLM gptq_gemm_rdna3) | **1.07x** | 84.07 | 166.18 | 89.85 † |
| down | 4 | 84.29 | 110.33 (vLLM gptq_gemm_rdna3) | **1.31x** | 84.29 | 172.56 | 110.33 † |
| down | 8 | 84.65 | 103.31 (vLLM gptq_gemm_rdna3) | **1.22x** | 84.65 | 159.29 | 103.31 † |
| down | 16 | 89.10 | 176.98 (vLLM gptq_gemm_rdna3) | **1.99x** | 89.10 | 182.55 | 176.98 † |
| down | 32 | 131.51 | 320.60 (vLLM gptq_gemm_rdna3) | **2.44x** | 131.51 | 335.71 | 320.60 † |
| down | 64 | 237.49 | 348.35 (vLLM gptq_gemm_rdna3) | **1.47x** | 237.49 | 663.10 † | 348.35 |

† relative error above 0.02 against the reference (looser than one rounding of a 16-bit output with fp32 accumulation); still below 0.1, so timed. The JSON has each arm's error.

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, symmetric form (repack_gptq(symmetric=True), as vllm_gptq loads it)
- vLLM gptq_gemm (exllama): vllm._custom_ops.gptq_gemm, exllama kernel
- vLLM gptq_gemm_rdna3: vllm._custom_ops.gptq_gemm_rdna3 (vLLM's RDNA3 GPTQ kernel, what its GPTQ layers run on gfx11)

- skipped: ba: left unquantized by the checkpoint

