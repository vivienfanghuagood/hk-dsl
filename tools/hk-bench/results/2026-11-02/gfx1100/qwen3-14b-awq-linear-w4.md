### linear-w4 at Qwen/Qwen3-14B-AWQ (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 40.01 | 92.07 (vLLM awq_gemm (Triton)) | **2.30x** | 40.01 | 197.11 | 92.07 |
| qkv | 2 | 40.00 | 93.68 (vLLM awq_gemm (Triton)) | **2.34x** | 40.00 | 207.51 | 93.68 |
| qkv | 4 | 40.15 | 94.71 (vLLM awq_gemm (Triton)) | **2.36x** | 40.15 | 207.78 | 94.71 |
| qkv | 8 | 40.56 | 95.55 (vLLM awq_gemm (Triton)) | **2.36x** | 40.56 | 208.62 | 95.55 |
| qkv | 16 | 41.81 | 96.92 (vLLM awq_gemm (Triton)) | **2.32x** | 41.81 | 208.41 | 96.92 |
| qkv | 32 | 61.10 | 100.17 (vLLM awq_gemm (Triton)) | **1.64x** | 61.10 | 208.77 | 100.17 |
| qkv | 64 | 112.21 | 179.95 (vLLM awq_gemm (Triton)) | **1.60x** | 112.21 | 210.12 | 179.95 |
| qkv | 256 | -- | 407.80 (vLLM AWQ dequantize + matmul) | -- | -- | 407.80 | 658.60 |
| o | 1 | 29.64 | 69.74 (vLLM awq_gemm (Triton)) | **2.35x** | 29.64 | 190.65 | 69.74 |
| o | 2 | 29.71 | 70.01 (vLLM awq_gemm (Triton)) | **2.36x** | 29.71 | 190.90 | 70.01 |
| o | 4 | 29.84 | 70.12 (vLLM awq_gemm (Triton)) | **2.35x** | 29.84 | 191.17 | 70.12 |
| o | 8 | 30.00 | 70.34 (vLLM awq_gemm (Triton)) | **2.35x** | 30.00 | 191.25 | 70.34 |
| o | 16 | 31.71 | 71.11 (vLLM awq_gemm (Triton)) | **2.24x** | 31.71 | 191.60 | 71.11 |
| o | 32 | 43.86 | 73.62 (vLLM awq_gemm (Triton)) | **1.68x** | 43.86 | 192.10 | 73.62 |
| o | 64 | 83.82 | 136.11 (vLLM awq_gemm (Triton)) | **1.62x** | 83.82 | 192.92 | 136.11 |
| o | 256 | -- | 295.83 (vLLM AWQ dequantize + matmul) | -- | -- | 295.83 | 474.75 |
| gate_up | 1 | 165.48 | 385.46 (vLLM awq_gemm (Triton)) | **2.33x** | 165.48 | 1466.76 | 385.46 |
| gate_up | 2 | 165.59 | 386.29 (vLLM awq_gemm (Triton)) | **2.33x** | 165.59 | 1453.99 | 386.29 |
| gate_up | 4 | 166.02 | 388.28 (vLLM awq_gemm (Triton)) | **2.34x** | 166.02 | 1450.67 | 388.28 |
| gate_up | 8 | 166.58 | 394.75 (vLLM awq_gemm (Triton)) | **2.37x** | 166.58 | 1435.00 | 394.75 |
| gate_up | 16 | 169.09 | 404.97 (vLLM awq_gemm (Triton)) | **2.40x** | 169.09 | 1468.39 | 404.97 |
| gate_up | 32 | 263.52 | 427.47 (vLLM awq_gemm (Triton)) | **1.62x** | 263.52 | 1480.30 | 427.47 |
| gate_up | 64 | 508.69 | 856.82 (vLLM awq_gemm (Triton)) | **1.68x** | 508.69 | 1488.77 | 856.82 |
| gate_up | 256 | -- | 2179.38 (vLLM AWQ dequantize + matmul) | -- | -- | 2179.38 | 3439.54 |
| down | 1 | 88.31 | 213.99 (vLLM awq_gemm (Triton)) | **2.42x** | 88.31 | 741.85 | 213.99 |
| down | 2 | 88.40 | 213.88 (vLLM awq_gemm (Triton)) | **2.42x** | 88.40 | 739.40 | 213.88 |
| down | 4 | 88.66 | 214.25 (vLLM awq_gemm (Triton)) | **2.42x** | 88.66 | 737.59 | 214.25 |
| down | 8 | 90.21 | 214.84 (vLLM awq_gemm (Triton)) | **2.38x** | 90.21 | 741.60 | 214.84 |
| down | 16 | 92.24 | 215.98 (vLLM awq_gemm (Triton)) | **2.34x** | 92.24 | 739.95 | 215.98 |
| down | 32 | 159.01 | 221.19 (vLLM awq_gemm (Triton)) | **1.39x** | 159.01 | 738.33 | 221.19 |
| down | 64 | 288.40 | 423.73 (vLLM awq_gemm (Triton)) | **1.47x** | 288.40 | 742.69 | 423.73 |
| down | 256 | -- | 1093.43 (vLLM AWQ dequantize + matmul) | -- | -- | 1093.43 | 1541.81 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

