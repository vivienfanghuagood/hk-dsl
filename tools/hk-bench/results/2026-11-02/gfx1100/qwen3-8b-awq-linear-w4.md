### linear-w4 at Qwen/Qwen3-8B-AWQ (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 29.24 | 65.38 (vLLM awq_gemm (Triton)) | **2.24x** | 29.24 | 155.79 | 65.38 |
| qkv | 2 | 29.29 | 65.44 (vLLM awq_gemm (Triton)) | **2.23x** | 29.29 | 155.97 | 65.44 |
| qkv | 4 | 29.37 | 65.56 (vLLM awq_gemm (Triton)) | **2.23x** | 29.37 | 155.84 | 65.56 |
| qkv | 8 | 29.57 | 65.69 (vLLM awq_gemm (Triton)) | **2.22x** | 29.57 | 155.85 | 65.69 |
| qkv | 16 | 30.20 | 78.22 (vLLM awq_gemm (Triton)) | **2.59x** | 30.20 | 157.44 | 78.22 |
| qkv | 32 | 41.09 | 70.25 (vLLM awq_gemm (Triton)) | **1.71x** | 41.09 | 157.84 | 70.25 |
| qkv | 64 | 64.26 | 134.71 (vLLM awq_gemm (Triton)) | **2.10x** | 64.26 | 159.13 | 134.71 |
| qkv | 256 | -- | 250.22 (vLLM AWQ dequantize + matmul) | -- | -- | 250.22 | 463.73 |
| o | 1 | 22.86 | 49.64 (vLLM awq_gemm (Triton)) | **2.17x** | 22.86 | 124.72 | 49.64 |
| o | 2 | 22.88 | 49.88 (vLLM awq_gemm (Triton)) | **2.18x** | 22.88 | 124.14 | 49.88 |
| o | 4 | 22.92 | 50.00 (vLLM awq_gemm (Triton)) | **2.18x** | 22.92 | 124.28 | 50.00 |
| o | 8 | 23.21 | 50.29 (vLLM awq_gemm (Triton)) | **2.17x** | 23.21 | 124.38 | 50.29 |
| o | 16 | 23.40 | 50.88 (vLLM awq_gemm (Triton)) | **2.17x** | 23.40 | 124.43 | 50.88 |
| o | 32 | 31.16 | 52.51 (vLLM awq_gemm (Triton)) | **1.69x** | 31.16 | 124.64 | 52.51 |
| o | 64 | 48.06 | 92.13 (vLLM awq_gemm (Triton)) | **1.92x** | 48.06 | 125.24 | 92.13 |
| o | 256 | -- | 191.03 (vLLM AWQ dequantize + matmul) | -- | -- | 191.03 | 313.76 |
| gate_up | 1 | 97.70 | 225.91 (vLLM awq_gemm (Triton)) | **2.31x** | 97.70 | 894.98 | 225.91 |
| gate_up | 2 | 98.02 | 226.01 (vLLM awq_gemm (Triton)) | **2.31x** | 98.02 | 889.70 | 226.01 |
| gate_up | 4 | 97.89 | 238.77 (vLLM awq_gemm (Triton)) | **2.44x** | 97.89 | 895.02 | 238.77 |
| gate_up | 8 | 99.06 | 228.90 (vLLM awq_gemm (Triton)) | **2.31x** | 99.06 | 888.70 | 228.90 |
| gate_up | 16 | 101.00 | 242.53 (vLLM awq_gemm (Triton)) | **2.40x** | 101.00 | 888.98 | 242.53 |
| gate_up | 32 | 148.25 | 257.85 (vLLM awq_gemm (Triton)) | **1.74x** | 148.25 | 894.79 | 257.85 |
| gate_up | 64 | 238.85 | 480.07 (vLLM awq_gemm (Triton)) | **2.01x** | 238.85 | 893.67 | 480.07 |
| gate_up | 256 | -- | 1163.24 (vLLM AWQ dequantize + matmul) | -- | -- | 1163.24 | 1921.86 |
| down | 1 | 56.45 | 126.79 (vLLM awq_gemm (Triton)) | **2.25x** | 56.45 | 350.24 | 126.79 |
| down | 2 | 56.29 | 128.69 (vLLM awq_gemm (Triton)) | **2.29x** | 56.29 | 348.95 | 128.69 |
| down | 4 | 56.36 | 129.73 (vLLM awq_gemm (Triton)) | **2.30x** | 56.36 | 348.85 | 129.73 |
| down | 8 | 57.76 | 129.74 (vLLM awq_gemm (Triton)) | **2.25x** | 57.76 | 349.76 | 129.74 |
| down | 16 | 59.99 | 130.68 (vLLM awq_gemm (Triton)) | **2.18x** | 59.99 | 347.19 | 130.68 |
| down | 32 | 88.83 | 133.82 (vLLM awq_gemm (Triton)) | **1.51x** | 88.83 | 346.62 | 133.82 |
| down | 64 | 150.74 | 240.75 (vLLM awq_gemm (Triton)) | **1.60x** | 150.74 | 355.88 | 240.75 |
| down | 256 | -- | 548.94 (vLLM AWQ dequantize + matmul) | -- | -- | 548.94 | 888.44 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

