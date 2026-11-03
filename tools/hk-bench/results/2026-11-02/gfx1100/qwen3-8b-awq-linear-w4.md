### linear-w4 at Qwen/Qwen3-8B-AWQ (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 29.10 | 65.12 (vLLM awq_gemm (Triton)) | **2.24x** | 29.10 | 153.33 | 65.12 |
| qkv | 2 | 29.16 | 65.33 (vLLM awq_gemm (Triton)) | **2.24x** | 29.16 | 153.28 | 65.33 |
| qkv | 4 | 29.30 | 65.41 (vLLM awq_gemm (Triton)) | **2.23x** | 29.30 | 153.51 | 65.41 |
| qkv | 8 | 29.61 | 65.88 (vLLM awq_gemm (Triton)) | **2.23x** | 29.61 | 153.65 | 65.88 |
| qkv | 16 | 29.92 | 77.05 (vLLM awq_gemm (Triton)) | **2.58x** | 29.92 | 154.02 | 77.05 |
| qkv | 32 | 40.66 | 69.55 (vLLM awq_gemm (Triton)) | **1.71x** | 40.66 | 154.29 | 69.55 |
| qkv | 64 | 63.46 | 132.61 (vLLM awq_gemm (Triton)) | **2.09x** | 63.46 | 156.25 | 132.61 |
| qkv | 256 | -- | 249.91 (vLLM AWQ dequantize + matmul) | -- | -- | 249.91 | 471.89 |
| o | 1 | 22.84 | 49.76 (vLLM awq_gemm (Triton)) | **2.18x** | 22.84 | 122.65 | 49.76 |
| o | 2 | 22.87 | 50.31 (vLLM awq_gemm (Triton)) | **2.20x** | 22.87 | 122.20 | 50.31 |
| o | 4 | 22.91 | 50.15 (vLLM awq_gemm (Triton)) | **2.19x** | 22.91 | 122.29 | 50.15 |
| o | 8 | 23.20 | 50.43 (vLLM awq_gemm (Triton)) | **2.17x** | 23.20 | 122.36 | 50.43 |
| o | 16 | 23.41 | 51.02 (vLLM awq_gemm (Triton)) | **2.18x** | 23.41 | 122.34 | 51.02 |
| o | 32 | 31.18 | 52.45 (vLLM awq_gemm (Triton)) | **1.68x** | 31.18 | 122.57 | 52.45 |
| o | 64 | 48.09 | 92.16 (vLLM awq_gemm (Triton)) | **1.92x** | 48.09 | 123.03 | 92.16 |
| o | 256 | -- | 190.77 (vLLM AWQ dequantize + matmul) | -- | -- | 190.77 | 314.13 |
| gate_up | 1 | 97.89 | 226.04 (vLLM awq_gemm (Triton)) | **2.31x** | 97.89 | 891.66 | 226.04 |
| gate_up | 2 | 97.78 | 226.78 (vLLM awq_gemm (Triton)) | **2.32x** | 97.78 | 885.40 | 226.78 |
| gate_up | 4 | 98.13 | 237.10 (vLLM awq_gemm (Triton)) | **2.42x** | 98.13 | 881.39 | 237.10 |
| gate_up | 8 | 99.19 | 230.12 (vLLM awq_gemm (Triton)) | **2.32x** | 99.19 | 894.65 | 230.12 |
| gate_up | 16 | 101.04 | 242.20 (vLLM awq_gemm (Triton)) | **2.40x** | 101.04 | 889.28 | 242.20 |
| gate_up | 32 | 148.37 | 256.32 (vLLM awq_gemm (Triton)) | **1.73x** | 148.37 | 888.39 | 256.32 |
| gate_up | 64 | 236.82 | 481.19 (vLLM awq_gemm (Triton)) | **2.03x** | 236.82 | 898.08 | 481.19 |
| gate_up | 256 | -- | 1151.50 (vLLM AWQ dequantize + matmul) | -- | -- | 1151.50 | 1966.49 |
| down | 1 | 56.04 | 128.81 (vLLM awq_gemm (Triton)) | **2.30x** | 56.04 | 343.04 | 128.81 |
| down | 2 | 56.10 | 128.66 (vLLM awq_gemm (Triton)) | **2.29x** | 56.10 | 345.60 | 128.66 |
| down | 4 | 56.23 | 130.38 (vLLM awq_gemm (Triton)) | **2.32x** | 56.23 | 345.69 | 130.38 |
| down | 8 | 57.69 | 129.04 (vLLM awq_gemm (Triton)) | **2.24x** | 57.69 | 343.21 | 129.04 |
| down | 16 | 60.40 | 131.01 (vLLM awq_gemm (Triton)) | **2.17x** | 60.40 | 344.16 | 131.01 |
| down | 32 | 88.80 | 133.45 (vLLM awq_gemm (Triton)) | **1.50x** | 88.80 | 346.04 | 133.45 |
| down | 64 | 150.82 | 240.52 (vLLM awq_gemm (Triton)) | **1.59x** | 150.82 | 347.69 | 240.52 |
| down | 256 | -- | 547.95 (vLLM AWQ dequantize + matmul) | -- | -- | 547.95 | 889.64 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

