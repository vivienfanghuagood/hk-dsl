### linear-w4 at Qwen/Qwen3-14B-AWQ (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 39.95 | 92.41 (vLLM awq_gemm (Triton)) | **2.31x** | 39.95 | 195.43 | 92.41 |
| qkv | 2 | 40.02 | 93.39 (vLLM awq_gemm (Triton)) | **2.33x** | 40.02 | 205.70 | 93.39 |
| qkv | 4 | 40.08 | 95.28 (vLLM awq_gemm (Triton)) | **2.38x** | 40.08 | 205.61 | 95.28 |
| qkv | 8 | 40.50 | 94.84 (vLLM awq_gemm (Triton)) | **2.34x** | 40.50 | 205.80 | 94.84 |
| qkv | 16 | 41.78 | 96.69 (vLLM awq_gemm (Triton)) | **2.31x** | 41.78 | 205.97 | 96.69 |
| qkv | 32 | 61.02 | 99.87 (vLLM awq_gemm (Triton)) | **1.64x** | 61.02 | 206.43 | 99.87 |
| qkv | 64 | 112.00 | 179.40 (vLLM awq_gemm (Triton)) | **1.60x** | 112.00 | 207.43 | 179.40 |
| qkv | 256 | -- | 409.11 (vLLM AWQ dequantize + matmul) | -- | -- | 409.11 | 661.14 |
| o | 1 | 29.64 | 70.23 (vLLM awq_gemm (Triton)) | **2.37x** | 29.64 | 188.07 | 70.23 |
| o | 2 | 29.67 | 70.20 (vLLM awq_gemm (Triton)) | **2.37x** | 29.67 | 188.23 | 70.20 |
| o | 4 | 29.79 | 70.45 (vLLM awq_gemm (Triton)) | **2.37x** | 29.79 | 188.72 | 70.45 |
| o | 8 | 29.87 | 70.45 (vLLM awq_gemm (Triton)) | **2.36x** | 29.87 | 188.74 | 70.45 |
| o | 16 | 31.59 | 71.38 (vLLM awq_gemm (Triton)) | **2.26x** | 31.59 | 188.99 | 71.38 |
| o | 32 | 43.77 | 73.61 (vLLM awq_gemm (Triton)) | **1.68x** | 43.77 | 189.34 | 73.61 |
| o | 64 | 83.88 | 132.72 (vLLM awq_gemm (Triton)) | **1.58x** | 83.88 | 190.61 | 132.72 |
| o | 256 | -- | 295.36 (vLLM AWQ dequantize + matmul) | -- | -- | 295.36 | 475.17 |
| gate_up | 1 | 165.10 | 385.42 (vLLM awq_gemm (Triton)) | **2.33x** | 165.10 | 1444.98 | 385.42 |
| gate_up | 2 | 165.35 | 386.85 (vLLM awq_gemm (Triton)) | **2.34x** | 165.35 | 1449.70 | 386.85 |
| gate_up | 4 | 165.74 | 388.90 (vLLM awq_gemm (Triton)) | **2.35x** | 165.74 | 1444.43 | 388.90 |
| gate_up | 8 | 166.43 | 393.17 (vLLM awq_gemm (Triton)) | **2.36x** | 166.43 | 1464.03 | 393.17 |
| gate_up | 16 | 168.80 | 404.13 (vLLM awq_gemm (Triton)) | **2.39x** | 168.80 | 1466.69 | 404.13 |
| gate_up | 32 | 263.63 | 433.29 (vLLM awq_gemm (Triton)) | **1.64x** | 263.63 | 1473.52 | 433.29 |
| gate_up | 64 | 510.02 | 852.33 (vLLM awq_gemm (Triton)) | **1.67x** | 510.02 | 1478.37 | 852.33 |
| gate_up | 256 | -- | 2161.06 (vLLM AWQ dequantize + matmul) | -- | -- | 2161.06 | 3486.08 |
| down | 1 | 88.19 | 214.34 (vLLM awq_gemm (Triton)) | **2.43x** | 88.19 | 735.59 | 214.34 |
| down | 2 | 88.31 | 214.53 (vLLM awq_gemm (Triton)) | **2.43x** | 88.31 | 736.25 | 214.53 |
| down | 4 | 88.53 | 214.27 (vLLM awq_gemm (Triton)) | **2.42x** | 88.53 | 734.24 | 214.27 |
| down | 8 | 90.18 | 216.21 (vLLM awq_gemm (Triton)) | **2.40x** | 90.18 | 738.17 | 216.21 |
| down | 16 | 92.17 | 217.37 (vLLM awq_gemm (Triton)) | **2.36x** | 92.17 | 736.43 | 217.37 |
| down | 32 | 159.03 | 222.34 (vLLM awq_gemm (Triton)) | **1.40x** | 159.03 | 733.31 | 222.34 |
| down | 64 | 288.63 | 424.35 (vLLM awq_gemm (Triton)) | **1.47x** | 288.63 | 739.94 | 424.35 |
| down | 256 | -- | 1091.27 (vLLM AWQ dequantize + matmul) | -- | -- | 1091.27 | 1579.15 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

