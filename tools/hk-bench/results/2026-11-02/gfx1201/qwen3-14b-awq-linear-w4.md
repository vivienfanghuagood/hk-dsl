### linear-w4 at Qwen/Qwen3-14B-AWQ (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 35.91 | 76.51 (vLLM awq_gemm (Triton)) | **2.13x** | 35.91 | 244.60 | 76.51 |
| qkv | 2 | 36.10 | 76.76 (vLLM awq_gemm (Triton)) | **2.13x** | 36.10 | 244.11 | 76.76 |
| qkv | 4 | 36.29 | 76.47 (vLLM awq_gemm (Triton)) | **2.11x** | 36.29 | 243.25 | 76.47 |
| qkv | 8 | 36.32 | 77.28 (vLLM awq_gemm (Triton)) | **2.13x** | 36.32 | 246.59 | 77.28 |
| qkv | 16 | 36.99 | 79.20 (vLLM awq_gemm (Triton)) | **2.14x** | 36.99 | 244.75 | 79.20 |
| qkv | 32 | 49.17 | 84.43 (vLLM awq_gemm (Triton)) | **1.72x** | 49.17 | 247.85 | 84.43 |
| qkv | 64 | 87.11 | 162.72 (vLLM awq_gemm (Triton)) | **1.87x** | 87.11 | 255.53 | 162.72 |
| qkv | 256 | -- | 295.38 (vLLM AWQ dequantize + matmul) | -- | -- | 295.38 | 665.11 |
| o | 1 | 27.01 | 59.13 (vLLM awq_gemm (Triton)) | **2.19x** | 27.01 | 98.77 | 59.13 |
| o | 2 | 27.22 | 58.70 (vLLM awq_gemm (Triton)) | **2.16x** | 27.22 | 99.43 | 58.70 |
| o | 4 | 27.41 | 59.11 (vLLM awq_gemm (Triton)) | **2.16x** | 27.41 | 100.13 | 59.11 |
| o | 8 | 27.65 | 59.89 (vLLM awq_gemm (Triton)) | **2.17x** | 27.65 | 101.00 | 59.89 |
| o | 16 | 28.03 | 61.31 (vLLM awq_gemm (Triton)) | **2.19x** | 28.03 | 102.47 | 61.31 |
| o | 32 | 36.35 | 66.30 (vLLM awq_gemm (Triton)) | **1.82x** | 36.35 | 102.73 | 66.30 |
| o | 64 | 70.37 | 102.97 (vLLM AWQ dequantize + matmul) | **1.46x** | 70.37 | 102.97 | 126.60 |
| o | 256 | -- | 180.46 (vLLM AWQ dequantize + matmul) | -- | -- | 180.46 | 475.69 |
| gate_up | 1 | 155.41 | 328.04 (vLLM awq_gemm (Triton)) | **2.11x** | 155.41 | 1437.12 | 328.04 |
| gate_up | 2 | 156.32 | 331.29 (vLLM awq_gemm (Triton)) | **2.12x** | 156.32 | 1436.57 | 331.29 |
| gate_up | 4 | 157.61 | 331.73 (vLLM awq_gemm (Triton)) | **2.10x** | 157.61 | 1435.43 | 331.73 |
| gate_up | 8 | 159.81 | 339.80 (vLLM awq_gemm (Triton)) | **2.13x** | 159.81 | 1438.68 | 339.80 |
| gate_up | 16 | 162.89 | 351.08 (vLLM awq_gemm (Triton)) | **2.15x** | 162.89 | 1438.23 | 351.08 |
| gate_up | 32 | 210.24 | 382.74 (vLLM awq_gemm (Triton)) | **1.82x** | 210.24 | 1443.20 | 382.74 |
| gate_up | 64 | 399.33 | 790.85 (vLLM awq_gemm (Triton)) | **1.98x** | 399.33 | 1449.02 | 790.85 |
| gate_up | 256 | -- | 1541.87 (vLLM AWQ dequantize + matmul) | -- | -- | 1541.87 | 3433.47 |
| down | 1 | 81.57 | 180.19 (vLLM awq_gemm (Triton)) | **2.21x** | 81.57 | 705.42 | 180.19 |
| down | 2 | 81.57 | 180.58 (vLLM awq_gemm (Triton)) | **2.21x** | 81.57 | 705.11 | 180.58 |
| down | 4 | 81.60 | 181.31 (vLLM awq_gemm (Triton)) | **2.22x** | 81.60 | 705.52 | 181.31 |
| down | 8 | 81.58 | 182.08 (vLLM awq_gemm (Triton)) | **2.23x** | 81.58 | 706.15 | 182.08 |
| down | 16 | 82.38 | 184.04 (vLLM awq_gemm (Triton)) | **2.23x** | 82.38 | 708.14 | 184.04 |
| down | 32 | 120.49 | 192.52 (vLLM awq_gemm (Triton)) | **1.60x** | 120.49 | 709.97 | 192.52 |
| down | 64 | 199.71 | 373.40 (vLLM awq_gemm (Triton)) | **1.87x** | 199.71 | 714.43 | 373.40 |
| down | 256 | -- | 799.81 (vLLM AWQ dequantize + matmul) | -- | -- | 799.81 | 1543.56 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

