### linear-w4 at Qwen/Qwen3-14B-AWQ (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 35.85 | 76.42 (vLLM awq_gemm (Triton)) | **2.13x** | 35.85 | 238.60 | 76.42 |
| qkv | 2 | 35.97 | 76.71 (vLLM awq_gemm (Triton)) | **2.13x** | 35.97 | 242.19 | 76.71 |
| qkv | 4 | 36.24 | 77.12 (vLLM awq_gemm (Triton)) | **2.13x** | 36.24 | 244.00 | 77.12 |
| qkv | 8 | 36.51 | 77.23 (vLLM awq_gemm (Triton)) | **2.12x** | 36.51 | 241.93 | 77.23 |
| qkv | 16 | 36.91 | 78.90 (vLLM awq_gemm (Triton)) | **2.14x** | 36.91 | 241.94 | 78.90 |
| qkv | 32 | 49.15 | 84.33 (vLLM awq_gemm (Triton)) | **1.72x** | 49.15 | 247.69 | 84.33 |
| qkv | 64 | 84.55 | 158.36 (vLLM awq_gemm (Triton)) | **1.87x** | 84.55 | 251.34 | 158.36 |
| qkv | 256 | -- | 293.42 (vLLM AWQ dequantize + matmul) | -- | -- | 293.42 | 635.14 |
| o | 1 | 26.99 | 58.82 (vLLM awq_gemm (Triton)) | **2.18x** | 26.99 | 99.10 | 58.82 |
| o | 2 | 27.18 | 58.47 (vLLM awq_gemm (Triton)) | **2.15x** | 27.18 | 99.64 | 58.47 |
| o | 4 | 27.38 | 58.88 (vLLM awq_gemm (Triton)) | **2.15x** | 27.38 | 98.60 | 58.88 |
| o | 8 | 27.40 | 59.88 (vLLM awq_gemm (Triton)) | **2.19x** | 27.40 | 99.40 | 59.88 |
| o | 16 | 27.82 | 59.78 (vLLM awq_gemm (Triton)) | **2.15x** | 27.82 | 100.15 | 59.78 |
| o | 32 | 35.34 | 63.14 (vLLM awq_gemm (Triton)) | **1.79x** | 35.34 | 102.44 | 63.14 |
| o | 64 | 66.74 | 102.03 (vLLM AWQ dequantize + matmul) | **1.53x** | 66.74 | 102.03 | 116.11 |
| o | 256 | -- | 171.81 (vLLM AWQ dequantize + matmul) | -- | -- | 171.81 | 451.99 |
| gate_up | 1 | 155.69 | 326.31 (vLLM awq_gemm (Triton)) | **2.10x** | 155.69 | 1437.85 | 326.31 |
| gate_up | 2 | 156.29 | 326.81 (vLLM awq_gemm (Triton)) | **2.09x** | 156.29 | 1435.42 | 326.81 |
| gate_up | 4 | 157.74 | 330.00 (vLLM awq_gemm (Triton)) | **2.09x** | 157.74 | 1437.45 | 330.00 |
| gate_up | 8 | 159.96 | 335.02 (vLLM awq_gemm (Triton)) | **2.09x** | 159.96 | 1440.81 | 335.02 |
| gate_up | 16 | 162.88 | 344.56 (vLLM awq_gemm (Triton)) | **2.12x** | 162.88 | 1440.51 | 344.56 |
| gate_up | 32 | 207.84 | 374.66 (vLLM awq_gemm (Triton)) | **1.80x** | 207.84 | 1445.25 | 374.66 |
| gate_up | 64 | 388.13 | 775.38 (vLLM awq_gemm (Triton)) | **2.00x** | 388.13 | 1448.31 | 775.38 |
| gate_up | 256 | -- | 1547.61 (vLLM AWQ dequantize + matmul) | -- | -- | 1547.61 | 3338.35 |
| down | 1 | 81.57 | 177.56 (vLLM awq_gemm (Triton)) | **2.18x** | 81.57 | 705.32 | 177.56 |
| down | 2 | 81.51 | 179.07 (vLLM awq_gemm (Triton)) | **2.20x** | 81.51 | 705.67 | 179.07 |
| down | 4 | 81.52 | 179.93 (vLLM awq_gemm (Triton)) | **2.21x** | 81.52 | 705.60 | 179.93 |
| down | 8 | 81.71 | 180.59 (vLLM awq_gemm (Triton)) | **2.21x** | 81.71 | 706.34 | 180.59 |
| down | 16 | 82.68 | 180.54 (vLLM awq_gemm (Triton)) | **2.18x** | 82.68 | 708.50 | 180.54 |
| down | 32 | 120.25 | 188.95 (vLLM awq_gemm (Triton)) | **1.57x** | 120.25 | 710.08 | 188.95 |
| down | 64 | 191.82 | 362.68 (vLLM awq_gemm (Triton)) | **1.89x** | 191.82 | 714.67 | 362.68 |
| down | 256 | -- | 796.99 (vLLM AWQ dequantize + matmul) | -- | -- | 796.99 | 1491.55 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

