### linear-w4 at Qwen/Qwen3-8B-AWQ (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk w4a16 | vLLM AWQ dequantize + matmul | vLLM awq_gemm (Triton) |
|---|---:|---:|---|---:|---:|---:|---:|
| qkv | 1 | 26.26 | 55.68 (vLLM awq_gemm (Triton)) | **2.12x** | 26.26 | 102.83 | 55.68 |
| qkv | 2 | 26.77 | 55.47 (vLLM awq_gemm (Triton)) | **2.07x** | 26.77 | 102.95 | 55.47 |
| qkv | 4 | 26.90 | 55.91 (vLLM awq_gemm (Triton)) | **2.08x** | 26.90 | 102.75 | 55.91 |
| qkv | 8 | 27.12 | 56.53 (vLLM awq_gemm (Triton)) | **2.08x** | 27.12 | 103.04 | 56.53 |
| qkv | 16 | 27.45 | 56.82 (vLLM awq_gemm (Triton)) | **2.07x** | 27.45 | 102.85 | 56.82 |
| qkv | 32 | 33.19 | 60.58 (vLLM awq_gemm (Triton)) | **1.82x** | 33.19 | 103.22 | 60.58 |
| qkv | 64 | 58.48 | 104.36 (vLLM AWQ dequantize + matmul) | **1.78x** | 58.48 | 104.36 | 113.75 |
| qkv | 256 | -- | 178.11 (vLLM AWQ dequantize + matmul) | -- | -- | 178.11 | 436.57 |
| o | 1 | 18.69 | 40.68 (vLLM awq_gemm (Triton)) | **2.18x** | 18.69 | 73.88 | 40.68 |
| o | 2 | 19.12 | 40.05 (vLLM awq_gemm (Triton)) | **2.10x** | 19.12 | 74.08 | 40.05 |
| o | 4 | 19.11 | 40.22 (vLLM awq_gemm (Triton)) | **2.10x** | 19.11 | 74.10 | 40.22 |
| o | 8 | 19.25 | 40.59 (vLLM awq_gemm (Triton)) | **2.11x** | 19.25 | 74.88 | 40.59 |
| o | 16 | 19.31 | 53.14 (vLLM awq_gemm (Triton)) | **2.75x** | 19.31 | 74.50 | 53.14 |
| o | 32 | 23.70 | 44.35 (vLLM awq_gemm (Triton)) | **1.87x** | 23.70 | 76.24 | 44.35 |
| o | 64 | 42.08 | 86.03 (vLLM awq_gemm (Triton)) | **2.04x** | 42.08 | 90.27 | 86.03 |
| o | 256 | -- | 292.71 (vLLM awq_gemm (Triton)) | -- | -- | 300.90 | 292.71 |
| gate_up | 1 | 91.00 | 192.16 (vLLM awq_gemm (Triton)) | **2.11x** | 91.00 | 802.20 | 192.16 |
| gate_up | 2 | 92.25 | 192.58 (vLLM awq_gemm (Triton)) | **2.09x** | 92.25 | 788.53 | 192.58 |
| gate_up | 4 | 92.88 | 193.27 (vLLM awq_gemm (Triton)) | **2.08x** | 92.88 | 795.68 | 193.27 |
| gate_up | 8 | 93.76 | 197.71 (vLLM awq_gemm (Triton)) | **2.11x** | 93.76 | 793.94 | 197.71 |
| gate_up | 16 | 95.32 | 203.30 (vLLM awq_gemm (Triton)) | **2.13x** | 95.32 | 790.50 | 203.30 |
| gate_up | 32 | 118.89 | 217.23 (vLLM awq_gemm (Triton)) | **1.83x** | 118.89 | 788.97 | 217.23 |
| gate_up | 64 | 215.75 | 428.63 (vLLM awq_gemm (Triton)) | **1.99x** | 215.75 | 810.16 | 428.63 |
| gate_up | 256 | -- | 870.86 (vLLM AWQ dequantize + matmul) | -- | -- | 870.86 | 1952.26 |
| down | 1 | 48.66 | 103.44 (vLLM awq_gemm (Triton)) | **2.13x** | 48.66 | 410.69 | 103.44 |
| down | 2 | 51.77 | 102.89 (vLLM awq_gemm (Triton)) | **1.99x** | 51.77 | 410.13 | 102.89 |
| down | 4 | 51.55 | 103.59 (vLLM awq_gemm (Triton)) | **2.01x** | 51.55 | 410.44 | 103.59 |
| down | 8 | 53.31 | 104.80 (vLLM awq_gemm (Triton)) | **1.97x** | 53.31 | 411.53 | 104.80 |
| down | 16 | 53.53 | 116.96 (vLLM awq_gemm (Triton)) | **2.19x** | 53.53 | 413.25 | 116.96 |
| down | 32 | 65.84 | 111.33 (vLLM awq_gemm (Triton)) | **1.69x** | 65.84 | 415.58 | 111.33 |
| down | 64 | 111.38 | 215.89 (vLLM awq_gemm (Triton)) | **1.94x** | 111.38 | 418.28 | 215.89 |
| down | 256 | -- | 830.66 (vLLM awq_gemm (Triton)) | -- | -- | 1425.96 | 830.66 |

**Setup of each arm.**
- hk w4a16: hk.ops.w4a16.matmul, zero points (repack_awq, as vllm_int4 loads it)
- vLLM AWQ dequantize + matmul: vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's compiled graph runs (VLLM_USE_TRITON_AWQ=1)
- vLLM awq_gemm (Triton): vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod below 256 tokens, eager

