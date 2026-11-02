### linear at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 34.03 | 32.60 (vLLM) | **0.96x** | 34.03 | 46.51 | 32.72 | 34.74 | 34.09 | 32.60 |
| qkv | 2 | 34.05 | 32.78 (vLLM) | **0.96x** | 34.05 | 46.26 | 36.59 | 34.74 | 34.07 | 32.78 |
| qkv | 4 | 34.17 | 33.72 (vLLM) | **0.99x** | 34.17 | 46.31 | 36.69 | 34.73 | 34.07 | 33.72 |
| qkv | 8 | 34.32 | 34.13 (hipBLASLt+TunableOp) | **0.99x** | 34.32 | 45.66 | 36.83 | 34.78 | 34.13 | 34.20 |
| qkv | 16 | 34.51 | 34.10 (hipBLASLt+TunableOp) | **0.99x** | 34.51 | 47.53 | 37.10 | 34.96 | 34.10 | 34.11 |
| qkv | 32 | 35.84 | 34.37 (vLLM) | **0.96x** | 35.84 | 48.11 | 35.30 | 35.35 | 34.38 | 34.37 |
| qkv | 64 | 44.10 | 35.60 (hipBLASLt+TunableOp) | **0.81x** | 44.10 | 48.98 | 50.55 | 35.70 | 35.60 | 35.61 |
| qkv | 256 | 64.02 | 49.12 (hipBLASLt) | **0.77x** | -- | 64.02 | 71.35 | 49.12 | 61.48 | 61.53 |
| qkv | 1024 | 170.74 | 168.50 (vLLM) | **0.99x** | -- | 170.74 | 173.29 | 180.97 | 169.92 | 168.50 |
| o | 1 | 17.36 | 16.02 (vLLM) | **0.92x** | 17.36 | 25.92 | 17.02 | 28.57 | 17.07 | 16.02 |
| o | 2 | 17.45 | 16.38 (vLLM) | **0.94x** | 17.45 | 25.77 | 18.87 | 28.60 | 17.10 | 16.38 |
| o | 4 | 17.57 | 16.95 (hipBLASLt+TunableOp) | **0.96x** | 17.57 | 26.04 | 18.97 | 28.78 | 16.95 | 17.29 |
| o | 8 | 17.62 | 17.18 (hipBLASLt+TunableOp) | **0.97x** | 17.62 | 26.13 | 19.06 | 29.01 | 17.18 | 17.19 |
| o | 16 | 17.60 | 17.02 (vLLM) | **0.97x** | 17.60 | 26.10 | 19.25 | 29.55 | 17.04 | 17.02 |
| o | 32 | 18.28 | 17.28 (vLLM) | **0.94x** | 18.28 | 27.44 | 18.10 | 29.65 | 17.29 | 17.28 |
| o | 64 | 25.78 | 17.76 (hipBLASLt+TunableOp) | **0.69x** | 25.78 | 27.07 | 20.41 | 29.02 | 17.76 | 17.77 |
| o | 256 | 31.86 | 27.60 (hipBLASLt+TunableOp) | **0.87x** | -- | 31.86 | 33.91 | 33.98 | 27.60 | 28.59 |
| o | 1024 | 94.28 | 88.18 (Inductor max-autotune) | **0.94x** | -- | 94.28 | 88.18 | 91.75 | 88.94 | 88.94 |
| qkvz | 1 | 44.10 | 42.59 (vLLM) | **0.97x** | 44.10 | 60.20 | 43.87 | 44.22 | 43.89 | 42.59 |
| qkvz | 2 | 44.15 | 42.89 (vLLM) | **0.97x** | 44.15 | 60.28 | 49.20 | 44.20 | 43.91 | 42.89 |
| qkvz | 4 | 44.31 | 43.78 (vLLM) | **0.99x** | 44.31 | 60.96 | 49.04 | 44.36 | 44.02 | 43.78 |
| qkvz | 8 | 44.60 | 44.06 (hipBLASLt+TunableOp) | **0.99x** | 44.60 | 61.16 | 44.39 | 44.39 | 44.06 | 44.06 |
| qkvz | 16 | 44.89 | 44.38 (hipBLASLt+TunableOp) | **0.99x** | 44.89 | 59.52 | 50.12 | 44.56 | 44.38 | 44.40 |
| qkvz | 32 | 45.97 | 44.46 (hipBLASLt+TunableOp) | **0.97x** | 45.97 | 63.53 | 44.91 | 44.92 | 44.46 | 44.46 |
| qkvz | 64 | 54.12 | 45.63 (hipBLASLt) | **0.84x** | 54.12 | 66.88 | 45.66 | 45.63 | 45.73 | 45.70 |
| qkvz | 256 | 90.85 | 66.13 (hipBLASLt) | **0.73x** | -- | 90.85 | 88.65 | 66.13 | 67.80 | 68.38 |
| qkvz | 1024 | 226.69 | 213.64 (vLLM) | **0.94x** | -- | 226.69 | 232.46 | 238.11 | 214.74 | 213.64 |
| ba | 1 | 3.85 | 2.71 (Inductor max-autotune) | **0.71x** | 3.85 | -- | 2.71 | 13.75 | 4.68 | 3.07 |
| ba | 2 | 3.84 | 3.19 (vLLM) | **0.83x** | 3.84 | -- | 7.43 | 13.72 | 7.02 | 3.19 |
| ba | 4 | 3.83 | 3.57 (vLLM) | **0.93x** | 3.83 | -- | 7.55 | 13.74 | 7.13 | 3.57 |
| ba | 8 | 3.83 | 7.03 (vLLM) | **1.84x** | 3.83 | -- | 7.61 | 13.75 | 7.04 | 7.03 |
| ba | 16 | 3.83 | 7.06 (vLLM) | **1.84x** | 3.83 | -- | 7.67 | 13.63 | 7.07 | 7.06 |
| ba | 32 | 5.19 | 7.09 (vLLM) | **1.36x** | 5.19 | -- | 7.15 | 13.37 | 7.09 | 7.09 |
| ba | 64 | 7.95 | 7.07 (vLLM) | **0.89x** | 7.95 | -- | 7.12 | 12.91 | 7.07 | 7.07 |
| ba | 256 | -- | 7.48 (Inductor max-autotune) | -- | -- | -- | 7.48 | 10.31 | 8.55 | 8.54 |
| ba | 1024 | -- | 7.78 (vLLM) | -- | -- | -- | 10.44 | 10.38 | 7.79 | 7.78 |
| out | 1 | 17.23 | 15.96 (vLLM) | **0.93x** | 17.23 | 25.64 | 16.97 | 28.17 | 16.97 | 15.96 |
| out | 2 | 17.34 | 16.29 (vLLM) | **0.94x** | 17.34 | 25.66 | 18.80 | 28.22 | 17.01 | 16.29 |
| out | 4 | 17.45 | 16.87 (hipBLASLt+TunableOp) | **0.97x** | 17.45 | 25.66 | 18.91 | 28.74 | 16.87 | 17.20 |
| out | 8 | 17.55 | 17.11 (vLLM) | **0.97x** | 17.55 | 25.90 | 19.02 | 28.64 | 17.11 | 17.11 |
| out | 16 | 17.52 | 16.99 (hipBLASLt+TunableOp) | **0.97x** | 17.52 | 25.90 | 19.17 | 28.52 | 16.99 | 16.99 |
| out | 32 | 18.20 | 17.21 (vLLM) | **0.94x** | 18.20 | 27.27 | 18.01 | 28.98 | 17.22 | 17.21 |
| out | 64 | 25.80 | 17.73 (vLLM) | **0.69x** | 25.80 | 27.35 | 20.35 | 29.61 | 17.74 | 17.73 |
| out | 256 | 31.83 | 28.35 (hipBLASLt+TunableOp) | **0.89x** | -- | 31.83 | 33.75 | 33.63 | 28.35 | 28.68 |
| out | 1024 | 95.31 | 87.74 (vLLM) | **0.92x** | -- | 95.31 | 88.12 | 94.49 | 87.88 | 87.74 |
| shared_gate_up | 1 | 6.52 | 5.96 (vLLM) | **0.91x** | 6.52 | 15.72 | 6.13 | 14.15 | 7.85 | 5.96 |
| shared_gate_up | 2 | 6.56 | 6.08 (vLLM) | **0.93x** | 6.56 | 15.73 | 8.79 | 14.13 | 7.84 | 6.08 |
| shared_gate_up | 4 | 6.57 | 6.34 (vLLM) | **0.96x** | 6.57 | 15.73 | 8.84 | 14.12 | 7.87 | 6.34 |
| shared_gate_up | 8 | 6.59 | 7.89 (vLLM) | **1.20x** | 6.59 | 15.65 | 8.95 | 14.40 | 7.90 | 7.89 |
| shared_gate_up | 16 | 6.61 | 7.67 (vLLM) | **1.16x** | 6.61 | 16.12 | 9.15 | 14.42 | 7.69 | 7.67 |
| shared_gate_up | 32 | 7.12 | 7.40 (vLLM) | **1.04x** | 7.12 | 20.61 | 7.91 | 14.51 | 7.42 | 7.40 |
| shared_gate_up | 64 | 11.82 | 7.60 (vLLM) | **0.64x** | 11.82 | 20.66 | 8.19 | 13.54 | 7.60 | 7.60 |
| shared_gate_up | 256 | 20.86 | 10.84 (Inductor max-autotune) | **0.52x** | -- | 20.86 | 10.84 | 16.27 | 10.92 | 10.87 |
| shared_gate_up | 1024 | 25.61 | 28.58 (hipBLASLt+TunableOp) | **1.12x** | -- | 25.61 | 30.00 | 29.21 | 28.58 | 28.66 |
| shared_down | 1 | 5.04 | 4.24 (Inductor max-autotune) | **0.84x** | 5.04 | 5.12 | 4.24 | 8.21 | 5.21 | 4.25 |
| shared_down | 2 | 5.05 | 4.33 (Inductor max-autotune) | **0.86x** | 5.05 | 5.14 | 4.33 | 8.21 | 5.19 | 4.50 |
| shared_down | 4 | 5.08 | 4.37 (Inductor max-autotune) | **0.86x** | 5.08 | 5.13 | 4.37 | 8.15 | 5.19 | 4.70 |
| shared_down | 8 | 5.09 | 4.34 (Inductor max-autotune) | **0.85x** | 5.09 | 5.20 | 4.34 | 8.21 | 5.18 | 5.18 |
| shared_down | 16 | 5.24 | 4.43 (Inductor max-autotune) | **0.85x** | 5.24 | 5.28 | 4.43 | 8.24 | 5.11 | 5.12 |
| shared_down | 32 | 5.75 | 4.43 (Inductor max-autotune) | **0.77x** | 5.75 | 6.19 | 4.43 | 8.20 | 5.65 | 5.65 |
| shared_down | 64 | 6.21 | 4.31 (Inductor max-autotune) | **0.69x** | 7.50 | 6.21 | 4.31 | 8.22 | 6.07 | 6.07 |
| shared_down | 256 | 6.66 | 5.91 (Inductor max-autotune) | **0.89x** | -- | 6.66 | 5.91 | 8.47 | 7.11 | 7.10 |
| shared_down | 1024 | 15.36 | 14.15 (Inductor max-autotune) | **0.92x** | -- | 15.36 | 14.15 | 17.40 | 16.44 | 16.81 |
| lm_head | 1 | 816.06 | 812.38 (vLLM) | **0.99x** | 816.06 | -- | 814.04 | 815.82 | 814.04 | 812.38 |
| lm_head | 2 | 817.74 | 812.36 (vLLM) | **0.99x** | 817.74 | -- | 818.00 | 817.90 | 815.98 | 812.36 |
| lm_head | 4 | 822.06 | 814.84 (vLLM) | **0.99x** | 822.06 | -- | 822.62 | 822.48 | 822.68 | 814.84 |
| lm_head | 8 | 829.64 | 829.62 (hipBLASLt+TunableOp) | **1.00x** | 829.64 | -- | 830.46 | 830.52 | 829.62 | 829.80 |
| lm_head | 16 | 841.80 | 840.66 (vLLM) | **1.00x** | 841.80 | -- | 841.84 | 843.66 | 842.14 | 840.66 |
| lm_head | 32 | 858.48 | 847.96 (Inductor max-autotune) | **0.99x** | 858.48 | -- | 847.96 | 859.30 | 855.94 | 850.08 |
| lm_head | 64 | 896.20 | 869.12 (Inductor max-autotune) | **0.97x** | 896.20 | -- | 869.12 | 886.22 | 875.48 | 876.46 |
| lm_head | 256 | -- | 1105.66 (hipBLASLt+TunableOp) | -- | -- | -- | 1161.18 | 1150.58 | 1105.66 | 1200.60 |

TunableOp: 62 of 62 (M, N, K) tuned here (M = 1 included), 147 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a: FlyDSL not available: its RDNA3 GEMM is for gfx11, and this is gfx1201
