### linear at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 33.99 | 32.59 (vLLM) | **0.96x** | 33.99 | 46.75 | 32.65 | 34.70 | 34.01 | 32.59 |
| qkv | 2 | 34.10 | 32.81 (vLLM) | **0.96x** | 34.10 | 46.43 | 36.66 | 34.76 | 34.10 | 32.81 |
| qkv | 4 | 34.24 | 33.76 (vLLM) | **0.99x** | 34.24 | 46.79 | 36.67 | 34.81 | 34.19 | 33.76 |
| qkv | 8 | 34.42 | 34.19 (hipBLASLt+TunableOp) | **0.99x** | 34.42 | 45.79 | 36.72 | 34.90 | 34.19 | 34.20 |
| qkv | 16 | 34.62 | 34.13 (hipBLASLt+TunableOp) | **0.99x** | 34.62 | 47.60 | 37.14 | 35.05 | 34.13 | 34.17 |
| qkv | 32 | 36.05 | 34.39 (hipBLASLt+TunableOp) | **0.95x** | 36.05 | 48.31 | 35.37 | 35.36 | 34.39 | 34.40 |
| qkv | 64 | 44.21 | 35.57 (hipBLASLt+TunableOp) | **0.81x** | 44.21 | 48.76 | 50.52 | 35.66 | 35.57 | 35.59 |
| qkv | 256 | 62.70 | 48.74 (hipBLASLt) | **0.78x** | -- | 62.70 | 73.23 | 48.74 | 52.87 | 52.32 |
| qkv | 1024 | 151.33 | 157.85 (hipBLASLt+TunableOp) | **1.04x** | -- | 151.33 | 183.18 | 175.56 | 157.85 | 159.57 |
| o | 1 | 17.34 | 16.00 (vLLM) | **0.92x** | 17.34 | 25.81 | 16.95 | 28.53 | 17.08 | 16.00 |
| o | 2 | 17.43 | 16.36 (vLLM) | **0.94x** | 17.43 | 25.88 | 18.88 | 28.44 | 17.10 | 16.36 |
| o | 4 | 17.54 | 16.95 (hipBLASLt+TunableOp) | **0.97x** | 17.54 | 25.98 | 18.95 | 29.04 | 16.95 | 17.28 |
| o | 8 | 17.58 | 17.20 (hipBLASLt+TunableOp) | **0.98x** | 17.58 | 26.07 | 19.14 | 29.03 | 17.20 | 17.20 |
| o | 16 | 17.56 | 17.05 (hipBLASLt+TunableOp) | **0.97x** | 17.56 | 26.26 | 19.27 | 28.70 | 17.05 | 17.05 |
| o | 32 | 18.32 | 17.32 (vLLM) | **0.94x** | 18.32 | 27.39 | 18.06 | 29.44 | 17.32 | 17.32 |
| o | 64 | 25.70 | 17.78 (vLLM) | **0.69x** | 25.70 | 26.95 | 20.36 | 29.27 | 17.79 | 17.78 |
| o | 256 | 31.90 | 23.94 (vLLM) | **0.75x** | -- | 31.90 | 34.01 | 34.29 | 24.13 | 23.94 |
| o | 1024 | 71.38 | 69.46 (vLLM) | **0.97x** | -- | 71.38 | 86.15 | 73.20 | 71.63 | 69.46 |
| qkvz | 1 | 44.05 | 42.54 (vLLM) | **0.97x** | 44.05 | 60.08 | 43.79 | 44.20 | 43.95 | 42.54 |
| qkvz | 2 | 44.14 | 42.92 (vLLM) | **0.97x** | 44.14 | 60.67 | 49.15 | 44.29 | 44.05 | 42.92 |
| qkvz | 4 | 44.38 | 43.82 (vLLM) | **0.99x** | 44.38 | 60.75 | 48.86 | 44.37 | 44.10 | 43.82 |
| qkvz | 8 | 44.71 | 44.20 (vLLM) | **0.99x** | 44.71 | 59.94 | 44.48 | 44.43 | 44.21 | 44.20 |
| qkvz | 16 | 44.96 | 44.57 (hipBLASLt) | **0.99x** | 44.96 | 59.52 | 50.17 | 44.57 | 44.66 | 44.61 |
| qkvz | 32 | 46.19 | 44.59 (hipBLASLt+TunableOp) | **0.96x** | 46.19 | 63.21 | 45.01 | 44.98 | 44.59 | 44.62 |
| qkvz | 64 | 54.68 | 45.55 (Inductor max-autotune) | **0.83x** | 54.68 | 66.03 | 45.55 | 45.59 | 45.62 | 45.61 |
| qkvz | 256 | 86.84 | 55.71 (hipBLASLt+TunableOp) | **0.64x** | -- | 86.84 | 89.11 | 65.55 | 55.71 | 56.37 |
| qkvz | 1024 | 210.65 | 208.26 (hipBLASLt+TunableOp) | **0.99x** | -- | 210.65 | 231.56 | 230.74 | 208.26 | 209.90 |
| ba | 1 | 3.81 | 2.68 (Inductor max-autotune) | **0.70x** | 3.81 | -- | 2.68 | 13.65 | 4.65 | 3.05 |
| ba | 2 | 3.79 | 3.14 (vLLM) | **0.83x** | 3.79 | -- | 7.32 | 13.60 | 6.93 | 3.14 |
| ba | 4 | 3.75 | 3.50 (vLLM) | **0.93x** | 3.75 | -- | 7.40 | 13.53 | 6.97 | 3.50 |
| ba | 8 | 3.76 | 6.95 (hipBLASLt+TunableOp) | **1.85x** | 3.76 | -- | 7.57 | 13.57 | 6.95 | 6.95 |
| ba | 16 | 3.76 | 6.97 (vLLM) | **1.85x** | 3.76 | -- | 7.63 | 13.44 | 6.98 | 6.97 |
| ba | 32 | 5.15 | 6.99 (vLLM) | **1.36x** | 5.15 | -- | 7.09 | 13.19 | 7.00 | 6.99 |
| ba | 64 | 7.84 | 6.99 (hipBLASLt+TunableOp) | **0.89x** | 7.84 | -- | 7.07 | 12.77 | 6.99 | 6.99 |
| ba | 256 | -- | 7.51 (Inductor max-autotune) | -- | -- | -- | 7.51 | 10.02 | 8.48 | 8.48 |
| ba | 1024 | -- | 7.67 (hipBLASLt+TunableOp) | -- | -- | -- | 10.20 | 10.27 | 7.67 | 7.67 |
| out | 1 | 17.18 | 15.91 (vLLM) | **0.93x** | 17.18 | 25.38 | 16.88 | 27.88 | 16.93 | 15.91 |
| out | 2 | 17.31 | 16.23 (vLLM) | **0.94x** | 17.31 | 25.55 | 18.68 | 27.98 | 16.96 | 16.23 |
| out | 4 | 17.42 | 16.84 (hipBLASLt+TunableOp) | **0.97x** | 17.42 | 25.57 | 18.77 | 28.09 | 16.84 | 17.14 |
| out | 8 | 17.49 | 17.07 (vLLM) | **0.98x** | 17.49 | 25.94 | 18.94 | 28.06 | 17.07 | 17.07 |
| out | 16 | 17.50 | 16.95 (hipBLASLt+TunableOp) | **0.97x** | 17.50 | 25.80 | 19.15 | 28.74 | 16.95 | 16.96 |
| out | 32 | 18.20 | 17.17 (hipBLASLt+TunableOp) | **0.94x** | 18.20 | 26.80 | 17.99 | 28.92 | 17.17 | 17.17 |
| out | 64 | 25.74 | 17.67 (vLLM) | **0.69x** | 25.74 | 26.94 | 20.35 | 29.06 | 17.68 | 17.67 |
| out | 256 | 32.05 | 24.00 (vLLM) | **0.75x** | -- | 32.05 | 35.01 | 34.71 | 24.21 | 24.00 |
| out | 1024 | 74.71 | 71.35 (vLLM) | **0.95x** | -- | 74.71 | 85.40 | 78.33 | 74.06 | 71.35 |
| shared_gate_up | 1 | 6.53 | 5.97 (vLLM) | **0.91x** | 6.53 | 15.77 | 6.14 | 14.17 | 7.84 | 5.97 |
| shared_gate_up | 2 | 6.57 | 6.09 (vLLM) | **0.93x** | 6.57 | 15.77 | 8.87 | 14.26 | 7.89 | 6.09 |
| shared_gate_up | 4 | 6.58 | 6.36 (vLLM) | **0.97x** | 6.58 | 15.76 | 8.92 | 14.19 | 7.85 | 6.36 |
| shared_gate_up | 8 | 6.58 | 7.93 (hipBLASLt+TunableOp) | **1.21x** | 6.58 | 15.68 | 9.05 | 14.49 | 7.93 | 7.95 |
| shared_gate_up | 16 | 6.63 | 7.73 (hipBLASLt+TunableOp) | **1.17x** | 6.63 | 16.17 | 9.22 | 14.51 | 7.73 | 7.73 |
| shared_gate_up | 32 | 7.19 | 7.43 (vLLM) | **1.03x** | 7.19 | 20.68 | 8.03 | 14.46 | 7.44 | 7.43 |
| shared_gate_up | 64 | 11.84 | 7.60 (vLLM) | **0.64x** | 11.84 | 20.69 | 8.27 | 13.68 | 7.61 | 7.60 |
| shared_gate_up | 256 | 20.92 | 10.96 (Inductor max-autotune) | **0.52x** | -- | 20.92 | 10.96 | 16.28 | 11.35 | 11.38 |
| shared_gate_up | 1024 | 26.08 | 24.53 (vLLM) | **0.94x** | -- | 26.08 | 30.73 | 25.28 | 24.95 | 24.53 |
| shared_down | 1 | 5.09 | 4.25 (Inductor max-autotune) | **0.83x** | 5.09 | 5.13 | 4.25 | 8.21 | 5.20 | 4.25 |
| shared_down | 2 | 5.10 | 4.34 (Inductor max-autotune) | **0.85x** | 5.10 | 5.13 | 4.34 | 8.24 | 5.20 | 4.51 |
| shared_down | 4 | 5.15 | 4.34 (Inductor max-autotune) | **0.84x** | 5.16 | 5.15 | 4.34 | 8.27 | 5.21 | 4.72 |
| shared_down | 8 | 5.15 | 4.31 (Inductor max-autotune) | **0.84x** | 5.15 | 5.20 | 4.31 | 8.26 | 5.20 | 5.20 |
| shared_down | 16 | 5.28 | 4.43 (Inductor max-autotune) | **0.84x** | 5.29 | 5.28 | 4.43 | 8.28 | 5.16 | 5.15 |
| shared_down | 32 | 5.78 | 4.51 (Inductor max-autotune) | **0.78x** | 5.78 | 6.24 | 4.51 | 8.33 | 5.70 | 5.70 |
| shared_down | 64 | 6.28 | 4.35 (Inductor max-autotune) | **0.69x** | 7.59 | 6.28 | 4.35 | 8.32 | 6.13 | 6.13 |
| shared_down | 256 | 6.70 | 5.97 (Inductor max-autotune) | **0.89x** | -- | 6.70 | 5.97 | 8.35 | 7.03 | 7.03 |
| shared_down | 1024 | 13.20 | 14.10 (Inductor max-autotune) | **1.07x** | -- | 13.20 | 14.10 | 15.79 | 15.53 | 15.51 |
| lm_head | 1 | 814.90 | 811.52 (vLLM) | **1.00x** | 814.90 | -- | 813.14 | 815.24 | 814.36 | 811.52 |
| lm_head | 2 | 817.10 | 811.68 (vLLM) | **0.99x** | 817.10 | -- | 817.24 | 817.18 | 819.38 | 811.68 |
| lm_head | 4 | 821.14 | 813.92 (vLLM) | **0.99x** | 821.14 | -- | 821.80 | 821.96 | 821.76 | 813.92 |
| lm_head | 8 | 829.12 | 829.32 (hipBLASLt+TunableOp) | **1.00x** | 829.12 | -- | 830.08 | 829.54 | 829.32 | 829.38 |
| lm_head | 16 | 840.88 | 840.30 (vLLM) | **1.00x** | 840.88 | -- | 841.38 | 842.72 | 840.42 | 840.30 |
| lm_head | 32 | 861.38 | 851.70 (Inductor max-autotune) | **0.99x** | 861.38 | -- | 851.70 | 861.00 | 859.86 | 858.26 |
| lm_head | 64 | 894.78 | 873.50 (Inductor max-autotune) | **0.98x** | 894.78 | -- | 873.50 | 885.16 | 874.28 | 876.90 |
| lm_head | 256 | -- | 1080.07 (hipBLASLt) | -- | -- | -- | 1122.73 | 1080.07 | 1124.01 | 1115.29 |

TunableOp: 62 of 62 (M, N, K) tuned here (M = 1 included), 159 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a: FlyDSL not available: its RDNA3 GEMM is for gfx11, and this is gfx1201

