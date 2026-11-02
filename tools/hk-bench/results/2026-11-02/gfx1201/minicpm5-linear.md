### linear at openbmb/MiniCPM5-2B (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 20.66 | 19.34 (vLLM) | **0.94x** | 20.66 | 28.84 | 20.35 | 26.73 | 20.67 | 19.34 |
| qkv | 2 | 20.72 | 19.56 (vLLM) | **0.94x** | 20.72 | 29.00 | 22.41 | 26.80 | 20.86 | 19.56 |
| qkv | 4 | 20.73 | 20.50 (vLLM) | **0.99x** | 20.73 | 29.24 | 22.51 | 29.56 | 20.93 | 20.50 |
| qkv | 8 | 20.83 | 21.00 (hipBLASLt+TunableOp) | **1.01x** | 20.83 | 29.11 | 22.44 | 29.68 | 21.00 | 21.02 |
| qkv | 16 | 20.98 | 21.06 (hipBLASLt+TunableOp) | **1.00x** | 20.98 | 29.96 | 22.57 | 29.84 | 21.06 | 21.07 |
| qkv | 32 | 21.95 | 20.85 (vLLM) | **0.95x** | 21.95 | 30.89 | 21.84 | 27.05 | 20.85 | 20.85 |
| qkv | 64 | 29.99 | 21.22 (vLLM) | **0.71x** | 29.99 | 31.64 | 27.00 | 30.48 | 21.22 | 21.22 |
| qkv | 256 | 40.83 | 29.46 (vLLM) | **0.72x** | -- | 40.83 | 44.08 | 40.69 | 29.65 | 29.46 |
| qkv | 1024 | 89.08 | 87.40 (vLLM) | **0.98x** | -- | 89.08 | 94.93 | 90.10 | 88.43 | 87.40 |
| o | 1 | 17.39 | 15.99 (vLLM) | **0.92x** | 17.39 | 25.78 | 17.00 | 28.64 | 17.08 | 15.99 |
| o | 2 | 17.44 | 16.33 (vLLM) | **0.94x** | 17.44 | 25.75 | 18.89 | 28.77 | 17.09 | 16.33 |
| o | 4 | 17.51 | 16.93 (hipBLASLt+TunableOp) | **0.97x** | 17.51 | 25.83 | 18.93 | 29.17 | 16.93 | 17.27 |
| o | 8 | 17.56 | 17.19 (vLLM) | **0.98x** | 17.56 | 25.96 | 19.07 | 28.92 | 17.20 | 17.19 |
| o | 16 | 17.65 | 17.04 (hipBLASLt+TunableOp) | **0.97x** | 17.65 | 25.92 | 19.22 | 28.73 | 17.04 | 17.06 |
| o | 32 | 18.31 | 17.21 (hipBLASLt+TunableOp) | **0.94x** | 18.31 | 27.01 | 18.12 | 29.43 | 17.21 | 17.22 |
| o | 64 | 25.76 | 17.81 (vLLM) | **0.69x** | 25.76 | 26.84 | 20.38 | 29.19 | 17.82 | 17.81 |
| o | 256 | 32.06 | 24.16 (hipBLASLt+TunableOp) | **0.75x** | -- | 32.06 | 34.06 | 34.45 | 24.16 | 24.58 |
| o | 1024 | 71.58 | 72.21 (hipBLASLt) | **1.01x** | -- | 71.58 | 86.12 | 72.21 | 74.90 | 72.54 |
| gate_up | 1 | 84.02 | 82.27 (vLLM) | **0.98x** | 84.02 | 115.64 | 84.49 | 85.36 | 84.17 | 82.27 |
| gate_up | 2 | 84.18 | 82.77 (vLLM) | **0.98x** | 84.18 | 116.39 | 89.13 | 85.58 | 84.33 | 82.77 |
| gate_up | 4 | 84.46 | 83.60 (vLLM) | **0.99x** | 84.46 | 116.88 | 89.17 | 85.80 | 84.31 | 83.60 |
| gate_up | 8 | 85.23 | 86.09 (hipBLASLt) | **1.01x** | 85.23 | 115.36 | 87.31 | 86.09 | 86.16 | 86.44 |
| gate_up | 16 | 86.14 | 86.57 (hipBLASLt) | **1.00x** | 86.14 | 120.30 | 88.39 | 86.57 | 86.73 | 87.03 |
| gate_up | 32 | 88.24 | 86.98 (vLLM) | **0.99x** | 88.24 | 148.24 | 89.89 | 87.25 | 87.02 | 86.98 |
| gate_up | 64 | 98.53 | 88.78 (vLLM) | **0.90x** | 98.53 | 152.74 | 89.38 | 89.49 | 88.89 | 88.78 |
| gate_up | 256 | 162.95 | 108.74 (vLLM) | **0.67x** | -- | 162.95 | 158.91 | 114.57 | 109.46 | 108.74 |
| gate_up | 1024 | 410.13 | 405.01 (hipBLASLt) | **0.99x** | -- | 410.13 | 410.61 | 405.01 | 410.11 | 409.35 |
| down | 1 | 43.58 | 42.69 (vLLM) | **0.98x** | 43.58 | 60.27 | 43.64 | 64.46 | 43.30 | 42.69 |
| down | 2 | 43.65 | 43.12 (vLLM) | **0.99x** | 43.65 | 60.27 | 46.89 | 64.23 | 43.96 | 43.12 |
| down | 4 | 43.70 | 44.10 (hipBLASLt+TunableOp) | **1.01x** | 43.70 | 60.65 | 47.01 | 64.55 | 44.10 | 44.39 |
| down | 8 | 43.86 | 43.63 (vLLM) | **0.99x** | 43.86 | 60.48 | 47.35 | 64.81 | 43.63 | 43.63 |
| down | 16 | 44.10 | 43.82 (hipBLASLt+TunableOp) | **0.99x** | 44.10 | 60.71 | 47.69 | 65.51 | 43.82 | 43.84 |
| down | 32 | 45.75 | 44.08 (vLLM) | **0.96x** | 45.75 | 62.49 | 45.20 | 66.40 | 44.11 | 44.08 |
| down | 64 | 56.34 | 44.80 (hipBLASLt+TunableOp) | **0.80x** | 56.34 | 63.16 | 49.19 | 67.32 | 44.80 | 44.80 |
| down | 256 | 71.82 | 64.56 (hipBLASLt+TunableOp) | **0.90x** | -- | 71.82 | 78.06 | 75.31 | 64.56 | 64.91 |
| down | 1024 | 211.44 | 203.29 (hipBLASLt+TunableOp) | **0.96x** | -- | 211.44 | 206.65 | 209.88 | 203.29 | 206.15 |
| lm_head | 1 | 855.10 | 846.02 (vLLM) | **0.99x** | 855.10 | 896.92 | 851.54 | 856.02 | 854.58 | 846.02 |
| lm_head | 2 | 858.30 | 852.88 (vLLM) | **0.99x** | 858.30 | 896.50 | 858.68 | 858.64 | 858.62 | 852.88 |
| lm_head | 4 | 863.06 | 854.48 (vLLM) | **0.99x** | 863.06 | 908.26 | 863.42 | 863.42 | 863.32 | 854.48 |
| lm_head | 8 | 871.56 | 870.66 (vLLM) | **1.00x** | 871.56 | 913.80 | 872.56 | 872.42 | 870.84 | 870.66 |
| lm_head | 16 | 884.74 | 883.36 (hipBLASLt+TunableOp) | **1.00x** | 884.74 | 936.28 | 885.66 | 885.92 | 883.36 | 884.20 |
| lm_head | 32 | 903.92 | 895.92 (Inductor max-autotune) | **0.99x** | 903.92 | 1056.76 | 895.92 | 905.54 | 900.80 | 900.36 |
| lm_head | 64 | 938.06 | 907.34 (Inductor max-autotune) | **0.97x** | 938.06 | 1101.02 | 907.34 | 929.12 | 916.66 | 915.34 |
| lm_head | 256 | 1157.80 | 1119.64 (hipBLASLt) | **0.97x** | -- | 1157.80 | 1169.86 | 1119.64 | 1149.14 | 1149.86 |

TunableOp: 44 of 44 (M, N, K) tuned here (M = 1 included), 159 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a: FlyDSL not available: its RDNA3 GEMM is for gfx11, and this is gfx1201

