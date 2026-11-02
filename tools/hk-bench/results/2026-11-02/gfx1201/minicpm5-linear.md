### linear at openbmb/MiniCPM5-2B (TP1) on AMD Radeon AI PRO R9700 (gfx1201)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+ga50259bb3, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 20.64 | 19.37 (vLLM) | **0.94x** | 20.64 | 28.84 | 20.42 | 26.80 | 20.71 | 19.37 |
| qkv | 2 | 20.67 | 19.58 (vLLM) | **0.95x** | 20.67 | 28.90 | 22.39 | 26.77 | 20.86 | 19.58 |
| qkv | 4 | 20.64 | 20.47 (vLLM) | **0.99x** | 20.64 | 29.17 | 22.50 | 29.43 | 20.88 | 20.47 |
| qkv | 8 | 20.73 | 20.95 (hipBLASLt+TunableOp) | **1.01x** | 20.73 | 28.90 | 22.41 | 29.65 | 20.95 | 20.95 |
| qkv | 16 | 20.89 | 21.04 (vLLM) | **1.01x** | 20.89 | 29.98 | 22.54 | 29.76 | 21.07 | 21.04 |
| qkv | 32 | 21.88 | 20.82 (hipBLASLt+TunableOp) | **0.95x** | 21.88 | 30.37 | 21.77 | 27.04 | 20.82 | 20.82 |
| qkv | 64 | 29.90 | 21.13 (vLLM) | **0.71x** | 29.90 | 31.04 | 26.76 | 30.23 | 21.14 | 21.13 |
| qkv | 256 | 40.49 | 28.70 (hipBLASLt+TunableOp) | **0.71x** | -- | 40.49 | 44.24 | 40.64 | 28.70 | 28.89 |
| qkv | 1024 | 88.13 | 86.96 (hipBLASLt+TunableOp) | **0.99x** | -- | 88.13 | 89.19 | 89.17 | 86.96 | 87.23 |
| o | 1 | 17.39 | 16.02 (vLLM) | **0.92x** | 17.39 | 25.84 | 17.04 | 28.81 | 17.08 | 16.02 |
| o | 2 | 17.36 | 16.41 (vLLM) | **0.94x** | 17.36 | 25.74 | 18.87 | 28.96 | 17.05 | 16.41 |
| o | 4 | 17.49 | 17.00 (hipBLASLt+TunableOp) | **0.97x** | 17.49 | 25.97 | 18.96 | 28.92 | 17.00 | 17.35 |
| o | 8 | 17.49 | 17.13 (vLLM) | **0.98x** | 17.49 | 25.94 | 19.14 | 28.66 | 17.14 | 17.13 |
| o | 16 | 17.58 | 17.00 (vLLM) | **0.97x** | 17.58 | 25.95 | 19.30 | 28.71 | 17.01 | 17.00 |
| o | 32 | 18.20 | 17.15 (vLLM) | **0.94x** | 18.20 | 26.62 | 18.05 | 28.96 | 17.16 | 17.15 |
| o | 64 | 25.76 | 17.72 (hipBLASLt+TunableOp) | **0.69x** | 25.76 | 27.01 | 20.33 | 28.96 | 17.72 | 17.75 |
| o | 256 | 32.37 | 23.93 (vLLM) | **0.74x** | -- | 32.37 | 34.53 | 33.74 | 24.02 | 23.93 |
| o | 1024 | 71.81 | 70.24 (hipBLASLt+TunableOp) | **0.98x** | -- | 71.81 | 84.21 | 72.69 | 70.24 | 71.71 |
| gate_up | 1 | 84.06 | 82.26 (vLLM) | **0.98x** | 84.06 | 115.46 | 83.96 | 85.46 | 84.20 | 82.26 |
| gate_up | 2 | 84.12 | 82.94 (vLLM) | **0.99x** | 84.12 | 115.71 | 89.22 | 85.43 | 84.18 | 82.94 |
| gate_up | 4 | 84.49 | 83.70 (vLLM) | **0.99x** | 84.49 | 116.62 | 89.19 | 85.68 | 84.31 | 83.70 |
| gate_up | 8 | 85.28 | 85.88 (hipBLASLt) | **1.01x** | 85.28 | 114.59 | 87.59 | 85.88 | 86.25 | 86.39 |
| gate_up | 16 | 86.04 | 86.37 (hipBLASLt) | **1.00x** | 86.04 | 120.16 | 88.38 | 86.37 | 86.92 | 87.19 |
| gate_up | 32 | 88.07 | 86.54 (hipBLASLt+TunableOp) | **0.98x** | 88.07 | 148.27 | 89.80 | 87.01 | 86.54 | 86.67 |
| gate_up | 64 | 98.19 | 88.92 (vLLM) | **0.91x** | 98.19 | 155.06 | 88.97 | 89.37 | 89.12 | 88.92 |
| gate_up | 256 | 165.53 | 109.82 (hipBLASLt+TunableOp) | **0.66x** | -- | 165.53 | 161.56 | 115.94 | 109.82 | 110.65 |
| gate_up | 1024 | 393.05 | 400.49 (hipBLASLt+TunableOp) | **1.02x** | -- | 393.05 | 408.26 | 400.88 | 400.49 | 405.73 |
| down | 1 | 43.61 | 42.69 (vLLM) | **0.98x** | 43.61 | 60.59 | 43.67 | 64.72 | 43.32 | 42.69 |
| down | 2 | 43.65 | 43.06 (vLLM) | **0.99x** | 43.65 | 60.08 | 46.86 | 63.76 | 43.92 | 43.06 |
| down | 4 | 43.60 | 44.08 (hipBLASLt+TunableOp) | **1.01x** | 43.60 | 60.25 | 46.96 | 64.24 | 44.08 | 44.35 |
| down | 8 | 43.76 | 43.52 (vLLM) | **0.99x** | 43.76 | 60.30 | 47.36 | 64.91 | 43.54 | 43.52 |
| down | 16 | 44.04 | 43.72 (hipBLASLt+TunableOp) | **0.99x** | 44.04 | 60.81 | 47.49 | 65.10 | 43.72 | 43.74 |
| down | 32 | 45.65 | 44.00 (vLLM) | **0.96x** | 45.65 | 61.68 | 45.24 | 65.53 | 44.05 | 44.00 |
| down | 64 | 55.83 | 44.91 (hipBLASLt+TunableOp) | **0.80x** | 55.83 | 63.19 | 49.25 | 67.88 | 44.91 | 44.95 |
| down | 256 | 74.00 | 64.06 (vLLM) | **0.87x** | -- | 74.00 | 78.90 | 77.54 | 64.89 | 64.06 |
| down | 1024 | 203.02 | 198.45 (vLLM) | **0.98x** | -- | 203.02 | 213.99 | 215.45 | 209.89 | 198.45 |
| lm_head | 1 | 856.62 | 852.46 (vLLM) | **0.99x** | 856.62 | 898.46 | 854.50 | 856.34 | 854.18 | 852.46 |
| lm_head | 2 | 858.36 | 853.08 (vLLM) | **0.99x** | 858.36 | 896.22 | 858.58 | 858.52 | 858.54 | 853.08 |
| lm_head | 4 | 863.08 | 855.42 (vLLM) | **0.99x** | 863.08 | 907.56 | 863.40 | 863.50 | 863.12 | 855.42 |
| lm_head | 8 | 871.68 | 871.32 (vLLM) | **1.00x** | 871.68 | 912.74 | 872.46 | 872.14 | 871.74 | 871.32 |
| lm_head | 16 | 885.72 | 884.28 (hipBLASLt+TunableOp) | **1.00x** | 885.72 | 933.00 | 886.64 | 886.32 | 884.28 | 884.36 |
| lm_head | 32 | 904.06 | 897.78 (Inductor max-autotune) | **0.99x** | 904.06 | 1059.34 | 897.78 | 906.06 | 901.52 | 903.04 |
| lm_head | 64 | 940.20 | 908.80 (Inductor max-autotune) | **0.97x** | 940.20 | 1101.40 | 908.80 | 929.92 | 917.46 | 917.96 |
| lm_head | 256 | 1119.10 | 1097.04 (Inductor max-autotune) | **0.98x** | -- | 1119.10 | 1097.04 | 1116.08 | 1110.42 | 1109.28 |

TunableOp: 44 of 44 (M, N, K) tuned here (M = 1 included), 44 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a: FlyDSL not available: its RDNA3 GEMM is for gfx11, and this is gfx1201
