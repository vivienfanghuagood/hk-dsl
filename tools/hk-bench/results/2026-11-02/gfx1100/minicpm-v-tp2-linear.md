### linear at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | FlyDSL | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 29.17 | 28.15 (Inductor max-autotune) | **0.96x** | 29.17 | 59.27 | -- | 28.15 | 47.97 | 30.10 | 29.79 |
| qkv | 2 | 29.31 | 30.14 (hipBLASLt+TunableOp) | **1.03x** | 29.31 | 61.22 | -- | 49.63 | 45.97 | 30.14 | 32.36 |
| qkv | 4 | 29.48 | 30.29 (hipBLASLt+TunableOp) | **1.03x** | 29.48 | 61.41 | -- | 50.19 | 46.94 | 30.29 | 35.35 |
| qkv | 8 | 29.92 | 30.41 (hipBLASLt+TunableOp) | **1.02x** | 29.92 | 62.12 | -- | 46.91 | 47.94 | 30.41 | 30.45 |
| qkv | 16 | 30.51 | 30.64 (vLLM) | **1.00x** | 30.51 | 64.56 | -- | 55.22 | 49.02 | 30.73 | 30.64 |
| qkv | 32 | 33.18 | 29.50 (FlyDSL) | **0.89x** | 33.18 | 68.25 | 29.50 | 60.19 | 55.26 | 30.92 | 30.91 |
| qkv | 64 | 42.22 | 33.14 (FlyDSL) | **0.79x** | 42.22 | 72.98 | 33.14 | 66.00 | 68.45 | 33.45 | 33.38 |
| qkv | 256 | 105.64 | 67.65 (vLLM) | **0.64x** | -- | 105.64 | 81.63 | 127.52 | 125.96 | 69.47 | 67.65 |
| qkv | 1024 | 232.95 | 228.61 (FlyDSL) | **0.98x** | -- | 232.95 | 228.61 | 246.97 | 247.07 | 240.04 | 240.47 |
| o | 1 | 15.86 | 14.04 (Inductor max-autotune) | **0.89x** | 15.86 | 32.41 | -- | 14.04 | 23.37 | 22.40 | 15.41 |
| o | 2 | 15.91 | 17.73 (vLLM) | **1.11x** | 15.91 | 32.51 | -- | 23.53 | 23.41 | 22.81 | 17.73 |
| o | 4 | 15.96 | 18.66 (vLLM) | **1.17x** | 15.96 | 32.60 | -- | 23.23 | 23.61 | 22.48 | 18.66 |
| o | 8 | 16.13 | 23.40 (hipBLASLt+TunableOp) | **1.45x** | 16.13 | 33.21 | -- | 23.54 | 23.98 | 23.40 | 23.53 |
| o | 16 | 16.54 | 22.73 (hipBLASLt+TunableOp) | **1.37x** | 16.54 | 32.39 | -- | 27.32 | 24.35 | 22.73 | 22.73 |
| o | 32 | 18.42 | 19.01 (FlyDSL) | **1.03x** | 18.42 | 47.91 | 19.01 | 29.51 | 24.38 | 22.83 | 22.84 |
| o | 64 | 24.70 | 20.90 (FlyDSL) | **0.85x** | 24.70 | 48.24 | 20.90 | 31.01 | 41.53 | 25.05 | 25.03 |
| o | 256 | 51.82 | 37.35 (hipBLASLt+TunableOp) | **0.72x** | -- | 51.82 | 42.44 | 54.49 | 51.27 | 37.35 | 37.44 |
| o | 1024 | 120.91 | 120.75 (hipBLASLt+TunableOp) | **1.00x** | -- | 120.91 | 137.81 | 126.04 | 126.20 | 120.75 | 120.82 |
| qkvz | 1 | 37.23 | 36.31 (Inductor max-autotune) | **0.98x** | 37.23 | 84.73 | -- | 36.31 | 54.18 | 36.81 | 37.81 |
| qkvz | 2 | 37.40 | 36.85 (hipBLASLt+TunableOp) | **0.98x** | 37.40 | 86.58 | -- | 53.21 | 53.25 | 36.85 | 43.83 |
| qkvz | 4 | 37.55 | 37.11 (hipBLASLt+TunableOp) | **0.99x** | 37.55 | 87.60 | -- | 54.05 | 54.07 | 37.11 | 45.23 |
| qkvz | 8 | 38.06 | 37.28 (vLLM) | **0.98x** | 38.06 | 91.35 | -- | 67.18 | 54.95 | 37.31 | 37.28 |
| qkvz | 16 | 38.84 | 37.56 (hipBLASLt+TunableOp) | **0.97x** | 38.84 | 89.38 | -- | 56.56 | 56.89 | 37.56 | 37.56 |
| qkvz | 32 | 41.27 | 36.88 (FlyDSL) | **0.89x** | 41.27 | 94.24 | 36.88 | 79.22 | 79.28 | 37.84 | 37.87 |
| qkvz | 64 | 50.34 | 38.56 (FlyDSL) | **0.77x** | 50.34 | 102.12 | 38.56 | 107.49 | 109.16 | 39.19 | 39.20 |
| qkvz | 256 | 156.34 | 94.84 (vLLM) | **0.61x** | -- | 156.34 | 104.30 | 193.93 | 195.63 | 95.13 | 94.84 |
| qkvz | 1024 | 300.66 | 299.13 (FlyDSL) | **0.99x** | -- | 300.66 | 299.13 | 326.88 | 326.66 | 312.35 | 312.70 |
| ba | 1 | 4.67 | 3.42 (Inductor max-autotune) | **0.73x** | 4.67 | -- | -- | 3.42 | 10.89 | 6.15 | 3.82 |
| ba | 2 | 4.67 | 4.02 (vLLM) | **0.86x** | 4.67 | -- | -- | 10.15 | 10.66 | 10.57 | 4.02 |
| ba | 4 | 4.67 | 4.31 (vLLM) | **0.92x** | 4.67 | -- | -- | 10.08 | 10.64 | 10.31 | 4.31 |
| ba | 8 | 4.68 | 10.31 (vLLM) | **2.20x** | 4.68 | -- | -- | 11.79 | 10.90 | 10.31 | 10.31 |
| ba | 16 | 4.69 | 10.44 (hipBLASLt+TunableOp) | **2.23x** | 4.69 | -- | -- | 10.72 | 10.72 | 10.44 | 10.44 |
| ba | 32 | 6.03 | 10.16 (vLLM) | **1.68x** | 6.03 | -- | 16.34 | 13.26 | 10.54 | 10.16 | 10.16 |
| ba | 64 | 8.46 | 10.31 (vLLM) | **1.22x** | 8.46 | -- | 18.65 | 13.31 | 10.61 | 10.32 | 10.31 |
| ba | 256 | -- | 10.18 (hipBLASLt+TunableOp) | -- | -- | -- | 18.03 | 10.18 | 10.21 | 10.18 | 10.18 |
| ba | 1024 | -- | 9.95 (vLLM) | -- | -- | -- | 15.64 | 15.25 | 13.08 | 9.95 | 9.95 |
| out | 1 | 15.78 | 14.00 (Inductor max-autotune) | **0.89x** | 15.78 | 32.41 | -- | 14.00 | 23.39 | 22.33 | 15.39 |
| out | 2 | 15.86 | 17.75 (vLLM) | **1.12x** | 15.86 | 32.52 | -- | 23.50 | 23.54 | 23.02 | 17.75 |
| out | 4 | 15.91 | 18.63 (vLLM) | **1.17x** | 15.91 | 32.64 | -- | 23.24 | 23.65 | 22.45 | 18.63 |
| out | 8 | 16.08 | 23.18 (hipBLASLt+TunableOp) | **1.44x** | 16.08 | 33.07 | -- | 23.71 | 23.84 | 23.18 | 23.29 |
| out | 16 | 16.47 | 22.67 (vLLM) | **1.38x** | 16.47 | 32.42 | -- | 27.27 | 24.35 | 22.69 | 22.67 |
| out | 32 | 18.41 | 18.95 (FlyDSL) | **1.03x** | 18.41 | 47.97 | 18.95 | 29.41 | 24.47 | 22.76 | 22.84 |
| out | 64 | 24.45 | 20.96 (FlyDSL) | **0.86x** | 24.45 | 48.20 | 20.96 | 31.09 | 41.71 | 25.07 | 25.06 |
| out | 256 | 52.13 | 37.42 (vLLM) | **0.72x** | -- | 52.13 | 42.85 | 55.49 | 50.97 | 37.51 | 37.42 |
| out | 1024 | 121.22 | 121.02 (vLLM) | **1.00x** | -- | 121.22 | 138.20 | 126.03 | 126.32 | 121.20 | 121.02 |
| shared_gate_up | 1 | 6.68 | 5.99 (Inductor max-autotune) | **0.90x** | 6.68 | 24.19 | -- | 5.99 | 15.10 | 11.33 | 6.68 |
| shared_gate_up | 2 | 6.69 | 6.78 (vLLM) | **1.01x** | 6.69 | 24.37 | -- | 13.09 | 12.72 | 12.81 | 6.78 |
| shared_gate_up | 4 | 6.70 | 7.35 (vLLM) | **1.10x** | 6.70 | 24.41 | -- | 13.08 | 12.65 | 12.80 | 7.35 |
| shared_gate_up | 8 | 6.72 | 12.40 (Inductor max-autotune) | **1.85x** | 6.72 | 23.34 | -- | 12.40 | 12.75 | 12.91 | 12.86 |
| shared_gate_up | 16 | 6.74 | 12.49 (hipBLASLt+TunableOp) | **1.85x** | 6.74 | 23.18 | -- | 14.98 | 12.66 | 12.49 | 12.50 |
| shared_gate_up | 32 | 8.25 | 13.01 (vLLM) | **1.58x** | 8.25 | 41.29 | 16.96 | 16.24 | 13.71 | 13.02 | 13.01 |
| shared_gate_up | 64 | 10.54 | 12.77 (hipBLASLt) | **1.21x** | 10.54 | 41.36 | 17.00 | 16.50 | 12.77 | 13.15 | 13.13 |
| shared_gate_up | 256 | 41.86 | 18.60 (hipBLASLt) | **0.44x** | -- | 41.86 | 20.25 | 18.66 | 18.60 | 20.40 | 20.43 |
| shared_gate_up | 1024 | 44.50 | 37.16 (hipBLASLt+TunableOp) | **0.83x** | -- | 44.50 | 40.49 | 41.47 | 41.48 | 37.16 | 37.19 |
| shared_down | 1 | 5.82 | 4.67 (Inductor max-autotune) | **0.80x** | 5.82 | 6.77 | -- | 4.67 | 8.57 | 5.92 | 4.77 |
| shared_down | 2 | 5.83 | 5.12 (vLLM) | **0.88x** | 5.83 | 6.78 | -- | 5.65 | 7.60 | 5.93 | 5.12 |
| shared_down | 4 | 5.82 | 5.27 (Inductor max-autotune) | **0.91x** | 5.82 | 6.79 | -- | 5.27 | 7.61 | 5.93 | 5.70 |
| shared_down | 8 | 5.84 | 5.96 (hipBLASLt+TunableOp) | **1.02x** | 5.84 | 6.74 | -- | 5.97 | 7.57 | 5.96 | 5.96 |
| shared_down | 16 | 5.87 | 5.34 (Inductor max-autotune) | **0.91x** | 5.87 | 6.93 | -- | 5.34 | 7.44 | 6.04 | 6.03 |
| shared_down | 32 | 5.63 | 5.67 (hipBLASLt+TunableOp) | **1.01x** | 5.63 | 9.84 | 6.64 | 6.25 | 7.02 | 5.67 | 5.67 |
| shared_down | 64 | 8.37 | 5.90 (hipBLASLt+TunableOp) | **0.70x** | 8.37 | 9.94 | 6.73 | 6.10 | 14.74 | 5.90 | 5.90 |
| shared_down | 256 | 10.32 | 8.13 (vLLM) | **0.79x** | -- | 10.32 | 11.46 | 9.08 | 13.72 | 8.14 | 8.13 |
| shared_down | 1024 | 20.75 | 19.85 (hipBLASLt+TunableOp) | **0.96x** | -- | 20.75 | 29.38 | 23.68 | 25.34 | 19.85 | 19.87 |
| lm_head | 1 | 629.22 | 621.84 (Inductor max-autotune) | **0.99x** | 629.22 | -- | -- | 621.84 | 651.36 | 629.00 | 625.06 |
| lm_head | 2 | 632.60 | 628.78 (vLLM) | **0.99x** | 632.60 | -- | -- | 655.02 | 655.06 | 632.96 | 628.78 |
| lm_head | 4 | 636.80 | 638.48 (hipBLASLt+TunableOp) | **1.00x** | 636.80 | -- | -- | 659.36 | 659.48 | 638.48 | 641.16 |
| lm_head | 8 | 647.46 | 645.60 (vLLM) | **1.00x** | 647.46 | -- | -- | 672.16 | 672.28 | 645.80 | 645.60 |
| lm_head | 16 | 664.00 | 655.16 (hipBLASLt+TunableOp) | **0.99x** | 664.00 | -- | -- | 689.46 | 687.76 | 655.16 | 656.04 |
| lm_head | 32 | 691.24 | 672.62 (vLLM) | **0.97x** | 691.24 | -- | 700.54 | 708.80 | 1281.05 | 672.64 | 672.62 |
| lm_head | 64 | 737.06 | 708.94 (vLLM) | **0.96x** | 737.06 | -- | 714.18 | 843.24 | 878.52 | 709.36 | 708.94 |
| lm_head | 256 | -- | 1592.25 (vLLM) | -- | -- | -- | 1880.01 | 1618.13 | 1638.09 | 1595.49 | 1592.25 |

TunableOp: 62 of 62 (M, N, K) tuned here (M = 1 included), 101 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- FlyDSL, by row:
  - qkv M=32, o M=32, qkvz M=32, ba M=32, ba M=64, ba M=256, ba M=1024, out M=32, shared_gate_up M=32, shared_gate_up M=64, shared_down M=32, lm_head M=32: FlyDSL GEMM, tile 32x32x64w1
  - qkv M=64, qkv M=256, o M=64, o M=256, o M=1024, qkvz M=64, qkvz M=256, out M=64, out M=256, out M=1024, shared_gate_up M=256, shared_gate_up M=1024, shared_down M=64, shared_down M=256, shared_down M=1024, lm_head M=64, lm_head M=256: FlyDSL GEMM, tile 64x64x64w4
  - qkv M=1024, qkvz M=1024: FlyDSL GEMM, tile 128x128x32w4
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a at some shapes: no tile takes them
