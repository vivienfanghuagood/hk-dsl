### linear at openbmb/MiniCPM-V-4.7-35B-A3B (TP2) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | FlyDSL | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 29.31 | 28.25 (Inductor max-autotune) | **0.96x** | 29.31 | 57.77 | -- | 28.25 | 48.08 | 30.14 | 29.84 |
| qkv | 2 | 29.43 | 30.24 (hipBLASLt+TunableOp) | **1.03x** | 29.43 | 59.33 | -- | 49.38 | 46.06 | 30.24 | 32.36 |
| qkv | 4 | 29.68 | 30.36 (hipBLASLt+TunableOp) | **1.02x** | 29.68 | 62.24 | -- | 49.50 | 46.90 | 30.36 | 35.48 |
| qkv | 8 | 30.05 | 30.50 (vLLM) | **1.01x** | 30.05 | 64.16 | -- | 46.96 | 48.06 | 30.52 | 30.50 |
| qkv | 16 | 30.62 | 30.66 (hipBLASLt+TunableOp) | **1.00x** | 30.62 | 65.23 | -- | 55.27 | 48.95 | 30.66 | 30.70 |
| qkv | 32 | 33.21 | 29.66 (FlyDSL) | **0.89x** | 33.21 | 69.78 | 29.66 | 58.25 | 55.87 | 30.93 | 30.89 |
| qkv | 64 | 42.40 | 32.19 (FlyDSL) | **0.76x** | 42.40 | 74.54 | 32.19 | 71.96 | 71.12 | 33.68 | 33.72 |
| qkv | 256 | 103.51 | 68.12 (hipBLASLt+TunableOp) | **0.66x** | -- | 103.51 | 82.34 | 125.72 | 127.70 | 68.12 | 71.06 |
| qkv | 1024 | 233.31 | 226.79 (FlyDSL) | **0.97x** | -- | 233.31 | 226.79 | 245.81 | 245.86 | 238.43 | 237.64 |
| o | 1 | 15.83 | 14.01 (Inductor max-autotune) | **0.89x** | 15.83 | 32.37 | -- | 14.01 | 23.41 | 22.42 | 15.41 |
| o | 2 | 15.93 | 17.73 (vLLM) | **1.11x** | 15.93 | 32.55 | -- | 23.48 | 23.61 | 23.09 | 17.73 |
| o | 4 | 15.94 | 18.63 (vLLM) | **1.17x** | 15.94 | 32.72 | -- | 23.26 | 23.65 | 22.47 | 18.63 |
| o | 8 | 16.13 | 23.28 (vLLM) | **1.44x** | 16.13 | 33.25 | -- | 23.80 | 24.02 | 23.35 | 23.28 |
| o | 16 | 16.46 | 22.78 (hipBLASLt+TunableOp) | **1.38x** | 16.46 | 32.31 | -- | 27.48 | 24.41 | 22.78 | 22.80 |
| o | 32 | 18.43 | 19.04 (FlyDSL) | **1.03x** | 18.43 | 47.99 | 19.04 | 29.39 | 24.42 | 22.86 | 22.89 |
| o | 64 | 24.82 | 20.97 (FlyDSL) | **0.84x** | 24.82 | 48.34 | 20.97 | 31.17 | 41.86 | 25.08 | 25.12 |
| o | 256 | 51.85 | 37.43 (hipBLASLt+TunableOp) | **0.72x** | -- | 51.85 | 42.47 | 54.28 | 50.96 | 37.43 | 37.49 |
| o | 1024 | 121.03 | 120.78 (vLLM) | **1.00x** | -- | 121.03 | 137.76 | 125.53 | 125.71 | 120.96 | 120.78 |
| qkvz | 1 | 37.32 | 36.37 (Inductor max-autotune) | **0.97x** | 37.32 | 85.54 | -- | 36.37 | 54.37 | 36.90 | 37.87 |
| qkvz | 2 | 37.57 | 36.94 (hipBLASLt+TunableOp) | **0.98x** | 37.57 | 85.13 | -- | 53.47 | 53.61 | 36.94 | 44.09 |
| qkvz | 4 | 37.81 | 37.17 (hipBLASLt+TunableOp) | **0.98x** | 37.81 | 87.51 | -- | 54.20 | 54.34 | 37.17 | 45.55 |
| qkvz | 8 | 38.28 | 37.33 (hipBLASLt+TunableOp) | **0.97x** | 38.28 | 91.64 | -- | 67.86 | 55.28 | 37.33 | 37.36 |
| qkvz | 16 | 38.98 | 37.60 (vLLM) | **0.96x** | 38.98 | 90.91 | -- | 56.88 | 57.27 | 37.62 | 37.60 |
| qkvz | 32 | 41.38 | 37.23 (FlyDSL) | **0.90x** | 41.38 | 94.16 | 37.23 | 79.70 | 78.27 | 37.81 | 37.84 |
| qkvz | 64 | 50.20 | 38.78 (FlyDSL) | **0.77x** | 50.20 | 102.19 | 38.78 | 104.64 | 105.51 | 39.51 | 39.49 |
| qkvz | 256 | 156.06 | 93.34 (hipBLASLt+TunableOp) | **0.60x** | -- | 156.06 | 104.30 | 197.19 | 192.82 | 93.34 | 93.39 |
| qkvz | 1024 | 303.16 | 297.40 (FlyDSL) | **0.98x** | -- | 303.16 | 297.40 | 328.19 | 327.66 | 312.49 | 310.19 |
| ba | 1 | 4.67 | 3.42 (Inductor max-autotune) | **0.73x** | 4.67 | -- | -- | 3.42 | 10.89 | 6.15 | 3.82 |
| ba | 2 | 4.67 | 4.01 (vLLM) | **0.86x** | 4.67 | -- | -- | 10.16 | 10.66 | 10.57 | 4.01 |
| ba | 4 | 4.67 | 4.32 (vLLM) | **0.92x** | 4.67 | -- | -- | 10.08 | 10.64 | 10.31 | 4.32 |
| ba | 8 | 4.68 | 10.31 (hipBLASLt+TunableOp) | **2.20x** | 4.68 | -- | -- | 11.74 | 10.92 | 10.31 | 10.31 |
| ba | 16 | 4.69 | 10.44 (hipBLASLt+TunableOp) | **2.23x** | 4.69 | -- | -- | 10.72 | 10.72 | 10.44 | 10.44 |
| ba | 32 | 6.03 | 10.15 (vLLM) | **1.68x** | 6.03 | -- | 16.32 | 13.26 | 10.55 | 10.16 | 10.15 |
| ba | 64 | 8.43 | 10.52 (hipBLASLt) | **1.25x** | 8.43 | -- | 18.65 | 14.02 | 10.52 | 10.96 | 10.97 |
| ba | 256 | -- | 10.47 (hipBLASLt) | -- | -- | -- | 18.03 | 10.47 | 10.47 | 10.47 | 10.47 |
| ba | 1024 | -- | 10.18 (vLLM) | -- | -- | -- | 15.85 | 15.90 | 13.33 | 10.19 | 10.18 |
| out | 1 | 15.84 | 14.03 (Inductor max-autotune) | **0.89x** | 15.84 | 32.61 | -- | 14.03 | 23.47 | 22.35 | 15.42 |
| out | 2 | 15.93 | 17.77 (vLLM) | **1.11x** | 15.93 | 32.61 | -- | 23.58 | 23.42 | 22.84 | 17.77 |
| out | 4 | 15.95 | 18.63 (vLLM) | **1.17x** | 15.95 | 32.75 | -- | 23.22 | 23.67 | 22.42 | 18.63 |
| out | 8 | 16.14 | 23.28 (hipBLASLt+TunableOp) | **1.44x** | 16.14 | 33.30 | -- | 23.78 | 24.02 | 23.28 | 23.33 |
| out | 16 | 16.55 | 22.68 (vLLM) | **1.37x** | 16.55 | 32.43 | -- | 27.38 | 24.47 | 22.69 | 22.68 |
| out | 32 | 18.47 | 18.96 (FlyDSL) | **1.03x** | 18.47 | 48.03 | 18.96 | 29.57 | 24.47 | 22.78 | 22.83 |
| out | 64 | 24.52 | 21.05 (FlyDSL) | **0.86x** | 24.52 | 48.26 | 21.05 | 31.00 | 41.65 | 25.06 | 25.08 |
| out | 256 | 52.01 | 37.45 (vLLM) | **0.72x** | -- | 52.01 | 42.76 | 55.02 | 51.22 | 37.46 | 37.45 |
| out | 1024 | 121.44 | 121.15 (vLLM) | **1.00x** | -- | 121.44 | 138.22 | 126.57 | 125.99 | 121.26 | 121.15 |
| shared_gate_up | 1 | 6.69 | 6.00 (Inductor max-autotune) | **0.90x** | 6.69 | 24.31 | -- | 6.00 | 15.16 | 11.33 | 6.68 |
| shared_gate_up | 2 | 6.70 | 6.80 (vLLM) | **1.01x** | 6.70 | 24.25 | -- | 13.08 | 12.83 | 12.88 | 6.80 |
| shared_gate_up | 4 | 6.71 | 7.33 (vLLM) | **1.09x** | 6.71 | 24.35 | -- | 13.08 | 12.79 | 12.82 | 7.33 |
| shared_gate_up | 8 | 6.72 | 12.40 (Inductor max-autotune) | **1.85x** | 6.72 | 23.43 | -- | 12.40 | 12.85 | 12.94 | 12.94 |
| shared_gate_up | 16 | 6.74 | 12.50 (vLLM) | **1.85x** | 6.74 | 23.17 | -- | 14.97 | 12.76 | 12.51 | 12.50 |
| shared_gate_up | 32 | 8.26 | 13.03 (hipBLASLt+TunableOp) | **1.58x** | 8.26 | 41.25 | 16.97 | 16.24 | 13.58 | 13.03 | 13.04 |
| shared_gate_up | 64 | 10.55 | 12.91 (hipBLASLt) | **1.22x** | 10.55 | 41.35 | 17.01 | 16.53 | 12.91 | 13.18 | 13.16 |
| shared_gate_up | 256 | 41.90 | 18.36 (hipBLASLt) | **0.44x** | -- | 41.90 | 19.70 | 18.38 | 18.36 | 20.25 | 20.25 |
| shared_gate_up | 1024 | 44.41 | 37.02 (vLLM) | **0.83x** | -- | 44.41 | 41.40 | 41.08 | 41.10 | 37.09 | 37.02 |
| shared_down | 1 | 5.83 | 4.67 (Inductor max-autotune) | **0.80x** | 5.83 | 6.72 | -- | 4.67 | 8.57 | 5.93 | 4.74 |
| shared_down | 2 | 5.84 | 5.13 (vLLM) | **0.88x** | 5.84 | 6.78 | -- | 5.67 | 7.63 | 5.92 | 5.13 |
| shared_down | 4 | 5.84 | 5.28 (Inductor max-autotune) | **0.91x** | 5.84 | 6.80 | -- | 5.28 | 7.60 | 5.92 | 5.71 |
| shared_down | 8 | 5.86 | 5.95 (Inductor max-autotune) | **1.02x** | 5.86 | 6.74 | -- | 5.95 | 7.62 | 5.95 | 5.95 |
| shared_down | 16 | 5.88 | 5.35 (Inductor max-autotune) | **0.91x** | 5.88 | 6.93 | -- | 5.35 | 7.46 | 6.05 | 6.04 |
| shared_down | 32 | 5.64 | 5.67 (hipBLASLt+TunableOp) | **1.00x** | 5.64 | 9.85 | 6.66 | 6.25 | 7.02 | 5.67 | 5.67 |
| shared_down | 64 | 8.37 | 5.93 (hipBLASLt+TunableOp) | **0.71x** | 8.37 | 9.95 | 6.68 | 6.13 | 14.79 | 5.93 | 5.93 |
| shared_down | 256 | 10.33 | 8.22 (vLLM) | **0.80x** | -- | 10.33 | 11.45 | 9.10 | 13.78 | 8.23 | 8.22 |
| shared_down | 1024 | 20.90 | 20.22 (hipBLASLt+TunableOp) | **0.97x** | -- | 20.90 | 29.18 | 23.38 | 25.40 | 20.22 | 20.22 |
| lm_head | 1 | 629.87 | 623.11 (Inductor max-autotune) | **0.99x** | 629.87 | -- | -- | 623.11 | 651.30 | 629.04 | 625.40 |
| lm_head | 2 | 632.34 | 629.16 (vLLM) | **0.99x** | 632.34 | -- | -- | 656.23 | 654.50 | 634.97 | 629.16 |
| lm_head | 4 | 638.23 | 639.52 (hipBLASLt+TunableOp) | **1.00x** | 638.23 | -- | -- | 662.40 | 661.22 | 639.52 | 640.80 |
| lm_head | 8 | 647.75 | 645.88 (vLLM) | **1.00x** | 647.75 | -- | -- | 673.71 | 673.06 | 646.02 | 645.88 |
| lm_head | 16 | 665.25 | 655.26 (hipBLASLt+TunableOp) | **0.98x** | 665.25 | -- | -- | 691.02 | 690.75 | 655.26 | 656.47 |
| lm_head | 32 | 691.00 | 672.30 (hipBLASLt+TunableOp) | **0.97x** | 691.00 | -- | 699.18 | 708.33 | 1258.11 | 672.30 | 672.84 |
| lm_head | 64 | 735.37 | 709.33 (vLLM) | **0.96x** | 735.37 | -- | 713.61 | 863.15 | 878.23 | 711.37 | 709.33 |
| lm_head | 256 | -- | 1591.99 (hipBLASLt+TunableOp) | -- | -- | -- | 1921.55 | 1653.03 | 1637.39 | 1591.99 | 1601.61 |

TunableOp: 62 of 62 (M, N, K) tuned here (M = 1 included), 163 results, 256 MB rotating buffer.

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

