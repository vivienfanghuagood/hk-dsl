### linear at openbmb/MiniCPM5-2B (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | FlyDSL | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 18.52 | 18.05 (Inductor max-autotune) | **0.97x** | 18.52 | 35.35 | -- | 18.05 | 26.57 | 23.15 | 18.53 |
| qkv | 2 | 18.58 | 20.89 (vLLM) | **1.12x** | 18.58 | 35.56 | -- | 28.29 | 28.29 | 22.87 | 20.89 |
| qkv | 4 | 18.73 | 22.90 (hipBLASLt+TunableOp) | **1.22x** | 18.73 | 35.77 | -- | 27.00 | 28.81 | 22.90 | 23.67 |
| qkv | 8 | 18.92 | 23.30 (hipBLASLt+TunableOp) | **1.23x** | 18.92 | 36.21 | -- | 32.12 | 29.23 | 23.30 | 23.32 |
| qkv | 16 | 19.27 | 23.34 (vLLM) | **1.21x** | 19.27 | 35.76 | -- | 30.53 | 29.64 | 23.39 | 23.34 |
| qkv | 32 | 21.96 | 19.77 (FlyDSL) | **0.90x** | 21.96 | 51.16 | 19.77 | 29.58 | 30.85 | 23.40 | 23.39 |
| qkv | 64 | 28.07 | 21.89 (FlyDSL) | **0.78x** | 28.07 | 52.06 | 21.89 | 34.76 | 34.94 | 25.97 | 25.98 |
| qkv | 256 | 58.30 | 45.39 (vLLM) | **0.78x** | -- | 58.30 | 54.24 | 67.32 | 77.17 | 45.40 | 45.39 |
| qkv | 1024 | 156.85 | 142.36 (hipBLASLt+TunableOp) | **0.91x** | -- | 156.85 | 171.42 | 149.47 | 149.53 | 142.36 | 142.37 |
| qkv | 4096 | 509.49 | 508.95 (FlyDSL) | **1.00x** | -- | 509.49 | 508.95 | 549.41 | 530.79 | 527.58 | 530.55 |
| o | 1 | 15.80 | 14.01 (Inductor max-autotune) | **0.89x** | 15.80 | 32.48 | -- | 14.01 | 23.22 | 22.31 | 15.38 |
| o | 2 | 15.89 | 17.73 (vLLM) | **1.12x** | 15.89 | 32.61 | -- | 23.56 | 23.43 | 22.79 | 17.73 |
| o | 4 | 15.91 | 18.69 (vLLM) | **1.18x** | 15.91 | 32.37 | -- | 23.14 | 23.67 | 22.40 | 18.69 |
| o | 8 | 16.07 | 23.34 (vLLM) | **1.45x** | 16.07 | 33.31 | -- | 23.68 | 23.97 | 23.40 | 23.34 |
| o | 16 | 16.57 | 22.67 (vLLM) | **1.37x** | 16.57 | 32.59 | -- | 27.13 | 24.32 | 22.68 | 22.67 |
| o | 32 | 18.45 | 18.96 (FlyDSL) | **1.03x** | 18.45 | 47.95 | 18.96 | 29.76 | 24.52 | 22.71 | 22.71 |
| o | 64 | 24.78 | 20.96 (FlyDSL) | **0.85x** | 24.78 | 48.26 | 20.96 | 30.93 | 42.09 | 25.02 | 24.98 |
| o | 256 | 52.22 | 37.27 (vLLM) | **0.71x** | -- | 52.22 | 42.44 | 54.84 | 50.84 | 37.38 | 37.27 |
| o | 1024 | 121.04 | 120.93 (hipBLASLt+TunableOp) | **1.00x** | -- | 121.04 | 137.82 | 126.29 | 126.65 | 120.93 | 121.03 |
| o | 4096 | 402.62 | 404.06 (FlyDSL) | **1.00x** | -- | 402.62 | 404.06 | 443.86 | 446.69 | 426.95 | 428.64 |
| gate_up | 1 | 69.28 | 68.42 (Inductor max-autotune) | **0.99x** | 69.28 | 188.94 | -- | 68.42 | 88.94 | 69.14 | 69.84 |
| gate_up | 2 | 69.58 | 69.22 (hipBLASLt+TunableOp) | **0.99x** | 69.58 | 187.74 | -- | 104.36 | 87.39 | 69.22 | 76.37 |
| gate_up | 4 | 70.84 | 69.42 (hipBLASLt+TunableOp) | **0.98x** | 70.84 | 192.24 | -- | 106.27 | 88.64 | 69.42 | 78.77 |
| gate_up | 8 | 71.71 | 69.70 (hipBLASLt+TunableOp) | **0.97x** | 71.71 | 210.97 | -- | 117.79 | 90.66 | 69.70 | 69.77 |
| gate_up | 16 | 74.03 | 70.13 (hipBLASLt+TunableOp) | **0.95x** | 74.03 | 188.99 | -- | 92.89 | 93.26 | 70.13 | 70.90 |
| gate_up | 32 | 76.96 | 71.54 (hipBLASLt+TunableOp) | **0.93x** | 76.96 | 198.91 | 81.79 | 119.03 | 178.13 | 71.54 | 71.61 |
| gate_up | 64 | 87.87 | 73.44 (vLLM) | **0.84x** | 87.87 | 189.54 | 86.37 | 189.29 | 192.20 | 73.50 | 73.44 |
| gate_up | 256 | 300.40 | 177.76 (hipBLASLt+TunableOp) | **0.59x** | -- | 300.40 | 196.97 | 279.12 | 274.38 | 177.76 | 178.78 |
| gate_up | 1024 | 593.91 | 592.66 (FlyDSL) | **1.00x** | -- | 593.91 | 592.66 | 639.14 | 637.29 | 611.46 | 613.45 |
| gate_up | 4096 | 2301.03 | 2346.63 (FlyDSL) | **1.02x** | -- | 2301.03 | 2346.63 | 2570.66 | 2535.41 | 2550.32 | 2543.53 |
| down | 1 | 40.34 | 35.96 (Inductor max-autotune) | **0.89x** | 40.34 | 77.44 | -- | 35.96 | 49.21 | 49.30 | 38.13 |
| down | 2 | 40.70 | 40.77 (vLLM) | **1.00x** | 40.70 | 77.56 | -- | 49.16 | 49.21 | 49.36 | 40.77 |
| down | 4 | 41.09 | 42.87 (vLLM) | **1.04x** | 41.09 | 77.32 | -- | 49.62 | 49.49 | 49.65 | 42.87 |
| down | 8 | 41.48 | 50.02 (vLLM) | **1.21x** | 41.48 | 79.86 | -- | 51.37 | 50.13 | 50.07 | 50.02 |
| down | 16 | 43.83 | 51.02 (Inductor max-autotune) | **1.16x** | 43.83 | 78.95 | -- | 51.02 | 51.17 | 51.09 | 51.19 |
| down | 32 | 45.82 | 52.11 (vLLM) | **1.14x** | 45.82 | 127.69 | 62.71 | 52.25 | 52.39 | 52.14 | 52.11 |
| down | 64 | 64.57 | 66.33 (hipBLASLt+TunableOp) | **1.03x** | 64.57 | 128.57 | 69.16 | 69.10 | 102.60 | 66.33 | 66.76 |
| down | 256 | 135.25 | 109.16 (vLLM) | **0.81x** | -- | 135.25 | 132.65 | 131.52 | 130.25 | 109.20 | 109.16 |
| down | 1024 | 329.19 | 323.92 (vLLM) | **0.98x** | -- | 329.19 | 400.09 | 380.76 | 385.02 | 324.79 | 323.92 |
| down | 4096 | 1173.09 | 1225.02 (vLLM) | **1.04x** | -- | 1173.09 | 1230.51 | 1426.72 | 1396.48 | 1227.67 | 1225.02 |
| lm_head | 1 | 661.23 | 652.83 (Inductor max-autotune) | **0.99x** | 661.23 | 801.07 | -- | 652.83 | 681.43 | 656.27 | 656.04 |
| lm_head | 2 | 663.85 | 658.65 (vLLM) | **0.99x** | 663.85 | 812.65 | -- | 687.27 | 686.65 | 660.64 | 658.65 |
| lm_head | 4 | 669.21 | 670.07 (hipBLASLt+TunableOp) | **1.00x** | 669.21 | 812.59 | -- | 692.49 | 690.83 | 670.07 | 671.25 |
| lm_head | 8 | 678.87 | 677.39 (vLLM) | **1.00x** | 678.87 | 828.73 | -- | 703.63 | 704.05 | 678.53 | 677.39 |
| lm_head | 16 | 696.87 | 687.61 (vLLM) | **0.99x** | 696.87 | 809.53 | -- | 720.43 | 723.59 | 690.97 | 687.61 |
| lm_head | 32 | 722.59 | 708.03 (hipBLASLt+TunableOp) | **0.98x** | 722.59 | 956.85 | 734.99 | 741.85 | 1307.27 | 708.03 | 708.49 |
| lm_head | 64 | 771.85 | 728.59 (hipBLASLt+TunableOp) | **0.94x** | 771.85 | 970.61 | 747.13 | 910.09 | 894.65 | 728.59 | 730.39 |
| lm_head | 256 | 1643.99 | 1732.26 (hipBLASLt) | **1.05x** | -- | 1643.99 | 1764.98 | 1746.78 | 1732.26 | 1743.13 | 1747.28 |

TunableOp: 48 of 48 (M, N, K) tuned here (M = 1 included), 163 results, 256 MB rotating buffer.

**Setup of each arm.**
- hk decode_gemm: hk.ops.decode_gemm, its schedule for M
- hk gemm: hk.ops.gemm, its tiling for M
- FlyDSL, by row:
  - qkv M=32, o M=32, gate_up M=32, down M=32, lm_head M=32: FlyDSL GEMM, tile 32x32x64w1
  - qkv M=64, qkv M=256, qkv M=1024, o M=64, o M=256, o M=1024, gate_up M=64, gate_up M=256, down M=64, down M=256, down M=1024, lm_head M=64: FlyDSL GEMM, tile 64x64x64w4
  - qkv M=4096, o M=4096, gate_up M=1024, down M=4096: FlyDSL GEMM, tile 128x128x32w4
  - gate_up M=4096, lm_head M=256: FlyDSL GEMM, tile 256x256x32w16
- Inductor max-autotune: torch.compile mode=max-autotune-no-cudagraphs, autotuned before timing
- hipBLASLt: torch F.linear, TunableOp off
- hipBLASLt+TunableOp: TunableOp tuned here, 256 MB rotating buffer
- vLLM: torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, else hipBLASLt+TunableOp)

- AITER: n/a: ModuleNotFoundError: No module named 'aiter' (AITER's GEMM and MoE kernels target CDNA: gfx942, gfx950)
- FlyDSL: n/a at some shapes: no tile takes them

