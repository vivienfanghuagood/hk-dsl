### linear at openbmb/MiniCPM5-2B (TP1) on AMD Radeon Pro W7900D (gfx1100)

torch 2.13.0a0+rocm7.14.0a20260612, HIP 7.14.60850, vLLM 0.1.dev1+g3775d5fca, Triton 3.7.1, FlyDSL 0.4.0, AITER n/a (ModuleNotFoundError: No module named 'aiter'), hk 0.8.0. Same process, 5 interleaved rounds, the fastest kept; graph-timed, weights rotated over 576 MB a shape.

| shape | M | hk, us | best competitor, us | hk vs best | hk decode_gemm | hk gemm | FlyDSL | Inductor max-autotune | hipBLASLt | hipBLASLt+TunableOp | vLLM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 1 | 18.52 | 17.84 (Inductor max-autotune) | **0.96x** | 18.52 | 35.24 | -- | 17.84 | 26.62 | 23.16 | 18.55 |
| qkv | 2 | 18.56 | 20.94 (vLLM) | **1.13x** | 18.56 | 35.64 | -- | 28.25 | 28.29 | 22.86 | 20.94 |
| qkv | 4 | 18.65 | 22.90 (hipBLASLt+TunableOp) | **1.23x** | 18.65 | 35.54 | -- | 27.01 | 28.76 | 22.90 | 23.66 |
| qkv | 8 | 18.85 | 23.32 (vLLM) | **1.24x** | 18.85 | 35.82 | -- | 31.45 | 29.18 | 23.32 | 23.32 |
| qkv | 16 | 19.25 | 23.41 (hipBLASLt+TunableOp) | **1.22x** | 19.25 | 36.11 | -- | 30.53 | 29.65 | 23.41 | 23.42 |
| qkv | 32 | 21.94 | 19.83 (FlyDSL) | **0.90x** | 21.94 | 51.56 | 19.83 | 29.83 | 30.72 | 23.49 | 23.49 |
| qkv | 64 | 28.04 | 22.21 (FlyDSL) | **0.79x** | 28.04 | 52.41 | 22.21 | 34.70 | 34.86 | 25.89 | 25.93 |
| qkv | 256 | 59.39 | 44.96 (hipBLASLt+TunableOp) | **0.76x** | -- | 59.39 | 54.99 | 65.53 | 77.50 | 44.96 | 45.11 |
| qkv | 1024 | 156.54 | 144.30 (hipBLASLt+TunableOp) | **0.92x** | -- | 156.54 | 170.72 | 151.77 | 151.06 | 144.30 | 144.38 |
| qkv | 4096 | 503.84 | 506.07 (FlyDSL) | **1.00x** | -- | 503.84 | 506.07 | 532.81 | 531.35 | 507.56 | 507.73 |
| o | 1 | 15.81 | 13.99 (Inductor max-autotune) | **0.89x** | 15.81 | 32.50 | -- | 13.99 | 23.29 | 22.35 | 15.40 |
| o | 2 | 15.90 | 17.74 (vLLM) | **1.12x** | 15.90 | 32.65 | -- | 23.53 | 23.45 | 22.71 | 17.74 |
| o | 4 | 15.94 | 18.68 (vLLM) | **1.17x** | 15.94 | 32.71 | -- | 23.31 | 23.61 | 22.45 | 18.68 |
| o | 8 | 16.11 | 23.32 (vLLM) | **1.45x** | 16.11 | 33.17 | -- | 23.81 | 23.98 | 23.57 | 23.32 |
| o | 16 | 16.55 | 22.74 (hipBLASLt+TunableOp) | **1.37x** | 16.55 | 32.43 | -- | 27.19 | 24.39 | 22.74 | 22.74 |
| o | 32 | 18.48 | 18.96 (FlyDSL) | **1.03x** | 18.48 | 48.09 | 18.96 | 29.50 | 24.67 | 22.83 | 22.79 |
| o | 64 | 24.71 | 21.29 (FlyDSL) | **0.86x** | 24.71 | 48.38 | 21.29 | 30.99 | 42.56 | 25.09 | 25.06 |
| o | 256 | 52.37 | 36.74 (hipBLASLt+TunableOp) | **0.70x** | -- | 52.37 | 42.00 | 53.08 | 51.39 | 36.74 | 36.76 |
| o | 1024 | 120.60 | 122.24 (hipBLASLt+TunableOp) | **1.01x** | -- | 120.60 | 137.29 | 127.50 | 127.13 | 122.24 | 122.32 |
| o | 4096 | 398.42 | 402.49 (FlyDSL) | **1.01x** | -- | 398.42 | 402.49 | 438.73 | 441.10 | 413.10 | 414.25 |
| gate_up | 1 | 69.45 | 68.45 (Inductor max-autotune) | **0.99x** | 69.45 | 189.13 | -- | 68.45 | 88.43 | 69.10 | 69.83 |
| gate_up | 2 | 69.78 | 69.43 (hipBLASLt+TunableOp) | **0.99x** | 69.78 | 189.73 | -- | 103.40 | 87.46 | 69.43 | 76.52 |
| gate_up | 4 | 70.92 | 69.46 (hipBLASLt+TunableOp) | **0.98x** | 70.92 | 191.51 | -- | 105.00 | 88.18 | 69.46 | 78.91 |
| gate_up | 8 | 71.96 | 69.72 (vLLM) | **0.97x** | 71.96 | 209.17 | -- | 117.86 | 90.38 | 69.84 | 69.72 |
| gate_up | 16 | 74.12 | 70.98 (hipBLASLt+TunableOp) | **0.96x** | 74.12 | 180.19 | -- | 93.18 | 93.13 | 70.98 | 71.05 |
| gate_up | 32 | 77.05 | 71.68 (vLLM) | **0.93x** | 77.05 | 197.78 | 81.43 | 119.85 | 177.59 | 71.75 | 71.68 |
| gate_up | 64 | 87.86 | 73.54 (hipBLASLt+TunableOp) | **0.84x** | 87.86 | 195.52 | 83.99 | 192.76 | 194.92 | 73.54 | 73.68 |
| gate_up | 256 | 301.39 | 171.70 (hipBLASLt+TunableOp) | **0.57x** | -- | 301.39 | 197.37 | 277.84 | 279.39 | 171.70 | 172.33 |
| gate_up | 1024 | 590.56 | 592.27 (FlyDSL) | **1.00x** | -- | 590.56 | 592.27 | 635.69 | 634.52 | 613.51 | 614.34 |
| gate_up | 4096 | 2326.70 | 2347.54 (FlyDSL) | **1.01x** | -- | 2326.70 | 2347.54 | 2535.24 | 2554.43 | 2381.07 | 2388.06 |
| down | 1 | 40.05 | 36.10 (Inductor max-autotune) | **0.90x** | 40.05 | 77.69 | -- | 36.10 | 49.50 | 49.54 | 38.38 |
| down | 2 | 40.57 | 40.80 (vLLM) | **1.01x** | 40.57 | 77.73 | -- | 49.70 | 49.26 | 49.27 | 40.80 |
| down | 4 | 40.62 | 42.95 (vLLM) | **1.06x** | 40.62 | 77.89 | -- | 49.77 | 49.87 | 49.85 | 42.95 |
| down | 8 | 41.26 | 50.38 (hipBLASLt+TunableOp) | **1.22x** | 41.26 | 79.45 | -- | 51.51 | 50.49 | 50.38 | 50.48 |
| down | 16 | 43.34 | 51.15 (vLLM) | **1.18x** | 43.34 | 78.81 | -- | 51.26 | 51.45 | 51.22 | 51.15 |
| down | 32 | 45.39 | 52.23 (vLLM) | **1.15x** | 45.39 | 127.96 | 62.93 | 52.58 | 52.32 | 52.35 | 52.23 |
| down | 64 | 64.68 | 65.56 (hipBLASLt+TunableOp) | **1.01x** | 64.68 | 129.21 | 70.04 | 68.19 | 102.78 | 65.56 | 65.69 |
| down | 256 | 137.11 | 113.48 (vLLM) | **0.83x** | -- | 137.11 | 132.48 | 132.59 | 133.25 | 113.93 | 113.48 |
| down | 1024 | 324.69 | 324.42 (hipBLASLt+TunableOp) | **1.00x** | -- | 324.69 | 394.80 | 379.29 | 377.07 | 324.42 | 324.65 |
| down | 4096 | 1161.11 | 1219.95 (vLLM) | **1.05x** | -- | 1161.11 | 1237.48 | 1384.99 | 1305.41 | 1220.09 | 1219.95 |
| lm_head | 1 | 662.42 | 653.64 (Inductor max-autotune) | **0.99x** | 662.42 | 797.75 | -- | 653.64 | 682.60 | 658.98 | 657.48 |
| lm_head | 2 | 664.50 | 659.12 (vLLM) | **0.99x** | 664.50 | 803.88 | -- | 688.96 | 689.40 | 661.34 | 659.12 |
| lm_head | 4 | 670.46 | 671.36 (hipBLASLt+TunableOp) | **1.00x** | 670.46 | 809.50 | -- | 694.46 | 694.04 | 671.36 | 671.60 |
| lm_head | 8 | 680.30 | 678.48 (hipBLASLt+TunableOp) | **1.00x** | 680.30 | 828.78 | -- | 704.54 | 706.18 | 678.48 | 679.22 |
| lm_head | 16 | 698.54 | 690.48 (vLLM) | **0.99x** | 698.54 | 814.97 | -- | 723.90 | 724.98 | 690.80 | 690.48 |
| lm_head | 32 | 725.16 | 708.75 (vLLM) | **0.98x** | 725.16 | 958.30 | 735.87 | 743.78 | 1348.85 | 709.08 | 708.75 |
| lm_head | 64 | 772.09 | 727.32 (hipBLASLt+TunableOp) | **0.94x** | 772.09 | 970.67 | 749.22 | 886.00 | 901.28 | 727.32 | 727.98 |
| lm_head | 256 | 1615.83 | 1678.77 (FlyDSL) | **1.04x** | -- | 1615.83 | 1678.77 | 1748.21 | 1718.27 | 1723.61 | 1718.25 |

TunableOp: 48 of 48 (M, N, K) tuned here (M = 1 included), 48 results, 256 MB rotating buffer.

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
