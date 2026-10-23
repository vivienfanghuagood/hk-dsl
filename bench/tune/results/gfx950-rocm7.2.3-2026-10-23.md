# hk.tune scoreboard: gfx950, acceptance split, 2026-10-23

Device AMD Radeon Graphics; torch 2.10.0+rocm7.2.3.git1a270074, HIP 7.2.53211, hipcc `HIP version: 7.2.53211-c2d9476115`; triton 3.6.0+rocm7.2.3.git4ed88892.
hk-dsl at `f0dff9a` (claude/tune). Baselines missing: none. hk.tune budget 60.0 GPU-s a shape; scoring 3 sets x 7 rounds, interleaved, queued, weights rotated past the Infinity Cache.

| | before (routes) | after (hk.tune) |
|---|---|---|
| shapes >= 0.95x best | 3/25 (12%) | **19/25 (76%)** |
| median ratio | 0.709 | 0.972 |
| worst ratio | 0.252 | 0.821 |
| geomean ratio | 0.67 | 0.983 |

With a route to the engine's `F.linear` (hipBLASLt) where it is faster than hk's kernel -- what hk.integration's route tables do -- 22/25 shapes reach 0.95x the best baseline (median 0.997, worst 0.882); F.linear is the faster of the two at 13. The table's count is hk's own kernels.

Tuning cost: 0.11 GPU-min a shape on average, 0.29 at most, 2.7 in all; 0.23 min of wall clock a shape (the compiles). 21 of 25 shapes got a record; the rest kept their route.

Ratio = best baseline time / hk time (>1: hk faster), median over the scoring sets [their range].

| M x N x K | best baseline | best us | before | after | route before -> after | tune | GPU-s |
|---|---|---|---|---|---|---|---|
| 1x4608x9216 | tunableop | 19.82 | 0.795 [0.791-0.798] | 1.161 [1.154-1.169] | decode -> u:dp:16_x64_l16x64_ntb_sk6_s4_w1_g4_c0 | win +46.4% | 6.63 |
| 8x3072x8192 | tunableop | 12.21 | 0.795 [0.793-0.806] | 1.000 [0.992-1.005] | decode -> u:dp:32_x48_l16x64_ntb_sk8_s4_x1_c0 | win +27.7% | 3.26 |
| 8x55296x5120 | tunableop | 84.48 | 0.709 [0.703-0.711] | 0.998 [0.997-0.998] | decode -> u:dp:16_x32_k128_l16x64_ntb_pl0_s5_w1_g16_c16 | win +40.3% | 2.5 |
| 16x4096x7168 | tunableop | 13.3 | 0.775 [0.774-0.776] | 0.993 [0.988-1.004] | decode -> u:dp:32_x64_l16x64_ntb_sk4_s4_c16 | win +29.6% | 5.81 |
| 16x5120x5120 | torch | 15.17 | 0.682 [0.681-0.684] | 1.248 [1.241-1.250] | decode -> u:dp:16_x32_k128_l16x64_ntb_sp1_s5_w1_g16_c16 | win +79.7% | 2.67 |
| 16x7168x8192 | tunableop | 21.62 | 0.602 [0.599-0.602] | 1.045 [1.034-1.047] | decode -> u:dp:16_x32_k128_l16x64_ntb_pl0_sp1_s5_w1_g16_c0 | win +74.0% | 1.77 |
| 128x4096x512 | triton | 4.8 | 0.784 [0.783-0.784] | 1.289 [1.289-1.291] | dp64 -> u:dp:32_x64_k128_s4_g16 | win +63.8% | 7.61 |
| 128x4608x36864 | tunableop | 72.44 | 0.252 [0.251-0.252] | 0.888 [0.886-0.890] **miss** | dp64 -> u:dp:x192_l16x64_ntb_sk8_w1_x4_g4_c0 | win +256.0% | 2.35 |
| 128x5120x14336 | torch | 39.22 | 0.341 [0.341-0.342] | 0.904 [0.898-0.911] **miss** | dp64 -> u:pp:128_l16x64_ntb_sk8_g16 | win +167.6% | 1.95 |
| 512x4096x7168 | tunableop | 39.39 | 0.698 [0.694-0.699] | 0.956 [0.955-0.956] | dp64 -> u:pp:x128_es0_sk4_g16_c32 | win +37.8% | 6.78 |
| 512x7168x8192 | tunableop | 66.45 | 0.647 [0.646-0.651] | 1.039 [1.030-1.040] | dp64 -> u:pp:lb0_sk4_g16 | win +59.5% | 2.67 |
| 512x8192x3584 | tunableop | 37.22 | 0.712 [0.710-0.715] | 0.970 [0.960-0.973] | dp64 -> u:dp:s4_g4 | win +35.4% | 2.78 |
| 1024x5120x14336 | torch | 133.83 | 0.504 [0.501-0.507] | 0.846 [0.836-0.849] **miss** | dp128 -> u:pp:sk2_g4 | win +67.8% | 6.5 |
| 2048x4096x1024 | tunableop | 18.14 | 0.714 [0.712-0.720] | 0.881 [0.880-0.883] **miss** | dp128 -> u:pp:128_g4_c32 | win +22.4% | 4.36 |
| 2048x4608x4096 | tunableop | 62.71 | 0.573 [0.572-0.574] | 0.821 [0.821-0.821] **miss** | dp128 -> u:pp:g4_c16 | win +43.8% | 5.82 |
| 2048x7168x8192 | tunableop | 163.15 | 0.604 [0.596-0.606] | 0.977 [0.976-0.979] | dp128 -> u:pp:g4_c32 | win +65.1% | 8.5 |
| 2048x8192x3584 | torch | 83.11 | 0.608 [0.605-0.611] | 0.980 [0.978-0.980] | dp128 -> u:pp:g4_c32 | win +59.5% | 3.67 |
| 2048x16384x4096 | tunableop | 184.49 | 0.600 [0.597-0.605] | 0.978 [0.977-0.978] | dp128 -> u:pp:g4 | win +60.2% | 3.75 |
| 2048x55296x5120 | tunableop | 775.68 | 0.537 [0.534-0.537] | 0.964 [0.963-0.966] | dp128 -> u:pp:g16_c32 | win +79.2% | 10.77 |
| 4096x3072x4096 | tunableop | 80.29 | 0.959 [0.957-0.960] | 0.967 [0.944-0.969] | pp -> pp | tie | 6.59 |
| 4096x3072x8192 | torch | 150.02 | 0.984 [0.983-0.984] | 0.984 [0.983-0.984] | pp -> pp | tie | 6.98 |
| 4096x4608x4096 | tunableop | 121.58 | 0.774 [0.770-0.776] | 0.972 [0.963-0.978] | pp -> u:cols:wave:pp:c32|dp:pl0_sk2_s4_x4_c16 | win +24.6% | 17.65 |
| 8192x5120x5120 | tunableop | 305.41 | 0.954 [0.953-0.954] | 0.950 [0.949-0.951] | pp -> pp | tie | 17.55 |
| 8192x14336x8192 | tunableop | 1190.89 | 0.931 [0.931-0.934] | 0.940 [0.933-0.943] **miss** | pp -> pp | tie | 10.34 |
| 8192x18432x4608 | torch | 890.0 | 0.917 [0.910-0.922] | 0.951 [0.946-0.956] | pp -> u:pp:g4 | win +3.3% | 12.53 |

Every baseline's time (us):

| M x N x K | hipkittens | torch | triton | tunableop |
|---|---|---|---|---|
| 1x4608x9216 | - | 20.44 | 55.6 | 19.82 |
| 8x3072x8192 | - | 13.25 | 47.31 | 12.21 |
| 8x55296x5120 | - | 84.96 | 94.08 | 84.48 |
| 16x4096x7168 | - | 14.32 | 43.37 | 13.3 |
| 16x5120x5120 | - | 15.17 | 31.75 | 16.09 |
| 16x7168x8192 | - | 26.19 | 49.63 | 21.62 |
| 128x4096x512 | - | 5.5 | 4.8 | 5.17 |
| 128x4608x36864 | - | 79.55 | 226.06 | 72.44 |
| 128x5120x14336 | - | 39.22 | 88.93 | 43.27 |
| 512x4096x7168 | 136.38 | 41.16 | 59.81 | 39.39 |
| 512x7168x8192 | 142.05 | 78.43 | 97.57 | 66.45 |
| 512x8192x3584 | 73.33 | 43.41 | 46.81 | 37.22 |
| 1024x5120x14336 | 271.53 | 133.83 | 261.01 | 135.76 |
| 2048x4096x1024 | 26.89 | 20.57 | 27.48 | 18.14 |
| 2048x4608x4096 | 78.35 | 67.54 | 109.38 | 62.71 |
| 2048x7168x8192 | 170.54 | 173.11 | 240.56 | 163.15 |
| 2048x8192x3584 | 87.34 | 83.11 | 114.61 | 83.17 |
| 2048x16384x4096 | 196.76 | 185.09 | 256.92 | 184.49 |
| 2048x55296x5120 | 820.66 | 795.18 | 1101.36 | 775.68 |
| 4096x3072x4096 | 83.76 | 80.56 | 125.73 | 80.29 |
| 4096x3072x8192 | 153.8 | 150.02 | 236.75 | 150.17 |
| 4096x4608x4096 | 159.82 | 132.21 | 194.04 | 121.58 |
| 8192x5120x5120 | 320.93 | 322.85 | 429.36 | 305.41 |
| 8192x14336x8192 | 1272.2 | 1222.07 | 1836.89 | 1190.89 |
| 8192x18432x4608 | 972.34 | 890.0 | 1336.41 | 900.46 |

Misses (under 0.95x the best baseline after tuning), and what the numbers say about why:

- **128x4608x36864** (0.888x tunableop): best 600 TFLOP/s, 4.84 TB/s; hk 533 TFLOP/s, 4.30 TB/s; memory-bound: the best baseline streams at 60% of HBM, hk at 54%; hk ran dp 128x192 x8 K shares: 192 workgroups, 0.75 of 256 CUs
- **128x5120x14336** (0.904x torch): best 479 TFLOP/s, 3.87 TB/s; hk 433 TFLOP/s, 3.50 TB/s; neither limit is near: latency, launch or occupancy; hk ran pp 128x256 x8 K shares: 160 workgroups, 0.62 of 256 CUs
- **1024x5120x14336** (0.846x torch): best 1123 TFLOP/s, 1.39 TB/s; hk 945 TFLOP/s, 1.17 TB/s; compute-bound: the best baseline at 45% of the bf16 peak, hk at 38%; hk ran pp 256x256 x2 K shares: 160 workgroups, 0.62 of 256 CUs
- **2048x4096x1024** (0.881x tunableop): best 947 TFLOP/s, 1.62 TB/s; hk 833 TFLOP/s, 1.42 TB/s; neither limit is near: latency, launch or occupancy; hk ran pp 128x256: 256 workgroups, 1.00 of 256 CUs
- **2048x4608x4096** (0.821x tunableop): best 1233 TFLOP/s, 1.17 TB/s; hk 1013 TFLOP/s, 0.96 TB/s; compute-bound: the best baseline at 49% of the bf16 peak, hk at 41%; hk ran pp 256x256: 144 workgroups, 0.56 of 256 CUs
- **8192x14336x8192** (0.940x tunableop): best 1616 TFLOP/s, 0.51 TB/s; hk 1517 TFLOP/s, 0.48 TB/s; compute-bound: the best baseline at 65% of the bf16 peak, hk at 61%; hk ran pp 256x256: 1792 workgroups, 7.00 of 256 CUs

Arms dropped at a shape (wrong against fp32, or refused the shape):

- 1x4608x9216: base:hipkittens: ValueError: M=1, N=4608: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end 
- 8x3072x8192: base:hipkittens: ValueError: M=8, N=3072: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end 
- 8x55296x5120: base:hipkittens: ValueError: M=8, N=55296: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end
- 16x4096x7168: base:hipkittens: ValueError: M=16, N=4096: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end
- 16x5120x5120: base:hipkittens: ValueError: M=16, N=5120: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end
- 16x7168x8192: base:hipkittens: ValueError: M=16, N=7168: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the end
- 128x4096x512: base:hipkittens: ValueError: M=128, N=4096: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the en
- 128x4608x36864: base:hipkittens: ValueError: M=128, N=4608: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the en
- 128x5120x14336: base:hipkittens: ValueError: M=128, N=5120: both must be multiples of 256; the kernel computes whole 256x256 tiles and its C store is unpredicated, so a partial tile would write past the en
