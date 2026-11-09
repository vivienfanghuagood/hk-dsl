### SGLang-Diffusion routes' kernels at Qwen-Image 2.1's shapes on an AMD Radeon AI PRO R9700 (gfx1201)

`tools/hk-bench/sgld_kernels.py`, hk-dsl 0.9.17, SGLang 0.5.23, torch 2.10.0+rocm7.2.3; TunableOp on with the row's tuned
results (18/18 shapes). Per call: median of 5 interleaved rounds of 50 calls (CUDA events), the card under its node lock.

| route | shape | engine | hk | hk vs engine | |
|---|---|---:|---:|---:|---|
| sgld_linear | 4096x12288x4096 (packed QKV) | F.linear 3406 us | 3266 us | 1.04x | max rel err 4.4e-3 both |
| sgld_linear | 4096x12288x4096 + bias | 3423 us | 3208 us | 1.07x | |
| sgld_linear | 4096x4096x4096 (out) | 1230 us | 1146 us | 1.07x | |
| sgld_linear | 4096x4096x4096 + bias | 1270 us | 1149 us | 1.11x | |
| sgld_linear | 4096x4096x12288 (MLP down) | 3641 us | 3219 us | 1.13x | |
| sgld_linear | 4096x4096x12288 + bias | 3815 us | 3207 us | 1.19x | |
| sgld_rope | (1, 4141, 32, 128) bf16 | SGLang apply_rope 586 us | 92 us | 6.40x | bit-identical |
| sgld_rope | (1, 4096, 32, 128) bf16 | 572 us | 70 us | 8.21x | bit-identical |

The model's own projections are the three no-bias shapes (the route's per-shape decisions in a served request: 1.08x,
1.10x, 1.12x), all short of 1.15x, so `sgld_linear` stays off on gfx1201; `sgld_rope` is on there.
