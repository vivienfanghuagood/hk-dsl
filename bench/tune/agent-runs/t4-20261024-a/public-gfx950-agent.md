# T4 score: gfx950 (AMD Instinct MI355X), final

2026-10-24T07:26Z; hk-dsl at b4778dc; torch 2.12.0+rocm10.0.0, hip 7.15.26333, triton 3.8.0, vllm 0.32.1rc1.dev36+ge68efc685, flydsl 0.3.4.1, amd-aiter 0.1.25; 3 sets x 7 rounds; shapes: shapes_public.json.

**sol:solution**: PASS -- geomean 1.2079, min 1.0099, 12/12 shapes >= 0.95 (pass). Rule: correct + audit clean on every shape, geomean >= 0.95, min >= 0.85.

| shape | E | N | K | rows | roofline | torch_loop | torch_grouped_mm | triton | aiter_gmm | aiter_ck | vllm_fused_moe | hk_ops_moe | dense (proxy) | sol:solution | best | best/roofline | sol:solution ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe-1b-7b-gate_up-t4096 | 64 | 2048 | 2048 | 32768 | 109.1 | 1526.3 | 924.2 | 390.0 | 808.5 | 430.3 | 576.6 | x | 195.1 | 289.6 | triton | 0.28 | 1.345 (1.33-1.36) |
| olmoe-1b-7b-down-t512 | 64 | 2048 | 1024 | 4096 | 36.7 | 928.2 | 768.2 | 80.2 | 172.9 | 98.0 | 98.8 | x | 20.2 | 66.8 | triton | 0.46 | 1.200 (1.19-1.21) |
| mixtral-8x7b-tp4-gate_up-t2048 | 8 | 7168 | 4096 | 4096 | 95.4 | 580.1 | 411.8 | 358.3 | 368.8 | 316.9 | 414.2 | x | 169.3 | 248.3 | aiter_ck | 0.30 | 1.274 (1.27-1.28) |
| mixtral-8x7b-tp4-down-t4096 | 8 | 4096 | 3584 | 8192 | 95.4 | 504.7 | 337.1 | 339.4 | 740.1 | 328.9 | 426.2 | x | 161.3 | 233.8 | aiter_ck | 0.29 | 1.416 (1.41-1.43) |
| qwen1.5-moe-a2.7b-gate_up-t512 | 60 | 2816 | 2048 | 2048 | 89.0 | 884.2 | 725.1 | 175.8 | 450.5 | 168.1 | 177.9 | x | 31.1 | 153.6 | aiter_ck | 0.53 | 1.095 (1.09-1.10) |
| qwen1.5-moe-a2.7b-down-t4096 | 60 | 2048 | 1408 | 16384 | 57.4 | 1238.0 | 747.0 | 163.5 | 347.2 | 219.9 | 267.3 | x | 73.9 | 147.7 | triton | 0.35 | 1.108 (1.09-1.11) |
| granite-3.1-3b-a800m-gate_up-t1024 | 40 | 1024 | 1536 | 8192 | 21.0 | 668.7 | 501.7 | 73.3 | 130.1 | 99.0 | 124.9 | x | 27.9 | 46.2 | triton | 0.29 | 1.592 (1.58-1.60) |
| granite-3.1-3b-a800m-down-t8192 | 40 | 1536 | 512 | 65536 | 41.4 | 2059.2 | 512.6 | 188.6 | 350.8 | 394.9 | 576.6 | x | 127.0 | 141.1 | triton | 0.22 | 1.339 (1.32-1.34) |
| deepseek-v2-tp8-gate_up-t4096 | 160 | 384 | 5120 | 24576 | 112.5 | 2923.5 | 1970.5 | 230.8 | 760.2 | 293.8 | 314.8 | x | 95.8 | 229.5 | triton | 0.49 | 1.010 (1.01-1.01) |
| deepseek-v2-tp8-down-t1024 | 160 | 5120 | 192 | 6144 | 47.5 | 2288.0 | 1887.3 | 89.0 | 315.6 | 133.4 | 107.7 | x | 28.4 | 85.2 | triton | 0.53 | 1.048 (1.04-1.05) |
| qwen2-57b-a14b-tp8-gate_up-t256 | 64 | 640 | 3584 | 2048 | 38.9 | 944.8 | 772.0 | 78.1 | 259.7 | 91.9 | 95.7 | 215.8 | 24.0 | 74.9 | triton | 0.50 | 1.044 (1.04-1.06) |
| qwen2-57b-a14b-tp8-down-t2048 | 64 | 3584 | 320 | 16384 | 34.3 | 1202.0 | 791.4 | 110.3 | 215.3 | 226.9 | 194.8 | x | 61.1 | 94.7 | triton | 0.31 | 1.159 (1.16-1.16) |

Arms not timed:

- olmoe-1b-7b-gate_up-t4096 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- olmoe-1b-7b-down-t512 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- mixtral-8x7b-tp4-gate_up-t2048 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- mixtral-8x7b-tp4-down-t4096 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- qwen1.5-moe-a2.7b-gate_up-t512 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- qwen1.5-moe-a2.7b-down-t4096 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- granite-3.1-3b-a800m-gate_up-t1024 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- granite-3.1-3b-a800m-down-t8192 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- deepseek-v2-tp8-gate_up-t4096 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- deepseek-v2-tp8-down-t1024 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op
- qwen2-57b-a14b-tp8-down-t2048 `base:hk_ops_moe`: edge: VerifyError: this op needs '!hk.rt<bf16, 16x16, row>' in 16x32 fragments -- as an mma operand, or tied to a tile that is one, or that names them -- and 16x16 is not a whole number of them. On this target an mma op

Audit (one call under torch.profiler):

- olmoe-1b-7b-gate_up-t4096 `sol:solution`: ok; hk 284.3 us (100%), torch 0.0 us, foreign 0.0 us
- olmoe-1b-7b-down-t512 `sol:solution`: ok; hk 68.9 us (100%), torch 0.0 us, foreign 0.0 us
- mixtral-8x7b-tp4-gate_up-t2048 `sol:solution`: ok; hk 242.6 us (100%), torch 0.0 us, foreign 0.0 us
- mixtral-8x7b-tp4-down-t4096 `sol:solution`: ok; hk 231.7 us (100%), torch 0.0 us, foreign 0.0 us
- qwen1.5-moe-a2.7b-gate_up-t512 `sol:solution`: ok; hk 158.1 us (100%), torch 0.0 us, foreign 0.0 us
- qwen1.5-moe-a2.7b-down-t4096 `sol:solution`: ok; hk 143.9 us (100%), torch 0.0 us, foreign 0.0 us
- granite-3.1-3b-a800m-gate_up-t1024 `sol:solution`: ok; hk 47.9 us (100%), torch 0.0 us, foreign 0.0 us
- granite-3.1-3b-a800m-down-t8192 `sol:solution`: ok; hk 140.4 us (100%), torch 0.0 us, foreign 0.0 us
- deepseek-v2-tp8-gate_up-t4096 `sol:solution`: ok; hk 240.9 us (100%), torch 0.0 us, foreign 0.0 us
- deepseek-v2-tp8-down-t1024 `sol:solution`: ok; hk 84.5 us (100%), torch 0.0 us, foreign 0.0 us
- qwen2-57b-a14b-tp8-gate_up-t256 `sol:solution`: ok; hk 75.1 us (100%), torch 0.0 us, foreign 0.0 us
- qwen2-57b-a14b-tp8-down-t2048 `sol:solution`: ok; hk 93.0 us (100%), torch 0.0 us, foreign 0.0 us

Times are per call, us: the median over sets of each set's median round. ratio = best baseline / solution (> 1: faster), the median and range over sets. `x` = not timed (see above). roofline = max(FLOPs / bf16 peak, bytes / DRAM bandwidth) at the datasheets' peaks, bytes = x + out + the weights of every expert with a row; best/roofline = the fraction of it the best baseline reaches. `dense` is a proxy (one expert's weights for all rows), not the task's math.
