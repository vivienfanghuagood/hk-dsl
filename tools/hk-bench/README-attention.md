# Attention benchmark: hk against vLLM's kernels on Radeon

`attention_bench.py` times the attention kernels a vLLM server runs: decode steps and prefill.
It compares hk's kernels with vLLM's own on the same KV cache, at the shapes of models that are
served on Radeon, in one process. One GPU is enough. Most other scripts in this directory are
hk's internal sweeps; this one is meant to be rerun by anyone.

## Run it

You need hk-dsl, a ROCm build of PyTorch, and vLLM (its kernels are the baselines). The vLLM ROCm
images for Radeon have both PyTorch and vLLM.

```bash
pip install hk-dsl                      # or: a checkout, with python/ on PYTHONPATH
python3 tools/hk-bench/attention_bench.py decode  --model minicpm5
python3 tools/hk-bench/attention_bench.py decode  --model qwen3.8 --batch 1,2,4,8,16 --ctx 8192,32768
python3 tools/hk-bench/attention_bench.py prefill --model minicpm5 --reqs 0:8192,0:16384,16384:2048
```

Each shape prints one JSON row as it finishes. At the end the script prints a markdown table.
`--json FILE` also writes the rows to a file. `--arms` picks the arms; the default is
`hk,triton,rocm_attn`.

On a card shared with other jobs, timings are not trustworthy. Run on an idle GPU.

## What is compared

| arm | what it is | cache it reads |
|---|---|---|
| `hk` | decode: `hk.ops.paged_decode` (split-KV). Prefill: hk's vLLM route (`hk.integration.vllm_prefill`, the stand-in for `unified_attention`), which runs `hk.ops.paged_prefill` | TRITON_ATTN's, in place |
| `triton` | vLLM's `unified_attention`, called as TRITON_ATTN's `forward` calls it. For a decode batch below vLLM's threshold (128 / kv heads) that is the 3-D kernel (16 softmax segments plus a reduction); above it, the 2-D kernel | TRITON_ATTN's |
| `rocm_attn` | vLLM's ROCM_ATTN: `chunked_prefill_paged_decode`. Prefill goes to the prefix-prefill kernel; decode goes to ROCm's paged decode, or its Triton fallback where that kernel does not run | ROCM_ATTN's own (K x-split, V transposed), holding the same values |

The prefill `hk` arm goes through the same entry point a server uses. A call that hk's route does
not take falls back to vLLM's kernel there too. The row's `hk_calls` shows where the calls went.

## Shapes

`--model` gives one tensor-parallel rank's share of a model: query heads, kv heads, head_dim, and
the cache page and order vLLM uses for that model on Radeon.

| model | heads (q/kv) | head_dim | page | cache order |
|---|---|---|---|---|
| `minicpm5` (MiniCPM5-2B) | 16/2 | 128 | 16 | K, V planes |
| `qwen3-8b` | 32/8 | 128 | 16 | K, V planes |
| `qwen3.8` (Qwen3.8-27B, TP1) | 24/4 | 256 | 784 | a head's K and V side by side, head-major |
| `qwen3.8-tp2` | 12/2 | 256 | 784 | same |
| `minicpm-v-tp2` (MiniCPM-V-4.7-35B, TP2) | 8/1 | 256 | 544 | K and V side by side, token-major |
| `qwen3.6` (Qwen3.6-35B-A3B, bf16 KV) | 16/2 | 256 | 2064 | same |

Hybrid models (Mamba/GDN layers plus attention) get large attention pages: vLLM sizes them to the
mamba state. The newer vLLM keeps a head's K and V side by side.

- **Decode:** `--batch` requests, each with `--ctx` tokens in the cache and one new query.
- **Prefill:** `--reqs CTX:QLEN`, a request with CTX cached tokens and QLEN new ones. `+` joins
  several requests into one batch, for example `0:4096+8192:512`.

## How it measures

- **Correct first.** Every arm runs once on layer 0. Its output is compared with a float32
  reference of the same attention. An arm whose max error is above `--tol` is listed under
  `wrong` and is not timed.
- **Cold KV.** Each call reads its own layer's cache. The script makes enough layers that one
  cycle through them is at least 1 GiB, many times the last-level cache (96 MB of Infinity Cache
  on a W7900D, 64 MB on an R9700). A benchmark that rereads one layer measures the cache, not
  memory: decode on a 16 MB layer can show more than the card's DRAM bandwidth.
- **Pages scattered.** The block table is a random permutation of the cache's blocks, as an
  engine's allocator leaves them after a while.
- **Same process, interleaved.** Every arm is timed in one process: on the W7900D, separate
  processes alone have shown 13-16% differences. Arms alternate, three bursts each a round, and
  the best of `--rounds` is reported.
- **Graph timing for decode.** A decode burst (one call a layer) is captured as a CUDA graph and
  replayed, as vLLM replays its decode steps, so Python and launch overhead are not counted. Each
  kernel launch inside a graph still costs a few microseconds, which counts against arms that
  launch twice (a split and its merge, or vLLM's 3-D kernel and its reduction). Prefill calls take
  milliseconds and run eagerly.

## Reading the output

- `us`: microseconds a call (one layer).
- `gbs` (decode): the KV bytes the call has to read, K and V for every request and kv head, over
  its time. A W7900D reads about 820 GB/s from DRAM (a 1 GiB `torch.sum`).
- `tflops` (prefill): causal attention FLOPs (4 · head_dim · heads · the query-key pairs at or
  below the diagonal) over the time.
- `hk_speedup` (the table's last column): the faster stock kernel's time -- the faster of
  `triton` and `rocm_attn` -- over hk's.
- `splits` (decode): the number of KV-axis splits hk planned for the batch.

## Measured (W7900D, gfx1100)

One W7900D, in the MiniCPM5 production image (vLLM 0.1.dev1+g3775d5fca, ROCm 7.14), 2026-11-02.
Each arm's number is the median over 3 allocations of a shape of its best of 3 rounds (prefill:
one allocation, 2 rounds). Every arm's error against float32 was at most 1e-2 (bf16 output).

In brief:

- **Prefill.** hk is 1.2-2.9x the faster stock kernel on MiniCPM5 and 1.5-4.1x on the head_dim
  256 models, at 37-62 TF/s. ROCM_ATTN's prefix-prefill beats Triton from 8k tokens (d128) and
  everywhere at d256.
- **Decode, head_dim 256** (Qwen3.8, MiniCPM-V, Qwen3.6): hk is 1.02-4.0x the faster stock
  kernel. The large factors come at 1-4 requests and long contexts, where Triton's 16 segments a
  sequence leave the GPU idle. On these hybrid pages ROCM_ATTN runs a Triton fallback with no
  split, at 7-120 GB/s.
- **Decode, head_dim 128** (MiniCPM5): hk is 1.08-1.32x the faster stock kernel at 1-2 requests,
  and 1.0-1.16x from 4 on, where all three read at 600-740 GB/s of the card's ~820. At one
  request with 4-8k tokens, ROCM_ATTN's paged decode (`ll4mi`) is close (0.98-1.08x). On this
  card the hk number at one request also moved by up to 10% from one process to the next, which
  the other kernels' did not.

### Decode: MiniCPM5-2B (16/2 heads, d128, 16-token pages)

| shape | hk: us (GB/s) | rocm_attn: us (GB/s) | triton: us (GB/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| B1 ctx8192 | 26.88 (312.1) | 29.03 (289.0) | 48.16 (174.2) | 1.08x vs rocm_attn |
| B2 ctx8192 | 37.33 (449.5) | 44.56 (376.5) | 56.02 (299.5) | 1.194x vs rocm_attn |
| B4 ctx8192 | 59.77 (561.4) | 70.2 (478.0) | 69.14 (485.3) | 1.157x vs triton |
| B8 ctx8192 | 109.58 (612.4) | 116.96 (573.8) | 111.47 (602.0) | 1.017x vs triton |
| B16 ctx8192 | 200.65 (668.9) | 208.05 (645.1) | 213.32 (629.2) | 1.037x vs rocm_attn |
| B32 ctx8192 | 372.16 (721.3) | 397.21 (675.8) | 403.5 (665.3) | 1.067x vs rocm_attn |
| B1 ctx16384 | 38.7 (433.5) | 45.56 (368.3) | 88.7 (189.2) | 1.177x vs rocm_attn |
| B2 ctx16384 | 60.42 (555.4) | 79.8 (420.5) | 104.11 (322.3) | 1.321x vs rocm_attn |
| B4 ctx16384 | 109.33 (613.8) | 122.12 (549.6) | 127.78 (525.2) | 1.117x vs rocm_attn |
| B8 ctx16384 | 201.44 (666.3) | 211.36 (635.0) | 201.96 (664.6) | 1.003x vs triton |
| B16 ctx16384 | 379.85 (706.7) | 391.41 (685.8) | 394.01 (681.3) | 1.03x vs rocm_attn |
| B32 ctx16384 | 741.58 (724.0) | 770.58 (696.7) | 779.02 (689.2) | 1.039x vs rocm_attn |
| B1 ctx32768 | 61.49 (545.7) | 73.74 (455.0) | 161.7 (207.5) | 1.199x vs rocm_attn |
| B2 ctx32768 | 109.51 (612.8) | 119.7 (560.7) | 192.15 (349.3) | 1.093x vs rocm_attn |
| B4 ctx32768 | 201.14 (667.3) | 210.21 (638.5) | 236.89 (566.6) | 1.045x vs rocm_attn |
| B8 ctx32768 | 380.55 (705.4) | 389.99 (688.3) | 381.18 (704.2) | 1.002x vs triton |
| B16 ctx32768 | 746.08 (719.6) | 764.06 (702.7) | 759.6 (706.8) | 1.018x vs triton |
| B32 ctx32768 | 1448.22 (741.4) | 1527.2 (703.1) | 1490.4 (720.4) | 1.029x vs triton |

### Decode: Qwen3.8-27B TP1 (24/4, d256, 784-token pages)

| shape | hk: us (GB/s) | rocm_attn: us (GB/s) | triton: us (GB/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| B1 ctx8192 | 63.58 (527.7) | 1348.07 (24.9) | 95.72 (350.5) | 1.506x vs triton |
| B2 ctx8192 | 112.71 (595.4) | 1375.39 (48.8) | 117.18 (572.7) | 1.04x vs triton |
| B4 ctx8192 | 196.54 (682.9) | 1440.03 (93.2) | 236.23 (568.2) | 1.202x vs triton |
| B8 ctx8192 | 365.74 (733.9) | 1490.63 (180.1) | 403.3 (665.6) | 1.103x vs triton |
| B16 ctx8192 | 738.48 (727.0) | 1863.71 (288.1) | 753.08 (712.9) | 1.02x vs triton |
| B1 ctx32768 | 201.66 (665.6) | 5305.61 (25.3) | 328.93 (408.0) | 1.631x vs triton |
| B2 ctx32768 | 373.5 (718.7) | 5355.17 (50.1) | 400.55 (670.2) | 1.072x vs triton |
| B4 ctx32768 | 719.54 (746.1) | 5588.45 (96.1) | 880.84 (609.5) | 1.224x vs triton |
| B8 ctx32768 | 1393.17 (770.7) | 5776.39 (185.9) | 1495.79 (717.8) | 1.074x vs triton |
| B16 ctx32768 | 2844.43 (755.0) | 7258.59 (295.9) | 5773.87 (371.9) | 2.03x vs triton |

### Decode: MiniCPM-V-4.7-35B, one TP2 rank (8/1, d256, 544-token pages)

| shape | hk: us (GB/s) | rocm_attn: us (GB/s) | triton: us (GB/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| B1 ctx8192 | 29.11 (288.1) | 1111.98 (7.5) | 83.55 (100.4) | 2.87x vs triton |
| B2 ctx8192 | 40.78 (411.5) | 1111.04 (15.1) | 85.07 (197.2) | 2.086x vs triton |
| B4 ctx8192 | 65.73 (510.5) | 1099.66 (30.5) | 96.5 (347.7) | 1.468x vs triton |
| B8 ctx8192 | 112.45 (596.8) | 1118.89 (60.0) | 118.09 (568.3) | 1.05x vs triton |
| B16 ctx8192 | 199.92 (671.4) | 1209.73 (110.9) | 238.35 (563.1) | 1.192x vs triton |
| B1 ctx32768 | 71.15 (471.6) | 4356.59 (7.7) | 285.62 (117.5) | 4.014x vs triton |
| B2 ctx32768 | 109.69 (611.8) | 4359.15 (15.4) | 289.24 (232.0) | 2.637x vs triton |
| B4 ctx32768 | 201.74 (665.3) | 4337.58 (30.9) | 330.34 (406.3) | 1.637x vs triton |
| B8 ctx32768 | 372.87 (719.9) | 4382.53 (61.3) | 402.01 (667.7) | 1.078x vs triton |
| B16 ctx32768 | 722.58 (743.0) | 4618.63 (116.2) | 901.44 (595.6) | 1.248x vs triton |

### Decode: Qwen3.6-35B-A3B, bf16 KV (16/2, d256, 2064-token pages)

| shape | hk: us (GB/s) | rocm_attn: us (GB/s) | triton: us (GB/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| B1 ctx8192 | 41.56 (403.7) | 1919.1 (8.7) | 86.1 (194.9) | 2.072x vs triton |
| B2 ctx8192 | 66.06 (507.9) | 1900.74 (17.7) | 97.74 (343.3) | 1.48x vs triton |
| B4 ctx8192 | 110.64 (606.5) | 1924.74 (34.9) | 119.21 (562.9) | 1.077x vs triton |
| B8 ctx8192 | 196.72 (682.3) | 1975.31 (67.9) | 240.58 (557.9) | 1.223x vs triton |
| B16 ctx8192 | 365.24 (735.0) | 2029.07 (132.3) | 417.85 (642.4) | 1.144x vs triton |
| B1 ctx32768 | 110.46 (607.5) | 7581.46 (8.9) | 295.12 (227.4) | 2.672x vs triton |
| B2 ctx32768 | 202.41 (663.1) | 7515.94 (17.9) | 335.01 (400.6) | 1.655x vs triton |
| B4 ctx32768 | 370.53 (724.5) | 7536.15 (35.6) | 405.12 (662.6) | 1.093x vs triton |
| B8 ctx32768 | 716.9 (748.9) | 7703.61 (69.7) | 888.89 (604.0) | 1.24x vs triton |
| B16 ctx32768 | 1391.55 (771.6) | 7931.69 (135.4) | 1556.87 (689.7) | 1.119x vs triton |

### Prefill: MiniCPM5-2B (16/2 heads, d128, 16-token pages)

| shape | hk: us (TF/s) | rocm_attn: us (TF/s) | triton: us (TF/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| 0:2048 | 460.71 (37.31) | 1059.05 (16.23) | 898.02 (19.14) | 1.949x vs triton |
| 0:8192 | 4771.16 (57.62) | 10533.19 (26.1) | 13090.88 (21.0) | 2.208x vs rocm_attn |
| 0:16384 | 17659.26 (62.27) | 41780.44 (26.32) | 52488.46 (20.95) | 2.366x vs rocm_attn |
| 16384:2048 | 4820.87 (60.58) | 18211.24 (16.04) | 13924.65 (20.97) | 2.888x vs triton |
| 0:512+0:1024+4096:1 | 474.55 (11.4) | 644.23 (8.4) | 550.62 (9.82) | 1.16x vs triton |

### Prefill: Qwen3.8-27B TP1 (24/4, d256, 784-token pages)

| shape | hk: us (TF/s) | rocm_attn: us (TF/s) | triton: us (TF/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| 0:2048 | 1392.32 (37.04) | 2046.89 (25.19) | 8120.67 (6.35) | 1.47x vs rocm_attn |
| 0:8192 | 18495.46 (44.59) | 27021.8 (30.52) | 129372.46 (6.37) | 1.461x vs rocm_attn |
| 0:16384 | 72269.82 (45.64) | 106404.53 (31.0) | 523140.38 (6.31) | 1.472x vs rocm_attn |
| 16384:2048 | 18786.01 (46.64) | 71985.33 (12.17) | 136979.06 (6.4) | 3.832x vs rocm_attn |
| 0:512+0:1024+4096:1 | 975.37 (16.64) | 1707.45 (9.5) | 3095.31 (5.24) | 1.751x vs rocm_attn |

### Prefill: MiniCPM-V-4.7-35B, one TP2 rank (8/1, d256, 544-token pages)

| shape | hk: us (TF/s) | rocm_attn: us (TF/s) | triton: us (TF/s) | hk vs the faster of triton, rocm_attn |
|---|---|---|---|---|
| 0:2048 | 536.23 (32.05) | 961.11 (17.88) | 1786.39 (9.62) | 1.792x vs rocm_attn |
| 0:8192 | 5978.37 (45.98) | 9474.64 (29.02) | 30262.53 (9.08) | 1.585x vs rocm_attn |
| 0:16384 | 23015.79 (47.77) | 35924.74 (30.61) | 124291.34 (8.85) | 1.561x vs rocm_attn |
| 16384:2048 | 6085.07 (48.0) | 24735.78 (11.81) | 32327.65 (9.03) | 4.065x vs rocm_attn |
| 0:512+0:1024+4096:1 | 579.55 (9.33) | 1178.09 (4.59) | 1264.74 (4.28) | 2.033x vs rocm_attn |
