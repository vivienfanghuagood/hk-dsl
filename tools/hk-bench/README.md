# hk-bench

Measurement tools for hk's kernels: sweeps, A/B harnesses, the scoreboard. Most files here are
working tools; their docstrings say what each measures. The one meant for anyone to rerun is the
kernel benchmark below.

## The kernel benchmark: `kernel_bench.py`

hk's GEMM and MoE kernels against the best alternative on the GPU you run it on, at a real model's
shapes.

```bash
# dense linear layers (bf16): the decode GEMV and the GEMM against hipBLASLt, TunableOp, vLLM, Inductor, FlyDSL, AITER
python3 tools/hk-bench/kernel_bench.py --op linear --model openbmb/MiniCPM5-2B --out linear.json --md linear.md
# MoE experts (bf16) against vLLM's Triton fused_moe, at vLLM's config and at a config searched here
python3 tools/hk-bench/kernel_bench.py --op moe --model openbmb/MiniCPM-V-4_7-35B-A3B --tp 2
# MoE experts with 4-bit AWQ weights against vLLM's moe_wna16
python3 tools/hk-bench/kernel_bench.py --op moe-w4 --model Qwen/Qwen3.6-35B-A3B-AWQ
# 4-bit linear layers: GPTQ against vLLM's RDNA3 and exllama kernels, AWQ against vLLM's AWQ paths
python3 tools/hk-bench/kernel_bench.py --op linear-w4 --model Qwen/Qwen3-8B-AWQ
```

`--model` takes an HF name (from the local cache or the hub), a snapshot directory, or a
`config.json`. `--tp N` shards the shapes as vLLM does at tensor parallel N. `--m` picks the token
counts. `--competitors` limits the optional competitors.

### What is compared

Each competitor is set up the way its own users would set it up for production:

| op | hk | competitors |
|---|---|---|
| `linear` | `hk.ops.decode_gemm` (M up to 64) and `hk.ops.gemm` | hipBLASLt (torch's default); hipBLASLt through **TunableOp, tuned in the same process on exactly these shapes** (every M, M = 1 included, 256 MB rotating buffer); vLLM's `rocm_unquantized_gemm` (its skinny GEMMs at small M, else hipBLASLt + TunableOp); **Inductor** `max-autotune` (its Triton templates and aten, autotuned before timing); FlyDSL's GEMM; AITER's tuned GEMM |
| `linear-w4` | `hk.ops.w4a16.matmul` in the form its vLLM route loads (GPTQ: symmetric; AWQ: with zero points), up to 64 tokens | GPTQ: vLLM's `gptq_gemm_rdna3` (what its GPTQ layers run on gfx11) and `gptq_gemm` (exllama). AWQ: vLLM's `awq_dequantize` + `torch.matmul` (what its compiled graph runs) and its Triton `awq_gemm`. Layers the checkpoint leaves in 16 bits are skipped |
| `moe`, `moe-w4` | `hk.ops.moe.moe_gemm` / `moe_gemm_w4` (BLOCK_M 16, the schedule hk's vLLM route picks) | vLLM's Triton `fused_moe` / `moe_wna16`, at **the config vLLM picks and at a config searched in the same process** at each token count over 108 configs (BLOCK_M/N/K, GROUP_M, warps), as vLLM's `benchmark_moe` does; FlyDSL and AITER |

A competitor that does not run on the GPU (for example AITER on RDNA) is listed as n/a, with the reason. Each table states the version of every competitor and each arm's configuration. The JSON (`--out`) records every round.

### How it is timed

- **One process.** Every arm runs in round-robin rounds (5 by default), and the fastest round is reported, with the spread. Comparing separate processes is unreliable on these cards: drift between processes once produced a 13-16% difference out of nothing.
- **Graph timing.** Each arm's calls are captured once as a CUDA graph, and the replays are timed. No arm pays host or launch cost, as in a serving engine's captured decode step.
- **Cold weights.** Every call reads its own weights. Each shape's copies (or each MoE layer's experts) cover `--rotate-mb` (576 MB: six times a W7900D's Infinity Cache), so nothing is read from the cache. Each MoE call has its own routing, and a graph replays enough calls (up to 64) that the experts they read cover `--rotate-mb` too: at a few tokens a call reads only its top-k experts.
- **Correctness first.** Every arm is checked against an fp32 reference (linear) or vLLM's own kernel (MoE) before it is timed. Error is max |error| / max |reference|. An arm above 0.1 is shown as `wrong` and not timed. An arm between 0.02 and 0.1 is timed and marked `†`: looser than one rounding of a 16-bit output with fp32 accumulation. The JSON has each arm's error.
- **Conservative for hk.** Back-to-back calls of one shape favour a tuned hipBLASLt kernel. Inside a model's step it ran 2-5 us slower (hk-dsl #189). The ratios here are therefore lower bounds on what hk gains in serving.

### Reading a table

| column | meaning |
|---|---|
| hk, us | the faster of hk's kernels for the shape |
| best competitor | the fastest competitor at that row, named |
| hk vs best | best competitor's time / hk's time (above 1: hk is faster) |
| one column per arm | that arm's time, `wrong` or `error`; `--` where it doesn't take the shape |

hk's routes in vLLM and SGLang send a shape to hk only where hk wins it (`python/hk/tuned/routes_*`). A row where a competitor wins is a shape the route leaves on that engine.

## Results, 2026-11-02 (hk-dsl 2ac79b7)

One W7900D (gfx1100) and one Radeon AI PRO R9700 (gfx1201), each in the vLLM image its production serving uses. "hk vs best" is against the fastest competitor at each row. A row counts at most once in the columns below; the full tables, with every arm's time, version and configuration, are linked.

| arch | op | model (TP) | rows | hk ≥ 1.15x | 0.95–1.15x | hk < 0.95x | best ratio | table |
|---|---|---|---:|---:|---:|---:|---:|---|
| gfx1100 | linear | openbmb/MiniCPM5-2B (TP1) | 48 | 9 | 26 | 13 | 1.45x | [minicpm5-linear.md](results/2026-11-02/gfx1100/minicpm5-linear.md) |
| gfx1100 | linear | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 68 | 14 | 30 | 24 | 2.23x | [minicpm-v-tp2-linear.md](results/2026-11-02/gfx1100/minicpm-v-tp2-linear.md) |
| gfx1100 | moe | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 9 | 5 | 3 | 1 | 1.81x | [minicpm-v-tp2-moe.md](results/2026-11-02/gfx1100/minicpm-v-tp2-moe.md) |
| gfx1100 | moe-w4 | Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) | 8 | 8 | 0 | 0 | 2.21x | [qwen3.6-awq-moe-w4.md](results/2026-11-02/gfx1100/qwen3.6-awq-moe-w4.md) |
| gfx1100 | linear | Qwen/Qwen3.8-27B (TP2) | 69 | 4 | 45 | 20 | 2.73x | [qwen3.8-tp2-linear.md](results/2026-11-02/gfx1100/qwen3.8-tp2-linear.md) |
| gfx1100 | linear-w4 | palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) | 48 | 25 | 19 | 4 | 2.56x | [qwen3.8-gptq-linear-w4.md](results/2026-11-02/gfx1100/qwen3.8-gptq-linear-w4.md) |
| gfx1100 | linear-w4 | Qwen/Qwen3-8B-AWQ (TP1) | 32 | 28 | 0 | 0 | 2.59x | [qwen3-8b-awq-linear-w4.md](results/2026-11-02/gfx1100/qwen3-8b-awq-linear-w4.md) |
| gfx1100 | linear-w4 | Qwen/Qwen3-14B-AWQ (TP1) | 32 | 28 | 0 | 0 | 2.42x | [qwen3-14b-awq-linear-w4.md](results/2026-11-02/gfx1100/qwen3-14b-awq-linear-w4.md) |
| gfx1201 | linear | openbmb/MiniCPM5-2B (TP1) | 44 | 0 | 31 | 13 | 1.02x | [minicpm5-linear.md](results/2026-11-02/gfx1201/minicpm5-linear.md) |
| gfx1201 | linear | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 68 | 5 | 29 | 34 | 1.84x | [minicpm-v-tp2-linear.md](results/2026-11-02/gfx1201/minicpm-v-tp2-linear.md) |
| gfx1201 | moe | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 9 | 0 | 5 | 4 | 1.11x | [minicpm-v-tp2-moe.md](results/2026-11-02/gfx1201/minicpm-v-tp2-moe.md) |
| gfx1201 | moe-w4 | Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) | 8 | 8 | 0 | 0 | 1.91x | [qwen3.6-awq-moe-w4.md](results/2026-11-02/gfx1201/qwen3.6-awq-moe-w4.md) |
| gfx1201 | linear | Qwen/Qwen3.8-27B (TP2) | 69 | 2 | 52 | 15 | 1.31x | [qwen3.8-tp2-linear.md](results/2026-11-02/gfx1201/qwen3.8-tp2-linear.md) |
| gfx1201 | linear-w4 | palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) | 48 | 44 | 4 | 0 | 3.68x | [qwen3.8-gptq-linear-w4.md](results/2026-11-02/gfx1201/qwen3.8-gptq-linear-w4.md) |
| gfx1201 | linear-w4 | Qwen/Qwen3-8B-AWQ (TP1) | 32 | 28 | 0 | 0 | 2.75x | [qwen3-8b-awq-linear-w4.md](results/2026-11-02/gfx1201/qwen3-8b-awq-linear-w4.md) |
| gfx1201 | linear-w4 | Qwen/Qwen3-14B-AWQ (TP1) | 32 | 28 | 0 | 0 | 2.21x | [qwen3-14b-awq-linear-w4.md](results/2026-11-02/gfx1201/qwen3-14b-awq-linear-w4.md) |

**Reading.**
- **MoE with 4-bit AWQ experts: hk is ahead everywhere it was measured.** It is 1.82-2.21x faster on gfx1100 and 1.65-1.91x on gfx1201 than vLLM's `moe_wna16` at the best of 108 configs searched at each token count, from 1 to 256 tokens (2.5-4.5x against the config vLLM picks).
- **Dense AWQ int4: hk is ahead everywhere it was measured**, 1.39-2.59x on gfx1100 and 1.53-2.75x on gfx1201 against vLLM's Triton `awq_gemm` at 1-64 tokens (the four rows a table has at 256 tokens are past hk's decode kernel, `--`). vLLM's AWQ layers under torch.compile run the dequantize + matmul branch at every batch (docs/history/22), against which hk is 1.5-9.2x.
- **Dense GPTQ int4** (the layers the checkpoint quantizes; it keeps `in_proj_ba` in 16 bits): hk is ahead on every shape from M = 8 on gfx1100 (1.15-2.56x) and on gfx1201 (1.36-3.68x). At M = 1-4 on gfx1100, hk and vLLM's RDNA3 kernel are about even (0.92-1.14x; `down` 1.07-1.32x). On gfx1201, where vLLM's RDNA3 kernel doesn't run (`error` in the table) and the exllama kernel is the competitor, hk is 1.07-2.53x from M = 1.
- **MoE with bf16 experts:** on gfx1100, hk is 1.19-1.81x at 1-16 tokens against the best searched config and about even at 32-256 (1.04-1.08x). On gfx1201 it is about even at 1-8 tokens (1.03-1.11x) and behind from 16 (0.67-0.95x); hk's vLLM route there takes 1-2 tokens only.
- **Dense bf16 linear:** hk is clearly ahead only on small or latency-bound shapes at M ≈ 4-32 (`ba`, the shared expert, `o`/`out`: 1.15-2.7x). The large decode GEMVs run at memory bandwidth for hk and tuned hipBLASLt alike, so they tie (0.94-1.07x). hk loses at M = 1 to Inductor's Triton GEMV, and at M = 64-256 to TunableOp-tuned hipBLASLt and FlyDSL.
- hk's vLLM routes send a (shape, M) to hk only where it measured faster in the model's context (`hk.autotune.routes`, timing "context"). The rows where a competitor wins stay on that engine.

### Route-level gain of each default-on route

What a route gains over the layers it covers, each shape weighted by how many times a decode step
calls it: (sum of the engine's times) / (sum of the routed times), at M tokens a step. A routed
shape takes hk's time and the others the engine's. The engine is what vLLM runs without hk (`vLLM`
for dense layers, `gptq_gemm_rdna3`, vLLM's Triton MoE at its own config). `vllm_gemm` sends a
(shape, M) to hk where it measured faster in context by 3% (the run-to-run noise's 90th
percentile, docs/history/32); here that decision is approximated from these tables.

| route | arch | model (TP) | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm_gptq | gfx1100 | Qwen3.8-27B-GPTQ-Int4 (TP1) | 1.01x (in serving 1.13x) | 1.01x | 1.10x | 1.22x | 1.98x | 2.50x | 1.33x | -- |
| vllm_moe | gfx1100 | MiniCPM-V-4.7-35B-A3B (TP2) | 1.73x | 1.85x | 1.94x | 1.65x | 1.36x | 1.18x | 1.11x | 1.23x |
| vllm_moe | gfx1201 | MiniCPM-V-4.7-35B-A3B (TP2) | 1.07x | 1.22x | -- | -- | -- | -- | -- | -- |
| vllm_gemm | gfx1100 | MiniCPM5-2B (TP1) | 1.00x | 1.07x | 1.11x | 1.13x | 1.11x | 1.07x | 1.00x | 1.01x |
| vllm_gemm | gfx1100 | MiniCPM-V-4.7-35B-A3B, dense layers (TP2) | 1.00x | 1.09x | 1.13x | 1.21x | 1.19x | 1.14x | 1.04x | 1.00x |
| vllm_gemm | gfx1100 | Qwen3.8-27B (TP2) | 1.00x | 1.00x | 1.01x | 1.02x | 1.03x | 1.05x | 1.00x | 1.00x |
| vllm_gemm | gfx1201 | MiniCPM5-2B (TP1) | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| vllm_gemm | gfx1201 | MiniCPM-V-4.7-35B-A3B, dense layers (TP2) | 1.00x | 1.00x | 1.00x | 1.04x | 1.04x | 1.02x | 1.00x | 1.00x |
| vllm_gemm | gfx1201 | Qwen3.8-27B (TP2) | 1.00x | 1.00x | 1.00x | 1.02x | 1.02x | 1.00x | 1.00x | 1.00x |

`vllm_moe` on gfx1201 takes 1-2 tokens (`vllm_moe.BLOCK16_MAX_TOKENS_AT`); `vllm_gptq` takes 1-64.
`vllm_int4` (AWQ linear layers, 1-64 tokens), over Qwen3-8B-AWQ / Qwen3-14B-AWQ at M = 1, 8, 32 and 64: against vLLM's dequantize + matmul 7.4 / 7.3 / 4.9 / 3.1x and 8.0 / 7.9 / 5.0 / 2.7x on gfx1100, 7.5 / 7.2 / 5.7 / 3.3x and 8.3 / 8.1 / 6.1 / 3.4x on gfx1201; against its Triton awq_gemm 2.3 / 2.3 / 1.7 / 1.9x and 2.4 / 2.4 / 1.6 / 1.6x on gfx1100, 2.1 / 2.1 / 1.8 / 2.0x and 2.1 / 2.1 / 1.7 / 1.9x on gfx1201. The SGLang routes have no table here yet.

### The GPTQ benchmark against serving

At one token on gfx1100, a trace of Qwen3.8-27B-GPTQ-Int4 served by vLLM gives each shape's time
inside the decode step. That trace is rocprofv3 over 20 decode steps of one 8k request, with
medians of 1280 calls per shape. vLLM's kernel there matches this benchmark within 1-3%. hk's is
12-14% faster in the server than here on the shapes with N of 14336 or more, so this benchmark
under-rates hk on them. vLLM's per-call fill kernel is in both (it is part of
`gptq_gemm_rdna3`). The route takes a shape by its in-server time.

| shape (N x K) | hk, bench | hk, in server | vLLM, bench | vLLM, in server | hk vs vLLM, bench | in server |
|---|---:|---:|---:|---:|---:|---:|
| qkv (14336 x 5120) | 67.5 | 59.4 | 61.8 | 62.1 | 0.92x | 1.05x |
| o, out (5120 x 6144) | 33.2 | 35.4 | 31.6 | 31.9 | 0.95x | 0.90x |
| qkvz (16384 x 5120) | 78.4 | 68.1 | 72.8 | 74.0 | 0.93x | 1.09x |
| gate_up (34816 x 5120) | 162.2 | 139.8 | 161.3 | 166.2 | 0.99x | 1.19x |
| down (5120 x 17408) | 84.1 | 74.2 | 92.8 | 82.7 | 1.10x | 1.12x |

Over the step this is 1.116x in serving with every shape on hk, and 1.128x with o and out left
on vLLM's kernel at 1-2 tokens, which is what the route does (`vllm_gptq.MIN_M_NK`).
