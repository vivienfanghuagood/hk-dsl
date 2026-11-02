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
```

`--model` takes an HF name (from the local cache or the hub), a snapshot directory, or a
`config.json`. `--tp N` shards the shapes as vLLM does at tensor parallel N. `--m` picks the token
counts. `--competitors` limits the optional competitors.

### What is compared

Each competitor is set up the way its own users would set it up for production:

| op | hk | competitors |
|---|---|---|
| `linear` | `hk.ops.decode_gemm` (M up to 64) and `hk.ops.gemm` | hipBLASLt (torch's default); hipBLASLt through **TunableOp, tuned in the same process on exactly these shapes** (every M, M = 1 included, 256 MB rotating buffer); vLLM's `rocm_unquantized_gemm` (its skinny GEMMs at small M, else hipBLASLt + TunableOp); **Inductor** `max-autotune` (its Triton templates and aten, autotuned before timing); FlyDSL's GEMM; AITER's tuned GEMM |
| `moe`, `moe-w4` | `hk.ops.moe.moe_gemm` / `moe_gemm_w4` (BLOCK_M 16, the schedule hk's vLLM route picks) | vLLM's Triton `fused_moe` / `moe_wna16`, at **the config vLLM picks and at a config searched in the same process** at each token count over 108 configs (BLOCK_M/N/K, GROUP_M, warps), as vLLM's `benchmark_moe` does; FlyDSL and AITER |

A competitor that does not run on the GPU (for example AITER on RDNA) is listed as n/a, with the reason. Each table states the version of every competitor and each arm's configuration. The JSON (`--out`) records every round.

### How it is timed

- **One process.** Every arm runs in round-robin rounds (5 by default), and the fastest round is reported, with the spread. Comparing separate processes is unreliable on these cards: drift between processes once produced a 13-16% difference out of nothing.
- **Graph timing.** Each arm's calls are captured once as a CUDA graph, and the replays are timed. No arm pays host or launch cost, as in a serving engine's captured decode step.
- **Cold weights.** Every call reads its own weights. Each shape's copies (or each MoE layer's experts) cover `--rotate-mb` (576 MB: six times a W7900D's Infinity Cache), so nothing is read from the cache. MoE calls also rotate through 8 routings.
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
| gfx1100 | moe | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 9 | 5 | 3 | 1 | 1.56x | [minicpm-v-tp2-moe.md](results/2026-11-02/gfx1100/minicpm-v-tp2-moe.md) |
| gfx1100 | moe-w4 | Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) | 8 | 8 | 0 | 0 | 2.12x | [qwen3.6-awq-moe-w4.md](results/2026-11-02/gfx1100/qwen3.6-awq-moe-w4.md) |
| gfx1100 | linear | Qwen/Qwen3.8-27B (TP2) | 69 | 4 | 45 | 20 | 2.73x | [qwen3.8-tp2-linear.md](results/2026-11-02/gfx1100/qwen3.8-tp2-linear.md) |
| gfx1100 | linear-w4 | palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) | 49 | 25 | 10 | 14 | 4.34x | [qwen3.8-gptq-linear-w4.md](results/2026-11-02/gfx1100/qwen3.8-gptq-linear-w4.md) |
| gfx1201 | linear | openbmb/MiniCPM5-2B (TP1) | 44 | 0 | 31 | 13 | 1.02x | [minicpm5-linear.md](results/2026-11-02/gfx1201/minicpm5-linear.md) |
| gfx1201 | linear | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 68 | 5 | 29 | 34 | 1.84x | [minicpm-v-tp2-linear.md](results/2026-11-02/gfx1201/minicpm-v-tp2-linear.md) |
| gfx1201 | moe | openbmb/MiniCPM-V-4.7-35B-A3B (TP2) | 9 | 0 | 4 | 5 | 1.10x | [minicpm-v-tp2-moe.md](results/2026-11-02/gfx1201/minicpm-v-tp2-moe.md) |
| gfx1201 | moe-w4 | Qwen/Qwen3.6-35B-A3B (an AWQ 4-bit checkpoint) (TP1) | 8 | 8 | 0 | 0 | 2.36x | [qwen3.6-awq-moe-w4.md](results/2026-11-02/gfx1201/qwen3.6-awq-moe-w4.md) |
| gfx1201 | linear | Qwen/Qwen3.8-27B (TP2) | 69 | 2 | 52 | 15 | 1.31x | [qwen3.8-tp2-linear.md](results/2026-11-02/gfx1201/qwen3.8-tp2-linear.md) |
| gfx1201 | linear-w4 | palmfuture/Qwen3.8-27B-GPTQ-Int4 (TP1) | 49 | 42 | 7 | 0 | 4.68x | [qwen3.8-gptq-linear-w4.md](results/2026-11-02/gfx1201/qwen3.8-gptq-linear-w4.md) |

**Reading.**
- **MoE with 4-bit AWQ experts: hk is ahead everywhere it was measured.** It is 1.85-2.12x faster on gfx1100 and 1.62-2.36x on gfx1201 than vLLM's `moe_wna16` at the best of 108 configs searched at each token count, from 1 to 256 tokens.
- **Dense GPTQ int4:** hk is ahead on every shape from M = 16 on gfx1100 (1.30-4.33x) and from M = 8 on gfx1201 (1.23-4.68x). At M = 1-4 on gfx1100, vLLM's RDNA3 kernel wins on the large shapes (hk 0.80-0.94x), while hk stays ahead on the small ones (`ba` 2.0-3.0x). On gfx1201, where vLLM's RDNA3 kernel doesn't run (`error` in the table) and the exllama kernel is the competitor, hk is even at M = 1-2 (0.97-0.99x on the large shapes).
- **MoE with bf16 experts:** hk is 1.20-1.56x at 1-16 tokens on gfx1100 and about even at 32-256. On gfx1201 it is not ahead (0.67-1.10x).
- **Dense bf16 linear:** hk is clearly ahead only on small or latency-bound shapes at M ≈ 4-32 (`ba`, the shared expert, `o`/`out`: 1.15-2.7x). The large decode GEMVs run at memory bandwidth for hk and tuned hipBLASLt alike, so they tie (0.94-1.07x). hk loses at M = 1 to Inductor's Triton GEMV, and at M = 64-256 to TunableOp-tuned hipBLASLt and FlyDSL.
- hk's vLLM routes send a (shape, M) to hk only where it measured faster in the model's context (`hk.autotune.routes`, timing "context"). The rows where a competitor wins stay on that engine.
