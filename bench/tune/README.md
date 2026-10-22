# The tuning benchmark

What judges `hk.tune` (and, later, an agent that tunes): real model GEMM
shapes no shipped table holds, per GPU, against every baseline that is
available there, measured the project's way. The goal and its acceptance are
[docs/design/tuning-goal.md](../../docs/design/tuning-goal.md) (T3); the tuner
is [docs/tune.md](../../docs/tune.md).

## The shapes

`shapes_gfx950.json` and `shapes_gfx1100.json`, drawn by `shapes.py` with a
fixed seed and committed (a test holds that the files are what the seed
draws):

- bf16 C = A @ B.T, B as (N, K) -- F.linear's layout -- from the linear
  layers (qkv, o, gate_up, down) of models hk has not tuned: Llama-3-70B
  (TP 4, 8), Mixtral-8x7B (TP 2, 4; experts as dense GEMMs), Qwen2.5-32B
  (TP 1, 2, 4), Qwen2.5-72B (TP 4, 8), Gemma-2-27B (TP 1, 2, 4), Phi-3-medium
  (TP 1, 2), Phi-3-mini, Mistral-Nemo-12B (TP 1, 2), Llama-3.2-3B, and
  DeepSeek-V2-Lite's dense layers (MLA projections, the dense MLP, the shared
  experts);
- no (N, K) any shipped table or record holds (`python/hk/tuned/*.json`,
  either GPU), and none hk's GEMMs cannot take (N % 128, K % 64; the excluded
  ones are listed in the file with the reason -- Qwen2.5-72B at TP 8 is one);
- M in decode (1, 8, 16), 128-512, 1024-2048 and 4096-8192, six of each
  class a split, under a FLOP cap per GPU (a call of at most ~10 ms);
- grouped by (N, K), shuffled by the seed and dealt to a **tuning** split
  (where hk.tune was developed) and an **acceptance** split (where it is
  judged, untouched during development); every M of an (N, K) lands in the
  same split. 2048x16384x4096 -- named by the goal, 0.76x hipBLASLt on the
  route rule -- is in gfx950's acceptance split by hand.

## The baselines (`baselines.py`)

| name | what | GPUs |
|---|---|---|
| `torch` | `F.linear`: hipBLASLt's default heuristic | both |
| `tunableop` | `F.linear` under PyTorch TunableOp: hipBLASLt and rocBLAS solutions searched at the first call -- hipBLASLt tuned, as a user reaches it | both |
| `aiter` | aiter's tuned GEMM dispatcher (`tgemm.mm`): its tuned CSVs (asm, hipBLASLt solutions, FlyDSL, Triton, skinny) where it has the shape, else its default | gfx950 |
| `aiter_triton` | aiter's Triton `gemm_a16w16` with aiter's per-arch configs | gfx950 |
| `flydsl` | FlyDSL's GEMM: gfx950 `gemm_a16w16_gfx950` at its default schedule (as aiter ships it); gfx1100 its RDNA3 GEMM at its dispatcher's tile | both |
| `hipkittens` | HipKittens' CDNA4 GEMM (`hk.ops.hipkittens`), where its 256x256 tiles take the shape | gfx950 |
| `triton` | the Triton tutorial matmul autotuned per shape over blocks, warps, stages, `waves_per_eu` (and MFMA size, `kpack` on CDNA) -- `triton_gemm.py` | both |

Every arm is checked against fp32 (relative max error < 2e-2) before it is
timed; an arm that refuses a shape or is wrong is dropped from "best
available" for that shape, with its reason in the results. **Best available**
is the fastest correct baseline per shape.

Where they run: gfx950 in the vLLM ROCm image (`vllm/vllm-openai-rocm`
nightly-rocm100, the one engine image with aiter: ROCm 10, torch 2.12, aiter
0.1.25, FlyDSL 0.3.4.1, Triton 3.8), with hk built there
(`~/hk-dx/bin/hkx950v`), so every arm is in one process; gfx1100 in the
hk-rdna3 pod (ROCm 7.2.4, torch 2.10, Triton 3.6, FlyDSL 0.4.0.dev).

## Running it

```
python3 -m bench.tune.run --split acceptance --only 0:3 --out DIR    # tune, then score; a few shapes a call
python3 -m bench.tune.run --split acceptance --score-only --out DIR  # score with DIR/records as they are
python3 -m bench.tune.scoreboard DIR --write bench/tune/results      # results/<arch>-<date>.{json,md}
```

Per shape, `run.py`:

1. **tunes** it with `hk.tune.tune` at the default budget, writing its record
   into `DIR/records` (`HK_TUNE_DIR`), never into the package;
2. **scores** it in one process: hk before (the current route), hk after
   (`hk.ops.matmul`, which reads the record) and every baseline, in 3 sets of
   7 rounds on fresh activations, interleaved, each pass streaming three times
   the Infinity Cache in weight copies and queued behind a GPU spin longer
   than its launch (so a slow launcher is not timed as a slow kernel), an
   arm's time the median over sets of each set's median round;
3. appends a line to `DIR/<arch>-<split>.jsonl`.

The ratio of a shape is best baseline time / hk time (>1: hk faster), the
median over the sets and their range. The scoreboard: the share of shapes at
>= 0.95x, the median and worst ratio and the GPU-minutes of tuning a shape,
before and after.

## Results

`results/<arch>-<date>.{json,md}`; the acceptance split is what is reported
(`docs/tune.md` summarizes). Development runs on the tuning split are kept
beside them as `results/<arch>-<date>-tuning.*`.
