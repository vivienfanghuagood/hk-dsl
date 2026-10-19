# The agent loop: where a step's time goes, whether a kernel helped, and a new op on the rails

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Two questions come before every kernel this project writes: where should hk
hit next, and did it help. Until now each answer was a harness written for
the occasion. `tools/hk-bench/vllm_profile.py` profiled one whole `generate`,
and that is where the 39% / 0.9% split of [chapter 07](07-against-the-engine.md)
came from. `ab` printed tables of ratios for a person to read. A new kernel
started from a copy of whichever example was nearest. This chapter makes
the three into commands, with machine-readable output, so an agent can run
the loop without inventing a measurement:

```bash
python -m hk.integration headroom --model M --batch B --context C --json h.json
python -m hk.integration new-op NAME --dir DIR
python -m hk.integration ab --model M --arms stock,ARM --json v.json
```

[docs/agents.md](../agents.md) is the manual. Everything below was measured
in the engines' pod (vLLM 0.31.0, torch 2.13.0+rocm7.2, FlyDSL 0.4.0), on
GPU 3, Qwen3-8B in bf16, with `hk._mlir` built in a directory of its own.

## `headroom`

**A step, not a request.** `vllm_profile.py` profiled a whole `generate`,
which mixes the batch's prefill with its decode steps in proportions set by
`--tokens`. `headroom` profiles the two phases apart. For decode, it waits
until every request is past its prefill, then runs `--windows` windows of
`--steps` steps at the full batch without the profiler and one window under
it. For prefill, the batch generates one token each, which is how `ab`
times prefill. vLLM's steps are driven one at a time (`LLMEngine.step`, its
engine core in the process) and counted where the scheduler hands them out,
with their token and request counts. So a window of 8 steps is 8 steps even
under vLLM 0.31's asynchronous scheduling, and the prefill's M is known: at
16 x 4096 without chunked prefill, 8 steps of 8192 tokens, two prompts each.
`--tokens T` adds chapter 07's view, the shares over a prefill and T - 1
decode steps.

**The trace, not the summary.** Device time comes from the profiler's
chrome trace. On this ROCm `key_averages()` reports
`self_device_time_total = 0` for every kernel, which is the trap
`vllm_profile.py` already recorded.

**Families, and whose kernel.** Each kernel goes into a family by its name:
linear, attention, norm, activation, rotary, kv_write, sampling, embedding,
elementwise, copy, or other. The patterns were written from vLLM 0.31's
eager and compiled traces, Inductor's names and hk's own. A kernel is hk's
when it is `<name>_kernel(globals)`, the signature hk generates; FlyDSL's
GEMM (`wmma_gemm_kernel_<n>`) counts as hk's choice, since only hk's table
routes calls to it. Each family is then set against what hk has: the
`hk.ops` that compute it, the `ab` arm and switch that route it, and for the
linear layers the routing table's path for each of the model's five (N, K)
at this step's M.

**A share is not a saving.** Each row gives the family's device time per
step as its **upper bound**: what the step would save if the family took
zero time, and less where the host holds the step up. The rank is that
bound times a **ceiling**, the fraction hk could credibly take, 1 - 1/r for
a ratio r of hk's kernel to the engine's. The ratio comes from, in order:
1. another arm in the same run (`--arms stock,hk+gemm`), where that arm ran
   hk's kernels for the family;
2. the routing table's per-call times at this M, for linear layers;
3. the record, cited, and only against the kernel it was measured on. The
   first version applied chapter 20's 0.83-0.95x for hk's decode attention
   to stock 0.31's `kernel_paged_attention_2d`. That ratio was measured
   against TRITON_ATTN's kernel, and on stock's it gave the family a
   ceiling of 0, which was wrong (below). A record entry now names the
   kernel it holds for.

Where no ratio exists, the ceiling is the **roofline**, labelled "at most":
the card's peak against the family's work, computed from the model's config
and the step. For the linear layers each call is max(FLOPs / 122.6 TFLOP/s,
bytes / 864 GB/s). For decode attention the KV cache is streamed once; for
prefill attention it is 2 h n^2 d FLOPs a layer and causal prompt. The
peaks are `_rounds.py`'s for gfx1100, and an arch without them gets no
roofline. The step's wall time the device was idle is a row of its own: no
kernel wins it back.

**CUDA graphs.** The default is eager. `--graphs` runs the engine as it
ships, and on this ROCm the replays are seen: a decode window of 8 steps
held 8 graph launches and all 145 linear calls a step that eager makes. At
16 x 4096 the linear and attention families agreed between the two modes
within 0.4% (39.71 / 39.63 ms, 28.48 / 28.38 ms). The small families differ
because torch.compile fuses them (the norms 0.57 ms eager, 0.24 compiled). A
trace whose graph launches leave most of the step without device time did
not see its replays; that phase is reported as blind, not ranked.

## What it says about Qwen3-8B

**Stock vLLM 0.31, 16 requests x 4096, eager**, with `hk+gemm` beside it.
Per step; "of peak" is the family's roofline over its time:

| phase | family | share of step | ms | of peak | ratio, measured in this run | score |
|---|---|---|---|---|---|---|
| decode (71.3 ms) | linear | 55.8% | 39.76 | 44% | 2.00x (route table at M=16: 1.85x) | 19.9 |
| | attention | 40.1% | 28.59 | 39% | 1.71x | 11.9 |
| | GPU idle | 1.6% | 1.17 | | | 1.2 |
| prefill (2292 ms) | linear | 75.9% | 1740.8 | 53% | 1.39x (table at M=8192: 1.32x) | 489.9 |
| | attention | 20.6% | 473.2 | 17% | 2.36x | 272.6 |
| | norm, activation | 1.3% each | 30.6, 30.5 | | none | 30.6, 30.5 |

`hk+gemm`'s steps were 39.7 ms (1.80x) and 1528 ms (1.50x). The score is a
prediction: the linear layers' 19.9 ms is what hk's GEMV should take off a
decode step. The `ab` run below, the `gemm` arm alone under vLLM's defaults
(graphs, a step inferred from two lengths), shortened the decode step at
4096 x 16 from 69.72 to 49.86 ms, by 19.9 ms. Chapter 35's run of the same
measured 20.8.

**The same run, ranked from `hk+gemm`**: where hk should hit next once its
arms are on.

| phase | family | share | ms | of peak | ceiling (roofline) | score |
|---|---|---|---|---|---|---|
| decode (39.7 ms) | attention, hk's paged decode | 42.1% | 16.72 | 67% | 0.33 | 5.5 |
| | linear, hk's GEMV | 50.1% | 19.88 | 89% | 0.11 | 2.3 |
| prefill (1528 ms) | linear, hk's GEMM and FlyDSL's | 81.8% | 1251.0 | 74% | 0.26 | 321.2 |
| | attention, hk's paged prefill | 13.1% | 200.7 | 40% | 0.60 | 120.0 |

**TRITON_ATTN**, the backend hk's replaces and the one the record measured
against, with `hk` beside it:

| phase | family | share | ms | ratio, measured | score |
|---|---|---|---|---|---|
| decode (57.5 ms) | linear | 69.3% | 39.83 | 1.85x (table) | 18.3 |
| | attention | 25.4% | 14.62 | 0.87x | 0 |
| prefill (3668 ms) | attention | 53.8% | 1972.7 | 9.25x | 1759.4 |
| | linear | 44.1% | 1617.5 | 1.32x (table) | 391.9 |

Over a request of 24 tokens, chapter 07's view: prefill attention 51.5% of
the device time, decode attention 1.1%. Chapter 07 measured 39.1% and 0.9%
on vLLM 0.16. hk's decode attention measured 0.87x TRITON_ATTN's, which
chapter 20 recorded as 0.83-0.95x. Its prefill attention measured 9.25x,
where chapter 17 had 3.6-7.9x at its shapes.

**One request x 1024**: the decode step (24.9 ms eager, 24.1 with graphs)
ranks attention first: stock's kernel at 4% of its roofline, 4.2 ms a step
for 4 MB of KV a layer. That is consistent with chapter 20, where
TRITON_ATTN's step at 1024 x 1 was 4.3 ms shorter than stock's. The linear
layers rank last: the table leaves M = 1 on vLLM's skinny GEMMs (ratio 1.0),
which run at 94% of the DRAM roofline, and that is chapter 35's "at 1 and 4
requests the gain is prefill's alone".

**SGLang 0.5.18**, 4 requests x 1024, eager, stock: the decode step (46.0
ms) is 85.4% linear layers, F.linear's GEMMs at 45% of the DRAM roofline,
and SGLang's table puts hk's GEMV at 1.90x per call at M = 4, a score of
18.6 ms a step. Chapter 24 found the same GEMVs at 38% of bandwidth and
served this path 1.49-1.81x. The prefill (one step of 4096 tokens, 913 ms)
is 89% linear, at 57% of the bf16 peak, 1.30x by the table. Getting there
took three runs:
- With the CPU's activity in the profile, SGLang's scheduler segfaulted in
  the profiler's op recording when its by-stage profile moved from the
  prefill steps to the decode steps. `headroom` records the GPU's alone,
  one stage a session.
- The radix cache was on, and the same prompts were used for the warm,
  timed and profiled runs. A prefill of cached prompts is not a prefill:
  the cache is off now, as `ab` turns vLLM's prefix caching off, and every
  run has prompts of its own.
- SGLang's overlap scheduler schedules a step while the GPU runs the one
  before, so the windows cut steps. A decode window held 7.2 steps' kernels
  and 8 embedding lookups, and the prefill window held 20 of a decode
  step's 36 attention calls. With the overlap scheduler off, a step is
  counted from the trace as the calls of the attention kernel each layer
  makes once.

**What the record did not have.** Stock vLLM 0.31's decode attention is 40%
of a decode step at 16 x 4096. Chapter 07's 0.9% was a share of a whole
request, on 0.16 and another backend, and it does not carry to a step of
0.31's default backend. Chapter 20's step times (stock 72.1 ms against hk's
backend 59.0 at 4096 x 16) had the gap without saying where it was. And the
ratio that says hk's decode kernel loses is against TRITON_ATTN's kernel.
Against the backend vLLM 0.31 picks on this card, hk's is 1.71x.

## `ab --json`

`ab` keeps its tables and adds a verdict per shape, metric (total, prefill,
decode step) and arm, every row with the same keys:

```
{shape, metric, arm, ms, stock_ms, ratio, band, verdict, why, rounds, noise, tokens_same}
```

- `ms` and `stock_ms` are the medians over rounds, each round the best of
  its reps.
- `ratio` is stock / arm per round, rounds paired by number (a pair ran
  back to back, arms rotating), then the median.
- `band` is max(2%, noise), with noise measured the way the performance gate
  measures it: the larger of the ratio's spread across rounds (max |r_i /
  median - 1|) and the two arms' rep spreads within a round, summed.
- `verdict`: `win` or `loss` past the band; `tie` within a band of at most
  5%; `noise` for one round, a band past 15% (the gate's cap), or a ratio
  within a band wider than 5%. A band that wide cannot tell a tie from the
  2-5% this project wins and loses by, so calling it a tie would say more
  than was measured.

The first version also had "noise: rounds beyond the band on both sides".
That could never fire: the band contains every round's distance from the
median, so no round lies past it. The 5% rule replaced it. `--out` now keeps
every rep and each round's number, and `--from` judges an earlier run's raw
file.

**A known verdict, reproduced.** The `gemm` arm on Qwen3-8B under vLLM's
defaults (CUDA graphs, prefix caching and chunked prefill on), 72 tokens,
the slope against 32, three rounds; the configuration of chapter 35's
Qwen3-8B run:

| ctx x batch | total | prefill | decode step | noise, across / within (decode) |
|---|---|---|---|---|
| 1024 x 1 | 1.029x win | 1.284x win | 0.999x tie | 0.2% / 0.1% |
| 1024 x 16 | 1.444x win | 1.285x win | 1.677x win | 0.4% / 0.5% |
| 4096 x 1 | 1.054x win | 1.203x win | 1.000x tie | 0.3% / 0.4% |
| 4096 x 16 | 1.205x win | 1.253x win | 1.404x win, band 2.6% | 2.6% / 1.8% |

Chapter 35 measured 1.444x and 1.206x per request at 16 requests, and an
unchanged decode step at one request, where vLLM's skinny GEMMs keep M = 1.
Greedy tokens matched stock's, 72 of 72, at all four. Every band but one sat
at the 2% floor. The decode step at 4096 x 16 had a round 2.6% off the
median, and its band widened to that.

`--from` on chapter 35's own raw file (two rounds, no reps recorded) gives
the same verdicts: 1.446x and 1.206x wins at 16 requests, ties for the
decode step at one.

## `new-op`

`python -m hk.integration new-op residual_add --dir DIR` writes two files.
The op module has [`examples/11`](../../examples/11_vllm_integration.py)'s
shape, made generic. Its parts are a kernel (as written, a working stub,
c = a + b on bf16 rows), its `reference`, `out` and `route`, registered by
`register.op` as `torch.ops.hk_user.residual_add`; `exact()`, inputs whose
answer is known without the reference; `check()`; `scoreboard_cases()`; and
`install()` with an empty `PATCH_TARGET`. The test file runs the IR tier (the
kernel traces and translates; `out` matches the reference's shapes) and the
GPU tier (the exact answer bit for bit; `check` at four shapes, one of them
2047 rows, which takes the bounded loads). The scoreboard takes the op's file
with `--cases` (or `$HK_SCOREBOARD_CASES`) as a group named after the op.

End to end on GPU 3, with nothing edited:
- the generated tests, the repository's `tests/hk/gpu/test_new_op.py` and
  `test_register.py`: 15 passed;
- `python3 residual_add.py`: the exact case bit for bit, max error 6.1e-3
  of the magnitude at the random shapes, 308 calls routed to hk and none to
  the reference;
- the scoreboard (7 rounds of 50 calls, the gate's settings): 0.74x torch
  at 16 rows (2.4 against 1.8 us), 0.77x at 2048 rows (29.0 against
  22.4 us), a tie at 8192 rows (279.2 against 277.3 us). The stub is a
  template and the scoreboard says so. Its Inductor arm now compiles the
  reference once per shape (`dynamic=False`), so Dynamo does not make the
  rows dynamic at the second shape; this run predates that, and torch was
  the best other arm at every shape in it, so its verdicts stand;
- `ab --arms stock,user` with the op registered and nothing patched, a
  control. Its result, 1024 x 16, three rounds, eager: total 0.999x, prefill
  0.998x, decode step 0.993x, each a `tie` inside a 2% band, and greedy
  tokens 48 of 48. The noise it measured is the floor `ab` can resolve
  there: 0.2% across rounds and 0.3% within one for the total, 0.9% and
  1.1% for the decode step.

**An add of two bf16 numbers is often an exact tie.** The stub's random
shapes differ from torch by up to one bf16 ulp. HipKittens rounds fp32 to
bf16 to nearest with ties away from zero (`HK_BF16_ROUND=1`, base_types.cuh),
on the grounds that "exact ties are a measure-zero event on real data". For
a sum of two bf16 values they are not. The fp32 sum has at most a few bits
below bf16's precision, and torch breaks ties to even. Measured on GPU 3, a
and b standard normal at 16 x 4096 and 2048 x 4096: 21.5% and 21.3% of the
fp32 sums are exact ties, the kernel differs from torch at 10.7% and 10.6% of
the elements, every difference is on a tie, and the kernel equals
ties-away rounding of the fp32 sum at every element. So the rounding is
working as written; the comment's premise holds for computed values like
attention's probabilities, and not for a sum of bf16 inputs, such as a
residual add. This is why `exact()` uses small integers, whose sums need no
rounding: there an exact answer exists whichever way ties break.

## Tests

In the IR tier, with no GPU and no torch: `test_headroom.py` (36: every
traced kernel name in its family, vLLM's, SGLang's, Inductor's and hk's; hk's
and FlyDSL's kernels told from the engine's; steps counted by the attention
kernel; the rank and its order of evidence, the record keyed by kernel, the
roofline as the ceiling where there is no ratio; a blind graph trace; the
CLI on a trace from elsewhere), `test_ab_verdicts.py` (11: the four
verdicts from known spreads, pairing when a round failed, the fixed keys,
`--from`) and `test_new_op.py` (5: the files, their trace, their no-GPU
tests, the scoreboard's discovery, refused names and existing files). In
the engines tier, `test_vllm_contract.py` checks the vLLM internals
`headroom` drives (21 passed on vLLM 0.31; its first form expected
`schedule()` without arguments, and 0.31 passes one). On GPU 3 and on an
MI355X, `tests/hk/gpu/test_new_op.py` and the generated tests passed.

## What is left

1. **The record's ratios are few.** Most families rank by their bound or by
   a roofline, which no kernel reaches. Each `--arms` run measures a family
   where an arm has hk's kernel for it; the record should grow from those
   runs rather than from chapters.
2. **The rooflines cover linear layers and attention.** Norms, activations
   and the KV write need a footprint model, with the Infinity Cache in it,
   to be more than a guess. They are small at these shapes, and they rank
   by their bound.
3. **SGLang's path ran on one configuration, stock.** Its arms (`linear`,
   `moe`) set the plugin's switches and were not run here.
4. **`headroom` runs one round per arm.** A family ratio between arms is
   cross-process; it is a step-level number, read past the 2-5% noise.
5. **The stub's template loses to torch below 8192 rows.** Its 16-row
   workgroups are the shape examples/11 used, not a tuned elementwise
   kernel.
6. **On gfx950 `headroom` has families and nothing else.** No peaks (so no
   rooflines) and no record entries, and the patterns have not seen aiter's
   or CK's kernel names: whatever they miss lands in `other`, by name. The
   MI355X needs its own peaks measured (as `--probe-ceilings` measured
   gfx1100's cache rate) and its own `--arms` runs.
7. **A kernel that adds bf16 inputs in fp32 tiles disagrees with torch on
   about a tenth of its elements**, by one ulp on exact ties
   (`HK_BF16_ROUND=1`, ties away). Ties to even (`HK_BF16_ROUND=2`) would
   agree. base_types.cuh puts its cost at 4-6% on attention against 2-3%,
   which is why it is not the default; a per-kernel choice is the
   alternative. Nothing here changes it.

## Reproduction

```
# in the engines' pod, GPU 3, PYTHONPATH=python
python3 -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096 \
    --arms stock,hk+gemm --tokens 24 --json hr_stock.json
python3 -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096 \
    --arms TRITON_ATTN,hk --tokens 24 --json hr_triton.json
python3 -m hk.integration headroom --model Qwen/Qwen3-8B --batch 1 --context 1024 [--graphs]
python3 -m hk.integration headroom --from hr_stock.json      # ranked again, no GPU
/workspace/venv-sgl/bin/python -m hk.integration headroom --engine sglang --model Qwen/Qwen3-8B \
    --batch 4 --context 1024 --json hr_sgl.json

python3 -m hk.integration new-op residual_add --dir newop
python3 -m pytest newop/test_residual_add.py tests/hk/gpu/test_new_op.py
python3 newop/residual_add.py
python3 tools/hk-bench/scoreboard.py --cases newop/residual_add.py --rounds 7 --calls 50
HK_VLLM_USER=newop/residual_add.py python3 -m hk.integration ab --arms stock,user \
    --context 1024 --batch 16 --tokens 48 --short 16 --reps 3 --json newop_ab.json

HK_EAGER=0 HK_VLLM_DEFAULTS=1 python3 -m hk.integration ab --model Qwen/Qwen3-8B \
    --arms stock,gemm --context 1024,4096 --batch 1,16 --tokens 72 --short 32 --reps 3 \
    --out ab_gemm_raw.json --json ab_gemm.json
python3 -m hk.integration ab --from ab_gemm_raw.json --json ab_gemm.json
```
