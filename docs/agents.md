# For agents: where to hit next, and whether it helped

**Writing or tuning one kernel** rather than a model's step: start from
[llms.txt](../llms.txt) (the map, the ten-line kernel, the pitfalls) and the
[hk-kernel skill](../skills/hk-kernel/SKILL.md) (the loop), and check every
edit with `python -m hk.check FILE.py:NAME --json out.json`
([hk.check](check.md)): resources without a GPU, correctness against a torch
reference, and timing against torch eager and `torch.compile` interleaved in
one process, as this page's tools time an engine. The API reference,
generated from the code, is [docs/api](api/README.md). What follows is the
loop one level up: a kernel inside vLLM or SGLang.

Three commands answer the two questions without a harness of your own:

```bash
python -m hk.integration headroom --model M --batch B --context C --json h.json   # where the step's time is, and what hk has for it
python -m hk.integration new-op NAME --dir DIR                                     # a kernel on the rails: reference, route, exact test, bench row
python -m hk.integration ab --model M --arms stock,ARM --json v.json               # did the request get faster: win | loss | tie | noise
```

and three more hold a result once you have one: the scoreboard
(`tools/hk-bench/scoreboard.py --cases DIR/NAME.py`, per call, every
contender in one process), the performance gate (`scripts/perf_gate.sh`, so
it cannot quietly regress) and an exact-answer test (so it cannot quietly be
wrong). This page says how to use each, and which numbers not to believe.
Everything here was measured on a W7900D (gfx1100) under vLLM 0.31; for
another machine see [the last section](#on-another-machine).

- [Find the gap: `headroom`](#find-the-gap-headroom)
- [Add a kernel: `new-op`](#add-a-kernel-new-op)
- [Prove it helps](#prove-it-helps)
- [Numbers not to trust](#numbers-not-to-trust)
- [On another machine](#on-another-machine)

## Find the gap: `headroom`

```bash
python -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096 \
    --arms stock,hk+gemm --tokens 24 --json h.json
```

It builds the engine the way `ab` does (one child process per arm, vLLM's
engine core in it), and profiles two phases:

- **decode**: once every request is past its prefill, `--windows` windows of
  `--steps` steps at the full batch, timed without the profiler, then one
  window under it;
- **prefill**: the batch generating one token each, which is how `ab` times
  prefill. Without chunked prefill a step holds as many prompts as fit
  `max_num_batched_tokens`: at 16 x 4096 on Qwen3-8B, 8 steps of 8192 tokens.

vLLM's steps are driven one at a time (`LLMEngine.step`) and counted where
its scheduler hands them out, so a window of K steps is K steps even with
async scheduling. Device time comes from the profiler's chrome trace: on
ROCm 7.2's torch, `key_averages()` reports `self_device_time_total = 0` for
every kernel.

Each kernel goes into a family (linear, attention, norm, activation,
rotary, kv_write, sampling, embedding, elementwise, copy, other) by its
name, and is marked hk's (`<name>_kernel(globals)`, or FlyDSL's GEMM, which
only hk's routing table sends calls to) or the engine's. Then, per phase,
one row per family:

| column | what it is |
|---|---|
| step | the family's share of the step's wall time (unprofiled) |
| ms/step | its device time per step |
| bound | the **upper bound**: what the step would save if the family took zero time. At most that, and less where the host holds the step up |
| ratio | hk's kernel against the engine's, from the best evidence there is (below) |
| ceiling | 1 - 1/ratio: the fraction of the bound hk could take (0 where hk's kernel loses); with no ratio, the roofline's |
| score | bound x ceiling, what the rows are ranked by; with neither, the bound itself, flagged `?` |
| hk has | the `hk.ops` for it, the `ab` arm and switch that route it, on or off in this run |
| roofline | the family's shortest possible time at the card's peak, and the share of it the family runs at |

The ratio comes from, in this order:
1. **this run**: the same family in another `--arms` arm whose kernels for
   it are hk's (each arm is its own process, so read it as a step-level
   number, not a kernel-level one);
2. **the routing table**, for linear layers: the tuner's per-call times of
   the engine's op and of the path the table routes, at this step's M, for
   each of the model's (N, K) (measured interleaved in one process,
   `hk.autotune.routes`);
3. **the record**: a ratio a chapter measured, cited, and only where the
   engine's largest kernel in the family is the one it was measured
   against. hk's paged decode is 0.86-1.12x TRITON_ATTN's kernel under
   triton-rocm 3.7.1 at 16-64 requests ([chapter 38](history/38-rdna-losses.md)),
   and that says nothing about stock 0.31's `kernel_paged_attention_2d`.

With no ratio, the ceiling is the **roofline**'s, labelled "at most": for
the linear layers each of the model's calls at max(FLOPs / peak, bytes /
DRAM bandwidth), for decode attention the KV cache streamed once, for
prefill attention 2 h n^2 d FLOPs a layer and causal prompt, all from the
model's config and the step's M. The peaks are per arch (`PEAKS`: gfx1100's
from `tools/hk-bench/_rounds.py`); an arch without them gets none. No
kernel reaches its roofline, so this ceiling is itself an upper bound, but
it is what tells hk's own kernels apart once its arms are on: a GEMV at 89%
of the card's DRAM rate has little left, an attention at 40% of the bf16
peak has more.

`(GPU idle)` is a row too: the step's wall time the device was not busy.
No kernel wins it back; graphs or fewer launches do. `--tokens T` adds the
shares over one request batch of T output tokens, which is what a profile
of a whole `generate` shows (chapter 07's view).

**Qwen3-8B, 16 requests x 4096, eager**, `--arms stock,hk+gemm`
([chapter 36](history/36-the-agent-loop.md)). "of peak" is the family's
roofline over its time:

| phase | family | share of step | ms | of peak | ratio | score |
|---|---|---|---|---|---|---|
| decode, 71.3 ms | linear | 55.8% | 39.8 | 44% | 2.00x, measured by the `hk+gemm` arm | 19.9 |
| | attention | 40.1% | 28.6 | 39% | 1.71x, measured by the `hk+gemm` arm | 11.9 |
| prefill, 2292 ms | linear | 75.9% | 1741 | 53% | 1.39x, measured | 490 |
| | attention | 20.6% | 473 | 17% | 2.36x, measured | 273 |

Ranked from `hk+gemm` instead, the next targets are its own kernels: decode
attention at 67% of its roofline (5.5 ms a step left at most), the GEMV at
89% (2.3 ms); the prefill GEMMs at 74% of the bf16 peak, prefill attention
at 40%. Ranked from TRITON_ATTN (`--arms TRITON_ATTN,hk`), hk's decode
attention measures 0.87x its kernel and ranks last, as chapter 20 found,
while its prefill attention is 53.8% of a prefill step and measures 9.25x.

**CUDA graphs.** `--graphs` profiles the engine as it ships (graphs and
torch.compile). On this ROCm the profiler records the kernels a graph
replays: the decode step is one graph launch, with every kernel of the eager
step inside it. If a run's trace has graph launches but device time for
under half the step, the phase is reported as blind, not ranked. Eager is
the default. At 16 x 4096 the linear and attention families agreed with
graphs' within 0.4%; the small ones differ, because torch.compile fuses the
norms and activations into kernels of its own.

**Other inputs.** `--trace decode=PATH` ranks a chrome trace from elsewhere
(vLLM's or SGLang's own profiler, another machine); `--from h.json` ranks an
earlier result again from the per-kernel totals it kept. `--engine sglang`
profiles SGLang in its scheduler's process, with its radix cache and overlap
scheduler off, one session around the prefill and its profiler's first
decode steps; a step is counted by the attention kernel each layer calls
once, and the linear routes come from SGLang's own table. On Qwen3-8B at 4 x
1024 its decode step was 85% linear layers, at 45% of the DRAM roofline,
and its table put hk's GEMV at 1.90x there.

## Add a kernel: `new-op`

```bash
python -m hk.integration new-op residual_add --dir my_ops
```

writes `my_ops/residual_add.py` and `my_ops/test_residual_add.py`. The op
module is [`examples/11_vllm_integration.py`](../examples/11_vllm_integration.py)'s
shape, made generic:

- `kernel`, an `@hk.kernel` (inputs, then outputs). As generated it is a
  working stub, c = a + b on bf16 rows, so every part below runs before you
  change any of it;
- `reference`, `out` and `route`, registered by `register.op` as
  `torch.ops.hk_user.residual_add` (`registered()`): `out` is the fake
  torch.compile traces, `route` the calls the kernel takes, decided inside
  the op where shapes are numbers;
- `exact()`, inputs whose answer is known without the reference, and
  `check()`, which holds the kernel to it bit for bit and then to the
  reference at every shape in `SHAPES`;
- `scoreboard_cases()`, the scoreboard's rows: the op beside torch and
  Inductor on the same tensors;
- `install()` and `PATCH_TARGET`, the engine patch, run in vLLM's own
  process by its plugin when `HK_VLLM_USER` names the file. Empty patches
  nothing.

The test file runs in the IR tier (the kernel traces and translates, `out`
matches the reference's shapes; no GPU) and on a GPU (the exact answer bit
for bit, `check` at every shape). The repository holds the template to the
same: `tests/hk/ir/test_new_op.py` generates an op and runs its no-GPU tests,
`tests/hk/gpu/test_new_op.py` runs its `check()`.

Change `kernel`, `reference`, `out`, `route` and `exact()` together, and keep
`exact()` exact: attention's is V all ones, where every output element is
exactly 1.0 whatever Q and K are (`tests/hk/gpu/test_attn.py`). A tolerance
is where a wrong kernel hides; a half-ulp bias in a bf16 conversion passed
all 31 of attention's tolerance cases while changing what a model said
([phase 6](history/05-phase-6-torch-ops.md)). Random inputs do not make an
exact case in general: a kernel's fp32 arithmetic need not be torch's, in
order or in rounding. For the stub's add it happens to be: a sum of two bf16
values is an exact tie a fifth of the time, HipKittens' stores rounded those
ties away from zero where torch rounds them to even, and the stub differed
from torch at 10.7% of its elements ([chapter 36](history/36-the-agent-loop.md));
RDNA's stores now round to even and it agrees at every element
([chapter 38](history/38-rdna-losses.md)). The stub's exact case is still
small integers, which need no rounding at all.

## Prove it helps

Three levels, each one necessary, none sufficient alone.

**Per call: the scoreboard.**

```bash
python3 tools/hk-bench/scoreboard.py --cases my_ops/residual_add.py --rounds 7 --calls 50 --json sb.json
```

Every arm in one process, checked against an fp32 reference first, timed
from the kernels' own records in a profiler session per arm, rounds
rotating, each round's calls queued behind a GPU spin. Rounds that lost
records, beat the card at the case's footprint, or stand alone are rejected
(`_rounds.py`); an arm's time is the median of the rest. The row's verdict
is a tie inside max(2%, the two arms' noise). A file's `scoreboard_cases`
joins as a group named by its `NAME`; `$HK_SCOREBOARD_CASES` does the same.

**End to end: `ab`.**

```bash
HK_VLLM_USER=my_ops/residual_add.py python -m hk.integration ab --model Qwen/Qwen3-8B \
    --arms stock,user --context 1024,4096 --batch 1,16 --tokens 72 --short 32 --reps 3 \
    --out raw.json --json verdicts.json
```

Each arm is a fresh engine in its own process, rounds rotate the arms, and
each round reports the best of its reps. `--json` writes one row per shape,
metric (total, prefill, decode step) and arm, every row with the same keys:

```
{shape, metric, arm, ms, stock_ms, ratio, band, verdict, why, rounds, noise, tokens_same}
```

- `ms`, `stock_ms`: the medians over rounds;
- `ratio`: stock / arm per round, rounds paired by number (a pair ran
  back to back), then the median; above 1 the arm is faster;
- `band`: max(2%, noise), the noise being the larger of the ratio's spread
  across rounds and the two arms' summed rep spread within one, as the
  performance gate measures noise;
- `verdict`: `win` / `loss` past the band; `tie` inside a band of at most
  5%; `noise` for one round, a band past 15% (the gate's cap), or a ratio
  inside a band wider than 5%, which cannot tell a tie from the gaps this
  project wins and loses by. Tie and noise are answers, not missing data.
- `tokens_same`: how many leading greedy tokens of the first request match
  stock's.

`--from raw.json` judges an earlier run again. **Qwen3-8B, the `gemm` arm**,
vLLM's defaults, three rounds ([chapter 36](history/36-the-agent-loop.md)):

| ctx x batch | total | prefill | decode step |
|---|---|---|---|
| 1024 x 1 | 1.029x win | 1.284x win | 0.999x tie |
| 1024 x 16 | 1.444x win | 1.285x win | 1.677x win |
| 4096 x 1 | 1.054x win | 1.203x win | 1.000x tie |
| 4096 x 16 | 1.205x win | 1.253x win | 1.404x win (band 2.6%) |

and an arm that registers an op and patches nothing, the control: 0.999x,
0.998x and 0.993x at 1024 x 16, three ties inside 2%. The noise that control
measured, 0.2-0.3% for a request and about 1% for a decode step, is what a
run of `ab` can resolve here.

**Where the time went: `headroom` again**, with your arm beside stock
(`--arms stock,user`). A family whose kernels became hk's gets a measured
ratio; one that did not move says so.

**So it stays: the performance gate.** A scoreboard case is gated once the
baseline measures it: add its group to `check_regression.py`'s `ARGS` and
refresh the baseline from three runs or more (`scripts/perf_gate.sh
--refresh`, [testing](testing.md#the-performance-gate)). Its tolerance is
then its own measured noise, times three, at least 3%.

## Numbers not to trust

Each of these has cost this project a wrong conclusion once.

- **A kernel A/B across processes.** On these GPUs it manufactures 13-16%
  differences out of nothing ([phase 5](history/04-phase-5-autotune.md)):
  kernels are compared in one process, interleaved. `ab` is cross-process
  because an engine is; that is why its rounds alternate, its ratio is paired
  round by round and its band is measured, and why one round is `noise`.
- **A baseline that did not run its own autotuner.** FlyDSL's GEMM autotuner
  had never searched, and Triton's ranked cold calls; retuned, hk's GEMM went
  from 1.03-1.08x to 0.95-0.98x ([chapter 26](history/26-the-scoreboard-retuned.md)).
  Every scoreboard arm says how it was tuned; an opponent that did not try
  is not a comparison.
- **A gap under ~2-5%.** Noise until shown otherwise. Two attention tuning
  sweeps crowned contradictory winners at 0.1% and 2.8%; two runs of the
  route tuner on one tree disagree at 2% of their points
  ([chapter 31](history/31-remeasured.md)). The scoreboard's band floor is
  2%; `ab` calls nothing inside 5% a tie unless the band is that narrow.
- **A microsecond kernel launched onto an idle GPU.** A launch that reaches
  it 5-16 us after it went idle costs about 10 us more, which the call's own
  record counts ([chapter 32](history/32-measuring-the-measurements.md)).
  The scoreboard queues every round behind a spin; an eager `headroom` step
  has the same cost in its small kernels when the batch is small, so look at
  the `(GPU idle)` row and compare with `--graphs`.
- **A per-call win as an end-to-end one.** FlyDSL's M = 1024 rows measured
  0.2% ahead per call, summed, and the 1024-token prefill step was 1.2%
  slower with them ([chapter 31](history/31-remeasured.md)); Triton's decode
  kernel beats hk's at every batch, and routing 16+ requests to it made the
  captured step slower ([chapter 20](history/20-vllm-031.md)). The request is
  what a server pays: `ab`.
- **A share as a saving, or one configuration's share as another's.** A
  family at 40% of a step is a bound of 40%, not a gain; the ceiling is what
  hk's kernel measured against the engine's. And shares move: decode
  attention was 0.9% of a request in chapter 07's profile (vLLM 0.16), and
  is 40% of a decode step at 16 x 4096 on vLLM 0.31's default backend,
  against 25% on TRITON_ATTN's. Profile the engine, backend, batch and
  context you serve, and say whether a share is of a step or a request.
- **An opponent's version nobody checked.** A Triton 3.8.0 wheel was
  installed over the triton-rocm 3.7.1 vLLM 0.31 is built for, and its
  decode kernel is up to 40% slower; hk's walk read as 1.03-1.30x "Triton"
  where it is 0.86-1.12x the Triton chapter 20 measured
  ([chapter 38](history/38-rdna-losses.md)). Print `triton.__version__`
  (and torch's, FlyDSL's) beside a ratio, and re-run the record's own tool
  before calling a record wrong. In vLLM's venv, `python3 -m
  hk.integration.stack` says whether the stack is the measured one, and
  `ab` refuses to run when it is not.
- **A kernel test with only a tolerance.** Every new kernel needs a case
  with an exact answer (V = 1 for attention, small integers for an add);
  `new-op` writes one, and `check()` runs it first.
- **`self_device_time_total`, and the profiler's range-to-kernel
  correlation.** The first is 0 for every kernel on this ROCm; the second is
  unreliable for anything launched outside ATen. Read the chrome trace, and
  time one arm per profiler session.
- **A patch that routed nothing.** It times like stock: the first vLLM V1
  run reported 1.001x with the patch applied in a process the model never
  ran in ([integration](integration.md#on-by-default-and-turning-it-off)). `ab` prints each
  patch's counters; `headroom` says whether a family's kernels are hk's.
- **A round faster than the card.** An 8192x4096 RMSNorm "ran" at 1.38 TB/s
  on 864 GB/s of GDDR6 ([chapter 32](history/32-measuring-the-measurements.md)):
  the scoreboard rejects rounds past the card's ceiling and reports the
  median of the rest.

## On another machine

`headroom`'s families, its hk-or-engine split, the step counting and the
trace parsing do not depend on the arch or the engine's version; its
ceilings do. The record's ratios are keyed by arch (gfx1100's say nothing
about gfx950), the routing table is the arch's own
(`routes_vllm_linear_<arch>.json`), and a family with neither ranks by its
bound. On a new machine: run `headroom` stock, then with each arm hk has
there (`--arms stock,hk,gemm`) so the ratios are this machine's; check the
`other` family for engine kernels no pattern names; and keep the rules
above, which are about the measuring, not the card.
