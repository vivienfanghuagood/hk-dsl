# Measuring the measurements: impossible rounds, margins inside the noise, a shadowed package, and launches that came too soon

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 31](31-remeasured.md) re-measured the gate's baseline and the
linear routes, and in doing so found five weaknesses in the tools the
project's claims rest on:
- the scoreboard kept each arm's fastest round, including one faster than
  the card can go;
- `route_tune.py` decided by margins (3% and 2%) inside its own
  run-to-run noise;
- FlyDSL's GEMM could not be imported beside SGLang, whose venv has
  HuggingFace's `kernels` package;
- every gate run on 0.4.0 said the baseline wanted a refresh, though no
  kernel had changed;
- hk's one- and 32-row norms took 2.4-3.9 us depending on the process, so
  four cases had no gate.

This chapter fixes them. Everything was measured in the engines' pod, in the
vLLM 0.31 venv (torch 2.13.0+rocm7.2, Triton 3.8.0, FlyDSL 0.4.0) and
SGLang 0.5.18's venv beside it, on GPU 3, with hk at `28ffe21` plus this
chapter's changes, and the last four baseline runs and the final gate on
`47b40aa` (chapter 33) plus them. `hk._mlir` was built there against the
MLIR copy chapter 31 left, in a build directory of its own.

## The rounds the scoreboard believed

The scoreboard times each arm in 7 rounds of 50 calls, each round a
profiler session whose kernel records it sums. Chapter 31's seventh run had
hk's 8192x4096 RMSNorm at 97.4 us in one round, 1.38 TB/s for the 134 MB
the call moves through 864 GB/s of GDDR6; its other rounds, like every round
of the six other runs, took 173. Keeping the fastest round kept it.

`tools/hk-bench/_rounds.py` now decides which rounds count, by three rules:
- **records**: every round of an arm launches the same kernels, so a round
  whose record count differs from the arm's usual one lost records (or
  gained some), and is rejected.
- **ceiling**: no round may beat the card. GEMM and attention FLOPs are held
  to 122.6 TFLOP/s. Bytes are held to a rate that depends on the call's
  footprint, the distinct bytes it reads and writes: inside the 96 MB
  Infinity Cache, back-to-back calls find their data resident and run far
  past GDDR6, so a single HBM ceiling would reject every 2048-row norm.
  Each ceiling has 15% of headroom.
- **lone**: the fastest remaining round, if it is more than the larger of
  5% and 4x the other rounds' spread ahead of the second.

The cache's ceiling is measured, by a copy and a two-input add timed as the
arms are (`scoreboard.py --probe-ceilings`), TB/s:

| footprint MB | 8 | 16 | 32 | 48 | 64 | 80 | 96 | 128 | 192 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy | 1.30 | 1.52 | 1.68 | 1.73 | 1.76 | 1.77 | 1.12 | 0.75 | 0.74 | 0.75 | 0.74 |
| add | 1.31 | 1.79 | 2.00 | 2.25 | 2.28 | 2.35 | 2.43 | 0.72 | 0.73 | 0.73 | 0.73 |

So the ceiling is 2.43 TB/s up to 96 MB (2.79 with the headroom) and 864
GB/s past it (0.99). Of every arm in every report chapter 31 kept, the
fastest were 1.82 TB/s inside the cache (hk's 2048-row RMSNorm), 0.78 past
it (the 8192-row norms) and 92.8 TFLOP/s.

**On chapter 31's reports.** Those keep only each arm's best round and the
second's distance from it. `check_regression.py --audit` ran the rules
over all 17 (the six baseline runs, the run left out, four gate runs, six
norm A/B runs):
- the ceiling rejects exactly one round: the 97.4 us one, against a ceiling
  of 135 us;
- `lone` needs the other rounds, so there it can only name candidates: 11
  best rounds sat 22-42% ahead of their second, all at 1 and 32 rows.

A refresh from all seven runs now leaves out that run's one pair (named in
the baseline's `left_out`) and keeps its other 39. It gated the same 40
pairs as the six chapter 31 chose by hand.

**What the arm's time is.** The best round is the statistic that one round
can set. On four runs of the gate's arguments, both kept every round, so
the two statistics could be compared on the same data:

| pairs | ratio's spread across runs, best round | median of kept rounds |
|---|---|---|
| 41 at 2048-8192 rows, GEMM, attention: median / 90th pct / max | 0.45% / 3.5% / 4.0% | 0.36% / 3.7% / 4.7% |
| 16 at 1 and 32 rows | 10.7% / 28% / 34% | 10.7% / 19% / 25% |

Alike where the gate gates, better at the small rows, and not settable by
one round: an arm's time is now the median of its kept rounds, its noise
their median absolute deviation, with the best kept round beside it. In
these four clean runs the rules rejected 2 of 2,072 rounds, both `lone`,
both at one or 32 rows.

## Launches that came too soon

With every round recorded, the small rows' noise had a shape. Within one
process, hk's one-row RMSNorm rounds ranged 2.73-4.65 us, and so did
torch's and FlyDSL's; Inductor's and Triton's stayed within 1-2%. So the
"per-process" times of chapter 31 were per round, and a minimum over 7
rounds sampled them.

`tools/hk-bench/small_rows.py` keeps every kernel record. Per call, the
times are two populations: about 2.4 us, and about 12-14.5 us. The share of
calls 5 us or more above the median, over the one-row RMSNorm and fused add,
21 rounds of 50 calls a process:

| condition | processes | hk | torch | FlyDSL | Inductor | Triton |
|---|---|---|---|---|---|---|
| launched as the scoreboard did | 7 | 4.6-18% | 10-29% | 9.6-19% | 0-0.1% | 0-0.4% |
| kernel arguments in host memory (`HIP_FORCE_DEV_KERNARG=0`) | 3 | 13-18% | 13-25% | 11-14% | 0-0.2% | 0-0.6% |
| hk through `HK_LAUNCH=codeobj` | 3 | 0.1-0.4% | 12-26% | 14-20% | 0-0.1% | 0.1-0.5% |
| a 5 us host pause after every call | 1 | 2.3-5.6% | 12-15% | 16-17% | 0.1% | 0.1-0.2% |
| a 10 us pause | 1 | 0.1-0.3% | 0-17% | 0.5-1.2% | 0.1-0.2% | 0-0.1% |
| a 20 or 50 us pause | 2 | 0-0.1% | 0-18% | 0.1-0.4% | 0-0.1% | 0% |
| queued behind a GPU spin | 2 | 0% | 0-0.1% | 0% | 0% | 0% |

torch's fused add is three kernels, and a pause after the third leaves the
other two launched back to back; its RMSNorm had none from 10 us on.
Ruled out:
- **the kernel**: hk's code object behaves like its pybind module once its
  calls arrive the same way (below);
- **where the kernel arguments are**: the pod sets `HIP_FORCE_DEV_KERNARG=1`
  for every process, so chapter 31's runs had them in VRAM too, and host
  memory changes nothing;
- **the launch API and `hipFuncSetAttribute`**: `launch_path.py` builds the
  same one-row kernel four ways and launches each as fast as it goes. The
  pybind module as generated, which sets the maximum dynamic LDS before
  every launch, had 10.5-15% slow calls in three processes; with the
  attribute set once, 12.7-16.6%; without it, 13-15%; the device-only code
  object through `hipModuleLaunchKernel`, 9.2-14.8%. Triton beside them
  had 0-0.4%.

What is left is when a call reaches the GPU. Each launcher's host cost,
measured behind a spin: hk 6.4-7.7 us, FlyDSL 5.2-6.7, torch 7-14,
Triton 21-22, Inductor 22-28. `HK_LAUNCH=codeobj`'s Python launch is slower
than the pybind entry, which is why it looked like a fix. A launch that
reaches the GPU 5-16 us after it went idle is slow 10-20% of the time; one
that reaches it after 20 us or more, or that is queued behind another
kernel, is not. In the profiler's timestamps the GPU had been idle 5-16 us
before slow and fast calls alike, and 28-30 us before Triton's. Something
in that window costs about 10 us, which the call's own record counts. The
mechanism is not identified; a power-state transition after the queue
empties fits the timing.

**The fix is in the measurement.** The scoreboard now puts a GPU spin in
front of each round, 1 ms a call (`gpu_rounds(queue=True)`), so the host
has launched every call before the first runs, and they run back to back,
as in a captured graph, whatever the launcher costs. (A spin of 40 us a
call was not enough: inside a profiler session a launch costs the host
more, and hk's one-row rounds still ranged 2.19-3.34 us behind it.) Queued,
one run, GPU us per call, median of the kept rounds:

| case | hk | Inductor | Triton | FlyDSL | torch |
|---|---|---|---|---|---|
| rms_norm 1x4096 | 1.95 | 2.04 | 2.06 | 2.17 | 3.62 |
| rms_norm 32x4096 | 2.17 | 2.27 | 2.27 | 2.37 | 3.77 |
| fused add 1x4096 | 2.12 | 2.14 | 2.20 | 2.64 | 7.16 |
| fused add 32x4096 | 2.33 | 2.56 | 2.48 | 2.84 | 7.57 |

Every arm's rounds there sit within 0.1-1.1% (median absolute deviation),
and hk is the fastest at all four: over three queued runs, 1.01-1.06x the
next by the median ratio. Across those three runs against the four
unqueued ones:

| pairs | ratio's spread across runs, unqueued | queued |
|---|---|---|
| 16 at 1 and 32 rows: median / 90th pct / max | 10.7% / 19% / 25% | 0.51% / 0.99% / 2.0% |
| 41 others | 0.36% / 3.7% / 4.7% | 0.21% / 0.77% / 1.1% |

Queuing moved the 41 larger pairs' ratios by a median 0.67% (at most 4.8%,
Inductor against the 8192-row softmax), and the small-row ratios by 41-76%:
those had mostly measured the launchers. hk's small-row kernels were never
the cause, so nothing in hk changed, and the four cases have a gate again
(below). In an eager engine the slow window is real time; under CUDA
graphs, or behind a longer kernel, it is not there.

## The refresh note

The baseline recorded hk 0.3.1, and `check_regression.py` compared every
key of its `meta`, so on 0.4.0 every gate run said the baseline was due a
refresh. 0.4.0 changed the dialect's version and no kernel.

A run and a baseline now record what can move a ratio besides a slower hk:
torch's, Triton's and FlyDSL's versions, the device, `kernels` (one hash
over the generated C++ and hipcc flags of every hk kernel the run built or
took from the cache, `hk.runtime.compile.built_fingerprint`; include paths
are left out, so two checkouts agree) and `hipcc` (the compiler's version).
The note is keyed on those, compared where both sides have them. hk's
version, the dialect's and `toolchain` (hipcc's version with every header's
contents) are kept beside them and not compared. The first draft compared
`toolchain` too, and the rebase showed why not: chapter 33 changed a CDNA
header, which moved `toolchain`, while every RDNA kernel's C++ stayed
byte-identical (the same `kernels` hash before and after). A HipKittens
header change that leaves the generated C++ alone is the one change the
note cannot see. Checked against the old baseline, 0.4.0's gate run no
longer prints it. `tests/hk/ir/test_check_regression.py` holds it: a new
version with the same kernels, or a header for another arch, prints
nothing; a change of the kernels, hipcc, torch, Triton or FlyDSL prints the
note; a key one side lacks is not compared.

## route_tune's margins

`route_tune.py` gave a kernel an M when it beat the engine by 3%, and
FlyDSL a row when it beat the runs' answer by 2%, each on the minimum of 3
rounds. Two runs of one tree decided 15-16 of 740 points differently.

To see why, two runs an hour apart timed every point in four sets of 3
rounds, each set on activations of its own (`--sets 4 --raw`). Per path:
- the rounds within a set spread a median 0.6% (90th percentile 3.1%), and
  the four sets about their median 1.0% (2.9%);
- **the first set was biased.** It runs straight after the point's
  correctness check, where the old tool timed. In it FlyDSL ran a median
  1.5% slower than in the three sets after (90th percentile 6.8%), hk's GEMM
  0.2% (4.3%), the engine 0.1% faster (2.8% slower). Deciding by one set
  against another, set 1 disagreed with each later set at 23-26 points and
  the later sets with each other at 5-9.

The rules, applied to those two runs, against a reference that pools every
set but the first of both:

| rule | points decided differently, run 1 vs 2 | vs the reference |
|---|---|---|
| one set, bare gains (the old tool) | 15 (2 in the runs, 13 in the overlay) | 22, 15 |
| a warm-up set thrown away, then one set | 6 (0, 6) | 3, 7 |
| the same, close calls re-measured with 2 sets more | 6 (0, 6) | 4, 2 |
| the same, margins also above 1x the noise | 10 (1, 9) | 11, 11 |
| the same, margins also above 2x the noise | 6 (2, 4) | 19, 19 |
| three sets every point, bare gains | 6 (0, 6) | 4, 2 |

So the lever is the warm-up. Re-measuring close calls (a decision within
2x its pair's noise of its bar: 16% of the points) brings a run closer to
the pooled answer without changing how often two runs disagree. Requiring
margins above the noise is a **negative result**: it disagrees as often,
and at 2x it drops FlyDSL rows that chapter 31 measured winning end to end,
gate_up at M = 8192 (prefill 1.5% shorter) and down at 128 (the decode
rows). The tool now throws away a warm-up set at every point, times one,
re-measures close calls with two more (`--warmup 1 --sets 1 --remeasure 2
--close-k 2`), decides by the bare gains (`--noise-k 0`), and writes each
path's noise and the number of sets beside its time. `--compare A B`
prints the points two tables decide differently.

**Two real runs of it**, an hour apart (16:05 and 17:01; 196 and 201 s,
against the old tool's 118): 12 points differently, none in the runs, all
in FlyDSL's overlay, at Ms where it and hk's GEMM are within 2% (down at
1024-2816, gate_up at 16384, and four smaller). The old tool's pairs: 13,
16 and 17, with 2-4 in the runs. So the runs no longer flip; the overlay
still does, by drift between sessions that no rounds inside one run see.
Keeping only the overlay's Ms that two runs both took (`--confirm`
an earlier table) leaves two such tables differing at 3-4 points, with
35-39 FlyDSL points rather than 40-43.

**The vLLM table was not re-shipped.** Against it, the new tool's two runs
differ at 22 and 20 points, the old tool's run of the same session at 19,
and none of the differences is at an M chapter 31's end-to-end runs reach
(prefill steps of 1024, 4096 and 8192 tokens, decode at 96 and 128). That
is the old tool's own disagreement, not a change of decisions, so there was
nothing for an end-to-end A/B to measure.

## FlyDSL beside HuggingFace's kernels

FlyDSL's kernels live in its checkout's `kernels` package, and import each
other as `kernels.gemm.*` and `kernels.common.*`. SGLang's venv has
HuggingFace's `kernels` 0.14.1, which SGLang imports for its FA3 path. So
`flydsl_gemm` found the wrong package there, and chapter 31's SGLang table
had no FlyDSL rows; the venv did not have the `flydsl` wheel either, which
went in with `--no-deps` (it declares none), the only line its `pip
freeze` gained.

`flydsl_gemm._import_beside` imports FlyDSL's GEMM from `FLYDSL_ROOT` with
the checkout first on `sys.path` and any other `kernels` modules out of
`sys.modules`, then puts both back. `kernels` itself names HuggingFace's
package again, imported or not yet; FlyDSL's submodules keep their names,
which FlyDSL's compile cache resolves its functions by. HuggingFace's
package has no `gemm` or `common`; a submodule both had would stay
HuggingFace's, FlyDSL's held privately. Without another `kernels`, as in
vLLM's venv, the loader does what it did. `tests/hk/ir/test_flydsl_gemm.py`
holds it on stub packages (not yet imported, already imported, a shared
name, no `FLYDSL_ROOT`). On GPU 3:
- vLLM's venv: `tests/hk/gpu/test_flydsl_gemm.py` (CUDA-graph capture and
  replay included), `test_vllm_gemm.py` and `test_engine.py`, 70 passed;
- SGLang's: `tests/hk/gpu/test_flydsl_gemm.py`, 17 passed, and in the same
  process HuggingFace's `from kernels import get_kernel, load_kernel` still
  imports.

**SGLang's table with FlyDSL.** `route_tune.py --engine sglang` with
FlyDSL, the new way, took 182 s and re-measured 160 points. Its runs equal
the shipped table's but for three teeth (qkv's M = 272 and down's 272-288
to hk's GEMM). FlyDSL's overlay takes 38 points. At chapter 24's
concurrency points, where a decode step's M is the number of requests and
a prefill batch 512 or 4096 tokens here, it changes four routes: qkv at
M = 32 to FlyDSL's 32x32 tile (decode at 32 requests), gate_up at 512 and
at 8192 and up, down at 2048. Everything else routes as before.

**End to end**, chapter 24's setup: `launch_server` (eager, Triton
attention), `bench_serving` with random ids, 512 tokens in and 128 out.
Each server was warmed with 32 requests at concurrency 32 and 2 at 1. Arms
in order: stock, hk with the shipped table, hk with the new one twice, the
shipped one again. All requests completed; every arm's server logged the
same prefill batches.

| concurrency | TPOT ms, stock | hk, shipped table | hk, with FlyDSL | output tok/s, stock | shipped | with FlyDSL |
|---|---|---|---|---|---|---|
| 1 | 43.29 | 23.55, 23.57 | 23.56, 23.54 | 22.88 | 41.81, 41.78 | 41.81, 41.83 |
| 8 | 45.79 | 26.10, 26.15 | 26.14, 26.11 | 172.46 | 300.89, 300.38 | 300.40, 300.66 |
| 32 | 54.34 | 34.37, 34.46 | 34.23, 34.25 | 458.10 | 685.70, 684.29 | 687.72, 687.28 |

At 32 requests the decode step is 0.4-0.6% shorter in both rounds, the
ranges apart; at 1 and 8, nothing to measure. Greedy, three prompts of 48
tokens: the four hk servers gave identical text, and two prompts matched
stock; the third took the other side of chapter 24's near-tie ("France is
Paris" against "Italy is Rome"), with and without FlyDSL. **The table is
shipped**, as the tool wrote it, with its FlyDSL rows.

## The refreshed baseline

The statistic and the queue change every ratio, so the baseline was
measured again: ten runs, six on `28ffe21` with this work (16:55-17:00 and
17:28-17:33) and four after rebasing onto `47b40aa`, which brought chapter 33
(20:35-20:45). All ten built the same kernels, one `kernels` hash, and the
scoreboard rejected no round in any of them.

| | chapter 31's baseline | this one |
|---|---|---|
| runs | 6 (one left out by hand) | 10 |
| arm times | best round, launched onto an idle GPU | median of kept rounds, queued |
| gated pairs, over cases | 40 over 13 | 57 over 17 |
| at the 3% floor | 28 | 37 |
| widest tolerance | 13.7%, Inductor against the 8192-row fused add | 12.8%, torch against the one-row RMSNorm |
| not gated | 17 | 0 |
| cases with no gate | RMSNorm and the fused add at 1 and 32 rows | none |

The GEMM's tolerances are 3.0-4.4%, attention's 3.3-7.3%. The 40 pairs both
baselines gate moved by a median 0.68%, at most 5.1% (Inductor against the
8192-row softmax), and their widest tolerance went from 13.7% to 7.3%.

**Does it fire?** The norms on the tile kernels (`HK_NORM_LAYOUT=tile`),
which chapter 27 measured slower than the strips everywhere but the
8192-row fused add, in one queued run: exit 1 on 35 pairs. The one- and
32-row norms fell 15-26%, the 2048-row ones 10-27%, the 8192-row RMSNorm
9-12%, three arms of the 8192-row softmax 5.8-7.0%. The GEMM, attention and
the 8192-row fused add moved -1.5% to +0.8% and passed. The check also said
hk's kernels had changed (`kernels` f3148ed2 against e5ad0e20).

**And on the tree it ships with**: `scripts/perf_gate.sh` on `47b40aa`,
after the ten runs, exit 0, all 57 pairs within tolerance, no note; the
largest change used 31% of its tolerance (Inductor against the 8192-row
softmax, -2.2% of 7.2%).

## So

- **The scoreboard no longer believes an impossible round**: a ceiling by
  footprint (GDDR6 past the 96 MB cache, 2.43 TB/s measured inside it),
  a record count, and a lone-round rule; on chapter 31's 17 reports the
  ceiling rejects exactly the 97.4 us round. Arm times are the median of
  the kept rounds.
- **The one- and 32-row "per-process" times were the launchers'**: a launch
  arriving 5-16 us after the GPU went idle took about 10 us longer, and
  the arms that launch in 5-8 us got 10-20% of such calls. Queued behind a
  spin, every arm is steady to about 1%, hk is the fastest at all four
  cases, and they are gated again.
- **route_tune's runs no longer flip between runs; its FlyDSL overlay still
  does** (12 points of 740 an hour apart, from 13-17), by drift between
  sessions. The lever was a biased first set, not the margins: requiring
  margins above the noise is a negative result. The vLLM table stands.
- **FlyDSL runs beside SGLang** without taking `kernels` from HuggingFace,
  and its rows shorten SGLang's decode step at 32 requests by 0.4-0.6%;
  SGLang's table now carries them.
- **The refresh note is keyed on the kernels and the opponents**, not hk's
  version.

## What is left

1. **FlyDSL's overlay drifts between sessions** at prefill Ms where it and
   hk's GEMM are within 2%. `--confirm` against an earlier run's table is
   the measured remedy (3-4 points between two such tables); the shipped
   tables are single runs.
2. **The slow-launch window is not identified.** It is outside hk and
   affects any fast launcher on this card; in an eager engine it costs real
   time. A probe would be a HIP microbenchmark sweeping the gap between
   dispatches, with and without the profiler.
3. **The table rows below 2% that paid end to end** (chapter 31's M = 1024)
   are a benchmark-versus-model difference, not noise; nothing here
   resolves it.
4. Chapter 30's and 31's other items stand: FlyDSL at Ms its tiles do not
   divide; FlyDSL's losses at 272-1024 rows.

## Reproduction

```
# in the engines' pod, GPU 3; hk._mlir built against chapter 31's MLIR copy
HK_LLVM_ROOT=/workspace/measure/llvm HK_MLIR_BUILD=$PWD/../hk-build \
    PYTHON=/workspace/measure/buildenv/bin/python scripts/build_hk_mlir.sh
export HIP_VISIBLE_DEVICES=3 FLYDSL_ROOT=/workspace/FlyDSL PYTHONPATH=python

python3 tools/hk-bench/scoreboard.py --probe-ceilings --rounds 7 --calls 50
python3 tools/hk-bench/check_regression.py --audit sb_*.json      # chapter 31's reports
python3 tools/hk-bench/small_rows.py --arms hk,torch,inductor,triton,flydsl [--gap-us G] [--queue-ms 2]
HK_LAUNCH=codeobj python3 tools/hk-bench/small_rows.py ...
python3 tools/hk-bench/launch_path.py
python3 tools/hk-bench/scoreboard.py $(python3 tools/hk-bench/check_regression.py --print-args) \
    --json sb_N.json                                                # six times
python3 tools/hk-bench/check_regression.py --refresh sb_*.json

python3 tools/hk-bench/route_tune.py --model Qwen/Qwen3-8B --warmup 0 --sets 4 --remeasure 0 \
    --raw --out full_N.json                                         # twice, an hour apart
python3 tools/hk-bench/route_tune.py --model Qwen/Qwen3-8B --out new_N.json   # twice
python3 tools/hk-bench/route_tune.py --compare new_1.json new_2.json

/workspace/venv-sgl/bin/pip install --no-deps flydsl==0.4.0
/workspace/venv-sgl/bin/python tools/hk-bench/route_tune.py --engine sglang \
    --model Qwen/Qwen3-8B --out routes_sglang.json
HK_SGLANG_LINEAR=1 [HK_SGLANG_ROUTES=routes_sglang.json] python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --attention-backend triton --disable-cuda-graph \
    --dtype bfloat16 --mem-fraction-static 0.8
python -m sglang.bench_serving --backend sglang --dataset-name random-ids \
    --random-input-len 512 --random-output-len 128 --random-range-ratio 1.0 \
    --num-prompts 64 --max-concurrency 32                           # and 1, 8
```
