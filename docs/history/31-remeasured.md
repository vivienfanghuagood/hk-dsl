# Re-measured: the gate's baseline and the linear routes, after strips and the GEMM schedule

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Two artifacts that [chapter 30](30-flydsl-routes-gate-contracts.md) shipped
were measured on kernels that have since changed:
- the performance gate's baseline, on hk as of
  [chapter 26](26-the-scoreboard-retuned.md), before
  [chapter 27](27-strips.md) put the norms and softmax on strips and read
  every RDNA kernel's arguments at entry;
- the routing tables. vLLM's FlyDSL rows were measured against hk's GEMM
  before [chapter 28](28-gemm-schedule.md) made it 2.9-6.5% faster, and
  SGLang's table ([chapter 24](24-sglang.md)) is older still.

This chapter re-measures both where they were measured: the vLLM 0.31 venv
of the engines' pod (torch 2.13.0+rocm7.2, Triton 3.8.0, FlyDSL 0.4.0 with
its v0.4.0 checkout), SGLang's venv beside it (torch 2.9.1), GPU 3, and hk
at `f155cd6`.

## Building there

The pod's venvs carry the 0.3.1 wheel, whose dialect bindings do not build
chapter 27's kernels, and the pod has no MLIR. So the dev pod's MLIR 23.1.3
install for CPython 3.12 (3.6 GB) was copied into the engines' pod.
`scripts/build_hk_mlir.sh` then built `hk._mlir` against it there, in 34 s,
for the venvs' `/usr/bin/python3.12`, with nanobind from a venv of its own.
The clone, the build directory and hk's compile cache were all new, so
nothing another process had mapped was rebuilt. `PYTHONPATH=python` puts the
checkout ahead of the installed 0.3.1.

lit passed 10 of 10. The IR tier passed 637, skipped 3 and failed one,
`test_an_absent_framework_reports_rather_than_raises`, which needs vLLM not
to be importable.

## The gate against the old baseline

One run of `scripts/perf_gate.sh`: exit 0, no regression, and 33 of the 45
gated pairs `better`. The gate compares ratios only; the times beside them
are the old baseline's four runs (median) against this run.

| case | ratio change | hk's time | the other arms' times |
|---|---|---|---|
| GEMM, five shapes, 14 pairs | +2.8% to +5.0% (11 `better`) | -3.3% to -4.8% | within 1.5% |
| RMSNorm, 2048 rows | +33% to +35% | 24.9 to 18.5 us | within 1.5% |
| RMSNorm, 8192 rows | +12% to +14% | 195.0 to 173.4 us | within 1.0% |
| fused add, 2048 rows | +9.6% to +11.2% | 40.4 to 36.2 us | within 1.7% |
| fused add, 8192 rows | +0.3% to +1.5% (`ok`) | 349.3 to 340.1 us | within 1.6% |
| softmax, 2048 rows | +27% | 23.8 to 18.7 us | within 0.2% |
| softmax, 8192 rows | +7% to +8% | 186.2 to 173.4 us | within 1.0% |
| attention, causal / non-causal | -0.3% / -2.8% (`ok`) | unchanged, 2.81 / 4.66 ms | torch's non-causal 3.0% faster |
| 1 and 32 rows, the 5 gated pairs (Inductor, Triton) | -4.5% to +18.9% | 3.2-3.6 to 2.8-3.9 us | 2.5-2.6 to 2.6-2.8 us |

So the gate saw what chapters 27 and 28 claimed: the GEMM 3-5% faster, the
2048-row norms and softmax 10-26% faster, attention unchanged.

Three pairs moved down, all inside their tolerances:
- torch against the non-causal attention, -2.8% of 8.6%. hk's kernel took
  4.66 ms in both; torch's SDPA was 3.0% faster in this, the session's
  first process. The first of the old baseline's runs had the same 0.962
  against 0.990-0.992 in the other three.
- Inductor and Triton against the one-row fused add, -4.5% of 8.8% and
  -4.4% of 7.8%.

### The one-row fused add, before refreshing

A pair inside its tolerance is not a regression by the gate's rule, but a
refresh would make its new ratio the normal one. So chapter 26's tree (the
copy chapter 30 measured with, on its 0.3.1 bindings) and this one ran the
norms alternately, three processes each, with the gate's 7 rounds of 50
calls (`scoreboard.py --only norm --arms hk,inductor,triton`):

| case | hk us, chapter 26's tree | this tree | Inductor / hk, chapter 26's (median) | this |
|---|---|---|---|---|
| fused add 1x4096 | 4.86, 3.46, 3.56 | 4.00, 2.97, 3.50 | 0.708 | 0.711 |
| fused add 32x4096 | 4.35, 3.56, 3.64 | 2.51, 3.28, 3.41 | 0.701 | 0.780 |
| RMSNorm 1x4096 | 4.78, 4.00, 3.26 | 3.73, 3.06, 3.25 | 0.622 | 0.775 |
| RMSNorm 32x4096 | 5.09, 3.23, 3.33 | 3.42, 3.37, 3.69 | 0.769 | 0.746 |
| fused add 2048x4096 | 40.28, 40.26, 40.31 | 36.15, 36.21, 36.21 | 0.898 | 1.002 |
| RMSNorm 2048x4096 | 25.06, 24.99, 25.06 | 18.46, 18.53, 18.52 | 0.829 | 1.121 |

The one-row fused add is the same on both trees, at 3.56 and 3.50 us. Of
the others at one and 32 rows, two are faster on this tree, and the 32-row
RMSNorm is 3% slower by its median, inside the spread of its processes. No
regression, so the refresh went ahead.

What the A/B also shows: hk's one- and 32-row kernels take anything from
2.5 to 5.1 us depending on the process, on both trees, while Inductor and
Triton take 2.4-2.8 in every process. In the seven gate-argument runs of this tree,
hk's one-row fused add took 3.88, 2.37, 3.74, 3.61, 3.12, 3.57 and 2.52 us,
and Inductor's 2.50-2.65. The fast processes match chapter 27's best (2.4 us
for the one-row RMSNorm); the slow ones, chapter 26's 3.1-3.6. The cause is
unknown.

## The refreshed baseline

`check_regression.py --refresh` on six runs of this tree: the gate run
above and three more within 15 minutes, then two after an 18-minute vLLM
run (below).

A seventh run, the first after the gate run, was left out. In it, hk's 8192-row
RMSNorm took 97.4 us in its best round, 1.38 TB/s for the 134 MB the call
moves, which HBM (0.86 TB/s) cannot do. Its second-best round took 173 us,
like every round of the other runs. The scoreboard keeps each arm's fastest
round, so one such round sets the arm's time. Kept, it would have put the
case's noise at 78% and left all four of its pairs ungated. The run is
named in the refresh's commit.

| | chapter 30's baseline | this one |
|---|---|---|
| hk | as of chapter 26 | `f155cd6`, chapters 27 and 28 |
| runs | 4 | 6 |
| gated pairs, over cases | 45 over 16 | 40 over 13 |
| at the 3% floor | 32 | 28 |
| widest tolerance | 12.7%, Inductor against the 32-row RMSNorm | 13.7%, Inductor against the 8192-row fused add |
| not gated | 12 | 17 |
| cases with no gate at all | the one-row RMSNorm | RMSNorm and the fused add at 1 and 32 rows |

- The GEMM ratios moved 0.1-1.6% across the six runs, and every GEMM
  tolerance is 3.0-4.8%. The -11.6% to -15.2% that chapter 30 caused on
  purpose is past all of them; that test was not repeated.
- The 2048- and 8192-row norms and softmax moved up to 3.6%. The 8192-row
  fused add moved 4.6%, because hk ran it 4.5% faster in one run (325 us
  against 340-341), which widened its tolerances to 12.8-13.7% and put
  Triton's pair, at 5.1% noise, past the cap.
- torch's non-causal attention is at 8.6%: the gate run above is in the
  six, with its fast SDPA.
- The 16 small-row pairs are ungated by noise of 18-52%. Chapter 30's
  baseline gated five of them, at tolerances of 3.0-12.7%: its four runs
  happened to catch hk at a steady 3.1-3.7 us.

Checked against it, the left-out run passes, with the four 8192-row
RMSNorm pairs reported as `better` by 77-80%. A fresh gate run four minutes
after the six used at most 34% of a pair's tolerance (Triton against the
2048-row RMSNorm, -3.7% of 10.8%). Two more passed later:
- after the graph-mode vLLM run below, 70 minutes on: at most 50% (torch
  against the 2048x4096x12288 GEMM, -1.5% of 3.0%);
- on the release, 0.4.0, whose kernels are this tree's (its dialect change
  is the summary and the compile cache's key): at most 51% (FlyDSL against
  the same GEMM). torch's non-causal attention, in that session's first
  process, was again 2.9% faster, inside the 8.6% the first run here had
  set. On 0.4.0 the check also prints that the baseline was made on 0.3.1.

## The vLLM table

`route_tune.py --model Qwen/Qwen3-8B` with FlyDSL on, in the vLLM venv,
took 98 s.

**The runs, which every install routes by, are the shipped table's but for
three teeth of the comb:** qkv's M = 272 and down's 272-288 go to hk's GEMM
instead of vLLM's, and o's M = 80 too. The decode GEMV still owns 4-64 (qkv)
and 6-64 (the rest), and vLLM's skinny GEMMs still keep the Ms below.

**FlyDSL's overlay shrank from 105 routed points to 41**, of the 304 where
one of its tiles divides M. By region, the routed points, and FlyDSL's
median time against the runs' answer over the five shapes:

| M | routed, chapter 30 | now | FlyDSL / runs' answer, chapter 30 | now |
|---|---|---|---|---|
| 64-256 | 25 | 23 | 0.84-0.99x | 0.84-1.02x |
| 272-1024 | 16 | 8 | 1.14-1.33x | 1.17-1.42x |
| above 1024 | 64 | 10 | 0.975-0.99x | 1.00-1.02x |

The decode-sized rows stay. At 64-256 rows FlyDSL's 32- and 64-row tiles
take a median 0.75-0.90x the time of hk's 128-row GEMM on the four layer
shapes (0.98x on the LM head). At 96 and 128 rows, the decode steps of
chapter 30's 96 and 128 requests, every shape routes exactly as before
(`route_for`, FlyDSL's tile check included): FlyDSL on qkv, o and gate_up,
on down at 128 (vLLM's op at 96), and hk's GEMM on the LM head.

Above 1024 rows FlyDSL's lead is gone. It clears the 2% bar at 10 scattered
Ms, against 64 before. The absolute times of the two measurements are from
different sessions, so the comparison that holds is against vLLM's op, which
both tables timed at every point. Each path's time over the op's, this
measurement against chapter 30's, by point:

| path | median | 10th-90th percentile |
|---|---|---|
| hk's GEMM, 80-256 rows | -2.6% | -5.3% to +2.1% |
| hk's GEMM, 272-1024 rows | -3.2% | -4.8% to +0.4% |
| hk's GEMM, above 1024 | -3.1% | -4.7% to +0.1% |
| FlyDSL, by region | -0.7% to +0.7% | -3.0% to +4.7% |
| hk's decode GEMV, 1-64 rows | +0.2% | -5.4% to +7.4% |

hk's GEMM gained 3% on the op, as chapter 28 said, and FlyDSL's dispatcher
did not move. That alone takes the 1-2.5% lead FlyDSL had above 1024 rows.
But the spread is the other finding. The GEMV and FlyDSL did not change,
yet their ratios' 10th-90th percentiles span -5% to +7% and -3% to +5%. One
`route_tune.py` run is noisier, point by point, than the 2% the overlay asks
it to resolve.

How much of that is between sessions: a second run of the same tree, two
hours later. Point by point, each path's ratio to vLLM's op moved -2% to +2%
(10th-90th percentile). The two runs routed 15 of the 740 points
differently, against 80-81 between either one and the shipped table. So the
table's change is the kernels', and about 2% of its points are the run's.
- The runs differed at three teeth: qkv's 272, down's 272-288 and o's 96.
  Two of the three teeth the first run had moved went back.
- FlyDSL took 41 points in the first run and 37 in the second, 33 of them in
  both. Above 1024 rows both kept gate_up from 8192 rows up (FlyDSL's
  256x256 tile) and down at 2048, 3328 and 3456.
- At 96 rows the second run sent down to FlyDSL's 32x32 tile rather than
  vLLM's op; the other decode rows at 96 and 128 agreed.

The table shipped is the first run's, the one measured end to end below.

## SGLang's table

`route_tune.py --engine sglang` in SGLang's venv, without FlyDSL as before
(HuggingFace's `kernels` package shadows FlyDSL's there), took 72 s.
One change: o's M = 80-96 goes to hk's GEMM, 3.0-4.1% faster than F.linear
there, where it had been 3.5-4.4% slower. Everything else routes as before.
Against F.linear, which both tables timed, hk's GEMM gained a median 3.9%
over all Ms and 4.2% above 1024 rows; the GEMV, unchanged, moved +0.4%.

SGLang was not re-run end to end. At chapter 24's concurrency points (1, 8
and 32) a decode step's M is at most 32, which both tables give to hk's GEMV,
and a prefill step's is a multiple of 512, which both give to hk's GEMM. The
only change is at 80-96 requests.

## End to end

**Eager, the shapes where chapter 30's prefill rows paid.** In vLLM 0.31's
own process, `hk+gemm` (the attention backend, linear layers by the shipped
table) against `hk+gemm@new` (the same, by the new table), with stock vLLM
beside them. Three rounds, arms rotating, best of each; this tree for both,
FlyDSL installed. Greedy tokens matched stock's, 72 of 72, in every arm.

| ctx x batch | prefill, shipped table ms | new table | change | total, shipped | new | change |
|---|---|---|---|---|---|---|
| 1024 x 1 | 188.65 | 190.94 | +1.2% | 1678.40 | 1680.82 | +0.1% |
| 1024 x 4 | 710.85 | 708.58 | -0.3% | 2373.18 | 2364.15 | -0.4% |
| 4096 x 1 | 776.68 | 773.35 | -0.4% | 2322.72 | 2321.58 | -0.0% |
| 4096 x 4 | 3117.40 | 3063.70 | -1.7% | 4990.75 | 4955.49 | -0.7% |

Each hk arm's three rounds agreed within 0.4% (stock's within 0.8%). The
two tables' decode steps were within 0.7% of each other. FlyDSL took 3,240 calls a round with the shipped table, as
in chapter 30, and 504 with the new one.
- **1024 x 1 is a loss, and a negative result for the table.** Its one
  prefill step has M = 1024, where the shipped table sent qkv, o and down
  to FlyDSL and the new one sends them to hk. Per call, the new measurement
  put FlyDSL ahead by 0.2% summed over the three; in the model, its rows
  were worth 2.3 ms of that 190 ms step, 1.2%.
- **4096 x 4 is a win from a new row.** vLLM batches 8192 tokens a step
  here, and the new table sends gate_up's M = 8192 to FlyDSL's 256x256
  tile, which the shipped one kept on hk.
- Against stock, `hk+gemm` measured 1.186-1.567x per request at these four
  shapes; chapter 30 had 1.179-1.556x.

**Graph mode, chapter 30's decode shapes** (512-token prompts, 160 tokens,
as vLLM ships). The two tables route the decode steps at 96 and 128
requests identically, so this checks that, and the prefill. The arms are
`hk+gemm@old` and `hk+gemm@new`, the shipped table and the new one, with
stock; three rounds, rotating, best of each. Greedy tokens matched stock's,
160 of 160.

| ctx x batch | prefill, shipped table ms | new table | change | decode ms/tok, shipped | new | total, shipped | new |
|---|---|---|---|---|---|---|---|
| 512 x 96 | 8367.64 | 8238.37 | -1.5% | 51.18 | 51.66 | 16544.46 | 16523.99 |
| 512 x 128 | 11214.53 | 11041.54 | -1.5% | 56.08 | 56.16 | 20016.65 | 19982.60 |

- **Decode is the same.** Each round's own decode step ranged
  51.18-51.55 ms against 51.43-51.83 at 96 requests, and 55.55-56.16
  against 55.58-56.21 at 128. Against stock that is 1.426x and 1.654x with
  the shipped table, 1.413x and 1.651x with the new one; chapter 30
  measured 1.394x and 1.666x. FlyDSL ran in the captured graphs, 2,052
  launches a round with the shipped table (chapter 30's count) and 1,908
  with the new one, which routes some of the smaller capture sizes
  differently.
- **Prefill is 1.5% shorter**, in every round. vLLM's prefill steps here
  are 8192 tokens, where the new table sends gate_up to FlyDSL's 256x256
  tile, as at 4096 x 4.
- Prefill with the shipped table took 8368 and 11215 ms, against chapter
  30's 8549 and 11447 on the same table, while stock's totals moved only
  +0.2-0.3%: hk's faster GEMM, about 2% of the step.

## So

- **The gate found no regression against chapter 26**, and 33 pairs better,
  by what chapters 27 and 28 claimed. The one pair worth a second look, the
  one-row fused add, measured the same on both trees.
- **The refreshed baseline gates less**: 40 pairs, and no gate at all on
  the one- and 32-row norms, because hk's kernels there run in 2.4-3.9 us
  depending on the process (chapter 26's, in 3.2-5.1). That is a property
  of hk's kernels on both trees, which a baseline of four quiet runs had
  hidden.
- **The vLLM table is shipped re-measured.** Its decode rows stand, and
  route 96 and 128 requests as before. Most of its prefill rows went,
  because hk's GEMM gained 3% on vLLM's op and FlyDSL's did not. End to end
  the new table is 1.5-1.7% faster at prefill where steps are 8192 tokens,
  1.2% slower at a 1024-token step, within 0.4% elsewhere, and the same at
  decode; per request, -0.7% to +0.1%.
- **The 1024-token step is a negative result.** The rows the table dropped
  there were worth 1.2% in the model and measured under 2% per call. Two
  runs of `route_tune.py` on one tree disagree at 2% of their points, so a
  2% bar resolves less than it appears to.
- **SGLang's table is shipped re-measured**: one change, o at 80-96 rows,
  which chapter 24's concurrency points do not reach.

## Tests

The tests that pin the tables' contents passed unchanged on the new ones:
in `tests/hk/ir`, `test_vllm_gemm.py`, `test_sglang_linear.py`,
`test_engine.py`, `test_flydsl_gemm.py` and `test_hipkittens.py` (the whole
tier: 639 passed, 2 skipped), and on GPU 3 `tests/hk/gpu/test_vllm_gemm.py`,
`test_flydsl_gemm.py` and `test_engine.py` (70 passed). The comb the IR test
pins at qkv (128, 144, 192, 208, 256, 288, 304) is the same; the tooth that
moved is 272, between two of its points.

## What is left

1. **hk's small-row kernels are fast in some processes and not in others**,
   2.4 against 3.1-3.9 us, on chapter 26's kernels and on the strips. Find
   what differs between processes: the kernel-argument line, the code
   object's placement, the clock. Until then those four cases have no gate.
2. **The overlay's 2% bar is inside one run's noise at prefill sizes.**
   Two ways: time FlyDSL and hk at more rounds there, or require a row to
   win in two runs.
3. **The minimum over rounds trusts an impossible round.** The scoreboard
   could drop a best round far ahead of the second, or report a median;
   either changes every ratio, so the baseline would be measured again.
4. **The gate's baseline records hk 0.3.1.** A release that bumps the
   version without changing a kernel will make the check print its "due a
   refresh" note.
5. Chapter 30's other items stand: FlyDSL at Ms its tiles do not divide;
   SGLang's table with FlyDSL rows (its venv shadows FlyDSL's `kernels`);
   FlyDSL's losses at 272-1024 rows.

## Reproduction

```
# in the engines' pod: MLIR from the dev pod, hk._mlir built for its python3.12
kubectl exec <dev-pod> -- tar -C /workspace/llvm -cf - install-cp312 \
    | kubectl exec -i <vllm-pod> -- tar -C <workspace>/measure/llvm -xf -
python3.12 -m venv buildenv && buildenv/bin/pip install nanobind==2.15.0 lit==23.1.3
HK_LLVM_ROOT=<workspace>/measure/llvm HK_MLIR_BUILD=<workspace>/measure/hk-build-cp312 \
    PYTHON=buildenv/bin/python scripts/build_hk_mlir.sh

export HIP_VISIBLE_DEVICES=3 FLYDSL_ROOT=/workspace/FlyDSL PYTHONPATH=python
PYTHON=/workspace/venv-vllm031/bin/python HK_GPU=3 scripts/perf_gate.sh   # against the old baseline
python3 tools/hk-bench/scoreboard.py --only norm --arms hk,inductor,triton \
    --rounds 7 --calls 50 --json norm.json                                # each tree, alternating
python3 tools/hk-bench/scoreboard.py $(python3 tools/hk-bench/check_regression.py --print-args) \
    --json sb_N.json                                                      # six times
python3 tools/hk-bench/check_regression.py --refresh sb_*.json

python3 tools/hk-bench/route_tune.py --model Qwen/Qwen3-8B --out routes_vllm.json
/workspace/venv-sgl/bin/python tools/hk-bench/route_tune.py --engine sglang \
    --model Qwen/Qwen3-8B --out routes_sglang.json                        # FLYDSL_ROOT unset
python3 -m hk.integration ab --context 1024,4096 --batch 1,4 --tokens 72 --short 24 \
    --arms hk+gemm,hk+gemm@new --routes new=routes_vllm.json
HK_EAGER=0 python3 -m hk.integration ab --context 512 --batch 96,128 --tokens 160 --short 128 \
    --arms hk+gemm@old,hk+gemm@new --routes old=routes_old.json,new=routes_vllm.json
```
