# FlyDSL on the routes, a performance gate, and the engines' contracts

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Three pieces of plumbing around the kernels, each the answer to a way the
project could be wrong without noticing:
- the integration layer only knew hk's kernels and the engine's, while
  [chapter 26](26-the-scoreboard-retuned.md) had FlyDSL's GEMM ahead of hk's;
- nothing caught a kernel getting slower, because absolute times on the
  shared node drift more than most regressions;
- every patch rebinds engine internals by name, and nothing caught an engine
  release moving one.

## FlyDSL's GEMM on the routes

Chapter 26 left a division: FlyDSL's RDNA3 GEMM, through its own dispatcher,
is 2-5% faster than hk's at prefill shapes, while hk's decode GEMV and
attention have no FlyDSL counterpart that beats them. hk-dsl's place is
beside FlyDSL, so the linear-layer op `hk_vllm::linear` now has a path for
it, `flydsl`, taken where a measured table says FlyDSL is fastest.

**Installing FlyDSL takes two halves.** PyPI's `flydsl` 0.4.0 is the
compiler and runtime. The kernels, `kernels/gemm/rdna3_f16_gemm_autotune.py`
and its siblings, are in FlyDSL's repository and not in the wheel. The
checkout has to match the wheel: the v0.4.0 tag, commit 1f35116, cloned
beside the vLLM venv as `/workspace/FlyDSL`. `FLYDSL_ROOT` names it.
The SGLang venv has HuggingFace's unrelated `kernels` package installed,
which would shadow FlyDSL's. `flydsl_gemm` says so rather than failing
with an import error three modules deep.

**What a call costs** (gfx1100, GPU 3):
- A new M is a compile, 0.4-0.8 s cold and 0.04 s from FlyDSL's disk cache.
  M is a compile-time constant of the kernel.
- A call is about 60 us of host time through `rdna3_gemm_dispatch`. The
  launcher alone is 46 us and `pick_tile` 2.7; hk's op takes 18.
- So calling around the public dispatcher would save 5 us, and the path uses
  the dispatcher. At the Ms FlyDSL wins the GEMM takes longer than the host
  does.

**The tile decides the speed, so the table names the tile.** FlyDSL's tiles
must divide the shape, with M a multiple of 32 at least, and its dispatcher
picks among the ones that do. At (4096, 4096):

| M | the tile FlyDSL picks | FlyDSL ms | hk ms |
|---|---|---|---|
| 128 | 64x64x64, 4 waves | 0.080 | 0.115 |
| 1024 | 128x128x32, 4 waves | 0.429 | 0.453 |
| 1056 | 32x32x64, 1 wave | 0.658 | 0.463 |
| 4096 | 128x128x32, 4 waves | 1.574 | 1.627 |

A route that said "FlyDSL from 1024 to 2048" would send 1056 rows to a
kernel 42% slower than hk's. So the table's FlyDSL rows are ranges per tile,
`[lo, hi, "128x128x32w4"]`. A call takes the path only if FlyDSL's
dispatcher would pick that same tile for its M (`flydsl_gemm.variant`).

**And they are an overlay, not routes.** The table's runs are measured
exactly as before, from hk's kernels against the engine's, and they are
what every install routes by. FlyDSL's rows sit beside them
(`RouteTable.preferred`). Where FlyDSL is not installed, or its tile there
was not the one measured, or no tile divides M, a call goes where a table
without FlyDSL would send it. Making FlyDSL one more path in the runs would
have been simpler, but wrong in two ways:
- an install without FlyDSL would then send those Ms to the engine instead
  of to hk;
- the runs would break into alternating rows: hk at the odd multiples of 16,
  FlyDSL at the even ones, and the engine's path at every M between two
  rows that disagree.

**`route_tune.py` times FlyDSL through the same call the op makes**, at
every M a tile divides, and records the tile. A FlyDSL row needs FlyDSL to
beat the engine by `min_gain` (3%) and the runs' own answer by `opt_gain`
(2%). Consecutive measured Ms of the same tile merge; Ms of other tiles,
and Ms where FlyDSL did not run, do not break a range, since the tile check
excludes them at run time. Qwen3-8B's five shapes took 214 s:
- the runs came out identical to the table shipped two days earlier, but for
  one tie at gate_up M = 3, between vLLM's skinny GEMM and hk's GEMV;
- FlyDSL ran at 304 of the 740 (shape, M) points and was routed at 105.

Against the runs' answer, FlyDSL's time by region:

| M | qkv | o | gate_up | down |
|---|---|---|---|---|
| 64-256, median | 0.84x | 0.85x | 0.87x | 0.97x |
| 272-1024, median | 1.19x | 1.14x | 1.33x | 1.22x |
| above 1024, median | 0.98x | 0.98x | 0.99x | 0.975x |

The first row is the useful finding. Below 256 rows hk's GEMM is a comb, one
128-row block computed whole however much of it is kept, while FlyDSL's 32-
and 64-row tiles fill the grid: 0.63-0.96x hk's time at 96-256 rows. The
middle row is FlyDSL's dispatcher with only small tiles left, and is never
routed. The last row is chapter 26's 2-5%, often under the 2% bar.

**End to end, two runs.** Three arms in vLLM 0.31's own process, alternating,
three rounds, best of each: stock; `hk+gemm` (the attention backend and the
table shipped before); `+FlyDSL` (the same, with the new table). Every arm's
greedy tokens matched stock's.

At the shapes every vLLM chapter has used (eager, 72 tokens), FlyDSL took
3,240 of hk's 30,816 GEMM calls a round. Those are the pure-prefill steps of
1024 and 4096 tokens, on qkv, o and down (at gate_up FlyDSL was 1-2% faster
at those Ms, under the 2% bar):

| ctx x batch | total, hk+gemm ms | +FlyDSL | prefill, hk+gemm ms | +FlyDSL |
|---|---|---|---|---|
| 1024 x 1 | 1686.58 | 1682.37 | 196.56 | 191.56 (-2.5%) |
| 1024 x 4 | 2392.51 | 2381.10 | 727.52 | 721.03 (-0.9%) |
| 1024 x 16 | 5020.04 | 5022.69 | 2934.70 | 2937.27 |
| 4096 x 1 | 2343.31 | 2339.26 | 795.19 | 786.91 (-1.0%) |
| 4096 x 4 | 5069.63 | 5055.22 | 3183.78 | 3183.20 |
| 4096 x 16 | 15732.51 | 15747.82 | 12728.54 | 12745.88 |

At 16 requests, and at 4096 x 4, every prefill step after the first also
carries decodes, so its M is 4096 plus a few, which no tile divides. Decode at 1-16 requests
never reaches FlyDSL: it is hk's GEMV or vLLM's skinny GEMM. So the FlyDSL
rows are worth 1-2.6% of prefill there, and nothing measurable in total.

At 96 and 128 requests (graph mode, as vLLM ships; 512-token prompts, 160
tokens, chapter 14's shape), a decode step's M is in FlyDSL's best range:

| ctx x batch | decode, stock ms | hk+gemm | +FlyDSL | total, hk+gemm x | +FlyDSL x |
|---|---|---|---|---|---|
| 512 x 96 | 71.72 | 58.58 (1.224x) | 51.44 (1.394x) | 1.266x | 1.343x |
| 512 x 128 | 92.32 | 62.95 (1.467x) | 55.42 (1.666x) | 1.368x | 1.442x |

The decode step is 12% shorter, and the request 5-6%. Prefill is unchanged,
8543 against 8549 ms. FlyDSL ran inside vLLM's captured decode graphs, 2,052
captured launches a round. `tests/hk/gpu/test_flydsl_gemm.py` holds that:
it captures and replays FlyDSL's path under the op's `graph="safe"`
contract, which makes a host sync raise.

**So, honestly:**
- The FlyDSL rows win end to end where a step's M is one FlyDSL's tile
  divides and was measured faster: decode at 96 and 128 requests, by 12%
  a step (the rows reach 192-384 requests, depending on the shape), and
  pure-prefill steps of tile-aligned length, by 1-2.6%.
- Everywhere else they change nothing, by construction.
- The table shipped in `hk/tuned/` now carries them. An install without
  FlyDSL routes exactly as before.
- The 2-5% chapter 26 measured on square prefill GEMMs is mostly not what
  pays. What pays is that FlyDSL's dispatcher has small tiles for the Ms
  where hk's 128-row GEMM is weakest.

## A performance gate on ratios

Nothing caught a kernel getting slower. The obvious gate, absolute times
against a stored number, cannot work here. The node is shared, and absolute
times drift 3-5% between sessions, which is more than most regressions worth
catching. A gate set loose enough not to fire on drift catches nothing.

What does not drift is a ratio taken inside one run. The scoreboard already
times every arm in one process, interleaved and rotated, so a node that is
slower this hour is slower for torch, Triton and FlyDSL too. The gate,
`tools/hk-bench/check_regression.py`, compares t_arm / t_hk for each case
and each arm against a baseline, and never compares a time with a time from
another run. `scripts/perf_gate.sh` runs the scoreboard with the baseline's
arguments, then the check:
- the out-of-the-box arms only, via a new `--arms` (the tuned arms search on
  every run, which takes minutes and is a noise of its own);
- every case at 7 rounds of 50 calls, so the microsecond norms are
  measurable at all;
- about 80 seconds a run on GPU 3.

The GitHub runner has no GPU, and a self-hosted runner needs the owner's
approval. So the gate runs on the dev pod: by hand, nightly, or as the last
group of `HK_PERF_GATE=1 scripts/gpu_suite.sh`.

**The tolerance is measured, per pair.** `--refresh` builds the baseline
from several runs, each its own process. For each (case, arm):
- the ratio is the median across the runs;
- the noise is the larger of the spread across runs (max |r_i / median - 1|)
  and the spread within a run (the two arms' second-best-round spreads,
  summed, at their worst);
- the tolerance is max(3%, 3 x noise);
- a pair whose tolerance would pass 15% is not gated, and is listed with its
  noise.

The file keeps every run's ratio beside the tolerance it produced, a line
summarizing the noise, and the rule itself. A tolerance can be checked
against its noise rather than loosened by hand.

The noise is very uneven, which is why it is per pair. Two runs a minute
apart:
- most GEMM, softmax and 2048-8192-row norm ratios agreed within 0.5%;
- FlyDSL's 32-row RMSNorm took 1.18x hk's time in one and 0.83x in the next,
  bimodal across processes with a quiet within-run spread (1.5%);
- one FlyDSL GEMM round wandered 5.6% within a run.

A single global tolerance would have been either loose for the GEMMs or
firing on the norms.

**The shipped baseline.** Four runs: two before and two after the 70-minute
vLLM run above, on the same GPU, so session drift is in it.
- 45 pairs are gated over 16 cases, and 32 of them sit at the 3% floor.
  The GEMM ratios moved 0.02-1.5% across the four runs.
- The widest gated tolerance is 12.7%, Inductor against the 32-row RMSNorm.
- 12 pairs are not gated: the torch and FlyDSL arms of the one- and 32-row
  norms, Triton's 32-row RMSNorm, every arm of the one-row RMSNorm (which
  therefore has no gate at all), and FlyDSL against the 2048x4096x4096 GEMM.

**Does it fire?** Run against it:
- **A fifth run, clean:** no regression; the largest change used 68% of its
  pair's tolerance (Triton against the 2048-row RMSNorm, -3.0% of 4.4%).
- **hk's GEMM made slower on purpose,** `k_step` 32 instead of 64 in
  `gemm.DEFAULT` (9-12% a call, measured first): exit 1, all 14 GEMM pairs
  flagged at -11.6% to -15.2% against tolerances of 3.0-4.6%. Nothing else
  was flagged.
- **Reverted:** no regression, the GEMM ratios back within 0.3%; the
  largest change used 63% of its tolerance.

`tests/hk/ir/test_check_regression.py` holds the arithmetic on made-up
reports:
- a node slower across the board passes;
- hk slower against its neighbours fails;
- a gated arm missing from the run is not a pass (exit 2: FlyDSL not
  installed, say);
- a pair noisier than the cap is not gated.

How and when to refresh is in
[building](../building.md#the-performance-gate).

The baseline was measured on hk as of chapter 26. While this work was
merging, [chapter 27](27-strips.md) landed. It puts the norms and softmax
on strip kernels and changes how every RDNA kernel reads its arguments, so
hk's side of most norm and softmax pairs has moved. The first gate run on
that tree will say which pairs moved, and the baseline then wants a refresh.
It could not be refreshed here: the dev pod's Python 3.12 venvs carry the
0.3.1 wheel's dialect bindings, which do not build chapter 27's kernels.

## The engines' contracts

The patches rebind, by name or by identity, about twenty symbols across vLLM
and SGLang. An engine release that renames one, reorders its parameters or
moves the call that reaches it breaks a patch quietly:
- the rebind finds nothing, and the patch reports "0 bindings rebound" in a
  log line nobody reads;
- or the stand-in forwards its arguments positionally into the wrong slots;
- or a backend hook stops being read, and becomes dead code.

`tests/hk/engines/` asks the installed engine. It is a tier of its own,
skipped where the engine is not importable. Three kinds of check:
- **Signatures, as the stand-ins depend on them.** A stand-in that forwards
  positionally must take the engine function's parameters in the same order,
  with the same defaults. That covers the two MoE entry points in each
  engine, SGLang's `topk_softmax`, the AWQ method and its two ops,
  `reshape_and_cache_flash`, and the `(layer, x, weight, bias)` that
  `dispatch_unquantized_gemm()` hands back.
- **Call sites.** `_contract.call_sites` disassembles every function and
  method of every loaded engine module, and lists those that load the
  patched name as a global. Each must hold the original object, or a rebind
  by identity misses it. For `dispatch_unquantized_gemm` on vLLM 0.31 that
  is `linear` (now in `UnquantizedLinearMethod.__init__`, which stores the
  result as `_gemm_impl`, so the patch must land before the model is built,
  as the plugin does) and `vocab_parallel_embedding` (the LM head, per call).
- **Behaviour that only a call shows.** hk's `triton_decode` falls back to
  hk's walk, silently, when vLLM's `unified_attention` lacks an argument it
  passes. The test drives it with CPU tensors and a capturing stand-in for
  the kernel, and fails on that fallback. SGLang's
  `try_get_optimal_moe_config(return_down_config=True)` is called to check
  the `(config, (down, max_block_m))` shape the stand-in unpacks. Its tuned
  file lookup needs a running server's context, so the test goes through
  SGLang's own override hook (`triton_utils.get_config`).

Run on the dev pod:

| venv | engine | passed |
|---|---|---|
| `venv-vllm031` | vLLM 0.31.0+rocm721 | 19 of 19 (SGLang's file skipped) |
| `venv-sgl` | SGLang 0.5.18, sglang-radeon 0.1.5 | 8 of 8 |
| `venv-sgl` | vLLM 0.16.1, installed beside SGLang | 18 of 19 |

The one failure on 0.16.1 is real: it has no `quantization.auto_awq`, so the
4-bit patch, written against 0.31, cannot apply there.

The first run also found something about 0.31 itself. Four of the attention
backend's overrides are not on vLLM 0.31's `AttentionBackend` at all, and
grep confirms 0.31 reads none of them off a backend:
- `accept_output_buffer`: forward's output buffer is now always passed;
- `get_kv_cache_shape`: the layout comes from `supported_kv_cache_layouts`;
- the backend's `use_cascade_attention`: vLLM asks the metadata builder;
- `validate_head_size`: hk's own, called by hk.

None of this breaks anything, since the hooks 0.31 does read are all there.
But a test that let them pass as live would let the next rename pass too. So
every override is now classified as read by vLLM, as one of the two layout
hooks (`supported_kv_cache_layouts` from 0.31, `get_kv_cache_shape`
before), as legacy, or as hk's own. An unclassified override fails the test.
So does a hook in the first group missing from the base class, or neither
layout hook being there. The run also pinned one assumption nothing had
checked: `forward_includes_kv_cache_update` is True on 0.31's base. hk's
forward writes K/V itself, so a backend that said otherwise would get a
second write from vLLM.

SGLang needs a GPU visible to import its kernels at all: `fp8_kernel` asks
device 0 for its arch at import time. Its tests launch nothing, but they
cannot run with the GPUs hidden.

The supported-version matrix is in
[integration](../integration.md#supported-versions-and-the-contract-tests).

## Along the way

**torch 2.13's attention is much faster.** The gate's baseline runs on the
vLLM 0.31 venv's torch 2.13.0+rocm7.2. There, SDPA runs the causal n4096
GQA-4 case in 3.24-3.27 ms; chapter 26 measured 10.83 ms on torch 2.10. hk's
attention is 1.16x torch causal and 0.99x non-causal on 2.13, where chapter
26 had 3.80x and 2.73x. hk's attention kernel did not change (2.81 ms against
2.85); the opponent did. The claim "hk wins attention" holds against torch
2.10 and the engines' Triton kernels, and against torch 2.13 only causally.

**torch 2.13 fails a host sync under capture differently.**
`tests/hk/gpu/test_engine.py`'s host-sync test failed on the 0.31 venv, and
it failed identically with the installed 0.3.1 wheel, so the failure was not
new code. On torch 2.13 a `.item()` under capture fails first in HIP, with
"operation not permitted when stream is capturing", before torch's
sync-debug mode can name it. `engine.no_host_sync` only recognized the
debug mode's message, so the error escaped without naming the patch. It
recognizes both now, and the GPU tests that cover the integration layer
(`test_engine`, `test_vllm_gemm`, `test_flydsl_gemm`, 70 tests) pass.

## What is left

1. **FlyDSL at Ms its tiles do not divide.** A mixed step's M is 4096 plus
   a few, and FlyDSL cannot take it. Padding M up to a tile, or splitting off
   the remainder to hk, would extend its wins to arbitrary prefill steps; the
   pad costs a copy of the activation. Not tried.
2. **SGLang's table with FlyDSL rows.** `route_tune.py --engine sglang`
   measures them the same way, and SGLang's linear goes through the same op.
   Not run here.
3. **Under 272-1024 rows FlyDSL loses 13-33%.** Its dispatcher picks small
   tiles there, and the table keeps hk. FlyDSL's autotuner, searching
   per M, might find better, but it compiles and times every candidate on
   every call while `FLYDSL_AUTOTUNE=1`, which no server can afford.
4. **Refresh the gate's baseline on a tree with chapter 27's strips,** on
   a pod whose hk dialect bindings match it (see above).
5. **The one-row RMSNorm has no gate.** Its arms move 7-19% between
   processes at 50 calls a round. A gate on it needs a steadier
   measurement: more calls, or a graph-replayed timing.
6. **The 0.31 legacy hooks** (`accept_output_buffer`, `get_kv_cache_shape`,
   the backend's `use_cascade_attention`) can go when 0.16.1 stops mattering.

## Reproduction

```
# FlyDSL beside the vLLM venv: the wheel, and the matching repository tag
pip install flydsl==0.4.0
git clone --depth 1 --branch v0.4.0 https://github.com/ROCm/FlyDSL /workspace/FlyDSL
export FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=3 PYTHONPATH=python

python3 tools/hk-bench/route_tune.py --model Qwen/Qwen3-8B --out routes.json
python3 -m hk.integration ab --context 1024,4096 --batch 1,4,16 --tokens 72 --short 24 \
    --arms hk+gemm,hk+gemm@fly --routes fly=routes.json
HK_EAGER=0 python3 -m hk.integration ab --context 512 --batch 96,128 --tokens 160 --short 128 \
    --arms hk+gemm,hk+gemm@fly --routes fly=routes.json

HK_GPU=3 scripts/perf_gate.sh                     # the gate
python3 -m pytest tests/hk/engines -q             # in each engine's venv
python3 -m pytest tests/hk/gpu/test_flydsl_gemm.py tests/hk/gpu/test_engine.py -q
```
