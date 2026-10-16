# Testing

- [The tiers](#the-tiers)
- [Running each tier](#running-each-tier)
- [What the GPU tier holds the line on](#what-the-gpu-tier-holds-the-line-on)
- [Examples, and checks at full size](#examples-and-checks-at-full-size)
- [CI](#ci)
- [The performance gate](#the-performance-gate)
- [The gates for a change to the generated code](#the-gates-for-a-change-to-the-generated-code)
- [Reproducing](#reproducing)

## The tiers

Five tiers, by what they need. The first three need no GPU:

```bash
ninja -C $HK_MLIR_BUILD check-hk                     # lit: the dialect, passes, translation
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
```

| tier | needs | what it catches |
|---|---|---|
| `mlir/test/` | the MLIR build | op verifiers, pass results and errors, the C++ for hand-written IR |
| `tests/hk/ir/` | hk._mlir | tracing, the language, pass errors from Python, generated C++ |
| `tests/hk/codegen/` | hipcc | compiles; spill/occupancy gates fire |
| `tests/hk/gpu/` | an AMD GPU + torch | numerics vs torch |
| `tests/hk/engines/` | vLLM or SGLang importable | an engine release moving a symbol a patch rebinds |

A tier that cannot run is skipped with a reason, never silently passed.
Performance has a gate of its own, ratios against the arms beside hk in one
scoreboard run ([the performance gate](#the-performance-gate)).

**Tests, at chapter 28.**
- `mlir/test` (lit): 10 files, for:
  - the op verifiers and a round trip through text;
  - `hk-verify`, `hk-lds-alloc`, `hk-lds-pipeline` and `hk-check-budgets` on
    hand-written IR, including the workgroup-operation checks, the budget
    warning, and the refusal of a plain barrier after an ordinary LDS store.
    The fifth pass, `hk-fragments`, is tested from Python
    (`tests/hk/ir/test_cdna.py`);
  - the translation of a kernel, of indexed and row accesses, and of a
    `while`.
- tests/hk/ir and tests/hk/codegen: 1674 collected (641 and 1033). The one
  xfail is conditional: HipKittens' CDNA4 attention does not compile with
  ROCm 7.2's clang.
- tests/hk/gpu: run in groups by `scripts/gpu_suite.sh` ([CI](#ci)). On
  gfx950 the attention and RDNA GEMM tests skip
  ([chapter 29](history/29-cdna.md)).

[The MLIR chapter](history/15-mlir-dialect.md) records the counts as they
stood at the move (990 no-GPU tests, 386 on the GPU).

## Running each tier

* **lit** — `ninja -C $HK_MLIR_BUILD check-hk`, where `$HK_MLIR_BUILD` is the
  directory `scripts/build_hk_mlir.sh` built in (default
  `/dev/shm/hk-llvm/hk-build-cpXY`; see [building](building.md#mlir-and-the-dialect)).
  The tests are `mlir/test/Dialect`, `mlir/test/Transforms` and
  `mlir/test/Target`, each file driven by its `RUN:` line through `hk-opt`,
  `hk-translate` and `FileCheck` ([the tools](ir.md#hk-opt-and-hk-translate)).
* **ir** — `python3 -m pytest tests/hk/ir -q`. Needs `hk._mlir`; no compiler,
  no GPU, no torch.
* **codegen** — `python3 -m pytest tests/hk/codegen -q`. Needs hipcc, no GPU;
  generates and compiles, never launches. Skipped, with that reason, when
  there is no working hipcc.
* **gpu** — `python3 -m pytest tests/hk/gpu -q`. Needs an AMD GPU and torch;
  skipped when either is missing. This is the tier that launches, and the
  only one that can catch an indexing error — a kernel that reads the wrong
  tile compiles perfectly and passes every gate. `tests/hk/conftest.py` pins
  `HK_ARCH=gfx1100` unless one is set, so on another GPU set `HK_ARCH` to it.
  `test_flydsl_gemm.py` also needs FlyDSL (`FLYDSL_ROOT`), and says so when
  it skips.
* **engines** — `python3 -m pytest tests/hk/engines -q`, in each engine's
  venv. Every symbol a vLLM or SGLang patch rebinds, with the signature its
  stand-in forwards to and the call sites the rebind must reach; a module is
  skipped when its engine is not importable. SGLang's needs a GPU visible,
  since it reads the arch while importing. The versions it was run against
  are in [integration](integration.md#supported-versions-and-the-contract-tests).

The gates are by directory, in `tests/hk/conftest.py`, which also puts
`python/` on `sys.path`, so the suite runs from a checkout without installing.
A test outside those directories opts into a gate with the `hipcc` or `gpu`
marker (`pyproject.toml`). `python3 -m pytest` with no path runs `tests/hk`.

## What the GPU tier holds the line on

Numerics against torch, with tolerances per dtype rather than global (bf16
carries about three decimal digits), plus a few tests with exact answers. Two
of those came out of a real bug: a biased bf16 conversion that a tolerance
sized for indexing errors could never have seen
([Phase 6](history/05-phase-6-torch-ops.md)):

Two tests now hold the line, both with exact answers rather than a tolerance
(`tests/hk/gpu/test_attn.py`): `V = 1` must come back bit-exact, and `Q = 0`
must give the mean of V. The file's existing `RTOL = 5e-2` could never have
seen this -- it is sized to catch a shifted mask, and a half-ulp bias passed
all 31 of its cases while changing what a model said.

The GPU tier also covers the integration layer: `torch.compile(fullgraph=True)`,
CUDA-graph capture and replay, and the strided-ViT-input contract in both
directions.

## Examples, and checks at full size

Every kernel in [`examples/`](../examples) is held at three tiers, through
`examples/_catalog.py`:
- tests/hk/ir/test_examples.py traces and translates it, and pins the facts
  its docstring teaches: the GEMM's waits step down 4 then 0, the attention's
  rescale is a runtime branch, and the control-flow example carries its
  scalars.
- tests/hk/codegen/test_example_builds.py compiles it, with no spill.
- tests/hk/gpu/test_example_runs.py runs every script with its check against
  torch.

The GPU tier's own shapes are small on purpose: they hunt indexing errors at
ragged edges, and a handful of workgroups is enough for that. It is not
enough for a race between workgroups that share a WGP. The cross-warp combine
had one for as long as hk had norms, and only a full-size run showed it
([the record](history/16-validation-and-the-cross-warp-race.md)). Two things
now run at the sizes a model produces:
- [`examples/validate.py`](../examples/validate.py) checks every shipped op at
  7-8B decoder shapes against an fp32 reference, then times it against torch.
  Its exit status is the number of failures.
- tests/hk/gpu/test_phase2.py runs rmsnorm, layernorm and softmax at
  8192x4096 and 16384x1024, and quantize at 4096x4096, each twice.

```bash
python3 examples/validate.py --quick --no-bench   # every op checked, smaller shapes
python3 examples/validate.py                      # full size, then timed
```

## CI

`.github/workflows/ci.yml` runs the two tiers that need no GPU on every push
and pull request, on GitHub's `ubuntu-22.04` runner:
- **MLIR is prebuilt.** It downloads MLIR 23.1.3, installed by
  `scripts/build_mlir.sh` for that runner's CPython 3.10, from the
  `ci-mlir-23.1.3` release (0.9 GB). Building LLVM on the runner would take
  hours.
- **The dialect is built.** `scripts/build_hk_mlir.sh` builds it against
  that MLIR in about 20 seconds.
- **The IR tier runs** (`tests/hk/ir`).
- **hipcc is installed from AMD's apt repository**, as `hip-dev` and
  `rocm-device-libs` of ROCm 7.1, about 2.3 GB.
- **The codegen tier runs** (`tests/hk/codegen`), under pytest-xdist. hk's
  compile cache survives between runs in `actions/cache` (about 100 MB).
  Cold, the tier took 19-30 minutes on the runner's four cores, and 4
  minutes on 8 workers in a clean Ubuntu 22.04 container. Warm, it took 10
  seconds, and the whole run 3 minutes. The first run on a new branch is
  cold, because `actions/cache` does not hand one branch's cache to another;
  pull requests read `main`'s.

A new MLIR, or a CPython other than the runner's, needs a new archive. Build
it with `scripts/build_mlir.sh`, pack the install tree with LLVM's and MLIR's
licence files, attach it to a release, and point `MLIR_RELEASE` and
`MLIR_ASSET` at it.

The GPU tier runs on the development pod, not in CI:
`HK_GPU=3 scripts/gpu_suite.sh OUTDIR` runs `tests/hk/gpu` in groups, three
at a time on one GPU, and exits non-zero if a group failed. Moving it into
CI means a self-hosted runner on that pod, which runs code it is handed. On a
public repository that runner must only take pushes to `main` and manual
dispatches, never a pull request from a fork, and it must stay on the GPUs
that are this project's. No such runner is registered.

## The performance gate

`scripts/perf_gate.sh` catches a kernel that got slower. It runs on the dev
pod, alone on one GPU, by hand or nightly; `HK_PERF_GATE=1
scripts/gpu_suite.sh` runs it after the GPU tier as group `perf`.

```bash
HK_GPU=3 FLYDSL_ROOT=/workspace/FlyDSL scripts/perf_gate.sh [OUTDIR]   # exit 1 on a regression
```

**What it compares.** The node is shared, and absolute times drift 3-5%
between sessions, more than most regressions worth catching. So the gate
never compares a time with a time from another run. It runs the scoreboard
(`tools/hk-bench/scoreboard.py`) with the out-of-the-box arms (hk, torch,
Inductor, Triton, FlyDSL), every case at 7 rounds of 50 calls. For each case
and each arm A it takes the ratio t_A / t_hk measured in that one run, and
`tools/hk-bench/check_regression.py` fails if the ratio fell below the
baseline's by more than that pair's tolerance. A slower node moves both
times and not the ratio; a slower hk kernel moves every ratio of its case.
The tuned arms are left out because they search on every run, which takes
minutes and adds a noise of its own.

**Where the tolerance comes from.** `tests/perf/baselines/gfx1100.json` is
built from several scoreboard runs, each its own process, spread over time.
For each (case, arm) the ratio is the median across runs and the noise is
the larger of two spreads: across runs, max |r_i / median - 1|; within a
run, the two arms' second-best-round spreads summed, at their worst. Then
tolerance = max(3%, 3 x noise). A pair whose tolerance would pass 15% is
not gated, and is listed as such with its noise. The file records every
run's ratio beside the tolerance it produced. To change a tolerance,
re-measure the noise; editing the number by hand loosens the gate blindly.

**What the shipped baseline measured.** Four runs on the dev pod, two before
and two after a 70-minute vLLM run on the same GPU, give 45 gated pairs over
16 cases:
- GEMM ratios moved 0.02-1.5% between runs, and the 2048- and 8192-row norms
  and softmax up to 2.9%. 32 pairs sit at the 3% floor, and the widest gated
  tolerance is 12.7% (Inductor against the 32-row RMSNorm).
- Twelve pairs are not gated, because their ratios moved 5-30% between
  processes. They are the microsecond norms' torch and FlyDSL arms (FlyDSL's
  32-row RMSNorm took 1.18x hk's time in one run and 0.83x in the next),
  Triton's 32-row RMSNorm, and FlyDSL against the 2048x4096x4096 GEMM. That
  leaves the one-row RMSNorm with no gated pair at all.
- Two later clean runs used at most 68% and 63% of any pair's tolerance.
- With hk's GEMM made slower on purpose (`k_step` 32 instead of 64 in its
  default schedule, 9-12% a call), the gate failed with exit 1 on all 14
  GEMM pairs (-11.6% to -15.2% against tolerances of 3.0-4.6%) and on
  nothing else. Reverted, it passed.

**The shipped baseline is due a refresh.** It was measured on hk as of
[chapter 26](history/26-the-scoreboard-retuned.md). Since then
[chapter 27](history/27-strips.md) moved the norms and softmax onto strips
and changed how every RDNA kernel reads its arguments, and
[chapter 28](history/28-gemm-schedule.md) changed the GEMM's schedule. Both
moved hk's side of their pairs. The first gate run on a tree with them will
report those pairs as `better` (or as a regression), and then the baseline
wants `--refresh`. That needs the dev pod's hk dialect bindings rebuilt to
match the tree; the pod's Python 3.12 venvs carry the 0.3.1 wheel's.

**Refreshing the baseline.** Refresh it when the opponent changed (a new
torch, Triton or FlyDSL: the check prints a note when the versions differ
from the baseline's), when hk got faster on purpose (pairs that beat their
tolerance are reported as `better`), or for another arch:

```bash
HK_GPU=3 FLYDSL_ROOT=/workspace/FlyDSL scripts/perf_gate.sh --refresh 4
git diff tests/perf/baselines/gfx1100.json      # every moved ratio is a claim
```

`--refresh N` runs the scoreboard N times (at least three) and rewrites the
baseline from them. Back-to-back runs see less drift than runs hours apart;
for a baseline that has to last, run `check_regression.py --refresh` on
reports from separate sessions. A refresh is a commit, with the reason in
its message.

## The gates for a change to the generated code

The move to MLIR was gated three times, each against the code it replaced, and
the record treats the same gates as the bar for later changes: a
canonicalize/DCE pass "will change the C++, so it needs Gate B again,
resources first". From [the MLIR chapter](history/15-mlir-dialect.md):

**Gate A — the dialect against the Python IR.** A temporary converter
turned every kernel the old front end traced into the dialect, and ran it
through the C++ passes and the C++ translation. For all 354 shipped
instantiations, and the constexpr points the tests use (708 translations),
the C++ was byte-identical to the Python emitter's. The whole no-GPU suite
ran under the gate as its corpus — 394 tests, the error paths included,
where the C++ passes had to refuse what the Python ones refused, in the same
words. Then the converter was deleted with the rest.

**Gate B — each kernel family against its old self** (elementwise, norm,
quant, fused, decode GEMV, GEMM, paged, attention):

* All 422 instantiations rendered both ways. 49 are byte-identical, 281
  differ only in a loop variable's name (`i` became the `kb` the source
  wrote), and 92 differ in constant folding, hoist order or names. Those 92
  were compiled: VGPRs, SGPRs, scratch, LDS and occupancy are identical in
  every one.
* tests/hk/gpu: 379 passed on gfx1100.
* Old and new in one process, interleaved for 7 rounds, the old as control
  (the table is in [performance](performance.md#after-the-mlir-move)). Every
  gap is smaller than that row's own round-to-round spread, which is the
  expected answer for kernels whose C++ differs in names.

**Gate C — the engine.** vLLM 0.16.1 serving Qwen3-8B on the W7900D with the
GEMM patch (`vllm_e2e.py`, eager, context 1024, batch 1 and 16, 72 output
tokens, three rounds of each arm alternating), once from the tree before the
move and once after, each against stock vLLM in its own run. The routing
counters match call for call (5,472 prefill GEMMs, 102,553 decode GEMVs, no
fallback). At both shapes the first request's greedy output is identical to
stock, and so to the other run's (72/72).

The first attempt at this gate measured nothing. With vLLM's engine in its
own process, the patch installed in the driver never reached the model, and
both arms timed as stock (0.995–1.000x, `hk 0, hk_decode 0`). The tool prints
the routing counters for exactly this reason. The gate is run with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, as every earlier measurement here was.

Gate B's resource comparison needs only hipcc, and it is the part the record
asks for first ("resources first"): a change that spills is a wrong kernel,
not a slow one ([resource gates](runtime.md#the-resource-gates)).

## Reproducing

```bash
scripts/build_mlir.sh && scripts/build_hk_mlir.sh
ninja -C $HK_MLIR_BUILD check-hk
python3 -m pytest tests/hk/ir tests/hk/codegen -q
python3 -m pytest tests/hk/gpu -q                       # gfx1100
python3 examples/validate.py                            # gfx1100, full size
python3 -m pytest tests/hk/engines -q                   # in each engine's venv
HK_GPU=3 scripts/perf_gate.sh                           # alone on one GPU
scripts/build_wheel.sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 HK_GPU_UTIL=0.85 python3 tools/hk-bench/vllm_e2e.py \
    --context 1024 --batch 1,16 --tokens 72 --short 24 --reps 3 --arms stock,gemm
```
