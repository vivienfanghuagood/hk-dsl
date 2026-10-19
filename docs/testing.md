# Testing

- [The tiers](#the-tiers)
- [Running each tier](#running-each-tier)
- [What the GPU tier holds the line on](#what-the-gpu-tier-holds-the-line-on)
- [Examples, and checks at full size](#examples-and-checks-at-full-size)
- [The agents' tools](#the-agents-tools)
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
- tests/hk/ir and tests/hk/codegen, in CI at `596522c`: 639 passed and 4
  skipped (the IR tier, with `tests/hk/engines` skipping), and 1033 passed.
  The one xfail is conditional: HipKittens' CDNA4 attention does not compile
  with ROCm 7.2's clang.
- tests/hk/gpu: run in groups by `scripts/gpu_suite.sh`, on main on a
  W7900D and an MI355X ([the GPU tier](#the-gpu-tier)). On
  gfx950 the attention and RDNA GEMM tests skip
  ([chapter 29](history/29-cdna.md)), and so do the register strip tests:
  strips are include/rdna3's, and there the norms' strip layout runs the
  tile kernels and is held to their tolerances.

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
  `HK_ARCH=gfx1100` unless one is set, and skips the tier, saying so, when
  the GPU is another arch; set `HK_ARCH` to the GPU's to run it there.
  `test_flydsl_gemm.py` also needs FlyDSL (`FLYDSL_ROOT`), and says so when
  it skips.

  On gfx950 every kernel the tier runs has a native path, so a failure is a
  failure; until [chapter 39](history/39-cdna-paged.md) a rule in
  `conftest.py` reported the kernels that had none (paged attention, the
  4-bit GEMM, two RDNA-only tests) as skips. What a target lacks by design
  -- FlyDSL's RDNA3 GEMM on gfx950, the strips on gfx942 -- is a skip
  condition in the test itself.
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

## The agents' tools

`headroom`, `ab --json` and `new-op` ([agents](agents.md)) are tested in the
IR tier, without a GPU or torch:
- `tests/hk/ir/test_headroom.py`: kernel names from vLLM 0.31's traces (eager
  and graphs), hk's and Inductor's land in their families; hk's and FlyDSL's
  kernels are told from the engine's; the rank is the upper bound times a
  ceiling, a ratio from another arm comes before the routing table's, the
  record's ratio is used only against the kernel it measured, and with no
  ratio the roofline is the ceiling; a graph trace without its replays is
  blind; the CLI ranks a trace from elsewhere.
- `tests/hk/ir/test_ab_verdicts.py`: win, loss, tie and noise from rounds
  with known spreads; rounds pair by number when one failed; every row has
  the same keys; `--from` judges a raw file again.
- `tests/hk/ir/test_new_op.py`: the scaffold writes the op and its test, the
  kernel traces and translates, the generated no-GPU tests pass, and the
  scoreboard's `--cases` finds the row (`tools/hk-bench/_cases.py`).

On the GPU, `tests/hk/gpu/test_new_op.py` generates an op and runs its
`check()`: the exact answer bit for bit, the reference at every shape, and a
call the route declines running the reference. The scoreboard takes the
generated file as it is: `scoreboard.py --cases DIR/NAME.py`.

## CI

`.github/workflows/ci.yml` runs the two tiers that need no GPU on every push
and pull request, on GitHub's `ubuntu-22.04` runner:
- **MLIR is prebuilt.** It downloads MLIR 23.1.3, installed by
  `scripts/build_mlir.sh` for that runner's CPython 3.10, from the
  `ci-mlir-23.1.3` release (0.9 GB). Building LLVM on the runner would take
  hours.
- **The dialect is built.** `scripts/build_hk_mlir.sh` builds it against
  that MLIR in about 20 seconds.
- **The IR tier runs** (`tests/hk/ir`), with `tests/hk/engines`, which
  skips there: the runner has neither vLLM nor SGLang.
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

### The GPU tier

`HK_GPU=0 scripts/gpu_suite.sh OUTDIR` runs `tests/hk/gpu` in groups, three
at a time on one GPU, and exits non-zero if a group failed. It runs on main
on two GPUs, and each commit it tests gets a status on GitHub:
- **`gpu/gfx1100`**: GPU 0 of the W7900D dev pod.
- **`gpu/gfx950`**: one MI355X of the workstation, whichever is idle when the
  run starts, in `rocm/pytorch`.

Its "Details" link opens the run's report: each group's result, the failing
tests and the tail of each failed group's log. Reports are assets of the
[`gpu-ci-logs`](https://github.com/vivienfanghuagood/hk-dsl/releases/tag/gpu-ci-logs)
prerelease.

It is not a GitHub Actions workflow. The cluster reaches github.com and
api.github.com only through the `gh-test` and `gh-api-test.anruicloud.com`
proxies, and not GitHub's Actions service, so no self-hosted runner can sit
next to the W7900D. So it is driven from the workstation, which can reach
GitHub (`scripts/gpu_ci/`):
- **`orchestrate.py`** checks every 10 minutes. It tests main's head when it
  changes, and again nightly. It starts the gfx1100 run in the pod over
  `ssh 0004` and `kubectl exec`, starts the gfx950 run locally, and posts
  each status when its run finishes. It runs twice over: from cron, and as
  `orchestrate.py daemon`, started once from an SSH login. Only the daemon
  can start gfx950: the workstation grants the docker, render and video
  groups through `pam_group`, to login sessions only, so a cron job has none
  of them. Until the daemon starts a commit's gfx950 run, the status says it
  is waiting for the launcher. After a reboot, start the daemon again.
- **`pod_run.sh`** runs in the pod. It fetches the commit itself, through the
  proxy, then builds the dialect and runs the suite. No code from the
  repository runs on the jump host.
- **`local_run.sh`** and **`container_run.sh`** run the gfx950 tier in a
  container that sees one GPU and CI's own directories, not `$HOME`.

Commits pushed together are tested together, as one head. A commit that
changes only documentation carries its parent's passing result. With a warm
compile cache a gfx1100 run takes a minute or two; a change to the dialect's
version recompiles every kernel and takes about half an hour.

```bash
python3 ~/hk-ci/orchestrate.py status          # what each tier last did
python3 ~/hk-ci/orchestrate.py run REF         # test a branch or commit now
python3 scripts/gpu_ci/orchestrate.py install  # copy the scripts to ~/hk-ci, add the cron entries
python3 ~/hk-ci/orchestrate.py daemon          # from an SSH login: the gfx950 launcher
```

A gfx1100 run holds `/workspace/ci/gpu0.lock` in the pod while it uses
GPU 0. A manual benchmark on GPU 0 should hold it too
(`flock /workspace/ci/gpu0.lock <command>`), so that neither times the
other.

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

**Which rounds count** (`tools/hk-bench/_rounds.py`). The scoreboard times
each arm in 7 rounds, each a profiler session of its own whose kernel
records it sums. It used to keep each arm's fastest round, and so kept a
round that could not have happened: hk's 8192x4096 RMSNorm at 97.4 us, 1.38
TB/s for the 134 MB the call moves through 864 GB/s of GDDR6, when every
other round of seven runs took 173 us. Now a round is rejected, and listed
with its reason in the run's log and `--json`, by any of three rules:
- **records**: a round whose kernel-record count differs from the arm's
  usual one. A session that lost records sums fewer calls than it divides
  by.
- **ceiling**: a round faster than the card can go at the case's footprint
  (the distinct bytes the call reads and writes). Up to the 96 MB Infinity
  Cache, back-to-back calls find their data resident, and the ceiling is the
  fastest cache-resident rate measured on this card, 2.43 TB/s (a two-input
  add at 80-96 MB, `scoreboard.py --probe-ceilings`; a copy reaches 1.77,
  hk's 2048-row RMSNorm 1.82). Past the cache it is GDDR6's 864 GB/s
  (a copy reaches 0.72-0.75, the 8192-row norms 0.78). GEMM and attention
  FLOPs are held to the bf16 peak, 122.6 TFLOP/s. Each ceiling carries 15%
  of headroom.
- **lone**: of the rounds left, the fastest when it is more than the larger
  of 5% and 4x the other rounds' spread (their median over the second best)
  ahead of the second best. Repeated while more than half the rounds remain.

**Each round's calls are queued.** A GPU spin of 1 ms a call goes first in
each session, so the host has launched every call before the first runs
and they run back to back, as in a captured graph, whatever the launcher
costs. Without it a microsecond kernel's time was its launcher's: a call
that reached the GPU 5-16 us after it went idle took about 10 us longer,
10-20% of the calls of the arms that launch in 5-8 us (hk's pybind entry,
torch, FlyDSL) and none of Triton's or Inductor's, which take 21-28 us. That,
and not hk's kernels, was the one- and 32-row cases' noise
([chapter 32](history/32-measuring-the-measurements.md)).

An arm's time is the **median** of the rounds it kept, and its noise their
median absolute deviation; the best kept round is reported beside it. On
four runs of the gate's arguments the median and the best round moved the
41 gated-size pairs' ratios alike between runs (median spread 0.36% and
0.45%, 90th percentile 3.7% and 3.5%), and the 16 small-row pairs less (90th
percentile 19% against 28%). The median cannot be set by one round.

Run on the reports kept from [chapter 31](history/31-remeasured.md) (six
baseline runs, the run it left out, four gate runs, six norm A/B runs:
`check_regression.py --audit`), the ceiling rejects exactly one round, the
97.4 us one. Those reports keep only each arm's best round and the second's
distance from it, so `lone` can only be named there as a candidate: 11
best rounds, all at 1 and 32 rows, sat 22-42% ahead of their second. On
the reports that keep every round, the rules rejected 2 of 2,072 rounds (296
arm-cases) in four clean runs, both `lone`, both at one or 32 rows.

`check_regression.py` holds reports to the same rules. A time in a report
that is still faster than the card can go (one written before the
scoreboard screened its rounds) is left out of a refresh, for that run and
pair only, and named in the baseline's `left_out`; in a check it is
`IMPLAUSIBLE` and the run cannot pass (exit 2). A run whose times are
another statistic than the baseline's (median of kept rounds, or best
round) is refused: every ratio would move.

**Where the tolerance comes from.** `tests/perf/baselines/gfx1100.json` is
built from several scoreboard runs, each its own process, spread over time.
For each (case, arm) the ratio is the median across runs and the noise is
the larger of two spreads: across runs, max |r_i / median - 1|; within a
run, the two arms' spreads (the median absolute deviation of their kept
rounds) summed, at their worst. Then
tolerance = max(3%, 3 x noise). A pair whose tolerance would pass 15% is
not gated, and is listed as such with its noise. The file records every
run's ratio beside the tolerance it produced. To change a tolerance,
re-measure the noise; editing the number by hand loosens the gate blindly.

**What the shipped baseline measured.** hk with
[chapter 32](history/32-measuring-the-measurements.md)'s scoreboard: queued
rounds, timed by the median of the rounds kept. Ten runs on GPU 3 of the
engines' pod (vLLM 0.31 venv: torch 2.13.0+rocm7.2, Triton 3.8.0, FlyDSL
0.4.0): six on `28ffe21` with this work, three of them half an hour after the
others, and four four hours later on `47b40aa`, after
[chapter 33](history/33-cdna-gemm.md). All ten built the same kernels (one
`kernels` hash) and none of them had a round rejected:
- 57 gated pairs over 17 cases, and none left out for noise. The one- and
  32-row RMSNorm and fused add are gated again: queued, hk's kernels there
  take 1.95-2.33 us and every arm's rounds sit within about 1%, where
  before the launch-timing artifact moved their ratios 18-52% between runs.
- 37 pairs sit at the 3% floor. GEMM tolerances are 3.0-4.4%, attention's
  3.3-7.3%, and the widest is 12.8% (torch against the one-row RMSNorm,
  whose ratio moved 4.3% across the runs).
- The 40 pairs both this and the previous baseline gate moved by a median
  0.68% (at most 5.1%, Inductor against the 8192-row softmax), from the
  statistic and the queue; their widest tolerance went from 13.7% to 7.3%.
- With the norms on the tile kernels instead (`HK_NORM_LAYOUT=tile`, a
  regression on purpose: chapter 27 measured the strips faster everywhere
  but the 8192-row fused add), the check failed with exit 1 on 35 pairs:
  the one- and 32-row norms by 15-26%, the 2048-row ones by 10-27%, the
  8192-row RMSNorm by 9-12% and three arms of the 8192-row softmax by
  5.8-7.0%. The GEMM, attention and the 8192-row fused add moved -1.5% to
  +0.8%, and passed. The check also printed that hk's kernels had changed.
- `scripts/perf_gate.sh` on `47b40aa` with this baseline, a run after the
  ten: exit 0, all 57 pairs within tolerance, no note, no round rejected;
  the largest change used 31% of its pair's tolerance (Inductor against the
  8192-row softmax, -2.2% of 7.2%).

The previous baseline, measured on hk as of
[chapter 26](history/26-the-scoreboard-retuned.md), was the one tested for
firing. With hk's GEMM made slower on purpose (`k_step` 32 instead of 64 in
its default schedule, 9-12% a call), the gate failed with exit 1 on all 14
GEMM pairs (-11.6% to -15.2% against tolerances of 3.0-4.6%) and on nothing
else; reverted, it passed. The GEMM tolerances are now 3.0-4.8%. Run
against that baseline, this tree came out with no regression and 33 pairs
`better`: 11 GEMM pairs by 3.5-5.0%, and 22 norm and softmax pairs by
6.6-35%.

**Refreshing the baseline.** Refresh it when the opponent changed (a new
torch, Triton or FlyDSL), when hk's kernels changed, when hk got faster on
purpose (pairs that beat their tolerance are reported as `better`), or for
another arch. The check prints a note for the first two: the baseline and
each run record torch's, Triton's and FlyDSL's versions, the device, a hash
of the generated C++ and hipcc flags of every hk kernel the run built or
loaded (`kernels`, `hk.runtime.compile.built_fingerprint`), and hipcc's
version (`hipcc`). hk's own version, the dialect's and `toolchain` (hipcc
with every header's contents) are recorded and not compared: 0.4.0 bumped
the first two and changed no kernel, and on it every gate run printed that
a 0.3.1 baseline was due a refresh; chapter 33's CDNA shim moved the third
and left every RDNA kernel's C++ as it was. A HipKittens header change that
leaves the generated C++ alone is the one change the note does not see.

```bash
HK_GPU=3 FLYDSL_ROOT=/workspace/FlyDSL scripts/perf_gate.sh --refresh 4
git diff tests/perf/baselines/gfx1100.json      # every moved ratio is a claim
```

`--refresh N` runs the scoreboard N times (at least three) and rewrites the
baseline from them. Back-to-back runs see less drift than runs hours apart;
for a baseline that has to last, run `check_regression.py --refresh` on
reports from separate sessions. Read the runs before using them
(`check_regression.py --audit` lists every rejected round): a round faster
than the hardware can go is now rejected by the scoreboard and a time left
out by the refresh, but a `spread` far above its neighbours' is still the
measurement and not the kernel, and it would widen every tolerance of its
case. A refresh is a commit, with the reason in its message, and with any
run left out named there.

## The gates for a change to the generated code

The move to MLIR was gated three times, each against the code it replaced,
and the same gates are the bar for any later change to the generated C++ ("it
needs Gate B again, resources first"). What each gate checks, with what it
found at the move; the full account is in
[the MLIR chapter](history/15-mlir-dialect.md):

- **Gate A — the C++ against the code it replaces.** Every shipped
  instantiation is rendered both ways and compared. At the move, all 354
  (708 translations with the tests' constexpr points) were byte-identical,
  and the whole no-GPU suite ran as the corpus, the error paths included:
  the new passes had to refuse what the old ones refused, in the same words.
- **Gate B — each kernel family against its old self.** Every instantiation
  is rendered both ways; those that differ beyond names are compiled, and
  VGPRs, SGPRs, scratch, LDS and occupancy must match. Then the GPU tier, and
  old and new timed in one process, interleaved, the old as control. At the
  move: 422 instantiations, 92 differing beyond names and all identical in
  resources; every timing gap inside its row's own round-to-round spread
  ([performance](performance.md#after-the-mlir-move)).
- **Gate C — the engine.** vLLM serving Qwen3-8B with the patch, from the
  tree before and after, each against stock vLLM: the routing counters must
  match call for call, and the greedy output must match stock's. Run it with
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Without it the patch installed in the
  driver never reaches the model, and both arms time as stock; the tool
  prints the routing counters to catch exactly that.

Gate B's resource comparison needs only hipcc, and it is the part to run
first: a change that spills is a wrong kernel, not a slow one
([resource gates](runtime.md#the-resource-gates)).

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
