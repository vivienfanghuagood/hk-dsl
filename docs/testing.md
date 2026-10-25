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
at a time on one GPU, and exits non-zero if a group failed. On every push to
main it runs on two GPUs, as `.github/workflows/gpu.yml`, whose two jobs are
main's required checks:
- **`gfx950`**: one MI355X of the workstation, the lowest-numbered one that
  is idle when the job starts, in `rocm/pytorch`.
- **`gfx1100`**: the four W7900D of the CI pod `hfang-hk-ci-0044` on node
  wx-ms-w7900d-0044 (`scripts/ci/k8s/gfx1100-ci-0044.yaml`), the suite's
  groups sharded over them. The pod requests `amd.com/gpu: 4` and is not
  privileged: the device plugin hands it those four cards alone, so it needs
  no node lock. Fallback: the dev pod `hfang-hk-rdna3-b`'s GPU 6 under the
  node lock (the job's `runs-on` back to `gfx1100`, its `env` removed).

Each job has a self-hosted runner of its own on the workstation:
`smci355-gfx950` and `smci355-gfx1100-x4`, labelled `self-hosted, hk-gpu,
gfx950` and `self-hosted, hk-gpu, gfx1100-x4`; `smci355-gfx1100` (`gfx1100`,
the dev pod) stays registered as the fallback. Both are ephemeral: registered
for one job, then registered again. A job's page has its counts as an
annotation and, in its summary, each group's result and the failing tests.
Its artifact (`gfx950-report`, `gfx1100-report`) holds `result.json`,
`report.txt` (with the tail of each failed group's log) and the build log;
gfx950's also every group's log, gfx1100's the pod's driver log.

The workflow runs on pushes to main and by hand
(`gh workflow run gpu.yml --ref BRANCH`), and never on pull requests: the
repository is public, and a fork's code must not reach these machines. Do
not add a `pull_request` or `pull_request_target` trigger. One run per arch
and ref at a time: a push waits for the run in progress, and replaces a push
that was waiting, so commits pushed close together are tested at the last.
With a warm compile cache the suite takes a minute or two; a change that
recompiles the kernels takes about 25 minutes on gfx950, and on gfx1100 32 on
the dev pod's single GPU (see the CI pod's numbers in the pull request that
moved it).

**What the jobs run** (`scripts/ci/`, from the job's checkout):
- **`run_gfx950.sh`** picks the GPU (under 10% use and 3% VRAM; it waits up
  to an hour for one) and starts `rocm/pytorch` with that GPU's render node,
  the checkout, the run's directory, CI's own directory, and the MLIR
  install read-only; nothing else of `$HOME`. In it, **`container_run.sh`**
  builds the dialect for the image's CPython 3.12 against MLIR 23.1.3's cp310
  install (the MLIR C++ libraries do not depend on the interpreter), then
  runs the suite.
- **`run_gfx1100.sh`** works over `ssh 0004` and `kubectl exec`: it installs
  **`pod_run.sh`** and `summarize.py` in the pod and starts the run there,
  detached, then follows its log and brings back its result. The pod
  fetches the commit itself, through the `gh-test.anruicloud.com` proxy: its
  cluster DNS resolves no GitHub name, and github.com is blocked by name, so
  git is pointed at the address the jump host resolves. No code from the
  repository runs on the jump host.
- **`summarize.py`** turns the suite's output into `result.json` and
  `report.txt`; **`report.py`** writes the annotation and the summary, and
  sets the job's exit status.

**GPUs and locks.** The MI355X are shared with whoever else is measuring on
the workstation; the idle test is the only claim, as it is for the CDNA
work. In the pod, a run takes GPU 6 when it is idle, and otherwise GPU 0,
each under the node-level lock of the physical card,
`/workspace/locks/gpu/<PCI address>.lock` (0000:c3:00.0 and 0000:03:00.0);
it never touches GPU 3, where the performance gate runs. The lock directory
is one directory on the node that every pod seeing these cards mounts at
`/workspace/locks` -- every pod on the node is privileged and sees all eight
GPUs, so a lock inside one pod is invisible to the others -- and a run
without it refuses to start. A manual benchmark takes the same lock
(`/workspace/locks/bin/gpulock IDX -- <command>`), so that neither times the
other. The minutes in a result count from when the GPU was taken.

**Required checks.** main's branch protection requires `gfx950` and
`gfx1100` from GitHub Actions, not up to date with main (`strict: false`),
without enforcing it on admins, and nothing else: no reviews, no other
rule. The workflow runs on main's pushes and by hand, never for a pull
request, so:
- a pull request can be merged once its head commit has passed both checks,
  and it gets them only if someone with write access runs the workflow by
  hand on a branch here at that commit. For a fork's pull request that would
  put the fork's code on these machines: do not. In practice only an admin,
  bypassing the rule, merges one;
- a non-admin's push to main is refused unless its commit has already
  passed both, the same way;
- the owner's pushes, the agents' among them, go through, and GitHub
  records each as a bypass. For them the checks are a record on each commit
  of main, after the fact, not a gate in front of it.

**The CI pod.** `kubectl apply -f scripts/ci/k8s/gfx1100-ci-0044.yaml` (from
the jump host) makes it and its 200 GiB node-local PVC, mounted at
`/workspace`. Once, the PVC is given what the image lacks, copied from the
dev pod: `/workspace/llvm/install-cp312` (the MLIR install the dialect builds
against), `/workspace/FlyDSL`, and in `/workspace/pyextra` the pip packages
flydsl, nanobind, lit and cmake (`pod_run.sh` puts it on `PYTHONPATH` and
`PATH`). Everything else -- clone, dialect build, compile cache, runs -- is
`/workspace/gha`, as on the dev pod.

**Where things live.** Host state is kept out of the checkout:
- `~/gha`: the runners (`runner-gfx950`, `runner-gfx1100`), `supervise.sh`,
  which keeps each registered, and its logs (`logs/gfx950.log`,
  `logs/gfx1100.log`; the runners' own are in `runner-*/_diag`);
- `~/gha/runs/gfx950/RUN_ID-ATTEMPT`: a gfx950 run's directory;
- `~/gha/x950/ci`: `/ci` in the container, with the dialect's build
  (`hk-build-cp312`), pip's packages (`pylib`) and the compile cache
  (`hk-cache`);
- `~/hk-ci/mlir/install-cp310`: the MLIR install, the `ci-mlir-23.1.3`
  release's asset unpacked, mounted read-only;
- `/workspace/gha` in the pod: its clone, `hk-build-cp312`, `hk-cache` and
  `runs/RUN_ID-ATTEMPT`.

`HK_CI_STATE` moves the gfx950 runs and `x950` (default `~/gha`),
`HK_CI_MLIR` the MLIR install and `HK_CI_IMAGE` the image; `HK_CI_POD`,
`HK_CI_JUMP`, `HK_CI_PROXY`, `HK_CI_POD_DIR` and `HK_CI_POD_GPUS` set the
pod's side. A job inherits its runner's environment, so set them where
`supervise.sh` is started.

**Starting and stopping the runners.** After a reboot, start them from an
SSH login: the workstation grants the docker, render and video groups
through `pam_group`, to login sessions only, and a runner started from cron
or a service could not start the gfx950 container.

```bash
cd ~/gha && rm -f STOP STOP-gfx950 STOP-gfx1100
for n in gfx950 gfx1100; do nohup setsid ./supervise.sh $n >> logs/$n.log 2>&1 < /dev/null & done
gh api repos/vivienfanghuagood/hk-dsl/actions/runners --jq '.runners[] | [.name, .status, .busy] | @tsv'
```

To stop them, `touch ~/gha/STOP` (both) or `~/gha/STOP-gfx950` /
`STOP-gfx1100` (one). Within 10 seconds the listener gets SIGTERM, which
cancels a job in progress, the runner is unregistered and `supervise.sh`
exits. Remove the file before starting again. Stop them this way rather
than by killing processes: `supervise.sh` registers a new runner whenever
its listener exits, and a runner that did not exit cleanly stays
registered, offline. While a runner is down its jobs queue, and GitHub
cancels a job that found no runner in 24 hours.

Until 19 October 2026 the tier was a polling CI on the same machines
(`scripts/gpu_ci/`, cron and a daemon in `~/hk-ci`), posting `gpu/gfx1100`
and `gpu/gfx950` commit statuses; older commits carry those, with reports
in the [`gpu-ci-logs`](https://github.com/vivienfanghuagood/hk-dsl/releases/tag/gpu-ci-logs)
prerelease. [Chapter 40](history/40-gpu-ci-on-actions.md) records the switch.

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

**What the first queued baseline measured.** hk with
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

[Chapter 38](history/38-rdna-losses.md) refreshed it from four
back-to-back runs on GPU 3, when RDNA's stores began rounding to even and
V's transpose was packed (a new `kernels` hash, Triton 3.8.0 in the venv):
57 gated pairs, tolerances 3.0-6.6%, the median pair moved 0.0%, attention
against torch +6-10%, and a gate run after it passed.

**On the matched stack (2026-10-23).** Every baseline up to here was
measured in a venv with PyPI's Triton 3.8.0 installed over the triton-rocm
3.7.1 that torch 2.13.0+rocm7.2 pins ([chapter 44](history/44-matched-stack.md)).
That venv is retired, and `/workspace/venv-vllm031` is now the matched venv
(torch 2.13.0+rocm7.2, triton-rocm 3.7.1, FlyDSL 0.4.0;
[integration](integration.md#supported-versions-and-the-contract-tests)).
The shipped baseline was refreshed there: eight runs on W7900D GPU 0, in two
blocks of four about 50 minutes apart. Each run was interleaved with a run
of the retired venv, on the same hk tree (`0df64ae`, one `kernels` hash),
with the arm order rotated. No round of the matched runs was rejected.
- **The change is the stack, not hk.** The eight retired-venv runs, on
  today's kernels, reproduce the old baseline. 55 of 57 pairs are within
  tolerance, and the median change per arm is 0.3-0.8%. A check of those
  runs against the old baseline exits 0 (with the note that hk's kernels
  changed). The matched runs differ from them in the arms that run Triton
  kernels. Against Triton the median ratio moves 3.2%: +2.3% to +3.6% on the
  GEMMs, +10.9% and +6.3% on the one- and 32-row fused add, -3.5% on the
  2048-row softmax. Against Inductor the median moves 1.2%. torch and
  FlyDSL move by a median 0.1-0.2% (at most 1.2%). In other words,
  triton-rocm 3.7.1's kernels are 2-4% slower than 3.8.0's on these GEMMs,
  6-11% slower on the small fused adds, and 3.5% faster on the 2048-row
  softmax. Checked against the old baseline, a matched run exits 1: a
  REGRESSION at softmax 2048x4096 / triton (-3.2% to -3.3% against 3.0%),
  7-8 pairs `better`, and the note that Triton changed from 3.8.0 to 3.7.1. The
  exception on the retired side is Inductor against the 32-row RMSNorm,
  1.23x against the old 1.04x, under 3.8.0 only: the other arms of that
  case are within 0.5%, so the move is Inductor's kernel (built in a fresh
  cache), not hk's.
- **The new baseline**: 57 gated pairs, none left out. Tolerances are
  3.0-13.7%, median 3.3%, and 25 pairs sit at the floor (the old baseline:
  51 at the floor, 3.0-6.6%, from four back-to-back runs). The runs span an
  hour, so the spread across runs counts for more: it sets the noise of 26
  pairs, as in the first ten-run baseline (37 at the floor, up to 12.8%).
  GEMM tolerances are 3.0-3.9% and attention's 3.0%. The widest are the
  four arms of the 8192-row fused add, 9.3-13.7%. In one of the eight runs,
  hk's kernel there took 0.3317 ms in all seven rounds, against 0.340-0.347
  ms in the other fifteen runs of either venv. That is a per-process mode,
  4% faster, and the gate now counts it as noise. A regression smaller
  than about 9% in that one case would pass.
- **Gate runs after the refresh**, `scripts/perf_gate.sh` in the matched
  venv. The first exited 1 on two 32-row pairs: torch against the RMSNorm
  (-4.6% of 4.0%) and Inductor against the fused add (-3.9% of 3.5%). In
  that run every arm of the one- and 32-row cases took 0.15-0.25 us longer
  a call (hk's 32-row RMSNorm 2.37 us, against 2.09-2.19 us in the eight
  runs), the large cases did not move, and the scoreboard took 130 s
  instead of 68 s. A run of the retired venv right after it was slowed the
  same way. A fixed cost added to every call compresses small cases' ratios
  towards 1, which a ratio does not cancel. GPU 0 was idle and the cause on
  the node was not found. A plain scoreboard run 25 minutes later read the
  usual times, and the gate run after it exited 0: all 57 pairs within
  tolerance, the largest drop -0.8% (of 3.0-4.0%). If the gate fails on
  the small cases alone, look at their absolute times before believing it.
- **The gate now catches the drift.** A run in the retired venv, checked
  against the new baseline, exits 1 with five regressions: Triton against
  the one- and 32-row fused add (-9.6%, -6.2%), the 2048-row fused add and
  two GEMMs. It also prints the note that Triton changed from 3.7.1 to
  3.8.0.

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
python3 -m hk.integration.stack                         # vLLM's venv: the measured stack?
HK_GPU=3 scripts/perf_gate.sh                           # alone on one GPU, in vLLM's venv
scripts/build_wheel.sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 HK_GPU_UTIL=0.85 python3 tools/hk-bench/vllm_e2e.py \
    --context 1024 --batch 1,16 --tokens 72 --short 24 --reps 3 --arms stock,gemm
```
