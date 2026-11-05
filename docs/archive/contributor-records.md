# Contributor records (to 2026-11)

> **A dated record.** These paragraphs used to sit in the contributor pages
> ([testing](../testing.md), [building](../building.md), [releasing](../releasing.md)).
> They record what was measured or went wrong at the time, with the versions and
> numbers of the time. The pages themselves say how to do each thing today.

## Test counts at chapter 28

From [testing](../testing.md#the-tiers).

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
  W7900D and an MI355X ([the GPU tier](../testing.md#the-gpu-tier)). On
  gfx950 the attention and RDNA GEMM tests skip
  ([chapter 29](../history/29-cdna.md)), and so do the register strip tests:
  strips are include/rdna3's, and there the norms' strip layout runs the
  tile kernels and is held to their tolerances.

[The MLIR chapter](../history/15-mlir-dialect.md) records the counts as they
stood at the move (990 no-GPU tests, 386 on the GPU).

## gfx950 skips before chapter 39

Until [chapter 39](../history/39-cdna-paged.md) a rule in `conftest.py`
reported the gfx950 kernels that had no native path (paged attention, the
4-bit GEMM, two RDNA-only tests) as skips.

## CI compile cache: why ten shards

From [testing](../testing.md#ci).

- Cold, six shards took 8-20 minutes a job (2.5 of setup, 2 of IR tier on the first);
  hence ten, with the IR tier apart.
- Before the cache key was cut to the headers a compile reads, every header was in
  every key, and a branch touching the CDNA shim compiled the whole tier cold on one 4-core runner,
  40-56 minutes against `main`'s warm 5 (PRs #28-#31).

## The GPU tier before GitHub Actions

Until 19 October 2026 the tier was a polling CI on the same machines
(`scripts/gpu_ci/`), posting `gpu/gfx1100` and `gpu/gfx950` commit statuses;
older commits carry those (the prerelease that held their reports has since
been removed). [Chapter 40](../history/40-gpu-ci-on-actions.md) records the switch.

## The gfx1201 runner and its baseline

From [testing](../testing.md#the-gfx1201-runner).

The `gfx1201` job ran `continue-on-error` until the port's GPU tier was green
there (RDNA4 R1, #71).

The gfx1201 performance baseline's eight runs on R9700s, in two sessions hours apart,
gate 34 pairs at 3.0-12.6% (median 4.5%). Not gated: the 1- and 32-row norms,
and the 8192-row fused add + RMSNorm, whose hk time is 445 us in some
processes and 481 in others (the same kernel source, the opponents unchanged);
a baseline from one session had gated it at 4.5%, and the later session's runs all
failed it.

The performance gate is not part of the gfx1201 job. Run there, after the GPU tier, it flagged
three pairs of an unchanged hk 3.3-6.2% slower (bf16 GEMM against Triton, the
fp8 GEMM at 2048 rows against hipBLASLt). On a card the tier has just kept busy
for a quarter of an hour, the ratios of hk's large GEMMs moved further than the
standalone baseline allows, and a required job cannot fail on a measurement
condition.

## The scoreboard's rounds: what the rules found

From [testing](../testing.md#the-performance-gate).

Before each round's calls were queued behind a GPU spin, a microsecond kernel's
time was its launcher's: a call
that reached the GPU 5-16 us after it went idle took about 10 us longer,
10-20% of the calls of the arms that launch in 5-8 us (hk's pybind entry,
torch, FlyDSL) and none of Triton's or Inductor's, which take 21-28 us. That,
and not hk's kernels, was the one- and 32-row cases' noise
([chapter 32](../history/32-measuring-the-measurements.md)).

On four runs of the gate's arguments the median and the best round moved the
41 gated-size pairs' ratios alike between runs (median spread 0.36% and
0.45%, 90th percentile 3.7% and 3.5%), and the 16 small-row pairs less (90th
percentile 19% against 28%).

Run on the reports kept from [chapter 31](../history/31-remeasured.md) (six
baseline runs, the run it left out, four gate runs, six norm A/B runs:
`check_regression.py --audit`), the ceiling rejects exactly one round, the
97.4 us one. Those reports keep only each arm's best round and the second's
distance from it, so `lone` can only be named there as a candidate: 11
best rounds, all at 1 and 32 rows, sat 22-42% ahead of their second. On
the reports that keep every round, the rules rejected 2 of 2,072 rounds (296
arm-cases) in four clean runs, both `lone`, both at one or 32 rows.

## The gfx1100 performance baselines

From [testing](../testing.md#the-performance-gate).

**What the first queued baseline measured.** hk with
[chapter 32](../history/32-measuring-the-measurements.md)'s scoreboard: queued
rounds, timed by the median of the rounds kept. Ten runs on one W7900D, in the
engines' container (vLLM 0.31 venv: torch 2.13.0+rocm7.2, Triton 3.8.0, FlyDSL
0.4.0): six on `28ffe21` with this work, three of them half an hour after the
others, and four four hours later on `47b40aa`, after
[chapter 33](../history/33-cdna-gemm.md). All ten built the same kernels (one
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

[Chapter 38](../history/38-rdna-losses.md) refreshed it from four
back-to-back runs on one W7900D, when RDNA's stores began rounding to even and
V's transpose was packed (a new `kernels` hash, Triton 3.8.0 in the venv):
57 gated pairs, tolerances 3.0-6.6%, the median pair moved 0.0%, attention
against torch +6-10%, and a gate run after it passed.

**On the matched stack (2026-10-23).** Every baseline up to here was
measured in a venv with PyPI's Triton 3.8.0 installed over the triton-rocm
3.7.1 that torch 2.13.0+rocm7.2 pins ([chapter 44](../history/44-matched-stack.md)).
That venv is retired, and vLLM 0.31's venv is now the matched one
(torch 2.13.0+rocm7.2, triton-rocm 3.7.1, FlyDSL 0.4.0;
[integration](../integration.md#supported-versions-and-the-contract-tests)).
The shipped baseline was refreshed there: eight runs on one W7900D, in two
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
  towards 1, which a ratio does not cancel. The GPU was idle and the cause
  was not found. A plain scoreboard run 25 minutes later read the
  usual times, and the gate run after it exited 0: all 57 pairs within
  tolerance, the largest drop -0.8% (of 3.0-4.0%). If the gate fails on
  the small cases alone, look at their absolute times before believing it.
- **The gate now catches the drift.** A run in the retired venv, checked
  against the new baseline, exits 1 with five regressions: Triton against
  the one- and 32-row fused add (-9.6%, -6.2%), the 2048-row fused add and
  two GEMMs. It also prints the note that Triton changed from 3.7.1 to
  3.8.0.

The previous baseline, measured on hk as of
[chapter 26](../history/26-the-scoreboard-retuned.md), was the one tested for
firing. With hk's GEMM made slower on purpose (`k_step` 32 instead of 64 in
its default schedule, 9-12% a call), the gate failed with exit 1 on all 14
GEMM pairs (-11.6% to -15.2% against tolerances of 3.0-4.6%) and on nothing
else; reverted, it passed. The GEMM tolerances are now 3.0-4.8%. Run
against that baseline, this tree came out with no regression and 33 pairs
`better`: 11 GEMM pairs by 3.5-5.0%, and 22 norm and softmax pairs by
6.6-35%.

Why hk's own version, the dialect's and `toolchain` (hipcc with every header's
contents) are recorded and not compared: 0.4.0 bumped
the first two and changed no kernel, and on it every gate run printed that
a 0.3.1 baseline was due a refresh; chapter 33's CDNA shim moved the third
and left every RDNA kernel's C++ as it was.

## Gates A and B at the MLIR move

From [testing](../testing.md#the-gates-for-a-change-to-the-generated-code).

- Gate A: all 354 shipped instantiations (708 translations with the tests'
  constexpr points) were byte-identical, and the no-GPU suite's error paths
  refused in the same words.
- Gate B: 422 instantiations, 92 differing beyond names and all identical in
  resources; every timing gap inside its row's own round-to-round spread
  ([performance](performance-2026-10.md#after-the-mlir-move)).

## Wheels and glibc, 0.3.1 to 0.8.0

From [building](../building.md#wheels).

0.3.1 through 0.7.0 built cp312 in rocm/pytorch on Ubuntu 24.04: `libHKPythonCAPI`
needed `GLIBC_2.38` (`__isoc23_strtol`, which glibc 2.38's headers substitute
for `strtol`), so the wheel was `manylinux_2_38`, and `pip install hk-dsl`
found no cp312 wheel in the vLLM ROCm images, which have glibc 2.35. 0.7.0 got
an extra cp312 `manylinux_2_35` wheel, built in the vLLM image, and from 0.8.0
it is the only cp312 wheel; pip on a newer glibc takes it too.

On 0.7.0's wheels, the cp310 `manylinux_2_35` and both cp312
wheels (`manylinux_2_38` at `GLIBC_2.38`, `manylinux_2_35` at `GLIBC_2.34`)
pass; the 2_38 wheel renamed to 2_35 fails.

Installed into a fresh venv, the retagged cp310 wheel pulls its
three dependencies from PyPI, traces, translates and builds outside the
checkout.

## PyPI uploads

0.3.1, the first version on PyPI, through 0.7.0 were uploaded by hand with
twine from the same manylinux wheels.

`.github/workflows/pypi.yml`, run for v0.6.0 (run 40186344465), downloaded the release's wheels and then PyPI refused the
exchange: "no corresponding publisher (Publisher with matching claims was not
found)". The claims it presented were repository `vivienfanghuagood/hk-dsl`,
workflow `.github/workflows/pypi.yml`, environment `pypi`. Releases since have gone up with twine
([building](../building.md#wheels) says what fixes it).

## Reproducible wheels: what 0.7.0 got wrong

From [building](../building.md#reproducible-wheels).

0.7.0's rc and tag wheels, built from the same tree, were not byte-identical: the cp310 build ran in a `mktemp -d`
directory, whose name ended up in two `__FILE__` strings in
`libHKPythonCAPI.so` and in the GNU build-id of the three libraries linked
against it. The zip layer differed too: every entry's timestamp was the
build time.

With the bindings linked with the build tree's absolute `_mlir_libs` path
instead of `$ORIGIN`, the three libraries that link `libHKPythonCAPI`
  were patched by `cmake --install`, in place: `.dynstr` kept the old string's
length, so the layout followed the length of the build directory's path, and
the build-id, hashed at link time, kept the old path itself. That, not code,
was the "only the build-id differs" of 0.7.0.
