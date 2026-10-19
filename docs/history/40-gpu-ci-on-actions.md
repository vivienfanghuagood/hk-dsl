# The GPU tier on GitHub Actions, and the finding it had been built around

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

On 16 October the GPU tier went onto main as a polling CI
(`scripts/gpu_ci/`): cron and a daemon on the MI355X workstation tested
main's head every ten minutes, on a W7900D of the dev pod and on an MI355X,
and posted `gpu/gfx1100` and `gpu/gfx950` commit statuses. It was built that
way because of one finding, written into `orchestrate.py`'s docstring and
[testing](../testing.md): the cluster reaches github.com and api.github.com
only through the gh-test and gh-api-test proxies, "and not GitHub's Actions
service, so no self-hosted runner can sit next to the W7900D". That was
wrong. This chapter records the measurement that replaced it, the move to two
self-hosted Actions runners, the burn-in with both CIs side by side, and what
was retired.

## What the cluster reaches

Measured on 19 October, connecting to each host from each place:

| from | api.github.com | the Actions service | github.com |
|---|---|---|---|
| the workstation | yes | yes | yes |
| jump host 0004; pod hfang-hk-vllm (DNS 8.8.8.8) | 10/10 | 10/10 | 9/10, `git ls-remote` works |
| pod hfang-hk-rdna3-b (cluster DNS 10.232.0.10) | by IP, 5/5 | by IP, 5/5 | no, from any of five IPs |

The Actions service is what a runner talks to: the broker, pipelines and
results-receiver hosts under `*.actions.githubusercontent.com`, and
`blob.core.windows.net`, where logs and artifacts go. hfang-hk-rdna3-b has
two problems, and neither is the Actions service: its cluster DNS resolves no
GitHub name, and github.com itself is blocked by name. Everything else
answers there by IP.

The first conclusion was drawn from github.com alone. From rdna3-b,
github.com fails at every address and no GitHub name resolves, so the
cluster looked cut off from GitHub but for the proxies. The Actions
service's hosts were never tried, from that pod or from anywhere else in the
cluster. gh-api-test.anruicloud.com, named in the finding, is a reverse proxy
of api.github.com only; nothing needs it. gh-test is still how rdna3-b
fetches a commit.

## Two runners, on the workstation

The tier is now `.github/workflows/gpu.yml`, on two ephemeral self-hosted
runners, `smci355-gfx950` and `smci355-gfx1100` (labels `self-hosted,
hk-gpu, gfx950` and `self-hosted, hk-gpu, gfx1100`), each kept registered by
`~/gha/supervise.sh`: one job, then registered again. It runs on pushes to
main and by hand, never on pull requests: the repository is public, and a
fork's code must not reach these machines. Its jobs, `gfx950` and `gfx1100`,
call `scripts/ci/` from the checkout:
- `run_gfx950.sh` runs the commit on the lowest-numbered idle MI355X, in
  `rocm/pytorch`, through `container_run.sh` (the polling CI's, moved), with
  a build directory, pip packages and compile cache of its own under
  `~/gha/x950/ci` and the MLIR install mounted read-only.
- `run_gfx1100.sh` and `pod_run.sh` relay to the pod as `orchestrate.py`
  did, over `ssh 0004` and `kubectl exec`, the pod fetching the commit
  through gh-test, into `/workspace/gha`. A run takes an idle one of GPUs
  5-7 under a lock of its own, else GPU 0 under `/workspace/ci/gpu0.lock`.
- `summarize.py` (moved) writes `result.json`; `report.py` turns it into an
  annotation with the counts, the job summary and the exit status.

Both runners live on the workstation, though a runner could now live in the
cluster: the workstation holds the MI355X and reaches everything, and the
gfx1100 relay had run for three days. A runner on 0004 or in hfang-hk-vllm
would work as measured; in rdna3-b it would need GitHub's names resolved
some other way.

Against the polling CI:
- every push to main is a run; the polling CI tested main's head once a
  tick. A push that waits behind a run in progress replaces one that was
  waiting.
- a documentation-only commit is run, not credited with its parent's
  result: with a warm compile cache the suite takes a minute or two.
- no nightly rerun; `gh workflow run gpu.yml --ref main` reruns by hand.
- gfx1100 tries GPUs 5-7 before GPU 0, which the polling CI used alone.
- the result is a check run, with the report as an artifact, instead of a
  status linking to an asset of the `gpu-ci-logs` prerelease.
- `gfx950` and `gfx1100` are main's required checks, without enforcement on
  admins: every push to main so far is the owner's, and goes through as a
  recorded bypass. What that does and does not enforce is in
  [testing](../testing.md#the-gpu-tier).

## Burn-in

Both CIs ran side by side from the first push of the workflow until the
polling CI was stopped. Counts are failed / passed / skipped; "by hand" is
`gh workflow run` on a branch at the commit and `orchestrate.py run` on its
SHA, for a commit of a batch whose push tested only its head.

| commit | how | gfx950, Actions | gfx950, polling | gfx1100, Actions | gfx1100, polling |
|---|---|---|---|---|---|
| `9f953b0`, the workflow | push | success 0/729/121 | success 0/729/121 | success 0/659/95 | success 0/659/95 |
| `e2f5a75`, chapter 39's head | push | success 0/784/66 | success 0/784/66 | success 0/659/95 | success 0/659/95 |
| `c669558`, inside chapter 39's batch | by hand | success 0/781/69 | success 0/781/69 | success 0/659/95 | success 0/659/95 |

Each pair agrees in state and in every count, and both followed the 850
gfx950 tests as chapter 39's batch turned skips into passes: 52 by `c669558`
(paged attention and the 4-bit GEMM), three more by its head. Before main,
the trial branch had run `~/gha`'s scripts on `95cd1d0`'s tree, gfx950
failure 5/718/121 and gfx1100 0/659/89, and the repository's on `0bc9bbb`'s,
0/729/121 and 0/659/95: the polling CI's results on those commits exactly,
the five failures the same tests. A second run of `9f953b0` by hand repeated
its counts. With warm caches a job took under two minutes; a commit that
recompiled the kernels took 23-24 minutes on gfx950 and 31-32 on gfx1100, in
either CI.

## Retired

- The two cron entries (`# hk-dsl gpu ci`) are removed and the
  `orchestrate.py daemon` is stopped, between ticks and with no run in
  progress; nothing posts `gpu/*` statuses any more. The commits it tested
  keep theirs, and the `gpu-ci-logs` prerelease keeps the reports they
  link to.
- `scripts/gpu_ci/` is gone from the repository. `container_run.sh` and
  `summarize.py` live on in `scripts/ci/`; `orchestrate.py`, `local_run.sh`
  and `pod_run.sh` are replaced by the jobs and `scripts/ci/pod_run.sh`.
- `~/hk-ci` is left in place, inert, with a note saying so. One part is still
  used: `mlir/install-cp310`, the MLIR install the gfx950 job mounts. The
  rest, about 2.5 GB (the polling CI's build, pip packages, compile cache and
  checkout, its mirror, runs, state, logs and scripts), can be deleted. In
  the pod, `/workspace/ci/gpu0.lock` is still GPU 0's lock; the rest of
  `/workspace/ci`, about 1.2 GB, is inert too.
