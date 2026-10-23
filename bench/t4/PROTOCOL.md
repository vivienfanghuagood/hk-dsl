# T4 run protocol (evaluator only)

How the lead runs criterion T4 of [docs/design/tuning-goal.md](../../docs/design/tuning-goal.md):
*a fresh agent given only the docs, hk.check, hk.profile, hk.tune and
hk.schedules takes a hidden variant from first kernel to >= 0.95x the best
baseline within a stated budget, on both GPUs; the run is recorded.*

The variant, the baselines and why are in [DESIGN.md](DESIGN.md). This file and
DESIGN.md are **not** given to the evaluated agent (`t4-make-checkout` deletes
them from its checkout).

## 0. What exists before a run

| what | where |
|---|---|
| the task, exactly as the agent receives it | `bench/t4/task.md` at the task commit |
| the harness (scorer, reference, baselines, public shapes, trivial solution) | `bench/t4/` at the task commit |
| the sealed set: models, master seed, the 14 hidden shapes per GPU | `~/hk-dx/t4/sealed/` on the MI355X workstation (mode 700; never committed, never copied where the agent can read it) |
| evaluator tools | `~/hk-dx/t4/bin/`: `t4-make-checkout`, `t4gpu`, `t4pull`, `t4seal` |

## 1. Set up the run (before the agent starts; none of it is on the agent's budget)

```bash
SHA=<the task commit>                     # the READY line in STATUS.md names it
RUN=$(date -u +%Y%m%d)-a                  # a run id
CK=~/hk-t4run-$RUN                        # the agent's checkout
~/hk-dx/t4/bin/t4-make-checkout $SHA $CK  # no history, no DESIGN/PROTOCOL, HipKittens at its pin
cat > ~/hk-dx/t4/run.conf <<EOF
T4_RUN=$RUN
T4_WT=$CK
T4_GPU950=3
T4_GPU1100=3
T4_BUDGET_H950=4.0
T4_BUDGET_H1100=4.0
EOF
# warm both builds so the agent's GPU time is not spent on them (~3 min; ~20 min for the pod's
# first sync and build):
HKX950V_MAXMEM=30 ~/hk-dx/bin/hkx950v $CK 3 python3 -c "import hk"
HKX1100_GPU=3 ~/hk-dx/bin/hkx1100k $CK python3 -c "import hk"
```

Check the checkout: `ls $CK/bench/t4` shows no DESIGN.md / PROTOCOL.md;
`git -C $CK log --oneline` shows one commit; `grep -rl hk-dx/t4/sealed $CK` is
empty. Pick GPUs that are idle (the MI355X's GPU 3 was E1's; the W7900D's GPU 3
is shared with K4's short checks -- tell K4 the window).

## 2. Launch the agent

- A **new** Claude Code session (record the model id and version), started in
  a directory that has no project memory or CLAUDE.md of this project (e.g.
  an empty `C:\t4run-<RUN>` on the Windows host), with **web tools disabled**
  (deny `WebFetch`, `WebSearch`; no GitHub MCP; `gh` not to be used) and no
  access to other sessions' tools.
- **The prompt** is, verbatim: the contents of `bench/t4/task.md` at `$SHA`,
  then the excerpt below (section 2a) with `$CK`, the GPUs and the budget filled in.
  Nothing else: no project history, no STATUS, no hints.
- Note the start time (UTC). Keep the session's transcript.

### 2a. The excerpt the agent gets after task.md

> ## Where and how to run
>
> - You work from a Windows host (Git Bash). The repository checkout is on a
>   Linux workstation, reached as
>   `ssh -q -o BatchMode=yes -o LogLevel=ERROR smci355-ccs-aus-n06-05.cs-aus.dcgpu '...'`,
>   at `$CK` (a snapshot of hk-dsl: no git history; do not push anything anywhere).
>   Edit files there (or write them on Windows and copy them over, then strip
>   CRs: `sed -i 's/\r$//' FILE` -- files are LF).
> - **GPUs**, only through these two commands on the workstation (they keep
>   your GPU-time ledger and enforce the budget):
>   - `~/hk-dx/t4/bin/t4gpu gfx950 CMD...` runs CMD in your checkout on an
>     MI355X (gfx950), in a container with ROCm, torch 2.12, Triton, aiter and
>     vLLM (the baselines), hk built for your checkout. Paths inside are
>     relative to the checkout.
>   - `~/hk-dx/t4/bin/t4gpu gfx1100 CMD...` syncs your checkout's changed files
>     to a W7900D (gfx1100) pod and runs CMD there (ROCm 7.2.4, torch 2.10,
>     Triton 3.6). Write outputs outside `src/` or under it and copy them back
>     with `~/hk-dx/t4/bin/t4pull PATH...`; `bench/t4/out/` is under src.
>   - e.g. `~/hk-dx/t4/bin/t4gpu gfx950 python3 -m bench.t4.score --solution bench/t4/solution.py --only olmoe`
> - Things that need no GPU (tracing, `--dump`, `kernel.source()`, `build()`,
>   hk.check without timing) run with `~/hk-dx/t4/bin/t4gpu cpu CMD...`: the
>   gfx950 container without a GPU, logged but not counted.
> - **Long commands:** never run a command that can stay silent for more than
>   ~5 minutes in the foreground: start it detached (`nohup ... > log 2>&1 &`)
>   and poll the log. Keep each gfx1100 command to minutes (score a few shapes
>   a call; `python3 -m bench.t4.score --merge A.json B.json` combines runs).
> - The machines are shared with other teams: touch nothing you did not create,
>   do not use other GPUs, keep scratch files inside your checkout.
> - Budget: GPU time is $BUDGET GPU-hours per GPU (the helpers print what is
>   left after every command); wall-clock 10 hours from now.

## 3. What is logged

| what | where |
|---|---|
| every GPU command: start, end, seconds, GPU, exit status, the command | `~/hk-dx/t4/runs/$RUN/ledger.tsv` (written by `t4gpu`) |
| what the tools said: the full output of every GPU command | `~/hk-dx/t4/runs/$RUN/gpu-logs/NNNN-<arch>.log` |
| every tool call, every file read and written, the agent's reasoning as shown | the session transcript (export it at the end: Claude Code's export, or the session's JSONL) |
| what it tried and concluded, in its words | `$CK/bench/t4/NOTES.md` (task.md asks for it) |
| wall-clock | session start/end (UTC), the first and last ledger rows |

GPU-hours per GPU = the sum of the ledger's seconds for that arch / 3600.
Tool calls = the transcript's tool_use count.

## 4. Stopping

- **GPU budget:** `t4gpu` refuses a command once a GPU's ledger reaches the
  budget, and cuts off a running one with `timeout` when it would pass it
  (on gfx1100 the remote process may outlive the cut-off ssh by up to a
  minute; the ledger still stops at the budget).
- **Wall-clock:** at 10 hours after the start the lead stops the session
  (interrupt it, then end it). Whatever `bench/t4/solution.py` is at that
  moment is the submission: copy it (with `sha256sum`) and NOTES.md out of the
  checkout immediately.
- The agent may stop earlier by saying it is done.

## 5. Scoring on the sealed shapes

After the stop, with the agent's session ended (the gfx1100 scoring copies the
sealed shapes into the pod for the duration of the run):

Score the submitted file in a **fresh checkout of the harness**, not in the
agent's: the agent could edit anything in its checkout, `score.py` and
`stats.py` included. The harness commit is the task commit, or a later one
that changes only how faithfully the harness measures (say which in the
record):

```bash
HSHA=<the harness commit>
SC=~/hk-t4score-$RUN
~/hk-dx/t4/bin/t4-make-checkout $HSHA $SC
cp ~/hk-dx/t4/runs/$RUN/submission/solution.py $SC/bench/t4/solution.py
CK=$SC                                   # below, CK is the scoring checkout
~/hk-dx/t4/bin/t4seal gfx950 $CK --solution bench/t4/solution.py --label agent-$RUN-1 \
    --picks /t4sealed/results/triton_picks.json
~/hk-dx/t4/bin/t4seal gfx1100 $CK --solution bench/t4/solution.py --only 0:7 --label agent-$RUN-1a
~/hk-dx/t4/bin/t4seal gfx1100 $CK --solution bench/t4/solution.py --only 7:14 --label agent-$RUN-1b
python3 -m bench.t4.score --merge ~/hk-dx/t4/sealed/results/gfx1100-agent-$RUN-1{a,b}.json \
    --out ~/hk-dx/t4/sealed/results --label agent-$RUN-1
```

(run detached and poll; check the GPUs are idle -- `rocm-smi` busy% and the
pod's `rocm-smi` -- before each). Then **confirm**: repeat both on another
process start (`-2` labels), at least an hour later. Also score the public
shapes once with the same solution (`--shapes public`) for the record.

## 6. What counts as a pass

Per GPU, from `VERDICT` (bench/t4/stats.py `verdict`), on the sealed shapes:

1. **correct** on every sealed shape (every checked routing within 2e-2 of
   fp32, every row written, inputs unchanged), and **audit-clean** (no
   forbidden import; every kernel of an audited call is hk's or torch
   bookkeeping, with no matrix multiply outside hk);
2. the **geometric mean** of the ratios (best baseline / solution) >= 0.95;
3. **no shape** below 0.85.

T4 passes when both GPUs pass in both scoring runs (1 and 2). Report the
per-shape ratios with their set ranges, the share of shapes >= 0.95, the
best baseline per shape, and the GPU-hours and wall-clock used; a fail is
recorded as a fail, with the shapes and the reason.

**Invalidating a run** (recorded, not scored as a pass): the transcript shows
the agent read the sealed directory or anything under `~/hk-dx/t4/` other than
running `bin/`, `bench/t4/baselines.py` or `bench/t4/triton_grouped.py`, the
baselines' library sources (aiter, vLLM, CK, Triton kernels), git history or
GitHub, or the web; or its solution special-cases the public shapes by name or
exact dimensions in a way that does not cover the guarantees of task.md. Check
with `grep` over the transcript for `sealed`, `hk-dx/t4`, `baselines.py`,
`triton_grouped`, `dist-packages/aiter`, `dist-packages/vllm`, `git log`,
`git show`, `gh `, `curl`, `wget`.

## 7. The record

`bench/tune/agent-runs/t4-$RUN/` (the goal's evidence column), committed
through a PR like any other change:

- `README.md`: the model and session, start/end, wall-clock hours, GPU-hours
  per GPU (from the ledger), tool calls, the outcome per GPU (verdict, geomean,
  min, shapes >= 0.95), and a page of what the agent tried, from NOTES.md and
  the transcript;
- `solution.py` (as submitted, with its sha256), `NOTES.md`, `ledger.tsv`;
- the sealed scores (`<arch>-agent-$RUN-{1,2}.{json,md}`) and the public one.

Publishing the sealed scores reveals the hidden shapes: the set is then
**spent**. A second fresh-agent run needs a new sealed set: new models (or new
TP/batch cases) and a new master seed in `~/hk-dx/t4/sealed/models.json`, then
`python3 -m bench.t4.shapes --models ~/hk-dx/t4/sealed/models.json --out
~/hk-dx/t4/sealed/shapes.json`, and its baselines re-checked (`t4seal ...
--baselines-only`).

## 8. Before the first run: the sealed baselines

`t4seal gfx950 ~/hk-dsl-t4 --baselines-only --label baselines-<date>` (and
the gfx1100 halves) shows every baseline runs and is checked on the sealed
shapes; E1 ran these when building the task (results in
`~/hk-dx/t4/sealed/results/`, the Triton picks in `triton_picks.json` there).
