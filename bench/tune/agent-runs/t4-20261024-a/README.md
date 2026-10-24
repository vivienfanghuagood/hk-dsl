# T4 run 20261024-a: a fresh agent writes MoE grouped GEMM in hk-dsl

Criterion T4 of [docs/design/tuning-goal.md](../../../../docs/design/tuning-goal.md),
run by [bench/t4/PROTOCOL.md](../../../t4/PROTOCOL.md).

**Outcome: FAIL.** The gfx950 half passed both sealed scoring runs; the gfx1100
half missed the 0.85 floor at one hidden shape in both, while passing the
geometric mean.

| | gfx950 (MI355X) | gfx1100 (W7900D) |
|---|---|---|
| run 1 | **PASS**: geomean 1.132, min 0.922, 13/14 >= 0.95 | **FAIL**: geomean 1.039, min **0.819**, 12/14 >= 0.95 |
| run 2 (an hour later, new processes) | **PASS**: geomean 1.131, min 0.921, 13/14 | **FAIL**: geomean 1.032, min **0.827**, 12/14 |
| correct + audit clean | 14/14 both runs | 14/14 both runs |

The failing shape is deepseek-v3-tp8-down-t4096 on gfx1100 (E = 256, N = 7168,
K = 256, 32,768 rows): 3,335-3,362 us against Triton's 2,732-2,767.

## The run

| | |
|---|---|
| agent | a new Claude Code session, model claude-opus-5-5 (Claude Code 2.1.306), headless (`claude -p`), started in an empty directory with no project memory; WebFetch/WebSearch and inter-session tools disallowed, no MCP servers |
| prompt | bench/t4/task.md at the task commit 56c0d6c, then PROTOCOL.md section 2a adapted to working on the workstation directly (the checkout path, `t4gpu`/`t4pull`, the budget) |
| start / end | 2026-10-23T22:55:34Z / about 08:29Z, when the agent stopped on its own: 9.56 h of the 10 h wall clock |
| tool calls | 333 (334 turns); cost $107.57 |
| GPU time (ledger.tsv) | gfx950 0.57 GPU-h in 63 commands; gfx1100 0.25 GPU-h in 8 commands (budget 4.0 each) |
| submission | `solution.py`, sha256 6ac7e19bc4e1ad80...ecd825e (solution.sha256), 1,076 lines |

**Validity.** The transcript was searched as PROTOCOL.md section 6 says
(sealed, hk-dx/t4, baselines.py, triton_grouped, dist-packages/aiter,
dist-packages/vllm, git log, git show, gh, curl, wget): its only uses of
`~/hk-dx/t4` were the permitted `bin/t4gpu` and `bin/t4pull`; it read no
baseline source, git history, GitHub or the web, and the solution has no
case for a public shape by name or dimensions.

**Scoring.** Both runs used a fresh checkout of the harness at 35ca007 with the
submitted `solution.py` copied in, not the agent's own checkout. 35ca007
differs from the task commit only by the audit fix the agent itself found
(torch.profiler on an MI355X lost a lone kernel in about a third of one-call
profiles, so a correct hk-only solution could fail the audit). On gfx1100 the
scoring checkout's sync to the pod (a 292 MB tarball through `kubectl exec` at
~14 KB/s) had not finished after 1.5 h; the score instead ran in a pod-side
tree made from the agent's built source with `scratch/` removed and
`bench/t4/` replaced by the pristine harness plus the submission -- the agent
had changed nothing outside those two. The sealed shapes were copied into the
pod for the scoring and deleted after it.

## What the agent built

One launch a call and no host synchronisation: every workgroup finds its expert
from the offsets on the GPU (a search over slot starts, each slot a block of
rows of one expert) and computes its tile with a grouped
version of hk's own GEMMs -- gemm_cdna's 256x256 pingpong and gemm_cdna_dp's pipelined ring on
gfx950, a small-K variant that keeps the whole K of x in LDS, and on gfx1100
the RDNA3 GEMM with one N block a workgroup. A first-call tuner picks among
2-8 candidates per (rows, N, K, E) on a synthetic skewed routing over weight
copies rotated past the LLC, checking each candidate's output. Its NOTES.md
is the full log, attempt by attempt.

Two things it found along the way that changed the evaluation or hk:
- the audit's dropped kernels (above), now fixed in the harness (35ca007);
- a sweep with checks off had scored a candidate 1.52x that computed 256 of
  384 columns; it added a guard and kept checks on.

## Why gfx1100 missed

From about 23:47Z (hour 1 of 10) the agent's gfx1100 commands stopped
returning: a candidate kernel it was timing (`rflat`, a flat (N block, K tile)
stream it had built for short K) hung on pod GPU 3 and held the GPU and its
lock for the next 9 hours, so every later gfx1100 command queued behind it.
The lead found and killed it before scoring. So the gfx1100 path the agent
submitted is the version it last measured before the hang (its own public
score there: geomean 1.107 and 1.075 over the two halves, min 0.932 at
deepseek-v2-down, K = 192) -- the short-K case the hidden set's deepseek-v3
down (K = 256) then failed on. On gfx950, where it kept working, it had added
a short-K variant, and the same hidden shape scored 1.112.

The harness lesson: a GPU command that times out on the pod has to take its
remote process with it; today a hung kernel outlives the cut-off ssh and holds
the GPU.

## Files

- `solution.py`, `solution.sha256`, `NOTES.md`: the submission as copied at the stop.
- `ledger.tsv`: every GPU command (start, end, seconds, arch, exit status, command).
- `{gfx950,gfx1100}-agent-20261024-a-{1,2}.{json,md}`: the sealed scores, runs 1 and 2.
- `public-gfx950-agent.{json,md}`: the agent's own last public score on gfx950 (PASS, geomean 1.208, min 1.010).

Publishing these scores reveals the hidden shapes: the sealed set of this run
is spent. A second fresh-agent run needs a new one (PROTOCOL.md section 7).
