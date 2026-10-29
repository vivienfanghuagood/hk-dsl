# T4: why this task, these baselines and this rule (evaluator only)

Criterion T4 of [docs/design/tuning-goal.md](../../docs/design/tuning-goal.md):
*a fresh agent given only the docs, `hk.check`, `hk.profile`, `hk.tune` and
`hk.schedules` takes a hidden variant -- one of fp8 GEMM, grouped/MoE GEMM, an
attention `score_mod` -- from first kernel to >= 0.95x the best baseline within
a stated budget, on both GPUs; the run is recorded.*

This note says how that sentence became `task.md`, `score.py` and the pass
rule. It is removed from the evaluated agent's
checkout: it names what the baselines are and how close
each is to the hardware, which the agent should find out by measuring.

## The variant: the grouped GEMM of an MoE layer

Of the three the goal names, only this one is new work for hk on both GPUs
and has strong, checkable baselines on both:

| variant | what hk has today | baselines | verdict |
|---|---|---|---|
| fp8 GEMM | `hk.templates.gemm` loads fp8 and stores fp8 C on gfx950; the tuned GEMM routes and `hk.tune` apply as they are | hipBLASLt fp8 on gfx950. gfx1100 has no fp8 matrix instruction (RDNA3's WMMA takes f16, bf16, iu8, iu4), so "both GPUs" would mean an fp8 *storage* GEMM there, with no baseline worth the name | mostly tuning an existing route, one GPU only |
| attention `score_mod` | `bench/dx` T6 (tanh soft-cap) and T7 (sliding window) are exactly this, solved from the docs alone at 1.6-4.1x torch.compile in phase 1 (both GPUs) | flex-attention / Triton | already measured, not hidden |
| **grouped / MoE GEMM** | `hk.ops.moe.moe_gemm`: an RDNA decode kernel (vLLM's fused-MoE layout, 16-row blocks). On gfx950 it does not build (a 16x16 bf16 tile is not a whole number of gfx950's 16x32 MFMA fragments); on gfx1100 it runs 1.3-3.4x slower than the best baseline at these batch sizes | a tuned grouped Triton GEMM on both; aiter's CK MoE GEMM, aiter's Triton gmm and vLLM's fused-MoE kernel on gfx950; `torch._grouped_mm` and hipBLASLt per expert on both | **chosen** |

The operation is the plain one every MoE layer runs after its tokens are
permuted: contiguous expert groups with their offsets on the GPU, both expert
GEMMs (`gate_up` = vLLM's `w13`, `down` = `w2`), bf16 in and out, fp32
accumulation. No routing weights, activation or gather are fused in: each
library fuses different things, and only the bare grouped GEMM is the same
math in every baseline. The interface (`solve(x, w, offsets, out)` with an
optional untimed `prepare(w)`) lets a solution pre-shuffle weights, as aiter's
CK kernel does.

What makes it a real task rather than a route: the group sizes are data on the
GPU and change on every call. A solution has to either read them on the host
(paying the synchronisation, which the timing charges) or schedule on the GPU
from them -- a persistent or block-table kernel, empty and one-row experts,
an M per expert anywhere from 0 to tens of thousands. hk's dense GEMMs assume a
fixed M.

## What a solution with no design gets

`hk.ops.matmul` once per non-empty expert, the offsets read to the host
(`scratch_hk_loop.py`, not committed; audit clean: 99% of the call's device
time in hk kernels, the rest torch bookkeeping), on three public shapes:

| shape | gfx950 ratio | gfx1100 ratio |
|---|---|---|
| olmoe-1b-7b-down-t512 (64 experts, ~64 rows each) | 0.081 | 0.337 |
| mixtral-8x7b-tp4-gate_up-t2048 (8 experts, ~512 rows) | 0.621 | 0.898 |
| mixtral-8x7b-tp4-down-t4096 (8 experts, ~1024 rows) | 0.721 | 0.975 |

Few big experts are within reach of hk's dense kernels; many small ones are
an order of magnitude off on gfx950 (a launch per expert, each too small to
fill 256 CUs). The gap is the task.

## The baselines

Each is checked for correctness on every shape before it is timed, and a wrong
one is never "best". On gfx950 they run in the vLLM ROCm 10 image (torch
2.12, Triton 3.8.0, aiter 0.1.25, vLLM 0.32.1rc1, FlyDSL 0.3.4.1); on gfx1100 in
the dev pod (ROCm 7.2.4, torch 2.10, Triton 3.6), where aiter and vLLM do not
run.

- **`triton`** (`triton_grouped.py`): a grouped GEMM written for this task in
  the style of the Triton grouped-GEMM tutorial, every program finding its
  expert from the offsets on the GPU. It is tuned per shape -- block sizes,
  warps, stages, and on CDNA the MFMA size and `waves_per_eu` -- and the
  picks for the public shapes are committed (`results/triton_picks.json`; the
  sealed shapes' picks are kept with the sealed set). It is the best baseline
  at 9 of 12 public shapes on gfx950 and all 12 on gfx1100. Tuning our own
  competitor is deliberate: an untuned baseline would make 0.95x mean little.
- **`aiter_ck`**: aiter's CK MoE second stage at top-1, with aiter's own
  block size and kernel choice and pre-shuffled weights -- the kernel SGLang
  and vLLM run on an MI355X. Best at 3 of 12 (the Mixtral and Qwen1.5 shapes:
  few experts, wide N).
- **`aiter_gmm`**, **`vllm_fused_moe`**: aiter's Triton grouped GEMM with its
  gfx950 configs, and vLLM's fused-MoE Triton GEMM with vLLM's config for the
  batch. Never best on the public shapes, kept because the hidden ones may
  differ.
- **`torch_grouped_mm`**, **`torch_loop`**: `torch._grouped_mm` and hipBLASLt
  per expert -- what a framework does without a kernel library.
- **`hk_ops_moe`**: hk's shipped kernel, where it builds.
- `proxy:dense`, never a baseline: the dense hipBLASLt GEMM of rows x N x K on
  one expert's weights, timed for scale.

The best baseline is far from the hardware's limit at every public shape: on
gfx950 at 0.22-0.53 of the datasheet roofline, on gfx1100 at 0.41-0.71 (the
`roofline` and `best/roofline` columns of `results/*-public-*.md`). The target
is reachable without beating physics.

## How it is timed

The project's measurement rules (history chapters 31-32), plus what a grouped
GEMM adds:

- Every arm is checked on three routings no timing uses: one with empty
  experts first and last, a one-row expert and one expert holding half the
  rows, and two drawn like the shape's.
- Interleaved, in one process: 3 sets x 7 rounds, the arms' order rotated
  every round; each arm's pass is several back-to-back calls over weight
  copies rotated past the last-level cache, queued behind a GPU spin longer
  than its host time, between two events.
- **New routings every round**, drawn from the shape's distribution (a skewed
  router: some experts get several times their share), so nothing derived
  from one call's offsets can be reused by the next -- the rule task.md states
  is also what the timing enforces.
- A call's time is device time from its first kernel to its last, plus any
  wait for the host it causes: reading the offsets on the host is allowed and
  paid for.
- An arm's time is the median over sets of each set's median round; a
  solution's ratio is best baseline / solution per set, reported as the median
  and the range over the sets.

## The audit

"Written in hk" is checked, not trusted: one call of the solution under
torch.profiler, every kernel classified as hk's (a kernel hk-dsl built), torch
bookkeeping (elementwise, scan, index, copy, fill), or foreign (hipBLASLt,
rocBLAS, CK, Triton, aiter, anything else). Foreign kernels fail it, and so
does a torch kernel that multiplies matrices; the solution file's imports are
checked statically (no `triton`, `aiter`, `vllm`, `flydsl`, `sglang`, or
anything under `bench`). `sol:torch_loop` is the check of the check: it is
correct, fast at a few shapes, and fails the audit at all twelve on both GPUs.

## The pass rule

Per GPU, on the sealed shapes: correct and audit-clean on every one, the
**geometric mean** of the ratios >= 0.95, and **no shape below 0.85**. The goal
says ">= 0.95x the best baseline"; asking it of every one of 14 shapes, each
against whichever of up to seven libraries happens to win there, would fail a
solution on the noise of its worst shape (the range between sets reached 9% of the
ratio at one public shape on gfx950). The geometric mean holds the level, the floor stops a
solution from buying it with a shape it gave up on. T4 passes when both GPUs
pass in two scoring runs started at least an hour apart.

## The hidden set

Fourteen shapes per GPU, from seven MoE models that are not among the public
six (two cases a model, each one of its expert GEMMs at a batch size,
the routings from a master seed fixed before any solution existed), drawn by `shapes.py` within task.md's
guarantees. They are kept outside the repository and
are read only through `$HK_T4_SEALED`; every baseline was run and checked on
them before the task commit. Publishing a scored run
reveals them, and the set is then spent.

## The budget

10 hours of wall-clock and 4.0 GPU-hours on each GPU, compiles included,
kept by the run's GPU-time ledger. For scale: `hk.tune`'s acceptance runs spent
0.09-0.21 GPU-minutes a shape, a full public scoring run takes a few minutes
per GPU, and a gfx1100 command pays a sync to the remote machine. Four GPU-hours is room
for a few hundred scored attempts, not for a blind sweep.
