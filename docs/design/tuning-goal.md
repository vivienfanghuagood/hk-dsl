# Goal: an agent tunes a kernel to the hardware's limit

Status: proposed 2026-10-22, for the milestone after 0.5.0. Follows
[dx-goal.md](dx-goal.md) (writing a fast kernel easily); this one is about the
last 5-20%: getting from "faster than torch.compile" to "as fast as the best
kernel that exists for this GPU", on shapes and variants nobody tuned.

## The goal

Given a GEMM- or attention-shaped workload it has never seen -- a shape no
table holds, or a variant (fp8, grouped/MoE GEMM, W4A16, an attention
modifier) -- a coding agent using hk-dsl's tools reaches **>= 0.95x the best
available kernel** (hipBLASLt, aiter, FlyDSL, tuned Triton, HipKittens) on
gfx1100 and gfx950, **correct, measured, within a stated GPU-time budget**.

## Why this, and not a compiler that infers everything

- **What is left is search, not derivation.** The open gaps are shape- and
  schedule-specific: on an MI355X, a 1.75-wave grid leaves an epilogue's VALU
  work unhidden (Llama gate_up at M=1024, 1.09x the plain GEMM); a shape outside
  `routes_gemm_gfx950.json` falls back to `route()`'s rule and runs 0.76x
  hipBLASLt (2048x16384x4096); the CDNA GEMM went from 0.41x to 0.97x
  hipBLASLt by trying ping-pong, weight streaming and CU-filling blocks
  (chapters 33 and 41), not by a rule that would have predicted them.
- **hk-dsl's passes make aggressive search safe.** Wait counts are derived,
  LDS hazards across warps are checked, spills are refused: a schedule either
  compiles to a correct kernel or is refused with a reason. That is what lets
  an agent try things a person would not risk -- and it is what Triton,
  TileLang and CuTe do not give a searcher in the same way.
- **Layout/pipeline inference competes with large teams** (Triton, TileLang,
  CuTe, hipBLASLt, aiter) for months with an uncertain payoff, on the shapes
  where vendor libraries are already near the roofline. hk-dsl routes to them
  where they win; its value is the shapes and variants they do not cover.

So the compiler work in this milestone is chosen by what the agent loop needs:
a schedule space that is data, legality and diagnostics, and a baseline tuner.

## Where it stands (2026-10-22)

- Both GEMMs are already parameterized -- `hk.ops.gemm.gemm_kernel` (gfx1100:
  block_m/n, k_step, dot_slice, warps, warp_rows, n_split, wgm, write_pos,
  lds_buffers, stagger, stream_a, setprio, sched) and
  `hk.ops.gemm_cdna.gemm_kernel` (gfx950: block_m/n, k_step, warps_m/n, wgm,
  xcd_chunk, splits, lds_shape, nt, early_store, late_b, ...) -- but as two
  unrelated keyword sets, with legality spread over ValueErrors, hand-picked
  `CANDIDATES`, and per-arch route tables written by bespoke tools
  (`tools/hk-bench/route_tune.py`, `gemm_cdna_routes.py`).
- `hk.check` says *whether* a kernel is fast (time vs baselines, registers,
  occupancy, spills); nothing says *why* it is slow. rocprofv3 and
  rocprof-compute are installed on both machines and unused by the tools.
- `python -m hk.integration tune` measures missing shapes at model load, among
  a fixed candidate set; there is no search over the schedule space.

## Acceptance criteria

| id | criterion | evidence |
|---|---|---|
| T1 | **Schedules are data.** One `Schedule` type per kernel family (GEMM first, attention second) covering every knob of both GPUs' kernels, with `check(shape, arch)` that refuses an illegal point with reasons before compiling, `enumerate(shape, arch)` over the legal space, serialization into the tuned records, and `hk.templates.gemm(..., schedule=)`. Every shipped route reproduces byte-identical C++ through it. | `hk.schedules`, tests, docs |
| T2 | **Why it is slow.** `hk.profile` (and `hk.check --profile`) runs a kernel under rocprofv3 and reports, per kernel, % of the compute and memory roofline, matrix-core busy, the top stall reasons (vmem/LDS waits, barrier, dependency), LDS bank conflicts, occupancy and the instruction mix of the main loop; JSON for agents and a one-line bottleneck verdict. Validated on known cases: it must name the 1.75-wave tail and the unhidden epilogue above, and the 0.76x shape's cause. | both GPUs |
| T3 | **A tuner that is a baseline, not the answer.** `python -m hk.tune` searches a `Schedule` space for a shape (noise-aware: interleaved, medians, a margin from measured spread, confirmation rounds), writes records/routes, and reports GPU-time spent. On a hidden set of >= 20 GEMM shapes per GPU that no table holds, >= 80% reach >= 0.95x the best baseline within the budget; 2048x16384x4096 on gfx950 reaches >= 0.95x hipBLASLt. | `bench/tune/`, scoreboard |
| T4 | **The agent loop closes.** A fresh agent given only the docs, `hk.check`, `hk.profile`, `hk.tune` and `hk.schedules` takes a hidden variant (one of: fp8 GEMM, grouped/MoE GEMM, an attention `score_mod`) from first kernel to >= 0.95x the best baseline within a stated budget, on both GPUs; the run (time, GPU-hours, what it tried, what the tools told it) is recorded. | the first run (2026-10-24), FAIL (gfx950 passed, geomean 1.13; gfx1100 one hidden shape at 0.82) |

Every number is measured interleaved in one process against its baselines,
with versions and GPU stated, and losses recorded as losses.

## Workstreams (first phase)

| stream | scope |
|---|---|
| K1 schedules | T1 for GEMM on both GPUs: the `Schedule` type, legality, enumeration, the byte-identical round trip, the template's `schedule=` |
| K2 profile | T2: rocprofv3/rocprof-compute counters to per-kernel metrics and a verdict, both GPUs; the three validation cases |
| K3 tune | T3: the search, noise-aware acceptance, the hidden-shape set and its baselines (hipBLASLt everywhere; aiter, FlyDSL, HipKittens on gfx950; tuned Triton), the scoreboard |
| later | attention schedules; fp8 and grouped GEMM; T4's fresh-agent runs |

## Not in scope

- Inferring layouts or pipelines from a loop nest (a polyhedral or
  Triton-style compiler).
- Beating hipBLASLt or aiter where they are at the roofline: route to them.

## Risks

- **GPU time on shared machines.** A search costs GPU-minutes per shape;
  budgets are part of every criterion, and searches yield to other teams' jobs.
- **Tuning noise into wins.** The measurement rules of chapters 31-32 (medians,
  spread-based margins, confirmation) apply to every accepted result.
- **Overfitting.** Acceptance uses shapes and variants held out from the tuner
  and the agents.
