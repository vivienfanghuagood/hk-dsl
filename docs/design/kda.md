# KDA: the default way a new kernel is built

**Decision (owner, 2026-10-24):** every new kernel in this project is built as a KDA task. KDA
is the AMD/hk port of NVlabs' Kernel Design Agents. A task can be done by a lead agent, by a
headless session, or by a person. This document says what a task is and how its record is kept.
It also explains how the record decides which DSL capability gets built next.

## A task

A task is one workspace: `~/hk-dx/kda/<task>/`, made by `_flow/new_task.sh`.

```
docs/contract.md   objective, correctness, target, allowed approaches, validation command,
                   evaluation command, promotion criteria -- fixed before the first round;
                   changed only by the owner, with the change dated in the file
docs/draft.md      the first design and what it rests on
docs/plan.md       the rounds planned, kept current
candidates.jsonl   one record per round (_flow/tools/kda_record.py: schema, failure class)
benchmark.csv      every timed number, with its run and GPU
harness.py         inputs, the reference, compare(); eval*.py / validate.sh
profile/           hk.profile reports the rounds cite
.claude/skills/    the three KDA skills (kda-loop, hk-kernelwiki, hk-profile-report),
                   installed per workspace, never globally
```

Each round follows the same loop:
1. Draft a candidate.
2. Validate it against the reference at the real shapes. A benchmark never times a config
   that has not been checked.
3. Evaluate it, in one process and interleaved with the baselines.
4. Profile it when it misses the target.
5. Record it with one failure class.

The round is not done until the record is written and a dated STATUS line says what changed.
A result that later turns out wrong is not deleted: a correction record is appended, citing
the evidence.

## Failure classes

Every record that did not succeed gets exactly one class. The class is the cause, not the
symptom.

| class | meaning | example |
|---|---|---|
| `dsl` | the language could not express it | hk refused int8 register tiles on gfx1100 (T-SA, T-CR) |
| `codegen` | expressible, but the generated code was wrong or unusable; register spills and LDS overflow count here | 8-warp stage 1 spilling (K1 c12), LLVM serialising int8 WMMAs (T-SA) |
| `harness` | the measurement, reference or tooling was wrong | -- |
| `algorithm` | correct and as written, but not faster | weights through LDS (K1 c6/c22/c27) |
| `env` | the machine, driver or runtime | torch reporting WGPs as CUs on gfx1100 (T-SA) |
| `correctness` | a wrong result that reached a reported number | K1's partial grid and depth-3 remainder (c8-correction) |

## The cross-task table

`~/hk-dx/kda/_flow/tools/failure_table.py --write` regenerates it from every task's
`candidates.jsonl`, and from the controlled experiments' arms (`~/kda-<exp>-<arm>/kda`). It
writes the result to `~/hk-dx/kda/_flow/failure-summary.md`. Regenerate it whenever a task
finishes and copy the table here.

As of 2026-10-26 (`failure_table.py` after both experiments' reruns from main and the megakernel
tasks; the experiments' arms are `~/kda-<exp>-<arm>/kda`, their reports `~/hk-dx/kda/runs/`):

| task | rounds | dsl | codegen | harness | algorithm | env | correctness |
|---|---|---|---|---|---|---|---|
| convrot-int8-gemm (T-CR) | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| k1-moe-a8w4 | 39 | 3 | 3 | 1 | 14 | 0 | 1 |
| k2-mxfp8-gemm | 22 | 0 | 3 | 0 | 5 | 0 | 0 |
| k3-dit-attn | 6 | 0 | 0 | 0 | 2 | 0 | 0 |
| mk-decode | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk-primitives | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk-tp2-ar | 3 | 0 | 2 | 0 | 0 | 0 | 0 |
| experiment 2, hk arm (K2 MXFP8; main + 2 MX commits) | 13 | 2 | 1 | 0 | 5 | 0 | 0 |
| experiment 2, Triton arm (K2 MXFP8) | 19 | 0 | 7 | 0 | 5 | 0 | 0 |
| experiment 2 rerun, hk arm from main `3fd779f` | 20 | 1 | 3 | 1 | 4 | 0 | 0 |
| experiment 1, hk arm (T-SA) | 25 | 2 | 2 | 0 | 6 | 1 | 0 |
| experiment 1, Triton arm (T-SA) | 10 | 1 | 2 | 0 | 2 | 0 | 0 |
| experiment 1 rerun, hk arm on `claude/rdna3-int8` (T-SA) | 15 | 3 | 1 | 0 | 6 | 0 | 0 |
| experiment 1 rerun 2, hk arm from main `3fd779f` (T-SA) | 14 | 0 | 2 | 0 | 3 | 0 | 0 |
| **all** | 195 | **13** | **26** | **2** | **52** | **1** | **1** |

The reruns from main (2026-10-25) are what the capability merges (#28-#31) bought: experiment 2's
hk arm went from 0.86x to 1.234x the best baseline (Triton's arm 1.118x), and experiment 1's hk
arm wrote no C++ outside the DSL for the first time (1.505x the oracle, correct at round 1).

## The named gaps so far

Each row is a `dsl` or `codegen` record with its evidence, and its status in main as of 2026-10-26.

| gap | class | found in | status |
|---|---|---|---|
| int8 register tiles / iu8 WMMA refused on gfx1100 | dsl | T-SA run 1 (blocked: C++ rewrite) | in main (#31) |
| i8 tile loaded global -> registers is silently wrong on gfx1100 | dsl (correctness) | T-SA rerun c0 | fixed in main (#31): HipKittens 5cb93472 (every rdna3 elementwise copy indexes the packed word) + test |
| register-tile bitcast (i32 <-> fp32) | dsl | T-SA rerun c7 (C++ rewrite) | in main (#28, `Tile.bitcast`, same width only) |
| paired-lane accumulator -> operand exchange (P^T) | dsl | T-SA run 1 and rerun c2 (C++ rewrite) | open; rerun 2 went through a per-warp LDS tile instead, and HipKittens' accumulator -> operand `copy()` costs a permlanex16 an element (~216 of 475 VALU per 64 WMMA, codegen) |
| `granules()` only for 2-byte LDS tiles; no fp32 scalar load | dsl | T-SA rerun | open (still so in rerun 2: int8 K cannot use the granule path, and the hoisted address VGPRs cost a workgroup a WGP) |
| scattered atomic row add (top-k reduce in the epilogue) | dsl | K1 c29 | in main (#30, `Global.scatter(add=True)`) |
| row-contiguous lanes for atomics: an epilogue re-layout through LDS | dsl | K1 c29 | in main (#30, `Global.scatter_add_lds`: 307 vs 85 G pk-adds/s). K1 does not use it: bf16 atomics cap a top-6 reduce at ~0.8 ms at 16k tokens, 4x a sum pass (c33) |
| MX-quantizing epilogue store (exact E8M0, fp8 pack, 32-row scale layout) | dsl | K1 structural conclusion, c31 | in main as one fused op (#29, `mx_quant_store`); the general primitives (floor/ceil, bit ops on tiles, byte scatter) are still missing |
| AGPR operand placement | codegen | K1 c30 | not needed: LLVM already puts the weights in AGPRs. Its residue is addresses parked in AGPRs (~200 copies / 192 MFMA) |
| spill gate refused SGPR -> VGPR-lane spills (ScratchSize 0) | codegen (tooling) | K1 c32 | in main (#30: the gate refuses only memory spills) |
| stale docs/api after a new API turned branch CI red (8 + 4 + 4 runs) | harness (tooling) | owner, 2026-10-25 | pre-commit hook (gen_api_docs --check) on every ~/hk-dsl worktree; kda_record `--repo` |
| loop-rewriter liveness errors point at the loop, not the cause | dsl (front end) | T-SA rerun | open; rerun 2: a tuple-unpacked loop variable in a runtime loop is reported as a rebinding |
| fp8 (1-byte) tiles in `group_load_async` / LDS-DMA on **main** | dsl | experiment 2 hk arm: the ceiling (0.86x vs Triton 1.12x) | in main (#28: `g.load(wait=False)` of fp8 tiles, the 16-byte LDS DMA) |
| a synchronous `g.load` drains vmcnt and lgkmcnt to 0 | dsl / codegen | experiment 2 hk arm (1900 of 4550 cycles a K-tile) | inherent with in-order vmcnt (#28); the asynchronous copies avoid it |
| `tileVgprs` prices fp8 tiles as fp32 (4x) | codegen (estimate) | experiment 2 hk arm (576 estimated vs ~200) | fixed in main (#28: 8-bit tiles priced by their bits) |
| pingpong halves on disjoint LDS regions not expressible to hk-lds-pipeline | dsl | experiment 2 hk arm | open |
| control of LLVM's AGPR allocation (`amdgpu-agpr-alloc`) | codegen | experiment 2 Triton arm +9%; K1 c30's copies | open: a cheap hk experiment |
| the validator reuses one weight per shape (stale weight-cache bugs pass) | harness | experiment 2, both arms | open |

Found by the reruns from main (`~/hk-dx/kda/runs/validation-reruns-report.md`, 2026-10-25):

| gap | class | found in | status |
|---|---|---|---|
| no byte / bit operations on tiles; `MxScales` cannot be stored, built from a tile or loaded from LDS; the only raw-byte form (`loose`) is one byte load per 16-row tile and K-step -- so the E8M0 activation-scale repack stays in torch, the largest loss on short shapes | dsl | experiment 2 rerun | open |
| the 4-wave kernel and the persistent pingpong spill at K = 1152 / 1280 (accumulators displaced while addresses sit in AGPRs); the spill gate refuses them rightly, but hk has no allocation lever beyond the program's shape | codegen | experiment 2 rerun | open |
| a 128x256 schedule with `loose` A scales at K = 1152 not bit-identical (max-rel 0.78, no spill) | correctness (open lead) | experiment 2 rerun | cause found: the compiler copied an inline-asm `ds_read`'s destination registers before hk's wait; main refuses such kernels at build time (#39, the wait gate) |
| HipKittens' LDS allocator hands out generic pointers, so converting LDS stores become `flat_store_b16`, unordered against the inline-asm `ds_read` (a real race that passed 10 bitwise repeats) | codegen | experiment 1 rerun 2 | open; harness lesson: bitwise repeats do not prove LDS ordering |
| rocprofv3 counters read zero on node 0043 (GRBM_GUI_ACTIVE = 0) | env | experiment 1 rerun 2 | open |

## How the table chooses DSL work

A `dsl` or `codegen` record is a capability gap with a name. The DSL backlog is ordered as
follows:

1. **Tasks blocked by the gap.** A `dsl` failure that stopped a task, or forced work outside
   the language, comes first. Example: T-SA's int8 tiles, since added in main (#31).
2. **How often it recurs across tasks.** This counts recurring `codegen` classes too, such as
   register pressure from operands the DSL cannot place.
3. **The core kernels' remaining targets.** For example, the K1 structural conclusion
   (`~/hk-dx/kda/k1-moe-a8w4/docs/structural-conclusion.md`) asks for an MX-quantizing epilogue
   store and explicit AGPR operand placement.

`algorithm` failures are the normal cost of search and drive nothing in the DSL. A
`correctness` record always gets a KernelWiki pitfall, plus a guard in the op or the harness
that would have caught it.
