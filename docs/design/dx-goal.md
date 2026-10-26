# Goal: fast kernels that are easy to write, for people and for agents

Status: active, from 2026-10-20. Owner: the `claude/*` workstreams. Progress: the
pull requests from those branches, and `<workspace>/STATUS.md` on the workstation.

## The goal

On a Radeon (gfx1100) and an Instinct (gfx950), a developer or a coding agent
who has read only hk-dsl's documentation writes a kernel that is correct and at
least as fast as `torch.compile` — and close to what an hk expert would write —
in one sitting, without reading the library's source.

## Where it stands (2026-10-20, measured)

A GELU(tanh)-and-mul written from the docs, as the first page teaches (16x64
register tiles), is correct on both GPUs, 2-3x `torch.compile` on decode-sized
inputs, but **0.61x `torch.compile` at 4096x14336 on an MI355X** (0.93x on a
W7900D). Rewritten on register strips — which takes the language reference's
deep sections and a per-architecture constant — it reaches 0.88x; the tuned
`hk.ops.silu_mul` reaches 1.05x. `examples/01_vector_add.py` itself is 0.59x
`torch.add` on an MI355X. A GEMM with a fused epilogue has no route but copying
`hk/ops/gemm.py` (RDNA) or `hk/ops/gemm_cdna.py` (CDNA). The docs describe
`main`, which is ahead of the wheel on PyPI (gfx950 strips are refused by
0.4.0), list no catalogue of math functions, and the same op takes different
arguments on the two architectures (`silu_mul`'s strided halves).

## Acceptance criteria (milestone 0.5)

| id | criterion | evidence |
|---|---|---|
| G1 | **A DX suite**: 8 kernel-writing tasks with LLM shapes, a torch reference and baselines (torch eager, `torch.compile`, Triton where it runs, the matching `hk.ops` kernel where one exists), run on gfx1100 and gfx950 by one command that writes a JSON scoreboard; runnable on the GPU runners | `bench/dx/`, scoreboard JSON for both GPUs |
| G2 | **Templates with hooks**: `hk.templates.elementwise` / `rowwise`, `gemm(epilogue=...)`, `attention(score_mod=..., mask_mod=...)`. A task solved with one is <= 30 lines of user code and reaches >= 0.95x the tuned `hk.ops` kernel for the same job, or >= 1.0x the best baseline where hk has none, on both GPUs | DX suite rows "template" |
| G3 | **An agent interface**: `python -m hk.check` (correctness against a reference, timing against baselines, the resource report, JSON out); an API reference generated from the code and an `llms.txt`; an agent skill in the repository. A fresh agent given only the docs solves >= 6 of the 8 tasks at >= 0.9x `torch.compile` | `hk.check`, `docs/api/`, `llms.txt`, `skills/`, a recorded fresh-agent run |
| G4 | **A releasable project**: 0.5.0 cut from main with the docs saying which version has what; the same op takes the same arguments on every GPU; CONTRIBUTING, issue templates, an accurate repository description | the release, a green CI |

Every number is measured on the machines below, interleaved against its
baselines in one process, with versions stated; a loss is recorded as a loss.

## Workstreams

| stream | branch | scope |
|---|---|---|
| W1 dxsuite | `claude/dxsuite` | G1, and the "written from the docs" solutions that are today's baseline |
| W2 rowwise | `claude/rowwise` | G2: `hk.templates.elementwise`, `hk.templates.rowwise` |
| W3 gemm | `claude/gemm-epilogue` | G2: `hk.templates.gemm(epilogue=...)` on the tuned RDNA and CDNA GEMMs |
| W4 agent | `claude/agent-interface` | G3: `hk.check`, `docs/api/`, `llms.txt`, the skill; the cross-GPU API audit |
| later | | G2's attention template (after W2/W3 fix the template conventions), G3's fresh-agent evaluation, G4's release |

The work Cursor's agent is finishing in parallel (the matched-stack
re-validation, CDNA C3, the README/performance reorganisation) is not touched
by these streams.
