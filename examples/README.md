# Examples

Eleven kernels written from scratch, in the order they introduce the language, two files of kernels built from templates,
two about gfx950's XCDs (a task graph on `hk.comm`, and workgroup placement),
and a script that checks the shipped kernels against torch. Each kernel
example checks its result against an fp32 torch reference before it times
anything, and times itself against the torch call it would replace and,
where there is one, the tuned kernel in `hk.ops` or the Triton kernel it
ports (7-10 are Triton tutorials and the kernels a serving engine runs
beside its model).

| | example | what it shows |
|---|---|---|
| 1 | [`01_vector_add.py`](01_vector_add.py) | a complete kernel: tensors, a grid, tile loads and stores, `+` |
| 2 | [`02_rmsnorm.py`](02_rmsnorm.py) | a runtime loop, a row reduction accumulated in place, a cross-warp combine, `scale[:, None]` |
| 3 | [`03_softmax.py`](03_softmax.py) | three passes over a row: max, sum of exponentials, normalise |
| 4 | [`04_gemm.py`](04_gemm.py) | LDS staging by the workgroup, asynchronous LDS reads, a derived `lgkmcnt`, `acc += a @ b.T` on WMMA |
| 5 | [`05_attention.py`](05_attention.py) | flash attention: scores computed transposed, an online softmax, a uniform rescale branch, `hk.scope()` |
| 6 | [`06_control_flow.py`](06_control_flow.py) | `for` / `if` / `while` on device values, a carried scalar, the C++ they become |
| 7 | [`07_dropout.py`](07_dropout.py) | `hk.rand`, Triton's Philox bit for bit; a runtime seed via `load_scalar`; `(r - p).step()` as the keep mask |
| 8 | [`08_embedding.py`](08_embedding.py) | `table.gather(T, ids, at)`: rows by index, an out-of-range id read as zeros without a branch |
| 9 | [`09_kv_scatter.py`](09_kv_scatter.py) | `cache.scatter(t, slots, at)`: rows to slots, a `-1` slot dropped by the buffer store |
| 10 | [`10_layernorm_backward.py`](10_layernorm_backward.py) | a row reduction and a column reduction (`sum(axis=0)`) in one kernel, tiles held in registers across two passes, partial sums without a lock |
| 11 | [`11_vllm_integration.py`](11_vllm_integration.py) | your kernel in vLLM: `register.op` with a reference and a route, `op.check`, `register.patch`, and the end-to-end A/B (`python -m hk.integration ab`) |
| 12 | [`12_templates_rowwise.py`](12_templates_rowwise.py) | `hk.templates.elementwise` and `rowwise`: GELU-and-mul, RMSNorm, the fused add, softmax and a cross-entropy over 151936 columns, each the math of one element or one row; the template picks the plan per GPU ([the guide](../docs/templates/elementwise-rowwise.md)) |
| 13 | [`13_templates_gemm.py`](13_templates_gemm.py) | `hk.templates.gemm`: linear + bias + GELU in one kernel -- your epilogue traced into the store of hk's tuned GEMM, on both GPUs ([docs/templates/gemm.md](../docs/templates/gemm.md)) |
| 14 | [`14_task_graph.py`](14_task_graph.py) | `hk.comm` (gfx950): a persistent `resident=True` grid taking tasks from a queue (`hk.comm.claim`), each layer waiting for the one before (`wait` with a timeout into an error element, `arrive`), through `"device"`-scoped tensors; `hk.arch.xcd_id()` shows the tasks spread over the 8 XCDs ([docs/design/comm.md](../docs/design/comm.md)) |
| 15 | [`15_placement.py`](15_placement.py) | which XCD runs which workgroup: `placement=hk.Placement.cluster(...)`, `Kernel.placed()`, `placement="auto"` and `placement_choice()`; outputs bitwise the unplaced kernel's, `hk.check --placement` with no GPU; the identity on gfx1100 ([design](../docs/design/placement-pass.md)) |
| | [`validate.py`](validate.py) | the shipped `hk.ops` kernels at typical LLM shapes: correctness, then speed against torch |

## Running

From a clone, with the release installed (`pip install hk-dsl`) -- or with this
checkout's `hk._mlir` built, which the examples then use instead
([building](../docs/building.md)):

```bash
python3 examples/01_vector_add.py              # on the GPU: check against torch, then time
python3 examples/04_gemm.py --dump ir          # the hk MLIR module, no GPU needed
python3 examples/04_gemm.py --dump cpp         # the HipKittens C++ it translates to
python3 examples/05_attention.py --build       # compile with hipcc and print the resources
python3 examples/05_attention.py --causal
python3 examples/validate.py --quick           # every shipped op, smaller shapes
python3 examples/validate.py --only gemm,attention
```

`--dump` and `--build` need no GPU, so the examples can be read and compiled
on any machine with hipcc. `--arch gfx1201` (or `gfx942`, `gfx950`) builds
for a part that is not in the machine. `--no-bench` checks correctness
without timing.

On an MI355X (gfx950) the GPU tier runs them too. `05_attention.py` walks
its 32-key blocks on the matrix core there, Pᵀ read in place
([chapter 39](../docs/history/39-cdna-paged.md)), and `validate.py` passes
every case, attention included ([CDNA3/4](../docs/archive/design/cdna.md)).

The timings follow the protocol every number in this repository was taken
with: all contenders in one process, interleaved round by round, best round
reported. Absolute times drift between processes; ratios taken in one
process drift much less, though for microsecond kernels they still moved
5-30% from one process to the next
([the performance gate](../docs/testing.md#the-performance-gate)).

## What they leave out

These are the shapes of the kernels, not the tuned versions. The GEMM is
single-buffered and overlaps nothing but its two LDS reads; the attention
reads K and V from global memory in every warp instead of sharing them
through LDS; the vector add uses 16x64 register tiles, the matrix core's
layout, where a streaming kernel is faster on register strips (02 shows both;
[the language](../docs/language.md#register-strips)). For speed, start from
a template: 12 and 13 are at the tuned kernels' speed, and
[`hk.templates.gemm`](../docs/templates/gemm.md) is the tuned GEMM with your
epilogue. The tuned kernels themselves are `hk.ops`
([API](../docs/api/hk.ops.md)), and the history in
[docs/history](../docs/history/README.md) says how they got that way. None of the
examples handle a ragged last tile: rows and columns are multiples of the
tile, and each example's docstring says which. `bounded=True` loads and
stores are how the shipped kernels take the rest.
