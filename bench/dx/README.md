# The DX suite

Eight kernel-writing tasks an LLM developer actually has, each with a torch
reference, realistic shapes and baselines, and one command that checks and
times every solution on a GPU and writes a scoreboard. It is the yardstick for
[the DX goal](../../docs/design/dx-goal.md): a kernel written by a person or an
agent who has read only the docs should be correct and at least as fast as
`torch.compile`, close to an hk expert's, on gfx1100 and gfx950.

| id | task | what makes it a test |
|---|---|---|
| `t1_gelu_tanh_and_mul` | Gemma's MLP activation, `(M, 2I) -> (M, I)` | the first elementwise kernel anyone writes; strided halves, ragged M |
| `t2_add_rmsnorm_quant` | residual add + RMSNorm + per-token dynamic int8 quant | three outputs, a row reduction, then a second reduction (absmax) on the normed row |
| `t3_cross_entropy` | per-token cross-entropy over a vocabulary up to 151936 | a row reduction over a very long row, a gather, ignore_index |
| `t4_linear_bias_gelu` | `gelu_tanh(x @ W.T + b)` | a GEMM with an epilogue |
| `t5_linear_residual` | `x @ W.T + residual` (o_proj / down_proj) | a GEMM with a second input in the epilogue |
| `t6_attn_softcap` | causal GQA prefill attention with tanh soft-capping (Gemma-2) | attention with a score modifier |
| `t7_attn_sliding_window` | causal sliding-window GQA prefill attention (Mistral) | attention with a mask modifier and skipped blocks |
| `t8_moe_router` | softmax + top-k + renormalise over E experts | a tiny-row reduction with a selection |

Each task's `task.md` is the problem exactly as a user or an agent receives it.

## Running

```bash
python3 -m bench.dx.run --list                       # tasks, shapes, solutions
python3 -m bench.dx.run --tasks t1 --quick           # three shapes, 5 rounds
python3 -m bench.dx.run --tasks t1,t2 --solutions torch_compile,docs_only --json out.json
python3 -m bench.dx.run --small                      # the test's tiny shapes, correctness only
```

On the workstation (bench/dx needs the worktree's `hk._mlir`, which the helpers build):

```bash
hkx950 <checkout> 5 python3 -m bench.dx.run --json bench/dx/results/gfx950-DATE.json
hkx1100 <checkout> python3 -m bench.dx.run --tasks t1 --json bench/dx/results/x.json   # one task per call
```

Options: `--tasks`, `--solutions`, `--kinds`, `--shapes` (substring of shape
names), `--quick`, `--small`, `--check-only`, `--rounds`, `--no-graphs`,
`--cold-cache` (fresh hk / Inductor / Triton caches, so `first_call_s` is a cold
compile), `--json OUT` (writes `OUT` and `OUT` with `.md`), `--merge A.json B.json
--json OUT` (one GPU's runs of different tasks in one scoreboard). Set
`HK_DX_COMMIT` (and `HK_DX_GPU`) when the run cannot see the git checkout or the
machine's GPU index (the containers cannot).

## The protocol

For each task and shape: the inputs come from `reference.make_inputs(shape,
seed=0)`, and the golden output from `reference.reference(...)` (fp32). Then,
for each solution:

1. `supports(**inputs)` -- a solution may decline a shape, with a reason.
2. **Correctness first.** The solution runs once on a fresh copy of the inputs;
   the wall time of that first call, `setup()` included, is `first_call_s` (the
   compile, unless the caches were warm -- see `--cold-cache`). The outputs go
   through the task's `check()`. A wrong answer, an exception, an OOM, or an
   input the solution modified (unless the task allows it) is recorded and the
   solution is **not timed**.
3. **Timing**, of the correct ones only, all in one process, interleaved: every
   round runs every contender once, in a rotating order, each as a batch of
   back-to-back calls long enough (>= 1 ms) that the events' resolution does
   not matter; the median round is reported. Two numbers:
   - `host_us`: back-to-back eager calls -- what a Python caller pays, host
     overhead (Python, dispatch, guards, launch) included. For decode-sized
     kernels this is often the larger number.
   - `device_us`: the same calls replayed from a CUDA graph -- the GPU's time,
     as a serving engine's captured step sees it. If a solution cannot be
     captured, the reason is in `graph_error` and ratios fall back to host time.
   Inputs stay resident between calls (hot L2 / MALL), as they do inside a
   model's layer loop for the small shapes; the large shapes exceed the caches.
   Every graph is checked before it is timed: an empty one, or one whose
   replay does not follow its inputs (the float inputs halved, the graph
   replayed and compared with an eager call), gets no `device_us`.
   **On RDNA** hk's default launch (the pybind entry point) goes to the null
   stream, which neither torch's events nor a CUDA graph wait for
   (docs/runtime.md, "The stream"), so the runner sets `HK_LAUNCH=codeobj`
   there: the same device code launched on torch's stream, with a Python
   launch in front of it. The result files say so (`hk_launch`).
4. **Ratios** are `other / this` (> 1 = faster): `x_tc` against `torch.compile`,
   `x_best` against the fastest *baseline* at that shape (torch eager,
   torch.compile, Triton, hk.ops), `x_hk_ops` against the hk.ops kernel where
   the task has one; `_dev` on device time, `_host` on host time.

Every result file carries the GPU, the arch, torch, HIP, Triton, the hk version
and commit, and the command.

## The interfaces

### A task: `bench/dx/tasks/<id>/reference.py`

| name | what |
|---|---|
| `SHAPES` | list of dicts, each with a `name` and the task's parameters |
| `QUICK` | names of the shapes `--quick` runs |
| `SMALL` | tiny shapes for `tests/dx` (correctness on every GPU, fast) |
| `make_inputs(shape, device="cuda", seed=0)` | the inputs, a dict of tensors and Python scalars, deterministic |
| `reference(**inputs)` | the golden output(s), in fp32 (or exact) |
| `torch_impl(**inputs)` | the natural PyTorch code: the *torch eager* baseline, and what `torch.compile` compiles |
| `compile_impl(**inputs)` | optional: what `torch.compile` compiles when that differs (e.g. `flex_attention`) |
| `check(out, ref, inputs)` | raises `AssertionError` on a wrong answer; returns a one-line summary |
| `MUTABLE_INPUTS` | optional: input names a solution may modify in place |
| `work(shape)` | optional: `{"bytes": ..., "flops": ...}` per call |

### A solution: `bench/dx/solutions/<id>/<name>.py`

One file per solution; the file name is the solution's name in the tables. The
runner adds two built-ins to every task, `torch_eager` and `torch_compile`.

```python
KIND = "template"            # see below
LABEL = "hk.templates.elementwise with a gelu hook"   # one line, shown in the tables

def solve(**inputs):         # required (or setup): the task's inputs by name ->
    ...                      # the task's outputs (a tensor, or a tuple in task.md's order)

def setup(**inputs):         # optional: work that is not part of the op (allocating
    ...                      # outputs, pre-transposing a weight, picking a schedule)
    return lambda: ...       # -> a zero-argument callable returning the outputs;
                             # it is what is checked and timed

def supports(**inputs):      # optional: True, or a string saying why not
    return True
```

- `solve` takes the task's input names as keyword arguments (`solve(x)` for t1,
  `solve(q, k, v, scale, cap)` for t6; `task.md` lists them) and returns new
  output tensors. A solution must not modify its inputs unless the task's
  `MUTABLE_INPUTS` allows it.
- With `setup`, only the returned callable is timed; anything that is a
  property of the model (weights, a plan) may be prepared there, anything that
  depends on the activations may not.
- `KIND` is one of:

| KIND | who writes it | used as |
|---|---|---|
| `baseline` | the suite: torch eager, torch.compile, Triton | a baseline |
| `hk_ops` | the suite: the matching `hk.ops` kernel(s), when they compute the task | a baseline; `x_hk_ops` |
| `proxy` | the suite: a kernel that computes *different* math at the same cost (say what in `PROXY = "..."`) | timed for scale, never checked, never a baseline |
| `docs_only` | written only from README.md, docs/ and examples/ -- today's DX | contender |
| `template` | with `hk.templates` (W2/W3) | contender |
| `agent` | by a coding agent from the docs (W4's fresh-agent runs) | contender |
| `expert` | by someone who knows the library's insides | contender |

- Lines: the tables count a solution file's code lines (no blanks, comments or
  docstrings), so keep one solution per file and its helpers in it.
- A `docs_only` (or `agent`) solution records, in its docstring, what had to be
  looked up and every doc gap or error on lines starting `GAP:`;
  [FINDINGS.md](FINDINGS.md) collects them.

## Files

```
bench/dx/run.py          the runner (python3 -m bench.dx.run)
bench/dx/common.py       discovery, the check helper, line counting, versions
bench/dx/tasks/<id>/     task.md, reference.py
bench/dx/solutions/<id>/ one file per solution
bench/dx/results/        <arch>-<date>.json / .md: the committed scoreboards
bench/dx/FINDINGS.md     the doc gaps the docs-only solutions hit, and the numbers
tests/dx/                imports every task and solution; correctness on SMALL shapes (GPU tier)
```
