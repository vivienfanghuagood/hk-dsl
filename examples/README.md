# Examples

Eleven kernels written from scratch, in the order they introduce the language,
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
| | [`validate.py`](validate.py) | the shipped `hk.ops` kernels at typical LLM shapes: correctness, then speed against torch |

## Running

From a checkout with `hk._mlir` built (see [docs/getting-started.md](../docs/getting-started.md)):

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
on any machine with hipcc. `--arch gfx1201` builds for a part that is not in
the machine. `--no-bench` checks correctness without timing.

The timings follow the protocol every number in this repository was taken
with: all contenders in one process, interleaved round by round, best round
reported. Absolute times drift between processes; the ratios do not.

## What they leave out

These are the shapes of the kernels, not the tuned versions. The GEMM is
single-buffered and overlaps nothing but its two LDS reads; the attention
reads K and V from global memory in every warp instead of sharing them
through LDS. The kernels they are compared against -- `hk/ops/gemm.py` and
`hk/ops/attn.py` -- are the next thing to read, and the history in
[docs/history](../docs/history) says how they got that way. None of the
examples handle a ragged last tile: rows and columns are multiples of the
tile, and each example's docstring says which. `bounded=True` loads and
stores are how the shipped kernels take the rest.
