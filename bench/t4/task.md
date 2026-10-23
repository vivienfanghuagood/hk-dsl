# T4: the grouped GEMM of a mixture-of-experts layer, on two GPUs

Write, with hk-dsl, the grouped GEMM that a mixture-of-experts (MoE) layer runs
for its experts, and make it as fast as the best kernel that exists for it on
**both** an AMD Radeon PRO W7900D (gfx1100, RDNA3) and an AMD Instinct MI355X
(gfx950, CDNA4). It is scored on shapes you will not see.

## The operation

An MoE layer routes each token to `top_k` of its `E` experts. After the
batch's (token, expert) pairs are permuted so that each expert's rows are
contiguous -- expert 0's rows first -- every expert multiplies its rows by its
own weight matrix:

```python
# x       (rows, K)   bf16, contiguous   -- the permuted (token, expert) rows
# w       (E, N, K)   bf16, contiguous   -- expert g's weights w[g], F.linear's (N, K) layout
# offsets (E + 1,)    int32, on the GPU  -- 0, the running row counts, rows: expert g owns
#                                           rows offsets[g] .. offsets[g+1]-1 (possibly none)
# out     (rows, N)   bf16, contiguous
for g in range(E):
    a, b = offsets[g], offsets[g + 1]
    out[a:b] = (x[a:b].float() @ w[g].float().T).to(bf16)     # fp32 accumulation
```

This is both expert GEMMs of a layer: `gate_up` (vLLM's `w13`: N = 2 x the
expert's intermediate size, K = the model's hidden size) and `down` (`w2`:
N = hidden, K = intermediate), at the batch sizes a server prefills and
decodes. `rows` = tokens x top_k.

What every shape guarantees (the hidden ones included):

| | |
|---|---|
| E | 8 to 256 experts |
| top_k | 1 to 8 |
| N | a multiple of 128, 256 to 8192 |
| K | a multiple of 64 (not always of 128), 128 to 8192 |
| rows | 1 to 65536 |
| groups | any size, including 0 (empty experts, at any position, first and last included) and 1; the sizes follow a router's imbalance (some experts get several times their share) and change on every call |

**Correct** means: relative max error `max|out - ref| / max|ref|` below
`2e-2` against the fp32 reference (`bench/t4/reference.py`), on every routing,
with every row written (the scorer fills `out` with NaN before checking), and
`x`, `w` and `offsets` unchanged.

## Your solution

One file, **`bench/t4/solution.py`**, the same file on both GPUs (branch on the
architecture inside it if you need to):

```python
def prepare(w):                      # optional. Called once per weight tensor, untimed: a
    ...                              # model-load step (relayout, pre-shuffle). It must keep
    return w_prepared                # the bf16 values (no quantization); solve gets its result.

def solve(x, w, offsets, out):       # required. w is prepare(w) if you defined prepare.
    ...                              # Write out (and return it or None), or return a new
                                     # (rows, N) bf16 tensor.
```

`bench/t4/solutions/torch_loop.py` is a complete (and slow) example of the
interface.

Rules:

- **The GEMM work runs in kernels hk-dsl builds** -- `hk.kernel`,
  `hk.templates`, `hk.ops` kernels that hk compiles. torch may allocate memory
  and do small bookkeeping on `offsets` (elementwise, scan, index kernels), but
  no matrix multiply: no `torch.mm`/`matmul`/`F.linear`/`_grouped_mm`, nothing
  that reaches hipBLASLt, rocBLAS or CK. The file may import `hk`, `torch`,
  `numpy` and the standard library -- not `triton`, `aiter`, `vllm`, `flydsl`,
  `sglang` or anything under `bench`. The scorer checks both: the imports
  statically, and every GPU kernel one call launches, under the profiler. A
  solution that fails this audit fails, however fast.
- **Every call is a new routing.** You may cache what depends on the shape
  (E, N, K, rows): compiled kernels, workspaces, plans. Nothing derived from
  the values of `offsets` or `x` may be reused across calls. Reading `offsets`
  on the host is allowed, and paid for: the time of a call includes any wait
  for the host it causes.
- One solution for every shape: it may choose a kernel or schedule per shape
  at run time, but it must handle any shape within the guarantees above, not
  a list of known ones.

## How it is scored

```bash
python3 -m bench.t4.score --solution bench/t4/solution.py                  # all public shapes
python3 -m bench.t4.score --solution bench/t4/solution.py --only olmoe     # names or substrings
python3 -m bench.t4.score --solution bench/t4/solution.py --check-only     # correctness only
python3 -m bench.t4.score --solution bench/t4/solution.py --sets 1 --rounds 3 --baselines triton
```

For each shape, `score.py` (its docstring is the full protocol):

1. checks every arm on three routings no timing uses (one of them with empty
   experts, a one-row expert and one expert holding half the rows);
2. times the correct ones interleaved in one process: 3 sets of 7 rounds, each
   round on routings drawn fresh for it, each arm's pass several back-to-back
   calls over weight copies rotated past the GPU's last-level cache, queued
   behind a GPU spin, between two events;
3. takes each arm's median round per set, the median over sets, and the
   **ratio = best baseline time / your time** (> 1: you are faster), with its
   range over the sets;
4. audits one call of your solution with the profiler.

It writes `bench/t4/out/<arch>-<label>.json` and `.md` and prints `SCORE` and
`VERDICT` lines. `bench/t4/results/` has the baselines' numbers on the public
shapes; the `triton` baseline uses the configs tuned for them there
(`triton_picks.json`) and tunes any other shape on its first call. A full
public run takes a few minutes per GPU.

**The baselines** (each checked for correctness before it is timed; the
fastest correct one at a shape is "best"):

| name | what | GPUs |
|---|---|---|
| `triton` | a grouped Triton GEMM, tuned per shape (blocks, warps, stages; MFMA size and `waves_per_eu` on CDNA) | both |
| `aiter_ck` | aiter's CK MoE GEMM (its fused-MoE second stage at top-1, aiter's own block size and kernel choice, pre-shuffled weights) | gfx950 |
| `aiter_gmm` | aiter's Triton grouped GEMM with its gfx950 configs | gfx950 |
| `vllm_fused_moe` | vLLM's fused-MoE Triton GEMM with vLLM's config for the batch | gfx950 |
| `torch_grouped_mm` | `torch._grouped_mm` | both |
| `torch_loop` | `torch.mm` (hipBLASLt) per expert | both |
| `hk_ops_moe` | hk's shipped MoE GEMM (`hk.ops.moe.moe_gemm`, decode-sized), where it runs | both |

`proxy:dense` in the tables is the dense hipBLASLt GEMM of the same rows x N x K
(one expert's weights for every row): not the task's math, shown for scale.

**The hidden set.** About a dozen shapes per GPU from the MoE layers of other
real models (not the public ones), in the ranges above, at decode-to-prefill
batch sizes and router skews like the public ones, scored by the same command
after your run. **Pass, per GPU:** correct and audit-clean on every hidden
shape, the geometric mean of the ratios >= 0.95, and no shape below 0.85.
T4 passes when both GPUs pass.

**The public shapes** (`bench/t4/shapes_public.json`, the same twelve on both
GPUs; develop on these):

| shape | E | top_k | N | K | rows |
|---|---|---|---|---|---|
| olmoe-1b-7b-gate_up-t4096 | 64 | 8 | 2048 | 2048 | 32768 |
| olmoe-1b-7b-down-t512 | 64 | 8 | 2048 | 1024 | 4096 |
| mixtral-8x7b-tp4-gate_up-t2048 | 8 | 2 | 7168 | 4096 | 4096 |
| mixtral-8x7b-tp4-down-t4096 | 8 | 2 | 4096 | 3584 | 8192 |
| qwen1.5-moe-a2.7b-gate_up-t512 | 60 | 4 | 2816 | 2048 | 2048 |
| qwen1.5-moe-a2.7b-down-t4096 | 60 | 4 | 2048 | 1408 | 16384 |
| granite-3.1-3b-a800m-gate_up-t1024 | 40 | 8 | 1024 | 1536 | 8192 |
| granite-3.1-3b-a800m-down-t8192 | 40 | 8 | 1536 | 512 | 65536 |
| deepseek-v2-tp8-gate_up-t4096 | 160 | 6 | 384 | 5120 | 24576 |
| deepseek-v2-tp8-down-t1024 | 160 | 6 | 5120 | 192 | 6144 |
| qwen2-57b-a14b-tp8-gate_up-t256 | 64 | 8 | 640 | 3584 | 2048 |
| qwen2-57b-a14b-tp8-down-t2048 | 64 | 8 | 3584 | 320 | 16384 |

## Budget

- **Wall-clock: 10 hours** from your first action to your final message.
- **GPU time: 4.0 GPU-hours on each GPU**, counted as the wall time of every
  command you run through the GPU helpers you are given (compiles included,
  since they run inside them). The helpers keep the ledger and refuse to start
  a command once a GPU's budget is spent. Time spent without a GPU (reading,
  writing, tracing and building kernels with `--dump`/`trace()`/`source()`/
  `build()`) is not GPU time.
- Stop when either runs out, or earlier if you are done. Whatever
  `bench/t4/solution.py` is at that moment is what is scored.

## What you may use

- Everything under `docs/` (the history chapters included), `examples/`,
  `skills/`, `README.md`, `llms.txt`, `llms-full.txt`, the hk package's source
  (`python/hk/`), `tests/`, `tools/` and `bench/` -- **except** two files in
  this directory: `bench/t4/baselines.py` and `bench/t4/triton_grouped.py` are
  the competitors' code; run them through `score.py`, do not read them, and do
  not read the baselines' libraries (aiter, vLLM, CK, the Triton kernels) either.
- The hk tools: `python -m hk.check`, `hk.profile` (`python -m hk.profile`,
  `hk.check --profile`), `python -m hk.tune` and `hk.schedules`, and
  `bench/t4/score.py`.
- Not: the internet, GitHub (issues, pull requests, other branches), git
  history beyond your checkout, other people's directories on the machines, or
  any file about this evaluation that you were not given.

## What to leave behind

- `bench/t4/solution.py`.
- `bench/t4/NOTES.md`, kept as you go: a dated line for every attempt -- what
  you changed, what the tools said (numbers, verdicts, errors), what you
  concluded and what you tried next -- and at the end, your last public score
  on each GPU and what you would do with more time.
- Your last full public-shape score on each GPU (`bench/t4/out/*.json` and `.md`).
