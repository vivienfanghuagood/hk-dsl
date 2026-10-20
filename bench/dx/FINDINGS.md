# What the DX suite found (2026-10-20)

The baseline: eight LLM kernels written by someone who had read only hk-dsl's
documentation, measured against torch eager, `torch.compile`, Triton and the
`hk.ops` kernel where one exists, on an Instinct MI355X (gfx950) and a Radeon
PRO W7900D (gfx1100). The numbers are the committed scoreboards
([gfx950](results/gfx950-2026-10-20.md), [gfx1100](results/gfx1100-2026-10-20.md));
the doc gaps are the `GAP:` lines in each `solutions/<task>/docs_only*.py`,
collected and ranked here.

## How the docs-only solutions were written

- **Who:** t1 by the suite's author, as the first page teaches (and a second
  attempt on register strips); t2-t8 by four fresh coding agents, one per task
  group, each given the task statement and told it may read `README.md`,
  `docs/` (the history chapters included, when a doc links there),
  `examples/`, and `help()` / `dir()` at run time -- and not `python/hk`,
  `include/`, `tests/` or `tools/`. They could not call `hk.ops`.
- **Time:** 1-2.5 hours per task, about half of it reading. Every one of them
  ended up in `docs/history/` chapters for the facts that make a kernel fast,
  and in `help()` for API the docs do not show.
- **So:** these are what a *determined* reader gets from today's docs, not
  what the first page gets you. The first-page version is t1's `docs_only.py`
  (16x64 register tiles): 0.54-1.28x `torch.compile`.

## The scoreboard, docs-only against the baselines

Device time (a CUDA graph of the calls), geomean over the task's shapes;
speedups (> 1 is faster) against `torch.compile` (`x TC`) and against the
fastest of torch eager, `torch.compile`, Triton and hk.ops at each shape
(`x best baseline`); "fastest baseline at" counts the shapes each baseline
won. Per-shape tables: the result files. tests/dx passes on both GPUs.

| task | docs-only solution | lines | gfx950: x TC (min-max) | x best baseline | hk.ops x TC | fastest baseline at (shapes) | gfx1100: x TC (min-max) | x best baseline | hk.ops x TC | fastest baseline at (shapes) |
|---|---|---:|---|---:|---|---|---|---:|---|---|
| t1_gelu_tanh_and_mul | docs_only | 16 | 0.78 (0.54-1.28) | 0.66 | - | triton 6, torch_compile 2 | 0.83 (0.71-0.92) | 0.80 | - | triton 5, torch_compile 3 |
| t1_gelu_tanh_and_mul | docs_only_strips | 20 | 1.11 (0.97-1.60) | 0.95 | - | triton 6, torch_compile 2 | 1.07 (0.98-1.22) | 1.02 | - | triton 5, torch_compile 3 |
| t2_add_rmsnorm_quant | docs_only | 50 | 1.30 (1.09-1.53) | 1.10 | hk_ops 0.97 | triton 6, hk_ops 1, torch_compile 1 | 1.38 (1.01-2.19) | 1.25 | hk_ops 0.69 (7/8) | triton 5, hk_ops 2, torch_compile 1 |
| t3_cross_entropy | docs_only | 116 | 2.68 (1.84-3.56) | 1.86 | - | triton 5, torch_compile 1 | 1.60 (1.32-3.52) | 1.17 | - | triton 6 |
| t4_linear_bias_gelu | docs_only | 106 | 1.14 (0.89-1.42) | 1.02 | hk_ops 0.98 / hk_ops_decode 0.74 (2/8) | triton 2, hk_ops 2, torch_eager 2, torch_compile 2 | 0.90 (0.65-1.26) | 0.58 | hk_ops 1.41 / hk_ops_decode 2.18 (2/8) | hk_ops 5, triton 2, hk_ops_decode 1 |
| t5_linear_residual | docs_only | 136 | 1.15 (0.81-1.58) | 1.02 | hk_ops 1.11 / hk_ops_decode 0.98 (2/10) | hk_ops 7, torch_eager 2, torch_compile 1 | 0.84 (0.67-1.04) | 0.52 | hk_ops 1.44 / hk_ops_decode 3.86 (2/10) | hk_ops 8, hk_ops_decode 2 |
| t6_attn_softcap | docs_only | 213 | 4.10 (2.54-5.65) | 2.29 | - | triton 6 | 2.32 (2.16-2.56), 4/6 shapes | 0.73 | - | triton 6 |
| t7_attn_sliding_window | docs_only | 198 | 3.12 (2.98-3.23) | 0.96 | hk_attention 5.59 (2/5) | triton 3, hk_attention 2 | 1.65 (1.59-1.75) | 0.57 | hk_attention 2.88 (2/5) | triton 3, hk_attention 2 |
| t8_moe_router | docs_only | 60 | 9.36 (6.95-14.52) | 1.16 | - | triton 9 | 7.32 (5.43-9.93) | 1.18 | - | triton 9 |

hk.ops columns: `fused_add_rms_norm` + `quantize` (t2), `matmul` + a torch
epilogue (t4/t5; `decode_gemm` where it takes the shape), `hk.attention` where
it computes the task (t7 at N <= 4096). Tasks without one: t1 (`silu_mul` is not
GELU; it is in the tables as a proxy), t3, t6 (no soft-cap; `hk.attention`
without it is a proxy, a lower bound), t8.

What the numbers say:

- **Row kernels (t1-t3, t8) beat `torch.compile` on both GPUs once written on
  register strips:** t2 1.30x / 1.38x (gfx950 / gfx1100), t3 2.68x / 1.60x,
  t8 9.4x / 7.3x (the compiled router is several launches), t1 1.11x / 1.07x.
  Against the best baseline (Triton, at most shapes) they are 0.95-1.86x. On
  the first page's 16x64 tiles t1 is 0.78x / 0.83x (0.54x at 4096x14336 on
  gfx950).
- **One fused docs-only kernel beats hk's own composition for t2:** 1.35x
  `fused_add_rms_norm` + `quantize` on gfx950. On gfx1100 that composition is
  0.69x `torch.compile` (0.34-0.65x at hidden size 4096; `quantize` refuses a
  single token there).
- **GEMM + epilogue (t4/t5).** gfx950: docs-only wins at M <= 128 (1.26-1.58x)
  and loses at M = 4096 (0.81-0.96x) to hipBLASLt; `hk.ops.matmul` + a torch
  epilogue is no better there (0.84-0.96x). gfx1100: the docs show only the
  single-buffered example GEMM, so docs-only is 0.90x / 0.84x `torch.compile`
  and 0.58x / 0.52x the best baseline, which there is hk's own tuned GEMM
  (1.41x / 1.44x `torch.compile`; the decode GEMV 1.6-4.0x at M = 16):
  hipBLASLt is weak on RDNA, and hk.ops is the yardstick.
- **Attention with a modifier (t6/t7).** gfx950: docs-only 4.10x / 3.12x
  `torch.compile` (flex_attention), 2.3x the Triton tutorial port for t6, and
  0.46-0.67x `hk.attention` where that computes t7 exactly. gfx1100: the
  docs-only one-warp RDNA body is 2.32x / 1.65x `torch.compile` but 0.73x /
  0.57x the Triton port, which there ties `hk.attention` (2.87x vs 2.88x at
  N <= 4096); head dim 256 does not fit the register file at all (spills), so
  t6 runs 4 of its 6 shapes.
- **Against G3's bar** (>= 0.9x `torch.compile`): the best docs-only solution
  is at or above it at every shape of 6 of 8 tasks on gfx950 (not t4: 0.89 at
  M = 4096; not t5: 0.81) and of 5 of 8 on gfx1100 (not t4, t5, and t6, which
  declines head dim 256). These were written by agents spending 1-2.5 hours
  each, deep in the history chapters -- an upper bound on what the docs give
  today, not the fresh-agent result G3 asks for.

## The doc gaps, ranked by what they cost

Each is evidence from a solution's DX log (task named) and, where it is a
measurement, the number.

1. **What makes a kernel fast is not in the docs proper.** The examples are,
   in their own words, "the shapes of the kernels, not the tuned versions";
   for everything past them `examples/README.md`, `04_gemm.py` and
   `05_attention.py` say "read `hk/ops/gemm.py` / `attn.py` next" -- library
   source. A docs-only reader assembles a GEMM pipeline, an LDS-shared
   attention, a split-K or a per-shape tiling from one-sentence mentions in
   `docs/history/27, 28, 29, 33, 34, 37, 41`. Cost: 30-60 min of reading per
   task (t4, t6), ~40 configurations swept by hand for tile shapes (t4), and
   the remaining gaps: GEMM 0.81-0.96x `torch.compile` at M = 4096 on gfx950,
   attention 0.46-0.67x `hk.attention` (t4, t5, t6, t7). On gfx1100 no doc or
   example shows a double-buffered GEMM at all, so the docs-only GEMM there is
   `04_gemm.py`'s single-buffered one.
2. **The first page's tiling is the slow one, and nothing says so.**
   getting-started's `add` (16x64 register tiles, one warp) is 0.54x
   `torch.compile` at 4096x14336 on gfx950 (t1 `docs_only.py`); register strips,
   which `language.md` introduces as the layout for kernels that "only stream
   rows", are 1.03x -- but need the wave size (32 or 64) as a per-architecture
   constant, and the docs give no rule for warps or columns per workgroup
   (t1, t2: the shipped rule from history/37 was 25% slower at decode sizes).
3. **No catalogue of the math.** `language.md` says "`hk.math`: elementwise
   functions". `dir(hk.math)` is exp, exp2, log, log2, abs, relu, gelu, dgelu,
   sqrt, rsqrt, silu, add, sub, mul, div, maximum, minimum: no tanh, sigmoid,
   erf, reciprocal, round or clamp. `gelu` is the tanh form and fp32-only,
   neither documented (`help(hk.math.gelu)` says "kittens::gelu"). Soft-capping
   needs tanh, built as `cap - 2*cap*rsqrt(E+1)**2` (t6); int8 quantization
   needs round/clamp, which turn out to happen when an fp32 strip is stored
   into an i8 tensor (t2) -- a guess that worked.
4. **Statements about gfx950 that are false.** `language.md` "Limits": "On
   gfx950, hk.ops' GEMM and attention have no native form yet";
   `examples/README.md`: "attention has no CDNA path yet and is refused while
   tracing, and validate.py skips the GEMM"; history/34: "05_attention.py is
   still skipped on gfx950". README and design/cdna.md say the opposite, and
   `05_attention.py --causal` runs on an MI355X at 167 TFLOP/s (t4, t6).
   `help(hk.RS)` says a strip's chunk is `32*run` columns: it is 64*run on
   gfx950 (t2, t3). `help(Tile.step)` says "Tiles only": it works on strips (t8).
5. **On RDNA a kernel is launched on the null stream, invisibly.** `k(...)`,
   as the first page teaches, launches on the null stream on gfx1100 (one
   paragraph of `runtime.md`, "The stream"). torch's events and a CUDA graph
   do not see it: the suite first timed `hk.ops.silu_mul` at 0.16 us replayed
   against 3.5 us eager, and t2's agent saw a "device time" of 0.1-2.2 us and
   host time above the card's bandwidth -- no error anywhere. In a serving
   engine's captured decode step such a kernel would not run at all. The fix
   (`hk.torch_op(k)`, or `HK_LAUNCH=codeobj`) is in `runtime.md` only (t2, t8,
   and the runner, which now sets `HK_LAUNCH=codeobj` on RDNA).
6. **RDNA rounds differently, and only the history says so.** In registers,
   `t.to(hk.bf16)` rounds ties away from zero on gfx1100 and to even on
   gfx950: t2's updated residual was one ulp off on 10.7% of elements on
   gfx1100 only, until `defines={"HK_BF16_ROUND": 2}`, which history/18 and
   history/38 mention and `language.md` (which documents `defines`) does not.
7. **API the solutions needed that only `help()` / `dir()` show:**
   the bounded *store* spelling (`o[idx].store(v, bounded=True)`; only bounded
   loads are shown; t1, t6); one value per row out of a strip (`store` and
   `store_scalar` raise errors that send you to each other; the way out,
   `hk.as_vec(v, S, uniform=True)`, is an unchecked assertion; t2, t3);
   multi-buffered LDS (`hk.ST(..., count)` and `hk.lds.alloc(count=)`, two
   spellings; t4, t6); `triu_` / `tril_` / `right_fill_` / `left_fill_` (no
   docstrings; `triu_(d, v)` fills where `row - col > d`, the opposite sign of
   numpy's -- probed with a scratch kernel; t3, t7, t8); `hk.full`;
   `Reduction.value()`; `load_scalar` reading int32 only (no way to read one
   bf16 / fp32 element; t3); no int64 (labels; t3); no reflected `-` or `/`
   (`1.0 - t`, `127.0 / amax` raise; t2, t8); no top-k, argmax, iota, compare
   or select (t8 builds them from `lane_id()`, `right_fill_` and `step()`).
8. **No hooks on the shipped kernels.** `hk.attention` takes no score or mask
   modifier, and `hk.ops.matmul` no epilogue, so a soft-cap, a window, a bias +
   GELU or a residual means writing the whole kernel (t4-t7). This is G2's
   template work.
9. **Undocumented rules met as errors** (the errors themselves were good):
   a DMA copy (`g.load(wait=False)`) must be a whole number of 16-byte copies
   per thread, which rules out 256x224 blocks (t4); register tiles cannot be
   bounded by columns, so the block width must divide N (t4); a pipelined loop
   must not end with a copy in flight (t4); 8 warps cap a wave at 256 VGPRs, and
   `s_setprio` decided whether a 256x256 kernel spilled (t4, t6).
10. **Advice that is wrong for this workload.** history/41's non-temporal
    weight streaming was 20-35% slower here (inputs hot in the MALL; t4);
    chapter 11's `wg_arrive` one-launch split is 8-15x slower than two launches
    on gfx950 (t3), with no warning.
11. **Which version has what.** The docs describe main, which is ahead of the
    PyPI 0.4.0 wheel (gfx950 strips are refused by 0.4.0), and main also says
    `hk.__version__ == "0.4.0"`: a reader cannot tell which features they have.
12. **No way to see the ISA.** t8's agent extracted the disassembly by hand
    (llvm-objcopy + clang-offload-bundler + llvm-objdump on `Build.so_path`) to
    find 398 `s_nop`s in 1108 instructions; nothing documents how.

## What the baselines found about hk itself

- **The same op takes different arguments on the two GPUs.** `hk.ops.silu_mul`
  takes a gate/up output's strided halves on gfx950 and refuses them on gfx1100
  (copied to contiguous tensors in setup there), and on gfx1100 refuses a
  decode batch of one ("a is 1x14336 and silu_mul needs at least 16x64"),
  which gfx950 takes.
- **`hk.ops.matmul` does not route decode batches to the decode GEMV on
  gfx1100** (it does on gfx950); `hk.ops.decode_gemm.matmul` called directly
  is faster at M = 16 there (see t4/t5's `hk_ops_decode` rows).
- **hk's two-kernel add + norm + quant** (`fused_add_rms_norm` then `quantize`)
  rounds the normed row to bf16 between the kernels; its q is up to 1.27 steps
  from `y / scale` (the check allows 2, as vLLM's own kernel rounds twice), and
  it is 0.97x `torch.compile` on gfx950 -- one fused docs-only strip kernel is
  1.35x it.
- **Head dim 256 has no hk attention** (`HEAD_DIMS = (64, 128)`): Gemma-2-9B's
  attention is out of reach of `hk.attention` even without the soft-cap. The
  docs-only kernel runs it on gfx950 (4 warps; 8 spill), not on gfx1100 (the
  one-warp body spills).

## The suite's own caveats

- Inputs stay resident between calls (hot L2 / MALL): decode shapes are
  measured as they run inside a layer loop; t2's agent suspects its 1.8-2.3x
  at H = 5120 / 8192 on gfx1100 is partly that.
- `torch.compile` for t6/t7 is `flex_attention` (what torch offers for a
  modified attention); HF's eager Gemma-2 attention, scores in bf16, is 7% off
  at these logits and fails the check, so the eager baseline keeps the scores
  in fp32.
- On gfx1100 hk kernels are timed through the code-object launch (see 5), which
  adds a Python launch to their `host_us`; `device_us` is unaffected.
- gfx1100 is shared by every stream and the CI: its runs were one task per
  call, with warm caches (gfx950's `first_call_s` are cold compiles).
- The docs-only agents saw the task's `reference.py` and could iterate against
  the check; a user would have only `task.md`.
