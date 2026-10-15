# Against Triton and FlyDSL: the scoreboard, and what it moved

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

> The opponents here were under-tuned: Triton ran the tutorial's seven GEMM
> configs and FlyDSL its bare 128x128x32 default rather than its own
> dispatcher. Against FlyDSL's dispatcher the GEMM loses; [chapter 26](26-the-scoreboard-retuned.md)
> re-measures every row with each DSL tuned.

The goal this chapter serves is that hk's kernels be no slower than the
faster of Triton and FlyDSL at the shapes an LLM runs, on this card. That
needs a scoreboard first.

**FlyDSL on gfx1100.** ROCm/FlyDSL (0.4.0.dev970) builds for gfx1100 and runs
in the same process as hk -- the two MLIRs do not collide. What runs here is
its RDNA3 WMMA GEMM, rmsnorm, fused add + rmsnorm and softmax. Its flash
attention does not run: it stages K/V through `raw_ptr_buffer_load_lds`, a
global-to-LDS load that gfx11 does not have (`tools/hk-bench/flydsl_probe.py`).

**The harness.** `tools/hk-bench/scoreboard.py` runs each operation at 8B
shapes through every way there is to compute it, in one process:
- hk.ops;
- torch eager (hipBLASLt, aotriton, ATen);
- torch.compile, which is the Triton vLLM runs for its norms;
- a Triton kernel as a Triton user writes it (`triton_ref.py`, the tutorial
  kernels with their own autotune);
- FlyDSL's own kernel.

Every arm is checked against fp32 before it is timed. GPU time comes from the
profiler's kernel records, attributed by kernel name. Attributing by profiler
range does not work on ROCm: the correlation of anything launched outside
ATen is unreliable there, and it once credited a GEMM with 347 TFLOP/s on a
123 TFLOP/s card.

The first full run (hk against the fastest other arm):
- the GEMM at 0.90-0.96x;
- attention at 2.7-3.7x aotriton;
- the norms at 0.08-0.77x;
- softmax at 0.46-0.85x.

**The norms a model runs.** The losing rows were ops hk did not have:
- `hk.ops.rms_norm(x, weight)`;
- `hk.ops.fused_add_rms_norm(x, residual, weight)`, which is in place, like
  vLLM's.

Four things moved them, each measured:
- **Fold below 4096 rows.** One workgroup per row, the row's sixteen chunks
  as a tile's rows, and the weight folded the same way. A weight tile loaded
  at the data tile's coordinates is then the weight of every element in it.
  Above 4096 rows the unfolded hold wins, and the planner switches.
- **Widen in registers.** A bounded bf16 load into an fp32 tile went element
  by element. HipKittens' `load_bounded` now takes `load()`'s widening path:
  the lane's 16-element run in two b128 loads, then its own entries of it.
  Loading bf16 tiles and `copy()`ing them was the workaround, and it moves
  every element across the wave halves -- 0.86 us of a 3.5 us one-row kernel
  (`tools/hk-bench/norm_stages.py`).
- **Straight-line vector LDS access.** An `align_l` register vector looped
  over sixteen entries and skipped, by lane, the ones its half does not
  hold. That made one divergent branch per entry, each a flat load through
  a generic pointer -- sixteen serialized round trips in a cross-warp sum.
  It now loops over the slots a lane holds, through an `address_space(3)`
  pointer (fork 4d7ea9fb).
- **The packed store's exchange on the VALU.** It was `__shfl_xor`, which is
  `ds_bpermute` plus a wait each. Now it is `v_permlanex16`: storing one
  16x64 fp32 tile as bf16 went from 32 bpermutes and 52 waits to none and 21
  (fork 9542a0f4).

The residual is rounded ties-to-even (`HK_BF16_ROUND=2`, through a new
per-kernel `defines`). A sum of two bf16s is exact in fp32, so its tie bit is
set about one time in ten, and ties away from zero mismatched torch on 10.7%
of elements. Rounded like torch, it is bit-identical.

**The GEMM, which was one buffer away.** Triton's autotune picks the same
tile as hk at every shape -- 128x128x64, 8 warps, 2 stages -- and was 9%
faster. The difference was in the resource lines:

| | VGPRs | occupancy | LDS | workgroups per WGP | waves per SIMD |
|---|---|---|---|---|---|
| Triton | 168 | 9 | 32 KB, one buffer | 4 | 8 |
| hk | 211 | 7 | 64 KB, two buffers | 2 | 4 |

A WGP has 128 KB of LDS, so hk's two buffers -- the next K-tile written while
this one is read, one barrier a tile -- capped it at two workgroups. One
buffer costs a second barrier per tile and lets three in
(`gemm_kernel(lds_buffers=1)`). TFLOP/s (`tools/hk-bench/gemm_variants.py`,
one process, every arm checked):

| shape | two buffers, k64 | one buffer, k64 | two, k32 | one, k32 | Triton |
|---|---|---|---|---|---|
| 4096x4096x4096 | 74.5 | 85.9 | 74.5 | 76.1 | 82.1 |
| 2048x6144x4096 | 76.5 | 85.8 | 75.3 | 75.3 | 80.2 |
| 2048x4096x4096 | 74.4 | 83.4 | 71.7 | 71.5 | 77.9 |
| 2048x24576x4096 | 77.2 | 86.9 | 75.1 | 72.3 | 78.1 |
| 2048x4096x12288 | 74.5 | 84.0 | 71.9 | 70.8 | 76.0 |

One buffer at k64 is now the default. The shipped tuning record for
4096x4096x4096, k_step 32 at a 1.2% gain over the old default, is withdrawn:
it beat a default that no longer exists, and under one buffer it is the
slower choice. The 64x64 warp tiles were tried again too (256x128 and
128x256 blocks), and every one still spills.

**Softmax, online.** The old kernel had three passes and two exps per
element: the max, the sum of exp, and exp again to write. `v_exp_f32` is
quarter rate on gfx1100. Online, a tile held in registers becomes its own
exp in place and is scaled once at the write, so there is one exp per
element; streamed, it is two passes over memory instead of three. The write
stores fp32 and narrows in registers. That moved 2048x4096 from 0.56x to
0.63x and 8192x4096 from 0.91x to 0.93x. It also removed the reason softmax
alone folded first: unfolded is now the faster plan for it as for the others
(8192x4096: 0.186 against 0.198 ms). So the planner folds by row count, for
every op, and no longer by kind.

**The scoreboard now** (GPU ms, hk against the fastest other arm):

| case | hk | torch | Inductor | Triton | FlyDSL | hk vs best |
|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.612 | 1.999 | -- | 1.687 | 1.743 | 1.05x |
| gemm 2048x6144x4096 | 1.202 | 1.613 | -- | 1.300 | 1.306 | 1.08x |
| gemm 2048x4096x4096 | 0.837 | 1.043 | -- | 0.903 | 0.930 | 1.08x |
| gemm 2048x24576x4096 | 4.847 | 5.944 | -- | 5.090 | 5.075 | 1.05x |
| gemm 2048x4096x12288 | 2.490 | 2.958 | -- | 2.645 | 2.571 | 1.03x |
| attention n4096 causal, GQA 4 | 2.994 | 10.924 | -- | -- | -- | 3.65x |
| attention n4096, GQA 4 | 4.823 | 13.240 | -- | -- | -- | 2.75x |
| rms_norm 1x4096 | 0.0057 | 0.0044 | 0.0025 | 0.0026 | 0.0023 | 0.41x |
| rms_norm 32x4096 | 0.0048 | 0.0044 | 0.0026 | 0.0026 | 0.0023 | 0.47x |
| rms_norm 2048x4096 | 0.0274 | 0.0264 | 0.0196 | 0.0193 | 0.0191 | 0.70x |
| rms_norm 8192x4096 | 0.1949 | 0.1812 | 0.1810 | 0.1807 | 0.1791 | 0.92x |
| fused add + rms_norm 1x4096 | 0.0074 | 0.0080 | 0.0027 | 0.0031 | 0.0032 | 0.37x |
| fused add + rms_norm 32x4096 | 0.0051 | 0.0063 | 0.0026 | 0.0026 | 0.0028 | 0.51x |
| fused add + rms_norm 2048x4096 | 0.0505 | 0.0537 | 0.0436 | 0.0438 | 0.0394 | 0.78x |
| fused add + rms_norm 8192x4096 | 0.3560 | 0.4621 | 0.3664 | 0.3460 | 0.3831 | 0.97x |
| softmax 2048x4096 | 0.0305 | 0.0313 | 0.0582 | 0.0261 | 0.0192 | 0.63x |
| softmax 8192x4096 | 0.1893 | 0.1775 | 0.2271 | 0.1781 | 0.1757 | 0.93x |

Before this chapter, rms_norm was 0.11-0.77x and the fused add 0.08-0.43x.
The fused add reached 1.00-1.02x at 8192 rows in two runs and 0.97x in this
one, which is the noise at that size.

Lines per kernel (no docstrings, comments or blanks; FlyDSL's are whole
files, helpers included):

| kernel | hk | Triton | FlyDSL |
|---|---|---|---|
| GEMM | 124 | 30 | 402 |
| attention | 127 | -- | 181 |
| rms_norm + fused add | 63 | 14 | 1784 |
| softmax | 25 | 10 | 21 |

**What is left, and why.** The gate is not met for the norms and softmax:
- **Small M is fixed latency, not bandwidth.** A one-row norm is 8 KB. hk's
  kernel is a 16x64 fp32 tile per warp: a longer instruction stream, through
  more conversion and packing, than Triton's one-dimensional block. On the
  same row, the floor of an hk tile kernel -- load a tile, store it -- is
  2.6 us where Triton's whole RMSNorm is 3.0 (`norm_stages.py`, noisy by
  about a microsecond at this size).
- **Mid-size softmax** loses to FlyDSL's 1.75 TB/s from the Infinity Cache.
  hk spends two LDS round trips with barriers per reduction (cross-warp,
  then fold), twice.
- **At 8192 rows** every op is within 3-8% of the fastest. That is
  bandwidth, and hk's packed 16-bit store is the likely remainder.
