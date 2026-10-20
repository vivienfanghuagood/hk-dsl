# gfx950's last rows: arguments the compiler could not see through, a floor that follows the buffer, and a RoPE of eight-byte runs

Chapter 41 left gfx950's norm, elementwise and fused rows at 120 of 132
cases tying or beating the fastest of torch, Inductor, Triton and FlyDSL,
each tuned. Re-measured on this tree before any change (`scoreboard.py
--only llm-norm,llm-quant,llm-pointwise --rounds 7 --calls 50`, one idle
MI355X) it was 117: the cases at 1-32 rows and at the edges of the 2% band
move between runs. Fifteen losses, 0.95-0.98x:

- six at the one-row floor (3.89-3.94 us against 3.80-3.83);
- the fused add + RMSNorm at 2048x4096 and 2048x8192 (0.974, 0.953);
- layernorm 8192x8192, softmax 2048x8192, quantize 2048x8192 and 8192x4096
  (0.974-0.978);
- RoPE at 2048x40, 2048x64 and 8192x40 heads (0.961, 0.968, 0.950).

Now, with the arms' output buffers rotated (below), 131 and 128 of 132 tie
or win in two runs. The losses are one- and 32-row cases at 0.978-0.979x
against a 2.0% band, different ones each run, and in one run RoPE at
8192x40, 0.977x (0.983x and 0.985x in the other two runs on this tree).

## The floor is the card's, and it follows the buffer

Kernels from `hipModuleLaunchKernel` through the scoreboard's own timing
(`gpu_rounds`: queued behind a spin, kernel records, median of 15 rounds of
50 calls):

| kernel | us |
|---|---|
| empty | 3.80-3.81 |
| 1x8192 add, three pointer arguments, 64-1024 threads a block | 3.86-3.88 |
| the same, the arguments a 144-byte struct | 3.87-3.89 |
| hk's add, at each of its seven plans | 3.87-3.90 |
| torch's add | 3.89 |

So a one-row kernel is the empty kernel plus 0.05-0.1 us, whatever its
launch path, argument form or plan. Which arm gets the 0.1 depends on where
its buffers sit. The scoreboard gives every arm its own output; built as it
builds them, with the final tree, each arm on its own buffer and on the
other's (`tools/hk-bench/floor_placement.py`), us:

| case | hk, own | other, own | hk on the other's | other on hk's |
|---|---|---|---|---|
| rmsnorm 32x4096 (FlyDSL) | 3.905 | 3.795 | 3.805 | 3.912 |
| add 32x8192 (torch) | 3.900 | 3.837 | 3.793 | 3.909 |
| mul 1x5120 (torch) | 3.886 | 3.868 | 3.870 | 3.886 |
| add 1x8192 (torch) | 3.879 | 3.865 | 3.869 | 3.870 |
| exp 1x5120 (torch) | 3.867 | 3.877 | 3.877 | 3.877 |

The process matters too: one had hk and torch at 3.89 on every buffer of
add 1x8192, the next at 3.80. Carving the output out of one allocation at
seven offsets moved hk 3.78-3.88 and torch 3.80-3.87.

`scoreboard.py --rotate-buffers` passes the output buffers one arm along
each round (`buffer_rotation`, from the per-arm dicts the cases' closures
read at call time), so every arm runs on the same buffers and the round
spread carries the placement. Of the 66 one- and 32-row cases, two lost
in the first rotated run (the fused add at 32x4096, 0.969x, whose FlyDSL
arm keeps the buffers it was built with; mul 1x8192, 0.977x), none in the
second, one in the third and three in the fourth (0.978-0.979x). exp
1x5120 went 1.015, 1.000, 0.997 and 0.979x over the first four runs, its
rounds 3.785-3.924 us for hk and 3.773-3.962 for torch in the fourth.

## Arguments the compiler could not see through

On gfx950 a strip kernel was `void k(const globals g)`: five `strided_gl`s
for the fused add, 240 bytes passed by reference into the kernarg segment.
Two things follow. LLVM cannot preload a by-reference aggregate into SGPRs
(its kernarg preload stops at the first one), and the pointers inside it
cannot be `__restrict__`, so a store through `o` may alias any later load
through `x`. At 128 elements a lane -- tuned Triton's pick for the
2048x8192 fused add, one warp a row -- the struct form took 230 VGPRs.

The same generated body with the fields as scalar arguments (pointers
first), compiled by hand and launched by `hipModuleLaunchKernel`, three
processes, us:

| form | 2048x8192 fused add |
|---|---|
| hk's op (pybind, struct) | 19.47-19.84 |
| its code object by hand (struct) | 19.37-19.78 |
| flat arguments | 19.58-19.97 |
| flat, `__restrict__` | 19.34-19.37 (164 VGPRs) |
| flat, `__restrict__`, preloaded | 19.00-19.02 |
| tuned Triton (`num_warps=1, waves_per_eu=2`) | 19.37-19.38 |

The pybind entry point costs nothing; `__restrict__` is the change, and
preload helps only on top of it. On CDNA a kernel declared
`hk.kernel(..., noalias=True)` now takes its globals' fields as scalars:
HKToCpp emits it as `hk_flat::<name>_kernel(T *__restrict__ hk_x, ...,
int hk_x_rs, ..., int hk_x_r, ..., hk_x_c, hk_x_b, hk_x_d ...)` -- every
pointer, then every row stride, then rows, cols, batch, depth -- rebuilds
`g` from them (`strided_gl`'s constructor is now `__host__ __device__`), and
`launch_on` passes them. The device scaffold's layout table describes the
parameters (`hk_kernarg`, the struct HKToCpp emits beside the kernel), so
the code-object launcher packs them as it packed `globals`; it finds the
mangled symbol in the code object's symbol table. `noalias` says no tensor
is written where another is read except by the lane that read it, after it
did -- an in-place call is that -- and every strip kernel in `hk.ops` (the
norms, softmax, quantize, the elementwise ops, silu_mul, both RoPEs)
declares it. On RDNA nothing changes: the attribute is not set there, and
gfx1100's and gfx1201's generated sources are identical for every kernel
(`tools/cdna-probes/dump_sources.py`; gfx950's 494 strip kernels changed).

## Preload, past 32 MB

`Kernel.preloaded()` is the kernel built with
`-mllvm -amdgpu-kernarg-preload-count=16` (`Target.kernarg_preload`, on for
gfx950): the CP writes the flat arguments into SGPRs before the waves start.
It has a cost: an empty kernel is the same 3.80 us either way, but a one-row
fused add went 3.89 -> 4.08 us and a raw add 3.86 -> 4.05, and Triton, which
preloads its kernels on gfx950, sits at 4.0-4.1 at one row. Per scoreboard
case, the planner's plan without and with (`tools/hk-bench/preload_ab.py`,
one process, buffers rotated), us:

| case | moves | plain | preloaded |
|---|---|---|---|
| quantize 2048x4096 | 24 MB | 4.77 | 4.93 |
| rms_norm 2048x4096 | 32 MB | 5.40 | 5.66 |
| exp 2048x4096 | 32 MB | 5.23 | 5.47 |
| RoPE 2048x32 | 32 MB | 5.29 | 5.42 |
| rms_norm 2048x5120 | 40 MB | 6.66 | 6.44 |
| exp 2048x5120 | 40 MB | 6.57 | 6.15 |
| RoPE 2048x40 | 40 MB | 6.26 | 6.02 |
| add 2048x4096 | 48 MB | 7.30 | 6.95 |
| quantize 2048x8192 | 48 MB | 8.59 | 8.37 |
| layernorm 2048x8192 | 64 MB | 10.44 | 9.99 |
| layernorm 8192x4096 | 128 MB | 20.04 | 19.79 |

At 8192 rows every op gained 0.4-1.8% (RoPE at 40 heads lost 1.4% on its
old plan); the fused add gained nothing anywhere (0.997-1.005x), silu_mul
nothing. The line is the eight XCDs' 32 MB of L2: the norms, quantize, the
elementwise ops and RoPE take the preloaded twin once a call moves more
than 32 MB (`norm.STRIP_PRELOAD_PAST`); the fused add keeps its plain
build.

## The fused add: a warp a row

With the flat form the per-lane plans were measured again
(`STRIP_MAX_PER_LANE_WAVE64` now lets the fused add hold 128 a lane), us,
one process:

| 2048 rows | 16 | 32 (was) | 64 | 128 | tuned Triton |
|---|---|---|---|---|---|
| 4096 | 7.93 | 7.79 | 7.73 | (64) | 7.60 |
| 8192 | 20.17 | 20.41 | 19.81 | 19.17 | 19.40 |

At 2048 rows the fused add now holds 64 a lane at 4096 columns and 128 at
8192: one warp a row, as Triton's pick. Preloading these moved neither
(7.75, 19.21).

## RoPE: eight-byte runs

The 8192x40 RoPE was the widest gap. It is not the angles: the 16-byte
kernel with its angle loads replaced by constants still took 25.16 us
against Triton's 24.26. Nor where x sits: after spacers of 0-62 MB hk took
24.74-24.83 and Triton 24.25-24.31. It is the shape of the work: Triton's
pick gives a 512-column group of a token two warps of 4-byte accesses,
where hk gave it one warp of 16-byte ones. `rotary16` now takes the run as
a constexpr; at RUN 4 a 128-wide head is 32 lanes and its halves 16 apart,
one `v_permlane16_swap` (`rotate_half`). Rotated buffers, one process, us:

| plan | 8192x40 | 2048x64 | 8192x64 |
|---|---|---|---|
| RUN 8, one warp a group (was, at 40 heads) | 24.89 | -- | -- |
| RUN 8, the shipped plan | -- | 10.44 | 39.65 |
| RUN 8, the shipped plan, preloaded | -- | 10.20 | 39.33 |
| RUN 4, two warps a group | 24.72 | -- | -- |
| RUN 4, two warps a group, preloaded | 24.53 | -- | -- |
| tuned Triton | 24.29 | 10.19 | 39.08 |

A token that does not split into 4, 8 or 16 warps of two chunks (40 heads)
now runs two warps of eight-byte runs a group; every RoPE plan is
preloaded past 32 MB.

One thing about RoPE stays unexplained. Interleaved with Inductor's arm,
with the buffers rotated, hk's 8192x40 rounds took 25.36-25.55 us on two of
three buffers and 24.7 on its own, while Triton's stayed at 24.2; without
the Inductor arm, or without rotation, every round was 24.7. No other case
showed it.

## The scoreboard now

`--rotate-buffers`, 7 rounds of 50 calls, the verdict a tie within the
larger of 2% and the two arms' noise (columns: hk over the fastest other
arm, tuned level):

| op | 1-32 rows | 2048 rows | 8192 rows | verdicts |
|---|---|---|---|---|
| rms_norm | 1.00 | 1.02-1.14 | 1.00-1.02 | 7 tie, 5 win |
| rmsnorm | 1.00 | 1.03-1.07 | 1.00-1.01 | 9 tie, 3 win |
| layernorm | 1.00-1.05 | 1.02-1.22 | 0.99-1.02 | 5 tie, 7 win |
| fused add + RMSNorm | 0.99-1.00 | 0.99-1.01 | 1.00-1.01 | 12 tie |
| softmax | 0.99-1.00 | 0.99-1.04 | 1.02-1.04 | 9 tie, 3 win |
| int8 quantize | 1.04-1.05 | 0.99-1.04 | 0.99-1.04 | 3 tie, 9 win |
| add | 0.99-1.01 | 1.00-1.04 | 1.00-1.01 | 11 tie, 1 win |
| mul | 0.99-1.01 | 1.00-1.05 | 1.00-1.01 | 11 tie, 1 win |
| exp | 0.98-1.02 | 1.04-1.06 | 1.01-1.02 | 1 loss, 8 tie, 3 win |
| silu_mul | 1.04-1.06 | 1.00-1.03 | 1.10-1.12 | 2 tie, 10 win |
| RoPE | 1.03-1.06 | 0.99-1.05 | 0.98-1.01 | 4 tie, 8 win |

131 of 132 (50 win, 81 tie). A second rotated run gave 128: the fused add
at 32x4096, mul and exp at 1x8192 (0.978-0.979x) and RoPE 8192x40 (24.70
against 24.13 us, 0.977x). Without `--rotate-buffers`, as chapter 41
measured, the same tree gives 126: all 66 cases past 32 rows tie or win
(the lowest RoPE 8192x40 at 0.985x), and the six losses are one- and 32-row
cases at 0.972-0.980x, each arm on the buffer it was given. Across the
three runs ten cases lost once or twice, nine of them at the floor. Chapter
41's twelve, in the first rotated run, hk against the fastest other arm,
us:

| case | hk | other | ratio |
|---|---|---|---|
| rmsnorm 32x4096 | 3.81 | 3.82 (FlyDSL tuned) | 1.003 |
| fused add 32x4096 | 3.91 | 3.93 (FlyDSL) | 1.005 |
| add 1x8192 | 3.88 | 3.88 (torch) | 1.000 |
| mul 1x5120 | 3.89 | 3.86 (torch) | 0.992 |
| exp 1x5120 | 3.88 | 3.80 (torch) | 0.979 |
| fused add 2048x4096 | 7.75 | 7.65 (tuned Triton) | 0.987 |
| fused add 2048x8192 | 19.17 | 19.38 (tuned Triton) | 1.011 |
| layernorm 8192x4096 | 19.86 | 19.67 (tuned Triton) | 0.990 |
| layernorm 8192x8192 | 37.94 | 38.13 (tuned Triton) | 1.005 |
| quantize 2048x8192 | 8.38 | 8.32 (tuned Triton) | 0.993 |
| RoPE 2048x64 | 10.27 | 10.16 (tuned Triton) | 0.990 |
| RoPE 8192x40 | 24.61 | 24.20 (tuned Triton) | 0.983 |

## What did not work

- **Preload at the floor**: 0.19-0.2 us onto every one- and 32-row kernel,
  which is what Triton's one-row arms pay; hence the 32 MB line.
- **Preload without `__restrict__`**: flat arguments alone ran as the
  struct (19.58-19.97 against 19.47-19.84 us), and preloading them
  changed nothing (19.65-20.01).
- **Preload on the fused add**: 0.997-1.005x at every shape.
- **Counter profiles of the fused add** (rocprofv3): collecting counters
  serializes the dispatches, and serialized the ranking reversed -- hk 20.8
  us against 21.6 for Triton at `num_warps=8, waves_per_eu=4` -- so the
  comparisons above were made queued.
- **RoPE plans at 8192x40 other than two warps of eight-byte runs**: RUN 8
  at two warps of a chunk 26.16-26.38 us, one warp of two chunks
  26.36-26.64, five warps 25.38-25.59, ten 25.18-25.35; RUN 2 at four warps
  a group 37.6.
- **Angle loads**: removing them from the 16-byte RoPE saved 0.3 of the
  0.9 us gap.

## Tests

lit: `Target/flat.mlir` (the flat signature, `hk_kernarg`, the rebuilt `g`
and the launch on gfx950; `globals` without the attribute and on gfx1100).
`tests/hk/ir`: the strip kernels declare `noalias`, the flat source and its
mangled-prefix symbol on gfx950, the unchanged module on gfx1100, the layout
table, which builds ask for preload, the 32 MB line and the fused add's
plans, RoPE's new plans, `buffer_rotation`. `tests/hk/codegen`: every strip
kernel builds flat for gfx950 without spilling, the ISA checks find the
flat symbol, and the preloaded twin preloads its pointers. `tests/hk/gpu`: a
flat kernel and its preloaded twin as code objects write what the pybind
module writes. Every kernel a test builds is a pybind module, each takes one
of a process's 1024 pthread TSS keys, and with the strips built twice past
32 MB the one process that ran the rest of the tier ran out of them
(`thread_specific_storage constructor: could not initialize the TSS key!`
at test_rms_norm_weighted); `scripts/gpu_suite.sh` now runs it as two. The
gfx950 tier: 913 passed, 150 skipped.

## Reproduction

```
# one idle MI355X, rocm/pytorch 2.10 image, cp312 hk._mlir over python/hk/_mlir
export FLYDSL_ROOT=/path/to/FlyDSL HK_ARCH=gfx950
python3 tools/hk-bench/scoreboard.py --only llm-norm,llm-quant,llm-pointwise \
    --rounds 7 --calls 50 --rotate-buffers --json rows.json
python3 tools/hk-bench/floor_placement.py "add 32x8192:torch" "exp 1x5120:torch"
python3 tools/hk-bench/preload_ab.py --rows 2048,8192 --hidden 4096,5120,8192
```
