# Strips: a register layout for rows, and kernel arguments read too late

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 26](26-the-scoreboard-retuned.md) ended with every row kernel
losing to tuned Triton or FlyDSL: RMSNorm 0.48-0.91x, the fused add +
RMSNorm 0.44-0.97x, softmax 0.83-0.94x. It named two causes and left a
third open:
- hk's register tiles have WMMA's lane layouts, and a row kernel wants a lane
  to own a contiguous run of one row;
- at 8192 rows the weight was read element by element;
- one-row hk timings moved between 3.7 and 5.3 us for no known reason.

This chapter adds the missing layout to the language, end to end, rewrites
the norms and softmax on it, and finds the third cause, which was not in the
kernels' bodies at all.

## The layout

`hk.RS(dtype, rows, cols, run=8)` is a register strip:
`!hk.rt<f32, 1x1024, strip<8>>` in the dialect, `kittens::rs<float, 1,
1024, 8>` in HipKittens. Lane l owns, of every row, the runs

    [j*32*run + l*run, j*32*run + l*run + run)     j = 0 .. cols/(32*run) - 1

A chunk of 32*run columns is one contiguous stretch of the row, so a load of
it is one fully coalesced access per lane: a bf16 run of 8 is one
`global_load_b128`, and nothing is held twice. It is Triton's blocked layout
for one warp, and what FlyDSL's tiled copies build.

Its vectors follow. The col_vec, one value per row, is `rs_cv<T, rows>`, held
by every lane: a row reduction folds the lane's runs as a tree, then a
butterfly across the wave (DPP `row_xmask` 1, 2, 4, 8 and `v_permlanex16`),
which leaves the same bits in every lane, since each stage combines a value
with its partner's and the partner computes the same pair the other way
round. The row_vec, one value per column, is `rs<T, 1, cols, run>`, the
strip's own column layout, so a weight vector loads with the same 16-byte
accesses as the rows it scales. That is the 8192-row weight fixed in passing.

Where it lives:
- **HipKittens** (`include/rdna3`): `types/register/rs.cuh`,
  `ops/warp/register/strip` (maps, conversions, row and column reductions,
  the `*_row` and `*_col` broadcasts, `right_fill`/`left_fill` by column),
  `ops/warp/memory/strip` (loads and stores, plain and bounded, and the
  col_vec through an LDS vector). The bounded forms use a buffer resource
  that ends at the global's last row, and give a run at or past the last
  column an offset past any range: it reads zero and is not written.
- **The dialect**: a third layout of `!hk.rt`, a `run` parameter defaulting
  to 0 on `RegTileType` and `RegVecType`, so existing builders and printed IR
  are unchanged. The maps, loads, stores, reductions and broadcasts take a
  strip because it is an `!hk.rt`; the ops whose C++ exists only for WMMA
  layouts refuse it in their verifiers (mma, LDS tile loads and stores,
  transposes, gathers, row-limited loads, the upper/lower fills,
  `philox_uniform`, `load_w4`, `atomic_add`), and a strip access needs an
  element coordinate.
- **The translator** names it `rs_fp32_1x1024_r8` and its vectors
  `rv_fp32_1x1024_r8_col`.
- **The DSL**: `hk.RS`, the same operators as a tile. `cross_warp` combines a
  strip's col_vec, and no longer puts a barrier in front of a scratch it has
  just allocated outside any loop: LDS is allocated linearly, so nothing else
  has read it.

A one-row weighted RMSNorm at 4096 columns, from the device assembly, per
wave:

| | instructions | b128 loads | b128 stores | barriers |
|---|---|---|---|---|
| Triton's, 16 warps (its tuned pick) | 221 | 2 | 1 | 2 |
| hk's tile kernel, folded plan, 4 warps | 846 | 16 | 8 (+32 b16 unaligned) | 3 |
| hk's strip kernel, 8 warps | 231 | 4 (row and weight) | 2 | 1 |

## The one-row penalty was the kernel arguments

With strips the one-row kernels came in at Triton's instruction count, and
still took 3.1 us where Triton's took 2.6. Stranger, the strip fused add +
RMSNorm, which does strictly more, took 2.3 us, and the strip softmax 2.3.

Taking the fused add apart one piece at a time (one row, 4096 columns, us):

| fused add kernel | us |
|---|---|
| as written | 2.28 |
| without the residual store | 2.27 |
| without the residual load | 3.24 |
| without either | 3.19 |
| without the weight | 3.37 |

So removing work made it slower, and the instruction stream did not explain
it. The difference turned out to be where the output tensor sits in the
kernel's arguments. A kernel's arguments are the `globals` struct, 48 bytes a
tensor. The same no-weight RMSNorm with its parameters ordered (x, w, o, s),
o at offset 0x60, took 3.17 us; ordered (x, o, w, s), o at 0x30, 2.20.

Every dispatch's arguments are written into a fresh slot of the runtime's
kernarg pool, so every scalar-cache line of them is a cold miss, to host
memory by default. LLVM sinks each argument's `s_load` to its first use. An
output's pointer and row stride are first used by the final store, so when
they sit in a second 64-byte line, that line is fetched at the end of the
kernel, after everything else has finished, and nothing overlaps it. The
fused add was fast because its early residual store pulled the late line in
early.

`kittens::touch_kernargs<sizeof(globals)>()` loads one dword of every line at
entry, XORed into a single asm operand so that the loads go out in one
clause with one wait, beside the miss on the first line every kernel pays.
hk-dsl now emits it first in every RDNA kernel. One row, 4096 columns, us:

| | arguments read where used | read at entry |
|---|---|---|
| strip RMSNorm, o at 0x60 | 3.10 | 2.09 |
| strip RMSNorm, o at 0x30 | 2.16 | 2.11 |
| hk's shipped tile RMSNorm (folded, one tile) | 3.63 | 2.59 |
| the same, `HIP_FORCE_DEV_KERNARG=1` | 3.20 | 2.39 |

The last row says the cost is the miss, not PCIe: with the arguments in
device memory the late fetch still costs 0.8 us. It is also chapter 26's
unexplained 3.7-5.3 us. Which field lands in a late line, and how late it is
first used, depends on the plan and the kernel, so the penalty came and went
between harnesses and plans while the kernels' bodies stayed the same. The
tile RMSNorm read at entry already beats tuned Triton's 2.63 us. The helper
also reads the struct's last dword: a kernarg segment is only 16-byte
aligned, so 240 bytes of arguments can span five lines rather than four.

What is left of the instability is the clock, not the kernel. In one
process, after a 5-second burn-in, with 21 interleaved rounds of 50 calls,
the strip fused add measures 2.07-2.16 us at one row and 2.20-2.27 at 32,
best and second-best round within 4%. Inside the scoreboard, whose arms run
in profiler sessions with idle gaps between them, the same kernel's best
round ranged from 2.5 to 3.2 us across five runs, and a process started
with a one-second burn-in read 2.9-3.6. The scoreboard's band rule absorbs
most of that; one run in five scored it a loss.

## The norms on strips

`hk.ops.norm` now runs RMSNorm, LayerNorm, softmax, the weighted RMSNorm and
the fused add on one strip kernel, `_strip_kernel`: one workgroup a row,
warp w holding columns `[w*CW, (w+1)*CW)` as a one-row strip, the row read
once. All five kinds are 59 lines.
- **The plan** (`norm.strip_plan`): the run is the largest power of two up to
  16 bytes that divides the row (8 for bf16, 4 for fp32, smaller for an odd
  width), so every run stays aligned; the warp's chunks come from a per-lane
  target; the warps cover the row, at most 16, and the chunks are rebalanced
  so no warp holds more than it must. Any warp count is allowed (5120
  columns is 10 warps of two chunks).
- **Tails**: a width the warps do not cover exactly takes the bounded
  accesses, and softmax's max fills the tail with -inf first. LayerNorm
  subtracts the mean before squaring: the row is held, so a second
  reduction costs a barrier and not a pass over memory, and the variance no
  longer cancels at a large offset (the test now holds it to 1e-4 where the
  tile kernel was allowed 5e-3).
- **Width**: a lane holds at most 64 elements, so a row up to 32768 wide fits;
  wider rows go to the tile kernels' streaming plan, as do all rows with
  `HK_NORM_LAYOUT=tile`.
- **Occupancy**: when a call moves more bytes than the Infinity Cache holds,
  48 KB of LDS is reserved so a WGP holds two workgroups (below).

The per-lane target, measured across 1024-8192 columns and 1-8192 rows
(`tools/hk-bench/strip_plan.py`): two 16-byte runs a lane win or tie for the
read-only kernels nearly everywhere, missing by 1-3% at a few shapes. The
fused add, which reads and writes both of its row operands, is slowest at two
runs of the three targets in the cache-resident regime, by 5-19%, and takes
one. us:

| | 8 a lane | 16 a lane | 32 a lane |
|---|---|---|---|
| rms_norm 2048x2048 | 10.33 | 9.62 | 11.23 |
| rms_norm 8192x1024 | 19.07 | 18.32 | 20.95 |
| softmax 2048x4096 | 20.23 | 18.82 | 20.00 |
| fused add 2048x1024 | 10.81 | 12.04 | 10.89 |
| fused add 2048x2048 | 19.14 | 21.17 | 19.85 |

The strip kernels beat the tile kernels at every shape swept, by 1.9-2.5x
at one row, 1.2-2.5x at 32 and 1.1-1.5x at 2048 and 8192, with one
exception: the in-place fused add at 8192x4096, where the tile plan
measured 342.8-350.7 us across three sweeps against the strip's
348.2-351.2, inside the regime's noise.

**Rows in flight.** Past the Infinity Cache the strip kernels ran 2% behind
tuned Triton at 8192 rows, and the reason was occupancy, not the kernel.
Triton's tuned fused add at that size (`waves_per_eu=4`, 16 warps) allocates
241 VGPRs: one workgroup a WGP. hk's strip takes about 50, and a WGP holds
four. Reserving LDS so that a WGP holds fewer (`hk.lds.reserve`; a WGP has
128 KB), at 8192x4096, us, 50 calls a round:

| | no cap | 5 a WGP | 3 a WGP | 2 a WGP | tuned Triton |
|---|---|---|---|---|---|
| rms_norm, 8 warps | 177.0 | 177.4 | 175.8 | 173.9 | 173.7 |
| softmax, 8 warps | 177.1 | 178.7 | 175.5 | 173.7 | 174.5 |
| fused add, 16 warps | 350.1 | 353.3 | 348.1 | 346.5 | 337.4 |

Inside the cache it is the other way round (the same cap took a 2048x4096
RMSNorm from 18.4 to 22.3 us), so the plan reserves 48 KB only when the bytes
a call moves exceed the 96 MB cache.

The same sweep showed how much of an 8192-row comparison is where the
tensors sit: the same kernel on a second set of buffers moved by 2.4%
(hk's RMSNorm) and 3.6% (tuned Triton's fused add). The scoreboard gives
each arm its own buffers, so at this size a difference under about 3% is
the allocator's as much as the kernel's.

## The scoreboard, tuned against tuned

`tools/hk-bench/scoreboard.py --only norm,softmax`, GPU ms per call. The
2048- and 8192-row rows are best of 5 rounds of 10 calls; the rows marked *
best of 9 rounds of 50. "Triton" and "FlyDSL" are the faster of each DSL's
out-of-the-box and tuned arms; the verdict is hk against the fastest other,
a tie within the larger of 2% and the two arms' noise. The last column is
the range of that ratio over the runs made with this chapter's kernels
(five at 1-2048 rows, two at 8192, where the occupancy cap came last).

| case | hk | torch | Inductor | Triton | FlyDSL | tuned: hk vs best | over the runs | chapter 26 |
|---|---|---|---|---|---|---|---|---|
| rms_norm 1x4096* | 0.0024 | 0.0042 | 0.0026 | 0.0026 | 0.0025 | 1.04x win | 1.04-1.19x | 0.48x |
| rms_norm 32x4096* | 0.0023 | 0.0044 | 0.0027 | 0.0027 | 0.0025 | 1.12x, a tie by noise | 1.07-1.14x | 0.66x |
| rms_norm 2048x4096 | 0.0184 | 0.0263 | 0.0197 | 0.0193 | 0.0190 | 1.04x win | 1.03-1.04x | 0.77x |
| rms_norm 8192x4096 | 0.1699 | 0.1771 | 0.1775 | 0.1744 | 0.1741 | 1.02x win | 1.00-1.02x | 0.91x |
| fused add + rms_norm 1x4096* | 0.0032 | 0.0060 | 0.0026 | 0.0027 | 0.0029 | 0.83x loss | 0.83-1.08x | 0.44x |
| fused add + rms_norm 32x4096* | 0.0025 | 0.0064 | 0.0029 | 0.0028 | 0.0030 | 1.11x, a tie by noise | 1.00-1.11x | 0.64x |
| fused add + rms_norm 2048x4096 | 0.0362 | 0.0488 | 0.0396 | 0.0363 | 0.0358 | 0.99x tie | 0.99-1.01x | 0.90x |
| fused add + rms_norm 8192x4096 | 0.3390 | 0.4547 | 0.3708 | 0.3280 | 0.3690 | 0.97x loss | 0.97-1.02x | 0.97x |
| softmax 2048x4096 | 0.0187 | 0.0304 | 0.0573 | 0.0255 | 0.0191 | 1.02x tie | 1.02-1.03x | 0.83x |
| softmax 8192x4096 | 0.1686 | 0.1755 | 0.2240 | 0.1725 | 0.1725 | 1.02x win | 1.00-1.02x | 0.94x |

How to read it:
- **RMSNorm and softmax win or tie at every size.** At one and 32 rows by
  4-19% (from 0.48-0.66x), at 2048 by 2-4% (from 0.77-0.83x), at 8192 they
  tie or win by 2% at the HBM roof (0.79 TB/s against torch's elementwise
  copy at 0.77).
- **The fused add ties at 32, 2048 and 8192 rows over the runs, with one
  loss at each end.** At one row it was a tie or a win in four runs of five
  and a 0.83x loss in the fifth, the clock state above. At 8192 rows the run
  in the table scored it 0.97x against untuned Triton, which in that run was
  faster than its tuned arm (0.3280 against 0.3369), and the other run 1.02x;
  both inside the buffer-placement spread measured above.
- Every other arm is the same as in chapter 26: Triton and FlyDSL tuned as
  that chapter describes, hk's row kernels on its planner.

## What did not work

- **Non-temporal accesses.** `HK_STRIP_NT` makes the strip loads or stores
  non-temporal. At 8192x4096, back to back on the same tensors, an RMSNorm
  with non-temporal stores read 80.5 us against Triton's 173.7: past what
  HBM moves. The output no longer evicts the 67 MB input from the 96 MB
  Infinity Cache, so every call after the first reads its input from there.
  That is the benchmark, not the kernel; in a model the next kernel reads
  the output, and that traffic would go to HBM. Left off, and kept only so
  that the measurement can be repeated (non-temporal loads 125.7 us, both
  168.1).
- **Several rows a strip.** `ROWS` lets a warp hold two or four rows of the
  same columns. It won by more than noise only for the fused add at
  2048x1024 (7%), and is not in the plan.
- **The residual store after the barrier.** gfx11's `s_barrier` waits for
  every store in flight, so the fused add's residual store, issued before the
  cross-warp combine, is a write acknowledgment every wave waits for. Issued
  after the barrier instead it landed on the kernel's tail: 0.1-0.2 us slower
  at 1-32 rows and no faster at 2048 and 8192. It stays where it was.

## Tests

- `tests/hk/ir/test_strip.py`: the type and its spelling, the translation
  (aliases, `touch_kernargs` first, one barrier in a fresh combine), the IR
  round trip, every refusal, gfx11 only, the plan.
- `tests/hk/codegen/test_strip_codegen.py`: the shipped strip kernels compile
  without spilling at occupancy 8 or more; the weighted kernel's ISA is 4
  `global_load_b128` and 2 `global_store_b128` with a DPP butterfly and no
  `ds_bpermute`; the arguments' lines load in one clause.
- `tests/hk/gpu/test_strip.py`: loads, stores, both reductions across warps,
  both broadcasts and the fills, on bf16, fp16 and fp32 globals, runs of 1, 2,
  4 and 8, one to four rows a strip, ragged rows and columns, a 16-bit strip
  copied bit for bit, a bounded store leaving the rest of a strided view
  alone.
- The norm tests run on both layouts (`norm.LAYOUT`); new ragged widths for
  the weighted norms on every dtype, the residual still bit-exact against
  torch.

## Reproduction

```
cd /workspace/hk-dsl
export FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 PYTHONPATH=python
python3 tools/hk-bench/scoreboard.py --only norm,softmax --rounds 9 --calls 50
python3 tools/hk-bench/scoreboard.py --only norm,softmax --rounds 5 --json sb.json
python3 tools/hk-bench/strip_norm.py          # strip warps against the tiles, Triton, FlyDSL
python3 tools/hk-bench/strip_plan.py --rows-per 1,2,4   # the per-lane target, rows a strip
```
