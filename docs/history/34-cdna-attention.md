# gfx950's attention in hk: P read in place, masks the register allocator copied, and a division per element

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 29](29-cdna.md) put HipKittens' CDNA4 GQA forward behind an hk op,
because hk could not write attention for gfx950: its softmax's P comes out of
the matrix core as an accumulator, P·V wants it as an operand, and
`hk-fragments` refused the cast between the two. [Chapter 33](33-cdna-gemm.md)
wrote HipKittens' GEMM in hk. This chapter writes its attention: flash
attention forward, bf16, GQA, head dims 64 and 128, causal or not, in
`hk.ops.attn_cdna`. At the ten points it was set against, it runs at
0.98-1.08x the stronger of HipKittens' and FlyDSL's kernels, and
1.30-3.13x torch's SDPA. `hk.attention` and the SDPA drop-in run it on gfx950,
so the attention tests that were skipped there run and pass.

Hardware and software as in chapter 33: one MI355X (gfx950) of the shared
node, GPU 5, idle at the start of every run, in `rocm/pytorch:rocm7.2.3 …
py3.12 torch 2.10` (HIP 7.2.53211); FlyDSL 0.4.0 with its checkout's
`kernels/` at `88f1969`; HipKittens' GQA at the pinned submodule, built with
ROCm 7.1's hipcc (7.2.3 cannot compile it). hk's kernel is built with 7.2.3
and checked under 7.1 too. Every arm is checked against fp32 first -- three
32-row blocks of queries against chunked SDPA in fp32, and for hk two checks
with exact answers (V = 1 gives exactly 1; Q = 0 gives the mean of V) -- then
timed interleaved in one process; numbers are the best of 7 rounds, spreads
in `tools/hk-bench/attn_cdna.py`'s JSON.

## First, torch's stream

On gfx950 every hk op launched on the null stream: the CDNA pybind entry
point took no stream, so a CUDA graph capture came out empty and an op was
not ordered against torch's work on another stream. The entry point now takes
one (`bind_function_on_stream`) and `Kernel.__call__` passes torch's current
stream, 0.2 µs a call; `tests/hk/gpu/test_stream_cdna.py` captures a graph of
`hk.ops.gemm` and replays it against fresh inputs. That landed first, as its
own commit (`535be1c`).

## What hk could not say

HipKittens' `kernels/cdna4/attn/gqa` against hk at `535be1c`:

| piece | hk before | now |
|---|---|---|
| Sᵀ = K·Qᵀ on the 32x32x16 MFMA, a 32x32 accumulator | no: accumulators were 16x16 | `frag=(32, 32)` on an accumulator (wave64 only) |
| P·V reading P where the softmax left it | no: `hk-fragments` refused an accumulator cast into an operand | `Tile.as_operand`: the stride-4 operand fragment (`frag=(16, 32, 4)`, `rt_16x32_4`) of the same rows, and the MMA's other operand given the same stride |
| one 64-row score tile whose halves take exp2 in different phases | no | `Tile.subtile(rows, i)`, an alias (`hk.rt_subtile`) |
| the lazy rescale's vote | no | `Vec.any_above(other, threshold)`, one int a wave |
| a phase's results held to its phase | no | `hk.arch.pin(*values)` |
| K and V blocks, one LDS type read two ways | the translator gave a type one `st_shape` | a shape per allocation |
| the DMA, the pingpong, `setprio`, interleave hints | chapter 33 | the same |

**The operand read off an accumulator.** In a 32x32 accumulator lane l holds
column l%32, rows 4(l/32) + 8k + s for k, s < 4: four runs of 4 rows. The
16x32 operand fragment holds 8 consecutive rows a lane. Read the
accumulator's registers as an operand anyway and the K index of the product
runs down the rows in another order -- harmless if the other operand is
permuted the same way. So the type carries the lane run (`frag 16x32 stride
4`), `hk-fragments` gives V the stride P has, and V's tile is read out of LDS
with the transposing `ds_read_b64_tr_b16` in that order. Pᵀ needs no register
move between the softmax and the second MFMA: `as_operand` converts fp32 pairs
to bf16 into registers that already hold the right rows.

## The kernel

Eight warps of 32 query rows, 256 rows a workgroup; K and V walked in 64-row
blocks, double-buffered in LDS by the DMA; each step four phases a barrier
apart, two of math and two of memory, the two halves of the workgroup a
barrier apart so one wave of each SIMD is at the matrix core while the other
is at memory -- HipKittens' schedule. What differs:
- **The rescale.** HipKittens rescales O after a quarter of P(t)·V, so when
  the max jumps three quarters of block t are added at the wrong scale. Here
  the vote and the new max are taken without a branch and O is rescaled at
  the top of the next math phase.
- **Any N**, and (B, H, N, D), torch's layout: Q rows past N read zeros and
  are not stored, a KV block past N_kv is masked.
- **Causal** (below).

## What the compiler did with it

Most of the work was between the kernel as written and the code that ran.

**A hazard in inline asm.** With both interleave hints on, some lanes came
out wrong. An inline-asm `v_cvt_pk_bf16_f32` read a `v_exp_f32` result one
instruction later; the hardware wants a wait state there, and the compiler
cannot see inside asm to insert it. The conversion is now compiler-visible
(`__builtin_convertvector` to `__bf16x2`).

**Work that left its phase.** LLVM sinks a value to its use, and in this
schedule the use is a phase -- a barrier -- away; `sched_barrier` fences the
scheduler, not the passes before it. The subtraction S − m sank into the next
math phase's head, ahead of its first MFMA, and P's conversion with it.
`hk.arch.pin` binds the values at the end of their phase. Pinning the score
tile too, with HipKittens' split of the softmax (the PV phase takes the max
and exp2 of half the block, the QK phase the rest), put every exp2 under an
MFMA: non-causal 0.94x to 0.96x the best baseline at N = 4096 and 8192.

**Masks the register allocator copied.** The causal mask was three-way per
32-row half: all −inf, the diagonal's triu, or nothing. LLVM placed the −inf
fill's moves before the branch, into 16 registers of their own, and copied the
scores into them on every path that masked nothing -- every step. Making the
fill instructions of their own (asm) took half of the copies away (causal
947 to 1013 TFLOP/s at N = 4096); the diagonal's path still left the tile in
new registers. One branch, "does this block reach past any of this warp's
queries", with one element-wise triu over the whole block -- −inf throughout
a block wholly past them -- and the triu's update in its own registers (a
tied `v_pk_add_f32`) removed the rest: 245 to 232 VGPRs, 1050 TFLOP/s. The
triu itself was then two compares, two `s_nop` and two selects a pair of
elements, the compares through VCC; four compares into SGPR pairs and four
selects in one asm block have no nops.

**A division per element.** O / l compiled to an IEEE division for each of a
lane's 64 elements, ten VALU apiece -- 640 VALU a warp at the end of every
workgroup. One division a query and a multiply an element: timed at N_q =
1024 with N_kv from 512 to 8192 and fitted to a line, the fixed part of the
16x64-head grid's time fell from 121 to 98 µs (FlyDSL: 107), and N = 1024
went from 0.92x to 0.98x.

**Packed fp32 that came back unpacked.** S − m and the column sums are pairs
of the same query, so `v_pk_add_f32` halves them. LLVM unpacked every pair
that sat in an MFMA's shadow, where two single VALU co-issue and one packed
one does not; no change in time, and the overloads were not kept (without
them the final build ran about 1% faster at every point).

**The copy's descriptor.** Each DMA copy computes its buffer descriptor:
based at the tile, its range clamped to 32 bits. Written as a 64-bit compare
-- however written -- it was a `v_cmp_lt_u64` and a VCC-to-SALU hop a copy;
with the range ending at the slab's last element it also depended on the
column, which the GEMM's K loop moves. The range now counts whole rows from
the tile and saturates by the product's high word (`s_mul_hi_u32`): scalar,
and loop-invariant in the GEMM. Against the copy as chapter 33 left it,
attention is 6% faster (1218 against 1150 TFLOP/s at 16x64x8x4096x128) and
hk's GEMM 1.2% faster (1470 against 1452 at 4096³, 1549 against 1530 at
8192³).

**A gate that miscounted.** `hk.runtime.cocheck` reported every attention
build as reading Q before its load landed. It left the DMA's `buffer_load …
lds` out of vmcnt's queue -- it writes no register -- so a load into
registers issued before the copies looked unretired by the wait that retires
it. It counts them now.

## Causal

Under causal a workgroup walks the KV axis up to its last query, in steps of
two blocks. With masking timed out (wrong answers), causal ran 13% faster
than with it; after the single-branch mask, 2%. The rest was wasted work: with
256 queries a workgroup and 64-key blocks, the last two steps are wholly past
the queries of the first half of the workgroup. Those warps now skip both
steps' math and keep their barriers and their wait for their share of V's last
block, so the steps' math phases are the other half's alone. A branch may not
hold a barrier (`hk-verify`), and a branch around each phase's math held the
scores, P and V live into the steps and spilled; the region is written as a
loop of 0 or 1 trips beside one of the other, each with the steps' seven
barriers. Causal at N = 1024 went from 0.95x to 0.98x, at 4096 from 0.94x to
0.95x.

## Where it stands

TFLOP/s (4·B·H·N²·D, halved under causal), best of 7 interleaved rounds;
`tools/hk-bench/attn_cdna.py`:

| B, H, H_KV, N, D | causal | hk | HipKittens | FlyDSL | torch SDPA | hk / best | hk / SDPA |
|---|---|---|---|---|---|---|---|
| 16, 64, 8, 1024, 128 | no | 1056 | 978 | 1055 | 574 | 1.00x | 1.84x |
| | yes | 739 | 681 | 746 | 256 | 0.99x | 2.89x |
| 16, 64, 8, 4096, 128 | no | 1230 | 1233 | 1247 | 690 | 0.99x | 1.78x |
| | yes | 1107 | 1091 | 1122 | 431 | 0.99x | 2.57x |
| 16, 64, 8, 8192, 128 | no | 1259 | 1277 | 1283 | 773 | 0.98x | 1.63x |
| | yes | 1209 | 1202 | 1226 | 497 | 0.99x | 2.43x |
| 1, 32, 8, 4096, 128 | no | 1164 | 1157 | 1178 | 653 | 0.99x | 1.78x |
| | yes | 1104 | 902 | 1024 | 353 | 1.08x | 3.13x |
| 16, 64, 8, 4096, 64 | no | 938 | 939 | 942 | 724 | 1.00x | 1.30x |
| | yes | 857 | 804 | 862 | 539 | 0.99x | 1.59x |

The spread between rounds is 0.3-2.5% except at N = 1024 and at one batch
of 32 heads (6-16% for every arm: a few hundred microseconds a call). At the
start of the chapter's tuning the kernel was at 0.94x non-causal and
0.86-0.88x causal at N = 4096 and 8192.

**What is left is per step.** Held at N_q = 1024 with N_kv from 512 to 8192
(a straight-line fit), hk's kernel costs 419.5 µs per 1024 keys of the
16x64-head grid against FlyDSL's 410.8 -- 2% more -- and 93 µs fixed against
FlyDSL's 103. Its two math phases issue what HipKittens' issue, opcode for
opcode (16 MFMAs, 16-17 exp2, 49-58 VALU each). Counted on the device (N =
4096 against 8192), hk issues 176.5 VALU and 30.5 SALU a step a wave,
HipKittens 170.5 and 18.8, FlyDSL 156.5 and 8.5; at a workgroup's ends hk
issues 337 VALU a wave, HipKittens 668, FlyDSL 521. The VALU in the math
phases already fits under the MFMAs (packing it changed nothing), so the
next gains are in the memory phases' issue -- the copies' descriptors are
most of the 30.5 SALU, against FlyDSL's 8.5 -- and in how the phases
overlap, not in the math.

## hk.attention and the SDPA drop-in

`hk.ops.attn` routes to `attn_cdna` on gfx950 (`_native_cdna`), with the CDNA
kernel's rules: bf16, contiguous (B, H, N, D), head dim 64 or 128, H_KV
dividing H, causal only for N_kv = N_q, any non-empty length. `hk.ops.sdpa`
goes through the same rules, so `F.scaled_dot_product_attention` patched by
`sdpa.patch()` runs the kernel on gfx950 behind a `TORCH_LIBRARY` op on
torch's stream. The kernels are named `attn_fwd_d{64,128}[_causal]_cdna`.

The gfx950 GPU tier was 615 passed, 170 skipped, 0 failed at `535be1c`; it
is 658 passed, 127 skipped, 0 failed: the 43 are `test_attn.py` and the
`test_sdpa.py` cases that run the kernel. `examples/validate.py` checks
attention on gfx950 too (2.1-2.2x SDPA at 1x32x1024x128). New tests:
`tests/hk/ir/test_cdna_attn.py` (the language and the kernel's shape),
`tests/hk/codegen/test_cdna_attn_builds.py` (the gate and the copy count in
the ISA), lit cases for the new ops and type, and cocheck's DMA count.

## What is still missing

- **Paged prefill and decode** on gfx950 still have no path: `hk.ops.paged`
  and `hk.ops.paged_prefill` are RDNA kernels (16-wide operand tiles, LDS
  granules) and are refused at trace time. Prefill is this kernel with its
  K and V copies addressed through the block table: a 64-row block spans
  64 / page_size pages, so a copy needs a per-lane page offset (a gathering
  DMA, which `g.load` cannot say today) or one copy per page. Decode is a
  different schedule: one to eight query rows a sequence (the GQA group)
  leave a 256-row workgroup idle, so it wants a split-KV kernel over the
  sequence -- 16-row MFMA tiles or VALU dot products, and a combining pass
  -- as FlyDSL's `pa_decode` kernels are.
- **`examples/05_attention.py`**, the one-warp teaching kernel, is written
  with 16x16 operand tiles and is still skipped on gfx950.
- **gfx942** compiles the shim but has no attention: the kernel uses
  gfx950's 32x32x16 MFMA, `ds_read_b64_tr_b16` and the 16-byte DMA.
