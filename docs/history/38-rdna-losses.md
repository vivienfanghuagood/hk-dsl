# The RDNA losses: a transpose that was most of a page, stores rounded as torch rounds, and LayerNorm's backward on strips

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The record listed four places where hk's kernels lost on the W7900D
(gfx1100):
1. paged decode attention at 16-64 requests, 0.83-0.95x Triton's kernel on
   hk's cache ([chapter 20](20-vllm-031.md));
2. LayerNorm backward at 2048 columns and wider, 0.66-0.92x the Triton
   tutorial ([chapter 19](19-rows-by-index.md));
3. the MoE GEMM above 256 tokens ([chapter 21](21-moe.md));
4. W4A16 at 1-4 tokens against `gptq_gemm_rdna3` ([chapter 22](22-int4.md)).

[Chapter 36](36-the-agent-loop.md) added a fifth that is not a speed:
HipKittens' fp32-to-bf16 conversion rounds ties away from zero, torch rounds
them to even, and on a residual add the two disagree at 10.7% of the
elements.

This chapter closes the second and the fifth; closes the first at every
capture size but 24 and 56, where the engine still does better on hk's
kernel than on Triton's; and checks that routing covers the third and
fourth. Everything was measured in the engines' pod
(vLLM 0.31.0, torch 2.13.0+rocm7.2, triton-rocm 3.7.1), on GPU 3, with
`hk._mlir` built in a directory of its own.

## Paged decode attention

**Two Tritons.** `tools/hk-bench/paged_variants.py` (new) measures on the
cache vLLM 0.31 keeps, one tensor a layer, `[blocks, heads, 32, 2 x
head_dim]`, K and V side by side in each token's row: every arm walks
enough layers' caches to read K/V cold, each timed as a CUDA graph's replay,
all arms interleaved in one process, and every arm is first checked against
Triton and against V all ones (every output element exactly 1.0). Triton's
arm is `unified_attention` as TRITON_ATTN calls it, its 16-segment 3-D
kernel up to 16 requests and its 2-D kernel above.

The pod's vLLM venv has two Tritons installed. vLLM 0.31 was built against
triton-rocm 3.7.1, which chapter 20 measured; on October 15 a Triton 3.8.0
wheel was installed over the same `triton` package, and that is what the
engines and every measurement here import unless told otherwise. Its 2-D
kernel is up to 40% slower at 32-64 requests: 0.9462 ms a layer at 4096 x
32 where 3.7.1 takes 0.6907. So both are measured below, 3.7.1 from its own wheel (`triton_rocm-3.7.1`, the same
`libtriton.so` hash as the venv's record of it) on `PYTHONPATH`, each in a
process of its own with hk's walk beside it. `paged_ab.py`, chapter 20's
tool, reproduces both: under 3.7.1 Triton takes 0.6870 ms at 4096 x 32 (the
record's 0.6907), under 3.8.0 0.9637.

The cache's layout is not what moved: Triton on vLLM 0.16's two planes
(`<arm>@2p`) runs 0.95-1.02x its speed on 0.31's cache, and on TRITON_ATTN's
own layout (16-token blocks, a token's heads side by side) it is no faster
either. Nor is the order of the blocks: a fresh engine's sequential tables
(`--sequential`) time like shuffled ones, within the 1-4% two runs differ
by.

hk's walk as main shipped it was 0.70-0.96x Triton 3.8.0 at 16 requests
and at 1024 x 32 and 64, 1.04-1.14x at 4096 x 32 and 64; against 3.7.1 it
was the record's 0.83-0.95x.

**Where the walk's time went.** `tools/hk-bench/paged_attrib.py`
(`--cases`, new) deletes stages of the walk at trace time. At 4096 x 16,
the walk with no K or V stream ran in 66 us of its 379: the kernel waits on
memory, at 708-735 GB/s. Its ISA said what the 66 us was. The page loop
was about 3,350 instructions, and about 2,900 of them were `v_cndmask`, DPP
moves and `v_perm` on half words: the transposes of V. The walk computes
Oᵀ += Vᵀ Pᵀ, V arrives [kv, d] and WMMA wants the reduction axis inside the
lane, so each of a page's 16 V fragments is transposed in registers
(`transpose_sep`). HipKittens' rdna3 tree ran the four butterfly stages on
each 16-bit value: about 180 instructions a fragment.

**A transpose on words.** In an operand tile, element pair (2k, 2k+1) is
32-bit word k. Element bits 1-3 are word bits 0-2, so three of the four
stages swap whole words between lanes, and only stage 0 works inside a word,
as one DPP move and one `v_perm_b32` a word. That is about 58 instructions
a fragment, and the page loop is 1,425. It is in the fork's rdna3 tree
(`transpose_operand_words`), so every RDNA op that transposes an operand
takes it, with outputs bit for bit as before. `HK_RDNA_XPOSE_WORDS=0` keeps
the generic stages, which is how the A/B below ran in one process. Attention
forward transposes V the same way, and gained as much: at n = 4096, 32
heads, d = 128, 4.687 -> 4.287 ms non-causal and 2.782 -> 2.507 causal
(1.09x and 1.11x, 15 rounds in one process, outputs bit-identical), which
puts it at 1.05x and 1.24x torch 2.13's SDPA in the perf gate (from 0.99x
and 1.13x). The
GPU tests of every op that transposes (paged, prefill, attention, SDPA,
GEMM, decode GEMM, MoE, W4A16, the backend's decode) passed, 200 of 200.

**Then the split plan was wrong by four.** `plan_splits` aimed for 1,024
one-warp workgroups, a number chapter 06 measured when the walk was slower.
With a page's work cut by 57%, fewer workgroups walking more pages each won
at every batch (ms a layer, walk plus merge):

| ctx | batch | s1 | s2 | s4 | s8 | s16 | s32 | best | workgroups |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.0946 | 0.0520 | 0.0322 | 0.0237 | **0.0218** | 0.0237 | 16 | 128 |
| 1024 | 4 | 0.0976 | 0.0571 | 0.0451 | **0.0430** | 0.0517 | 0.0647 | 8 | 256 |
| 1024 | 8 | 0.1027 | 0.0805 | **0.0637** | 0.0817 | 0.1010 | 0.1110 | 4 | 256 |
| 1024 | 16 | 0.1360 | **0.1121** | 0.1220 | 0.1524 | 0.1699 | 0.1774 | 2 | 256 |
| 1024 | 32 | **0.1989** | 0.2157 | 0.2434 | 0.2693 | 0.2913 | 0.3052 | 1 | 256 |
| 1024 | 64 | **0.3922** | 0.4266 | 0.4552 | 0.4785 | 0.5115 | 0.5458 | 1 | 512 |
| 4096 | 1 | 0.3512 | 0.1852 | 0.1024 | 0.0593 | 0.0462 | **0.0438** | 32 | 256 |
| 4096 | 4 | 0.3589 | 0.1971 | 0.1410 | **0.1150** | 0.1251 | 0.1516 | 8 | 256 |
| 4096 | 8 | 0.3738 | 0.2615 | **0.2041** | 0.2177 | 0.2480 | 0.2646 | 4 | 256 |
| 4096 | 16 | 0.5023 | **0.3824** | 0.3981 | 0.4270 | 0.4511 | 0.4709 | 2 | 256 |
| 4096 | 32 | **0.7447** | 0.7679 | 0.7862 | 0.8167 | 0.8435 | 0.8781 | 1 | 256 |
| 4096 | 64 | **1.4931** | 1.4983 | 1.5329 | 1.5629 | 1.5998 | 1.6643 | 1 | 512 |

`TARGET_WORKGROUPS` is 256 now. At one request, the one point it misses,
the backend runs Triton's kernel anyway.

**Against Triton** (ms a layer; Triton 3.8.0's run is 9 rounds with
every arm interleaved, 3.7.1's a run of 7 rounds of its own with hk's walk
beside it, and each ratio is from its own run -- hk's times in the two
runs agree within 3.3%; "own layout" is Triton 3.8.0 on the cache
TRITON_ATTN keeps; ratios are Triton on hk's cache over hk):

| ctx x batch | Triton 3.7.1 | Triton 3.8.0 | 3.8.0, own layout | hk before | transpose only | hk now | now vs 3.7.1 | now vs 3.8.0 |
|---|---|---|---|---|---|---|---|---|
| 1024 x 16 | 0.1254 | 0.1246 | 0.1219 | 0.1782 | 0.1517 | 0.1155 | **1.12x** | **1.08x** |
| 1024 x 32 | 0.1838 | 0.2582 | 0.2802 | 0.2791 | 0.2442 | 0.1990 | 0.93x | **1.30x** |
| 1024 x 64 | 0.3530 | 0.4413 | 0.4971 | 0.4618 | 0.4307 | 0.3915 | 0.90x | **1.13x** |
| 4096 x 16 | 0.3777 | 0.3988 | 0.3870 | 0.4656 | 0.4260 | 0.3881 | 0.99x | **1.03x** |
| 4096 x 32 | 0.6907 | 0.9462 | 1.0197 | 0.8281 | 0.7862 | 0.7451 | 0.93x | **1.27x** |
| 4096 x 64 | 1.3467 | 1.6118 | 1.7625 | 1.5493 | 1.5031 | 1.4889 | 0.90x | **1.08x** |

The walk is 1.04-1.54x faster than main's; "transpose only", the packed
transpose at main's split counts, is 1.03-1.18x of that, the rest the
plan. With lengths ragged by up to 64 tokens, against 3.8.0, it is the
same: 1.04-1.27x. hk's walk now runs at 581-721 GB/s, 67-83% of the card's
864; Triton 3.7.1's 2-D kernel reaches 730-810 at 24-64 requests, and that
is the gap left.

**The capture sizes in between.** vLLM pads a decode batch to 16, 24, 32,
40, 48, 56 or 64 (ms a layer, hk at its planned split):

| batch | ctx | Triton 3.7.1 | Triton 3.8.0 | hk | hk vs 3.7.1 | hk vs 3.8.0 |
|---|---|---|---|---|---|---|
| 24 | 1024 | 0.1365 | 0.1500 | 0.1658 (s2) | 0.86x | 0.90x |
| 24 | 4096 | 0.4985 | 0.5173 | 0.5762 (s2) | 0.87x | 0.90x |
| 40 | 1024 | 0.2229 | 0.2824 | 0.2497 (s1) | 0.89x | **1.13x** |
| 40 | 4096 | 0.8497 | 1.0168 | 0.9542 (s1) | 0.89x | **1.07x** |
| 48 | 1024 | 0.2944 | 0.3135 | 0.2917 (s1) | **1.01x** | **1.07x** |
| 48 | 4096 | 1.1237 | 1.1212 | 1.1019 (s1) | **1.02x** | **1.02x** |
| 56 | 1024 | 0.3174 | 0.3466 | 0.3364 (s1) | 0.95x | **1.03x** |
| 56 | 4096 | 1.2102 | 1.2204 | 1.2728 (s1) | 0.95x | 0.96x |

At 24 requests one split ties Triton 3.8.0 at 1024 tokens (1.00x) and no
split reaches it at 4096 (0.89-0.90x). Kernel for kernel, then, hk's walk
leads Triton 3.7.1 at 1024 x 16 and at 48, is level at 4096 x 16, and
trails it by 5-14% at 24, 32, 40, 56 and 64. The routing can hand batches
to Triton by range, `HK_TRITON_DECODE_ALSO="17-24"`
(`vllm_backend.triton_takes`); it is off by default, for the reason below.

**Below 16 requests Triton keeps its batches.** At one and four requests
hk's walk, at its planned split, is 0.66-0.99x Triton's 3-D kernel under
either Triton, and at eight they tie (0.97-1.05x). vLLM pads 9-15 requests to 16, so
`TRITON_DECODE_MAX` stays at 8.

**Chapter 20's paradox is not the 2-D kernel.** There, routing 16 and 32
requests to Triton made the captured step slower (4096 x 16: 59.0 -> 74.2
ms) though the kernel measured faster. The backend calls Triton with
`seq_threshold_3D = 8`, so at 16 requests it ran the 2-D kernel where
TRITON_ATTN runs the 3-D one. That would have explained it, and it does not:
forced to 2-D at 16 requests Triton takes 0.4351 ms a layer at 4096 tokens
against 0.3988, 1.3 ms a step, not 15; at 1024 tokens the 2-D kernel is the
faster.

**In the engine the paradox is still there, and it is Triton's kernel.**
`python -m hk.integration headroom --batch 24 --context 4096 --arms hk
--graphs` profiles eight captured decode steps; requests finish at different
times, so the steps held 15 to 22 requests, each replaying the graph
captured for 16 or 24. With hk's walk the step took 62.17 ms and decode
attention 16.40 ms of it (`paged_decode_fused`, two splits, 0.451 ms a
call). With `HK_TRITON_DECODE_ALSO=17-24` the six steps of 17-22 requests
went to Triton: the step took 111.64 ms, and `kernel_unified_attention`
52.04 ms over its 27 calls a step, 1.93 ms a call -- 3.7x the same call on
the bench at 24 requests (0.517 ms; the engine imports Triton 3.8.0), and
slower than the 3-D kernel there (0.614). hk's walk took less in the step
than on the bench (0.451 against 0.576), as it should with 15-22 requests
where the bench has 24; Triton's took more. The kernel and the cache's
layout are the same, and neither the layout nor the blocks' order moves it
on the bench (above); what is left is the engine's cache and allocator and
the arguments of a captured call. Why is still open. Chapter 20 saw the
same under triton-rocm 3.7.1, so the routing does not depend on which
Triton is installed: in the step hk's walk is the faster, and the switch
stays off.

**Measured and not levers.**
- *The head_dim split across warps.* Four warps each owning 32 columns of
  Q, O, K and V, S summed through LDS: the fp32 accumulator's LDS store and
  load go element by element in include/rdna3, LLVM hoists every element's
  swizzled address out of the page loop, and it spilled 24-118 VGPRs at two
  to eight warps.
- *Vᵀ through LDS.* The row written as it was read, read back a column a
  lane: the same hoisting, 233 VGPRs spilled. A version with raw
  `ds_load_u16_d16` reads cannot win either: each fragment's round trip
  (about 120 cycles) is longer than the packed transpose's latency, its wait
  cannot be split from its use, and at the best split counts one or two
  waves share a SIMD, so the transpose's VALU throughput is not what the
  walk waits on.
- *A page's loads issued together.* Q kept in LDS (64 VGPRs freed), both K
  rows at a page's start, both V rows as soon as S is computed, the next
  page's first K row under the current page's products: level with the
  fused walk at every point (4096 x 16: 0.3816 against 0.3824 ms; 4096 x 32:
  0.7430 against 0.7447), and one VGPR short of spilling under ROCm 7.2
  without the prefetch. The walk does not wait on how many loads a wave has
  in flight.
- *The half-split loads* (`HK_RT_HALF_SPLIT=1`: each 32-byte run loaded
  once, the halves swapped with `v_permlanex16`): slower at every point,
  0.71-0.95x.

**End to end.** `python -m hk.integration ab --json` on Qwen3-8B, vLLM's
defaults (CUDA graphs, prefix caching, chunked prefill), 72 output tokens
(128 for the 64-request shape, whose short run must hold 64), 2 rounds, the
tree before this chapter and after it each against its own stock arm in its
own run. Decode is the step, the slope between the short and the long run;
ratios are stock over the arm:

| shape | arm | total before | total after | decode step before | decode step after |
|---|---|---|---|---|---|
| 1024 x 16 | hk | 1.042x | 1.060x | 48.45 ms (1.027x) | 46.54 ms (1.063x) |
| 1024 x 32 | hk | 1.080x | 1.096x | 51.60 ms (1.131x) | 48.99 ms (1.183x) |
| 1024 x 64 | hk | 1.108x | 1.126x | 60.46 ms (1.215x) | 57.91 ms (1.246x) |
| 4096 x 16 | hk | 1.359x | 1.373x | 58.68 ms (1.197x) | 56.51 ms (1.227x) |
| 4096 x 32 | hk | 1.417x | 1.434x | 72.58 ms (1.418x) | 68.70 ms (1.472x) |
| 1024 x 16 | hk+gemm | 1.544x | 1.587x | 27.84 ms (1.788x) | 25.98 ms (1.904x) |
| 1024 x 32 | hk+gemm | 1.500x | 1.539x | 32.65 ms (1.787x) | 29.79 ms (1.946x) |
| 1024 x 64 | hk+gemm | 1.498x | 1.530x | 42.84 ms (1.715x) | 40.41 ms (1.786x) |
| 4096 x 16 | hk+gemm | 1.798x | 1.837x | 38.25 ms (1.836x) | 35.57 ms (1.949x) |
| 4096 x 32 | hk+gemm | 1.829x | 1.862x | 52.39 ms (1.964x) | 49.31 ms (2.051x) |

Every row is a win past its band, before and after; the decode step is 1.9-
3.9 ms shorter after at every shape (4-9%), and prefill moved by at most
1.1% (the prefill kernel did not change). The greedy outputs were
identical to stock's in every run (72/72, 128/128).

**Headroom.** `python -m hk.integration headroom --arms stock,hk+gemm` at
16 x 4096 (eager, chapter 36's configuration), each tree in its own run:
decode attention is 28.6 ms of stock's 71.2 ms step either way, and in the
hk+gemm arm it went from 16.73 to 14.12 ms a step. The ratio headroom
measures for the family went from 1.71x to 2.02x, and the share of the
family's time hk takes back from 41% to 51%; with the linear layers'
2.00x, attention no longer trails them. The record's row for the decode
(`headroom.RECORD`), which headroom uses where a step runs Triton's kernel,
is now 0.86-1.12x, the ratio against triton-rocm 3.7.1: still no ceiling
there, since hk's walk does not beat that kernel at every batch.

## bf16 stores round as torch rounds

HipKittens' rdna3 tree narrows fp32 to bf16 in one place, `convertor<bf16,
float>`, rounding to nearest with ties away from zero (`HK_BF16_ROUND=1`).
Its comment gives the reason: ties-even is one add dearer, attention pays it
on every probability tile, and an exact tie is "a measure-zero event on real
data". That holds for computed values and not for a store whose input is a
sum of two bf16 values ([chapter 36](36-the-agent-loop.md)).

So the stores now round apart from the registers. `store_convertor<bf16,
float>` (`HK_BF16_STORE_ROUND`, default 2: nearest even, torch's rounding)
is what every store that narrows uses -- tiles, bounded and scattered rows,
vectors and strips -- and in-register conversions (`copy()`, attention's P)
keep `HK_BF16_ROUND`. Mode 3 also keeps a NaN whose low mantissa bits the
rounding add would carry into the exponent (torch keeps every NaN); mode 1
is ties away as before. CDNA's stores already rounded to even through hk's
shim (`include/hk/cdna.cuh`); its output is unchanged.

`tools/hk-bench/bf16_store_round.py` builds each case three ways in one
process -- the stores at ties away as before, at nearest even, and at
nearest even with the NaN guard -- and compares each with torch bit for bit
(15 rounds, the arms' order rotated each round; "differ" is the elements
where the old and new stores disagree, each an exact tie):

| case | = torch before | after | differ | us before | even (2) | even + NaN (3) |
|---|---|---|---|---|---|---|
| add, 16 x 4096 | 89.33% | **100%** | 10.67% | 12.92 | +0.3% | +0.6% |
| add, 2048 x 4096 | 89.36% | **100%** | 10.64% | 36.48 | -0.9% | +0.2% |
| add, 8192 x 4096 | 89.34% | **100%** | 10.66% | 289.20 | +0.1% | +0.2% |
| fused add + RMSNorm, 1 x 4096 | 89.11% | **100%** | 10.89% | 12.60 | 0.0% | +0.6% |
| fused add + RMSNorm, 2048 x 4096 | 89.36% | **100%** | 10.64% | 40.92 | +1.7% | -0.3% |
| RMSNorm, 2048 x 4096 | 99.9990% | 99.9996% | 0.0006% | 27.12 | +0.1% | +1.3% |
| LayerNorm, 2048 x 4096 | 99.9990% | 99.9995% | 0.0006% | 27.32 | +0.4% | +1.9% |
| softmax, 2048 x 4096 | 99.9988% | 99.9990% | 0.0005% | 27.48 | +0.4% | +1.0% |
| GEMM 4096³ | 99.737% | 99.737% | 0 | 1544.9 | +0.7% | -- |
| attention d128, 4096 | 59.007% | 59.007% | 0.0007% | 2298.9 | 0.0% | +0.1% |

The add -- new-op's template, a residual add -- and the fused add's
residual are torch's at every element. Computed values are ties a few
times in a million: the norms and softmax moved closer to torch by those,
and the GEMM and attention differ from torch for other reasons (the order
of their sums), not the rounding. Even rounding cost within about 1% (the
fused add's +1.7% at 2048 rows and the add's -0.9% are the noise either
way); the NaN guard cost 1-2% more on the norms, and with it the perf gate
put `rms_norm` at one row 3.0% behind Inductor, so 2 is the default. The
GEMM had no tie and paid 0.7%, so its kernels keep ties away
(`defines={"HK_BF16_STORE_ROUND": 1}` in `hk.ops.gemm`), where its C++
compiles to what it was.

**The perf gate.** With the stores and the transpose changed, `scripts/
perf_gate.sh` against the old baseline found no regression (57 pairs) and
printed that hk's kernels had changed; the baseline was refreshed from four
runs (`--refresh 4`), and a fifth passed against it. Of the 57 pairs the
median moved 0.0%; the norms and softmax moved -2.8% to +2.4%, attention
against torch +6.2% (non-causal) and +10.1% (causal), the GEMM not at all.

`tests/hk/gpu/test_store_round.py` holds it to exact answers: 1 + 2⁻⁸ and
three other ties go to the even neighbour on both sides of zero, a NaN stays
a NaN and an overflow an infinity, and an add of random bf16 rows -- tile
and strip -- is torch's bit for bit. The exact-answer tests that were exact
stay exact; 213 GPU tests of the ops that store bf16 passed.

## LayerNorm backward on strips

`examples/10_layernorm_backward.py` held a 16-row tile of x-hat and dy
across its two passes, which fits a row of 2048 at most; past that it read
each row twice. A strip ([chapter 27](27-strips.md)) is one row, each lane
holding 16-byte runs of it and nothing twice, so a row of any width is read
once:
- a workgroup walks rows g, g + G, ..., its warps covering the row, two
  16-byte runs a lane, the next row's x and dy loaded while this one is
  reduced;
- c1 and c2 of every warp go into one double-buffered LDS vector stack, one
  barrier a row (`hk.collective.cross_warp` inside a loop allocated its
  scratch inside the loop, where the translator had no allocator: the
  exchange is written out instead);
- each workgroup keeps its rows' dy x x-hat and dy in registers and writes
  them once, as one partial row; the second kernel sums G partial rows.

G is about 768 waves' worth (`groups_for`). Two sweeps of G agreed (4096
rows, ms, interleaved with the Triton tutorial and torch):

| width | tiles | strips, G=48 | 96 | 192 | 384 | Triton tutorial | strips vs Triton |
|---|---|---|---|---|---|---|---|
| 1024 | 0.039 | 0.059 | 0.036 | 0.026 | **0.026** | 0.050 | **1.90x** |
| 2048 | 0.066 | 0.063 | 0.040 | **0.034** | 0.036 | 0.058 | **1.69x** |
| 4096 | 0.152 | 0.100 | **0.091** | 0.112 | 0.139 | 0.105 | **1.15x** |
| 8192 | 0.531 | **0.301** | 0.304 | 0.320 | 0.347 | 0.346 | **1.15x** |

The tiles measured 1.27x, 0.88x, 0.69x and 0.65x in the same runs (the
record: 1.12x, 0.92x, 0.69x, 0.66x). dx is within 2.7e-3 of fp32 autograd
and dw, db within 3.3e-7, at every G. The 1024- and 2048-wide cases fit the
96 MB Infinity Cache, which is why they run past the card's HBM rate.

## MoE above 256 tokens and W4A16 at 1-4 tokens: covered by routing

Neither kernel loss reaches a request, because the routing does not hand
those calls to hk. Each patch counts its calls (the `routing:` line `ab`
prints), and `ab` runs it against stock with graphs on and vLLM's defaults:
- **MoE above 256 tokens.** `vllm_moe.BLOCK16_MAX_TOKENS = 256`: a call
  with more tokens goes to vLLM's `fused_moe`. OLMoE-1B-7B, 1024 and 3072
  tokens at 1 and 16 requests: 960 calls to hk and 2,272 to Triton, every
  captured call hk's. Prefill, where the long calls are, ties stock at
  every shape (0.999-1.004x); decode wins 1.09-1.38x and the request
  1.05-1.34x; outputs 72/72.
- **W4A16 at 1-4 tokens.** `gptq_gemm_rdna3` is vLLM's kernel for GPTQ
  checkpoints, which hk's int4 patch leaves alone. On
  Qwen2.5-7B-Instruct-GPTQ-Int4 at 1 and 4 requests the patch took 0 calls,
  vLLM logged `RDNA3W4A16LinearKernel` in both arms, and the request tied
  stock (1.002-1.004x, decode 1.004x, 72/72). hk's W4A16 runs on AWQ
  checkpoints, where stock's kernel is Triton's AWQ GEMM: on Qwen3-8B-AWQ
  at 1, 4 and 16 requests, 3,600 calls to hk (every captured one) and 4,176
  prefill calls to stock; decode 3.84-4.60x, the request 1.68-3.88x,
  prefill a tie or a win (1.005-1.023x), outputs 72/72.

hk's W4A16 at 1-4 tokens is still 0.69-0.90x `gptq_gemm_rdna3` kernel for
kernel. Its weight path is relabel-only transposes, with nothing like the
decode's 2,900 instructions to remove; closing the gap needs a dot-product
GEMV for 1-4 tokens instead of the WMMA tile, which is not a cheap change
and was not made.

## What is left

- Kernel for kernel, hk's walk still trails triton-rocm 3.7.1's 2-D kernel
  by 5-14% at 24, 32, 40, 56 and 64 requests (it leads at 1024 x 16 and at
  48, and is level at 4096 x 16): 581-721 GB/s against 730-810. More
  workgroups do not close it (two splits at 32 requests are slower than
  one), and neither did issuing a page's loads together; what Triton's
  code does differently at the instruction level was not found.
- Triton's decode kernel runs 3.7x slower in a captured vLLM step than on
  the bench (chapter 20's paradox, reproduced above). hk's walk does not
  have the problem, so in the engine hk is the faster and no routing waits
  on it, but the cause is unknown.
- The pod's venv imports Triton 3.8.0, installed over the triton-rocm
  3.7.1 vLLM 0.31 was built for; its decode kernel is up to 40% slower at
  32-64 requests (level at 48, and at 56 x 4096). Which one a deployment runs decides what "Triton's
  kernel" means.
- One to eight requests stay on Triton's 3-D kernel (hk 0.66-0.99x at 1 and
  4).
- The W4A16 GEMV at 1-4 tokens against `gptq_gemm_rdna3` (above).
- The MoE GEMM above 256 tokens is still 0.41-0.87x `fused_moe`, routed to
  it.

## Reproduction

```
# in the engines' pod, GPU 3, PYTHONPATH=python
python3 tools/hk-bench/paged_attrib.py --cases 4096:16:8/4/16,4096:32:4/2/8,1024:64:2/1/4,4096:64:2/1/4
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 1,4,8,16,32,64 \
    --arms triton,trinative,hk:fused:1,hk:fused:2,hk:fused:4,hk:fused:8,hk:fused:16,hk:fused:32
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 8,16,24,32,48,64 --rounds 9 \
    --arms triton,trinative,tri2d,hk:fused:plan1024:xw0,hk:fused:plan1024,hk:fused:plan,hk:fused:plan:hs
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 24,40,56 --rounds 9 \
    --arms triton,tri3d,hk:fused:1,hk:fused:2,hk:fused:4
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 16,32,64 --ragged --rounds 9 \
    --arms triton,hk:fused:plan1024:xw0,hk:fused:plan
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 16,32,64 --rounds 7 \
    --arms triton,triton@2p,hk:fused:plan,hk:fused:plan@2p
python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 --batch 16,24,32,64 --rounds 7 --sequential \
    --arms triton,hk:fused:plan
# triton-rocm 3.7.1 from https://download.pytorch.org/whl/triton_rocm-3.7.1-cp312-cp312-linux_x86_64.whl
pip install --no-deps --target tri371 triton_rocm-3.7.1-cp312-cp312-linux_x86_64.whl
PYTHONPATH=tri371:python python3 tools/hk-bench/paged_variants.py --ctx 1024,4096 \
    --batch 1,4,8,16,24,32,40,48,56,64 --rounds 7 --arms triton,hk:fused:plan
PYTHONPATH=tri371:python python3 tools/hk-bench/paged_ab.py --ctx 1024,4096 --batch 16,32 --layers 36
python3 tools/hk-bench/bf16_store_round.py --rounds 15
python3 -m pytest tests/hk/gpu/test_store_round.py
python3 examples/10_layernorm_backward.py --groups 48,96,192,384 --rounds 9
# end to end: HK_EAGER=0 HK_VLLM_DEFAULTS=1, each tree its own run
AB="python3 -m hk.integration ab --tokens 72 --short 32 --reps 2"
$AB --model Qwen/Qwen3-8B --arms stock,hk,hk+gemm --context 1024,4096 --batch 16,32 --json out.json
$AB --model Qwen/Qwen3-8B --arms stock,hk,hk+gemm --context 1024 --batch 64 --tokens 128 --short 64 --json out.json
$AB --model allenai/OLMoE-1B-7B-0125-Instruct --arms stock,moe --context 1024,3072 --batch 1,16 --json out.json
VLLM_USE_TRITON_AWQ=1 $AB --model Qwen/Qwen3-8B-AWQ --dtype float16 --arms stock,int4 --context 1024 --batch 1,4,16 --json out.json
$AB --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --dtype float16 --arms stock,int4 --context 1024 --batch 1,4 --json out.json
python3 -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096 --arms stock,hk+gemm --tokens 24
python3 -m hk.integration headroom --model Qwen/Qwen3-8B --batch 24 --context 4096 --arms hk --graphs
HK_TRITON_DECODE_ALSO=17-24 python3 -m hk.integration headroom --model Qwen/Qwen3-8B \
    --batch 24 --context 4096 --arms hk --graphs
```
