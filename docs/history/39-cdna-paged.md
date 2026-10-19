# gfx950's paged attention: the decode on the matrix core, the copies' runs, and a merge a query row

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 34](34-cdna-attention.md) left paged attention as the one family
hk could not run on gfx950: `hk.ops.paged` and `hk.ops.paged_prefill` are
RDNA kernels (16-wide WMMA operands, V transposed in registers, LDS
granules) and were refused while tracing. This chapter writes them for
the MI355X -- the split-KV decode and its merge in a new
`hk.ops.paged_cdna`, the prefill as chapter 34's attention schedule over
the page table -- and takes the GPU tier's last "no native gfx950 path"
skips with them.

Against what vLLM runs on an MI355X, Qwen3-8B's decode attention at 1-128
requests and 512-8192 tokens: at 8-128 requests hk is 0.90-1.05x aiter's
gluon unified attention, the fastest kernel there, and 1.04-1.27x
ROCM_ATTN, vLLM's default; at one or two requests with 2048 tokens or more
it is 0.71-0.88x, and the merge and a lone wave's walk are what is left.
The prefill is 1.10-1.24x aiter's at batches of prompts and chunks, 2.5-13x
ROCM_ATTN's.

One MI355X of the shared node (GPU 2 for the measurements, GPU 5 for the
GPU tier), idle at the start of every run. The GPU tier runs in
`rocm/pytorch:rocm7.2.3 … py3.12 torch 2.10` as the CI does. The
comparisons run in `vllm/vllm-openai-rocm:nightly-rocm100-0cbac6cd`: ROCm 10
(HIP 7.15), torch 2.12, vLLM 0.30.1rc1.dev558, triton 3.8, and aiter 0.1.23,
which earlier chapters could not install and this image ships. hk is built
in each image with its own hipcc; its decode times agree between the two to
within 3% at large batches and 10% at one request
(`tools/hk-bench/paged_cdna.py`).

## The inventory

The gfx950 GPU tier on main (`0bc9bbb`) was 729 passed, 121 skipped. 55
of the skips were `tests/hk/conftest.py`'s `_NO_CDNA_PATH` rule, which
reported a kernel with no CDNA path as a skip:

| kernel | tests | reason |
|---|---|---|
| paged decode (`KERNELS`, `split_kernel`) | test_paged.py, 19 | hk-fragments: 16-wide operand tiles |
| paged decode, fused, and its merge | test_paged_fused.py, 14 | the same |
| paged prefill | test_paged_prefill.py, 9 | the same |
| W4A16 decode GEMM | test_w4a16.py, 10 | `load_w4` is include/rdna3's |
| the one-warp attention example | test_example_runs.py, 2 | 16-wide operand tiles |
| the code-object LDS test (RDNA GEMM) | test_codeobj.py, 1 | 16-wide operand tiles |

The other 66 are not missing CDNA paths: test_gemm_cdna.py's 31 are
schedules a shape does not fit; test_flydsl_gemm.py's 17 FlyDSL's RDNA3 GEMM;
test_hipkittens.py's 10 HipKittens' GQA, which ROCm 7.2.3's hipcc cannot
build; test_vllm_gemm.py's 4 a route table that sends nothing to the RDNA
GEMM; test_route_tune.py's 3 gfx1100's candidates; one test needs vLLM. After
this chapter the tier is 784 passed, 66 skipped, 0 failed, and the rule is
gone: such a failure is a failure again.

## The decode

`hk.ops.paged_cdna.decode_kernel` stands in for all three of hk.ops.paged's
contracts (`KERNELS`, `split_kernel` with merge_splits, `fused_kernel` with
`merge_kernel`), and hk.ops.paged routes to it on gfx950 only. One warp a
(split, kv head, request):

- **Sᵀ = K·Qᵀ** on the 16x16x32 MFMA, the GQA group's query rows padded to
  16 columns, so the softmax's statistics are a row vector of Sᵀ.
- **Pᵀ read in place.** Two 16-row accumulators are one 32x16 operand
  fragment with a lane run of 4 (`Tile.as_operand`, chapter 34's 16x16
  case), and V is read out of LDS with the same run by
  `ds_read_b64_tr_b16`: P·V moves no registers. hk-lds-pipeline did not
  count that read; it does (two a fragment).
- **K and V by DMA**, a page of each copied into a two-page LDS ring while
  the page before is worked, each copy bounded to the sequence's rows: a
  page past the split copies nothing, so a stale index in the table's unused
  columns is never followed, and the tail of the last page reads zero.

The first version was right everywhere the tests look, and 0.65-0.95x the
best alternative. What moved it, in order:

**The page index was in vmcnt's queue.** `load_scalar` is a global load and
a readfirstlane. Its wait counts vmcnt, in one queue with the DMA copies,
and LLVM hoisted the readfirstlane: each page's index waited out every copy
issued before it, which held the ring to one page in flight however deep
it was. `load_scalar(invariant=True)` is an `s_load_dword` through the
scalar cache, on lgkmcnt. One request, 1024 and 8192 tokens: 12.99 to
12.67 us and 18.65 to 18.05.

**Where a lone wave's time went.** At one request and 8192 tokens, a split
walking all 256 pages took 0.68 us a page: deeper rings (three or four
pages) took it to 0.62, and stayed slower overall because a workgroup's LDS
took slots on the CU. The softmax's two column reductions on a 16-column
accumulator were include/cdna4's shuffle through LDS -- three `ds_bpermute`,
each waited with `lgkmcnt(0)`, which also waited out the V reads in flight.
The shim reduces such a tile with `v_permlane32_swap` and
`v_permlane16_swap`: 0.60 us a page.

**The merge was most of a small batch.** hk.ops.paged's merge builds on
gfx950 as it is and is right, and the profiler put it at 3.7 us at 8 splits
and 7.5 at 32, against 3.0-4.0 us for ROCM_ATTN's and aiter's reductions; at
one request hk's walk was level with theirs or faster (4.9-6.1 us against
6.7-7.5 at 1024 tokens, 9.9 against 9.7-10.0 at 8192) and the merge was the
difference. It gives a warp 16 columns of a 16-row tile of which the
group's four rows are real, and reads the splits eight at a time. gfx950's
merge is a warp a (request, kv head, query row), the row a strip of head_dim
(`hk.RS`), folding the splits online eight a round, and only the rounds the
sequence's pages reach: 2.2-2.3 us up to 8 splits that saw pages, 3.1 at
16, 4.7 at 32. Reading every split's addresses at once spilled 110 SGPRs,
and issuing the next round under the fold 29: an address a split, held in
SGPRs, is what bounds it.

**The plan.** Measured at 1-128 requests and 512-8192 tokens (one to 32
splits each), the best split count put about 256 one-warp workgroups on the
card for 4096 tokens and more, and fewer for shorter contexts at one to two
requests. A captured step cannot see the length, so the plan aims at 256
(`paged_cdna.TARGET_WORKGROUPS`, hk.ops.paged.plan_splits on gfx950) and a
split takes at least four pages (`MIN_SPLIT_PAGES`): a 512-token sequence
under 32 planned splits walks four of them, and the merge reads only those.
One request: 9.2 us at 512 tokens (eight splits, the best there, 7.5; 32
with no floor, 10.7), 16.1 at 8192 (the best).

**The copies' runs.** Large batches sat at 5.9-6.2 TB/s against 6.5-6.6 for
aiter's gluon kernel, and the decode with its math taken out
(`ablate=("math",)`: copies and LDS reads only) ran at the same 6.1 TB/s:
the copy path was the limit. Neither a deeper ring nor K straight into
registers (`kreg`, with only V through LDS) nor interleaving the splits'
pages moved it (5.8-6.1 TB/s). A DMA copy writes 16 bytes a lane into
consecutive LDS bytes, so its global footprint is one subtile of the LDS
layout: the fragments' own layouts, st_16x32 for K and st_32x16 for V, make
each copy read 64- and 32-byte runs of a 256-byte row, every L2 line asked
for two or four times. aiter's kernel reads 1 KB contiguous a copy, with
`sc0 nt`. In hk, 128 requests x 8192 tokens, the walk alone, us
(`HK_DMA_CPOL` was a build define of the experiment):

| K layout | V layout | cache policy | full | copies + LDS reads only |
|---|---|---|---|---|
| st_16x32 | st_32x16 | 0 | 695 (6.18 TB/s) | 698 |
| st_16x32 | st_32x16 | sc0 nt | 896 (4.79) | 688 |
| st_16x64 | st_32x16 | sc0 nt | 799 (5.37) | 687 |
| st_16x32 | st_32x32 | sc0 nt | 681 (6.31) | 687 |
| st_16x64 | st_32x32 | 0 | 699 (6.15) | 698 |
| st_16x64 | st_32x32 | sc0 nt | 620 (6.93) | 687 |

Non-temporal copies in half-line runs fetched each line twice (4.8 TB/s);
in 128- and 64-byte runs, 6.9. The two layouts are include/cdna4's own,
whose generic loads also serve those fragments, so hk names them:
`hk.lds.alloc(..., shape="16x64" | "32x32")` (the translator holds every read
of the allocation to a fragment the layout serves) and
`g.load(..., nontemporal=True)`. The K reads out of the unswizzled st_16x64
conflict on banks more; at one request the walk is no slower.

### Where it stands

Qwen3-8B's attention (32 query heads, 8 KV heads, D 128), bf16, us a layer;
each arm at its own engine's cache layout holding the same K/V under a
scrambled page table, every arm checked against fp32 first, calls rotating
through enough layers' caches to read past the 256 MB Infinity Cache, each
arm's calls captured in a CUDA graph and replayed, rounds interleaved, the
median of 7 (`tools/hk-bench/paged_cdna.py decode`). aiter's gluon kernel
(`aiter.ops.triton.unified_attention`, ROCM_AITER_UNIFIED_ATTN) is timed on
hk's cache, where it is a little faster than on its own NHD layout;
ROCM_ATTN's is `ops.paged_attention_rocm` through
`chunked_prefill_paged_decode` on its 5-D pages of 16; TRITON_ATTN's is
vLLM's `unified_attention` with its builder's 3-D threshold; SDPA gathers the
pages first.

| requests | context | hk | aiter gluon | ROCM_ATTN | TRITON_ATTN | SDPA | hk / best | hk / ROCM_ATTN |
|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 9.2 | 9.1 | 9.3 | 7.5 | 43 | 0.81x | 1.01x |
| 1 | 1024 | 9.1 | 15.5 | 9.4 | 9.1 | 60 | 1.00x | 1.03x |
| 1 | 4096 | 13.5 | 9.6 | 10.2 | 18.3 | 169 | 0.71x | 0.75x |
| 1 | 8192 | 16.1 | 12.2 | 12.8 | 31.0 | 307 | 0.76x | 0.80x |
| 4 | 1024 | 9.9 | 15.9 | 10.1 | 11.1 | 74 | 1.02x | 1.02x |
| 4 | 8192 | 28.2 | 27.3 | 30.7 | 42.5 | 378 | 0.97x | 1.09x |
| 16 | 1024 | 17.3 | 17.4 | 18.3 | 22.7 | 149 | 1.01x | 1.05x |
| 16 | 8192 | 88.7 | 89.9 | 99.0 | 131.8 | 1026 | 1.01x | 1.12x |
| 32 | 1024 | 25.4 | 23.2 | 30.3 | 42.9 | 272 | 0.91x | 1.19x |
| 32 | 8192 | 171.4 | 166.1 | 194.7 | 301.5 | 2030 | 0.97x | 1.14x |
| 64 | 4096 | 167.3 | 170.9 | 188.7 | 210.0 | 2045 | 1.02x | 1.13x |
| 128 | 1024 | 84.0 | 85.0 | 104.4 | 118.4 | 1089 | 1.01x | 1.24x |
| 128 | 8192 | 659.2 | 644.6 | 725.8 | 898.1 | 8095 | 0.98x | 1.10x |

Of the 40 shapes measured (1, 2, 4, 8, 16, 32, 64 and 128 requests at 512,
1024, 2048, 4096 and 8192 tokens), hk is within 2% of the best alternative
or ahead at 15 and within 5% at 24; at 8-128 requests 0.90-1.05x; it is
faster than ROCM_ATTN at 32 (up to 1.27x) and slower at one to four
requests with 2048-8192 tokens (0.75-0.96x). aiter's `paged_attention_v1`
(ROCM_AITER_FA's decode) is slower than ROCM_ATTN at every shape. At 128 x
8192 hk reads 6.5 TB/s of K/V and aiter 6.7.

What is left:
- **One or two requests, 2048 tokens and more (0.71-0.88x).** At one request
  and 4096 tokens aiter takes 9.6 us in one kernel and its reduction; hk's
  32 splits of four pages walk in 8.4-9.5 us and merge in 4.7. The merge's
  bound is an address a split in SGPRs, and a lone wave's page costs 0.60
  us of dependent latency (copy, reads, QK, softmax, P·V), which a
  software-pipelined walk -- QK of the next page under P·V of this one, as
  chapter 34's phases do -- would overlap.
- **32 requests (0.91-0.97x).** One split, 256 workgroups, one wave a CU,
  each walking all of its sequence; two splits did not help.

Negative results: rings of three and four pages (slower: a workgroup's LDS
is a slot on the CU, and three pages fit 1.33 rounds of 1024 workgroups);
K straight into registers (`kreg`, level or slower); splits taking every
`splits`-th page instead of a contiguous run (no change); two or four warps
sharing a split, each walking its own pages and folded in LDS at the end
(17.3-17.9 us against 17.8 at one request and 8192 tokens: the warps' copy
descriptors and barriers cost what they saved); the merge reading every
split at once, or the next round under this one's fold (SGPR spills).

## The prefill

On gfx950 `hk.ops.paged_prefill` is chapter 34's attention forward
(`attn_kernel(paged=True)`) over the paged cache, with the RDNA kernel's
contract: q and o the engine's (tokens, heads x D) rows, a workgroup one
(request, query tile) entry of a work list, query i of a request at
position ctx + i, causal from the cached prefix on. The tile is 256 query
rows (8 warps of 32), so the work list is built with gfx950's tile, a
request's tiles last first. A 64-row KV block is two pages: each its own
DMA copy into its own LDS buffer, bounded to the sequence's rows, its index
through the scalar cache; the block's registers are read a page at a time,
Sᵀ written a 32-row subtile at a time and P·V read off P's two halves. At
D = 64 a page is half an 8-warp copy, so half the warps copy it and the
other half copy it again. The dense kernel's source is byte-identical.

Qwen3-8B, requests as cached-prefix + new tokens, us
(`tools/hk-bench/paged_cdna.py prefill`, the same method):

| requests | hk | aiter gluon | ROCM_ATTN | TRITON_ATTN | hk / best | hk / ROCM_ATTN |
|---|---|---|---|---|---|---|
| 0 + 2048 | 59.3 | 58.3 | 153.3 | 224.5 | 0.98x | 2.59x |
| 4096 + 512 | 106.2 | 92.0 | 283.7 | 228.3 | 0.87x | 2.67x |
| 4 x (2048 + 1024) | 171.8 | 192.6 | 600.6 | 920.8 | 1.12x | 3.50x |
| 8 x (0 + 2048) | 297.9 | 327.0 | 889.6 | 1511.2 | 1.10x | 2.99x |
| 0+300, 1000+77, 4000+1000, 64+2048 | 142.7 | 176.9 | 1908.3 | 683.9 | 1.24x | 13.4x |

A 512-token chunk on a 4096-token prefix is 2 query tiles x 32 heads, 64
workgroups for 256 CUs, where aiter's 128-row tiles make twice as many.

## The vLLM backend on gfx950

`hk.integration.vllm_backend` uses these kernels on gfx950 unchanged: the
page is 32 tokens, the cache head-major ([B, H, N, 2C] on vLLM 0.31's
layout, which the decode reads as row-strided views at the same speed), the
decode two launches on torch's current stream and the prefill one, all
capturable. `tools/hk-bench/paged_ab.py` in the vLLM image runs the
backend's forward against TRITON_ATTN's: vLLM's cache writes into the
head-major pages bit-identical, every path checked, the decode captured
whole. hk's decode against Triton's kernel on that cache at one to eight
requests is 0.81-0.91x at 512 tokens and one to four requests, 0.99-1.18x
at 1024 and 1.14-1.99x from 2048, so the backend hands no batch to Triton
there (`_triton_decode_max()` is 0 on gfx950).

Running it in that image found a crash first: after `import vllm`, hk's
first trace segfaulted creating the MLIR context. vLLM's `_C` links ROCm's
`libLLVM.so`, loaded first, and hk's libraries' calls into their own LLVM
and MLIR resolved to that copy. hk._mlir's libraries are linked
`-Bsymbolic` now (`-Bsymbolic-functions` alone failed on an attribute
lookup instead), and hk traces after any vLLM import. Every hk op that
vLLM's backend or linear patch traces inside the engine runs after that
import, so in this image hk's ops could not be traced inside a vLLM
process unless hk had traced before vLLM was imported.

What it still needs, for vLLM on gfx950: an end-to-end measurement in the
engine (the decode step and prefill against ROCM_ATTN, which vLLM picks by
default there); the backend's routing measured in a captured step rather
than kernel for kernel, as chapter 20 found the two can disagree; block
size 32 is the backend's (vLLM's ROCm default is 16); and no fp8 KV cache.

## The rest of the skips

- **`load_w4`** on gfx950 fills a 16x32 row operand fragment, whose lane
  holds eight columns of a row -- one word of hk's interleaved codes, the
  fragment's 64 lanes reading 256 contiguous bytes -- with none of
  include/rdna3's permlanex16. fp16 then failed in include/cdna4's
  `mma_ABt`, which takes fp16 operands only into an fp16 accumulator; the
  shim's overload issues `v_mfma_f32_16x16x32_f16` into fp32. Against
  vLLM's Triton AWQ kernel at Qwen3-8B's shapes: 2.3-4.0x at 1-16 tokens,
  1.5-2.3x at 32-64, on gfx1100's schedules (0.35-1.3 TB/s).
- **`examples/05_attention.py`** walks a 32-key block a step on wave64,
  Pᵀ read in place; its RDNA source is unchanged.
- **test_codeobj's LDS test** launches the pingpong GEMM's 128 KB on
  gfx950.

RDNA output is byte-identical (`tools/cdna-probes/dump_sources.py`, 976
kernels for gfx1100 and gfx1201), and so are gfx950's attention and GEMMs.
New tests: `tests/hk/ir/test_cdna_paged.py`,
`tests/hk/codegen/test_cdna_paged_builds.py` (the copies counted against
the ISA under ROCm 7.1's hipcc), `mlir/test/Target/scalar.mlir`.
