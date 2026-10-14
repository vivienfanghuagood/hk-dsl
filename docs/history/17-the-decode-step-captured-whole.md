# The decode step captured whole, and where hk's decode still loses

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Where the decoder backend stood before this: a decode step 1.14-1.19x slower
than vLLM's Triton attention, run eagerly between the pieces of vLLM's graphs,
with prefix caching and chunked prefill refused. Around the kernel, each layer
issued a dozen small operations:
- a torch scatter for the cache write;
- a padded copy of the query;
- a pybind launch on stream 0, which a CUDA graph cannot capture;
- a torch merge of the splits;
- a copy of the output.

The goal for this milestone was vLLM as it ships -- CUDA graphs, prefix
caching and chunked prefill on -- with Qwen3-8B faster end to end at every
point of a context x batch grid and the greedy tokens unchanged. All numbers
here are vLLM 0.16.1.dev0 on one W7900D.

**Making the step capturable.** Each of the following removed a launch or a
host dependency (`python/hk/integration/vllm_backend.py`):
- The cache write is vLLM's own `reshape_and_cache_flash`, through a
  `(blocks, block, heads, d)` view of the head-major pages. It takes its
  strides from the tensor, so the view is a layout it writes correctly --
  bit for bit against a torch scatter (`tools/hk-bench/paged_ab.py` checks it
  on every run). It also skips slot -1, which a graph's padding tokens carry;
  the scatter it replaced was six launches and sent them to the last slot of
  the last block.
- The query and the output are the engine's own `(tokens, kv_heads, group,
  d)` views. A group of four rows loads into a 16-row tile bounded by the
  global, so the pad rows read zeros and are never stored.
- The split count is planned from the batch size and max_model_len alone,
  never from the step's max_seq_len. A graph captured on a dummy batch of
  length-1 sequences keeps whatever it planned for every replay, and a split
  past the end of a short sequence is empty and merges with weight zero.
- The scratch is allocated once, at its largest.
- Every launch is a torch op on torch's stream.
- The merge is a launch of its own, a warp per 16 columns of head_dim. Folding
  it into the split that finishes last (`wg_arrive`) was measured and lost:
  that is one warp doing all of it in series, 18-47 us on every decode.

The builder then declares `UNIFORM_SINGLE_TOKEN_DECODE`. vLLM captures decode
steps whole and runs steps with prefill in them between its pieces, and
`forward` refuses with `GraphCaptureError` to be captured on one of those.

**The walk, from 56% to 83-86% of HBM** at large shapes, attributed with the
kernel's `ablate` switches (`tools/hk-bench/paged_attrib.py`). The page index
is prefetched one page ahead, K and V fragments are issued eight at a time
ahead of their WMMAs, and V's first row is in flight before the softmax.
Measured and not levers:
- `#pragma unroll` on the runtime page loop;
- unrolling it by hand, which made larger cold code;
- a floor of four pages a split (`MIN_SPLIT_PAGES`). A ctx-1024 decode went
  from 56 to 67 us, the main loop from 16 to 42.
- a merge step of 16 splits: each bounded read builds a 4-SGPR descriptor,
  and it spilled 16 SGPRs under hipcc 7.1 and 7.2.

One ablation was unfair and is worth knowing about: deleting the page-table
read also made every K/V read an L2 hit.

**Paged prefill.** The dense attention kernel, with K and V read through the
page table (`python/hk/ops/paged_prefill.py`). It needed one thing from the
language: a row bound that is a runtime value. `x[idx].load(t,
bounded=n)` is the new `hk.load_rows` op, whose HipKittens side is
`load_bounded(..., limit)`. The causal diagonal sits at `ctx = seq_len -
q_len`, so a cached prefix or an earlier chunk is just context in front of
the new tokens. The host builds a work list of (request, query block) for
the whole step, and one launch serves every prefill request in it. With
that, the refusal of prefix caching and chunked prefill came out.

Against `unified_attention` on the same batches (`paged_ab.py --prefill`,
ms):

| batch | Triton | hk | |
|---|---|---|---|
| prompt 2048 | 4.47 | 0.87 | 5.1x |
| prompt 8192 | 82.6 | 10.5 | 7.9x |
| chunk 512 after 4096 | 4.69 | 0.79 | 5.9x |
| 4 x 1024 after 1024 | 12.6 | 2.38 | 5.3x |
| 8 x 3000 + 200 | 10.2 | 2.85 | 3.6x |
| 16 decodes + chunk 512 | 3.03 | 0.72 | 4.2x |

**The measurement that hid a loss.** The first end-to-end run with vLLM's
defaults made decode steps 1.4-3.1% slower than stock at batch 16 and
below. At one request and 1024 tokens that was about 35 us a layer. The
layer-level A/B had said 6-12 us, and for two reasons it was the A/B that
was wrong:
- It replayed a one-layer graph over one cache. Every replay reread K/V the
  previous one had left in the 96 MB Infinity Cache, where a decode step
  reads 36 layers' worth cold.
- A one-layer graph's replay overhead, about 13 us, sat on top of both arms.

`paged_ab.py --layers 36` gives each arm 36 caches and walks them all in one
call; that is the number to believe. Graph clock, ms per layer:

| ctx | batch | Triton layer | hk layer | hk walk + merge alone |
|---|---|---|---|---|
| 1024 | 1 | 0.0206 | 0.0312 | 0.0263 |
| 1024 | 4 | 0.0424 | 0.0541 | 0.0485 |
| 1024 | 16 | 0.1323 | 0.1304 | 0.1237 |
| 4096 | 1 | 0.0444 | 0.0534 | 0.0480 |
| 4096 | 4 | 0.1182 | 0.1267 | 0.1211 |
| 4096 | 16 | 0.4198 | 0.4038 | 0.3961 |

Kernel by kernel (`tools/hk-bench/paged_kernels.py`, us per layer, same
caches), the loss is in every launch:

| ctx x batch | Triton: 3-D kernel + reduction + cache write | hk: walk + merge + cache write |
|---|---|---|
| 1024 x 1 | 13.5 + 3.1 + 2.6 = 19.2 | 21.0 + 8.0 + 4.1 = 33.0 |
| 1024 x 4 | 32.7 + 4.0 + 2.7 = 39.4 | 39.4 + 9.3 + 5.1 = 53.8 |
| 4096 x 1 | 35.6 + 3.2 + 2.6 = 41.4 | 40.9 + 9.2 + 5.2 = 55.3 |
| 4096 x 4 | 104.5 + 4.2 + 2.8 = 111.6 | 107.6 + 9.3 + 5.2 = 122.1 |

The causes:
- **The merge is twice Triton's reduction.** It reads 32 splits eight at a
  time, a round of latency each. Reading them unbounded straight into fp32 --
  no descriptor, so sixteen at a time fits -- was slower at every point,
  7.8-8.8 us to 11.6-14.8. Each unbounded read carries its alignment branch
  and its elementwise fallback, and in a launch this latency-bound the larger
  code is what it pays. The reads stayed as they were.
- **vLLM's C++ cache write** through the transposed view is ~2 us slower
  than the Triton one into vLLM's own layout. The Triton write cannot be used
  here: it treats a token's heads as contiguous.
- **The walk** at one page a split is a chain of dependent memory round
  trips -- the query and the page index, then K, then V, then the partial out
  -- with nothing to overlap them. Fewer splits make it worse, not better:
  each page then costs a full round trip in series (4096 tokens, one request:
  0.048 ms a layer at 32 splits, 0.099 at 8, 0.164 at 4).

**Triton's kernel on hk's cache.** `unified_attention` addresses K and V by
stride, so hk's head-major pages are a `(blocks, block, heads, d)` view to it
as well. On them it is bit-identical to the stock backend, and no slower: at
4096 tokens it is faster, 0.1109 against 0.1149 ms at four requests and
0.2078 against 0.2148 at eight, because a head's tokens are contiguous. So
the backend now routes pure-decode batches of up to eight requests
(`TRITON_DECODE_MAX`) to it, and keeps hk's decode from sixteen up. vLLM pads
a batch to a capture size first, so 9-15 requests are a 16.

After the routing the backend's layer is within 0.5 us of stock's at 1024
tokens -- the cache write -- and faster at 4096:

| ctx | batch | Triton layer | hk layer, routed |
|---|---|---|---|
| 1024 | 1 / 4 / 8 | 0.0207 / 0.0427 / 0.0725 | 0.0214 / 0.0431 / 0.0730 |
| 1024 | 16 | 0.1322 | 0.1328 |
| 4096 | 1 / 4 / 8 | 0.0447 / 0.1184 / 0.2187 | 0.0452 / 0.1157 / 0.2128 |
| 4096 | 16 | 0.4204 | 0.4080 |

This is a loss for hk, recorded as one. The gate was that hk's *kernel* not
be slower than Triton's at eight points, and it is slower at five: one and
four requests at both contexts, and eight at 4096. Where it wins it wins by
more: at 32 requests Triton's backend leaves its 16-segment kernel for the
unsplit one, and hk's is 2.6x it (0.2211 against 0.5832 ms a layer at 1024
tokens, 0.7626 against 2.0301 at 4096; 16 layers' caches).

**End to end.** Qwen3-8B, vLLM's defaults (CUDA graphs, prefix caching,
chunked prefill), 72 output tokens, best of two rounds
(`tools/hk-bench/vllm_e2e.py`, `HK_VLLM_DEFAULTS=1`). `hk` is the attention
backend alone, `hk+gemm` with the linear-layer patch as well; "step" is a
decode step at the full batch, and ratios are stock over patched:

| ctx x batch | stock ms | hk ms | x | hk+gemm ms | x | step, stock ms | hk | x | hk+gemm | x |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 3206 | 3184 | 1.007x | 1730 | 1.85x | 41.32 | 41.35 | 0.999x | 21.29 | 1.94x |
| 1024 x 4 | 4033 | 3924 | 1.028x | 2384 | 1.69x | 42.31 | 42.46 | 0.996x | 22.16 | 1.91x |
| 1024 x 16 | 7628 | 7160 | 1.065x | 5150 | 1.48x | 46.32 | 46.81 | 0.989x | 26.57 | 1.74x |
| 1024 x 32 | 13059 | 11264 | 1.159x | 8768 | 1.49x | 61.64 | 51.00 | 1.209x | 31.18 | 1.98x |
| 4096 x 1 | 4448 | 3986 | 1.116x | 2458 | 1.81x | 42.21 | 42.24 | 0.999x | 22.16 | 1.91x |
| 4096 x 4 | 9195 | 7274 | 1.264x | 5286 | 1.74x | 45.29 | 44.84 | 1.010x | 24.54 | 1.85x |
| 4096 x 16 | 27923 | 20254 | 1.379x | 16738 | 1.67x | 56.20 | 56.36 | 0.997x | 36.92 | 1.52x |
| 4096 x 32 | 55372 | 37439 | 1.479x | 31789 | 1.74x | 109.38 | 70.56 | 1.550x | 50.32 | 2.17x |

Greedy output tokens are identical to stock's. Every point is faster, with
the attention backend alone (1.007-1.479x) and with both patches
(1.48-1.85x). The attention backend's decode step is level with stock's up to
16 requests -- it is stock's kernel there, and at 16 hk's within a percent
-- and 1.21-1.55x at 32. Before the routing the same grid lost at 1024 x 1,
1.020x slower in total and 3.1% per step.

**What is left, in order.**
1. A decode kernel that wins below sixteen requests. Of the measured
   options, the biggest piece is launches: a cache write fused into the walk,
   and a merge that is not its own latency-bound launch. Next is
   cross-page pipelining in the walk.
2. Failing that, hk's own cache write. It is ~2 us a layer, which with the
   routing is now the whole difference at 1024 tokens.
3. The port to current vLLM, where every number here has to be re-measured.
