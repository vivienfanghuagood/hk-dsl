# vLLM routes: reference

To turn hk on in vLLM and check it, read [hk in vLLM](vllm.md) first. This page is the reference for the routes hk applies inside vLLM, one section each: what it replaces, where it runs, and what it was checked against. What "on by default" means, `--status`, the engine contract and the environment variables are in [integration](integration.md); the generated [route table](supported-hardware.md#engine-routes-on-each) says which routes are on for each GPU.

- [The SDPA patch: vision towers](#the-sdpa-patch-vision-towers)
- [Prefill attention on vLLM's own cache](#prefill-attention-on-vllms-own-cache)
- [The decoder attention backend](#the-decoder-attention-backend)
- [The linear-layer patch](#the-linear-layer-patch)
  - [A model no table has measured](#a-model-no-table-has-measured)
  - [FlyDSL's GEMM as an optional path](#flydsls-gemm-as-an-optional-path)
- [The MoE patch](#the-moe-patch)
- [The 4-bit patch](#the-4-bit-patch)
- [The Quark W4A16 patch](#the-quark-w4a16-patch)
- [The DeepSeek-V4.1 index launch](#the-deepseek-v41-index-launch)

## The SDPA patch: vision towers

The patch rebinds **by identity**, not by name: `_common.rebind` walks `sys.modules` under a prefix and replaces
every attribute that *is* `torch.nn.functional.scaled_dot_product_attention`,
which catches the modules that did `from torch.nn.functional import ...` at
import time and cannot be fooled by a same-named impostor.

The kernel takes bf16, head_dim 64 or 128, no mask and no dropout;
`hk.ops.sdpa` forwards everything else to torch, so a framework that calls it
with something else keeps working ([the drop-in](runtime.md#torch-ops-and-the-sdpa-drop-in)).

**What the patch can and cannot reach in vLLM**, read against vLLM at 1dfe9fb
and checked at runtime by `hk.integration.vllm.probe()`:

* **The decoder's attention is out of reach, structurally.** vLLM V1 keeps KV
  in a paged cache and dispatches through `AttentionImpl.forward(..., kv_cache,
  block_table, ...)` -- `RocmAttentionImpl` or `TritonAttentionImpl` on this
  chip. There is no dense `(B, H, N, D)` tensor on that path for an SDPA
  drop-in to intercept. That is what [the decoder backend](#the-decoder-attention-backend)
  is for.
* **The vision tower's attention is in reach, exactly.**
  `mm_encoder_attention._forward_sdpa` -> `torch.ops.vllm.torch_sdpa_wrapper`
  -> `apply_sdpa` is a literal `F.scaled_dot_product_attention` on a dense
  `(B, H, N, D)` tensor, and patching `F` reaches it because the attribute is
  resolved on the module at call time.
* **It is not the default on RDNA3.** `RocmPlatform.get_vit_attn_backend`
  prefers the Triton flash-attention whenever `flash_attn_triton_available()`;
  `--mm-encoder-attn-backend TORCH_SDPA` selects it deliberately, and that is a
  supported configuration rather than a trick.

`VLLM_ATTENTION_BACKEND` is not the lever: the variable no longer exists in
current vLLM, and `AttentionBackendEnum.TORCH_SDPA` carries the comment "this
tag is only used for ViT".

Which vision towers fit (`python/hk/integration/vllm.py`):

```
CLIP ViT-L/14 (LLaVA, many adapters)   1024 / 16 heads = 64   fits
InternViT-300M                         1024 / 16       = 64   fits
Pixtral                                1024 / 16       = 64   fits
SigLIP so400m                          1152 / 16       = 72   falls back
Qwen2-VL / Qwen2.5-VL                  1280 / 16       = 80   falls back
```

The drop-in counts which branch it took (`hk.ops.sdpa.STATS`), because a patch
that silently falls back produces identical answers and no speedup, which is
indistinguishable from one that works unless something counts.
`tools/hk-bench/vllm_verify.py` is the in-engine verification.

## Prefill attention on vLLM's own cache

With chunked prefill and prefix caching on, a long prompt's time on RDNA is
mostly prefill attention over the paged cache. The route (on by default on gfx1100 and gfx1201; `HK_VLLM_PREFILL=0` turns it off) replaces
only that kernel. The cache, its layout and block size, the cache write, and the
decode of decode-only steps all stay vLLM's. So does the rest of the backend
vLLM picked.

**How to read the numbers in this section.** The ratios below are **kernel-level**: hk's kernel against the
faster of TRITON_ATTN's and ROCM_ATTN's on the same batch, in one process, cold KV. Each one names the PR that
measured it and the first release that shipped it. They decide which batches the route takes; they are not
end-to-end claims. End to end, prefill runs outside CUDA graphs in every vLLM mode, and these are the
measured results, both arms compile + CUDA graphs (vLLM's default), stock in its best mode, released hk 0.9.20:

- MiniCPM-V-4.7-35B-A3B at TP2 on two W7900Ds, replaying real gateway traffic for one replica:
  busy-hour TTFT p90 11.3 s -> 0.24 s, median-hour e2e p50 36.7 s -> 0.94 s.
- Per kernel, the attention's GPU time in a 32k prefill goes from 12.0 s to 0.97 s.

On hybrid models whose shared KV cache is larger than 4 GiB (MiniCPM-V-4.7-1B: about 32 GiB), 0.9.20 leaves
prefill to vLLM; main takes it since #374, which is not yet in a release.

| backend | what is rebound | hk's kernel |
|---|---|---|
| `ROCM_ATTN` (ROCm's default) | `prefix_prefill.context_attention_fwd`, which `chunked_prefill_paged_decode` calls for requests with more than one query | `paged_prefill` `layout="rocm"`: K through a strided global over the x-split cache, V's pages already V^T |
| `TRITON_ATTN` | `unified_attention`, on batches with prefill in them | `paged_prefill` `layout="flash"`, `decodes=True`: the whole batch, a decode row a 16-row fragment |
| `TRITON_ATTN`, decode-only batches (the FULL CUDA graphs) | `unified_attention` at head size 256 (`HK_VLLM_DECODE`, below) | `paged_decode`: the KV axis split across the card |
| `ROCM_ATTN`, decode-only batches, opt-in | `chunked_prefill_paged_decode` at head size 128 (`HK_VLLM_ROCM_DECODE=1`, below) | `paged_decode` `kv_order="rocm"`: the same split, over the x-split K and V^T pages |

Head size 64, 128 and 256. At 256 the kernel uses `dsplit`, a query row
across a pair of warps. Where a kernel does not build spill-free under the
compiler at hand, its calls stay vLLM's: TRITON_ATTN at d256 under clang 22
(ROCm 7.2.3). It builds under the vLLM images' clang 23. gfx1100 and gfx1201, bf16
or fp16 (the prefill calls: fp16-served checkpoints such as GPTQ / AWQ models run
`--dtype float16`), pages of 16 or 32 tokens -- on TRITON_ATTN any multiple of 16: hybrid
models' attention pages (544 tokens for MiniCPM-V-4.7, 784 for Qwen3.8) hold
whole 32-key blocks or halves of them, and are read as zero past a
sequence's end since the rows there belong to the model's other layers. TRITON_ATTN's cache is read in either of vLLM's
orders: K and V planes a block, or a head's K and V side by side (vLLM's
per-layer layouts, token- or head-major). A call
with anything else falls back to vLLM's kernel unchanged and is counted:
ALiBi, sliding windows, sinks, softcap, an fp8 or quantized cache, fp8
output, non-causal, or a call made while a graph is captured.

Nothing is read on the host. The grid is (Q tiles of the longest query,
heads, requests), and a workgroup works out its request from its index.
On TRITON_ATTN the route first transposes the batch's V pages once into a
scratch of V^T pages (`vt_pages`), instead of in every workgroup;
`HK_VLLM_PREFILL_VT=0` turns that off. At head size 256 that kernel also
dispatches a request's Q tiles longest first.
`HK_VLLM_PREFILL_SHADOW=1` also runs vLLM's kernel on every call hk takes and
compares the rows. It is a diagnostic that syncs; don't serve with it.

**Batches the route leaves to vLLM on TRITON_ATTN.** On some batches hk's kernel
is not faster than vLLM's, so the route does not take them (the default-on rule:
never slower on a shape it takes). Two kinds:

- decode rows beside a small prefill: hk's kernel walks each decode's whole
  context in one workgroup, where vLLM's splits it. A 512-token prompt beside
  8 decodes over 4k tokens ran at 0.46-0.73x on a W7900D (kernel-level);
- a short chunk: too few query tiles to fill the card (a 128-token chunk of
  MiniCPM5-2B at 0.91x on a W7900D).

The route reads nothing on the host, so it decides from the sizes it is given:
it bounds the decode work from above ((requests - 1) x the longest context) and
the prefill's from below (the longest query's square over 2), and checks the
query rows times kv heads and the context per query token, against thresholds
measured per GPU and head size (`vllm_prefill.GUARDS`), set so that every
measured batch it takes is at least 1.05x the faster of TRITON_ATTN's and
ROCM_ATTN's kernels (`tools/hk-bench/results/2026-11-03/`). Such a batch goes to
vLLM's kernel unchanged and is counted as `declined`; the first one prints one
line. `HK_VLLM_PREFILL_GUARD=0` turns the guard off, to measure it.

**A mixed batch with a sizable prefill is split.** When the step's exact sizes
are known (vLLM's TRITON_ATTN metadata builder hands them over, host arrays only),
a batch declined for its decodes is taken in two parts. hk's kernel, built to
skip one-query requests, computes the prefill rows. vLLM's own kernel computes
the decode rows, as a batch of one query a request; it splits a decode's
context, which hk's prefill kernel does not. Everything stays on the device: no
sync, two more launches and a gather. The split costs those launches and gives the
decode rows to TRITON_ATTN's decode, which ROCM_ATTN's beats, so it is taken only
when the longest query has at least 1024 tokens and the decodes' contexts sum to
at most a measured share of the prefill's work (`vllm_prefill.SPLITS`). Measured
against the faster of TRITON_ATTN's and ROCM_ATTN's kernels on 68 shapes a model
and on the chunks vLLM's default 2048-token budget makes (1024-2040 tokens, first
or after a prefix, beside 4-16 decodes;
`tools/hk-bench/results/2026-11-04/<gpu>/prefill-split-sweep.json`; kernel-level, #249, first released in
0.9.1), every batch it splits is faster:

| GPU | MiniCPM5-2B (d128) | Qwen3.8-27B, MiniCPM-V-4.7 (d256) |
|---|---|---|
| gfx1100 (W7900D) | 28 batches, 1.08-2.59x | 62 batches, 1.40-5.81x |
| gfx1201 (R9700) | 17 batches, 1.33-2.58x | 50 batches, 1.23-2.81x |

With prompts of 128-512 tokens, and on gfx1201 with a first chunk beside heavier
decodes, some mixed batches were slower split (0.26-1.03x), so those stay declined. A split call is counted as `hk` and as `split`;
`HK_VLLM_PREFILL_SPLIT=0` turns the split off, and the guard then declines the
whole batch as before.

**A short chunk over a long context is split along the KV axis.** The new tokens
of a request whose prompt is mostly a prefix-cache hit, or a chunked prefill's
last chunk, are a few query rows over thousands of keys: a few query tiles, so
hk's prefill kernel and vLLM's both leave most of the card idle, and the guard
declined such calls. With the step's exact sizes, the route now takes them with
`hk.ops.paged_prefill`'s `kv_splits`: each query tile walks the context in
runs, a workgroup each, writes its normalised partial and (m, l), and
`paged_prefill.merge_kernel` combines a request's runs. Decode rows in the same
batch go to vLLM's kernel, as in the split above. It is taken where the longest
query is short and the longest sequence long enough, per GPU and head size
(`vllm_prefill.KVSPLITS`), from a sweep of every tile size and split count that
builds against TRITON_ATTN; the route as served was then measured against the
faster of TRITON_ATTN's and ROCM_ATTN's kernels on 36 shapes a model (contexts
of 1k-32k tokens, chunks of 16-2048, batches of short chunks, chunks beside
decodes; `tools/hk-bench/results/2026-11-05/<gpu>/prefill-kvsplit-sweep.json`),
and every call it takes clears the default-on rule's bar there. A KV-split call
is counted as `hk` and as `kvsplit`; `HK_VLLM_PREFILL_KVSPLIT=0` turns it off.

The shortest chunks over the longest contexts take GQA-packed tiles
(`paged_prefill`'s `pack`): a workgroup holds one kv head's query heads of 16
tokens instead of one query head's rows, so a kv head's K and V are read once a
tile rather than once a query head, and a 16-token chunk fills its tile. Which
calls take them is per GPU, head size and query heads a kv head
(`vllm_prefill.KVS_PACKED`), from the same kind of sweep against the unpacked
plan; the more query heads share a kv head, the further it reaches. Beside
decodes a packed call takes the batch too, the decode rows going to vLLM's
decode kernel apart, up to a measured number of decodes per GPU and model
shape (`KVS_PACK_MAX_DECODES`, with the decodes' work at most
`KVS_PACK_DEC_RATIO` times the prefill's): vLLM's mixed-batch kernel walks a
decode's context unsplit, so a short chunk beside long decodes is several
times faster this way.
`HK_VLLM_PREFILL_KVS_PACK=0` keeps the unpacked tiles (for measuring).

**On ROCM_ATTN no batch is declined.** ROCM_ATTN hands the prefix-prefill only
its requests of more than one query; its own paged decode keeps the decode rows.
So neither kind of batch above reaches hk's kernel there. Measured in one process
against ROCM_ATTN's own prefill (`attention_bench.py prefill --arms hk_rocm,rocm_attn`,
68 shapes a model: prompts, chunks after a prefix, prompts beside 1-16 decodes over
1k-16k tokens; kernel-level, #247, first released in 0.9.1), every shape is faster:

| GPU | MiniCPM5-2B | Qwen3-8B |
|---|---|---|
| gfx1100 (W7900D) | 1.54-5.35x, geomean 3.08x | 1.16-5.79x, geomean 2.75x |
| gfx1201 (R9700) | 1.08-3.07x, geomean 1.90x | 1.02-3.05x, geomean 1.91x |

(`tools/hk-bench/results/2026-11-04/`). The thinnest are a short prompt beside
4-16 decodes over 4k-16k tokens, where ROCM_ATTN's own decode, which the route
leaves alone, is most of the batch's time. MiniCPM-V-4.7's 544-token pages are not
taken on ROCM_ATTN (pages of 16 or 32 only, above), so they stay vLLM's there.

### Decode-only batches on TRITON_ATTN

A decode step reads the whole KV cache and does one query row a head of maths
on it. vLLM's Triton kernel cuts each sequence into 16 segments, so at 1-4
requests most of the card is idle. `hk.ops.paged_decode` reads the same cache
in place and cuts the KV axis into a number of splits planned from the batch
alone. A split's range is arithmetic on the device's sequence length, so the
count a CUDA graph was captured with is right for every replay. A second
launch merges the splits.

- `HK_VLLM_DECODE` unset: head size 256 at every batch. Those are hybrid
  models' attention layers (Qwen3.8, Qwen3.6, MiniCPM-V-4.7): there hk is
  1.02-4.0x (kernel-level; #220, hk 0.9.0) the faster of TRITON_ATTN's kernel and ROCM_ATTN's (which runs an
  unsplit Triton fallback on pages that are not a power of two).
- `HK_VLLM_DECODE=1`: head size 128 too, opt-in. It is for one or two
  requests decoding over long contexts -- a single user, local inference --
  and is kept as groundwork for that. It takes up to 4 (request, kv head)
  pairs (one or two requests of MiniCPM5-2B), with as many splits as the GPU
  has WGPs. Serving MiniCPM5-2B on a W7900D against TRITON_ATTN's decode, one
  request, both arms under CUDA graphs (#239, hk 0.9.0): TPOT 1.08x at 8k tokens, 1.20x at 16k, 1.40x at 32k. The rule
  cannot look at the context: a decode step runs as a CUDA graph captured
  once per batch size, so one or two requests over a short context are taken
  too, at about 1.03x (4k). Four requests were 0.98x at 4k-8k and stay
  vLLM's. It stays off by default because against ROCM_ATTN's paged decode
  the kernel is 0.99-1.27x at one request, and about 1.10x over a measured
  real serving mix, under the 1.15x the default-on rule asks for.
- `HK_VLLM_DECODE=0`: off.

### Decode-only batches on ROCM_ATTN (opt-in)

`HK_VLLM_ROCM_DECODE=1` sends ROCM_ATTN's decode-only calls at head size 128
to `hk.ops.paged_decode` reading ROCM_ATTN's own cache (`kv_order="rocm"`):
K's x-split pages and V's per-page V^T, both staged through LDS. A workgroup
is one warp, and the split count keeps four workgroups on every WGP
(`paged_decode.plan_rocm`: 96 splits at one request of a 2-kv-head model on a
W7900D, 48 at two, 24 at four), from the batch alone, so the CUDA graph a
batch size is captured with is right for every replay. It takes up to 16
(request, kv head) pairs, eight requests of MiniCPM5-2B; at 16 requests vLLM's
kernel is as fast. gfx1100 only.

On a W7900D with MiniCPM5-2B, against ROCM_ATTN's paged decode in one process
(`attention_bench.py decode --arms rocm_attn,triton,hk_rocm`; kernel-level, #302, hk 0.9.10): one request
1.19-1.29x from 16k tokens, 0.95x at 4k; two to eight requests 1.01-1.25x.
Serving (end to end), two engines in one process, both under CUDA graphs: TPOT 1.01x at one
request over 2k-8k tokens, 1.05x at 32k, 1.07x at 64k, 1.02-1.03x at two to
eight requests; TTFT unchanged. It stays opt-in: weighted by a measured real
serving mix the kernel is 1.10-1.14x against the faster stock kernel, under
the 1.15x the default-on rule asks for.

`tools/hk-bench/attention_bench.py` measures every shape against both stock
kernels (`tools/hk-bench/README-attention.md`).

## The decoder attention backend

vLLM V1 keeps KV in a paged cache and dispatches through `AttentionImpl`, so
the way in is to *be* a backend. `hk/ops/paged.py` is the kernel and
`hk/integration/vllm_backend.py` registers it, through
`vllm.v1.attention.backends.registry.register_backend`.

**Which slot.** Four facts about vLLM's plugin surface decide it, each written
down in `vllm_backend.py` because each cost a run to find:
1. `AttentionBackendEnum` is closed, and `register_backend` overrides a
   member rather than adding one.
2. The reserved `CUSTOM` member is rejected by `RocmPlatform`'s allowlist, so
   the override has to target a member the platform accepts.
3. That member is `TRITON_ATTN`, which is also ROCm's default -- hence the
   separate `HK_VLLM_BACKEND=1` opt-in.
4. `get_name()` must answer with the member it overrides, because vLLM
   round-trips the name through the enum.

`HK_VLLM_BACKEND_SLOT`, or a value of `HK_VLLM_BACKEND` other than
`1`/`on`/`yes`/`true`, names a different member to register over.
`register()` reports the slot it took, as
`HK_ATTN registered over TRITON_ATTN (--attention-backend TRITON_ATTN)`.

**The cache layout decides the kernel.** `gl` derives its strides from its
dims, so vLLM's default `(block, token, head, dim)` cannot yield a contiguous
`(block_size, head_dim)` tile for one head. A backend declares its own shape,
so this one declares `(2, num_blocks, num_kv_heads, block_size, head_size)` --
exactly `gl`'s `(b, d, r, c)`, so one page of one head is a plain tile load.
The page size must be the kernel's KV block, 32, and the head size one the
kernels have, 64 or 128; anything else is refused with a sentence, as are
ALiBi, sliding windows, sinks, a logits soft cap and a quantized KV cache.

**Every step's three paths.** Each step writes its new tokens' K/V with
vLLM's own `reshape_and_cache_flash`, through a `(blocks, block, heads, d)`
view of the head-major pages (it takes its strides from the tensor, and skips
the slot -1 a CUDA graph's padding carries). Then, with the batch ordered
decodes first by the builder:

* **Decode, up to eight requests**: vLLM's own Triton kernel
  (`unified_attention`, its 16-segment 3-D path), reading the head-major
  cache through the same transposed view. That is hk's decode losing, and it
  is recorded as such.
* **Decode, sixteen requests and up**: `hk.ops.paged`'s split walk and,
  with more than one split, its merge, reading the engine's query and
  writing its output in place. The split count is planned from the batch
  size and max_model_len alone (about 256 workgroups), so a graph captured
  on any batch is right for every replay. On a compact cache the two
  kernels are close; in vLLM, whose cache is one allocation of tens of GiB,
  Triton's 2-D kernel reads the same pages more slowly past about 2 GiB and
  hk's walk does not, which is why the walk takes these batches.
  Batches are padded to a capture size first, so 9-15 requests run as 16, on hk.
  `vllm_backend.TRITON_DECODE_MAX` is the boundary; 0 turns the routing off.
  `HK_TRITON_DECODE_ALSO="17-24,49-56"` gives Triton further ranges of
  batch sizes; off by default, because in the captured step Triton's
  kernel at 15-22 requests and 4096 tokens took (kernel-level, #13, hk 0.6.0) 0.877 ms a call under
  triton-rocm 3.7.1 (1.93 under Triton 3.8.0) against hk's 0.449, though on
  a compact bench cache it is the faster at 24.
* **Prefill**: `hk.ops.paged_prefill`, one launch for every prefill request
  in the step, reading K/V through the page table. A cached prefix or an
  earlier chunk is just context in front of the new tokens, so prefix caching
  and chunked prefill work as vLLM ships them. On gfx950, when the step's
  query tiles times its heads come short of the machine (a short chunk on a
  long prefix), the builder cuts each tile's walk over its cached prefix
  (`paged_prefill.split_plan`, from vLLM's host lengths) and a second launch
  merges the partials.

On gfx950 (MI355X) the same paths run gfx950's kernels (`hk.ops.paged_cdna`
and the paged form of `hk.ops.attn_cdna`), with the same page, cache layout
and capture, and hk's decode takes every batch (on this cache it is behind
Triton's kernel only at short contexts with one to four requests).
The planner aims at 256 one-warp workgroups there, a split taking at least
four pages.

`HK_SPLITS` is the one debugging switch: `off`, `0` or `1` forces one split;
`auto` or `plan`, the default, uses the planner; an integer forces that many,
capped by the page count and 32. Its comment records a fault from before the
decode was a captured launch -- under vLLM at batch 16 and context 4096 the
second `generate` aborted with `HSA_STATUS_ERROR_EXCEPTION 0x1016`, and
forcing one split made it go away -- which the current path has not
reproduced.

## The linear-layer patch

On gfx1100, vLLM 0.16.1's unquantized GEMMs all end in
`torch.nn.functional.linear`, which is hipBLASLt, because its skinny-GEMM
path is gated on gfx9.
vLLM 0.31 runs those skinny GEMMs on gfx11 too, at M ≤ 5. The routing table
is measured against whichever path the installed vLLM takes.
`hk/integration/vllm_gemm.py` installs the replacement, on its own switch
(`HK_VLLM_GEMM`, on by default on gfx1100 and gfx1201). It rebinds `dispatch_unquantized_gemm` by
identity in every vLLM module that imported it, and what that returns calls a
single opaque custom op, `hk_vllm::linear`, with a fake implementation. Under
vLLM's torch.compile the token count is symbolic, so branching on it inside
traced code would be a guard and a recompile; inside the op it is neither, and
CUDA-graph capture sees a launch on torch's stream.

On gfx1100 each call takes one of three paths, counted in `vllm_gemm.STATS`:
`hk` (the prefill GEMM), `hk_decode` (the decode GEMV, for small M) or
`fallback` — vLLM's own `rocm_unquantized_gemm` where it has one, else
`F.linear`. Which one is a [routing table](integration.md#routing-tables). A table may
also send a call to FlyDSL's GEMM (`flydsl`, below), and on gfx950 to
HipKittens' CDNA4 GEMM (`hipkittens`,
[design note](archive/design/cdna-hipkittens-ops.md)). A call also falls back when it is
not bf16, has a bias, has a non-contiguous weight, does not tile, or runs on a
device of another arch. With no table for the current arch the patch is not
installed at all, and says so.

### A model no table has measured

The shipped table is Qwen3-8B's, and a (N, K) it never measured takes vLLM's
path. So the patch measures the model it is given, on the machine it runs on
(`hk/integration/vllm_autotune.py`):

- **When.** It wraps vLLM's GPU worker's `load_model`. After the weights load
  it walks the model vLLM built, every unquantized bf16 linear layer and the
  LM head, sharded as this rank holds them. The shapes the table lacks are
  measured before vLLM's profile run sizes the KV cache, and before any CUDA
  graph is captured. The tuner's buffers are freed before the profile run.
- **What.** The tuner behind `route_tune.py` (`hk.autotune.routes`), against
  the installed vLLM's own op. It measures every decode M (1-64) of every
  missing shape first, largest weight stream first, then the prefill grid.
  A shape whose calls all carry a bias is not measured, since those calls
  fall back anyway.
- **How it times.** As a captured decode step runs the layer (`measure_points`,
  `timing="context"`): each candidate's calls are replayed from a CUDA graph,
  each one after a norm-like kernel that writes its input and before the
  model's other linear layers, on the routes they take at that M, each with
  an input kernel of its own. Every shape rotates through 576 MB of its own
  weight copies, so none is read from the Infinity Cache. The context's own
  graph is timed in the same rounds and taken off. A shape timed alone
  flatters a tuned hipBLASLt kernel, which runs faster back to back than
  after other kernels, while hk's decode GEMV does not change: on an R9700
  with MiniCPM5's TunableOp results the alone-timed tuner left M = 8 on
  vLLM, where the routes context timing picks shorten the served step. Once
  every decode M is in, the Ms whose context moved (another shape's route
  changed) are timed again, until none moves. The LM head, one call a step,
  is timed alone.
- **TunableOp.** Which kernel vLLM's GEMM runs at an M is what TunableOp's
  results file says. A record keeps the file's name and a hash of its
  contents (`meta.tunableop`, "off" without TunableOp), and a record
  measured under another state is measured again.
- **How long.** `HK_AUTOTUNE_BUDGET` seconds, 60 by default. A shape whose
  grid the budget cut short routes only up to the largest M measured; the
  next start carries on. Context timing costs more a point than a shape timed
  alone: on an R9700, MiniCPM5-2B's decode Ms took three 60-second starts, so
  its first start routed little. Run `python -m hk.integration tune` (below)
  before the server to start on the full routes. The patch's report line says what was measured,
  how long it took, and which shapes stayed on vLLM's path.
- **Where it goes.** This machine's records: `$HK_TUNE_DIR` (by default
  beside the compile cache), `routes_vllm_linear_<arch>.json`. Each shape
  there replaces the shipped entry for the same (N, K). The next start of
  the same model finds every shape and measures nothing.
- **Once per machine.** Writers hold a file lock. Tensor-parallel ranks, or
  two servers, measure a shape once: the others wait, then read it.
- **Then every layer is decided.** Each unquantized linear layer and the LM
  head is looked up once against the table it will run on. A layer the table
  routes at no M keeps vLLM's own GEMM with nothing of hk's in its path: an
  fp16 or biased weight, or a shape every measured M of which fell back. A
  wrapper there would still be an op of its own in every captured decode
  step. `[hk] linear layers at load: N through hk's op; unrouted at load: ...`
  says how many of each, by reason (`vllm_gemm.UNROUTED`). With
  `HK_AUTOTUNE_BUDGET=0` the same decision is taken on each layer's first call.

To pay the cost before a server starts instead, on a quiet card and with no
budget unless one is given:

```bash
python3 -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct          # --tp N, --force, --dry-run
```

It reads the model's `config.json` (a path, the local HF cache, or the hub)
and writes the same records. `HK_AUTOTUNE_BUDGET=0` turns the measuring at
load off. `HK_AUTOTUNE=0` ignores the records as well. A table named by
`$HK_VLLM_ROUTES` is used alone, and nothing is measured against it.

The patch also points `VLLM_CACHE_ROOT` at a `hk-linear` subdirectory.
vLLM keys its compiled graphs, including the AOT cache, by config and
source, not by what is patched. A patched engine sharing a cache root with
an unpatched one loads back graphs whose linear layers never call the op.

### FlyDSL's GEMM as an optional path

hk-dsl sits beside FlyDSL rather than replacing it: FlyDSL's RDNA3 GEMM is
faster than hk's at some shapes, and hk's decode GEMV and attention at
others. So the op
has another path, `flydsl`, which calls FlyDSL's dispatcher
(`rdna3_gemm_dispatch`: its tile heuristic and per-tile options, no
measurement) where a routing table measured it fastest
(`hk/integration/flydsl_gemm.py`).
- **Optional.** hk does not depend on FlyDSL. The path exists when `flydsl`
  is importable and so is FlyDSL's `kernels` package, which is in FlyDSL's
  repository, not its wheel: put a checkout of the installed version on
  `sys.path`, or name it in `FLYDSL_ROOT`. Where HuggingFace's unrelated
  `kernels` package is installed (beside SGLang), FlyDSL's is imported from
  `FLYDSL_ROOT` beside it, and `import kernels` still finds HuggingFace's;
  without `FLYDSL_ROOT` the patch's report line says the package is
  shadowed. `HK_FLYDSL=0` turns the path off.
- **Only at the tile that was measured.** FlyDSL compiles M into the kernel,
  so its tiles must divide the shape (M a multiple of 32 at least), and its
  speed depends on which tile M leaves it: an M that fits its large tile runs
  close to hk's time, one that fits only 64-row tiles much slower. A table's FlyDSL rows therefore name the tile, and a call takes the
  path only if FlyDSL's dispatcher would pick that tile for it.
- **Everything else goes where the table without FlyDSL sends it.** A
  bf16, contiguous call whose M no tile divides, at another tile, or on an
  install without FlyDSL, takes the table's own path, hk's or the engine's.
- **Costs.** Each new M is a FlyDSL compile, 0.4-0.8 s cold and 0.04 s from
  FlyDSL's disk cache. A call costs about 60 us of host time through
  FlyDSL's launch path, against 17.8 us for hk's op. Both are hidden behind a GEMM
  at the sizes where FlyDSL wins. The launch is on torch's stream with no
  host sync, so CUDA graphs capture it (`tests/hk/gpu/test_flydsl_gemm.py`).

The shipped Qwen3-8B table carries FlyDSL rows, mostly at decode-sized Ms,
each where FlyDSL measured faster than hk and the engine at that point;
SGLang's table carries FlyDSL rows too.

## The MoE patch

vLLM's unquantized MoE sorts each batch's (token, choice) pairs by expert,
then makes each of the layer's two GEMMs one call of
`invoke_fused_moe_triton_kernel` on the sorted lists.
`hk/integration/vllm_moe.py` rebinds that name, plus the
`try_get_optimal_moe_config` that sizes the sort's blocks, by identity in
every vLLM module that imported them:
- **Up to `BLOCK16_MAX_TOKENS` (256) tokens a step**, the config asks for
  16-row blocks and hk's GEMM (`hk.ops.moe`) runs both GEMMs. An arch can
  have its own limit (`BLOCK16_MAX_TOKENS_AT`): gfx1201's is 2, where vLLM
  skips the sort and hk streams the few chosen experts faster than Triton;
  from there Triton's kernel streams them at GDDR6's rate too.
- **Above that, or for a quantized, biased or otherwise untaken GEMM**,
  vLLM's own kernel runs.

The counters are `vllm_moe.STATS`. The MoE layer is a custom op the compiler
does not trace into, so the stand-in runs on every call, and the graphs
vLLM captures for decode record hk's launches.

## The 4-bit patch

On ROCm, vLLM runs an AutoAWQ checkpoint through `AutoAWQLinearMethod`. Its
GEMM is the Triton AWQ kernel: the C++ one is not built there, so
`VLLM_USE_TRITON_AWQ=1` has to be set. That kernel takes fp16 activations
only, so an AWQ model is served with `--dtype float16`.
`hk/integration/vllm_int4.py` wraps the method:
- after vLLM's own weight processing, each layer also gets hk's 4-bit layout
  (`hk.ops.w4a16.repack_awq`), a second copy of the codes;
- `apply` calls the opaque op `hk_vllm::w4a16`, inside which batches of up
  to 64 tokens take hk's decode GEMM, and the rest vLLM's own path on vLLM's
  own tensors;
- on gfx1201, fp16 batches of 65-512 tokens (prefill) take hk's large-batch
  W4A16 form instead, reading hk's AWQ layout with its zero points.
  `HK_VLLM_INT4_LARGE_MAX_M` moves the 512, and `0` turns the form off. gfx1100
  keeps vLLM's path there: its dequantize-and-matmul is as fast.

The counters are `vllm_int4.STATS`. The patch changes what compiled graphs
call, so it uses the compile-cache root `hk-int4`.

```bash
VLLM_USE_TRITON_AWQ=1 HK_PATCH_VLLM=1 HK_VLLM_INT4=1 \
    vllm serve Qwen/Qwen3-8B-AWQ --dtype float16
```

A 4-bit symmetric GPTQ checkpoint runs on RDNA3 through `AutoGPTQLinearMethod`
with `RDNA3W4A16LinearKernel`, vLLM's HIP kernel on GPTQ's shuffled codes.
`hk/integration/vllm_gptq.py` wraps that method the same way:
- each layer gets hk's layout (`hk.ops.w4a16.repack_gptq`) *before* vLLM's
  weight processing, which shuffles the codes in place. It is `load_w4`'s
  symmetric form (`repack_gptq(..., symmetric=True)`): zero point 8, no
  zero-times-scale tensor, scales in 16-row blocks;
- `apply` calls the opaque op `hk_vllm::gptq_w4a16`, inside which batches of
  1 to 64 tokens take hk's decode GEMM, and the rest vLLM's kernel. At one
  token hk's is one kernel a call, where vLLM's dispatches a fill before each
  GEMM.
- Act-order checkpoints (`desc_act`), other quant types, layers on another
  kernel and layers whose stored zero points are not 8 keep vLLM's path;
  `vllm_gptq.REFUSED` counts them by reason.

```bash
HK_PATCH_VLLM=1 vllm serve <a GPTQ-Int4 checkpoint> --dtype float16    # HK_VLLM_GPTQ=0 turns it off
```

## The Quark W4A16 patch

AMD Quark int4 weight-only checkpoints (for example
amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16) load through vLLM's `QuarkW4A16Int4`
scheme, which hands each layer to the same MPLinearKernel GPTQ layers get. The
int4 routes above hook AutoAWQ and AutoGPTQ, so they never see these layers;
`hk/integration/vllm_quark.py` (`HK_VLLM_QUARK`) wraps
`QuarkW4A16Int4.process_weights_after_loading` and `.apply_weights`:
- **gfx1201** (`RDNAHybridW4A16LinearKernel`, on by default): after vLLM's
  repack, hk reads its tensors in place. The fp16 batches vLLM would run on its
  Triton GEMM go to hk's 4-bit forms, the decode GEMM up to 64 tokens and the
  large-batch form above that; vLLM's own HIP skinny GEMM keeps its batches.
- **gfx1100** (`RDNA3W4A16LinearKernel`, on by default; opt-in in 0.9.15): hk writes its
  symmetric layout of Quark's codes first (`hk.ops.w4a16.repack_quark`), and
  16-bit batches of 1-64 tokens take hk's decode GEMM.
- Asymmetric checkpoints, other kernels, and shapes the forms do not take keep
  vLLM's path; `vllm_quark.REFUSED` counts them by reason.

Checked end to end on amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 on one R9700 (fp16,
TRITON_ATTN) and on one W7900D: GSM8K-200 equal to stock's on both. The measured numbers are in the
[switch's evidence](integration.md#on-by-default-and-turning-it-off).
`HK_VLLM_QUARK=0` turns the route off; `=1` turns it on where it is off by default.

## The Gemma RMSNorm route

vLLM's `GemmaRMSNorm` -- y = x_hat * (1 + w), the norm of Gemma, of the Qwen3.5 family and of
MiniCPM-V-4.7's text models -- has no fused kernel on CUDA or ROCm: its `forward_cuda` is its
`forward_native`, vLLM IR ops that run as PyTorch ops. Under torch.compile inductor fuses them; a
model served eager (`enforce_eager`, as MiniCPM-V-4.7's production configuration is) runs every
norm as about eight kernels, and a decode step at one request is bound by their launches:
MiniCPM-V-4.7-1B issues ~1060 kernels a token, 61 of them norms of ~8 ops, and the GPU idles ~70%
of the step. (`compilation_config.custom_ops=["+rms_norm"]` does not reach this class.)

`hk/integration/vllm_norm.py` (`HK_VLLM_NORM`, on by default on gfx1100 and gfx1201) wraps
`GemmaRMSNorm.forward_cuda`: hk's weighted norm kernels at weight offset 1 (the stored w, the one
added in fp32 inside the kernel, as vLLM's math does), the residual add fused, out of place as
vLLM's native path ((y, x + residual) or y), on torch's current stream. Inputs hk's kernels do
not take (not a contiguous 16-bit tensor, a weight of another dtype or shape, a residual unlike x)
keep vLLM's path; `vllm_norm.STATS` counts them by reason.

Under torch.compile (vLLM's default when `enforce_eager` is off) vLLM traces the layer's
`forward_native` and inductor generates a kernel for the norm. `HK_VLLM_NORM=compile` routes those
calls too: the route patches `forward_native` and calls hk's kernels through two opaque custom
ops, `torch.ops.hk_vllm.gemma_rms_norm` and `gemma_add_rms_norm`, so the compiled graph has one
node per norm (dynamic rows, no recompiles) and a CUDA graph captures one kernel with no host
sync. The engine gets a compile cache of its own (`.../hk-norm`). On a W7900D, with a linear
between norms as in a decoder layer and both sides compiled and captured, hk's norm is 4.4-7.5 us
faster than inductor's (hidden 1024 and 2048, 1 to 4,096 rows), closer to float64; this is
opt-in until it is measured end to end.

**Eager only.** The gains below are measured with `--enforce-eager` in both arms, as MiniCPM-V-4.7's production
deployment runs, and they come from launch overhead: in eager mode each norm is about 8 PyTorch kernels, and
hk's fused kernel replaces them. Under compile + CUDA graphs, vLLM's default and about 5x faster at one request
for these models, inductor already fuses the norm. There the route is off by default, and
`HK_VLLM_NORM=compile` changed TPOT by about 2% (MiniCPM-V-4.7-1B, W7900D, hk at #372 on top of 0.9.20;
not a claim). Quote these numbers only as eager-mode numbers.

Checked on MiniCPM-V-4.7 with its production args (eager) on W7900Ds, hk 0.9.18, the route off and on,
one process, engines rebuilt per block, 4 alternating rounds (ranges disjoint):

| model | TPOT at 1 / 4 / 8 requests | TTFT 8k | GSM8K-300 (off / on) |
|---|---|---|---|
| 1B | 21.39 -> 16.70 ms (1.28x) / 1.21x / 1.24x | 1.07x | 81.7 / 82.7% (McNemar p 0.65) |
| 35B-A3B, TP2 | 59.93 -> 52.28 ms (1.15x) / 1.11x / 1.10x | 1.06x | 96.0 / 97.0% (p 0.25) |

On an R9700 (gfx1201), eager in both arms, hk 0.9.19, the same design with 6 alternating rounds (ranges disjoint):

| model | TPOT at 1 / 4 / 8 requests | TTFT 2k / 8k | GSM8K-300 (off / on) |
|---|---|---|---|
| MiniCPM-V-4.7-1B | 21.41 -> 16.80 ms (1.27x) / 1.25x / 1.20x | 1.12x / 1.10x | 82.0 / 81.3% (p 0.79) |
| Qwen3.5-9B | 29.95 -> 27.29 ms (1.10x) / 1.09x / 1.10x | 1.09x / - | 95.0 / 94.7% (p 1.00) |

Qwen3.5-9B gains less because its larger GEMVs take a bigger share of the step than the norms do.

On the real calls of a request every output is within 1 bf16 ulp of vLLM's native path (0.10% of
a call's elements at most) and every residual bit-equal. Qwen3.8-27B also has GemmaRMSNorm but is
served compiled, where the route leaves the norm to inductor. `HK_VLLM_NORM=0` turns the route off.

## The DeepSeek-V4.1 index launch

With `--monokernel`, vLLM runs a DeepSeek-V4.1 decode step's backbone layer
as dsv41_mono's two persistent launches. Between them, each index layer (a
compressor and the sparse-attention indexer) runs vLLM's own chain,
`mono_decode._select`: 9-12 kernels. They are the compressor's projection,
ring, RMSNorm and compressed-cache insert, the indexer's K store, qr's MXFP8
quant and the wq_b GEMM, weights_proj, the Q RoPE / MXFP4 quant, then the
logits and the top-k. `hk/integration/vllm_dsv41_index.py` replaces that
function with one resident launch, `hk.ops.dsv41_index`, at decode steps of up
to 8 rows:
- **The whole chain, top-k included**, while a row fits the launch (the
  indexer's `max_model_len` up to `WIDTH_MAX`, 16640: its keys in LDS). While
  a row's 8-token blocks fit the indexer's candidate list (2048 of them, 16384
  tokens), the candidate source's block top-k keeps every block and the
  consumers' pool is the whole row, so both reduce to the dense selection.
  Past that the launch does the source's block top-k itself (the top 2048
  block maxima, `select_row` on the block keys), and a consumer selects among
  the source's candidate blocks only.
- **Past that, up to the logits' inputs**: vLLM's indexer op takes the
  launch's q, q_scale and head weights.
- **Anything else runs vLLM's `_select` unchanged**: more rows, a DSpark draft
  layer, a layer whose shapes or metadata the launch does not know.

The launch's mailbox tags come from mono's runner epoch, which every MoE launch moves,
so steps and layers never read each other's values, CUDA-graph replays
included.

The route patches a module vLLM imports late, with the model. The plugin
therefore installs an import hook rather than importing the model package
itself, which would cost every process seconds. The contract is read from the
module's source (`contract._vllm_dsv41_index`), and the signature is checked
again when the hook patches it.

What holds it, on vLLM e68efc685 (MI355X):
- **The captures.** On steps captured from a real-weight TP2 run (~3.5K-token
  prompts) at every layer kind, the launch was compared with vLLM's chain
  replayed on the same rows. The kinds are a dense source at ratio 2, the
  candidate source at ratio 1, and the consumers. The ring, the
  compressed-cache records, the indexer K cache, q and q_scale are byte for
  byte vLLM's. The head weights are exact. The top-512 sets and the source's
  candidate blocks equal the replay's on 8 of 8 rows.
- **In vLLM.** `HK_DSV41_INDEX_CHECK=N` runs vLLM's chain after the launch
  on the first N eager calls of each layer, then prints what differs. On all 8
  index layers the caches, q and q_scale are byte-identical. The top-k sets
  are equal at M=1; at M=8 they overlap at least 0.994, because vLLM's
  weights_proj GEMM rounds a few head weights one bf16 ulp differently.
- **The tests.** `tests/hk/gpu/test_vllm_dsv41_index.py` (the gfx950 vLLM
  tier) builds synthetic layers of each kind and checks the same bytes against
  vLLM's own kernels, at 1 and 8 rows and at max_model_len 16640's widths.
  One difference shows up there that the captures never hit. The launch sums
  the compressor's projection over its K split in a fixed order, not
  torch.mm's, so the ring's fp32 rows agree within about 1e-7. Now and then a
  latent element then rounds to the neighbouring bf16, which moves one value
  of a compressed-cache record by one code (1 byte in 8 records at 8000-token
  rows).

```bash
vllm serve <DeepSeek-V4.1-Flash> --monokernel --tensor-parallel-size 2 ...   # on by default on gfx950
HK_VLLM_DSV41_INDEX=0 vllm serve ...                    # vLLM's own chain
HK_DSV41_INDEX_TOPK=0 vllm serve ...                    # the launch without the logits / top-k
HK_DSV41_INDEX_CHECK=3 vllm serve ... --enforce-eager   # the in-place check
```

Serving results are on [the scorecard](performance/scorecard.md). With
`--monokernel` and hk's default routes, DeepSeek-V4.1-Flash's real-weight
GSM8K-1319 (greedy) agrees with stock's run-to-run noise (McNemar p 0.39-1.00
against stock).

The counters are `vllm_dsv41_index.STATS` (`hk`, `stock`); the layers that
stay on vLLM's chain, and why, are in `STOCK_LAYERS`.
