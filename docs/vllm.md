# hk in vLLM

The routes hk applies inside vLLM, one section each: what it replaces, where it runs, and what it was checked against. What "on by default" means, `--status`, the engine contract and the environment variables are in [integration](integration.md); the generated [route table](supported-hardware.md#engine-routes-on-each) says which routes are on for each GPU.

- [The SDPA patch: vision towers](#the-sdpa-patch-vision-towers)
- [Prefill attention on vLLM's own cache](#prefill-attention-on-vllms-own-cache)
- [The decoder attention backend](#the-decoder-attention-backend)
- [The linear-layer patch](#the-linear-layer-patch)
  - [A model no table has measured](#a-model-no-table-has-measured)
  - [FlyDSL's GEMM as an optional path](#flydsls-gemm-as-an-optional-path)
- [The MoE patch](#the-moe-patch)
- [The 4-bit patch](#the-4-bit-patch)
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
mostly prefill attention over the paged cache. `HK_VLLM_PREFILL=1` replaces
only that kernel. The cache, its layout and block size, the cache write, and the
decode of decode-only steps all stay vLLM's. So does the rest of the backend
vLLM picked.

| backend | what is rebound | hk's kernel |
|---|---|---|
| `ROCM_ATTN` (ROCm's default) | `prefix_prefill.context_attention_fwd`, which `chunked_prefill_paged_decode` calls for requests with more than one query | `paged_prefill` `layout="rocm"`: K through a strided global over the x-split cache, V's pages already V^T |
| `TRITON_ATTN` | `unified_attention`, on batches with prefill in them; decode-only batches (the FULL CUDA graphs) keep vLLM's | `paged_prefill` `layout="flash"`, `decodes=True`: the whole batch, a decode row a 16-row fragment |

Head size 64, 128 and 256. At 256 the kernel uses `dsplit`, a query row
across a pair of warps. Where a kernel does not build spill-free under the
compiler at hand, its calls stay vLLM's: TRITON_ATTN at d256 under clang 22
(ROCm 7.2.3). It builds under the vLLM images' clang 23. gfx1100, bf16, pages of 16 (and 32 on ROCM_ATTN). A call
with anything else falls back to vLLM's kernel unchanged and is counted:
ALiBi, sliding windows, sinks, softcap, an fp8 or quantized cache, fp8
output, non-causal, or a call made while a graph is captured.

Nothing is read on the host. The grid is (Q tiles of the longest query,
heads, requests), and a workgroup works out its request from its index.
`HK_VLLM_PREFILL_SHADOW=1` also runs vLLM's kernel on every call hk takes and
compares the rows. It is a diagnostic that syncs; don't serve with it.

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
  kernel at 15-22 requests and 4096 tokens took 0.877 ms a call under
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
[design note](design/cdna-hipkittens-ops.md)). A call also falls back when it is
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
- **How long.** `HK_AUTOTUNE_BUDGET` seconds, 60 by default. A shape whose
  grid the budget cut short routes only up to the largest M measured; the
  next start carries on. The patch's report line says what was measured,
  how long it took, and which shapes stayed on vLLM's path.
- **Where it goes.** This machine's records: `$HK_TUNE_DIR` (by default
  beside the compile cache), `routes_vllm_linear_<arch>.json`. Each shape
  there replaces the shipped entry for the same (N, K). The next start of
  the same model finds every shape and measures nothing.
- **Once per machine.** Writers hold a file lock. Tensor-parallel ranks, or
  two servers, measure a shape once: the others wait, then read it.

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
  own tensors.

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
  weight processing, which shuffles the codes in place. The zero point is
  vLLM's: 8, whatever a symmetric checkpoint stores;
- `apply` calls the opaque op `hk_vllm::gptq_w4a16`, inside which batches of
  4 to 64 tokens take hk's decode GEMM, and the rest vLLM's kernel. Below
  4 tokens, with GPTQ's 32-column groups, vLLM's kernel is the faster one.
- Act-order checkpoints (`desc_act`), other quant types and layers on another
  kernel keep vLLM's path; `vllm_gptq.REFUSED` counts them by reason.

```bash
HK_PATCH_VLLM=1 vllm serve <a GPTQ-Int4 checkpoint> --dtype float16    # HK_VLLM_GPTQ=0 turns it off
```

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
