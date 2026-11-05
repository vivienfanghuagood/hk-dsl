# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## 0.9.7

hk-dsl 0.9.7: on gfx1201, `vllm_gptq` also takes GPTQ prefill (the batches vLLM runs on Triton, above
64 tokens) on a new large-batch form of hk's 4-bit GEMM that reads vLLM's weights in place. No change
to the language or the dialect (`hk-dialect-8`).

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.7-cp312-cp312-manylinux_2_35_x86_64.whl

### Changes

**GPTQ prefill on hk on gfx1201** (#287). `vllm_gptq` now also takes the prefill batches vLLM's `RDNAHybridW4A16LinearKernel` runs on Triton (above 64 tokens), on a new large-batch form of hk's 4-bit GEMM (`hk.ops.w4a16.exllama_large_matmul`, `w4a16_large_kernel`). Like the decode forms, it reads vLLM's own weight tensors in place. A 128-row tile of W is decoded once into LDS and shared across 128 tokens.

### All pull requests

#287, #288.

## 0.9.6

hk-dsl 0.9.6: on gfx1201, Qwen3.8's shortest prefill chunks over 16k+ contexts beside up to 7 decodes
take the packed KV split; the language and templates pages' examples are now tested against torch on
the GPUs. No change to the language, the dialect (`hk-dialect-8`) or which routes are on by default.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.6-cp312-cp312-manylinux_2_35_x86_64.whl

### Changes

**vLLM's TRITON_ATTN on gfx1201: Qwen3.8's shortest chunks over 16k+ contexts beside up to 7 decodes take the packed KV split** (#281). The packed split took such a chunk beside at most 3 decodes on gfx1201 for Qwen3.8's shape; over a context of 16k tokens or more it now takes up to 7, while their tokens stay under a measured bound. `vllm_prefill.KVS_PACK_MAX_DECODES` entries are tiers by the longest sequence. Measured faster than both stock kernels on every batch it takes.

**Documentation and its tests** (#277, #278, #279, #283). The kernels on docs/language.md and the templates pages' examples are extracted and run in the test suite: they translate for gfx1100, gfx1201 and gfx950 and match torch on the GPU tiers. The concepts pages summarise and link to the language page instead of repeating it; the IR and placement pages move to the reference section. docs/e2e-loop.md drops two timings that are not on the scorecard.

### All pull requests

#277, #278, #279, #281, #282, #283.

## 0.9.5

hk-dsl 0.9.5: on gfx1201, MiniCPM5's shortest prefill chunks beside up to 15 decodes take the packed
KV split; the language page's first example stores its row sums where it says; documentation
reorganized. No change to the language, the dialect (`hk-dialect-8`) or which routes are on by default.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.5-cp312-cp312-manylinux_2_35_x86_64.whl

### Changes

**vLLM's TRITON_ATTN on gfx1201: MiniCPM5's shortest chunks beside more decodes take the packed KV split** (#271). The packed split took a short chunk over a long context beside at most 3 decodes on gfx1201; for MiniCPM5's shape (head size 128, 8 query heads a kv head) it now takes up to 15, while their tokens stay under a measured bound (`vllm_prefill.KVS_PACK_MAX_DECODES` entries are now (decodes, decode tokens)). Measured faster than both stock kernels on every batch it takes.

**Documentation** (#259, #260, #273, #274, #275). docs/language.md's first example (`row_sumsq`) stored its row sums along a row instead of down a column; it now stores them where the text says. Integration route defaults come from the generated route table; the README's comparison is one section; contributor docs (testing, building, releasing) say how, with dated records and version notes moved to docs/archive/.

**Tools** (#269, #276). `kernel_bench` linear-w4 gets vLLM's RDNA hybrid W4A16 kernel as a GPTQ arm (what vLLM runs for GPTQ on gfx1201). The SGLang toggle A/B's stock arm now turns hk's norm route off on layers already built.

### All pull requests

#259, #260, #269, #271, #272, #273, #274, #275, #276.

## 0.9.4

hk-dsl 0.9.4: `vllm_moe_wna16` takes prefill chunks too (a per-arch token cap: 2048 tokens a step on
gfx1100, 1024 on gfx1201), and on gfx1100 Qwen3.8's short prefill chunks beside decodes take the KV
split. No change to the language, the dialect (`hk-dialect-8`) or which routes are on by default.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.4-cp312-cp312-manylinux_2_35_x86_64.whl

### Changes

**vLLM's TRITON_ATTN on gfx1100: Qwen3.8's short chunks beside decodes take the KV split** (#267). A 16-128-token chunk beside decodes went to vLLM's kernel whole when the decodes' work passed a small share of the chunk's, and that kernel walks each decode's context unsplit. For Qwen3.8's shape (head size 256, 6 query heads a kv head) the KV split now takes such a batch up to a measured number of decode tokens (`vllm_prefill.KVS_DEC_MAX_TOKENS`), the decode rows on vLLM's decode kernel; measured faster than both stock kernels on every batch it takes.

**`vllm_moe_wna16` takes prefill chunks too** (#262). Its token cap is now per arch, set where hk's 4-bit MoE GEMM is still ahead of vLLM's kernel at the best of its searched configs: 2048 tokens a step on gfx1100 and 1024 on gfx1201 (32 elsewhere). A prompt or prefill chunk up to that size now runs on hk as well. On a Radeon AI PRO R9700 that shortens time to first token for a 1k-token prompt; decode is unchanged; GSM8K equal.

### All pull requests

#262, #267, #268.

## 0.9.3

hk-dsl 0.9.3: a fix for vLLM's attention routes on a KV cache past 4 GiB (hk's 32-bit buffer offsets
wrapped there; such a cache now goes to vLLM's kernel); a kernel made by a factory needs no
`body.__name__ = name`, name string or `lru_cache` (`@hk.specialize`, and kernels named after their
factory's values); and on vLLM's TRITON_ATTN the KV split's shortest chunks take GQA-packed tiles.
No change to the dialect (`hk-dialect-8`) or to which routes are on by default.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files. The API reference marks what 0.9.3 adds *New after 0.9.2*.

    pip install hk_dsl-0.9.3-cp312-cp312-manylinux_2_35_x86_64.whl

### Changes

**Fix: vLLM's attention routes no longer read a KV cache past 4 GiB** (#265). hk's kernels address a tensor through a buffer resource, with 32-bit byte offsets from its base. Over a layer's cache view of 4 GiB or more, the prefill (`vllm_prefill`, both backends) and the split decode read the wrong pages, or faulted; Qwen3.8's shape crosses it at about a million cached tokens a layer. Such a call now goes to vLLM's kernel (`vllm_prefill.SPAN_LIMIT`).

**`@hk.specialize`, and factory kernels that need no `body.__name__ = name`** (#263, #264). A kernel defined inside a factory and given no `name=` is now named after the factory values it reads (`<function>__<values>`, or `<factory>_body__<values>` for a `def body`; the same in every process), and a kernel's `__name__` is its name, so `hk.profile`, `hk.check` and tracebacks show it rather than `body`. `@hk.specialize` on a factory memoizes it on its arguments (defaults included) and names the kernel it returns `<factory>__<arguments>`, in place of `functools.lru_cache` and a hand-built name string ([the language](docs/language.md#kernels-made-by-a-factory)). `name=` and a hand-set `__name__` keep working and still win; every kernel in `hk.ops` keeps its name and its generated source.

**The KV split's shortest chunks take GQA-packed tiles** (#253). `hk.ops.paged_prefill` has `pack`: a workgroup holds one kv head's query heads of a few tokens instead of one query head's rows, so a kv head's K and V are read once a tile instead of once a query head, and a chunk of a few tokens fills its tile instead of a sliver of it. The route takes it for the shortest chunks over long contexts, per GPU, head size and query heads a kv head, where it was measured faster than the unpacked split (`vllm_prefill.KVS_PACKED`); every call it takes is faster than both stock kernels. Beside decodes, a packed call now takes the batch too: the prefill rows on the packed split, the decode rows on vLLM's decode kernel, which splits a decode's context where its mixed-batch kernel does not. It takes up to a measured number of decodes for each GPU and model shape (`KVS_PACK_MAX_DECODES`); before, such a batch went to vLLM's kernel whole. On gfx1201 with head size 128, the unpacked split no longer takes a short chunk over a context under 16k tokens beside decodes, where it was measured slower.

**Docs: hk in vLLM and SGLang in one command** (#257, #258). The README, the quickstart, the install page and the engine pages lead with the one command that enables hk in each engine; the install page no longer names a release.

**The first-JIT notice names the pre-build command for the GPU it runs on** (#256): `warm-routes` on gfx950 only.

### All pull requests

#253, #256, #257, #258, #261, #263, #264, #265.

## 0.9.2

hk-dsl 0.9.2: vLLM's 4-bit MoE GEMMs for AWQ checkpoints with zero points (`moe_wna16`) on hk's MoE
GEMM by default on gfx1100 and gfx1201. No change to the language or the dialect (`hk-dialect-8`).

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.2-cp312-cp312-manylinux_2_35_x86_64.whl

### Default changes

Since 0.9.1 ([the default-on rule](docs/testing.md#the-default-on-rule); `python3 -m hk.integration
--status` shows each route's state and why):

**On by default on gfx1100 and gfx1201: `vllm_moe_wna16`** (#252). vLLM's 4-bit MoE GEMMs for an AWQ checkpoint with zero points (`moe_wna16`) run on hk's MoE GEMM at decode batches up to 32 tokens a step. It reads vLLM's own expert tensors, so it repacks nothing and holds no second copy. It takes only the expert shapes it was validated on (Qwen3.6-35B-A3B-AWQ, group 32); `HK_VLLM_MOE_WNA16=force` reaches others, and `HK_VLLM_MOE_WNA16=0` turns it off. Measured end to end against stock-fair on a Radeon PRO W7900D and on a Radeon AI PRO R9700, three alternating starts an arm: decode TPOT ahead, TTFT level, GSM8K equal.

### All pull requests

#252, #254.

## 0.9.1

hk-dsl 0.9.1: vLLM's prefill attention on by default on gfx1201 (Radeon AI PRO R9700), as it is on
gfx1100 since 0.9.0, and on TRITON_ATTN it now takes, split, mixed batches with a sizable prompt and
short chunks over a long context; and vLLM's GPTQ-Int4 layers on gfx1201 on hk's 4-bit decode forms
by default. The dialect is `hk-dialect-8` (an `exllama` attribute on `load_w4` / `gemv_w4`), so
kernels cached by 0.9.0 are compiled again once; the wheels' layout is unchanged.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels below are the same files.

    pip install hk_dsl-0.9.1-cp312-cp312-manylinux_2_35_x86_64.whl

### Default changes

Since 0.9.0 ([the default-on rule](docs/testing.md#the-default-on-rule); `python3 -m hk.integration
--status` shows each route's state and why):

**On by default on gfx1201 too: `vllm_prefill`** (#244, #247). vLLM's prefill attention on hk's
kernels over vLLM's own cache, as on gfx1100 since 0.9.0: ROCM_ATTN's prefix-prefill, and TRITON_ATTN's
batches with prefill in them, where the guard leaves to vLLM's kernel the batches hk's is not faster on.
On gfx1201 it needed `HK_VLLM_PREFILL=1`; `HK_VLLM_PREFILL=0` turns it off. The decode-only
TRITON_ATTN path (`HK_VLLM_DECODE`) keeps its own architectures. The GPU tests run both entry points
the way vLLM calls them, 200 times over outputs poisoned with NaN and with a producer on another stream.

**`vllm_prefill` on TRITON_ATTN: a mixed batch with a sizable prompt is split instead of declined** (#249, #250). When a prompt or chunk of 1024 tokens or more is batched with decodes, hk's prefill kernel now computes the prompt's rows and vLLM's own kernel the decode rows; before, the guard sent such a batch to vLLM's kernel whole, since hk's prefill kernel walks a decode's context unsplit. It is taken only where every measured batch was faster than both stock kernels (TRITON_ATTN's and ROCM_ATTN's); Its kernels are built at vLLM's warm-up, not on the first such batch (#250). `HK_VLLM_PREFILL_SPLIT=0` turns it off.

**`vllm_prefill` on TRITON_ATTN: a short chunk over a long context is split along the KV axis** (#251). A prefill call with few query rows -- the new tokens of a request whose prompt is mostly a prefix-cache hit, a chunked prefill's last chunk -- left most of the card idle in hk's prefill kernel and in vLLM's, and the guard sent it to vLLM's kernel. `hk.ops.paged_prefill` now has `kv_splits`: each query tile walks the context in runs, a workgroup each, and a merge kernel (`paged_prefill.merge_kernel`) combines the runs' partials. The route takes such calls with the KV split where it was measured faster than both stock kernels (TRITON_ATTN's and ROCM_ATTN's) on each GPU and head size (`vllm_prefill.KVSPLITS`); decode rows in the same batch go to vLLM's kernel, as in the split above; its kernels are built at vLLM's warm-up. `HK_VLLM_PREFILL_KVSPLIT=0` turns it off.

**On by default on gfx1201 too: `vllm_gptq`** (#248). On gfx1201, vLLM serves a symmetric GPTQ-Int4 checkpoint with `RDNAHybridW4A16LinearKernel`. That kernel runs a HIP skinny GEMM for the first few tokens of a batch and a Triton GEMM above, and the Triton GEMM is several times slower at the switch. The route now takes those Triton batches, up to 64 tokens, on hk's 4-bit decode forms. It reads vLLM's own tensors in place, so no second copy of the weights is held. vLLM keeps its HIP kernel and every larger batch. The fallback calls vLLM's op without zero points for these symmetric layers: this vLLM passes their stored qzeros to it, and its Triton path then asserts at start. Measured on Qwen3.8-27B-GPTQ-Int4 on one Radeon AI PRO R9700 against vLLM's kernels in the same process, and end to end against the same image with that assert worked around. The batches the route takes are faster, single-request decode and prefill are level, and GSM8K is equal.

### Changes

**`hk.load_w4` / `hk.gemv_w4(..., exllama=True)` and `hk.ops.w4a16.exllama_matmul`** (#248). hk's 4-bit decode forms (the WMMA form and the small-batch form) can read vLLM's RDNA skinny W4A16 layout in place: row-major int32 `[N, K/8]` codes in ExLlama's nibble order, and scales `[N, K/group]`.

**`tools/hk-bench/serving/sglang_toggle_ab.py`: an SGLang A/B in one server process.** It starts one `sglang.launch_server` with hk's plugin on and switches hk's SGLang routes (`sglang_linear`, `sglang_norm`) off and on between rounds by a signal to every process of the server (`serving/sglang_toggle/`). Weights, KV cache and the GPU-to-GPU links are the same in both arms. Two separate server starts on a Radeon PRO W7900 can differ by 13-16%, more than most kernel wins. The arms alternate round by round, each switch is checked in every rank (state file, and hk's call counters move only in the hk arm), and a ratio is reported as a result only when the arms' per-round ranges are disjoint. It needs `--disable-cuda-graph`, since a captured graph does not see the switch.

**The measurement method as its own page** (#187): [how the numbers are taken](docs/performance/method.md),
beside the scorecard.

### All pull requests

#187, #242, #244, #245, #247, #248, #249, #250, #251.

## 0.9.0

hk-dsl 0.9.0: vLLM's prefill and decode attention on RDNA3 run on hk's kernels over vLLM's own
cache; GPTQ and AWQ 4-bit layers on hk's decode GEMM; engine contracts checked per route and held
against the production images in CI, with a [support matrix](docs/support-matrix.md); the build
gates extended to the VOPD and MFMA hazards current compilers miss; ROCm 10.1 (clang 24); and the
DSL's surface consolidated (`hk.map`, `hk.mma` with scaled operands, `hk.RowIndex`, register views),
with the old spellings deprecated, not removed. hk stays on by default in vLLM and SGLang for the
routes that meet [the default-on rule](docs/testing.md#the-default-on-rule) on your GPU:
`HK_DISABLE=1` turns every route off, `HK_<ROUTE>=0` one of them
([integration](docs/integration.md#on-by-default-and-turning-it-off)). Performance results are on
[the scorecard](docs/performance/scorecard.md), not in these notes.

On PyPI: `pip install hk-dsl`, for CPython 3.10 to 3.13 on Linux x86_64 with glibc 2.35 or newer
(Ubuntu 22.04 and later, and vLLM's ROCm images): one `manylinux_2_35` wheel per interpreter. The
wheels on the GitHub release are the same files; the manylinux copies are the ones on PyPI. Each
carries the MLIR bindings for its interpreter and the HipKittens headers, so at run time only
hipcc is needed. The API reference marks what 0.9.0 adds *New after 0.8.0*.

    pip install hk_dsl-0.9.0-cp312-cp312-manylinux_2_35_x86_64.whl

### Default changes

Which routes are on by default changes from release to release under
[the default-on rule](docs/testing.md#the-default-on-rule) ([stability](docs/stability.md)).
`python3 -m hk.integration --status` shows each route's state and why. Since 0.8.0:

**New, on by default on gfx950: `vllm_dsv41_index`, `vllm_sparse_mla`.** DeepSeek-V4.1's index layers
under `--monokernel` as one hk launch, and its sparse MLA prefill (below). `HK_VLLM_DSV41_INDEX=0` and
`HK_VLLM_SPARSE_MLA=0` turn them off.

**New, on by default on gfx1100: `vllm_prefill`, `vllm_gptq`, `sglang_norm`.**
- `vllm_prefill` (#158, #167, #190, #196): vLLM's prefill attention on vLLM's own paged cache
  (ROCM_ATTN's `context_attention_fwd`, and TRITON_ATTN's `unified_attention` on batches with
  prefill in them), described below. On gfx1201 it runs with `HK_VLLM_PREFILL=1` (#216).
- Within it, decode-only batches on TRITON_ATTN at head size 256 run on hk's split decode,
  `hk.ops.paged_decode` (#220). `HK_VLLM_DECODE` unset: head size 256 at every batch (hybrid
  models' attention layers: Qwen3.8, Qwen3.6, MiniCPM-V-4.7); `=1`: head size 128 too, up to 4
  (request, kv head) pairs, one split per WGP (opt-in, for one or two requests over long contexts;
  #239); `=0`, or `HK_VLLM_PREFILL=0`: vLLM's decode, as before. Calls are counted as the route's
  `decode` path.
- `vllm_gptq`: 4-bit symmetric GPTQ linear layers on hk's decode GEMM, below.
- `sglang_norm`: SGLang's Gemma-style RMSNorm on hk's norms, below.

**On by default on gfx1201 too: `vllm_gemm`, `vllm_moe`, `vllm_int4`** (Radeon AI PRO R9700, the
R9700 vLLM image; below).

### Compatibility notes

- **Python:** `requires-python` is `>=3.10`. Wheels for cp310, cp311, cp312 and cp313 (0.8.0 had
  cp310 and cp312 only; #150).
- **The compile cache rebuilds everything once.** The key now covers every header a kernel's
  compile reads, torch's and pybind11's versions and hipcc's appended flags, and the dialect
  version is `hk-dialect-7`. Entries from 0.8.0 are never read again and can be deleted.
- **Deprecated, still working** (each warns with `DeprecationWarning` and traces to the same IR as
  its replacement; removed at most two minor releases later,
  [migrating](docs/migrating.md), `tools/surface/migrate.py`): `hk.mma_ABt_mx` (`hk.mma` with
  `.scaled`), `hk.copy_mx_scales`, `hk.load_fp4_frags_lds`, `hk.fp8_operand`, `hk.fp4_operand`,
  `hk.mx_scales_loose` (register views), the `div=` / `sort_topk=` / `staged=` keywords and
  `g.gather_prep` (`hk.RowIndex`, `g.stage_rows`), `Tile.make_causal` (`tril_` / `triu_`). A call
  migrated with inline loads in its arguments must keep the old load order (see the S2 entry).
- **Removed:** `Global.scatter_add_lds` (no user; `SharedTile.load` + `Global.scatter(add=True)`
  does the same), `runtime.compile.target_headers_fingerprint`.
- **Semantics:** on gfx950 the fp32 max / min *reductions* are IEEE maximum / minimum: a NaN in
  the input gives NaN, and -0 orders below +0. `==` / `!=` of two strips make a mask (it was
  Python's identity test).
- **Engine plugins:** a route whose engine contract is broken now stays off alone, with one
  warning line; the engine's other routes are applied.
- **RDNA:** `k(...)` under a non-default torch stream warns once (`HK_STRICT_STREAM=1` raises);
  calls on the default stream and the engine routes are unchanged.
- **gfx942:** an op with no kernel there raises `NotImplementedError` at the call instead of
  failing in hipcc.

### Known issues

- **RDNA, `hk.ops` entry points under a side stream:** the ops' pybind entry points launch on the
  null stream and do not warn under a non-default torch stream (the warning covers `k(...)`
  launches). Calling them directly on a stream of your own is not ordered against it; the engine
  routes launch on torch's stream. Use `hk.torch_op` or `HK_LAUNCH=codeobj` for your own streams.
- **Compiler bugs hk gates rather than fixes:** LLVM's missing MFMA wait states behind a branch
  (llvm/llvm-project#225989, gfx950) and the VOPD interlock hazard (llvm/llvm-project#220348,
  gfx11) are in every ROCm release to date. hk rebuilds the affected kernels (`HK_MFMA_GATE`,
  `HK_VOPD_GATE`) and says so once a process ([runtime](docs/runtime.md#the-resource-gates)).
- **clang 24 (ROCm 10.1)** is tested by hand, not in CI ([toolchains](docs/toolchains.md)).
- **Under clang 22 (ROCm 7.2.3)** the head-size-256 prefill kernel does not build spill-free, so
  those calls stay vLLM's; the vLLM images' clang 23 builds it.
- **gfx942 is compile-only:** nothing has run on an MI300X.

### Changes

**vLLM prefill attention on vLLM's own cache (`vllm_prefill`, on by default on gfx1100; #158,
#167, #190, #196, #233, #238).** With chunked prefill and prefix caching on, a long prompt's time on
RDNA is mostly prefill attention over the paged KV cache. The route replaces only that kernel:
ROCM_ATTN's `prefix_prefill.context_attention_fwd`, and TRITON_ATTN's `unified_attention` on
batches with prefill in them. It reads each backend's cache in place, so the cache, its layout and
block size, the cache write and the rest of the backend stay vLLM's. bf16, head size 64 and 128,
and 256 on TRITON_ATTN (a query row across a pair of warps); pages of 16 or 32 tokens, and on
TRITON_ATTN any multiple of 16, which covers hybrid models' attention pages (544 tokens for
MiniCPM-V-4.7, 784 for Qwen3.8). Any other call falls back to vLLM's kernel, and is counted.
On TRITON_ATTN the batch's V pages are transposed once a call for the whole batch
(`HK_VLLM_PREFILL_VT=0` turns that off). `HK_VLLM_PREFILL_SHADOW=1` also runs vLLM's kernel on
every call hk takes and prints the worst row's difference (a diagnostic; it syncs).
- **Batches hk is not faster on stay vLLM's (#233, #238).** On TRITON_ATTN two kinds of batch were
  slower on hk's kernel: decode rows beside a small prefill (hk walks each decode's whole context in
  one workgroup; vLLM splits it), and a short chunk over a context (too few query tiles to fill the
  card). The route now declines them, deciding from each step's exact sizes, which it reads from
  the host arrays TRITON_ATTN's metadata builder already has, with no device sync (the builder is
  wrapped by `apply()` and restored by `revert()`).
- Its end-to-end results (long-prompt TTFT on MiniCPM5-2B and MiniCPM-V-4.7, quality within stock's
  spread) are on [the scorecard](docs/performance/scorecard.md).

**`hk.ops.paged_decode_portable`: one decode-attention source on gfx1100, gfx1201 and gfx950
(#235).** hk.ops.paged_decode's split-KV walk over vLLM's TRITON_ATTN cache, written once: its
tiles are legal operands on both matrix units, and the two places the hardware differs are the
DSL's (#223). On gfx1201, head_dim 256 with a GQA group of 6 on hybrid pages builds only with
clang 23 or newer; under clang 22 the op raises `ResourceError` (no result, never a wrong one). It
is an op and a benchmark: on RDNA the hand-tuned walk (#220) stays what `vllm_prefill` runs.

**`tools/hk-bench`: public kernel benchmarks (#206, #220).** `kernel_bench.py` times hk's linear,
4-bit and MoE GEMMs against the engines' own kernels at real models' shapes, with gfx1100 and gfx1201
results under `tools/hk-bench/results/` (`tools/hk-bench/README.md`); `attention_bench.py` does
paged prefill and decode against vLLM's kernels (`tools/hk-bench/README-attention.md`). The
[scorecard](docs/performance/scorecard.md)'s kernel rows are reproducible with them.

**Docs: a new structure.** Install and a quickstart with W7900D and MI355X tracks; concept pages
(how a kernel runs, tiles / strips / LDS, [the correctness gates](docs/correctness-gates.md), XCDs
and `hk.comm`; #186); per-engine pages ([vLLM](docs/vllm.md), [SGLang](docs/sglang.md)) with
correctness evidence (#184); an op catalog generated from the code (#185);
[supported hardware](docs/supported-hardware.md), [toolchains](docs/toolchains.md),
[API stability](docs/stability.md) (#211) and [releasing](docs/releasing.md) (#209); older design
documents and measurements moved to an archive.

**Fix (gfx950): LDS windows that are not whole subtiles of their tile's layout are refused
(#222, #217).** HipKittens' CDNA4 `st_subtile` points at the subtile that holds a window's
first element and keeps no offset inside it, so a window narrower or shorter than that subtile
read, or was copied into, the subtile's first rows and columns, whatever its index, with no
error. The C++ translation refused such windows only on a tile that names its layout. It now
checks every window against the layout its allocation is actually declared with: the one it
names, the one its reads' fragment picks, or its type's. The refusal says which. No existing
kernel is refused, and all 1160 gfx950 kernels generate byte-identical C++.

**RDNA: the CDNA forms of P·V build (#223, #212, #213).** `Tile.as_operand(bf16)` on gfx1100 and
gfx1201 is now the accumulator's col-layout cast (the WMMA operand `.to(bf16)` makes); it was
refused. A col-layout 16-bit operand read from LDS on RDNA is now one row-major b128 read and a
register transpose instead of 16 `ds_read_u16` a 16x16 tile (the budget pass's
`slow-smem-layout` warning, and spills in a head-dim-128 paged decode). So
`acc += v_s.load(RT(bf16, 32, 16, "col")).T @ s.as_operand(bf16)` means the same on gfx950 and
on RDNA, and one attention source builds for both. `load(wait=False)` of such an operand is
refused: the transpose waits for its data. No existing RDNA kernel changes (1049 kernels each on
gfx1100 and gfx1201, byte-identical C++).

**Engine contracts are per route.** A route whose engine contract is broken, or whose apply raises,
now stays off with one warning line of its own (`[hk] WARNING: vllm: route vllm_moe_a8w4 OFF --
engine contract FAILED ...`), and the engine's other routes are applied. Until now one broken
contract left every route in that engine off, which three engine updates did over a change in one
route. The plugins' summary line lists the routes that are off. `hk.integration.contract` gains
`check_each`, `refuse_route` and `apply_routes`; the vLLM, SGLang and SGLang-Diffusion plugins use
them. A user's own modules (`HK_VLLM_USER`) that fail to load are reported the same way.

**Engine contracts are held against the production images in CI.** `python3 -m hk.integration.contract --record` writes what a route's contract check reads from the installed engines (each check is now a gather of facts and a judge of them), and `tests/hk/ir/test_engine_facts.py` judges every route that is on by default against the facts recorded in each production image (`tests/hk/ir/engine_facts/`: the MiniCPM5, MiniCPM-V, Qwen3.6 template and sglang-radeon images on gfx1100, the R9700 vLLM image on gfx1201) and in each engine version hk claims (vLLM 0.31.0 on its matched stack and SGLang 0.5.18 on gfx1100, the vLLM 0.32.1 nightly the gfx950 routes were measured on), with no engine installed. A stand-in or check that one of those images would refuse now fails the IR tier, instead of turning that route off at serving time ([testing](docs/testing.md)).

**A support matrix: [docs/support-matrix.md](docs/support-matrix.md).** Per arch, the image each GPU tier of hk's own tests runs in (ROCm, Python, torch), and every serving-engine image whose contract facts are recorded: the engine version, torch, ROCm, Triton and Python it runs on, and which of hk's default-on routes hold in it. `scripts/gen_support_matrix.py` writes it from CI's files and `tests/hk/ir/engine_facts/`, and `tests/hk/ir/test_support_matrix.py` fails while it is stale. `python3 -m hk.integration.contract --record` now records the stack (`contract.stack()`) and takes `--role` (production, measured, development); the eight fixtures carry both.

**`hk.gemv_w4`: a few tokens against 4-bit weights without WMMA, and hk.ops.w4a16's small-batch
form on it.** At 1-4 tokens the w4a16 decode kernel was bound by instruction issue, not memory. A
WMMA operand holds every row of W in both halves of the wave, so each 16-byte load of codes was
decoded twice and exchanged: ~125 instructions a wave per 256 bytes.
`hk.gemv_w4` has each lane decode its own row's codes in packed fp16 and multiply with
v_dot2_f32_f16. It returns the warp's partial as the 16x16 fp32 accumulator tile an mma gives, so a
kernel reduces and stores it as before. Symmetric (GPTQ's `zero`, `s_blocked`) and asymmetric
(`zs`) forms, groups that are multiples of 32, 1-8 tokens, fp16, RDNA.

`hk.ops.w4a16.matmul` takes it for fp16 at 1-4 tokens on gfx1100 (`gemv_takes`; `GEMV_ARCHES`).
At Qwen3.8-27B-GPTQ's shapes at 1-4 tokens it measured faster than vLLM's gptq_gemm_rdna3 on every
projection, and faster than the WMMA form at one token (one process, graph-timed, cold weights).

**gfx1201: shipped vLLM linear routes for MiniCPM5-2B and Qwen3.8-27B (TP2).** `routes_vllm_linear_gfx1201.json` gains their twelve (N, K), measured with `route_tune --timing graph` in the R9700 vLLM image, so a server of either model no longer measures them at load (Qwen3.8's ran out of the 60 s budget). M 1-5 stay on vLLM's skinny GEMM on every shape but `in_proj_ba` (from M 5); hk's decode GEMM takes MiniCPM5's qkv, o and down from M 6. A machine's own records for these shapes still override the shipped rows.

**Fix: vLLM with torch.compile and CUDA graphs failed to start with `vllm_gemm` on (#177).** vLLM calls `dispatch_unquantized_gemm()` inside the forward it compiles, and #177's stand-in cached what it returned in a module dict on its first call. That first call was the profile run's traced forward, so Dynamo replayed the write after the graph (`_GEMMS.update`), and vLLM's CUDA-graph guard refused the forward: an `InternalTorchDynamoError` / `DecompilationError` from `vllm/compilation/wrapper.py`'s `bytecode_hook` at startup (MiniCPM5 in its production configuration). The stand-in and each layer's routing decision are now read-only while torch.compile traces: the cache, the decision kept on the layer and the unrouted counts are written only in eager calls, and a decision is kept for the table's generation (bumped by `set_routes`) rather than the table object. Whether it shows depends on the vLLM build: one whose `UnquantizedLinearMethod.apply` calls `dispatch_unquantized_gemm()` on every forward (the MiniCPM5 image's) failed, one that calls it once when the method is built (the MiniCPM-V image's) did not. `tests/hk/ir/test_vllm_gemm_compiled.py` compiles a vLLM-shaped forward whole under vLLM's guard, for the GEMM route and the AWQ / GPTQ `apply` stand-ins, and `tests/hk/engines/test_vllm_compiled_forward.py` does it through the installed vLLM's own `apply`.

**Use a kernel as a PyTorch op: the recipe, and a test of it.** A new page, docs/torch-ops.md,
shows a kernel of your own made a `torch.library` custom op by `hk.torch_op`, and says what works:
eager calls, `torch.compile(fullgraph=True)` with no graph break (also `dynamic=True` and
`mode="reduce-overhead"`), CUDA-graph capture, and `torch.library.opcheck`. Inductor does not lower
or fuse the op, and it has no autograd formula. tests/hk/gpu/test_torch_op_recipe.py runs that
recipe for both registrations (code object, `HK_TORCH_OP=library`) on every GPU of the GPU tier.

**Fix: `vllm_gptq` no longer turns off every vLLM route on a vLLM whose `gptq_gemm_rdna3` takes
`b_g_idx`.** vLLM builds differ in that GEMM's parameters: some take the act-order index, some do not.
The route's engine contract counted five arguments, so on the other builds it failed, and the
contract is all-or-nothing, so hk patched nothing in that engine. The fallback now calls vLLM's GEMM
by parameter name, whatever this build's signature. Weights are read by the names the kernel was
given. The contract checks only the two `AutoGPTQLinearMethod` entry points the route wraps. A build
that lacks something the route needs leaves those layers on vLLM's path, counted in
`vllm_gptq.REFUSED`; the engine's other routes stay on.

**The VOPD gate (gfx11.0, gfx11.5): kernels with the VOPD interlock hazard are rebuilt without dual
issue.** When X's and Y's SRC0, or their VSRC1, are distinct VGPRs with the same parity, gfx11's VOPD
forwarding drops the interlock, and an operand written just before can be read stale. LLVM fixed the
pairing in llvm/llvm-project#220348, scoped to gfx11.0 and gfx11.5 by #225880; no ROCm release has the
fix yet. hk's bounded strip RMSNorm with the fused add (1024 x 5120 bf16, 16 warps), built by AMD clang
23 (TheRock 7.14), had such a VOPD and was wrong in 25-65% of launches. In
tools/rdna3-probes/vopd_interlock_parity.hip, the same window is wrong in 88-93 of 100 launches under a
partial EXEC when the VSRC1s share a parity, and in none when they differ or EXEC is full.

`hk.runtime.compile.build` now applies the upstream rule (`cocheck.vopd_parity_hazard`) to every
gfx11.0 / gfx11.5 code object. A kernel with a hazardous VOPD is rebuilt with
`-mllvm -amdgpu-enable-vopd=0` (`HK_VOPD_GATE=fix`, the default; `refuse`, `report` and `0` as for the
MFMA gate). Every other kernel's code object is unchanged, and gfx12 is not affected. Cached builds
from before the gate are checked on their first hit, and `python3 -m hk.integration --status` lists
what was rebuilt. tests/hk/gpu/test_vopd_gate.py launches the strip kernel 200 times into NaN-filled
outputs (bf16 and fp16, weight offset 0 and 1).

**hk.ops.paged refuses a decode whose tensors are not the shapes its kernels read.** The
unsplit kernels (`paged.KERNELS[d]`) and `paged.decode` (the fused walk and its merge, which the vLLM
backend runs) took their tensors on trust. A wrong layout became a read or a write past an
allocation: the unsplit kernel given the fused walk's 4-row query read 12 rows past it and page-faulted
a W7900D. Both entry points now check the decode's contract (`paged.check_decode`) before launching:
- q and out are (requests, kv_heads, rows, head_dim), with a whole 16-row tile for the unsplit
  kernel and 1-16 rows for the fused walk;
- the caches are (blocks, kv_heads, KV_BLOCK, head_dim);
- block_table and seq_lens are int32 and cover every request;
- the fused scratch has a row per (request, split) and the group's rows.

A wrong call raises ValueError naming the tensor, its shape and the one the kernel needs. No kernel
changes; `decode` checks a set of shapes once. gfx950's decode (hk.ops.paged_cdna) is not affected.

**Fixed: a kernel the compiler refuses could crash the process.** An error raised inside the pass
pipeline (for example "a mask strip leaves its hk.map" on RDNA) was reported from MLIR's worker
thread, where the Python bindings' error capture has no Python thread state; when the function ran
on a worker the process segfaulted instead of raising, about once in a thousand refusals (and once in
the IR tests' CI). hk's MLIR context now runs passes on the calling thread. The generated code is
unchanged (V0: byte-identical on gfx950, gfx1100, gfx1201), and compiling is no slower.

**SGLang: Gemma-style RMSNorm on hk (`HK_SGLANG_NORM`, on by default on gfx1100).** Where
SGLang's `GemmaRMSNorm` would run its native torch path, which it does on ROCm without aiter or
vLLM (eleven kernels a norm), it goes through hk's weighted norm kernels. The route passes the
stored weight at `weight_offset=1`, fuses the residual add, writes out of place, and launches on
torch's current stream. Calls it does not take go back to SGLang's forward, counted by reason in
`hk.integration.sglang.NORM_STATS`. Where SGLang has its own fused norm (aiter, vLLM) the route
is not applied.

**RDNA: `k(...)` under a non-default torch stream warns (`HK_STRICT_STREAM=1`: raises).** On gfx1100
and gfx1201 the pybind entry point launches on the null stream, which is ordered against nothing torch
queued on a stream of its own: a direct `hk.ops` / `@hk.kernel` call made under `torch.cuda.stream(s)`
-- an engine's model stream, a side stream, a graph capture -- could read its inputs before they were
written, and nothing said so. Each such launch now checks torch's current stream and warns once per
kernel, naming the fix (`hk.torch_op(k)` or `HK_LAUNCH=codeobj`); `HK_STRICT_STREAM=1` makes it an
error. Calls on the default stream, and every engine route (they launch on torch's stream), are
unchanged; no kernel's source changes.

**`vllm_gemm`: a layer the table never routes keeps vLLM's GEMM, unwrapped.** Every unquantized
linear used to go through hk's op, which fell back inside for a call it did not take. A model whose
dense layers are fp16 (the tables are bf16) got no hk call but paid for one captured op per layer
per decode step. Now each layer is decided once, at load, after the missing shapes are measured. A
layer whose (dtype, N, K) the table routes at no M, or whose weight is biased or strided, calls
vLLM's own GEMM directly. The load reports `[hk] linear layers at load: N through hk's op;
unrouted at load: ...`.

**gfx1201: a single-row linear stays on vLLM's own GEMM.** On a Radeon AI PRO R9700, `vllm_gemm` no
longer sends M = 1 calls to hk's decode GEMM even where its table measured hk faster: vLLM's wvSplitK
already streams the weights at the memory roof there, and the routed shape lost end to end at batch 1.
`vllm_gemm.ENGINE_KEEPS_M1` names the arches; gfx1100's tables and routing are unchanged.

**The routes tuner times what a decode step runs (RDNA): each call in its model's context, TunableOp
in the records, and 12 gfx1100 rows re-measured.**
- **Two biases.** `hk.autotune.routes` (route_tune and the tuning at load) timed an unqueued burst
  on RDNA, which charged vLLM's Python dispatch at M <= 5 to the engine, so shapes went to hk that
  a captured graph runs faster on vLLM's path; InternVL3-2B lost decode time at one request to it.
  And a shape timed alone flatters a tuned hipBLASLt kernel: on an R9700 with MiniCPM5's TunableOp
  results, vLLM's o_proj at M = 8 ran faster back to back than in the served step, while hk's
  decode GEMV ran the same either way, so the tuning at load left M = 8 on vLLM.
- **Timing.** `measure_points(timing="context")`, the default on RDNA: each path's calls are
  captured in a CUDA graph and replayed. Every call runs after a norm-like kernel that writes its
  input, then the model's other linear layers follow on their routes at that M, each with an input
  kernel of its own. Every shape rotates through 576 MB of its own copies. The context's own graph
  is timed in the same rounds and taken off. "graph" (a shape's burst alone) stays; the LM head,
  one call a step, is timed so.
- **The routes move the context.** Once every decode M is in, the Ms whose context changed are
  timed again in the routes chosen so far, until none moves (at most 3 passes).
- **Records.** `meta.timing`, `meta.context_routes` (the context each point was timed in) and
  `meta.tunableop` (the results file and a hash of it, or "off"). A later start re-times what an
  earlier one measured in another context. A record timed another way or under another TunableOp
  state no longer counts: the overlay leaves it out, as it does a record measured against another
  engine, so the shipped row stands and the tuning at load measures the shape again. Before, a
  complete stale record shadowed the shipped row and was never re-measured.
- **route_tune.** It gains `--timing`, `--nk`, `--merge-into` and `--splice` (replace a row's
  points at the measured Ms only, re-derive its runs), and times a model's shapes in context.
- **Checked against serving (R9700, MiniCPM5, TunableOp, rocprofv3).** At decode batch 8, with
  the routes picked by context timing (qkv and o on hk's GEMV), the harness's prediction of a
  layer's linear time matched the served step's, for hk's routes and for vLLM's. Batches 1 and 32
  stay on vLLM's path.
- **The 12 rows.** The shipped gfx1100 rows that sent M <= 5 to hk against the burst floor were
  re-measured at M 1-63 with graph timing and spliced. Small-N rows give M <= 1-5 back to vLLM
  (2048x256: M <= 16). Large rows take hk from M = 2, where vLLM's M = 2-4 is slower than the
  decode GEMV in a graph. M = 1 stays vLLM's everywhere.
- **Gates.** 3 alternating server starts per arm against vLLM with TunableOp on, GSM8K level
  throughout, on gfx1201 (MiniCPM5-2B, routes tuned at load) and gfx1100 (MiniCPM5-2B,
  InternVL3-2B): no concurrency slower than stock, and the InternVL3-2B loss at one request gone.

**RDNA3's paired WMMA operand: `as_operand(k_order="even_odd")` (DSL-surface S11).** On gfx1100,
`S.as_operand(hk.fp16, k_order="even_odd")` turns a 32x16 fp32 pair of WMMA accumulators into two fp16
operands, the pair's even rows and its odd rows, one `permlanex16` a packed dword. `hk.ops.attn_k3`
writes its Pᵀ this way; its `PairPKernel` subclass, which spliced the same C++ into the generated
source and rewrote it with a regex, is gone -- no `hk.ops` kernel overrides `with_defines` any more.
attn_k3's ISA and resources are identical on gfx1100; on gfx1201, where its route never ran and RDNA4's
accumulator layout differs, tracing it is now refused.
**AWQ MoE at decode on RDNA3: `hk.load_w4(zp=, rows=True)`, `hk.ops.moe.moe_gemm_w4` and the vLLM
route `vllm_moe_wna16` (opt-in, `HK_VLLM_MOE_WNA16=1`).** vLLM runs an AWQ MoE checkpoint's routed
experts through moe_wna16: codes uint8 [E, N, K/2], scales [E, N, K/group], and 4-bit zero codes per
row and group. hk now reads those tensors as they are, with no repack and no second copy of the
experts (~17 GB on Qwen3.6-35B-A3B-AWQ):
- `hk.load_w4(..., zp=Z)`: the element is (q - z) * s, with z read from an int8 [N/2, K/group]
  tensor holding two rows to a byte. An fp16 tile decodes as the symmetric form does, so q - z is
  exact and the product rounds once to nearest even. `rows=True`: the codes are row-major, int32
  [N, K/8], each lane reading its own row's 16 bytes. On a Qwen3.6 microbenchmark this is as fast as
  hk's interleaved layout. RDNA only (include/hk/rdna/w4.cuh). The asymmetric and symmetric forms
  print exactly as before: V0 shows 0 differences over the catalog (1049 kernels) and over every
  w4a16 kernel the routes build (66), on gfx1100 and gfx1201.
- `hk.ops.moe.moe_gemm_kernel` gains `dtype` (fp16), `w4_group` (W in the layout above, expert e's
  rows at e * N) and `unroll`. The bf16 kernels are unchanged. `moe_gemm_w4` / `takes_w4` run it.
- `hk.integration.vllm_moe_wna16` rebinds `invoke_fused_moe_wna16_triton_kernel` and asks for
  16-row blocks at up to 32 tokens a step, on the (experts, N, K, group) it was validated on
  (`VALIDATED`: Qwen3.6-35B-A3B-AWQ's two expert GEMMs; `HK_VLLM_MOE_WNA16=force` takes any).
  Prefill chunks, other shapes, symmetric GPTQ without zero points and int8 weights stay on vLLM's
  kernel, counted by reason in `REFUSED`.
- Checked on Qwen3.6-35B-A3B-AWQ, one W7900D, the Radeon TheRock vLLM image, against vLLM with
  TRITON_ATTN, TunableOp and the int4 MoE config picked per M, 3 alternating starts each: decode
  faster at every measured concurrency (1, 4, 8, 16 requests), GSM8K-200 equal, and every captured decode call took hk's kernel. The
  route is opt-in; its serving numbers go on the [scorecard](docs/performance/scorecard.md).

**`hk.map` on RDNA strips (gfx1100, gfx1201).** The integer, mask and bit expressions below,
`hk.map(fn)` and `hk.map_cpp` lower on RDNA too, through include/hk/rdna/map.cuh (the trees'
`strip_each`, wave32), included only by a kernel with a map -- every other kernel's source is
unchanged. RDNA's strips hold no bool, so a mask must stay inside a map (used where it is made,
or read once by another map); one read twice or stored is a verifier error naming the line.
`hk-map-fuse` now also fuses across scalar index ops (`x.at[r, c]`'s coordinates), which a
store's index put between a mask and its `hk.where`. Tested on the GPU on all three targets
(tests/hk/gpu/test_map.py) and built by hipcc for each (tests/hk/codegen/test_map_builds.py).

**Qwen3.8-27B's linear layers on the gfx1100 routes (vLLM and SGLang, TP2).** Before this, neither
engine routed a single Qwen3.8 linear to hk, because neither table held its shapes.
- `hk.autotune.routes.linear_shapes` now reads hybrid decoders (Qwen3.5 / 3.6 / 3.8, Qwen3-Next):
  the attention projections on the full-attention layers only (q doubled by `attn_output_gate`),
  and the Gated-DeltaNet `in_proj_qkvz`, `in_proj_ba` and `out_proj` on the rest. `route_tune.py
  --model`, `hk.integration.tune` and `tools/hk-bench/vllm_gemm.py` use it.
- `routes_vllm_linear_gfx1100.json` and `routes_sglang_linear_gfx1100.json` gain Qwen3.8's seven
  TP2 per-rank (N, K), measured with `route_tune.py` in each engine's image (vLLM: TheRock ROCm
  7.14, clang 23; SGLang: sglang-radeon). Each shape records the run it came from.
- The tuner's held-out shape sets are redrawn: they exclude every shipped (N, K).

**GPTQ-Int4 linear layers in vLLM on RDNA3: `vllm_gptq` (`HK_VLLM_GPTQ`, on by default on gfx1100).**
vLLM runs a 4-bit symmetric GPTQ checkpoint on gfx1100 through `AutoGPTQLinearMethod` with
`RDNA3W4A16LinearKernel`. The new route keeps hk's own layout of the codes beside vLLM's.
Batches of 1-64 tokens run on hk's 4-bit decode GEMM, in the symmetric form below (its scales in
16-row blocks, no zero-times-scale tensor); larger ones stay on vLLM's kernel. Act-order
checkpoints, other quant types, layers on another kernel and layers whose stored zero points are not
8 are refused, and `vllm_gptq.REFUSED` counts them by reason. On Qwen3.8-27B-GPTQ-Int4 (W7900D,
TRITON_ATTN + TunableOp on both sides) decode is faster than vLLM's at every measured concurrency
(1 and 4-16 users).
- `hk.ops.w4a16`: a quantization group narrower than the load now narrows the load
  (`schedule_for(m, k, group)`). Before, GPTQ's 32-column groups fell to the dequantize path at
  every batch up to 16 tokens.
- `tools/hk-bench/gptq_w4a16.py` times hk against `gptq_gemm_rdna3` on GPTQ checkpoints.

**W4A16's symmetric form: `hk.load_w4(..., zero=, s_blocked=)`.** A 4-bit checkpoint whose zero
points are all one value (GPTQ's uint4b8: 8) loads without the zero-times-scale tensor:
`hk.load_w4(ty, qw, s, None, n0, k0, group=G, zero=8)` dequantizes `(q - 8) * s`, 12.5% fewer bytes
at 32-column groups. `s_blocked=True` reads scales laid out `[N/16, K/G * 16]`, so the 16 rows a
k-step reads take 32 contiguous bytes. On RDNA a half tile decodes in packed fp16 (include/hk/rdna/w4.cuh,
included only by a kernel that uses the form): `q - 8` exactly, then one multiply by `s` rounded to
nearest. bf16 tiles and CDNA keep the fp32 decode. `hk.ops.w4a16.repack_gptq(..., symmetric=True)`
returns `(qw, s in 16-row blocks, None)` and refuses a checkpoint whose zero points are not all 8;
`dequant`, `matmul` and `plan` take `zs=None` as the symmetric form. The asymmetric form (AWQ's too)
is unchanged: every catalog kernel's source is the same.

**One register-view op: `hk.lane_view` (DSL-surface S9a).** `Tile.as_fragment` and `Tile.as_scales`
(and the deprecated `hk.fp8_operand` / `hk.fp4_operand` / `hk.mx_scales_loose`) build one dialect op,
`hk.lane_view`, and one C++ function, `kittens::lane_view`, picked by the types -- instead of three of
each. The generated code is the same: ISA and resources identical on K1 and dsv41_index.
`hk.mfma_strip` is unchanged (S9b, deferred).

**`hk.lane_coords` of a strip type; `hk.iota` is its column (DSL-surface S6).**
`hk.lane_coords(ty)` -- of a strip type (`hk.RS`) -- gives two i32 strips of its shape,
element (r, c) holding r and c; of a register tile it is as before. `hk.iota(ty)` is
`hk.lane_coords(ty)[1]`, kept as a one-line alias. In the IR the `hk.iota` op is gone: an
`hk.lane_coord` without an operand is a strip's own coordinates (`kittens::lane_coord<AXIS>(dst)`,
`kittens::iota`'s formula). The generated source of the 14 catalog kernels that used `hk.iota`
changes; their ISA and resources are identical (all 1160 gfx950 builds compared against main).

**`hk.ops.rms_norm` / `fused_add_rms_norm(weight_offset=)`: weights stored as an offset from one.**
Gemma-style models (y = x_hat * (1 + w)) pass their stored `w` and `weight_offset=1.0`. The kernels
add the offset in fp32 after loading the 16-bit weight, so the result is within half a unit of the
output dtype of a float64 reference. Folding the one into a bf16 weight first rounds most of a
model's weights. The offset is fixed when a kernel is built (`hk.ops.norm.offset_kernel`), not a
constexpr argument, so every registered norm kernel generates the source it did before: the default
path is unchanged on every arch.

**gfx950: one copy body for a group's loads into LDS (DSL-surface S8).** A synchronous
`g.load(tile, view)` is the asynchronous copy's DMA body -- bounded at the tensor's last row and the
tile's -- followed by the waits it always had; hk/cdna.cuh's second, synchronous copy body is gone.
In this form the copy's descriptor is read from lane 0, so a descriptor LLVM cannot prove uniform is
not a waterfall loop around every copy. Only the two kernels with such loads change: `mxfp8.gemm`
(outputs bit for bit the same at four shapes, and no slower) and example 02's GEMM.
gfx942 is unchanged.

**gfx1100 under clang 24 (ROCm 10.1): the kernels the spill gate refused build again.** With clang
24, 139 of the codegen tier's gfx1100 kernels spilled and were refused: `hk.ops`' fp16/bf16
RMSNorm and LayerNorm, rope, the fp32 norm/softmax at the top of their range, examples 02 and 03,
and four decode schedules. The cause was HipKittens' element-by-element global<->register paths,
the fallback for unaligned rows and the only path for layouts that don't vectorize. They gave
every element a 64-bit pointer, and clang 24 kept a whole tile's worth of them live: an RMSNorm
that needs 79 VGPRs under clang 22 took 256. Under clang 24 these paths now go through a buffer
resource at 32-bit offsets (HipKittens#4, `HK_RT_GLOBAL_BUFFERED`):
- the codegen tier passes on gfx1100;
- the 256 kernels that already built use a median of 48 fewer VGPRs;
- 12 of them lose one occupancy step (attention, paged decode, elementwise min/max/gelu).

Under clang 22 and 23 nothing changes: the ISA is identical for all 1049 kernels of the gfx1100
catalog.

**Integer, mask and bit expressions on strips: `hk.map` (gfx950).** What needed `hk.map_cpp`'s
C++ strings is written with operators (docs/language.md, "Integer, mask and bit expressions on
strips"; docs/design/dsl-surface.md S7).
- i32 strips take C's `+ - * // % << >> & | ^ ~` (`//` and `%` truncate, `>>` is arithmetic);
  comparisons of i32 and fp32 strips make masks, combined with `& | ~` and used by
  `hk.where(m, a, b)`; `hk.min` / `hk.max` of i32 strips are signed. `hk.math.fma`, `rcp`,
  `popcount`, `ctz`, `ceil` and `floor` are new. `hk.map(fn, *args)` traces a Python function of
  the elements into one expression, with contraction off.
- In the IR this is one op, `hk.map`: a region of `arith` / `math` scalar ops (the `math`
  dialect is now registered), or a C++ body -- `hk.map_cpp` is that form, and prints the same
  C++ as before. The `hk-map-fuse` pass merges the one-op maps the operators build into one
  expression; it never moves arithmetic across a load, a store, a lane exchange or a barrier.
- `paged_cdna`'s and `sparse_mla`'s map_cpp calls are rewritten with the operators: the
  generated ISA and resources are identical on all 76 of their catalog builds, so their
  allowlist entries are gone. `dsv41_index`'s 190 stay for a later step.
- No float result changes: fp32 strip and tile operators keep their ops. `==` / `!=` of two
  strips now make a mask (it was Python's identity test); of WMMA tiles and vectors they still
  are identity.
- An `hk.Int` on the left of a strip operator (`hk.arch.warp_id() * 64 + col`) now gives the
  strip's operator instead of a `TypeError`.

**`hk.templates.rowwise` and `hk.templates.elementwise` stay callable after their module is
imported.** `import hk.templates.rowwise` (or `from hk.templates.rowwise import ...`) before the
first use bound the module to the package's name, and `hk.templates.rowwise(...)` then raised
"'module' object is not callable". The two modules now call through to their template, as
`hk.templates.gemm`'s already did.

**gfx942 (MI300X / MI325X): an op with no kernel there refuses at the call.** `hk.ops.matmul`,
`hk.attention`, the paged decode and prefill, `decode_gemm.matmul`, `moe.moe_gemm`, `w4a16.matmul`
and `hk.templates.gemm` used to get as far as hipcc and fail on a missing include/cdna3 member.
They now raise `NotImplementedError` at the call: "<op> has no kernel for gfx942; it runs on
gfx1100, gfx1201, gfx950", and the message says that gfx942 is compile-only. `hk.ops.sdpa` falls
back to torch there. The `hk.templates.rowwise` refusal now names gfx1201 among the arches with
register strips.

**gfx1100: the decode GEMM builds under the TheRock ROCm 7.14 compiler, so vLLM's linear routes
to hk in the W7900D production images again.** Under its AMD clang 23 the four-tile
`decode_gemm` spilled (256 -> 264 VGPRs), the spill gate refused it, tuning at load failed and
`vllm_gemm` routed nothing.
- The cause: include/rdna3 copied fp32 tiles between LDS and registers through generic
  pointers, a FLAT load and a 64-bit address an element; the decode GEMM's fp32 combine peaked
  there. Those copies now go through LDS pointers (HipKittens submodule bump): the four-tile
  kernel is 194 VGPRs under clang 23, 162 under clang 22; 30 of 180 decode GEMM schedules that
  spilled now build on both, none breaks. Outputs are bit-identical.
- `hk.kernel(max_occupancy=N)`: a cap on the waves a SIMD holds -- the kernel keeps enough
  VGPRs that no more fit. The freed registers raised the decode GEMM's occupancy, and more
  workgroups streaming W at once read it slower on the widest layers; its two- and four-tile
  schedules now carry the occupancy they were tuned at (`occupancy`: 6 and 5).
- Checked on a W7900D against the previous build at MiniCPM5-2B's and Qwen3-8B's shapes, M
  1-64: outputs bit-identical under both clang 23 (the production image) and ROCm 7.2's
  clang 22, and the four-tile kernel builds under clang 23.
- A kernel the compiler refuses no longer breaks the route: `vllm_gemm` sends that shape to
  vLLM's path (said once) instead of raising inside the forward, and tuning leaves the
  kernel out of what it measures instead of failing.
- `tests/hk/codegen/test_decode_gemm_builds.py`: every shipped schedule builds without
  spilling at its cap, under whichever hipcc runs it.
- The shipped gfx1100 vLLM table has rows for MiniCPM5-2B's and MiniCPM-V-4.7-35B-A3B's (TP2)
  linear shapes, measured in their production image, so they start without tuning at load
  (which had lengthened MiniCPM-V's startup) and every shape is covered --
  tuning at load had run out of its 60 s before MiniCPM-V's down_proj and in_proj_ba. M = 1 stays on vLLM's
  skinny GEMM there. MiniCPM5-2B end to end in that image: GSM8K 87.95% vs 88.02% (1319
  questions, McNemar p = 1.0).

**vLLM: the sparse MLA route (`vllm_sparse_mla`) is on by default on gfx950.** Validated on
the serving scorecard (DeepSeek-V4.1-Flash, MI355X TP2, real weights): GSM8K-1319 equal to
stock's. The serving result quoted for hk's default routes is on the
[scorecard](docs/performance/scorecard.md).
- **The trade-off is decode.** The route runs only where vLLM's own sparse MLA does, and
  that path also replaces AITER's decode kernel, so decode steps can be slightly slower at
  some batch sizes (an open item).

`HK_VLLM_SPARSE_MLA=0` turns it off. With `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, vLLM
takes AITER's path for prefill and decode and the route stands aside; its startup line
says so.

**`include/hk/cdna.cuh` split by kind (DSL-surface S12).** It is an umbrella over
`include/hk/cdna/{core,lib,formats,special,primitives,escape}.cuh`, a pure move of the definitions
in their dependency order; every gfx950 kernel's code object is unchanged, function by function.
The build cache sees the parts through the preprocessor's dependency list (#136).

**Register views: `Tile.as_fragment`, `Tile.as_scales` (DSL-surface S5).**
- `words.as_fragment(hk.RT(hk.fp8e4m3, 16, 128), fmt="fp8" | "fp4")` reads an i32 strip's
  registers as a block-scaled MMA operand fragment, and `bytes.as_scales()` reads one as loose E8M0
  scales. No data moves. They are the inverse of the planned `lane_strip`.
- They replace `hk.fp8_operand`, `hk.fp4_operand` and `hk.mx_scales_loose`, now deprecated shims
  that trace to the same IR. `dsv41_index` uses the new spellings.

**`rotate_half` is `swap_halves(span, negate_first=True)` (DSL-surface S4).**
- `Tile.swap_halves(span=None, negate_first=False)` exchanges the halves of each `span`-column
  segment (the whole row by default) and optionally negates the first. `rotate_half(span)` stays
  as sugar for it. The two dialect ops are one (`hk.swap_halves {span, negate_first}`), and the two
  C++ templates are one, `kittens::swap_halves<SPAN, NEG>`.
- Evidence: the 70 rotary kernels' code objects are identical on gfx950, function by function
  (ISA hash and resources).

**Rows by index take an `hk.RowIndex` (DSL-surface S3).**
- `hk.RowIndex(ids, at, count=, div=, sort_topk=)` says which rows a gather, a scatter or a
  gathered LDS copy moves, and how an index names a row: plain, `div=k`, or AITER's sort ids. It
  replaces the `div=` / `sort_topk=` / `staged=` keywords and the `(ids, at)` pair of
  `g.load(gather=)`. `Global.gather(ty, ids, at)` / `scatter(t, ids, at)` with plain ids stay.
- `g.stage_rows(ix, rows, keep_choice=)` copies an index into LDS and returns the staged index the
  copies read there. It replaces `g.gather_prep(ids, at, rows, raw=)`.
- A consumer now refuses a staged index of the other form, and one no `stage_rows` has copied.
  Both used to be a silent read of an LDS array no one wrote.
- `ix.staged(rows)` describes the staged index ahead of the loop body that copies it, and
  `ix.offset(n)` gives a later tile's positions.
- A gathered copy's row limit (#125) carries over: `g.load(..., gather=ix, rows=n)`, or the
  index's own `count` (rows = max(count - at, 0)) -- the same IR as `gather=(ids, at), rows=`.
- The old keywords and `gather_prep` are deprecated shims that trace to the same IR.
- K1 and the RDNA MoE GEMM use the new form; every shipped kernel's generated source is
  byte-identical.

**The block-scaled MMA is `hk.mma`; MX scales are register values (DSL-surface S2).**
- `hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb, fmt="fp4", at=(tile, step)), acc)` is gfx950's
  block-scaled MFMA: `Tile.scaled(scales, fmt=, at=)` makes an MX operand (`hk.Scaled`), and both
  operands carry one. `hk.mma_ABt_mx` is a deprecated shim for it.
- `MxScales.copy(into=dst)` replaces `hk.copy_mx_scales(dst, src)`, and `hk.load_fp4_frags`
  takes an LDS tile as its source, replacing `hk.load_fp4_frags_lds`. Both old names are
  deprecated shims (DeprecationWarning).
- `hk.MxScales` and `hk.Scaled` are exported and documented. K1, K2, gemm_cdna's MX mode and the
  DeepSeek-V4.1 index front (`dsv41_index`) use the new spellings. Every shipped kernel's generated source is byte-identical.
- Migrating a call with inline loads in its arguments: bind them to locals in the old argument
  order first (`hk.mma_ABt_mx` evaluated a, b, acc, sa, sb; `hk.mma` with `.scaled` evaluates a,
  sa, b, sb, acc), or the loads issue in another order and the kernel changes.
  `tools/surface/migrate.py` rewrites call sites and does this; [docs/migrating.md](docs/migrating.md).

**Removed: `Global.scatter_add_lds`; deprecated: `Tile.make_causal` (DSL-surface S1).**
- `scatter_add_lds` had no user since K1 chose the deterministic sum (K1 c33). Its dialect op
  and `kittens::scatter_add_lds` are gone. `SharedTile.load` + `Global.scatter(..., add=True)`
  does the same; `scatter(add=True)` stays, held by its GPU test (`test_scatter_add.py`).
- `Tile.make_causal(v)` is now `tril_(0, v)` on a copy, and `transposed=True` is `triu_(0, v)`.
  The call warns (`DeprecationWarning`). The `hk.make_causal` / `hk.make_causal_t` dialect ops
  are gone; no kernel used them.
- Every name the DSL-surface steps deprecate (S1, S2, S3, S5) is removed in the release after the
  last call from outside the repository is migrated, and at most two minor releases after the
  release that deprecated it. [docs/migrating.md](docs/migrating.md) lists them with their
  replacements.
  - On RDNA the old ops were the same C++ as `tril(0)` / `triu(0)`.
  - On gfx950 the old op only compiled as `make_causal` on fp32 tiles: there it matched
    `tril_(0, v)` and torch bit for bit (col layout, 16 to 64 square). include/cdna4 has no
    `make_causal_t` and no bf16 form, so the alias now covers more.
- Every shipped kernel's generated source is byte-identical (`tools/surface/snapshot.py
  sources`: gfx950 1145, gfx1100 1049, gfx1201 1049 kernels).

**The surface registry (DSL-surface S0).** `python/hk/dsl/surface.py` registers every name of
the language, every dialect op and every function in `include/hk/` with its kind (primitive,
format, escape hatch, experiment, internal, merge / delete / deprecated). The IR tier
(`tests/hk/ir/test_surface.py`) now fails on anything unregistered, and on an escape-hatch use in
a file of hk.ops, the templates or the examples that `scripts/escape_hatches_allow.txt` does not
list with its reason, or above that line's ceiling. A `Kernel` subclass may not override
`with_defines` / `source` unless listed there. Both ratchet: a registered name that is gone, or
uses under a ceiling, print a hint to tighten them and never fail.
- The API reference marks each non-primitive name with its kind and boundary, or its migration
  step.
- `tools/surface/snapshot.py` takes the Gate A / B evidence for a change to the generated code:
  every shipped kernel's generated source hashed (`sources`), or its resources and an ISA hash a
  function (`builds`), diffed between two checkouts.
- `tests/hk/gpu/test_specializations.py`: the 32x32 fp32 -> bf16 store, the 16-column column
  reduction and the -inf column masks (cdna.cuh's type-selected paths) give the generic results
  bit for bit on gfx950.
- A pull-request template with the reviewer checklist.

**ROCm 10.1 (AMD clang 24): every compile failed; it builds and runs now.** In
the official `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0`
(TheRock), even `python3 examples/01_vector_add.py --build` raised
`ResourceError: the build emitted no kernel-resource-usage remarks`. clang 24
builds a host+device module (`-shared`, the pybind and torch path) with its new
offload driver: clang-linker-wrapper generates the device code at link time and
prints the remarks bare, `k.cpp:3:0:     VGPRs: 2`, without clang's
`remark:` and `[-Rpass-analysis=kernel-resource-usage]`, which were all the
spill gate read (a device-only `--genco` build still prints those). The gate
now reads both shapes; a bare line counts only with a `file:line:col:`
location and one of the remark's keys. Captured output of clang 22, 23 and 24
is in `tests/hk/remark_fixtures` (`tests/hk/ir/test_resource_remarks.py`).
- The error sent people to check that `-Rpass-analysis` reached hipcc, which it
  had. It now names the compiler (`hipcc --version`'s clang line) and shows the
  first lines of its output.
- ROCm's LLVM tools -- `llvm-objdump`, `llvm-readelf`, `llvm-objcopy`,
  `clang-offload-bundler`, `llvm-cxxfilt` -- are found beside the clang hipcc
  runs (`InstalledDir`, or `HIP_CLANG_PATH`), then in `ROCM_PATH`'s `llvm/bin`,
  then on `PATH`. TheRock puts none on `PATH`, so `HK_LAUNCH=codeobj`, the wait
  gate and `hk.profile`'s disassembly died on `FileNotFoundError: 'llvm-objdump'`;
  a tool that is nowhere is now an error that says where it looked
  (`runtime.compile.llvm_tool`). `rocm_path()` finds TheRock's
  `_rocm_sdk_devel` (it fell back to a `/opt/rocm` that is not there).
- `HK_MFMA_GATE=fix` padded nothing in a `.so` built by clang 24: its pipeline
  has no device assembly step, so the rebuild ran unpadded and the kernel was
  refused. The rebuild plans such a command under `--no-offload-new-driver`,
  and a plan with no device assembly at all is now an error.
- Tested with the three compilers on an MI355X: `01_vector_add.py` (`--build`,
  run, `HK_LAUNCH=codeobj`), the README quick start, and `hk.check` on the
  getting-started kernel. On gfx1100 the fp16/bf16 RMSNorm and LayerNorm and
  the rope of `hk.ops`, and examples 02 and 03, still spilled under clang 24;
  the "gfx1100 under clang 24" entry above fixes them.

**`hk.check` named an MI355X "AMD Radeon Graphics"**, HIP's name for a device
its libdrm table does not list (rocm/pytorch 7.2.3). For that name it now
takes the board's own from sysfs (`AMD Instinct MI355 OAM`):
`runtime.arch.device_name`.

**`requires-python` is `>=3.10`** (was `>=3.9`, which nothing tested), with
classifiers; the wheels are cp310 and cp312, another 3.x builds from source.

**The compile cache served kernels built from old headers.** The key hashed a
fixed set of files -- kittens.cuh, pyutils/, the target's arch tree and its
extra_includes (include/hk/cdna.cuh) -- and nothing they include. An edit to
include/hk/cdna_agpr.cuh (which cdna.cuh includes) or include/pyutils/hk_bind.cuh
(which every pybind module includes), in a checkout or an installed wheel, kept
every key, so the next build loaded the binary from before the edit with no
sign of it: an A/B or benchmark after a header fix could measure the old code.
The key now covers the contents of every header the kernel's compile reads, as
the compiler lists them: on a miss, one preprocessor pass per side (device, and
host for a `.so` -- hipcc's `-MD` lists the host's only) writes clang's
dependency file, kept beside the cache as a manifest; a hit re-hashes the listed
files. An edit to a header a kernel does not include still leaves it cached.
torch's and pybind11's versions are in the key when their headers are on the
include path (a torch upgraded in place kept its key), and so are the flags
hipcc appends from HIPCC_COMPILE_FLAGS_APPEND / HIPCC_LINK_FLAGS_APPEND and
HIP_CLANG_PATH. A cache hit costs slightly more (it re-hashes the listed headers).
Every existing entry misses once (a new
key scheme); old entries are never read again and can be deleted.
`runtime.compile.target_headers_fingerprint` is gone. `tests/hk/ir/test_compile_cache.py`,
`tests/hk/codegen/test_compile.py`.

**gfx950: fp32 max and min reductions are IEEE 754-2019 maximum and minimum.**
Row and column `max` / `min` of register tiles and strips, `group_max` /
`group_min`, the combine of `cross_warp(v, "max" | "min")` and `fold_rows`,
and the online fold `m.max_(...)` / `m.min_(...)` are `llvm.maximum` /
`llvm.minimum` on gfx950 (`v_maximum3_f32`, `v_minimum3_f32`), where they were
`fmaxf` / `fminf`.
- Two semantic changes, on values no shipped kernel reduces: a NaN anywhere
  in what is reduced now gives NaN, however the reduction is split across
  lanes, warps or blocks (as `torch.amax` does; maxNum skipped it), and -0
  orders below +0 (the max of -0 and +0 is +0, the min -0; maxNum may return
  either). `tests/hk/gpu/test_reduce_nan.py` holds both, within a wave, across
  warps and across blocks.
- Unchanged: elementwise `hk.max` / `hk.min` (`maximum`, `minimum`, clamps)
  stay `fmaxf` / `fminf` everywhere; gfx942 (no `v_maximum3_f32`: LLVM would
  expand each maximum into compares and selects) and RDNA keep `fmaxf` for
  reductions too. docs/language.md has the two kinds.
- Why: in IEEE mode LLVM quiets a maxNum operand it cannot prove quiet with a
  `v_max_f32 x, x`. A reduction of 2k + 1 values pairs into `v_max3_f32`s but
  for one two-operand max, whose operands (MFMA results, loop-carried values)
  it never can: attention's row max took 3 instructions for that 1. The d128
  main loop is 693 -> 680 instructions, 348 -> 338 VALU.
- Outputs: every launch of every `tests/hk/gpu` test, re-run as a twin built
  from the previous headers and emitter, is bit-identical (1182 launches; the
  6 that differ also differ from themselves run to run: atomics). The GEMM and
  the MXFP4 MoE and MXFP8 linear kernels compile to the same instructions as
  before.

**CONTRIBUTING.md: new kernels add no kernel-specific C++ helpers or one-off DSL ops.**
- A needed exception goes through review with a written reason and is registered as an escape
  hatch with a usage boundary. The file also has a reviewer checklist.
- [docs/design/dsl-surface.md](docs/design/dsl-surface.md), the approved design, inventories
  every kernel-specific op and helper and classifies each: 17 to merge, 28 to keep, 2 to delete.
  It names the six general abstractions the merges fold into and gives a migration in which every
  kernel stays bit-identical and no kernel gets slower. The decisions on its open questions are
  recorded in it.

**gfx950: an LDS tile narrower than its read's layout was never copied.** A
tile with fewer rows or columns than the include/cdna4 st shape its read's
fragment picks -- a 32x16 bf16 tile read as the 32x16 row operand of the
32x32x16 MFMA, which picks st_32x32 -- held no whole subtile of that shape, and
include/cdna4's copies loop over subtiles: every store and load of it compiled
to nothing, and the MFMA read whatever the LDS held, with no error. The
translator now gives such an allocation the fragment's own shape (st_32x16
here) when that one tiles it, and refuses one that no shape tiles.
`tests/hk/ir/test_lds_narrow_tile.py`, `tests/hk/gpu/test_lds_narrow_tile.py`.

**gfx950: `hk.ops.sparse_mla`, DeepSeek-V4.1's sparse MLA prefill, and a vLLM route
for it (`HK_VLLM_SPARSE_MLA`, off by default).** The attention vLLM's ROCm path runs
for a prefill chunk (AITER's opus kernel at T >= 1024): 32 heads of 512, each
query's sliding window and top-k rows gathered by index into LDS by the DMA.
`prefill_mxfp8` writes instead the layer's inverse-RoPE'd MXFP8 rows for wo_a,
made from O in registers (`lane_strip`) and bit for bit vLLM's quant of `prefill`'s
bf16 rows; vLLM's bf16 rows are scratch, so none are written. Calls below T 512
are left to vLLM. The kernel's gathered copies are bounded by each
query's rows (`rows=`), so no index past a query's list is read: a real-weight server
faulted on a mixed step before. Serving with real weights (DeepSeek-V4.1-Flash,
MI355X TP2): GSM8K equal to stock's. The route is opt-in
(`HK_VLLM_SPARSE_MLA=1`; a later entry turns it on by default). `tests/hk/gpu/test_sparse_mla.py`,
`tests/hk/gpu/test_vllm_sparse_mla.py`.

**gfx950: fp32 LDS vectors and 16-bit col-layout LDS stores are ds ops, not FLAT.**
`hk.SharedVec`'s load and store (what `hk.collective.cross_warp` combines
through) and the store of a 16-bit col-layout register tile into LDS (an
accumulator converted to bf16 -- attention's P handed to other warps) went
through include/cdna4's generic-pointer paths, so the compiler emitted FLAT
loads and stores. FLAT counts in vmcnt as well as lgkmcnt: a sparse-MLA kernel's
per-block max exchange waited for its next KV block's DMA. `include/hk/cdna.cuh`
now has the same accesses through LDS pointers, as the fp32 tiles got in #59.
`tests/hk/codegen/test_cdna_fp32_lds.py`.

**gfx950: a matrix-core result as a strip, where it sits: `Tile.lane_strip`,
`hk.lane_coords`, `Tile.run_xor`, and group reductions over any xor offsets.**
`acc.lane_strip(run)` takes a register tile's registers, lane by lane, as a 1-row
strip. No data moves. `hk.lane_coords(acc, run)` gives each element's row and
column in the tile, so `hk.map_cpp`, `group_max` and `store_elems` work on an MMA
result without a round trip through memory: an epilogue that quantizes
attention's output to MXFP8 per 32-row group. `run_xor(mask)` takes an element's
partner inside its lane's run, such as a rotary pair. `group_max`/`group_min`/
`group_sum` take any distinct power-of-two xor offsets up to 32, not only the
neighbouring 1 .. 2^(k-1). A 32x32 result's 32-row column block is a lane's
run plus lane l ^ 32's: `group_max(32, xor=(32,))`.
`tests/hk/gpu/test_strip.py`, `tests/hk/ir/test_strip.py`.

**vLLM, DeepSeek-V4.1 under `--monokernel`: the index layers as one hk launch
(`HK_VLLM_DSV41_INDEX`, on by default on gfx950).** Between dsv41_mono's two
launches each index layer ran vLLM's 9-12-kernel chain (`mono_decode._select`).
At decode steps of up to 8 rows, `hk.integration.vllm_dsv41_index` runs it as
one resident launch, `hk.ops.dsv41_index`: compressor, compressed-cache insert,
indexer K and Q, and, for rows of up to 16640 entries, the logits and the
top-k -- the candidate source's block top-k and the consumers' candidate mask
included. The caches, q and q_scale are vLLM's byte for byte and the top-k sets
equal (docs/integration.md, "The DeepSeek-V4.1 index launch"). Serving on
MI355X (TP2, max_model_len 16640, with hk's other default routes): GSM8K-1319
within stock's noise.
The route patches through an import hook, since the plugin must not import the
model package, and its contract is read from the module's source.

**CDNA: the MFMA hazard gate pads what LLVM leaves short.** The hardware
does not interlock on an MFMA's result. LLVM before llvm/llvm-project#225989
-- every ROCm release so far, 7.2.3 and the 10.x images included -- pads too
little on the shortest path into a block it first reached the long way
(ROCm/llvm-project#5153), and gfx950's paged decode computed some scores
wrong from it. hk's build now checks every CDNA code object
(`hk.runtime.mfmahaz`: a shortest-distance walk backwards over the control
flow from each instruction that touches an MFMA's registers, against LLVM's
windows) and, by default, rebuilds a kernel that is short anywhere with the
`s_nop` padding the fixed compiler would emit (`HK_MFMA_GATE=fix`; `refuse`,
`report` and `0` are the other modes). The build log, the cache entry's
`MFMAHAZ` and `python3 -m hk.integration --status` name each padded kernel
and the places. A compiler that pads the gate's two-path probe is not
checked. Swept over every kernel the test tiers and the default routes build:
- ROCm 7.2.3 (CI's GPU image): gfx950's dense attention, head_dim 64 and 128,
  not causal -- `hk.attention`'s kernel -- reads S for the softmax's max 3-4
  wait states after the last QK MFMA on the taken path of a branch, where 12
  are needed (2 places a kernel). Padded, its output is bitwise the unpadded
  one's and its time unchanged within noise (6 shapes, one process).
- ROCm 10 (the vLLM image): HipKittens' causal attention (`attend_ker`,
  hk.ops.hipkittens), 3-6 wait states short at the join after an exec branch
  (6 places at head_dim 128, 3 at 64); and one `hk.ops.gemm_cdna_dp`
  candidate with a bias+GELU epilogue (an MFMA reading an overlapping
  accumulator 1 wait state early).
- The default-on routes' kernels (vllm_mxfp8, vllm_moe_a8w4, vllm_int4) are
  clean under both; gfx1100 / gfx1201 have no MFMAs.

**`warm-routes` builds every K2 kernel a server can pick.**
- `python3 -m hk.integration warm-routes` called K2 (the MXFP8 linear route) at
  M = 2048 / 5120 / 8192 / 16384 only. The four-warp `stages=2` schedule that
  `vllm_mxfp8.RANGES` takes at M 832-1376 of (5120, 4096) was never built, so
  every vLLM server compiled it on its first request in that range, in every
  TP worker (the `[hk] JIT: compiling kernel
  'mxfp8_4w_block_m128_block_n128_split_m1_split_n1_stages2'` line).
- `warm_routes.k2_plans(n, k)` now enumerates, for each routed (N, K), every
  (kind, schedule, tail recut) the route can launch at M up to 32768, and
  warm-routes makes one call at the smallest M of each.
  `tests/hk/ir/test_warm_routes.py` checks the cover against an independent
  scan of `schedule` and `mxfp8.tail_split`.

**gfx950: a gathered LDS copy takes a row limit, and reads no index past it.**
`g.load(..., wait=False, gather=(ids, at))` read `ids[at + r]` for every row r of the
tile. When an index list ended inside the tile (the last query of a CSR), the copy
read up to a tile of ids past the list's end. Usually that lands in the allocator's
slack; when the list ends its segment, it is a memory fault. In vLLM that was the
sparse-MLA prefill, under a real-weight server's mixed batch. `rows=` (refused with
`gather=` until now) now bounds a gathered copy: rows from the limit on read zero,
and their ids are not read; and no gathered copy or `g.gather_prep` reads an id past
the ids tensor (those rows read zero too). The copy stays branch-free, so its index
loads still issue together. `tests/hk/gpu/test_gather_load.py`,
`tests/hk/gpu/test_fenced.py::test_gathered_copy_reads_no_index_past_its_rows`.

**CDNA: `hk-mma-fence`, a DSL rule for matrix-core accumulators; `hk.arch.agpr`
to hold one in AGPRs.**
- `hk-mma-fence`, a pass in the kernel pipeline, puts an `hk.sched_barrier`
  between a matrix-core chain and the first other op that touches its
  accumulator. On gfx950 LLVM threaded the softmax's reads of AGPR scores
  through the two-page paged decode's MFMA chains and computed some scores
  2-4e-2 wrong, differently from one build to the next: one of those reads
  sat a branch after its MFMA, and LLVM's hazard recognizer, measuring the
  longer path into that block, padded 2 wait states where 8 are needed
  (fixed in LLVM by llvm/llvm-project#225989; ROCm 7.2.3 to 10.1 do not have
  it, ROCm/llvm-project#5153). The decode's hand-written fence is gone; the
  rule covers every CDNA kernel. `hk.arch.mma_fence(False)` turns it off for an
  A/B (`attn_kernel(mma_fence=)`, `decode_kernel(fence_qk=)`).
  - Where it lands: attention's prologue (its main loop fences its phases
    already), the decode's per-step scores and epilogue, GEMM epilogues.
  - Cost, measured in one process against the previous code (MI355X): dense
    attention unchanged within noise (8 shapes, outputs bitwise equal); paged
    decode unchanged within noise over 120 points, except one request on 1 kv
    head at 1k-4k tokens, slightly slower (the per-step fence of the one-page
    walk).
- `hk.arch.agpr(tile)` holds an fp32 accumulator that `hk.zeros` just made in
  AGPRs (CDNA): mmas into it are inline asm with C/D in AGPRs, other accesses
  go through a VGPR copy after the MFMAs' wait states
  (`include/hk/cdna_agpr.cuh`). Outputs are bitwise the VGPR form's.
  `attn_kernel(acc="agpr")` puts attention's O there; it stays off by default:
  it is slower on full-attention shapes, because LLVM cannot schedule around an
  asm MFMA. LLVM 22's register split spills the d128 kernel;
  the causal pairs walk (two O tiles) spills on either toolchain.

**Attention tests hold the kernels to their own error.** Every attention
test (test_attn, test_sdpa, test_paged, test_paged_fused, test_paged_prefill,
test_attn_k3, test_hipkittens' GQA, test_schedules' attention,
test_vllm_backend_decode, test_fenced's attention cases) compares against
attention in float64 (`tests/hk/gpu/_attn_ref.py`) instead of torch's fp32
SDPA, which moved by more than the kernel's error from run to run on ROCm,
through one check, `_attn_ref.check`:
- every output row within 4 bf16 ulps of that row's largest value (worst
  measured: 2.31 on gfx950, CI and vLLM images; 1.32 on gfx1100);
- and 1e-2 of the whole output's largest value (worst 7.1e-3 / 5.3e-3),
  where the tests asked 2e-2 to 5e-2.
The gfx950 decode that computed some scores wrong (ROCm/llvm-project#5153)
was 16-30 ulps off in its worst row and passed the old bounds. The value
caches of the paged tests now come from the test's own generator, so a case
sees the same data in any order.

**gfx950 paged decode: two pages a step, a cheaper merge, and no point
slower than before the 512-split plan.** The previous entry's plan was slower
than the one before it on short sequences (1-2k tokens); this one is not.
- Where (requests x kv heads) is under 64 or 256-511 -- about one workgroup a
  CU -- the walk takes two pages a step on splits of 8 pages or more and one
  on shorter ones, in one kernel (`paged_cdna.pages_per_step`,
  `decode_kernel(ppstep=2)`, `hk.ops.paged.fused_kernel(ppstep=)`).
- Two pages a step need a scheduling fence after QK (`decode_kernel(fence_qk=)`,
  on by default, no measurable cost). Without it the current LLVM threads the
  softmax's reads of the AGPR scores through the two pages' MFMA chains, and
  some scores come out wrong: 2-4e-2 off on a 128-token split, a different
  answer from one build to the next.
- Splits take at least 2 pages where the merge reads every split (up to 64
  splits, `paged_cdna.merge_reads_all`, `min_pages_for`) and, with more
  splits, on sequences of up to 128 pages (`floor_pages`); 4 otherwise. A
  short sequence on a plan for long ones walked 4 pages a split.
- The merge: up to a round of 8 splits a warp it reads every split without
  first loading the sequence length (the walk writes empty splits' zeros); it
  folds on at most 8 warps (`MERGE_WARPS`); and it sums only the warps that
  folded something.
- On many-round plans an empty split skips its query load and copies, and
  split indices are turned by the (request, kv head) pair so that a short
  sequence's splits spread over the XCDs.
- Measured on a decode grid (MI355X; 32/8, 64/8, 32/4 and 8/1 heads; 1-64
  requests; 1k-128k tokens; CUDA graphs) in one process against both earlier
  plans: no point is slower than before the 512-split plan beyond noise.
- Outputs change by the order pages and splits are folded in (within the fp32
  check; deterministic under another process on the GPU).

**gfx950 paged decode: a merge on several warps, and up to 512 splits.** The
split decode's merge (`hk.ops.paged_cdna.merge_kernel`) folded every split of
a (request, kv head, query row) on one warp, 8 splits a round. That capped
the plan at 32 splits, and a request on few kv heads walked its whole context
on 32 to 128 workgroups.
- The merge now folds on up to 16 warps (`paged_cdna.merge_warps`, about one
  round of 8 splits each), combined through LDS; one round stays one warp's.
- The plan (`hk.ops.paged.plan_splits` on gfx950, `paged_cdna.target_workgroups`)
  aims at 512 one-warp workgroups from 64 to 255 (request, kv head) pairs and
  256 otherwise, up to 512 splits (`paged_cdna.MAX_SPLITS`).
- The walk no longer writes partials for splits the merge does not read.
- The plan is for long contexts on few (request, kv head) pairs. Short sequences
  (1k tokens at 1-4 requests) got slower under it; the entry above fixes that.
- Outputs change by the order splits are folded in (within the fp32 check;
  deterministic). `HK_SPLITS` still forces a count.

**gfx950 paged prefill: long KV walks cut over the cached prefix.** A
short chunk on a long prefix, or a batch on few heads, is a few query tiles
of 256 rows walking the whole context, with most of the machine idle: 4096
cached + 512 new tokens on 32/8 heads is 64 workgroups for 256 CUs.
- When the tiles times the heads come short of 256 workgroups,
  `hk.ops.paged_prefill.split_plan` cuts each tile's walk over the prefix
  every one of its queries sees, at least 16 64-key blocks a piece, the last
  piece taking the rest with the diagonal. The cut tiles write normalised
  partials and (m, l) (`attn_cdna.attn_kernel(kv_split=True)`,
  `split_kernels_for`), and `attn_cdna.prefill_merge_kernel` weighs them.
- Its entries go longest first across requests. On one or two kv heads the
  vLLM backend takes that order even with nothing to cut. On more kv heads that
  order costs L2 reuse, and the backend keeps the single-tile work list unless
  the plan cuts.
- The vLLM backend plans it in the metadata builder, from
  `seq_lens_cpu_upper_bound` (exact for prefill rows); without it the
  prefill is the single launch as before.
- Measured on a prefill grid (MI355X; 32/8, 64/8, 32/4 and 8/1 heads; 13
  batches from fresh prompts to 65536 + 2048 chunks) in one process against the
  previous code: no batch slower beyond noise.
- Cut tiles' outputs change by the order partials are folded in (within the
  fp32 check; deterministic). Uncut ones are bitwise as before.

**gfx950 causal attention walks query blocks in pairs.** A causal workgroup of
`hk.ops.attn_cdna` (what `hk.attention` and the SDPA drop-in run on gfx950)
now walks query block nqb - 1 - y and then block y, so every workgroup does
the same work. Before, longest-first order inside each batch still ended
grids with few workgroups a batch on a round of long tiles.
- Outputs are bitwise the same (all 42 causal points of the attention grid,
  head_dim 128, 8/1 to 64/8 heads, N 1k-32k).
- It helps most where few heads leave the grid short (8 query / 1 KV heads, a
  70B-class model at TP8, and 16 / 2 heads, at short N); measured in one process
  against the single-block walk, no other point moves beyond noise either way.
- `attn_cdna.kernel_for(..., pairs=False)` and the `pairs` knob of
  `hk.schedules.AttnCdna` give back the single-block walk. The spatial
  numbering and the paged prefill do not change.

**The MXFP8 linear route below its MIN_M** (gfx950, `vllm_mxfp8.RANGES`). DeepSeek-V4.1's MXFP8
linears at decode and mid-size M were measured route level against the other available kernels:
hk now takes engram wkv (12800 x 6144) from M 160, wo_b (5120 x 4096) from 416, the shared
expert's w2 (5120 x 1152) from 512, wq_b (16384 x 1280) from 576, the shared expert's w1|w3
(2304 x 5120) from 960 and fused wq_a|wkv (1792 x 5120) from 1216, only in the ranges where it
measured faster in every round of two sweeps, on the four-warp kernel at 128 x 128 or the
eight-warp one. Bitwise vLLM's output at every point. At M <= 128, and for the indexer's wq_b
below its MIN_M, those calls stay with vLLM.

**Host time per call** (#91, cross-arch). An engine route's Python no longer
re-decides per call what only the call's shapes decide, and code-object
kernels launch from a per-shape memo without the torch dispatcher
(`hk.runtime.hostpath`, [integration](docs/integration.md#host-time-per-call)).
Same kernels, same arguments, bitwise the same outputs; `HK_HOST_FASTPATH=0`
restores every call's previous path.
- Host µs per call, before -> now: `vllm_gemm` decode 38.5 -> 14.0 and
  prefill 46.5 -> 17.6 (W7900D); `vllm_int4` 33.6 -> 12.5 (W7900D) and
  19.9 -> 9.4 (MI355X); `hk.ops.sdpa`'s DiT attention 82.5 -> 49.2; the a8w4 MoE route
  167.1 -> 122.6, and a MoE call it leaves to AITER 252.6 -> 208.6.
- Code-object launches in C (#92) where hk._mlir has `_hkLaunch` (built with the bindings),
  with less host time a launch than the pybind entry point and the Python launcher.
- New: `hk.runtime.hostpath` (`ENABLED`, `set_enabled`),
  `hk.runtime.codeobj.direct`, `hk.runtime.torch_ext.launcher`,
  `CodeObjectKernel.launcher` (`native=`, #92), `hk.ops.w4a16.plan`. They are runtime and
  integration names, which the API reference does not cover, so they carry
  no mark there; [the runtime](docs/runtime.md) and
  [integration](docs/integration.md#host-time-per-call) describe them.

**DeepSeek-V4.1 routes, new thresholds** (#84, gfx950, real weights, TP2).
- The a8w4 MoE route (`vllm_moe_a8w4`) takes M >= 2560, not 8192 (`HK_MOE_A8W4_MIN_M`), with
  #85's kernels. The threshold was measured on the real router's choices; with uniform routing,
  which DSV4.1's router does not produce, AITER is faster at the low end of that range (the
  reasons are at `MIN_M`).
- The MXFP8 linear route (`vllm_mxfp8`) takes the indexer's wq_b (N 4096, K 1280) from M 2825,
  and engram's wkv (12800 x 6144) from M 1024 instead of 2048; per (N, K) in `MIN_M_NK`.
  Bit-identical to vLLM's kernel at every new point, ragged M included.

**fp8 GEMM on gfx1201** (#76, RDNA4). `hk.ops.scaled_mm(a, b, scale_a, scale_b,
*, bias=None, out=None)` is `torch._scaled_mm`'s product, bf16
`(A @ B.T) * scale_a * scale_b (+ bias)` from OCP fp8 operands with fp32 accumulation, in
`hk.ops.matmul`'s (N, K) layout, with per-tensor or per-token x per-channel scales. It runs on
gfx1201's fp8 WMMA: `hk.ops.gemm`'s kernel takes `op_dtype=hk.fp8e4m3` / `hk.fp8e5m2` there,
and gfx1100 (no fp8 WMMA) and gfx950 (whose fp8 GEMMs are gemm_cdna's) refuse it. The decode
shapes (M 1-64) are not tuned yet.

**The MXFP4 a8w4 MoE kernel at prefill M: a shorter stage 2** (gfx950). Outputs are bit for
bit those of before; decode M unchanged. The route's thresholds are not touched.
- `store_scatter` stores 16x16 fp32 fragments as bf16 in 16-byte runs (`v_permlane16_swap` pairs
  lanes l and l+16): half the store instructions, on stage 2's scattered top-k rows.
- `Global.scatter(..., staged=N)` reads its row ids from an LDS copy made by
  `g.gather_prep(ids, at, N, raw=True)` (both new after 0.8.0). A global id load in an epilogue is a vmcnt
  wait, and on gfx950 that retires, in order, every load and store issued before it: in the
  resident stage 2, the next column tile's weight prefetch and the epilogue's own stores. Stage 2
  now stages its block's ids, and loads its rows' routing weights once, not once a column tile.
- Both stages issue the next token fragment's LDS read before the current fragment's MFMAs
  (`stage1_sorted(xpipe=True)`, the default).
- `load_fp4_frags` builds its per-expert buffer resource with a 32-bit compare and a select: the
  64-bit bounds check was two scalar branches at every K step.

**Attention on gfx1201** (#77, RDNA4). `hk.attention` (and the SDPA drop-in) runs its own
tiling on gfx1201 from 256 query rows: 16 warps (a 256-row Q tile), a head-first grid
that starts the longest rows of every head first, and, causal, warps that skip the KV
blocks they precede. The causal KV block depends on the compiler: 64 rows under ROCm
7.2.3's clang 22, 32 under the R9700 vLLM image's clang 23. A shorter Q keeps gfx1100's
192-row tiling, so what `hk.attention` accepts is the same on every RDNA arch.
`attn_kernel(warp_skip=True)` without `causal` raises.

**Element ops for a whole cache** (#88, gfx950). The forms DeepSeek-V4.1's
index chain needed as one launch on vLLM's caches:
- `Global.load_elems` / `store_elems(..., row=r)`: indices count from row `r`,
  its base formed in 64 bits and its descriptor that one row -- a paged cache
  of ~100 GB addressed a block at a time (a descriptor over the tensor wraps
  at 4 GB).
- `Global.load_elems(..., runs=True)`: a lane's run is one load of its bytes
  (masked by its first index), not one `buffer_load_dword` a value.
- `SharedTile.load_elems(..., runs=True)`: 4 elements a 16-byte LDS read. Index
  a 16-column tile: a wider `hk.ST` is 16x16 subtiles 1 KB apart, and a
  flat-indexed strip conflicts 8 ways on every read.
- `SharedTile.store_elems` into a bf16 tile (round to nearest even): an MMA
  operand built element by element. LDS element reads are branch-free (an
  index out of range still reads 0); a guarded read was one exec-masked block
  each, and a strip's reads ran one at a time.

**Norms on gfx1201** (#78, RDNA4). The strip plans take per-lane targets measured on the
R9700, for rms_norm and the fused add + rms_norm. gfx1100's and gfx950's plans are unchanged.

**Ops for an index top-k in one launch** (#98, gfx950), from DeepSeek-V4.1's
index logits and top-512 inside the same launch:
- `SharedTile.atomic_add_elems(val, idx)`: i32 adds into an LDS tile
  (`ds_add_u32`), every add to one element landing -- a radix select's
  histogram.
- `hk.fp4_operand(words)` / `hk.mx_scales_loose(bytes)`: an MXFP4 MMA operand
  and its E8M0 scales from a lane's own gathered dwords / byte, for operands in
  a layout no tile load knows (logits straight off a preshuffled paged K
  cache).
- `Tile.first_lane()`: lane 0's element of an i32 strip as a scalar
  (`v_readfirstlane`); one LDS word read by every wave after a barrier may
  then drive an `hk.uniform` branch (a select's exact-fit early exit).

**Ops for a MoE in one launch** (#104, gfx950), from DeepSeek-V4.1's MoE as one
resident hk launch, bitwise vs vLLM's `--monokernel` MoE:
- `while hk.uniform(t < n):` -- a runtime loop every warp runs the same
  iterations of may hold barriers: a work queue's loop over `hk.comm.claim`.
- `hk.fp8_operand(words)`: an MXFP8 MMA operand from a lane's eight dwords, for
  fp8 operands kept in LDS as words (an fp8 shared tile has no register path).
- `hk.Tensor[..., "stream"]`: non-temporal accesses for a weight stream a launch
  reads once. Not a scope: mailboxes, flags
  and symmetric memory refuse it.

**vLLM on gfx1201** (#80, RDNA4), in the Radeon AI PRO R9700 vLLM image
(`ghcr.io/inferstation/vllm-rocm-r9700-main`: TheRock ROCm 7.14, clang 23, torch 2.13).
- `vllm_gemm` is on by default on gfx1201, with a measured route table
  (`routes_vllm_linear_gfx1201.json`): tokens identical to stock and routing counts equal
  round to round on Qwen3-1.7B and Qwen3.5-9B.
- `vllm_int4` (AWQ) is on by default on gfx1201: tokens identical to the image's AWQ path on
  Qwen3-8B-AWQ.
- `vllm_moe` and `vllm_sdpa` stay off there: the MoE route is right but slower at 16 tokens,
  and the SDPA drop-in's per-call host time outweighs its kernel.
- The decode GEMM's four-tile schedule on gfx1201 is combine depth 2
  (`decode_gemm.SCHEDULES_AT`), the one that builds without a spill under both compilers.

**Attention on gfx1201 at head_dim 64** (#94, RDNA4). The 16-warp tilings now cover
head_dim 64 too: causal from 1024 query rows, full from 2048, under either compiler. Below
those lengths gfx1100's tiling stays, which is faster at a ViT's 577 and 1025 tokens.

**gfx1201's bf16 GEMM at 32-512 rows** (#102, RDNA4). hk ships tuned schedules for an
RDNA arch in `python/hk/tuned/routes_gemm_<arch>.json`, in `hk.tune`'s record format,
read behind a machine's own records (`hk.tune.records.shipped_route`); `hk.ops.gemm.matmul`,
`schedule_for`, `hk.templates.gemm` and the vLLM route take them. gfx1201's 18, tuned under
both ROCm 7.2.3's clang 22 and the R9700 vLLM image's clang 23 on Qwen3's four (N, K). The
gfx1201 vLLM table, re-measured with them, routes most of M 33-256 to hk: Qwen3.5-9B at
32-128 requests, tokens identical to stock's. `HK_AUTOTUNE=0` ignores the shipped records.

**vllm_moe on gfx1201, up to 2 tokens a step** (#108, RDNA4). `vllm_moe.BLOCK16_MAX_TOKENS_AT`
gives an arch its own token limit for hk's MoE GEMM; gfx1201's is 2, where vLLM skips the
sort and gives each (token, choice) pair a block of its own. There, OLMoE tokens are
identical to stock's and routing counts equal round for round; past the limit vLLM's Triton
kernel runs. `vllm_moe` is on by default on gfx1201.

**gfx1201's bf16 GEMM at 1024-8192 rows** (#114, RDNA4). 11 more shipped gfx1201 schedules
in `routes_gemm_gfx1201.json`, for Qwen3's four (N, K), tuned under both the R9700 vLLM
image's clang 23 and ROCm 7.2.3's clang 22 (each compiler's own champions disagree at these
sizes; the shipped one is the one that holds under both).

**Fixes**
- **`hk.schedules.lookup(arch=...)`, `hk.templates.gemm.plan(arch=...)` and hk.tune's
  incumbent read the current arch's tuned records, not the asked arch's** (#114): under
  `HK_ARCH=gfx1201`, `lookup(arch="gfx1100")` could return a gfx1201 schedule illegal on
  gfx1100. `hk.ops.gemm.user_route` / `schedule_for` take an `arch` now.
- **gfx950: `hk.ops.gemm.matmul` raised `ResourceError` in the vLLM image** (#84) at M 192-512 for
  N 28672 / 14336, K 4096: its table routes there to `dp128x224`, which ROCm 10's hipcc builds
  with a 13-VGPR spill, and the spill gate refuses it. The route now records a refused candidate
  (once a process, with a line on stderr) and takes the next-fastest one the table measured at
  that point (`pp128`, `ppk2`), else the rule's next; a refused kernel is never run.
- **A kernel registered as a torch op through libtorch's headers did not
  compile against TheRock's torch 2.13** (#79), the torch of the Radeon AI PRO
  R9700 vLLM image (ROCm 7.14), whose ROCm headers are hipified the newer way
  (`c10::cuda::CUDAGuard`, no `c10::hip::HIPGuard`): the C++ shim
  `hk.codegen.scaffold` writes for `hk.runtime.torch_ext` failed with
  `no type named 'HIPGuard' in namespace 'c10::hip'`. It now picks the names by
  `C10_HIP_API`, which only the older headers define; with those (rocm/pytorch's
  torch 2.10, the vLLM 0.31 venv's 2.13, the gfx950 images) it builds the same
  code as before.
- **The LDS wait check refused a correct kernel** (#87). `hk.runtime.cocheck`'s scan walked the code in
  address order and carried a block's in-flight LDS loads past an unconditional `s_branch` into a block
  that code never reaches. It now follows branches: a forward branch carries its states to its target.
- **An LDS tile allocated after one inside a branch not every warp takes landed at two addresses** (#96).
  HipKittens' `shared_allocator` hands out offsets in execution order, so the warps that skipped the branch saw
  every later tile elsewhere -- silently (DeepSeek-V4.1's index-chain top-k read two histograms). `hk.lds.alloc` now
  refuses an allocation that follows such a branch outside it; inside the branch, or in a workgroup-uniform one,
  stays allowed.

**CI and tests**
- A fourth GPU-tier job, `gfx1201`, on a Radeon AI PRO R9700 (RDNA4). It is continue-on-error and not a required
  check until the RDNA4 port's tier is green (#67,
  [testing](docs/testing.md#the-gfx1201-runner)).
- The `gfx1201` job is no longer continue-on-error, now that the RDNA4 port's GPU tier is green
  there: a red gfx1201 fails the run (#97). Whether it is one of main's required checks is a
  branch-protection setting.
- gfx1201 has CI legs of its own for the tiers that need no GPU: `IR tier, gfx1201` and
  `codegen tier (I/4), gfx1201`, with `HK_ARCH=gfx1201` and the same ROCm 7.1 hipcc (#106).
- A performance-gate baseline for gfx1201, `tests/perf/baselines/gfx1201.json` (34 pairs:
  gemm, attention, norms, softmax, fp8 and int8 against torch, Inductor and Triton);
  `check_regression.py` takes each arch's scoreboard arguments (`ARGS_AT`). Run by hand on an
  R9700, as gfx1100's is, not in the gfx1201 job (#106,
  [testing](docs/testing.md#the-gfx1201-runner)).
- Pull requests merge in batches through a merge train, `scripts/ci/merge_train.py`: ready
  PRs are stacked on main in order, tested together once (CI and the GPU tier), and
  squash-merged in that order if both are green; a PR that conflicts or turns the batch red
  is dropped from it (#113, [testing](docs/testing.md#the-merge-train)).
- A gfx950 job waits for an idle MI355X (a configurable limit, 60 minutes by default) instead
  of failing when its GPUs are busy; the gfx950 and gfx950-vllm jobs' timeout is 360 minutes,
  which covers the wait (#123).
- The merge train resolves a conflict in `CHANGELOG.md` by union instead of dropping the PR.
  The union is checked before it is kept: no conflict markers, no entry twice, and each
  version section kept as it was (#128, [testing](docs/testing.md#the-merge-train)).

### All pull requests

Every pull request merged since 0.8.0 (125), by number:

- [#67](https://github.com/vivienfanghuagood/hk-dsl/pull/67) CI: a gfx1201 job on the RDNA4 dev box (remote-registered ephemeral runner, continue-on-error)
- [#76](https://github.com/vivienfanghuagood/hk-dsl/pull/76) hk.ops.scaled_mm: fp8 GEMM on gfx1201's fp8 WMMA (RDNA4 R3b)
- [#77](https://github.com/vivienfanghuagood/hk-dsl/pull/77) hk.attention on gfx1201: its own tiling, ties AITER's FlyDSL kernel (RDNA4 R3c)
- [#78](https://github.com/vivienfanghuagood/hk-dsl/pull/78) norms on gfx1201: strip plans measured there, the last scoreboard losses go (RDNA4 R3d)
- [#79](https://github.com/vivienfanghuagood/hk-dsl/pull/79) codegen: the torch-op shim builds against both spellings of torch's ROCm headers
- [#80](https://github.com/vivienfanghuagood/hk-dsl/pull/80) gfx1201 in vLLM: the R9700 image's toolchain, and vllm_gemm on by default there (RDNA4 R4a)
- [#84](https://github.com/vivienfanghuagood/hk-dsl/pull/84) DeepSeek-V4.1 routes retuned from the MI355X scoreboard (K1 from M 4096, K2 indexer wq_b + engram wkv), and gemm_cdna routes around a candidate the spill gate refuses
- [#87](https://github.com/vivienfanghuagood/hk-dsl/pull/87) cocheck: the LDS wait scan follows branches (a false refusal)
- [#88](https://github.com/vivienfanghuagood/hk-dsl/pull/88) Element ops for a whole cache: row= past 4 GB, runs=True loads, 16-byte LDS reads, bf16 LDS element stores (gfx950)
- [#90](https://github.com/vivienfanghuagood/hk-dsl/pull/90) K1 MoE a8w4 prefill: 16-byte scatter stores, LDS-staged scatter ids, LDS read pipelining (gfx950)
- [#91](https://github.com/vivienfanghuagood/hk-dsl/pull/91) runtime, integration: host fast paths -- an eager route call no longer re-decides what its shapes decide
- [#92](https://github.com/vivienfanghuagood/hk-dsl/pull/92) runtime: code-object launches in C (_hkLaunch)
- [#93](https://github.com/vivienfanghuagood/hk-dsl/pull/93) docs/testing: the GPU allocation on the gfx950 workstation
- [#94](https://github.com/vivienfanghuagood/hk-dsl/pull/94) attention on gfx1201 at head_dim 64: the 16-warp tilings (RDNA4 R3e)
- [#95](https://github.com/vivienfanghuagood/hk-dsl/pull/95) post-release 0.8.0: API marks against the 0.8.0 wheel, RELEASE = None, CHANGELOG Unreleased
- [#96](https://github.com/vivienfanghuagood/hk-dsl/pull/96) lds.alloc: refuse a tile after one allocated in a branch not every warp takes (silent wrong addresses)
- [#97](https://github.com/vivienfanghuagood/hk-dsl/pull/97) gpu.yml: the gfx1201 job no longer continue-on-error (RDNA4 R2 done)
- [#98](https://github.com/vivienfanghuagood/hk-dsl/pull/98) Ops for an index top-k in one launch: LDS atomic adds, MXFP4 operands / scales from gathers, first_lane (gfx950)
- [#99](https://github.com/vivienfanghuagood/hk-dsl/pull/99) vllm_mxfp8: K2 below MIN_M where hk was measured faster -- six of DeepSeek-V4.1's seven MXFP8 linears from M 160-1216 (bitwise vLLM)
- [#100](https://github.com/vivienfanghuagood/hk-dsl/pull/100) attn_cdna: causal workgroups walk query blocks in pairs
- [#101](https://github.com/vivienfanghuagood/hk-dsl/pull/101) paged decode on gfx950: a merge on several warps, plans up to 512 splits
- [#102](https://github.com/vivienfanghuagood/hk-dsl/pull/102) gfx1201's bf16 GEMM at 32-512 rows: shipped hk.tune schedules, and the vLLM table re-measured with them (RDNA4 R3)
- [#103](https://github.com/vivienfanghuagood/hk-dsl/pull/103) paged prefill on gfx950: cut long KV walks over the cached prefix
- [#104](https://github.com/vivienfanghuagood/hk-dsl/pull/104) K2 MoE ops: work-queue loops with barriers, hk.fp8_operand, non-temporal stream tensors (gfx950)
- [#105](https://github.com/vivienfanghuagood/hk-dsl/pull/105) paged decode on gfx950: two pages a step, a cheaper merge, and no point slower than before the 512-split plan
- [#106](https://github.com/vivienfanghuagood/hk-dsl/pull/106) CI: gfx1201's IR and codegen tiers, and a gfx1201 performance gate (RDNA4 follow-up 1)
- [#107](https://github.com/vivienfanghuagood/hk-dsl/pull/107) CDNA: hk-mma-fence, a DSL rule for matrix-core accumulators; hk.arch.agpr to hold one in AGPRs
- [#108](https://github.com/vivienfanghuagood/hk-dsl/pull/108) vllm_moe on gfx1201, up to 2 tokens a step (RDNA4 follow-up 2)
- [#113](https://github.com/vivienfanghuagood/hk-dsl/pull/113) The merge train: READY PRs stacked twice a day, tested once, squash-merged in order
- [#114](https://github.com/vivienfanghuagood/hk-dsl/pull/114) gfx1201's bf16 GEMM at 1024-8192 rows: shipped schedules that hold under both compilers; records read by the asked arch (RDNA4 follow-up 3)
- [#115](https://github.com/vivienfanghuagood/hk-dsl/pull/115) vLLM: DeepSeek-V4.1's index layers under --monokernel as one hk launch
- [#116](https://github.com/vivienfanghuagood/hk-dsl/pull/116) tests: attention against float64, each output row within 4 bf16 ulps
- [#117](https://github.com/vivienfanghuagood/hk-dsl/pull/117) gfx950: an LDS tile narrower than its read's st shape was never copied
- [#118](https://github.com/vivienfanghuagood/hk-dsl/pull/118) gfx950: fp32 LDS vectors and 16-bit col-tile LDS stores are ds ops, not FLAT
- [#119](https://github.com/vivienfanghuagood/hk-dsl/pull/119) CDNA: an MFMA hazard build gate; pads what LLVM left short (default)
- [#120](https://github.com/vivienfanghuagood/hk-dsl/pull/120) gfx950: hk.ops.sparse_mla, DeepSeek-V4.1's sparse MLA prefill, and a vLLM route (off)
- [#121](https://github.com/vivienfanghuagood/hk-dsl/pull/121) gfx950: a matrix-core result as a strip -- lane_strip, lane_coords, run_xor, any group xor offsets
- [#122](https://github.com/vivienfanghuagood/hk-dsl/pull/122) docs: converging the DSL surface (approved design: inventory, target API, migration) + CONTRIBUTING rule for new kernels
- [#123](https://github.com/vivienfanghuagood/hk-dsl/pull/123) CI: the gfx950 jobs' wait for a GPU is HK_CI_PICK_WAIT_MIN (runner .env), timeout 360 min
- [#124](https://github.com/vivienfanghuagood/hk-dsl/pull/124) warm-routes: build every K2 kernel a server can pick (the stages=2 schedule included)
- [#125](https://github.com/vivienfanghuagood/hk-dsl/pull/125) gfx950: a gathered LDS copy takes a row limit and reads no index past it
- [#128](https://github.com/vivienfanghuagood/hk-dsl/pull/128) merge train: CHANGELOG.md conflicts are resolved by union, not kicked; readmit
- [#129](https://github.com/vivienfanghuagood/hk-dsl/pull/129) gfx950: fp32 max/min reductions are IEEE maximum/minimum (v_maximum3_f32)
- [#130](https://github.com/vivienfanghuagood/hk-dsl/pull/130) DSL surface S0: the surface registry and its lint (hard CI), the V0/V1 snapshot tool, specialization tests
- [#136](https://github.com/vivienfanghuagood/hk-dsl/pull/136) compile cache: key on every header a kernel's compile reads (a header edit served the old kernel)
- [#137](https://github.com/vivienfanghuagood/hk-dsl/pull/137) dsv41_index: the candidate source's block top-k and the consumers' mask in the launch (max_model_len 16640)
- [#139](https://github.com/vivienfanghuagood/hk-dsl/pull/139) Public hygiene: remove internal infrastructure details (no behaviour change)
- [#140](https://github.com/vivienfanghuagood/hk-dsl/pull/140) ROCm 10.1 (clang 24): read its bare resource remarks; find LLVM tools beside hipcc's clang
- [#141](https://github.com/vivienfanghuagood/hk-dsl/pull/141) DSL surface S1–S5 + S12: scaled hk.mma, RowIndex, swap_halves, register views, cdna.cuh split (one commit a step) + docs follow-up
- [#142](https://github.com/vivienfanghuagood/hk-dsl/pull/142) Public hygiene follow-up: CDNA header notes, CHANGELOG process text, removed release links
- [#143](https://github.com/vivienfanghuagood/hk-dsl/pull/143) gfx1100: the decode GEMM builds under clang 23 (TheRock 7.14) -- LDS-pointer fp32 copies + hk.kernel(max_occupancy=)
- [#144](https://github.com/vivienfanghuagood/hk-dsl/pull/144) docs: one-screen positioning -- README top, docs/positioning.md, one sentence everywhere
- [#145](https://github.com/vivienfanghuagood/hk-dsl/pull/145) vLLM: the sparse MLA route on by default on gfx950 (owner's decision)
- [#147](https://github.com/vivienfanghuagood/hk-dsl/pull/147) DSL: strided globals on RDNA -- hk.Tensor[dtype, "strided"], read by g.stage (#146)
- [#148](https://github.com/vivienfanghuagood/hk-dsl/pull/148) gfx1100 under clang 24 (ROCm 10.1): the kernels the spill gate refused build again
- [#149](https://github.com/vivienfanghuagood/hk-dsl/pull/149) docs: performance-claim purge -- only the scorecard is quoted (docs reset, step 2)
- [#150](https://github.com/vivienfanghuagood/hk-dsl/pull/150) Release wheels for CPython 3.11 and 3.13 (packaging only; cp310/cp312 unchanged)
- [#151](https://github.com/vivienfanghuagood/hk-dsl/pull/151) docs/supported-hardware.md: one hardware truth table, generated from hk.target and the route table
- [#152](https://github.com/vivienfanghuagood/hk-dsl/pull/152) S7 (G1): integer, mask and bit expressions on strips -- hk.map, hk.where, hk-map-fuse; paged_cdna and sparse_mla migrated
- [#153](https://github.com/vivienfanghuagood/hk-dsl/pull/153) hk.templates: rowwise / elementwise stay callable after their module is imported (fixes the test-order failure)
- [#154](https://github.com/vivienfanghuagood/hk-dsl/pull/154) gfx942 (MI300X): ops with no kernel there refuse at the call, not in hipcc
- [#155](https://github.com/vivienfanghuagood/hk-dsl/pull/155) examples: a GPU-pinned example (14, hk.comm) says which GPUs and why, and the GPU tier skips it there
- [#157](https://github.com/vivienfanghuagood/hk-dsl/pull/157) Default-on routes: a GPU test on every arch they're on by default, held by a coverage table
- [#158](https://github.com/vivienfanghuagood/hk-dsl/pull/158) vLLM prefill attention on vLLM's own cache (opt-in): paged_prefill rocm / flash / dsplit layouts + HK_VLLM_PREFILL route
- [#159](https://github.com/vivienfanghuagood/hk-dsl/pull/159) README/positioning: the hardware table is generated between hw-table markers (+ gfx1201 strips in the templates guide)
- [#160](https://github.com/vivienfanghuagood/hk-dsl/pull/160) S7b: hk.map on RDNA strips (gfx1100, gfx1201) -- include/hk/rdna/map.cuh; masks stay inside a map
- [#161](https://github.com/vivienfanghuagood/hk-dsl/pull/161) gfx1100 routes: Qwen3.8-27B's TP2 linears in the vLLM and SGLang tables; linear_shapes reads hybrid decoders
- [#162](https://github.com/vivienfanghuagood/hk-dsl/pull/162) vLLM GPTQ-Int4 on RDNA3: vllm_gptq (off by default) on hk's 4-bit decode GEMM; w4a16 takes GPTQ's 32-column groups
- [#163](https://github.com/vivienfanghuagood/hk-dsl/pull/163) S8 (E1): gfx950's synchronous g.load is the asynchronous copy's DMA body + its waits
- [#164](https://github.com/vivienfanghuagood/hk-dsl/pull/164) S6 (G3): hk.lane_coords of a strip type; hk.iota is its column alias; the hk.iota op goes
- [#165](https://github.com/vivienfanghuagood/hk-dsl/pull/165) vllm_gemm: on gfx1201 a single-row linear stays on vLLM's own GEMM (fixes the c1 regression)
- [#166](https://github.com/vivienfanghuagood/hk-dsl/pull/166) hk.ops norms: weight_offset= (Gemma-style weights, the one added in fp32 in the kernel)
- [#167](https://github.com/vivienfanghuagood/hk-dsl/pull/167) paged_prefill vt_pages: TRITON_ATTN's V pages transposed once a call (prefill route, step 2)
- [#168](https://github.com/vivienfanghuagood/hk-dsl/pull/168) S11 (D1): RDNA3 Tile.as_operand(fp16, k_order="even_odd"); attn_k3 drops PairPKernel
- [#169](https://github.com/vivienfanghuagood/hk-dsl/pull/169) W4A16's symmetric form (GPTQ uint4b8): hk.load_w4(zero=, s_blocked=); vllm_gptq from one token
- [#170](https://github.com/vivienfanghuagood/hk-dsl/pull/170) S9a (G2): one register-view op, hk.lane_view, behind as_fragment / as_scales
- [#172](https://github.com/vivienfanghuagood/hk-dsl/pull/172) RDNA: int8 into half / bf16 register tiles (i8 tile cast; int8 global -> 16-bit tile loads)
- [#173](https://github.com/vivienfanghuagood/hk-dsl/pull/173) docs/toolchains.md: the toolchain compatibility matrix (clang 20-24 x GPU, known issues for 0.8.0 and main)
- [#174](https://github.com/vivienfanghuagood/hk-dsl/pull/174) CI: a pull request moves a submodule's pin only on purpose ("HipKittens bump") and only forward
- [#175](https://github.com/vivienfanghuagood/hk-dsl/pull/175) SGLang: GemmaRMSNorm on hk's weighted norm kernels (HK_SGLANG_NORM, on by default on gfx1100)
- [#176](https://github.com/vivienfanghuagood/hk-dsl/pull/176) Fix: vllm_gptq binds vLLM's gptq_gemm_rdna3 by name, refuses per layer (main patched nothing on the MiniCPM5 image)
- [#177](https://github.com/vivienfanghuagood/hk-dsl/pull/177) vllm_gemm: decide each layer at load; one the table never routes keeps vLLM's GEMM unwrapped
- [#179](https://github.com/vivienfanghuagood/hk-dsl/pull/179) RDNA: k(...) under a non-default torch stream warns once (HK_STRICT_STREAM=1 raises)
- [#180](https://github.com/vivienfanghuagood/hk-dsl/pull/180) README: point to the generated route table instead of a hand-written on-by-default list (M1)
- [#181](https://github.com/vivienfanghuagood/hk-dsl/pull/181) Examples and --status say what they measured: example 15 over each byte once, pinned examples SKIPPED, MFMA gate only on CDNA (M1)
- [#182](https://github.com/vivienfanghuagood/hk-dsl/pull/182) docs: getting started in three pages (install, quickstart with W7900D/MI355X tracks, examples); nav in eight sections (M1)
- [#183](https://github.com/vivienfanghuagood/hk-dsl/pull/183) docs: an archive section (measurement record + superseded design under docs/archive/, banners); user pages stop citing history (M1)
- [#184](https://github.com/vivienfanghuagood/hk-dsl/pull/184) docs: engine pages -- integration.md for every engine, vllm.md and sglang.md per engine; correctness evidence, not speedups (M1)
- [#185](https://github.com/vivienfanghuagood/hk-dsl/pull/185) docs: the op catalog -- every hk.ops kernel family, generated from python/hk/ops/catalog.py and held to the code (M1)
- [#186](https://github.com/vivienfanghuagood/hk-dsl/pull/186) docs: concepts -- how a kernel runs, tiles/strips/LDS, the correctness gates, XCDs and hk.comm (M1)
- [#188](https://github.com/vivienfanghuagood/hk-dsl/pull/188) VOPD gate: rebuild gfx11.0/11.5 kernels with the VOPD interlock hazard (llvm#220348) without dual issue
- [#189](https://github.com/vivienfanghuagood/hk-dsl/pull/189) Routes tuner: context timing on RDNA (each call among its model's other layers), TunableOp in the records; 12 gfx1100 rows re-measured
- [#190](https://github.com/vivienfanghuagood/hk-dsl/pull/190) vllm_prefill: hybrid models' pages (any multiple of 16), vLLM's per-layer KV layouts, and a signature-agnostic ROCM_ATTN stand-in (prefill route, step 3)
- [#191](https://github.com/vivienfanghuagood/hk-dsl/pull/191) AWQ MoE at decode on RDNA3: hk.load_w4(zp=, rows=True), moe_gemm_w4, and the vLLM route vllm_moe_wna16 (opt-in)
- [#192](https://github.com/vivienfanghuagood/hk-dsl/pull/192) hk.ops.paged: refuse a decode whose tensors are not the shapes its kernels read
- [#193](https://github.com/vivienfanghuagood/hk-dsl/pull/193) compiler: run MLIR passes on the calling thread (a refusal inside the pipeline could segfault)
- [#194](https://github.com/vivienfanghuagood/hk-dsl/pull/194) gfx1201: shipped vLLM linear routes for MiniCPM5-2B and Qwen3.8-27B (TP2), graph-timed (on #189)
- [#195](https://github.com/vivienfanghuagood/hk-dsl/pull/195) Routes records: a stale local record no longer shadows the shipped row (timing, TunableOp); test_route_tune without microsecond asserts
- [#196](https://github.com/vivienfanghuagood/hk-dsl/pull/196) vllm_prefill on by default on gfx1100 (prefill route, step 4)
- [#197](https://github.com/vivienfanghuagood/hk-dsl/pull/197) Engine contracts: recorded per production image and claimed version, applied per route, and a support matrix
- [#199](https://github.com/vivienfanghuagood/hk-dsl/pull/199) docs: the default-on rule (a clear kernel win, the correctness suite, one e2e smoke per engine) replaces the multi-model gates
- [#201](https://github.com/vivienfanghuagood/hk-dsl/pull/201) docs + test: use a kernel as a PyTorch op (hk.torch_op) -- eager, torch.compile, CUDA graphs
- [#203](https://github.com/vivienfanghuagood/hk-dsl/pull/203) CI: the install matrix -- fresh stock-ROCm containers install the wheel, run the README quick start and one GPU test (gfx950/gfx1201/gfx1100 x ROCm 7.2/10.1)
- [#204](https://github.com/vivienfanghuagood/hk-dsl/pull/204) VOPD gate: one line per process at default verbosity; each kernel's VOPDs at INFO (hk.compile logger)
- [#205](https://github.com/vivienfanghuagood/hk-dsl/pull/205) hk.check: time cold by default (inputs rotated past the last-level cache, so GB/s is DRAM traffic); --hot for the old way
- [#206](https://github.com/vivienfanghuagood/hk-dsl/pull/206) tools/hk-bench: a public GEMM/MoE kernel benchmark, with gfx1100 and gfx1201 results
- [#209](https://github.com/vivienfanghuagood/hk-dsl/pull/209) Issue templates (bug, install, performance) and a release-candidate checklist (docs/releasing.md)
- [#210](https://github.com/vivienfanghuagood/hk-dsl/pull/210) hk.gemv_w4: 4-bit GEMV for a few tokens without WMMA; w4a16's small-batch form (gfx1100)
- [#211](https://github.com/vivienfanghuagood/hk-dsl/pull/211) docs: the API stability policy (stable / experimental / not covered), marked on every API reference page
- [#216](https://github.com/vivienfanghuagood/hk-dsl/pull/216) vllm_prefill on gfx1201 (opt-in): the arch gate takes RDNA4
- [#219](https://github.com/vivienfanghuagood/hk-dsl/pull/219) CONTRIBUTING: a PR that adds an op or a route carries its op-catalog row (test_ops_catalog says how)
- [#220](https://github.com/vivienfanghuagood/hk-dsl/pull/220) Decode attention on TRITON_ATTN's cache: hk.ops.paged_decode (head_dim 256 on by default), and a public paged-attention bench
- [#221](https://github.com/vivienfanghuagood/hk-dsl/pull/221) GPU tier: host's own video/render groups; fail when the container sees no GPU or nearly every test skips
- [#222](https://github.com/vivienfanghuagood/hk-dsl/pull/222) gfx950: refuse LDS windows that are not whole subtiles of the tile's layout, whichever way the layout was chosen (#217)
- [#223](https://github.com/vivienfanghuagood/hk-dsl/pull/223) RDNA: the CDNA forms of P.V -- as_operand and col-layout LDS operand reads (#212, #213)
- [#224](https://github.com/vivienfanghuagood/hk-dsl/pull/224) Fix: vllm_gemm writes nothing while torch.compile traces (vLLM with CUDA graphs failed to start since #177)
- [#231](https://github.com/vivienfanghuagood/hk-dsl/pull/231) compile cache: a damaged binary behind an OK marker is a miss; flush before the marker (#227)
- [#233](https://github.com/vivienfanghuagood/hk-dsl/pull/233) vllm_prefill: leave batches hk is not faster on to vLLM's kernel (TRITON_ATTN guard)
- [#234](https://github.com/vivienfanghuagood/hk-dsl/pull/234) tests: the SGLang-Diffusion routes' kernels at Qwen-Image 2.1's shapes (poisoned repeats, side-stream producer, fenced bounds)
- [#235](https://github.com/vivienfanghuagood/hk-dsl/pull/235) hk.ops.paged_decode_portable: one decode-attention source on gfx1100, gfx1201 and gfx950
- [#237](https://github.com/vivienfanghuagood/hk-dsl/pull/237) tests: a8w4 MoE route test against an fp64 MoE, not AITER's fused_moe (nondeterministically wrong)
- [#238](https://github.com/vivienfanghuagood/hk-dsl/pull/238) vllm_prefill guard: exact batch sizes from TRITON_ATTN's metadata builder (no sync)
- [#239](https://github.com/vivienfanghuagood/hk-dsl/pull/239) paged_decode: the opt-in head_dim-128 decode takes a split per WGP, up to 4 (request, kv head) pairs
- [#240](https://github.com/vivienfanghuagood/hk-dsl/pull/240) CI gfx1201: the container wrapper is scripts/ci/gfx1201/run_in_container
- [#241](https://github.com/vivienfanghuagood/hk-dsl/pull/241) CI gfx950: skip a card any process holds (KFD) and a reserved card; HK_CI_GPUS is a preference list

## 0.8.0

hk-dsl 0.8.0: workgroup placement as a compiler pass on gfx950; `hk.map_cpp`,
i32 strips, and the strip ops and element exchange a megakernel's roles need; gfx1201 (RDNA4) on
hardware; and reproducible wheels, one `manylinux_2_35` wheel per interpreter.
hk stays on by default in vLLM and SGLang for the routes validated on your GPU:
`HK_DISABLE=1` turns every route off, `HK_<ROUTE>=0` one of them
([integration](docs/integration.md#on-by-default-and-turning-it-off)).

On PyPI: `pip install hk-dsl`, for CPython 3.10 and 3.12 on Linux x86_64 with
glibc 2.35 or newer (Ubuntu 22.04 and later, and vLLM's ROCm images): one
`manylinux_2_35` wheel per interpreter. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed. The API reference marks what 0.8.0 adds *New after
0.7.0*.

    pip install hk_dsl-0.8.0-cp312-cp312-manylinux_2_35_x86_64.whl

**Workgroup placement as a pass** (#63, gfx950; design in
[docs/design/placement-pass.md](docs/design/placement-pass.md), #62).
Which XCD runs which workgroup, as one mechanism instead of hand-written
remaps:
- `hk.kernel(placement=None | "auto" | hk.Placement)`. `hk.Placement.cluster()`
  runs one sharing cluster an XCD at a time, its key found by the analysis
  (`hk.placement.analyze`: each global read's sharing key over the grid axes)
  or named with `hk.Digit`s; `hk.Placement.chunked(chunk, tail)` renumbers
  `block_idx.x` alone. `"auto"` launches the cluster placement where
  `hk.placement.choose` takes it: not for order-dependent or resident
  kernels, from five KV streams an XCD, and not for non-uniform work under
  twelve rounds an XCD (thresholds measured in #55). `Kernel.placed()` and
  `Kernel.placement_choice()` give the placed kernel and the decision.
- Outputs are bitwise the unplaced kernel's. `attn_cdna`'s `mapping="spatial"`
  (#55) is now `hk.Placement.cluster()`, and `gemm_cdna` / `gemm_cdna_dp`
  take `Placement.chunked` instead of their inline remaps: 11 GEMM shapes
  bitwise, new/old 0.997-1.007; 15 attention points bitwise, 0.993-1.003 of
  #55's hand-written numbering.
- `hk.arch.xcd_count()`: KFD's `num_xcc` (MI350P / DPX: 4, MI300A: 6), the
  arch's default otherwise, `HK_XCDS` to override. With 1 (gfx1100) every
  placement is the identity.
- `python -m hk.check FILE.py:NAME --placement --inputs ...` checks that a
  launch's placement covers its grid exactly once, evaluating the decode the
  kernel's own emitted ops compute ([check](docs/check.md#which-xcd-runs-which-workgroup---placement)).
- **M2** (#65): K2 (`mxfp8`), K1 (`moe_a8w4`) and the stream-K plan take it too, so hk.ops has no
  hand-written XCD remap left (and no `NUM_XCDS`); persistent kernels use
  `hk.placement.renumber` / `hk.placement.xcd_range`. K2 at the 18 DSV4.1 points bitwise,
  0.992-1.011; K1 bitwise, and **1.19-1.24x faster** at M where its old remap fell back to the
  hardware's order (stage-1 grid not a multiple of 8: M 9500, 10000, 14000, 15001), within 0.1%
  where it applied. `HK_XCDS=1` turns every placement off.
- A user guide, [docs/placement.md](docs/placement.md) (when to use it, examples, the measured
  tables, limits, turning it off); the design document rewritten as built; an
  [hk.placement](docs/api/hk.placement.md) API page. `hk.Digit.of`'s radix may be spelled
  `"tensor.dim"` (`"k.depth"`).

**`hk.map_cpp` and i32 strips** (#53, gfx950;
[the language](docs/language.md#integer-mask-and-bit-expressions-on-strips-hkmap)).
- `hk.map_cpp(expr, *args, dtype=None)` evaluates one C++ expression for each
  element of its strip arguments (`x0`, `x1`, ... bound to the elements, or
  to an `hk.Int` / Python int), into an fp32 or i32 strip. It is for what a
  kernel must reproduce bit for bit and the operators cannot say: bit
  manipulation, a fused multiply-add, one particular instruction. The
  expression compiles with floating-point contraction off (`a * b + c` is two
  roundings; an fma is `__builtin_fmaf`). Strips of fp32 and i32 together
  must say `dtype=`; wave32 targets refuse it.
- Strips may hold i32: they load, store, convert with `.to` and are
  `hk.map_cpp`'s arguments and results. Operators and reductions on them are
  refused, so integer arithmetic is written in `hk.map_cpp`.

**Strip ops for megakernel roles** (#72, gfx950), which DeepSeek-V4.1's K1
written in hk uses (bitwise to the hand-written K1 over ~496k layer calls a
rank):
- `hk.iota(ty)`: an i32 strip whose element (r, c) is c.
- `Global.load_elems(idx)` / `store_elems(val, idx)`: element gather and
  scatter at flat indices; an index below zero or past the tensor reads 0 and
  writes nothing.
- `Tile.group_sum` / `group_max` / `group_min(cols, lanes=, xor=)`:
  reductions over groups of lanes, in a stated order like
  `sum(axis=1, lanes=, xor=)`.
- `Tile.lane_gather(lanes)` (`ds_bpermute` of each element), and
  `hk.mfma_strip("f32_16x16x4", a, b, acc)`, one raw MFMA on strip registers
  for reproducing another kernel's MFMA bit for bit.

**Element exchange for megakernel hand-offs** (#73, gfx950), also from K1 in
hk:
- `Global.post_elems(val, idx, tag, runs=, bf16=)` / `take_elems(idx, tag,
  runs=, bf16=, timeout_s=)`: tagged mailbox pairs, each value and its tag in
  one 8-byte scoped store, as `hk.comm.post` / `take` per element. `runs=True`
  moves a lane's run as 16-byte pairs of pairs; `bf16=True` packs two bf16 a
  pair (values must already be bf16; refused without `runs`). A wave with no
  live pair polls nothing, and a timeout traps.
- `SharedTile.load_elems` / `store_elems`: an fp32 LDS tile addressed element
  by element, with `ds` operations (not FLAT, which would also count in vmcnt).
- Scoped `load_elems` / `store_elems`; `store_elems(runs=True)`, one store a
  run, fp32 to e4m3 packed four at a time; scoped register-tile loads take
  `bounded=n`.
- `hk.uniform(cond)` / `hk.if_(..., uniform=True)`: a branch declared
  workgroup-uniform may hold barriers, which is a persistent kernel's role
  dispatch; a divergent enclosing branch is still refused.

**gfx1201 (RDNA4) on hardware** (#70, #71). gfx1201 goes from compile-only to
hk's GPU tier passing on a Radeon AI PRO R9700 (914 passed, 0 failed;
[chapter 47](docs/history/47-gfx1201-on-hardware.md)):
- HipKittens moves to `9d182f93`: `include/rdna4` with gfx12's accumulator
  layout as measured (every MMA unit test had failed on the old one) and
  rdna3's work since the fork re-derived for gfx12. gfx1100 and gfx950 are
  unaffected: `include/rdna3` and the CDNA trees are byte-identical.
- The LDS reads a tile load issues come from the target: one on gfx1201, where
  hk assumed gfx1100's two. hk-lds-pipeline derives every LDS wait from that
  count, and a count too high makes every wait too shallow.
- The wait gate reads gfx12's waits (hex immediates, `s_wait_loadcnt_dscnt`,
  `s_wait_storecnt_dscnt`), the target is measured (VGPR granule 24, LDS pool
  128 KB, i8 WMMA, strips), and the schedules' register model is fitted on
  gfx1201's own builds.
- Its ceilings are measured on the R9700 (#74): bf16 / f16 WMMA 197 TFLOP/s,
  fp8 372, iu8 374 TOP/s; an add at 1.81 TB/s inside the 64 MB Infinity Cache,
  0.62 TB/s past it. `hk.check`, `hk.profile`, `hk.tune` and the scoreboard
  use them (they fell back to gfx1100's, which would have called hk's own 4096³
  GEMM faster than the card can), and `hk.tune` has a GEMM space there.
- The scoreboard on gfx1201, out of the box (gfx1100's plans), every arm in one
  process, with FlyDSL and AITER arms (#74,
  [chapter 48](docs/history/48-gfx1201-ceilings-and-first-scoreboard.md),
  [performance](docs/archive/performance-2026-10.md#part-c-rdna4-gfx1201-one-radeon-ai-pro-r9700)):
  hk's GEMM is the fastest at the five shapes (1.03-1.07x the best other arm).
  It loses attention to AITER's (0.83x causal, 0.97x full, at 4096 tokens,
  GQA 4) and fused add + rms_norm at 8192 rows (0.93x).
- No engine route is validated on gfx1201, so none is on by default there, and
  nothing is tuned there yet (its tables are gfx1100's).

**K1 (MoE a8w4) on gfx950, measured for this release** on the release's code
(after #85): real DeepSeek-V4.1-Flash weights, two layers, real, gate-only and
uniform routing, weights rotated past the caches, interleaved rounds in one
process; hk deterministic and within AITER's own error against an exact fp32
MoE everywhere.

| M | hk / AITER (tuned FlyDSL) | K1 now / K1 at #65 | K1 now / K1 before #65 |
|---|---|---|---|
| 64 | 0.77-0.84x | 1.26-1.45x | 1.12-1.20x |
| 128-512 | 0.74-0.85x | 1.05-1.30x | 1.04-1.31x |
| 2048 | 0.97-0.99x | 1.08-1.15x | |
| 4096 | 1.13-1.20x | 1.10-1.12x | |
| 8192 (routed) | 1.27-1.35x | 1.09-1.12x | 1.08-1.12x |
| 16384 (routed) | 1.16-1.23x | 1.04-1.06x | |

(The "K1 at #65" figures at M 2048, 4096 and 16384 compare each commit in its
own process, each normalised by its own in-process AITER; the other columns are
one process. The two methods agree within about 3%.)

- vLLM's K1 route takes M >= 8192, where hk wins at 12 of 12 points (before
  #85: 1.11-1.23x). Below that the route leaves the MoE to AITER, so decode in
  production is AITER's either way.
- At decode M (64-512) hk's K1 is still slower than AITER's tuned FlyDSL
  kernels, 0.74-0.85x, up from 0.53-0.78x at #65.

**Build gates**
- **The spill gate passes a scratch frame the code never touches** (#68). A
  kernel with ScratchSize > 0 and no VGPR spill builds when its code object has
  no `scratch_*` instruction, no `flat_*` memory instruction, no flat scratch
  and no private segment buffer (`cocheck.unused_scratch_frame`); the build
  records `scratch_note = "reserved-unused scratch frame"` and says so on
  stderr and in `build.log`. Found on K1's indexer build under ROCm 10 (36 B
  reserved, never used). A VGPR spill, any scratch access, and a code object
  that cannot be read are still refused.

**Fixes**
- **K1 MoE a8w4 at decode M ran slower after #65** (#85; M 64: 0.82-0.89x of
  K1 before #65, found by the MI355X scoreboard). The sorted kernels' grids are
  sized for the most sorted blocks a launch can have, and #65's
  `Placement.chunked(0, "spread")` dealt contiguous ranges of the whole grid, so
  the live blocks (235 of 390 at M 64) landed on the first five XCDs. They now
  renumber in the body over the live blocks only
  (`hk.placement.renumber(..., live=)`, new). Outputs are bitwise the same, and
  every M got faster (numbers in *K1 on gfx950* below).
- **gfx950: fp32 register tiles through LDS were FLAT loads and stores**
  (#59). `store(st<float>, rt<float>)` / `load(rt<float>, st<float>)` addressed
  the tile through a generic pointer, so every access was FLAT: slower to
  issue than `ds_read` / `ds_write`, and counted in vmcnt as well as lgkmcnt,
  so each wait for an LDS value also waited for every global load in flight.
  They are `ds_read` / `ds_write` now. Results are unchanged; affected are the
  cross-warp combines of fp32 partials (`hk.ops.decode_gemm`, `w4a16`, `moe`)
  and user kernels that move fp32 tiles through LDS on gfx950.

**Wheels** (#66)
- cp312 is one `manylinux_2_35` wheel, built on glibc 2.35 in vLLM's ROCm
  image, like cp310. Through 0.6.0 cp312 was built on Ubuntu 24.04 and needed
  glibc 2.38, so `pip install hk-dsl` found no cp312 wheel in vLLM's ROCm
  images; 0.7.0 added a second, `manylinux_2_35` cp312 wheel. (0.7.0's PyPI
  description still says CPython 3.12 needs glibc 2.38: that page cannot be
  edited after upload, and 0.8.0's states the wheels above.)
- `scripts/check_wheel_glibc.py` fails a wheel that needs a newer glibc (or
  GLIBCXX / CXXABI, with auditwheel) than its manylinux tag; CI's IR leg
  builds the cp310 wheel and runs it.
- Reproducible wheels: no build path in the binaries (`-ffile-prefix-map`,
  RUNPATH `$ORIGIN` at link time), entry times from `SOURCE_DATE_EPOCH`
  (the last commit's), fixed modes; the same commit builds byte-identical
  wheels ([building](docs/building.md#reproducible-wheels)).

**Docs**
- The API reference marks names against the 0.7.0 wheel (it marked them
  against 0.6.0, so `placement=` and `hk.arch.xcd_count` read as if 0.7.0 had
  them), and documents `hk.Placement`, `hk.Digit` and
  `hk.runtime.SymmetricBuffer`.
- [The language](docs/language.md) gains `hk.comm`, workgroup placement, the
  stated strip-reduction order (#52, #57), the bounded group copies on gfx950
  (#49) and LDS allocated in a runtime branch (#56).
- The docs site builds with `mkdocs build --strict` in CI: every page is in
  the nav (hk.profile, schedules, hk.tune and the end-to-end loop were not),
  heading ids follow GitHub's (so `#which-xcd-runs-which-workgroup---placement`
  resolves on both), and the links #54 broke (`#turning-them-on`) point at
  `#on-by-default-and-turning-it-off`.
- The install pages state 0.7.0's three wheels and that CPython 3.12 needs
  glibc 2.35, not 2.38; [the runtime](docs/runtime.md#which-gpu) has the arch
  detection order 0.7.0 introduced.
- [Releasing](docs/releasing.md): the release steps, and the documentation
  sync that must pass before a tag.
- Examples [14](examples/README.md) (an hk.comm task graph, gfx950) and
  [15](examples/README.md) (workgroup placement).
- `hk.map_cpp` and i32 strips in the language and the API reference (#81).
- No internal address, host name or user path in the repository (#83): docs,
  bench records and scripts use placeholders and environment variables, and
  `scripts/check_public_hygiene.py` (generic rules, an explicit allowlist)
  runs in CI's IR leg and before every tag.
- [docs/archive/design/e2e-goal.md](docs/archive/design/e2e-goal.md), the end-to-end goal
  refocused on three core kernels (#20, #22).

**CI and tests**
- Several gfx950 runners in parallel, each with its own GPUs and state, and
  GPU locks shared with manual measurements (#60,
  [testing](docs/testing.md#the-gpu-tier)).
- A cancelled GPU job no longer leaves its container on the GPU (#69,
  [testing](docs/testing.md#the-gpu-tier)).
- Tests of gfx1100's measured data are pinned to gfx1100
  (`@pytest.mark.arch`, #64).
- The fenced paged-attention checks (`test_fenced.py`) build their reference
  on the host from one readback of each tensor, read twice and compared, and a
  relative-error failure prints both maxima. The one intermittent paged failure
  seen on gfx950, `relative error 9.73e+29`, matches an all-zero reference
  (0.97265625 / 1e-30), not a wrong output: the paged decode did not fail in
  thousands of repetitions with poisoned allocations and polluted LDS (#89;
  tests only, no kernel change).

**Compatibility**
- No public DSL name or parameter was removed or renamed since 0.7.0 (checked
  name by name against the 0.7.0 wheel). The modules' `NUM_XCDS` constants in
  `hk.ops.attn_cdna`, `gemm_cdna`, `gemm_cdna_dp` and `moe_a8w4` are gone
  (#65): the XCD count is `hk.arch.xcd_count()`.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-6`. 0.7.0's cached builds are rebuilt once.
- The spill gate can build a kernel that 0.7.0 refused: one whose only
  scratch is a reserved frame its code never touches (#68).
- gfx950's attention, GEMMs, K1 and K2 now number their workgroups through
  the placement pass: outputs bitwise as before. `HK_XCDS=1` turns every
  placement off.
- cp312 has one wheel, `manylinux_2_35`; 0.7.0's cp312 `manylinux_2_38` wheel
  has no 0.8.0 counterpart, and pip on a newer glibc takes the 2_35 one.

**Known issues**
- A GPU-tier CI run on MI355X (gfx950) once reported a mismatch in the
  fenced-allocator test of the paged-decode split path
  (`tests/hk/gpu/test_fenced.py::test_paged_decode_and_prefill`, fused decode,
  head_dim 64, 32 splits). It did not recur in later runs or in several
  thousand targeted repetitions with poisoned allocations and polluted LDS; the
  reported error value matches an all-zero host-side reference rather than a
  wrong kernel output. The paged decode is used only by the opt-in vLLM
  attention backend (`HK_VLLM_BACKEND=1`, off by default) and is not reached by
  any default-on route. The test has since been hardened (#89, tests only): it
  builds its reference on the host from one stable readback of each tensor, so
  a recurrence names its cause.
- gfx942: the synchronous group copies are still unbounded (#49): on gfx942 a
  `g.load` / `g.store` at a partial last tile reads and writes past the
  tensor. There is no gfx942 here to test the bounded form on.
- The gfx1100 defaults were not run end to end. Each gfx1100 route's evidence
  is its earlier opt-in run with real models; the default-on configuration
  itself was checked by unit tests, the GPU tier and the engine contract.
- The vLLM decoder attention backend stays off by default: it replaces ROCm's
  default decoder attention for every model and raises rather than falls back
  on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and other head
  sizes; validated on Qwen3-8B only, and on gfx950 slower at one request on
  long contexts (0.71-0.88x).
- SGLang-Diffusion installs made before 0.7.0 stay opt-in until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  spills and the spill gate refuses it (rocm/pytorch 7.2.3 builds it). There,
  `hk.ops.gemm.matmul`'s gfx950 route picks it for N 28672 / 14336, K 4096 at
  M 192-512 and raises `ResourceError` instead of falling back. vLLM's linear
  route (`HK_VLLM_GEMM`) is off on gfx950, so no default route reaches it.
- gfx1201 runs hk's GPU tier on a dev box, not in CI's required tiers; no
  route is validated or on by default there and nothing is tuned. gfx942 is
  compile-only.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

## 0.7.0

hk-dsl 0.7.0: hk is on by default in vLLM and SGLang, for the routes validated
with real weights on your GPU; `HK_DISABLE=1` turns it off. Also `hk.comm` for
hand-offs between gfx950's XCDs and between GPUs, and fixes for two silent
errors that 0.6.0's DSL has in user kernels: on gfx950 a synchronous `g.store`
at a partial last tile overwrites the memory after the tensor, and LDS
allocated inside a runtime branch reads back as 0. No shipped op or route was
affected by either.

On PyPI: `pip install hk-dsl` (CPython 3.10 or 3.12 on glibc 2.35+, Linux
x86_64). Three wheels: cp310 `manylinux_2_35`, and cp312 both
`manylinux_2_35` and `manylinux_2_38`; pip picks the newest tag the system's
glibc takes. The cp312 `manylinux_2_35` wheel was added on release day, for
glibc 2.35 systems such as vLLM's ROCm images (Ubuntu 22.04), where the
`manylinux_2_38` one does not install. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed (and a ROCm build of torch to launch).

    pip install hk_dsl-0.7.0-cp312-cp312-manylinux_2_35_x86_64.whl

(This paragraph first said "3.12 on glibc 2.38+", as do 0.7.0's PyPI page and
the GitHub release notes; the `manylinux_2_35` cp312 wheel makes 2.35 enough.)

**hk is on by default** (#54). With hk-dsl installed, its vLLM and SGLang
plugins (and the SGLang-Diffusion boot) apply every route that was validated
with real weights on your GPU: outputs bit-identical to the engine's, or error
within the engine's own noise. Before, every route needed its own `HK_*=1`.
Turning hk off is now for comparison experiments:

```bash
HK_DISABLE=1 vllm serve <model>        # stock: nothing patched, in every engine
HK_VLLM_MXFP8=0 vllm serve <model>     # one route off (HK_<ROUTE>=0)
python3 -m hk.integration --status     # every route on or off, and which rule decided
```

- **On by default** (the evidence for each route is in
  [integration](docs/integration.md#on-by-default-and-turning-it-off)):

  | GPU | engine | routes (switch) |
  |---|---|---|
  | gfx950 (MI355X) | vLLM | K2 MXFP8 linear for DeepSeek-V4.1 at M >= 2048 (`HK_VLLM_MXFP8`); K1 MoE a8w4 for DeepSeek-V4.1 at M >= 8192 (`HK_VLLM_MOE_A8W4`); AWQ 4-bit (`HK_VLLM_INT4`) |
  | gfx1100 (W7900D) | vLLM | unquantized linear (`HK_VLLM_GEMM`); bf16 MoE (`HK_VLLM_MOE`); AWQ 4-bit (`HK_VLLM_INT4`); the encoder SDPA (`HK_VLLM_SDPA`) |
  | gfx1100 | SGLang | linear (`HK_SGLANG_LINEAR`); MoE and the gfx1100 top-k fix (`HK_SGLANG_MOE`) |
  | gfx1100 | SGLang-Diffusion | DiT attention (`HK_SGLD_ATTN`); DiT linear (`HK_SGLD_LINEAR`) |
  | gfx1100 | `hk.ops.sdpa` | K3 DiT attention (`HK_SDPA_K3`) |

  On a GPU a route was not validated on, it stays off: the gfx1100 routes on
  gfx950 (vLLM's linear, MoE and encoder SDPA among them), and every route on
  gfx942 and gfx1201.
- **Stays opt-in:**
  - **ConvRot INT8** (`HK_SGLD_CONVROT=1`, or `HK_SGLD=attn,linear,convrot`):
    it is lossy by design (PSNR 30.4 dB from bf16), a model-level choice and
    not a toolchain default.
  - **The vLLM decoder attention backend** (`HK_VLLM_BACKEND=1`): see the
    known issues.
  - **`HK_MXFP8_ANY`**, K2 at shapes it did not measure (no real-weight run).
- **Turning routes off.** The first rule that applies decides:
  1. `HK_DISABLE=1` turns every route off, in every engine. Nothing beats it.
  2. An engine's switch at `0` (`HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0`,
     `HK_SGLD=0`) turns off that engine's routes.
  3. A route's own switch decides that route: `HK_<ROUTE>=0` off, `=1` on (also
     a default-off route, or a GPU it was not validated on).
  4. The legacy switches still mean what they meant: `HK_SGLD=attn,linear,convrot`
     turns on exactly those parts, `HK_SGLD=1` attention and linear,
     `HK_PATCH_VLLM=1` also the encoder SDPA; `HK_SGLANG_MOE=topk` and
     `HK_VLLM_BACKEND=<slot>` work as before.
  5. Otherwise the default applies.

  `0`, `off`, `no` and `false` are off; an empty value counts as unset.
  `python3 -m hk.integration --status` prints every route, whether it is on,
  and which rule decided.
- **The engine contract is checked at startup** (`hk.integration.contract`).
  Before it patches anything, a plugin checks every route it is about to apply
  against the installed vLLM / AITER / SGLang: each patched or fallback entry
  point exists, with the parameters the route uses. If any check fails, for
  example after a version change, the plugin patches **nothing** in that
  engine, prints one warning line, and the engine runs fully stock:

  ```
  [hk] WARNING: vllm: engine contract FAILED -- hk patched NOTHING in this engine, it runs stock (...)
  ```

  It is all or nothing per engine, never half-patched: a route that raises
  while it is applied undoes the ones applied before it, with the same line.
- **The first JIT compile says so.** A route's kernels compile on first use,
  seconds each, in every worker. If they were not pre-built, the first
  compile in each process prints one line that suggests
  `python3 -m hk.integration warm-routes` (and `--warm` for attention), run
  with the server's `HK_CACHE_DIR` before the server starts.
- **GPU detection reads the KFD topology.** The vLLM ROCm image sets
  `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`. Without `HK_ARCH`, hk took the
  list's first entry, so it read an MI355X as gfx90a, built for gfx1100, and
  with default-on it would have left every gfx950 route off.
  `hk.runtime.arch.detect()` now goes `HK_ARCH`, then a variable naming exactly
  one arch, then the KFD topology in sysfs (no HIP context), then torch and
  rocminfo, and uses a list's first entry only as the last resort.
- **Comparison scripts must say stock.** Unset no longer means stock, so an
  A/B's stock arm sets `HK_DISABLE=1`. `python3 -m hk.integration ab` /
  `headroom` and `tools/hk-bench/` do this themselves.
- **SGLang-Diffusion:** an install made before 0.7.0 keeps its old boot, which
  needs `HK_SGLD=1`. Run `python3 -m hk.integration sglang-diffusion install`
  again for the defaults; `... sglang-diffusion status` says when the boot is
  the old one.
- **Fixed:** with several routes on in one vLLM (int4 and MXFP8 on gfx950), its
  compile cache root was nested again in every worker
  (`hk-int4/hk-mxfp8/hk-int4/...`). It is now one directory named for all the
  routes (`hk-int4+mxfp8`), the same in every process.
- `hk.ops.sdpa.reset_stats()` also resets K3's counters.

**Fixes**
- **gfx950: synchronous `g.load` / `g.store` went out of bounds at a partial
  last tile** (#49). This affects 0.6.0's DSL. In 0.6.0 on gfx950, in a
  user-written kernel whose synchronous `g.load` covers a partial last tile
  (the tile's rows run past the tensor's last row):
  - the load reads up to a tile's rows past the tensor: an intermittent memory
    fault (10 of 10 fenced cases faulted, at M = 1, 63, 65, 130 and 1000);
  - its `g.store` **silently overwrites the memory after the tensor**: with
    slack after the output, a probe found 63 of 63 rows past it overwritten,
    with no error.

  No shipped `hk.ops` op and no vLLM / SGLang route was affected, in 0.6.0 or
  in this release: all run clean under an end-fenced allocator on gfx950 and
  gfx1100, and no shipped kernel issues a `g.store`. The fix: on CDNA, the
  synchronous `g.load` stops at the tensor's last row (rows past it read zero),
  and `g.store` drops the rows past it (`group_load_sync`, `group_store_bounded`
  in `include/hk/cdna.cuh`). `hk.ops.mxfp8.gemm`, the one shipped CDNA kernel
  with a synchronous group copy, is bit-identical and within 0.6% of its speed.
  gfx942 still uses the unbounded copies (no hardware to test). RDNA3 already
  bounded them. `tests/hk/gpu/test_fenced.py` now runs every `hk.ops` family at
  ragged shapes under the end-fenced allocator, in CI on gfx950 and gfx1100.
- **LDS allocated inside a runtime branch was not part of the launch's LDS**
  (#56). The `hk-lds-alloc` pass sized a kernel's launch from the
  allocations in the kernel's own block only, while the generated C++ bumps
  its allocator along the path the kernel takes. So a tile that a kernel
  allocated with `hk.lds.alloc` inside a runtime `if` (a branch on a device
  value) lay past the end of the LDS the launch requested: its reads returned
  0 and its writes were dropped, with no error. An allocation inside a runtime
  loop was past the end the same way. Found porting DeepSeek-V4.1's K1 to hk,
  where a reduction tile allocated in a role branch read back as zeros (87.5%
  of q wrong). Every released version through 0.6.0 has the pass in that form,
  so a user kernel that allocates LDS in a runtime branch or loop is affected
  there. No shipped kernel is: a scan of every kernel the IR and codegen tiers
  translate (both arches) and the gfx950 GPU suite builds found no LDS
  allocation outside a kernel's own block, in 0.6.0 or in this release. The
  pass now walks every region: the two arms of an `if` start at the same
  offset and the larger one continues, and an allocation inside a runtime
  `for` / `while` is refused, with a message to allocate it before the loop.
  `tests/hk/gpu/test_lds_branch.py` reads such a tile back across lanes (all
  zeros before the fix, exact after).
- **`mxfp8.quant` read up to 63 rows past the activation** (#47). At
  M % 64 != 0, K2's fused quant (#41, after 0.6.0) copied whole 64-row tiles of
  x. When x ended an allocator segment, that read took a GPU memory fault in
  vLLM with `HK_VLLM_MXFP8` on: the first three real-weight DeepSeek-V4.1 runs
  with K1 and K2 on faulted, and since it depends on the allocator's layout,
  later runs of the same arm completed. It only ever affected main after
  0.6.0; the released 0.6.0 does
  not contain `mxfp8.quant`. The last block's copy now stops at row M, with the
  output unchanged: 48 of 48 ragged-M cases are bitwise equal to vLLM's quant +
  GEMM. The end-fenced allocator test (`test_fenced.py`, which #49 extended)
  came with it.

- **A stated strip-reduction order was not the documented one for products**
  (#57). With #52's `lanes=` / `xor=`, `(x * x).sum(axis=1, lanes="seq",
  xor=...)` compiled to fused multiply-adds: HIP's FMA contraction fused each
  product into the running sum across hk's inlined ops, so the sum was not the
  documented one of rounded products (4288 of 32768 lanes differed from it on
  MI355X). Only an ordered reduction of products was affected, and no `hk.ops`
  kernel uses one. #52 came after 0.6.0, so 0.6.0 is not affected. The ordered
  reduction now keeps each summand's rounding (no fma reaches in), and
  `sum(..., times=y)` is the fused form, stated.

- **Tests: `test_sdpa` compares against fp32 attention on torch's math
  backend** (#61), test-only: no kernel or library change. Its intermittent
  head_dim-64 failure on a shared MI355X was torch's reference, not hk: in the
  vLLM ROCm 10 image, torch's fused bf16 SDPA (AOTriton 0.13.50) gave wrong,
  nondeterministic tiles under same-GPU load, while hk's kernel gave identical
  bits on every call.

**New in the language: hk.comm (gfx950)** -- hand-offs between the XCDs of
one GPU and between GPUs, each lowered to what a memory-model litmus suite
(20000 rounds a combination, across XCDs and across two MI355X) found correct.
The memory model is in [docs/design/comm.md](docs/design/comm.md).
- **Scoped tensors, signal / wait, symmetric memory** (#38):
  - `hk.Tensor[dtype, "device" | "system"]`: every load, store and scalar
    atomic through it carries the scope's cache-policy bits (on gfx950 both
    sides of a hand-off need them, so the scope belongs to the tensor);
  - `hk.comm.signal` (drain, barrier, one scoped flag store) and
    `hk.comm.wait` (a scoped poll with a timeout that sets an error element,
    or traps, instead of hanging);
  - `hk.comm.arrive` / `hk.comm.barrier`: an epoch'd grid barrier;
  - `hk.kernel(..., resident=True)` and `Kernel.resident_capacity()`: a grid
    that cannot all be resident raises `ResourceError` instead of deadlocking;
  - `hk.runtime.SymmetricBuffer`: one buffer per rank with every peer's mapped;
    while one is open, a launch refuses it bound to an unscoped parameter.
- **Every scalar fetch-op, and scoped col_vecs** (#45): `View.atomic_min_scalar`,
  `atomic_or_scalar`, `atomic_and_scalar`, `atomic_xor_scalar` and
  `atomic_exch_scalar` beside `atomic_add_scalar` / `atomic_max_scalar`, each
  returning the old value, plain or at a scoped tensor's scope; a row tile's 32-bit `col_vec` loaded from or
  stored to a scoped tensor (a per-row scale next to its tile).
- **Work queues and per-XCD aggregation** (#48): `hk.comm.claim` (a work
  queue's take: one scoped atomic add, no drain) for persistent kernels and
  task graphs; `hk.arch.xcd_id()`, read from the hardware; and
  `hk.comm.xcd_barrier`, a grid barrier with one global atomic per XCD a round
  (2.53 us against 3.71 us flat at 256 workgroups). The tile-signal pattern
  through a double-buffered symmetric buffer is tested across two GPUs.
- **Tagged mailboxes** (#50): `hk.comm.post` / `hk.comm.take` send a one-int32
  message and its tag as one 8-byte scoped store and one poll, no drain: a
  1.51 us round trip between two MI355X, against 3.40 us for a tile with
  signal / wait. A mailbox orders nothing else.

**New in the language: strips**
- **A stated row-reduction order** (#52, #57, gfx950): `t.sum(axis=1,
  lanes="seq" | "tree", xor=(32, 16, 8, 4, 2, 1))`, and the same for `max`,
  `min` and `prod`, for a result that must match another kernel's bit for bit.
  The guaranteed order:
  - a lane's own values are combined left to right (`"seq"`) or as a balanced
    tree (`"tree"`), then the lanes over the given xor offsets in that order,
    each of 1, 2, 4, 8, 16 and 32 once;
  - the strip's values are summed as they are: `(x * y).sum(...)` adds the
    rounded products, and the compiler does not fuse them into the sum;
  - `x.sum(axis=1, lanes="seq", xor=..., times=y)` is the fused form, spelled
    with fma: `s = fma(x0, y0, r(x1 * y1))`, then `s = fma(xi, yi, s)`, then the
    xor sums. That is what a compiled `s += a * b` loop computes, and it is
    dsv41_mono's and aiter's sum of squares in their norms (`times=x`).

  Without the arguments nothing changes (a tree, then 1 .. 32). Refused on
  wave32 targets, on columns and on WMMA tiles.

**Kernels and serving**
- **gfx950 attention: a spatial workgroup numbering, routed by measurement**
  (#55). `attn_cdna` (dense flash attention and paged prefill) gains
  `mapping="spatial"`: each XCD takes a contiguous range of (batch, KV head)
  clusters, so the workgroups resident on an XCD read one K/V stream. Outputs
  are bitwise the default numbering's. `hk.ops.attn` and the SDPA ops pick it
  when an XCD would otherwise hold five or more KV heads' streams, except for
  causal grids with fewer than twelve rounds of workgroups an XCD. Against the
  default, MHA at D 128 and 8K-128K tokens: 1.05-1.18x at 128 KV heads, up to
  1.08x at 64 and 1.05x at 40 (at 128 heads, 32K, the L2 hit rate goes from
  0.47 to 0.96). The routed choice was never below the default at the 74 timed
  points. The paged prefill accepts the option but is not
  routed (unmeasured).
- **K2's activation quant and its scale layout in one launch** (#41):
  `hk.ops.mxfp8.quant(x, packed=True)`, and `hk.mx_quant_store` writes the
  packed scale layout. The vLLM route makes one opaque call
  (`hk_vllm::mxfp8_linear`) instead of vLLM's quant, hk's scale packing and the
  GEMM: 1.01-1.58x per call at the 12 DeepSeek-V4.1 TP2 shapes, bit-identical.
- **Serving readiness** (#43, #44): the engine-contract test, a gfx950 GPU tier
  in the vLLM deployment image (ROCm 10), and `python3 -m hk.integration
  warm-routes --tp 2`, which builds the DeepSeek-V4.1 routes' kernels before
  the engine starts (31.2 s on an empty cache, 0.9 s warm).
  [docs/archive/design/serving-readiness.md](docs/archive/design/serving-readiness.md) lists what
  0.6 lacked for a deployment; [docs/design/comm.md](docs/design/comm.md) has
  hk.comm's roadmap (#51); [docs/design/placement-pass.md](docs/design/placement-pass.md)
  designs workgroup placement as a compiler pass (#62, design only).
- CI: pushes run only on main, and a PR's superseded runs are cancelled (#42);
  GPU-tier runs write no core dumps (#46).

**Compatibility**
- **Unset now means on.** With hk-dsl installed, vLLM and SGLang (and
  SGLang-Diffusion, after `install`) run with hk's validated routes unless
  `HK_DISABLE=1` or a route's switch is `0`. Every existing `HK_*=1` still
  works.
- No public DSL name or parameter was removed or renamed since 0.6.0 (checked
  name by name against the 0.6.0 wheel). Changed signatures only gained
  parameters: the strip reductions' `lanes=` and `xor=` (and `sum`'s
  `times=`), and `hk.kernel`'s `resident=`. One integration op was
  replaced: vLLM's MXFP8 route calls `hk_vllm::mxfp8_linear`, which quantizes
  inside, instead of `hk_vllm::mxfp8_block32` (#41).
- On gfx950, a synchronous `g.load` now reads zero for a tile's rows past the
  tensor, and `g.store` no longer writes them. In-bounds rows are unchanged.
- An LDS allocation inside a runtime loop is now refused at build time (it
  was silently past the launch's LDS); allocate it before the loop. One inside
  a runtime branch now counts toward the launch's LDS, so such a kernel asks
  for more LDS than 0.6.0's did.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-5`. 0.6.0's cached builds are rebuilt once.

**Known issues**
- **gfx942: the synchronous group copies are still unbounded** (#49). There
  is no gfx942 here to test the bounded form on, so on gfx942 a `g.load` /
  `g.store` at a partial last tile still reads and writes past the tensor.
- **The gfx1100 defaults were not run end to end.** Each
  gfx1100 route's evidence is its earlier opt-in run with real models. The
  default-on configuration itself was checked by the unit tests, the gfx1100
  GPU tier and the engine contract in each engine's environment, not by a
  model run on a W7900D with nothing set. The real-weight default-on smoke ran
  on MI355X (DeepSeek-V4.1-Flash, TP2).
- **The vLLM decoder attention backend stays off by default.** It replaces
  ROCm's default decoder attention for every model, and it raises rather than
  falls back on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and
  other head sizes. It was validated on Qwen3-8B only, and on gfx950 it is
  slower at one request on long contexts (0.71-0.88x).
- **SGLang-Diffusion installs made before 0.7.0 stay opt-in** until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  at 130x1792x768 spills and is refused. rocm/pytorch 7.2.3 builds it.
- gfx1201 and gfx942 are compile-only, and no route is on by default there.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

See docs/api (marks "New after 0.6.0"), [integration](docs/integration.md),
[docs/design/comm.md](docs/design/comm.md) and
[docs/testing.md](docs/testing.md).
