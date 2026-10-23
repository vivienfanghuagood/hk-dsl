# Using hk inside vLLM and SGLang

`hk.integration` patches a running vLLM or SGLang so that some of its calls
run on generated kernels. Every patch is opt-in, reports what it took and what
it left alone, and falls back to what the engine would have run for any call
it cannot serve. The code is `python/hk/integration/`; its module docstrings
and the [development record](history/README.md) are the source for this page,
and what each patch measured is in [performance](performance.md).

- [The patches](#the-patches)
- [Turning them on](#turning-them-on)
- [The SDPA patch: vision towers](#the-sdpa-patch-vision-towers)
- [The decoder attention backend](#the-decoder-attention-backend)
- [The linear-layer patch](#the-linear-layer-patch)
  - [FlyDSL's GEMM as an optional path](#flydsls-gemm-as-an-optional-path)
- [The MoE patch](#the-moe-patch)
- [The 4-bit patch](#the-4-bit-patch)
- [The engine layer: `Op`, `Patch`, `RouteTable`](#the-engine-layer-op-patch-routetable)
- [Routing tables](#routing-tables)
- [The CUDA-graph contract](#the-cuda-graph-contract)
- [SGLang](#sglang)
- [Ahead-of-time builds](#ahead-of-time-builds)
- [Supported versions, and the contract tests](#supported-versions-and-the-contract-tests)
- [What is deliberately not patched](#what-is-deliberately-not-patched)
- [Your own kernel, in four steps](#your-own-kernel-in-four-steps)
- [Environment variables](#environment-variables)

## The patches

| patch | module | what it replaces | how it is turned on |
|---|---|---|---|
| SDPA drop-in | `vllm.py`, `sglang.py`, `_common.py` | `F.scaled_dot_product_attention`, rebound by identity | `HK_PATCH_VLLM=1` (vLLM); `apply()` in-process |
| decoder attention backend | `vllm_backend.py` | vLLM's `TRITON_ATTN` backend slot | `HK_PATCH_VLLM=1 HK_VLLM_BACKEND=1` |
| linear layers | `vllm_gemm.py` | `dispatch_unquantized_gemm`, rebound by identity | `HK_PATCH_VLLM=1 HK_VLLM_GEMM=1` |
| MoE GEMMs | `vllm_moe.py` | `invoke_fused_moe_triton_kernel` and `try_get_optimal_moe_config`, rebound by identity | `HK_PATCH_VLLM=1 HK_VLLM_MOE=1` |
| AWQ linear layers | `vllm_int4.py` | `AutoAWQLinearMethod.process_weights_after_loading` and `.apply` | `HK_PATCH_VLLM=1 HK_VLLM_INT4=1` |
| your kernels | `register.py` | whatever `register.patch` names | `HK_PATCH_VLLM=1 HK_VLLM_USER=<modules or files>` |
| SGLang linear layers | `sglang.py`, `_sglang_plugin.py` | `UnquantizedLinearMethod.apply` | `HK_SGLANG_LINEAR=1` (SGLang); `apply_linear()` in-process |
| SGLang MoE layers | `sglang_moe.py`, `_sglang_plugin.py` | `topk_softmax`, `invoke_fused_moe_kernel` and `try_get_optimal_moe_config`, rebound by identity | `HK_SGLANG_MOE=1`, or `=topk` for the routing alone |

`engine.py` is what every patch but the two SDPA rebinds shares: the custom
op, the counters, the routing table and the CUDA-graph contract.

## Turning them on

**In vLLM, through its plugin entry point.** The package registers
`hk_attention = "hk.integration._vllm_plugin:register"` under
`vllm.general_plugins` (`pyproject.toml`), which vLLM loads in the driver
*and* every worker. That is the only place a patch reaches the model:

* **vLLM V1 runs the model in a spawned process.** `apply()` in the process
  that built `LLM(...)` patches a process the model never runs in. The first
  end-to-end run reported `{'kernel': 0, 'fallback': 0}` with byte-identical
  answers and a plausible-looking 1.001x. Production therefore needs the
  `vllm.general_plugins` entry point (`hk/integration/_vllm_plugin.py`), which
  vLLM loads in the driver *and* every worker; it is opt-in on
  `HK_PATCH_VLLM=1`, because installing a kernel library must not silently
  change the numerics of every vLLM on the machine.

With `HK_PATCH_VLLM=1` the plugin applies the SDPA patch. The decoder
backend, the linear layers and the MoE layers are each a further opt-in,
because each replaces kernels a server was already running: `HK_VLLM_BACKEND`
registers over the slot ROCm picks by default, `HK_VLLM_GEMM` changes every
unquantized GEMM in the model, and `HK_VLLM_MOE` every bf16 expert GEMM up
to 256 tokens a step. A variable is off when unset or `0`, `off`, `no` or empty
(`python/hk/integration/_vllm_plugin.py`). The invocations, as the module
docstrings give them:

```bash
# the vision tower's attention (SDPA drop-in)
HK_PATCH_VLLM=1 vllm serve OpenGVLab/InternVL2-2B --mm-encoder-attn-backend TORCH_SDPA

# the linear layers
HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B

# the MoE layers
HK_PATCH_VLLM=1 HK_VLLM_MOE=1 vllm serve allenai/OLMoE-1B-7B-0125-Instruct

# the decoder attention backend, with vLLM's defaults: CUDA graphs, prefix
# caching and chunked prefill all on
HK_PATCH_VLLM=1 HK_VLLM_BACKEND=1 vllm serve <model> --attention-backend TRITON_ATTN
```

The plugin prints one `[hk] ...` line per process saying what each patch did.
The end-to-end measurements build the engine with
`attention_backend="TRITON_ATTN"` for the backend arm and vLLM's defaults
otherwise (`HK_VLLM_DEFAULTS=1`, read by `python -m hk.integration ab`,
which `tools/hk-bench/vllm_e2e.py` wraps; the backend asks for its own page
size, 32).

**In SGLang, through its plugin entry point.** The package also registers
`hk = "hk.integration._sglang_plugin:register"` under `sglang.srt.plugins`.
SGLang loads it in the launcher and in every scheduler process, which is where
its model runs. It prints the same `[hk] ...` line per process. Each patch is
opt-in on its own switch: `HK_SGLANG_LINEAR=1` for the linear layers and
`HK_SGLANG_MOE=1` for the MoE layers.

```bash
HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --attention-backend triton --disable-cuda-graph
HK_SGLANG_MOE=1 python -m sglang.launch_server \
    --model-path allenai/OLMoE-1B-7B-0125-Instruct \
    --attention-backend triton --disable-cuda-graph
```

If `SGLANG_PLUGINS` is set, it has to name `hk` for SGLang to load it.

**In-process**, for a script or a framework without plugins:

```python
import vllm                      # let vLLM import its modules first
import hk.integration.vllm as hkv
print(hkv.apply())               # before the engine is built
print(hkv.probe())               # what it can actually reach
```

`apply()` has to run *after* the framework's modules are imported and *before*
a model runs (`python/hk/integration/_common.py`), and with vLLM V1 it only
reaches the model if the engine runs in the same process; the measurements in
the record were run with `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
`hk.integration.apply()` patches every framework that is importable, and
`revert()` and `status()` do what they say. From the shell:

```
python3 -m hk.integration --status          # what is importable, what is patched
python3 -m hk.integration --patch           # apply every available patch and report
python3 -m hk.integration --warm -j 32      # build the torch ops ahead of time
python3 -m hk.integration headroom ...      # a step's device time by kernel family, against what hk has
python3 -m hk.integration ab ... --json f   # end to end against stock, with verdicts
python3 -m hk.integration new-op NAME       # a new op's kernel, registration, tests and bench row
```

The last three are the loop [for agents](agents.md): find the gap, add the
kernel, prove it helped.

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
`tools/hk-bench/vllm_verify.py` is the in-engine verification; the run, the
bf16 rounding bug it found and the tie that is left are in
[Phase 6](history/05-phase-6-torch-ops.md).

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
  on any batch is right for every replay. Kernel for kernel the walk is
  0.86-1.12x triton-rocm 3.7.1's kernel at 16-64 requests (1.03-1.30x
  Triton 3.8.0's at 16, 32 and 64), but in the captured step Triton's
  kernel is the slower ([chapter 38](history/38-rdna-losses.md)).
  Batches are padded to a capture size first, so 9-15 requests run as 16, on hk.
  `vllm_backend.TRITON_DECODE_MAX` is the boundary; 0 turns the routing off.
  `HK_TRITON_DECODE_ALSO="17-24,49-56"` gives Triton further ranges of
  batch sizes; off by default, because in the captured step Triton's
  kernel at 17-22 requests took 1.93 ms a call against hk's 0.451, though
  on the bench it is the faster at 24.
* **Prefill**: `hk.ops.paged_prefill`, one launch for every prefill request
  in the step, reading K/V through the page table. A cached prefix or an
  earlier chunk is just context in front of the new tokens, so prefix caching
  and chunked prefill work as vLLM ships them.

On gfx950 (MI355X) the same paths run gfx950's kernels (`hk.ops.paged_cdna`
and the paged form of `hk.ops.attn_cdna`), with the same page, cache layout
and capture, and hk's decode takes every batch: against Triton's kernel on
this cache it is 0.81-0.91x only at 512 tokens and one to four requests, and
1.14-1.99x from 2048 ([gfx950's paged attention](history/39-cdna-paged.md)).
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
path is gated on gfx9 ([the linear layers](history/08-linear-layers.md)).
vLLM 0.31 runs those skinny GEMMs on gfx11 too, at M ≤ 5. The routing table
is measured against whichever path the installed vLLM takes
([vLLM 0.31](history/20-vllm-031.md)).
`hk/integration/vllm_gemm.py` installs the replacement, behind its own opt-in
(`HK_PATCH_VLLM=1 HK_VLLM_GEMM=1`). It rebinds `dispatch_unquantized_gemm` by
identity in every vLLM module that imported it, and what that returns calls a
single opaque custom op, `hk_vllm::linear`, with a fake implementation. Under
vLLM's torch.compile the token count is symbolic, so branching on it inside
traced code would be a guard and a recompile; inside the op it is neither, and
CUDA-graph capture sees a launch on torch's stream.

On gfx1100 each call takes one of three paths, counted in `vllm_gemm.STATS`:
`hk` (the prefill GEMM), `hk_decode` (the decode GEMV, for small M) or
`fallback` — vLLM's own `rocm_unquantized_gemm` where it has one, else
`F.linear`. Which one is a [routing table](#routing-tables). A table may
also send a call to FlyDSL's GEMM (`flydsl`, below), and on gfx950 to
HipKittens' CDNA4 GEMM (`hipkittens`,
[design note](design/cdna-hipkittens-ops.md)). A call also falls back when it is
not bf16, has a bias, has a non-contiguous weight, does not tile, or runs on a
device of another arch. With no table for the current arch the patch is not
installed at all, and says so.

### A model no table has measured

The shipped table is Qwen3-8B's, and a (N, K) it never measured takes vLLM's
path. So the patch measures the model it is given, on the machine it runs on
(`hk/integration/vllm_autotune.py`, [chapter 35](history/35-any-model.md)):

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
python -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct          # --tp N, --force, --dry-run
```

It reads the model's `config.json` (a path, the local HF cache, or the hub)
and writes the same records. `HK_AUTOTUNE_BUDGET=0` turns the measuring at
load off. `HK_AUTOTUNE=0` ignores the records as well. A table named by
`$HK_VLLM_ROUTES` is used alone, and nothing is measured against it.

The patch also points `VLLM_CACHE_ROOT` at a `hk-linear` subdirectory.
vLLM keys its compiled graphs, including the AOT cache, by config and
source, not by what is patched. A patched engine sharing a cache root with
an unpatched one loads back graphs whose linear layers never call the op
([vLLM 0.31](history/20-vllm-031.md)).

### FlyDSL's GEMM as an optional path

hk-dsl sits beside FlyDSL rather than replacing it: FlyDSL's RDNA3 GEMM is
faster than hk's at some shapes, and hk's decode GEMV and attention at
others ([the scoreboard](history/26-the-scoreboard-retuned.md)). So the op
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
  speed depends on which tile M leaves it. At (4096, 4096), 1024 rows run a
  128x128 tile at 1.02x hk's time, while 1088 rows fit only 64-row tiles,
  at 1.34x. A table's FlyDSL rows therefore name the tile, and a call takes the
  path only if FlyDSL's dispatcher would pick that tile for it.
- **Everything else goes where the table without FlyDSL sends it.** A
  bf16, contiguous call whose M no tile divides, at another tile, or on an
  install without FlyDSL, takes the table's own path, hk's or the engine's.
- **Costs.** Each new M is a FlyDSL compile, 0.4-0.8 s cold and 0.04 s from
  FlyDSL's disk cache. A call costs about 60 us of host time through
  FlyDSL's launch path, against 17.8 us for hk's op. Both are hidden behind a GEMM
  at the sizes where FlyDSL wins. The launch is on torch's stream with no
  host sync, so CUDA graphs capture it (`tests/hk/gpu/test_flydsl_gemm.py`).

The shipped Qwen3-8B table carries FlyDSL rows, mostly at decode-sized Ms.
End to end they shorten a decode step at 96-128 requests by 12%. Since hk's
GEMM got faster ([chapter 28](history/28-gemm-schedule.md)), FlyDSL beats it
by 2% at few prefill Ms: the re-measured table routes 41 points to FlyDSL
instead of 105. Of its prefill rows, the one measured end to end, gate_up's
8192-token steps, makes those prefills 1.5-1.7% shorter; dropping the rows
at M = 1024 cost a 1024-token prefill 1.2%
([performance](performance.md#flydsls-gemm-on-the-routes)). SGLang's table
carries FlyDSL rows too (38 points); served at 32 requests they make the
decode step 0.4-0.6% shorter, and at 1 and 8 requests change nothing
measurable ([chapter 32](history/32-measuring-the-measurements.md)).

## The MoE patch

vLLM's unquantized MoE sorts each batch's (token, choice) pairs by expert,
then makes each of the layer's two GEMMs one call of
`invoke_fused_moe_triton_kernel` on the sorted lists.
`hk/integration/vllm_moe.py` rebinds that name, plus the
`try_get_optimal_moe_config` that sizes the sort's blocks, by identity in
every vLLM module that imported them:
- **Up to `BLOCK16_MAX_TOKENS` (256) tokens a step**, the config asks for
  16-row blocks and hk's GEMM (`hk.ops.moe`) runs both GEMMs.
- **Above that, or for a quantized, biased or otherwise untaken GEMM**,
  vLLM's own kernel runs.

The counters are `vllm_moe.STATS`. The MoE layer is a custom op the compiler
does not trace into, so the stand-in runs on every call, and the graphs
vLLM captures for decode record hk's launches
([the MoE GEMM](history/21-moe.md)).

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
call, so it uses the compile-cache root `hk-int4`
([4-bit weights](history/22-int4.md)).

```bash
VLLM_USE_TRITON_AWQ=1 HK_PATCH_VLLM=1 HK_VLLM_INT4=1 \
    vllm serve Qwen/Qwen3-8B-AWQ --dtype float16
```

## The engine layer: `Op`, `Patch`, `RouteTable`

The GEMM patch and the attention backend each wrote the same parts by hand.
`hk.integration.engine` writes them once:

* **`Op`** is an opaque torch custom op with a fake implementation.
* **`Patch`** counts each path a call takes, and holds the patch's CUDA-graph
  contract.
* **`RouteTable`** maps a shape to a kernel, and replaces the transcribed
  rule.

`vllm_gemm` is an `Op` routed by a table. `vllm_backend` is a `Patch`: its
metadata builder's `_cudagraph_support` is derived from its declaration
instead of being a second statement of it.

The counters are Python, so a CUDA-graph replay — which runs no Python — is
not counted; calls made while capturing are counted apart, in
`Patch.captured` (`python/hk/integration/engine.py`).

## Routing tables

`tools/hk-bench/route_tune.py` measures each (N, K) of the model at every M up
to 64, every 16 rows up to 1024, and coarser above. At each point it times the
engine's path, the prefill GEMM and the decode GEMV, each gated against fp32
first, interleaved in one process, with weights rotated through 3x the
Infinity Cache. A kernel takes an M only where it beat the engine by 3%; a tie
stays the engine's. How a point is timed:
- a set of 3 rounds on fresh activations is thrown away first: straight
  after the correctness check FlyDSL ran a median 1.5% slow;
- one set is timed, and a point with a decision within 2x its noise of the
  bar gets two sets more, its time then the median of the sets;
- each path's noise and the number of sets are written beside its time.

Two runs an hour apart decide about 12 of 740 points differently, all in
FlyDSL's overlay, where it and hk's GEMM are within 2%; the runs themselves
agree. `route_tune.py --compare A B` lists the differences, and `--confirm`
an earlier run's table keeps only the overlay points both runs took
([chapter 32](history/32-measuring-the-measurements.md)). The table answers
an M as follows:

* inside a run of measured Ms with the same winner, that winner;
* between two runs, which means between two measured Ms that disagreed, the
  engine's path;
* above the largest measured M, the winner measured there;
* below the smallest, or for a shape never measured, the engine's path, and
  the shape is recorded as a miss;
* for a shape whose grid a tuning budget cut short (`partial`), above the
  largest M measured, the engine's path.

The measuring is a library, `hk.autotune.routes`: `tune_shape(n, k, ...)`
returns one shape's entry, and `route_tune.py`, `python -m hk.integration
tune` and the [measuring at load](#a-model-no-table-has-measured) are thin
shells over it. `engine.shipped_table` lays this machine's records over the
shipped table, shape by shape. A record measured against another engine's
path (vLLM upgraded, say) is left out, and measured again at the next load.

Tables are per arch, per model and per engine, because the path a kernel has
to beat is the engine's. The shipped ones, in `hk/tuned/`, named
`routes_<engine>_<kind>_<arch>.json`:
- `routes_vllm_linear_gfx1100.json`: Qwen3-8B against vLLM 0.31's op, raw
  timings included, with FlyDSL rows (below). `$HK_VLLM_ROUTES` names
  another.
- `routes_sglang_linear_gfx1100.json` (`route_tune.py --engine sglang`):
  against the `F.linear` SGLang calls on ROCm, with FlyDSL rows.
  `$HK_SGLANG_ROUTES` names another. It disagrees with vLLM's below M = 4:
  vLLM runs its own skinny GEMMs there and ties hk's GEMV, while `F.linear`
  is 1.3-2.3x slower than it.
- `routes_vllm_linear_gfx950.json`: on an MI355X, against hipBLASLt, with
  HipKittens' CDNA4 GEMM as the candidate. It takes 37 of 206 points, all at
  M ≥ 1792 ([design note](design/cdna-hipkittens-ops.md)).

A table measured on another arch is refused, and an `$HK_VLLM_ROUTES` that
names a missing file raises (`engine.shipped_table`). There are no buckets:
GEMM time is a comb in M, not a curve, and a power-of-two bucket either
routes the teeth or gives up the gaps between them. What the table bought
over the hand-written rule is in [its chapter](history/14-routing-table.md).

**Optional paths are an overlay.** With FlyDSL installed, `route_tune.py`
also times FlyDSL's GEMM, through the same call the op makes, at every M one
of its tiles divides (`--no-flydsl` leaves it out). FlyDSL never enters the
runs above, which are what every install routes by. Instead each shape gets
`prefer: {"flydsl": [[lo, hi, tile], ...]}`: for each tile FlyDSL's
dispatcher picked, the measured Ms where it beat the engine by `min_gain` and
the runs' own answer by `opt_gain` (2%), merged over consecutive measured Ms
of that tile. `hi` null means "and above". `RouteTable.preferred` answers which
ranges cover an M; `vllm_gemm.route_for` takes one only if FlyDSL's dispatcher
would pick that tile, and otherwise answers from the runs.

## The CUDA-graph contract

A patch declares `graph="safe"` or `graph="never"`, and
"never" must give its reason. Under capture:

* A "never" patch raises `GraphCaptureError` with that reason, instead of
  being frozen at the capture batch's values.
* A "safe" one runs with torch's sync debug mode set to error. A host sync
  in its wrapper then raises at the line that does it, naming the patch,
  instead of failing the capture somewhere else, or succeeding.
* Calls made while capturing are counted apart from eager calls, since a
  replay runs no Python.
* Changing the routes after a capture warns, because graphs captured before
  keep the old routes.

The limit is torch's: its sync debug mode is a prototype that sees torch's
own syncs (`.item()`, `.tolist()`, blocking copies, `synchronize()`), not a
raw HIP call inside an extension.

The linear-layer op is `graph="safe"`: its route depends on shapes alone,
which a graph captured per batch size fixes. So is the attention backend, and
its builder declares `UNIFORM_SINGLE_TOKEN_DECODE`: vLLM captures pure-decode
steps whole and runs steps with prefill in them between its pieces. What makes
the decode capturable is that nothing in it depends on more than shapes -- the
split count comes from the batch size and max_model_len, the scratch is
allocated once at its largest, the first-call metadata check that reads the
device is skipped under capture, and the hk-or-Triton choice is the batch
size. A prefill step reads its query lengths on the host, so `forward`
refuses, with `GraphCaptureError`, to be captured on one (`vllm_backend.PATCH`).

Not ported: the two SDPA rebinding patches (`vllm.py`, `sglang.py`) still
keep their own counters.

## SGLang

Launched with the plugin as in [turning them on](#turning-them-on), or patched
in-process:

```python
import sglang                      # let SGLang import its modules first
import hk.integration.sglang as hks
print(hks.apply_linear())          # unquantized linear layers -> hk_vllm::linear
print(hks.apply())                 # SDPA, before the engine is built
```

**The linear layers are measured inside SGLang.** The setup was SGLang 0.5.18
with AMD-AIM/sglang-radeon 0.1.5 and its prebuilt gfx1100 kernel, torch
2.9.1 on ROCm 7.2.1, and Qwen3-8B in bf16, eager with Triton attention as
sglang-radeon runs it ([chapter 24](history/24-sglang.md)).
- `apply_linear()` rebinds `UnquantizedLinearMethod.apply`, whose ROCm path
  without aiter is `F.linear`, to `torch.ops.hk_vllm.linear`, routed by the
  SGLang table. Every call took an hk kernel.
- Served with `launch_server` and `bench_serving` (512 tokens in, 128 out),
  output throughput rose 1.81x at concurrency 1, 1.70x at 8 and 1.49x at 32,
  and time per output token fell from 43.5 to 24.0 ms at concurrency 1.
- The patch does not change accuracy: hk's linear and F.linear land equally
  far from an fp64 matmul, to three digits. In a greedy check, two of three
  prompts matched stock over 48 tokens; the third took the other side of a
  near-tie at its ninth token.

**MoE models run, and the expert GEMMs are measured inside SGLang**
([chapter 25](history/25-sglang-moe.md)).
- **Without the patch they do not run.** sgl_kernel's `topk_softmax` fails
  on gfx1100 on its own, with `unspecified launch failure`, before any
  expert GEMM. So no softmax-routed MoE model ran in SGLang on this card.
- **What the patch changes.** `sglang_moe.apply_topk()` computes the top-k
  in torch, as SGLang's own `fused_topk_torch_native` does, and with only
  that, OLMoE runs on SGLang's Triton MoE kernel. `sglang_moe.apply()` also
  routes the bf16 expert GEMMs through `hk.ops.moe`, asking for 16-row
  blocks up to 256 tokens a step.
- **Results.** Against Triton's GEMMs, OLMoE served 1.06x faster at
  concurrency 1, 1.32x at 8 and 1.26x at 32, with greedy output identical
  over SGLang's correctness prompts. Prefill steps of more than 256 tokens
  stay on Triton.

**Limits.**
- **No CUDA graphs.** sglang-radeon disables them on RDNA3 because HIP graph
  capture hangs there, so these are eager numbers. Even so, the decode step
  is GPU-bound: at batch 1 the linear layers' timings, measured one by one,
  add up to 42.1 ms of a 44.3 ms step.
- **Attention is not verified.** SGLang's RDNA3 path selects its Triton
  attention backend, which does not call `F.scaled_dot_product_attention`.
  On a text model `apply()` therefore reports that it rebound nothing; it
  bites with `--attention-backend torch_native` and in the vision encoders
  of multimodal models.

## Ahead-of-time builds

`python3 -m hk.integration --warm -j 32` fills the cache for the torch half
specifically. A torch op is a device-only code object by default
([runtime](runtime.md#over-a-device-only-code-object)): 1.1-1.5 s a kernel
cold on the pod, where the libtorch registration it replaced took 15, so a
server's first request compiles in seconds even unwarmed. A `--aot` pass
built for pybind does not cover it, since the artifacts differ. Run it and
`python3 -m hk.autotune --aot` both ([warming the cache](runtime.md#warming-the-cache)).

## Supported versions, and the contract tests

Every patch rebinds engine internals by name, so an engine release can break
one without an error: the rebind finds nothing, a stand-in forwards
arguments into the wrong slots, or a backend hook becomes dead code.
`tests/hk/engines/` checks each patched symbol against the engine installed,
and is skipped where the engine is not importable (CI, a laptop):
- **the signature a stand-in forwards to**, parameter for parameter and
  default for default (`invoke_fused_moe_triton_kernel`,
  `try_get_optimal_moe_config`, `topk_softmax`, `invoke_fused_moe_kernel`,
  `UnquantizedLinearMethod.apply`, the AWQ method and ops,
  `reshape_and_cache_flash`, the keyword arguments hk passes to vLLM's
  `unified_attention`);
- **every call site a rebind by identity has to reach**: each engine module
  with a function that loads the patched name as a global is found by
  disassembly, and must hold the original object, so the rebind redirects it;
- **the attention backend**: it registers over `TRITON_ATTN`, is not abstract,
  takes `AttentionImpl`'s arguments in order, and every hook it overrides is
  one the installed vLLM reads, or is listed as legacy or as hk's own;
- **the plugin groups** each engine loads, and that hk's entry points are in
  them.

```bash
python3 -m pytest tests/hk/engines/test_vllm_contract.py -q        # vLLM's venv
HIP_VISIBLE_DEVICES=3 python3 -m pytest tests/hk/engines -q        # SGLang's
```

SGLang reads the GPU's arch while importing its kernels, so its file needs a
GPU visible, though nothing is launched.

What was run, on the dev pod (`hfang-hk-vllm`, gfx1100), 2026-10-16:

| patch | vLLM 0.31.0+rocm721 | vLLM 0.16.1 | SGLang 0.5.18 |
|---|---|---|---|
| linear layers (`dispatch_unquantized_gemm`, its call sites in `linear` and `vocab_parallel_embedding`, `rocm_unquantized_gemm`) | holds | holds | -- |
| MoE (`invoke_fused_moe_triton_kernel`, `try_get_optimal_moe_config`) | holds | holds | -- |
| AWQ (`auto_awq.AutoAWQLinearMethod`, `awq_gemm`, `awq_dequantize`) | holds | **breaks**: no `quantization.auto_awq` | -- |
| attention backend (registry, `AttentionImpl`, ROCm builder, cache layout, `unified_attention`, `reshape_and_cache_flash`) | holds | holds; the layout comes from `get_kv_cache_shape` | -- |
| SDPA (`vit_attn_wrappers` through `F`) and the plugin group | holds | holds | -- |
| SGLang linear (`UnquantizedLinearMethod.apply`) | -- | -- | holds |
| SGLang MoE (`topk_softmax`, `invoke_fused_moe_kernel`, `try_get_optimal_moe_config` and what it returns) | -- | -- | holds |
| SGLang plugin group, loaded by the scheduler | -- | -- | holds |
| tests passed | 19 of 19 | 18 of 19 | 8 of 8 |

The supported versions are the ones measured end to end: **vLLM 0.31.0**
(torch 2.13.0+rocm7.2, triton-rocm 3.7.1, Python 3.12;
[vLLM 0.31](history/20-vllm-031.md), [MoE](history/21-moe.md),
[4-bit](history/22-int4.md), [chapter 30](history/30-flydsl-routes-gate-contracts.md),
re-measured on that stack in [chapter 44](history/44-matched-stack.md))
and **SGLang 0.5.18** with sglang-radeon 0.1.5 (torch 2.9.1, ROCm 7.2.1;
[chapter 24](history/24-sglang.md), [chapter 25](history/25-sglang-moe.md)).
vLLM 0.16.1 is the copy installed beside SGLang. Its contract holds except
for AWQ, but nothing has been run end to end on it since the port to 0.31.

Check which Triton the venv imports. vLLM 0.31's requirements include
xgrammar, which requires `triton` with no version; triton-rocm, which
torch 2.13.0+rocm7.2 pins, is a different distribution name, so installing
them after torch puts PyPI's newest `triton` over triton-rocm's files in the
same package. That is how the venv the record measured in from chapter 24
to 38 came to import Triton 3.8.0 ([chapter 44](history/44-matched-stack.md)).
`python -c "import triton; print(triton.__version__)"` should print 3.7.1;
if it does not, `pip uninstall triton` and reinstall the triton-rocm 3.7.1
wheel with `--no-deps --force-reinstall`. vLLM 0.31's own ROCm image builds
yet another Triton, ROCm/triton's 3.8.x branch, and under 3.7.1 vLLM logs
that its vendored `triton_kernels` (MXFP4 MoE) cannot be imported; nothing
hk patches uses them.

On 0.31, three of the backend's hooks are no longer read
(`accept_output_buffer`, `get_kv_cache_shape`, the backend's
`use_cascade_attention`). They are kept for older vLLM, and the test lists
them as legacy rather than letting them pass as live. A new vLLM gets the
same treatment: run the file in its venv, and a failure names the symbol
and what moved.

## What is deliberately not patched

Two things are deliberately *not* patched, and the reason is the same both
times -- the patch would be slower than what it replaces:

* **RMSNorm.** Every framework's RMSNorm applies a learned weight, and
  vLLM's fuses the residual add. `hk.ops.rms_norm` and
  `hk.ops.fused_add_rms_norm` are those two ops, bit-identical to torch on
  the residual. Under torch.compile vLLM's norms come from Inductor, fused
  with their neighbours, and an opaque op would cost Inductor those fusions.
  On Qwen3-8B shapes the tile kernels of the time also took 44-64% longer
  than Inductor's norms before multiplying any weight
  ([performance](performance.md#against-what-vllm-runs-under-torchcompile)).
  That was before chapter 27's strips, which beat Inductor's standalone
  RMSNorm at 1-8192 rows; inside vLLM's compiled graph it has not been
  re-measured.
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves. A kernel's globals take such halves since
  [strided globals](history/12-strided-globals.md), but the `hk.ops.silu_mul`
  wrapper still asks for contiguous operands (it reshapes them to a faster
  2-D shape), so the patch would copy both halves first. And at (4096,
  2x12288) the kernel on contiguous halves ran 429.7 µs to Inductor's fused
  409.5, so the patch was not built.

## Your own kernel, in four steps

`hk.integration.register` is the path from a kernel you wrote to a vLLM that
runs it. [`examples/11_vllm_integration.py`](../examples/11_vllm_integration.py)
is the whole thing, for vLLM's SiLU-and-mul.

**1. Register it** as a torch op with the reference it replaces:

```python
from hk.integration import register

op = register.op(
    "silu_mul",                                # torch.ops.hk_user.silu_mul
    kernel=silu_mul,                           # @hk.kernel: inputs, then outputs
    reference=lambda x: F.silu(x[:, :x.shape[1] // 2]) * x[:, x.shape[1] // 2:],
    out=lambda x: x.new_empty(x.shape[0], x.shape[1] // 2),
    route=lambda x: (x.shape[1] // 2) % 256 == 0,
)
```

`out` allocates the outputs, and on meta tensors it is the op's fake, which
torch.compile traces. `route` runs inside the op, where shapes are numbers;
a call it declines runs `reference`. The op launches on torch's stream, so
CUDA graphs capture it, and it counts its calls (`op.summary()`).

**2. Check it**: `op.check(x)` runs kernel and reference and raises past
`rel` (default 2e-2).

**3. Patch vLLM**: `register.patch("vllm.<module>.<Class>.<method>", fn)`.
Patch the method vLLM actually calls. Under its default compile config
custom ops are off, so for a `CustomOp` that is `forward_native`, not
`forward_cuda`. A patch inside vLLM also moves the engine's compile cache to
a root of its own, because the cache does not know what is patched into it
([vLLM 0.31](history/20-vllm-031.md)).

**4. A/B it** in vLLM's own process. vLLM builds the model in the process
that loads its plugins, so the module that registers and patches is named
to the plugin, which imports it there:

```bash
HK_VLLM_USER=examples/11_vllm_integration.py \
    python -m hk.integration ab --model Qwen/Qwen3-8B --arms user \
    --context 1024 --batch 1,16 --short 16 --tokens 48
HK_PATCH_VLLM=1 HK_VLLM_USER=my_kernels.py vllm serve Qwen/Qwen3-8B
```

`ab` runs stock and your arm alternately in fresh engines. It prints total,
prefill and decode-step time per shape, and whether the first request's
greedy tokens match stock's. `HK_VLLM_USER` takes comma-separated module
names or `.py` paths. `--json PATH` writes one row per shape, metric and arm
with a verdict, `win`, `loss`, `tie` or `noise`, against a band measured
from the rounds' own spread; `--out` keeps every round raw, and `--from`
judges such a file again ([agents](agents.md#prove-it-helps)).

`python -m hk.integration new-op NAME --dir DIR` writes the four steps' code
for a new op, as a working stub with an exact-answer test and a scoreboard
row, and `python -m hk.integration headroom` says which of the engine's
kernels are worth one: [the agents' page](agents.md) is the whole loop.

**Measure first.** The example's kernel is 0.62-0.93x the SiLU-and-mul vLLM
compiles with Inductor, which is the result chapter 12 recorded when it
dropped the same patch. It is there for the mechanics. A kernel that wins
only at some shapes should keep only those, through `route`.

## Environment variables

Read by `python/hk/integration` and the kernels it routes to, exactly as
spelled:

| variable | read by | effect |
|---|---|---|
| `HK_PATCH_VLLM` | `_vllm_plugin.py` | the consent: without it the plugin does nothing. With it, the SDPA patch is applied |
| `HK_VLLM_BACKEND` | `_vllm_plugin.py` | register the decoder attention backend; a value other than `1`/`on`/`yes`/`true` names the slot |
| `HK_VLLM_BACKEND_SLOT` | `vllm_backend.py` | the `AttentionBackendEnum` member to register over (default `TRITON_ATTN`) |
| `HK_VLLM_GEMM` | `_vllm_plugin.py` | route the unquantized linear layers |
| `HK_VLLM_MOE` | `_vllm_plugin.py` | route the bf16 MoE GEMMs up to 256 tokens a step |
| `HK_VLLM_INT4` | `_vllm_plugin.py` | route AWQ linear layers' fp16/bf16 batches up to 64 tokens |
| `HK_VLLM_USER` | `_vllm_plugin.py` | comma-separated modules or `.py` files to import in vLLM's process: your `register.op` / `register.patch` calls |
| `HK_VLLM_ROUTES` | `vllm_gemm.py`, via `engine.shipped_table` | a routing table to use instead of the shipped one and this machine's records; nothing is measured at load against it |
| `HK_AUTOTUNE_BUDGET` | `vllm_autotune.py` | seconds the linear patch may spend at load measuring the model's shapes no table has (default 60); `0` measures nothing, `inf` without a limit |
| `HK_AUTOTUNE` | `hk.autotune`, `engine.shipped_table`, `vllm_autotune.py` | `0`: ignore this machine's records -- schedules and routes -- and measure nothing at load |
| `HK_TUNE_DIR` | `hk.autotune` | where this machine's records go (default `$HK_CACHE_DIR/autotune`): `routes_<engine>_linear_<arch>.json` for routes |
| `HK_FLYDSL` | `flydsl_gemm.py` | `0`/`off`/`no`/`false`: never take a table's FlyDSL rows, even with FlyDSL installed |
| `FLYDSL_ROOT` | `flydsl_gemm.py` | a FlyDSL checkout whose `kernels/` package matches the installed `flydsl`, appended to `sys.path`; where another `kernels` is installed, imported from beside it |
| `HK_SGLANG_LINEAR` | `_sglang_plugin.py` | route SGLang's unquantized linear layers |
| `HK_SGLANG_ROUTES` | `sglang.py`, via `engine.shipped_table` | a routing table for SGLang's linear layers instead of the shipped one |
| `HK_SGLANG_MOE` | `_sglang_plugin.py` | `1`: the torch top-k and hk's expert GEMMs; `topk`: the torch top-k alone |
| `HK_SPLITS` | `ops/paged.py` | the decode backend's KV split: `auto` (default), `off`/`0`/`1`, or a count |
| `HK_TRITON_DECODE_ALSO` | `vllm_backend.py` | decode batch sizes past `TRITON_DECODE_MAX` that run Triton's kernel on hk's cache, as ranges: `17-24,49-56` (default none) |
| `HK_NORM_LAYOUT` | `ops/norm.py` | `strip` (default): the norms and softmax run on the strip kernel wherever a row fits one; `tile`: always on the tile kernels |
| `HK_NORM_FOLD` | `ops/norm.py` | the tile kernels' plan for the weighted norms: `auto` (default), `on`, `off`. Reached only for rows too wide for a strip (over 32768 elements) or under `HK_NORM_LAYOUT=tile` |
| `HK_HIPKITTENS_KERNELS` | `ops/hipkittens.py` | HipKittens' `kernels/cdna4` to build the wrapped CDNA4 kernels from (default: the submodule's) |
| `HK_VLLM_DEFAULTS` | `ab.py` | `1`: the A/B builds vLLM with prefix caching and chunked prefill on, as vLLM ships |
| `HK_EAGER` | `ab.py` | `1` (default): the A/B runs vLLM eager; `0`: with CUDA graphs |
| `HK_GPU_UTIL` | `ab.py` | the A/B's `gpu_memory_utilization` (default 0.90) |

`HK_PATCH_VLLM`, `HK_VLLM_BACKEND`, `HK_VLLM_GEMM`, `HK_VLLM_MOE`,
`HK_VLLM_INT4`, `HK_SGLANG_LINEAR` and `HK_SGLANG_MOE` are off when unset or
`0`, `off`, `no` or empty. The benchmark harnesses in `tools/hk-bench/` read
more; those are theirs, not the library's. The runtime's own variables are in
[the runtime](runtime.md#environment-variables).
