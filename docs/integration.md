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
- [The engine layer: `Op`, `Patch`, `RouteTable`](#the-engine-layer-op-patch-routetable)
- [Routing tables](#routing-tables)
- [The CUDA-graph contract](#the-cuda-graph-contract)
- [SGLang](#sglang)
- [Ahead-of-time builds](#ahead-of-time-builds)
- [What is deliberately not patched](#what-is-deliberately-not-patched)
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

`engine.py` is what the last two share: the custom op, the counters, the
routing table and the CUDA-graph contract.

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
otherwise (`HK_VLLM_DEFAULTS=1` in `tools/hk-bench/vllm_e2e.py`; the backend
asks for its own page size, 32).

**In SGLang, through its plugin entry point.** The package also registers
`hk = "hk.integration._sglang_plugin:register"` under `sglang.srt.plugins`.
SGLang loads it in the launcher and in every scheduler process, which is where
its model runs. It is opt-in on `HK_SGLANG_LINEAR=1` and prints the same
`[hk] ...` line per process:

```bash
HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
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
```

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
`AttentionBackendEnum` is closed and `register_backend` overrides rather than
adds; the reserved `CUSTOM` member is rejected by `RocmPlatform`'s allowlist,
so the override has to target a member the platform accepts (`TRITON_ATTN`,
which is also ROCm's default -- hence the separate `HK_VLLM_BACKEND=1`
opt-in); `get_name()` must answer with the member it overrides because vLLM
round-trips the name through the enum. `HK_VLLM_BACKEND_SLOT`, or a value of
`HK_VLLM_BACKEND` other than `1`/`on`/`yes`/`true`, names a different member
to register over. `register()` reports the slot it took, as
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
* **Decode, sixteen requests and up**: `hk.ops.paged`'s split walk and its
  merge, two launches, reading the engine's query and writing its output in
  place. The split count is planned from the batch size and max_model_len
  alone, so a graph captured on any batch is right for every replay. On
  vLLM 0.31 Triton's 2-D kernel is 5-20% faster than the walk kernel for
  kernel, but giving it these batches made the captured step slower (4096 x
  16: 59.0 -> 74.2 ms), and the step is what is kept
  ([vLLM 0.31](history/20-vllm-031.md)). Batches are padded to a capture
  size first, so 9-15 requests run as 16, on hk.
  `vllm_backend.TRITON_DECODE_MAX` is the boundary; 0 turns the routing off.
* **Prefill**: `hk.ops.paged_prefill`, one launch for every prefill request
  in the step, reading K/V through the page table. A cached prefix or an
  earlier chunk is just context in front of the new tokens, so prefix caching
  and chunked prefill work as vLLM ships them.

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

Each call takes one of three paths, counted in `vllm_gemm.STATS`: `hk` (the
prefill GEMM), `hk_decode` (the decode GEMV, for small M) or `fallback` —
vLLM's own `rocm_unquantized_gemm` where it has one, else `F.linear`. Which
one is a [routing table](#routing-tables). A call also falls back when it is
not bf16, has a bias, has a non-contiguous weight, does not tile, or runs on a
device of another arch. With no table for the current arch the patch is not
installed at all, and says so.

The patch also points `VLLM_CACHE_ROOT` at a `hk-linear` subdirectory.
vLLM keys its compiled graphs, including the AOT cache, by config and
source, not by what is patched. A patched engine sharing a cache root with
an unpatched one loads back graphs whose linear layers never call the op
([vLLM 0.31](history/20-vllm-031.md)).

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
stays the engine's. The table answers an M as follows:

* inside a run of measured Ms with the same winner, that winner;
* between two runs, which means between two measured Ms that disagreed, the
  engine's path;
* above the largest measured M, the winner measured there;
* below the smallest, or for a shape never measured, the engine's path, and
  the shape is recorded as a miss.

Tables are per arch, per model and per engine, because the path a kernel has
to beat is the engine's. `hk/tuned/routes_vllm_linear_gfx1100.json` is
Qwen3-8B's against vLLM 0.31's op, raw timings included, and `$HK_VLLM_ROUTES`
names another. `routes_sglang_linear_gfx1100.json` (`route_tune.py --engine
sglang`) is measured against the F.linear SGLang calls on ROCm, and
`$HK_SGLANG_ROUTES` names another. The two disagree below M = 4: vLLM runs its
own skinny GEMMs there and ties hk's GEMV, while F.linear is 1.3-2.3x slower
than it. A
table measured on another arch is refused, and an `$HK_VLLM_ROUTES` that names
a missing file raises (`engine.shipped_table`). There are no buckets: GEMM time
is a comb in M, not a curve, and a power-of-two bucket either routes the teeth
or gives up the gaps between them. What the table bought over the hand-written
rule is in [its chapter](history/14-routing-table.md).

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

```bash
HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --attention-backend triton --disable-cuda-graph
```

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

**Limits.**
- **No CUDA graphs.** sglang-radeon disables them on RDNA3 because HIP graph
  capture hangs there, so these are eager numbers. Even so, the decode step
  is GPU-bound: at batch 1 the linear layers' timings, measured one by one,
  add up to 42.1 ms of a 44.3 ms step.
- **MoE models do not run in SGLang on RDNA3.** Its Triton fused-MoE faults
  on gfx1100. hk's MoE GEMM, which runs in vLLM, is not wired into SGLang.
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

## What is deliberately not patched

Two things are deliberately *not* patched, and the reason is the same both
times -- the patch would be slower than what it replaces:

* **RMSNorm.** Every framework's RMSNorm applies a learned weight, and
  vLLM's fuses the residual add. `hk.ops.rms_norm` and
  `hk.ops.fused_add_rms_norm` are those two ops, bit-identical to torch on
  the residual; they are not patched in because under torch.compile vLLM's
  norms come from Inductor, fused with their neighbours, and at a decode
  step's handful of rows they lose to it ([performance](performance.md)).
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves and `hk.ops.silu_mul` has no stride support, so the
  patch would add two copies to save one pass.

The stride half of that was built later and the patches were measured before
being written: under torch.compile both ops come from Inductor, fused with
their neighbours, and the kernels lose to it, so both integrations were
dropped ([strided globals](history/12-strided-globals.md)).

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
names or `.py` paths.

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
| `HK_VLLM_ROUTES` | `vllm_gemm.py`, via `engine.shipped_table` | a routing table to use instead of the shipped one |
| `HK_SGLANG_LINEAR` | `_sglang_plugin.py` | route SGLang's unquantized linear layers |
| `HK_SGLANG_ROUTES` | `sglang.py`, via `engine.shipped_table` | a routing table for SGLang's linear layers instead of the shipped one |
| `HK_SPLITS` | `ops/paged.py` | the decode backend's KV split: `auto` (default), `off`/`0`/`1`, or a count |
| `HK_NORM_FOLD` | `ops/norm.py` | the weighted norms' plan: `auto` (default, folded below 4096 rows), `on`, `off` |

`HK_PATCH_VLLM`, `HK_VLLM_BACKEND`, `HK_VLLM_GEMM`, `HK_VLLM_MOE`,
`HK_VLLM_INT4` and `HK_SGLANG_LINEAR` are off when unset or `0`, `off`, `no`
or empty. The benchmark harnesses in `tools/hk-bench/` read
more (`HK_GPU_UTIL`, `HK_EAGER`, ...); those are theirs, not the library's.
