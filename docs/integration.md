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

With `HK_PATCH_VLLM=1` the plugin applies the SDPA patch. The decoder backend
and the linear layers are each a further opt-in, because each replaces kernels
a server was already running: `HK_VLLM_BACKEND` registers over the slot ROCm
picks by default, and `HK_VLLM_GEMM` changes every unquantized GEMM in the
model. A variable is off when unset or `0`, `off`, `no` or empty
(`python/hk/integration/_vllm_plugin.py`). The invocations, as the module
docstrings give them:

```bash
# the vision tower's attention (SDPA drop-in)
HK_PATCH_VLLM=1 vllm serve OpenGVLab/InternVL2-2B --mm-encoder-attn-backend TORCH_SDPA

# the linear layers
HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B

# the decoder attention backend: no paged prefill, so the last two are required
HK_PATCH_VLLM=1 HK_VLLM_BACKEND=1 vllm serve <model> --attention-backend TRITON_ATTN \
    --no-enable-prefix-caching --enable-chunked-prefill=False
```

The plugin prints one `[hk] ...` line per process saying what each patch did.
The end-to-end measurements built the engine with `block_size=32`,
`attention_backend="TRITON_ATTN"` for the backend arm, and prefix caching and
chunked prefill off (`tools/hk-bench/vllm_e2e.py`).

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

**What it refuses.** Scope, stated rather than discovered: no paged *prefill*
kernel, so prefix caching and chunked prefill are refused rather than silently
attending over part of a context. Prefill is served by the dense kernel
(`hk.ops.attn`) over the request's own tokens; prompts too short for it go to
torch, because zero-padded keys are not masked out, they are keys with score
zero and `exp(0)` is a real weight in the softmax sum.

**It is slower than Triton at decode.** The kernel splits the KV axis
(`plan_splits` in `python/hk/ops/paged.py`, aiming at about a thousand
workgroups), and what is left is a flat 1.14-1.19x per decode step, which is
the dozen small torch ops the backend issues per layer around the kernel
([Phase 7](history/06-phase-7-paged-decode.md)). It pays where prefill
dominates, because prefill goes to the dense kernel
([Against the engine](history/07-against-the-engine.md)).

Two debugging switches are read by the code: `HK_SPLITS` (`off`, `0` or `1`
forces one split; `auto` or `plan`, the default, uses the planner; an integer
forces that many, capped by the page count and 32), and `HK_NO_PREFILL` (any
non-empty value sends the backend's prefill to
`F.scaled_dot_product_attention`). The comment on `HK_SPLITS` records an open
fault: under vLLM at batch 16 and context 4096 the second `generate` aborted
with `HSA_STATUS_ERROR_EXCEPTION 0x1016`, and forcing one split made it go
away. That comment calls the split opt-in; the code's default is `auto`.

## The linear-layer patch

On gfx1100 every one of vLLM's unquantized GEMMs ends in
`torch.nn.functional.linear`, i.e. hipBLASLt; the installed vLLM's skinny-GEMM
path is gated on gfx9, and correctly ([the linear layers](history/08-linear-layers.md)).
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

Tables are per arch and per model. `hk/tuned/routes_vllm_linear_gfx1100.json`
is Qwen3-8B's, raw timings included, and `$HK_VLLM_ROUTES` names another. A
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
which a graph captured per batch size fixes. The attention backend is
`graph="never"`: the decode split count is planned on the host from
`max_seq_len`, the first call per layer checks the metadata with a device
read, and scratch regrown after capture would leave the graph writing to freed
memory, so vLLM captures the model piecewise around it and runs this attention
eagerly between the pieces (`vllm_backend.PATCH`).

Not ported: the two SDPA rebinding patches (`vllm.py`, `sglang.py`) still
keep their own counters.

## SGLang

```python
import sglang                      # let SGLang import its modules first
import hk.integration.sglang as hks
print(hks.apply())                 # before the engine is built
```

Not verified inside an SGLang server: there was no SGLang build for RDNA3 on
the machine this was developed on. `tests/hk/gpu/test_sdpa.py` verifies the
drop-in itself; the module's reading of SGLang's internals is what is
untested. SGLang's RDNA3 path normally selects its Triton attention backend,
which does not call `F.scaled_dot_product_attention`, so on a text model the
patch reports that it rebound nothing; it bites with
`--attention-backend torch_native` and in the vision encoders of multimodal
models (`python/hk/integration/sglang.py`).

## Ahead-of-time builds

`python3 -m hk.integration --warm -j 32` fills the cache for the torch half
specifically: libtorch's include and ABI flags are part of the cache key, so a
`--aot` pass built for pybind does not spare a server its first attention
compile. It builds the six distinct attention registrations in 17.8 s, which
is the number that matters for a server's first request. Run it and
`python3 -m hk.autotune --aot` both ([warming the cache](runtime.md#warming-the-cache)).

## What is deliberately not patched

Two things are deliberately *not* patched, and the reason is the same both
times -- the patch would be slower than what it replaces:

* **RMSNorm.** Every framework's RMSNorm applies a learned weight;
  `hk.ops.rmsnorm` is unweighted, so the patch would add an elementwise pass to
  save one.
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves and `hk.ops.silu_mul` has no stride support, so the
  patch would add two copies to save one pass.

The stride half of that was built later and the patches were measured before
being written: under torch.compile both ops come from Inductor, fused with
their neighbours, and the kernels lose to it, so both integrations were
dropped ([strided globals](history/12-strided-globals.md)).

## Environment variables

Read by `python/hk/integration` and the kernels it routes to, exactly as
spelled:

| variable | read by | effect |
|---|---|---|
| `HK_PATCH_VLLM` | `_vllm_plugin.py` | the consent: without it the plugin does nothing. With it, the SDPA patch is applied |
| `HK_VLLM_BACKEND` | `_vllm_plugin.py` | register the decoder attention backend; a value other than `1`/`on`/`yes`/`true` names the slot |
| `HK_VLLM_BACKEND_SLOT` | `vllm_backend.py` | the `AttentionBackendEnum` member to register over (default `TRITON_ATTN`) |
| `HK_VLLM_GEMM` | `_vllm_plugin.py` | route the unquantized linear layers |
| `HK_VLLM_ROUTES` | `vllm_gemm.py`, via `engine.shipped_table` | a routing table to use instead of the shipped one |
| `HK_SPLITS` | `ops/paged.py` | the decode backend's KV split: `auto` (default), `off`/`0`/`1`, or a count |
| `HK_NO_PREFILL` | `vllm_backend.py` | non-empty: the backend's prefill goes to torch SDPA |

`HK_PATCH_VLLM`, `HK_VLLM_BACKEND` and `HK_VLLM_GEMM` are off when unset or
`0`, `off`, `no` or empty. The benchmark harnesses in `tools/hk-bench/` read
more (`HK_GPU_UTIL`, `HK_EAGER`, ...); those are theirs, not the library's.
