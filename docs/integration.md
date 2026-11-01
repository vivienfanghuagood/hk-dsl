# Using hk inside vLLM and SGLang

`hk.integration` patches a running vLLM or SGLang so that some of its calls
run on generated kernels. **hk is on by default**: installed, its engine
plugins apply every route validated with real weights on your GPU
([on by default](#on-by-default-and-turning-it-off)); `HK_DISABLE=1` turns
all of it off. Every patch reports what it took and what it left alone, and
falls back to what the engine would have run for any call it cannot serve.
The code is `python/hk/integration/`, and serving results are on
[the scorecard](performance/scorecard.md).

This page is what applies to every engine. Each engine's routes, one section
a route, are on their own pages: [vLLM](vllm.md) and
[SGLang and SGLang-Diffusion](sglang.md). Adding a kernel of your own:
[your own kernel, in four steps](#your-own-kernel-in-four-steps), then
[the end-to-end loop](e2e-loop.md) and [the agent tools](agents.md).

- [On by default, and turning it off](#on-by-default-and-turning-it-off)
- [The patches](#the-patches)
- [The plugins](#the-plugins)
- [The engine layer: `Op`, `Patch`, `RouteTable`](#the-engine-layer-op-patch-routetable)
- [Host time per call](#host-time-per-call)
- [Routing tables](#routing-tables)
- [The CUDA-graph contract](#the-cuda-graph-contract)
- [Ahead-of-time builds](#ahead-of-time-builds)
- [Supported versions, and the contract tests](#supported-versions-and-the-contract-tests)
- [What is deliberately not patched](#what-is-deliberately-not-patched)
- [Your own kernel, in four steps](#your-own-kernel-in-four-steps)
- [Environment variables](#environment-variables)

## On by default, and turning it off

In hk's toolchain hk is on by default. Turning it off is for comparison
experiments. When vLLM or SGLang loads hk's plugin, the plugin applies every
**route** (one patch of one engine) that is on in `hk.integration.switches`.
A route is on by default only if it meets two conditions:
- it was validated with real weights on that GPU;
- its outputs were bit-identical to the engine's, or its error was within
  the engine's own run-to-run noise.

Lossy routes stay off, and so do routes without that evidence.
`python3 -m hk.integration --status` prints every route, whether it is on,
and which rule decided.

| route | switch | default | evidence |
|---|---|---|---|
| MXFP8 linear (DeepSeek-V4.1), M >= 2048 (indexer wq_b >= 2825, engram wkv >= 1024: `MIN_M_NK`), and measured M ranges below that (`RANGES`) | `HK_VLLM_MXFP8` | **on, gfx950** | bit-identical to vLLM's kernel; real weights, GSM8K 94.69% vs stock 94.24-94.39%, with the a8w4 MoE route ARC 96.76% vs 96.93-97.10% (McNemar p >= 0.175; [serving readiness](design/serving-readiness.md) B1) |
| MoE a8w4, MXFP4 weights (DeepSeek-V4.1), M >= 2560 (`MIN_M`) | `HK_VLLM_MOE_A8W4` | **on, gfx950** | error vs fp32 3.5-6.3e-3 against AITER's 0.97-2.1e-2 on real layers; GSM8K 94.92% (B1) |
| DeepSeek-V4.1 index layers under `--monokernel`: one launch for the compressor, indexer K / Q, logits and top-k, at decode steps of up to 8 rows | `HK_VLLM_DSV41_INDEX` | **on, gfx950** | caches and q byte-identical to vLLM's chain in vLLM, top-k sets equal ([the index launch](vllm.md#the-deepseek-v41-index-launch)); with `--monokernel` and hk's other default routes, real-weight GSM8K-1319 within stock's noise (McNemar p >= 0.39) |
| AWQ 4-bit linear | `HK_VLLM_INT4` | **on, gfx1100, gfx950 and gfx1201** | greedy tokens identical to stock's, Qwen3-8B/14B-AWQ; on gfx1201 Qwen3-8B-AWQ tokens identical |
| GPTQ 4-bit linear (symmetric, grouped; RDNA3W4A16LinearKernel) | `HK_VLLM_GPTQ` | **on, gfx1100** | Qwen3.8-27B-GPTQ-Int4 against stock-fair (TRITON_ATTN, TunableOp with a 256 MB rotating buffer): GSM8K-200 equal, TPOT in the RDNA scorecard; act-order checkpoints are refused and counted (`vllm_gptq.REFUSED`) |
| unquantized linear (+ tuning at load) | `HK_VLLM_GEMM` | **on, gfx1100 and gfx1201** | tokens identical, or a bf16 near-tie; on gfx1201 tokens identical and routing counts equal round for round, Qwen3-1.7B and Qwen3.5-9B, also with gfx1201's shipped small-M schedules; on gfx950 it was measured to change nothing against aiter, so it stays off there |
| bf16 MoE GEMMs | `HK_VLLM_MOE` | **on, gfx1100 and gfx1201** | OLMoE tokens identical; with aiter on gfx950 it routes nothing; on gfx1201 up to 2 tokens a step only, OLMoE tokens identical |
| encoder SDPA (`--mm-encoder-attn-backend TORCH_SDPA`) | `HK_VLLM_SDPA` | **on, gfx1100** | InternVL2-2B inside the spread of vLLM's own backends; on gfx1201 the drop-in's per-call Python makes the call slower than the R9700 image's torch SDPA, so it stays off there |
| SGLang linear | `HK_SGLANG_LINEAR` | **on, gfx1100** | Qwen3-8B up to a bf16 near-tie; error equal to F.linear's |
| SGLang MoE (and the gfx1100 top-k fix) | `HK_SGLANG_MOE` | **on, gfx1100** | OLMoE tokens identical |
| SGLang-Diffusion attention | `HK_SGLD_ATTN` | **on, gfx1100** | Qwen-Image 2.1, 42.1 dB from stock and closer to an fp32 reference than stock |
| SGLang-Diffusion linear | `HK_SGLD_LINEAR` | **on, gfx1100** | 40.0 dB from stock, closer to fp32 than stock |
| DiT attention in `hk.ops.sdpa` (non-causal, head_dim 128, on `hk.ops.attn_k3`) | `HK_SDPA_K3` | **on, gfx1100** | per-call error 2.95e-5 against the current path's 4.50e-5 (#24) |
| decoder attention backend | `HK_VLLM_BACKEND` | off | replaces ROCm's default decoder attention for every model, and raises rather than falls back on sliding windows, sinks, softcap, an fp8 KV cache, fp16 or other head sizes; validated on Qwen3-8B only |
| prefill attention on vLLM's own cache | `HK_VLLM_PREFILL` | off: until its end-to-end gates pass | ROCM_ATTN's prefix-prefill and TRITON_ATTN's batches with prefill in them, in place on each backend's cache; decode-only steps, cache write and block size stay vLLM's; any call it does not take falls back. In-situ against vLLM's kernels on every call (`HK_VLLM_PREFILL_SHADOW=1`, MiniCPM5-2B, about 22,000 calls on each backend): at most 2 bf16 ulps of a row's max |
| ConvRot INT8 (`--quantization convrot_int8`) | `HK_SGLD_CONVROT`, or `HK_SGLD=...,convrot` | **off: opt-in** | lossy by design (PSNR 30.4 dB from bf16, visible drift): a model-level choice, not a toolchain default |
| MXFP8 at any (N, K) | `HK_MXFP8_ANY` | off | a mode of the MXFP8 linear route; no real-weight run |
| DeepSeek-V4.1 sparse MLA prefill, calls of T >= 512 queries (`MIN_T`), and the layer's MXFP8 rows for wo_a made in the kernel from O in registers. It runs where vLLM's own sparse MLA does: with `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` vLLM takes AITER's path for prefill and decode and the route stands aside | `HK_VLLM_SPARSE_MLA` (`=0` turns it off) | **on, gfx950** | real weights, TP2: GSM8K-1319 equal to stock's (pooled McNemar p 0.12). **Its cost is at decode:** the route needs vLLM's own sparse MLA, whose decode kernel replaces AITER's, so decode steps can be slightly slower than with AITER's (an open item). Serving results: [the scorecard](performance/scorecard.md) |

On a GPU a route was not validated on (gfx942 for every route, and gfx1201 for
the routes not marked for it above), it is off unless you turn it on.

**Turning it off.** The first rule that applies decides:

1. `HK_DISABLE=1` turns every route off, in every engine. Nothing overrides it.
2. An engine's own switch at `0` turns off all of that engine's routes:
   `HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0` or `HK_SGLD=0`.
3. A route's own switch decides that route: `HK_VLLM_MXFP8=0` turns the MXFP8 linear route off, and
   `=1` turns on a default-off route, or a route on a GPU it was not validated on.
4. The legacy opt-ins still mean what they meant. `HK_SGLD=attn,linear,convrot`
   turns on exactly the parts it names, and `HK_SGLD=1` turns on attention
   and linear. `HK_PATCH_VLLM=1` turns on the encoder SDPA patch.
   `HK_SGLANG_MOE=topk` replaces only the top-k. `HK_VLLM_BACKEND=<slot>`
   names the slot.
5. Otherwise the default in the table applies.

A value of `0`, `off`, `no` or `false` is off. An empty value counts as unset.

```bash
vllm serve <model>                       # hk on: the default routes
HK_DISABLE=1 vllm serve <model>          # stock vLLM: the plugin patches nothing
HK_VLLM_MOE_A8W4=0 vllm serve <model>    # every default route but the a8w4 MoE
```

**Comparison scripts must say stock.** Leaving the variables unset no longer
means stock, so an A/B's stock arm sets `HK_DISABLE=1`. `python -m
hk.integration ab`, `headroom` and `tools/hk-bench/` set it themselves, and
apply each arm's patches explicitly.

**The engine contract, at startup.** Before it patches anything, a plugin
checks every route it is about to apply against the installed engine
(`hk.integration.contract`). It checks that each patched or fallback entry
point still exists, with the parameters the route uses: the MXFP8 kernel
class's two methods, `aiter.fused_moe.fused_moe`'s parameters, the AWQ
methods, and so on. If any contract is broken, for example by a vLLM or AITER
other than the one measured, the plugin patches **nothing** in that engine.
It prints one line to stderr, and the engine runs stock:

```
[hk] WARNING: vllm: engine contract FAILED -- hk patched NOTHING in this engine, it runs stock (vllm 0.x; routes measured on vllm 0.32.1rc1.dev36+ge68efc685): vllm_moe_a8w4: aiter.fused_moe.fused_moe lost ['gate_mode']. ...
```

It is all or nothing per engine, not per route. The routes in one engine patch
different entry points, but one broken contract means the engine is not the
build they were measured on. A half-patched engine was never measured, and
nobody chose it. A route that raises while it is being applied gets the same
treatment: the plugin undoes the routes already applied and prints the same
kind of line.

**The first JIT compile says so.** A route's kernels compile on first use,
seconds each, in every worker. Run `python3 -m hk.integration warm-routes`
(and `--warm`) with the server's `HK_CACHE_DIR` first. If you do not, the
first compile in each process prints one line:

```
[hk] JIT: compiling kernel '...' on first use (pid N): the vLLM plugin's routes were not pre-warmed into ..., so this call waits for hipcc (seconds a kernel, in every worker). Run python3 -m hk.integration warm-routes (and --warm for attention) with the server's HK_CACHE_DIR before starting the server.
```

## The patches

| patch | module | what it replaces | switch (default) |
|---|---|---|---|
| SDPA drop-in | `vllm.py`, `sglang.py`, `_common.py` | `F.scaled_dot_product_attention`, rebound by identity | `HK_VLLM_SDPA` (on, gfx1100); `apply()` in-process |
| decoder attention backend | `vllm_backend.py` | vLLM's `TRITON_ATTN` backend slot | `HK_VLLM_BACKEND=1` (off) |
| prefill attention | `vllm_prefill.py` | `prefix_prefill.context_attention_fwd` (ROCM_ATTN) with its parameters, and `triton_unified_attention.unified_attention` (TRITON_ATTN) with its signature, rebound by identity | `HK_VLLM_PREFILL` (off) |
| linear layers | `vllm_gemm.py` | `dispatch_unquantized_gemm`, rebound by identity | `HK_VLLM_GEMM` (on, gfx1100 and gfx1201) |
| MoE GEMMs | `vllm_moe.py` | `invoke_fused_moe_triton_kernel` and `try_get_optimal_moe_config`, rebound by identity | `HK_VLLM_MOE` (on, gfx1100 and gfx1201) |
| AWQ linear layers | `vllm_int4.py` | `AutoAWQLinearMethod.process_weights_after_loading` and `.apply` | `HK_VLLM_INT4` (on) |
| GPTQ linear layers | `vllm_gptq.py` | `AutoGPTQLinearMethod.process_weights_after_loading` and `.apply` | `HK_VLLM_GPTQ` (on, gfx1100) |
| MXFP8 linear layers | `vllm_mxfp8.py` | `RocmDotScaledMxfp8LinearKernel.process_weights_after_loading` and `.apply_weights` | `HK_VLLM_MXFP8` (on, gfx950) |
| MXFP4 MoE, a8w4 | `vllm_moe_a8w4.py` | `aiter.fused_moe.fused_moe`, with its signature | `HK_VLLM_MOE_A8W4` (on, gfx950) |
| DeepSeek-V4.1 index layers, `--monokernel` | `vllm_dsv41_index.py` | `mono_decode._select`, through an import hook | `HK_VLLM_DSV41_INDEX` (on, gfx950) |
| sparse MLA prefill (DeepSeek-V4.1) | `vllm_sparse_mla.py` | `rocm_sparse_attn_prefill` and `rocm_inverse_rope_mxfp8_rows`, rebound by identity, and `DeepseekV41ROCMAiterMLAAttention._forward_prefill`, with their signatures | `HK_VLLM_SPARSE_MLA` (on, gfx950) |
| your kernels | `register.py` | whatever `register.patch` names | `HK_VLLM_USER=<modules or files>` |
| SGLang linear layers | `sglang.py`, `_sglang_plugin.py` | `UnquantizedLinearMethod.apply` | `HK_SGLANG_LINEAR` (on, gfx1100); `apply_linear()` in-process |
| SGLang MoE layers | `sglang_moe.py`, `_sglang_plugin.py` | `topk_softmax`, `invoke_fused_moe_kernel` and `try_get_optimal_moe_config`, rebound by identity | `HK_SGLANG_MOE` (on, gfx1100), or `=topk` for the routing alone |
| SGLang-Diffusion attention, linear, ConvRot | `sglang_diffusion.py` | see [SGLang-Diffusion](sglang.md#sglang-diffusion-text-to-image-and-video) | `HK_SGLD_ATTN`, `HK_SGLD_LINEAR` (on, gfx1100), `HK_SGLD_CONVROT` (off) |

`engine.py` is what every patch but the two SDPA rebinds shares: the custom
op, the counters, the routing table and the CUDA-graph contract.

## The plugins

**In vLLM, through its plugin entry point.** The package registers
`hk_attention = "hk.integration._vllm_plugin:register"` under
`vllm.general_plugins` (`pyproject.toml`), which vLLM loads in the driver
*and* every worker. That is the only place a patch reaches the model:

* **vLLM V1 runs the model in a spawned process.** `apply()` in the process
  that built `LLM(...)` patches a process the model never runs in. The first
  end-to-end run reported `{'kernel': 0, 'fallback': 0}` with byte-identical
  answers and no speedup. Production therefore needs the
  `vllm.general_plugins` entry point (`hk/integration/_vllm_plugin.py`), which
  vLLM loads in the driver *and* every worker. It applies the routes that
  are on ([on by default](#on-by-default-and-turning-it-off)), after
  checking their engine contract.

The decoder backend stays off by default: it registers over the slot ROCm
picks by default, so it replaces the decoder attention of every model.
The invocations, as the module docstrings give them:

```bash
# the vision tower's attention (SDPA drop-in; on by default on gfx1100)
vllm serve OpenGVLab/InternVL2-2B --mm-encoder-attn-backend TORCH_SDPA

# the linear and MoE layers: on by default on gfx1100
vllm serve Qwen/Qwen3-8B
vllm serve allenai/OLMoE-1B-7B-0125-Instruct

# DeepSeek-V4.1's MXFP8 linear layers and a8w4 MoE: on by default on gfx950 (warm their kernels first)
HK_CACHE_DIR=/cache/hk python3 -m hk.integration warm-routes --tp 2
HK_CACHE_DIR=/cache/hk vllm serve <DeepSeek-V4.1-Flash> --tensor-parallel-size 2
# ... and with dsv41_mono, the index launch too
HK_CACHE_DIR=/cache/hk vllm serve <DeepSeek-V4.1-Flash> --tensor-parallel-size 2 --monokernel

# the decoder attention backend (opt-in), with vLLM's defaults: CUDA graphs,
# prefix caching and chunked prefill all on
HK_VLLM_BACKEND=1 vllm serve <model> --attention-backend TRITON_ATTN

# stock vLLM, for comparison
HK_DISABLE=1 vllm serve <model>
```

The plugin prints one `[hk] ...` line per process. The line names the routes
that are on and why, and what each patch did.
The end-to-end measurements build the engine with
`attention_backend="TRITON_ATTN"` for the backend arm and vLLM's defaults
otherwise (`HK_VLLM_DEFAULTS=1`, read by `python3 -m hk.integration ab`,
which `tools/hk-bench/vllm_e2e.py` wraps; the backend asks for its own page
size, 32).

**In SGLang, through its plugin entry point.** The package also registers
`hk = "hk.integration._sglang_plugin:register"` under `sglang.srt.plugins`.
SGLang loads it in the launcher and in every scheduler process, which is where
its model runs. It prints the same `[hk] ...` line per process. Both routes
are on by default on gfx1100. `HK_SGLANG_LINEAR=0` or `HK_SGLANG_MOE=0`
turns one off, and `HK_DISABLE=1` turns both off.

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --attention-backend triton --disable-cuda-graph
python3 -m sglang.launch_server --model-path allenai/OLMoE-1B-7B-0125-Instruct \
    --attention-backend triton --disable-cuda-graph
HK_DISABLE=1 python3 -m sglang.launch_server ...       # stock SGLang
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
python3 -m hk.integration --status          # every route on or off and why; what is patched
python3 -m hk.integration --patch           # apply every available patch and report
python3 -m hk.integration --warm -j 32      # build the torch ops ahead of time
python3 -m hk.integration headroom ...      # a step's device time by kernel family, against what hk has
python3 -m hk.integration ab ... --json f   # end to end against stock, with verdicts
python3 -m hk.integration new-op NAME       # a new op's kernel, registration, tests and bench row
```

The last three are the loop [for agents](agents.md): find the gap, add the
kernel, prove it helped.

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

## Host time per call

An eager engine runs every route's Python on every call, and where the GPU
work of a step is short -- decode, a 4-bit model -- that Python sets the
step's pace. Most of it was deciding again what only the call's shapes
decide: the table lookup and the schedule (`vllm_gemm`), `takes` and the
schedule twice (`vllm_int4`), `inspect.signature` of AITER's `fused_moe` on
every MoE layer (`vllm_moe_a8w4`), and imports inside function bodies. Calls hk does not
take paid it too.

So each route keeps its decision per shape, and the kernels behind it launch
from a per-shape memo of their grid and argument without the torch
dispatcher ([the runtime](runtime.md#torch-ops-and-the-sdpa-drop-in)). The
memos are bounded; `vllm_gemm`'s is dropped whenever its routing table
changes. `HK_HOST_FASTPATH=0` turns them off: every call then takes the path
it took before, line for line. Outputs are bitwise the same either way.

Host time of one engine call, µs, same process, the switch flipped between
rounds (`hk.runtime.hostpath.set_enabled`):

| route, shape | GPU | before | now | the engine's own path |
|---|---|---|---|---|
| `vllm_gemm`, decode (M 16, hk_decode) | W7900D | 38.5 | 14.0 | 17.9 |
| `vllm_gemm`, prefill (M 2048, hk) | W7900D | 46.5 | 17.6 | 23.9 |
| `vllm_gemm`, a call it leaves to vLLM (M 1) | W7900D | 41.2 | 25.1 | 18.0 |
| `vllm_int4` (M 1) | W7900D | 33.6 | 12.5 | -- |
| `vllm_int4` (M 1) | MI355X | 19.9 | 9.4 | 33.6 |
| head_dim-128 attention (`hk.ops.sdpa` drop-in, B1 H16 N256) | W7900D | 82.5 | 49.2 | 11.6 |
| `vllm_moe_a8w4` (M 8192) | MI355X | 167.1 | 122.6 | 210.7 |
| `vllm_moe_a8w4`, a call it leaves to AITER (M 1024) | MI355X | 252.6 | 208.6 | 202.7 |
| `vllm_mxfp8` (M 2048) | MI355X | 30.8 | 27.0 | 54.4 |

The engine's own path for a call hk leaves to it is the engine's function
called directly; inside the engine it sits behind a custom op of its own,
which costs about what hk's does.

Serving results, end to end, are on [the scorecard](performance/scorecard.md).

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
an earlier run's table keeps only the overlay points both runs took. The table answers
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
returns one shape's entry, and `route_tune.py`, `python3 -m hk.integration
tune` and the [measuring at load](vllm.md#a-model-no-table-has-measured) are thin
shells over it. `engine.shipped_table` lays this machine's records over the
shipped table, shape by shape. A record measured against another engine's
path (vLLM upgraded, say) is left out, and measured again at the next load.

Tables are per arch, per model and per engine, because the path a kernel has
to beat is the engine's. The shipped ones, in `hk/tuned/`, named
`routes_<engine>_<kind>_<arch>.json`:
- `routes_vllm_linear_gfx1100.json`: Qwen3-8B against vLLM 0.31's op, raw
  timings included, with FlyDSL rows (below), and the production W7900D
  models' shapes -- MiniCPM5-2B's and MiniCPM-V-4.7-35B-A3B's per TP2 rank --
  measured in their TheRock ROCm 7.14 image against its vLLM's op, each row
  with its own `meta` (engine, toolchain, model). Those models then start
  without measuring at load. M = 1 stays on vLLM's skinny GEMM in those rows:
  the margin there is a few percent either way and moved between images.
  `$HK_VLLM_ROUTES` names another.
- `routes_sglang_linear_gfx1100.json` (`route_tune.py --engine sglang`):
  against the `F.linear` SGLang calls on ROCm, with FlyDSL rows.
  `$HK_SGLANG_ROUTES` names another. It disagrees with vLLM's below M = 4:
  vLLM runs its own skinny GEMMs there and ties hk's GEMV, while `F.linear`
  is slower than it.
- `routes_vllm_linear_gfx950.json`: on an MI355X, against hipBLASLt, with
  HipKittens' CDNA4 GEMM as the candidate. It takes 37 of 206 points, all at
  M ≥ 1792 ([design note](design/cdna-hipkittens-ops.md)).

A table measured on another arch is refused, and an `$HK_VLLM_ROUTES` that
names a missing file raises (`engine.shipped_table`). There are no buckets:
GEMM time is a comb in M, not a curve, and a power-of-two bucket either
routes the teeth or gives up the gaps between them.

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
At startup each plugin runs `hk.integration.contract.check` on the routes it
is about to apply, and leaves the engine stock if one is broken
([on by default](#on-by-default-and-turning-it-off)).
`tests/hk/gpu/test_engine_contract.py` runs that check, and a simulated AITER
mismatch, in the vLLM image (the gfx950-vllm GPU tier).
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

What was run, on the dev pod (the vLLM pod, gfx1100), 2026-10-16:

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
(torch 2.13.0+rocm7.2, triton-rocm 3.7.1, Python 3.12)
and **SGLang 0.5.18** with sglang-radeon 0.1.5 (torch 2.9.1, ROCm 7.2.1).
vLLM 0.16.1 is the copy installed beside SGLang. Its contract holds except
for AWQ, but nothing has been run end to end on it since the port to 0.31.

**The Triton is part of the stack, and hk checks it.** vLLM 0.31's
requirements include xgrammar, which requires `triton` with no version;
triton-rocm, which torch 2.13.0+rocm7.2 pins, is a different distribution
name, so installing them after torch puts PyPI's newest `triton` over
triton-rocm's files in the same package. That is how a venv used for
measurements came to import Triton 3.8.0, with Triton's decode kernel slower
in vLLM's allocation and no error anywhere. So:
- `python3 -m hk.integration.stack` prints the vLLM, torch and Triton the
  venv has and exits 1 when they are not the measured stack (`hk.integration.stack.MATCHED`:
  for vLLM 0.31, torch 2.13.0+rocm7.2 and triton-rocm 3.7.1), or when two
  distributions install the `triton` package -- the overwrite's signature.
  It prints the commands that restore it: uninstall `triton`, then
  reinstall triton-rocm 3.7.1 with `--no-deps --force-reinstall`.
- `python3 -m hk.integration ab` refuses to run (exit 2) on a mismatched
  stack, and records the versions in its `--json` meta. The vLLM plugin
  prints the mismatch to stderr, framed, once a process,
  and carries on. `HK_STACK_CHECK` overrides either: `refuse`, `warn`, `off`.
- `tests/hk/engines/test_vllm_contract.py` fails in a venv whose Triton is
  not the one its vLLM was measured with.
- `scripts/engines/vllm031_setup.sh` builds the venv with the pin: after
  vLLM's requirements it removes whatever `triton` they brought and
  reinstalls the triton-rocm 3.7.1 wheel by its sha256, then checks the
  versions and stops if they are wrong.

A vLLM the table does not list is not checked: its stack has not been
measured. vLLM 0.31's own ROCm image builds yet another Triton, ROCm/triton's
3.8.x branch, and under 3.7.1 vLLM logs that its vendored `triton_kernels`
(MXFP4 MoE) cannot be imported; nothing hk patches uses them.

On the dev pod, `/workspace/venv-vllm031` is the matched venv: since
2026-10-23 it is a symlink to `/workspace/venv-vllm031-tri371`, and the
venv that had Triton 3.8.0 is kept, unused, as
`/workspace/venv-vllm031.retired-triton380-20261023`
(`/workspace/README-venvs.txt` in the pod).

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
  Whether hk's strip norms beat Inductor's fused ones inside vLLM's compiled
  graph has not been measured.
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves. A kernel's globals take such halves (strided
  globals), but the `hk.ops.silu_mul`
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
a root of its own, because the cache does not know what is patched into it.

**4. A/B it** in vLLM's own process. vLLM builds the model in the process
that loads its plugins, so the module that registers and patches is named
to the plugin, which imports it there:

```bash
HK_VLLM_USER=examples/11_vllm_integration.py \
    python3 -m hk.integration ab --model Qwen/Qwen3-8B --arms user \
    --context 1024 --batch 1,16 --short 16 --tokens 48
HK_VLLM_USER=my_kernels.py vllm serve Qwen/Qwen3-8B
```

`ab` runs stock and your arm alternately in fresh engines. It prints total,
prefill and decode-step time per shape, and whether the first request's
greedy tokens match stock's. `HK_VLLM_USER` takes comma-separated module
names or `.py` paths. `--json PATH` writes one row per shape, metric and arm
with a verdict, `win`, `loss`, `tie` or `noise`, against a band measured
from the rounds' own spread; `--out` keeps every round raw, and `--from`
judges such a file again ([agents](agents.md#prove-it-helps)).

`python3 -m hk.integration new-op NAME --dir DIR` writes the four steps' code
for a new op, as a working stub with an exact-answer test and a scoreboard
row, and `python3 -m hk.integration headroom` says which of the engine's
kernels are worth one: [the agents' page](agents.md) is the whole loop.

**Measure first.** The example's kernel is slower than the SiLU-and-mul vLLM
compiles with Inductor; it is there for the mechanics. A kernel that wins
only at some shapes should keep only those, through `route`.

## Environment variables

Read by `python/hk/integration` and the kernels it routes to, exactly as
spelled:

| variable | read by | effect |
|---|---|---|
| `HK_DISABLE` | `switches.py`, every plugin, the SGLang-Diffusion boot, `ops/sdpa.py` | `1`: every route off, in every engine; beats every other switch |
| `HK_PATCH_VLLM` | `switches.py`, `_vllm_plugin.py` | `0`: every vLLM route off (and `HK_VLLM_USER`). `1`: also the encoder SDPA patch, as it always applied |
| `HK_VLLM_SDPA` | `switches.py` | the encoder SDPA patch (default on, gfx1100) |
| `HK_VLLM_BACKEND` | `_vllm_plugin.py` | register the decoder attention backend (default off); a value other than `1`/`on`/`yes`/`true` names the slot |
| `HK_VLLM_BACKEND_SLOT` | `vllm_backend.py` | the `AttentionBackendEnum` member to register over (default `TRITON_ATTN`) |
| `HK_VLLM_PREFILL` | `_vllm_plugin.py` | prefill attention on vLLM's own cache, ROCM_ATTN and TRITON_ATTN (default off) |
| `HK_VLLM_PREFILL_SHADOW` | `vllm_prefill.py` | `1`: also run vLLM's kernel on every call hk takes and print the worst row's difference in bf16 ulps (a diagnostic; syncs) |
| `HK_VLLM_GEMM` | `_vllm_plugin.py` | route the unquantized linear layers (default on, gfx1100 and gfx1201) |
| `HK_VLLM_MOE` | `_vllm_plugin.py` | route the bf16 MoE GEMMs up to 256 tokens a step (2 on gfx1201; default on, gfx1100 and gfx1201) |
| `HK_VLLM_INT4` | `_vllm_plugin.py` | route AWQ linear layers' fp16/bf16 batches up to 64 tokens (default on) |
| `HK_VLLM_GPTQ` | `_vllm_plugin.py` | route 4-bit symmetric GPTQ layers' fp16/bf16 batches of 4-64 tokens (default on, gfx1100) |
| `HK_VLLM_MXFP8` | `_vllm_plugin.py` | DeepSeek-V4.1's block-scaled MXFP8 linear layers at M >= 2048, two shapes from their own M (`vllm_mxfp8.MIN_M_NK`), six with measured M ranges below that (`vllm_mxfp8.RANGES`) (default on, gfx950) |
| `HK_MXFP8_ANY` | `vllm_mxfp8.py` | `1`: the MXFP8 route takes any (N, K) of whole tiles, not only the measured TP2 shapes (default off) |
| `HK_VLLM_MOE_A8W4` | `_vllm_plugin.py` | DeepSeek-V4.1's AITER a8w4 MoE at M >= `HK_MOE_A8W4_MIN_M` (2560) (default on, gfx950) |
| `HK_VLLM_DSV41_INDEX` | `_vllm_plugin.py` | DeepSeek-V4.1's index layers under `--monokernel` as one hk launch at decode steps of up to 8 rows (default on, gfx950) |
| `HK_DSV41_INDEX_TOPK` | `vllm_dsv41_index.py` | `0`: the index launch stops at the logits' inputs; vLLM's indexer op does the logits and top-k (default on) |
| `HK_DSV41_INDEX_CHECK` | `vllm_dsv41_index.py` | `N`: on the first N eager calls of each index layer, run vLLM's chain after the launch and print what differs (default 0) |
| `HK_VLLM_USER` | `_vllm_plugin.py` | comma-separated modules or `.py` files to import in vLLM's process: your `register.op` / `register.patch` calls |
| `HK_VLLM_ROUTES` | `vllm_gemm.py`, via `engine.shipped_table` | a routing table to use instead of the shipped one and this machine's records; nothing is measured at load against it |
| `HK_AUTOTUNE_BUDGET` | `vllm_autotune.py` | seconds the linear patch may spend at load measuring the model's shapes no table has (default 60); `0` measures nothing, `inf` without a limit |
| `HK_AUTOTUNE` | `hk.autotune`, `engine.shipped_table`, `vllm_autotune.py` | `0`: ignore this machine's records -- schedules and routes -- and measure nothing at load |
| `HK_TUNE_DIR` | `hk.autotune` | where this machine's records go (default `$HK_CACHE_DIR/autotune`): `routes_<engine>_linear_<arch>.json` for routes |
| `HK_FLYDSL` | `flydsl_gemm.py` | `0`/`off`/`no`/`false`: never take a table's FlyDSL rows, even with FlyDSL installed |
| `FLYDSL_ROOT` | `flydsl_gemm.py` | a FlyDSL checkout whose `kernels/` package matches the installed `flydsl`, appended to `sys.path`; where another `kernels` is installed, imported from beside it |
| `HK_PATCH_SGLANG` | `switches.py` | `0`: every SGLang route off |
| `HK_SGLANG_LINEAR` | `_sglang_plugin.py` | route SGLang's unquantized linear layers (default on, gfx1100) |
| `HK_SGLANG_ROUTES` | `sglang.py`, via `engine.shipped_table` | a routing table for SGLang's linear layers instead of the shipped one |
| `HK_SGLANG_MOE` | `_sglang_plugin.py` | `1` (default on, gfx1100): the torch top-k and hk's expert GEMMs; `topk`: the torch top-k alone |
| `HK_SGLD` | `sglang_diffusion.py` and its boot | `0`: every SGLang-Diffusion route off; a list (`attn,linear,convrot`) names exactly the parts; `1`: attention and linear |
| `HK_SGLD_ATTN`, `HK_SGLD_LINEAR` | `switches.py` | SGLang-Diffusion's attention and linear (default on, gfx1100) |
| `HK_SGLD_CONVROT` | `switches.py` | ConvRot INT8, lossy (default off: opt-in) |
| `HK_SDPA_K3` | `ops/sdpa.py` | `hk.ops.sdpa`'s non-causal head_dim-128 calls on `attn_k3` (default on, gfx1100) |
| `HK_SPLITS` | `ops/paged.py` | the decode backend's KV split: `auto` (default), `off`/`0`/`1`, or a count |
| `HK_TRITON_DECODE_ALSO` | `vllm_backend.py` | decode batch sizes past `TRITON_DECODE_MAX` that run Triton's kernel on hk's cache, as ranges: `17-24,49-56` (default none) |
| `HK_NORM_LAYOUT` | `ops/norm.py` | `strip` (default): the norms and softmax run on the strip kernel wherever a row fits one; `tile`: always on the tile kernels |
| `HK_NORM_FOLD` | `ops/norm.py` | the tile kernels' plan for the weighted norms: `auto` (default), `on`, `off`. Reached only for rows too wide for a strip (over 32768 elements) or under `HK_NORM_LAYOUT=tile` |
| `HK_HIPKITTENS_KERNELS` | `ops/hipkittens.py` | HipKittens' `kernels/cdna4` to build the wrapped CDNA4 kernels from (default: the submodule's) |
| `HK_VLLM_DEFAULTS` | `ab.py` | `1`: the A/B builds vLLM with prefix caching and chunked prefill on, as vLLM ships |
| `HK_EAGER` | `ab.py` | `1` (default): the A/B runs vLLM eager; `0`: with CUDA graphs |
| `HK_GPU_UTIL` | `ab.py` | the A/B's `gpu_memory_utilization` (default 0.90) |
| `HK_HOST_FASTPATH` | `runtime/hostpath.py`, every route | `0`: no per-shape memos and no launches without the dispatcher; every call as before them ([host time](#host-time-per-call)) |
| `HK_STACK_CHECK` | `stack.py`, from `ab.py` and `_vllm_plugin.py` | what to do when vLLM's Triton or torch is not the stack it was measured on: `refuse` (`ab`'s default: exit 2), `warn` (the plugin's default: print to stderr and carry on), `off` |

A route's switch is off at `0`, `off`, `no` or `false`. Unset or empty, the
route's default applies. The precedence is in
[on by default](#on-by-default-and-turning-it-off). The benchmark harnesses in `tools/hk-bench/` read
more; those are theirs, not the library's. The runtime's own variables are in
[the runtime](runtime.md#environment-variables).
