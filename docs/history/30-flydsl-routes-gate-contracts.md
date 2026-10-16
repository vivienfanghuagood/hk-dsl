# FlyDSL on the routes, a performance gate, and the engines' contracts

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Three pieces of plumbing around the kernels, each the answer to a way the
project could be wrong without noticing:
- the integration layer only knew hk's kernels and the engine's, while
  [chapter 26](26-the-scoreboard-retuned.md) had FlyDSL's GEMM ahead of hk's;
- nothing caught a kernel getting slower, because absolute times on the
  shared node drift more than most regressions;
- every patch rebinds engine internals by name, and nothing caught an engine
  release moving one.

## The engines' contracts

The patches rebind, by name or by identity, about twenty symbols across vLLM
and SGLang. An engine release that renames one, reorders its parameters or
moves the call that reaches it breaks a patch quietly:
- the rebind finds nothing, and the patch reports "0 bindings rebound" in a
  log line nobody reads;
- or the stand-in forwards its arguments positionally into the wrong slots;
- or a backend hook stops being read, and becomes dead code.

`tests/hk/engines/` asks the installed engine. It is a tier of its own,
skipped where the engine is not importable. Three kinds of check:
- **Signatures, as the stand-ins depend on them.** A stand-in that forwards
  positionally must take the engine function's parameters in the same order,
  with the same defaults. That covers the two MoE entry points in each
  engine, SGLang's `topk_softmax`, the AWQ method and its two ops,
  `reshape_and_cache_flash`, and the `(layer, x, weight, bias)` that
  `dispatch_unquantized_gemm()` hands back.
- **Call sites.** `_contract.call_sites` disassembles every function and
  method of every loaded engine module, and lists those that load the
  patched name as a global. Each must hold the original object, or a rebind
  by identity misses it. For `dispatch_unquantized_gemm` on vLLM 0.31 that
  is `linear` (now in `UnquantizedLinearMethod.__init__`, which stores the
  result as `_gemm_impl`, so the patch must land before the model is built,
  as the plugin does) and `vocab_parallel_embedding` (the LM head, per call).
- **Behaviour that only a call shows.** hk's `triton_decode` falls back to
  hk's walk, silently, when vLLM's `unified_attention` lacks an argument it
  passes. The test drives it with CPU tensors and a capturing stand-in for
  the kernel, and fails on that fallback. SGLang's
  `try_get_optimal_moe_config(return_down_config=True)` is called to check
  the `(config, (down, max_block_m))` shape the stand-in unpacks. Its tuned
  file lookup needs a running server's context, so the test goes through
  SGLang's own override hook (`triton_utils.get_config`).

Run on the dev pod:

| venv | engine | passed |
|---|---|---|
| `venv-vllm031` | vLLM 0.31.0+rocm721 | 19 of 19 (SGLang's file skipped) |
| `venv-sgl` | SGLang 0.5.18, sglang-radeon 0.1.5 | 8 of 8 |
| `venv-sgl` | vLLM 0.16.1, installed beside SGLang | 18 of 19 |

The one failure on 0.16.1 is real: it has no `quantization.auto_awq`, so the
4-bit patch, written against 0.31, cannot apply there.

The first run also found something about 0.31 itself. Four of the attention
backend's overrides are not on vLLM 0.31's `AttentionBackend` at all, and
grep confirms 0.31 reads none of them off a backend:
- `accept_output_buffer`: forward's output buffer is now always passed;
- `get_kv_cache_shape`: the layout comes from `supported_kv_cache_layouts`;
- the backend's `use_cascade_attention`: vLLM asks the metadata builder;
- `validate_head_size`: hk's own, called by hk.

None of this breaks anything, since the hooks 0.31 does read are all there.
But a test that let them pass as live would let the next rename pass too. So
every override is now classified as read by vLLM, as one of the two layout
hooks (`supported_kv_cache_layouts` from 0.31, `get_kv_cache_shape`
before), as legacy, or as hk's own. An unclassified override fails the test.
So does a hook in the first group missing from the base class, or neither
layout hook being there. The run also pinned one assumption nothing had
checked: `forward_includes_kv_cache_update` is True on 0.31's base. hk's
forward writes K/V itself, so a backend that said otherwise would get a
second write from vLLM.

SGLang needs a GPU visible to import its kernels at all: `fp8_kernel` asks
device 0 for its arch at import time. Its tests launch nothing, but they
cannot run with the GPUs hidden.

The supported-version matrix is in
[integration](../integration.md#supported-versions-and-the-contract-tests).
