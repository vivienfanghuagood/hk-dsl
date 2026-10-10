"""The `vllm.general_plugins` entry point.

Why this file exists, and it is not a convenience. `vllm.LLM` with V1 runs the
model in a **separate process** (`EngineCore_DP0`, spawned), so patching
`torch.nn.functional` in the process that called `LLM(...)` patches a process
the model never runs in. Measured: a full InternVL2-2B request with
`hk.integration.vllm.apply()` called first reported `{'kernel': 0,
'fallback': 0}` -- the patched function was never entered, the answers were
byte-identical to unpatched, and the end-to-end time was 1.001x. Nothing about
that failure is visible without a counter; it looks exactly like a kernel that
is the same speed as torch.

vLLM's own answer to "run this in every process" is a general plugin, which it
loads in the driver and in every worker. So that is where the patch belongs.

**Opt-in, by environment.** Installing a kernel library must not silently
change the numerics of every vLLM on the machine; `HK_PATCH_VLLM=1` is the
consent. The plugin is otherwise a no-op that costs an entry-point load.

    HK_PATCH_VLLM=1 vllm serve OpenGVLab/InternVL2-2B \\
        --mm-encoder-attn-backend TORCH_SDPA

`load_general_plugins` warns that plugins may be loaded more than once in one
process, so `register()` is idempotent: `apply()` is guarded and `sdpa.patch()`
refuses to capture itself as the original.
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("HK_PATCH_VLLM", "0") not in ("0", "", "off", "no")


def register() -> None:
    """The entry point vLLM calls. Never raises: a plugin that throws takes
    the engine's startup with it, and a kernel that did not get patched is a
    slower server, not a broken one."""
    if not enabled():
        return
    try:
        from . import vllm as hkv  # noqa: PLC0415

        report = hkv.apply()
    except BaseException as e:  # noqa: BLE001
        print(f"[hk] vllm plugin failed, leaving torch attention in place: "
              f"{type(e).__name__}: {e}", flush=True)
        return
    print(f"[hk] {report} (pid {os.getpid()})", flush=True)
