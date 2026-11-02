"""The `sglang.srt.plugins` entry point.

SGLang runs the model in a scheduler process the launcher spawns, so a patch
made in the process that ran `launch_server` never reaches it. SGLang loads
its general plugins in the launcher and again in every scheduler process
(sglang/srt/managers/scheduler.py), which is where this belongs -- the same
reason the vLLM patch is a vLLM general plugin.

**On by default**, on the GPU the routes were validated on (gfx1100), one switch per route
(`hk.integration.switches`):
* `HK_SGLANG_LINEAR` routes SGLang's unquantized linear layers through
  hk's GEMM op (`sglang.apply_linear`);
* `HK_SGLANG_MOE` replaces the softmax top-k that fails on gfx1100 and
  routes the bf16 expert GEMMs through hk's (`sglang_moe.apply`);
  `HK_SGLANG_MOE=topk` replaces only the top-k (`sglang_moe.apply_topk`).
* `HK_SGLANG_NORM` runs SGLang's Gemma-style RMSNorm on hk's weighted norm
  kernels where SGLang would take its native torch path (`sglang.apply_norm`).

    python -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
        --attention-backend triton --disable-cuda-graph          # hk's routes on
    HK_DISABLE=1 python -m sglang.launch_server ...               # stock SGLang
    HK_SGLANG_MOE=0 python -m sglang.launch_server ...            # one route off

If SGLANG_PLUGINS is set, it has to name `hk` for SGLang to load this at all.
Before patching, the plugin checks the engine contract of each route it is about to apply
(`hk.integration.contract`). A route whose contract is broken, or whose apply raises, stays off
with one warning line of its own; the others are applied.
`load_plugins()` runs once per process, and each patch reports "already
patched" if it is reached twice anyway.
"""

from __future__ import annotations

import os
from typing import Tuple


def enabled() -> bool:
    """Whether any SGLang route is on (hk.integration.switches)."""
    from . import switches  # noqa: PLC0415

    return bool(switches.on("sglang"))


def register() -> None:
    """The entry point SGLang calls. Never raises: a plugin that throws takes
    the server's startup with it, and a layer that did not get patched is a
    slower server, not a broken one."""
    from . import contract, switches  # noqa: PLC0415

    pid = os.getpid()
    if switches.disabled():
        print(f"[hk] SGLang: off ({switches.DISABLE}=1), stock (pid {pid})", flush=True)
        return
    routes = switches.on("sglang")
    if not routes:
        return
    from . import sglang as hks  # noqa: PLC0415

    def moe() -> str:
        from . import sglang_moe  # noqa: PLC0415

        topk_only = routes["sglang_moe"].value.lower() == "topk"
        return sglang_moe.apply_topk() if topk_only else sglang_moe.apply()

    # Each route's contract, then each route that holds, one at a time: a broken or raising
    # route is off with its own warning line, and the others are applied (contract.py).
    notes, applied, off = contract.apply_routes(
        "sglang", routes,
        [("sglang_linear", hks.apply_linear), ("sglang_norm", hks.apply_norm), ("sglang_moe", moe)])
    if not applied:
        print(f"[hk] SGLang: no route applied ({len(off)} off, see the warnings), stock "
              f"(pid {pid})", flush=True)
        return
    from ..runtime.compile import arm_jit_notice  # noqa: PLC0415

    arm_jit_notice("SGLang plugin")
    on = ", ".join(f"{n} ({d.why})" for n, d in routes.items() if n in applied)
    left = f"; OFF: {', '.join(off)}" if off else ""
    print(f"[hk] SGLang routes on: {on}{left}; {'; '.join(notes)} (pid {pid})", flush=True)


__all__: Tuple[str, ...] = ("enabled", "register")
