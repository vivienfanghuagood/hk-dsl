"""The `sglang.srt.plugins` entry point.

SGLang runs the model in a scheduler process the launcher spawns, so a patch
made in the process that ran `launch_server` never reaches it. SGLang loads
its general plugins in the launcher and again in every scheduler process
(sglang/srt/managers/scheduler.py), which is where this belongs -- the same
reason the vLLM patch is a vLLM general plugin.

**Opt-in, by environment.** `HK_SGLANG_LINEAR=1` routes SGLang's unquantized
linear layers through hk's GEMM op (`sglang.apply_linear`); without it this is
a no-op that costs an entry-point load. If SGLANG_PLUGINS is set, it has to
name `hk` for SGLang to load this at all.

    HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
        --attention-backend triton --disable-cuda-graph

`load_plugins()` runs once per process, and `apply_linear()` reports "already
patched" if it is reached twice anyway.
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("HK_SGLANG_LINEAR", "0") not in ("0", "", "off", "no")


def register() -> None:
    """The entry point SGLang calls. Never raises: a plugin that throws takes
    the server's startup with it, and a layer that did not get patched is a
    slower server, not a broken one."""
    if not enabled():
        return
    try:
        from . import sglang as hks  # noqa: PLC0415

        note = hks.apply_linear()
    except BaseException as e:  # noqa: BLE001
        note = f"linear not patched ({type(e).__name__}: {e})"
    print(f"[hk] {note} (pid {os.getpid()})", flush=True)
