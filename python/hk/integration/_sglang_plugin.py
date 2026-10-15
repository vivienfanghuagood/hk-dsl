"""The `sglang.srt.plugins` entry point.

SGLang runs the model in a scheduler process the launcher spawns, so a patch
made in the process that ran `launch_server` never reaches it. SGLang loads
its general plugins in the launcher and again in every scheduler process
(sglang/srt/managers/scheduler.py), which is where this belongs -- the same
reason the vLLM patch is a vLLM general plugin.

**Opt-in, by environment**, one switch per patch; with neither, this is a
no-op that costs an entry-point load:
* `HK_SGLANG_LINEAR=1` routes SGLang's unquantized linear layers through
  hk's GEMM op (`sglang.apply_linear`);
* `HK_SGLANG_MOE=1` replaces the softmax top-k that fails on gfx1100 and
  routes the bf16 expert GEMMs through hk's (`sglang_moe.apply`);
  `HK_SGLANG_MOE=topk` replaces only the top-k (`sglang_moe.apply_topk`).

    HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
        --attention-backend triton --disable-cuda-graph

If SGLANG_PLUGINS is set, it has to name `hk` for SGLang to load this at all.
`load_plugins()` runs once per process, and each patch reports "already
patched" if it is reached twice anyway.
"""

from __future__ import annotations

import os

_OFF = ("0", "", "off", "no")


def enabled() -> bool:
    return any(os.environ.get(v, "0") not in _OFF
               for v in ("HK_SGLANG_LINEAR", "HK_SGLANG_MOE"))


def register() -> None:
    """The entry point SGLang calls. Never raises: a plugin that throws takes
    the server's startup with it, and a layer that did not get patched is a
    slower server, not a broken one."""
    notes = []
    if os.environ.get("HK_SGLANG_LINEAR", "0") not in _OFF:
        try:
            from . import sglang as hks  # noqa: PLC0415

            notes.append(hks.apply_linear())
        except BaseException as e:  # noqa: BLE001
            notes.append(f"linear not patched ({type(e).__name__}: {e})")
    moe = os.environ.get("HK_SGLANG_MOE", "0")
    if moe not in _OFF:
        try:
            from . import sglang_moe  # noqa: PLC0415

            notes.append(sglang_moe.apply_topk() if moe == "topk" else sglang_moe.apply())
        except BaseException as e:  # noqa: BLE001
            notes.append(f"moe not patched ({type(e).__name__}: {e})")
    if notes:
        print(f"[hk] {'; '.join(notes)} (pid {os.getpid()})", flush=True)
