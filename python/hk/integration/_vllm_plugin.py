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

**On by default.** With hk-dsl installed, vLLM loads this entry point in every
process, and it applies every route `hk.integration.switches` has on: the ones
validated with real weights on this GPU. Turning hk off is for comparisons:

    HK_DISABLE=1 vllm serve ...              # stock vLLM: no route, nothing patched
    HK_VLLM_MXFP8=0 vllm serve ...           # one route off
    HK_PATCH_VLLM=0 vllm serve ...           # every vLLM route off

Before it patches anything, the plugin checks the engine contract of every route it is about to
apply (`hk.integration.contract`). If one is broken, for example by a vLLM or AITER other than
the one measured, it patches nothing, prints one warning line, and vLLM runs stock.

`load_general_plugins` warns that plugins may be loaded more than once in one
process, so `register()` is idempotent: every route's apply is guarded and
`sdpa.patch()` refuses to capture itself as the original.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple


def enabled() -> bool:
    """Whether any vLLM route is on (hk.integration.switches)."""
    from . import switches  # noqa: PLC0415

    return bool(switches.on("vllm"))


def _sdpa() -> Tuple[str, Callable]:
    from . import vllm as hkv  # noqa: PLC0415

    return hkv.apply(), hkv.revert


def _backend() -> Tuple[str, Callable]:
    # On ROCm it is registered *over* an existing enum member (see vllm_backend.SLOT), the one
    # ROCm picks by default -- which is why it is not on by default (switches.ROUTES).
    from . import switches, vllm_backend  # noqa: PLC0415

    v = switches.value("vllm_backend")
    return vllm_backend.register(v or None), lambda: None


def _gemm() -> Tuple[str, Callable]:
    from . import vllm_gemm  # noqa: PLC0415

    return vllm_gemm.apply(), vllm_gemm.revert


def _int4() -> Tuple[str, Callable]:
    from . import vllm_int4  # noqa: PLC0415

    return vllm_int4.apply_patch(), vllm_int4.revert


def _moe_a8w4() -> Tuple[str, Callable]:
    from . import vllm_moe_a8w4  # noqa: PLC0415

    return vllm_moe_a8w4.apply_patch(), vllm_moe_a8w4.revert


def _mxfp8() -> Tuple[str, Callable]:
    from . import vllm_mxfp8  # noqa: PLC0415

    return vllm_mxfp8.apply_patch(), vllm_mxfp8.revert


def _dsv41_index() -> Tuple[str, Callable]:
    from . import vllm_dsv41_index  # noqa: PLC0415

    return vllm_dsv41_index.apply_patch(), vllm_dsv41_index.revert


def _moe() -> Tuple[str, Callable]:
    from . import vllm_moe  # noqa: PLC0415

    return vllm_moe.apply(), vllm_moe.revert


#: route -> its apply, returning (report, revert). In the order they are applied.
APPLY: Dict[str, Callable[[], Tuple[str, Callable]]] = {
    "vllm_backend": _backend, "vllm_gemm": _gemm, "vllm_int4": _int4,
    "vllm_moe_a8w4": _moe_a8w4, "vllm_mxfp8": _mxfp8,
    "vllm_dsv41_index": _dsv41_index, "vllm_moe": _moe, "vllm_sdpa": _sdpa,
}


def register() -> None:
    """The entry point vLLM calls. Never raises: a plugin that throws takes
    the engine's startup with it, and a kernel that did not get patched is a
    slower server, not a broken one. The one exception is asked for:
    `HK_STACK_CHECK=refuse` on a stack vLLM was not measured on."""
    from . import contract, switches  # noqa: PLC0415

    pid = os.getpid()
    if switches.disabled():
        print(f"[hk] vLLM: off ({switches.DISABLE}=1), stock (pid {pid})", flush=True)
        return
    routes = switches.on("vllm")
    master = os.environ.get(switches.ENGINES["vllm"], "")
    user = "" if switches.is_off(master) else os.environ.get("HK_VLLM_USER", "")
    if not routes and not user:
        why = switches.decide("vllm_mxfp8").why
        print(f"[hk] vLLM: no route on ({why}), stock (pid {pid})", flush=True)
        return
    notes = []
    # The stack first: a Triton other than the one this vLLM was measured
    # with changes every Triton kernel's speed and nothing's correctness, so
    # it is invisible unless something says so (hk.integration.stack). Warns
    # by default; HK_STACK_CHECK=refuse makes it stop the engine.
    from . import stack  # noqa: PLC0415

    try:
        notes.append(stack.enforce("vLLM plugin", default="warn"))
    except stack.StackMismatch:
        raise
    except BaseException as e:  # noqa: BLE001
        notes.append(f"stack not checked ({type(e).__name__}: {e})")
    # The contract of every route about to be applied: all of them or none (contract.py).
    problems = contract.check(routes)
    if problems:
        contract.refuse("vllm", problems)
        return
    applied: List[Tuple[str, Callable]] = []
    current = ""
    try:
        for name, how in APPLY.items():
            if name in routes:
                current = name
                report, revert = how()
                applied.append((name, revert))
                notes.append(report)
        # A user's own kernels: the modules that register and patch them,
        # imported here because this is the process the model runs in.
        if user:
            current = "HK_VLLM_USER"
            from . import register as hk_register  # noqa: PLC0415

            notes.append(f"user: {', '.join(hk_register.load_user(user))}")
    except BaseException as e:  # noqa: BLE001 -- never half-patched: undo what was applied
        for _, revert in reversed(applied):
            try:
                revert()
            except BaseException:  # noqa: BLE001,S110
                pass
        contract.refuse("vllm", [f"{current} raised {type(e).__name__}: {e}"], contract.RAISED)
        return
    from ..runtime.compile import arm_jit_notice  # noqa: PLC0415

    arm_jit_notice("vLLM plugin")
    on = ", ".join(f"{n} ({d.why})" for n, d in routes.items())
    print(f"[hk] vLLM routes on: {on}; {'; '.join(notes)} (pid {pid})", flush=True)
