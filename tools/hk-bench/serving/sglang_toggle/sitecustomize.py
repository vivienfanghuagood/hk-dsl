"""hk on/off inside a running SGLang server, for a same-process A/B (`sglang_toggle_ab.py`).

On PYTHONPATH with HK_SGLANG_TOGGLE_DIR set, every Python process of the server (the scheduler of
each tensor-parallel rank included) installs three signal handlers:

    SIGRTMIN+3  stock: hk's SGLang routes that were applied are reverted (SGLang's own methods back)
    SIGRTMIN+4  hk:    the same routes applied again
    SIGRTMIN+5  write the state and hk's call counters again

After each signal a process writes `<dir>/<pid>`: its arm and hk's counters (`none` where hk is not
loaded, which is every process but the ranks'). The driver signals every pid in the directory and
checks those files. Between switches the server keeps its weights, its KV cache and its compiled
kernels; only which method a layer calls changes.

Valid only where the routes are not baked into CUDA graphs: a captured graph replays the kernels it
captured whatever the method is now. Run SGLang with `--disable-cuda-graph`.
"""

import os
import signal
import sys

_DIR = os.environ.get("HK_SGLANG_TOGGLE_DIR")


def _chain():
    """Run the next sitecustomize on the path too (images ship their own)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        if not d or os.path.abspath(d) == here:
            continue
        p = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    exec(compile(fh.read(), p, "exec"), {"__name__": "sitecustomize_chained"})
            except Exception:  # noqa: BLE001 -- theirs, not ours
                pass
            return


_chain()

#: The routes this process had applied before the first switch to stock: re-applied by "hk".
_WAS_ON: dict = {}


def _routes():
    return sys.modules.get("hk.integration.sglang")


def _counters() -> str:
    out = []
    G = sys.modules.get("hk.integration.vllm_gemm")
    if G is not None and getattr(G, "PATCH", None) is not None:
        out.append("linear " + str({k: v for k, v in dict(G.PATCH.stats).items() if v}))
    S = _routes()
    if S is not None and hasattr(S, "norm_stats"):
        out.append("norm " + S.norm_stats())
    return " | ".join(out)


def _write(state: str) -> None:
    try:
        with open(os.path.join(_DIR, str(os.getpid())), "w") as fh:
            fh.write(f"{state}\n{_counters()}\n")
    except OSError:
        pass


def _set(on: bool) -> None:
    S = _routes()
    if S is None:
        _write("none")
        return
    try:
        if on:
            if _WAS_ON.pop("linear", False) and S._LINEAR is None:
                S.apply_linear()
            if _WAS_ON.pop("norm", False) and S._NORM is None:
                S.apply_norm()
            if hasattr(S, "pause_norm"):
                S.pause_norm(False)
        else:
            if S._LINEAR is not None:
                from hk.integration import vllm_gemm

                cls, orig = S._LINEAR
                cls.apply = orig
                S._LINEAR = None
                vllm_gemm.set_routes(None)
                _WAS_ON["linear"] = True
            if S._NORM is not None and hasattr(S, "pause_norm"):
                # the layers keep the bound forward_hip SGLang cached at construction: pause the
                # route in place rather than restoring the class attribute
                S.pause_norm(True)
            elif S._NORM is not None:
                cls, orig = S._NORM
                cls.forward_hip = orig
                S._NORM = None
                _WAS_ON["norm"] = True
        state = "hk" if on else "stock"
        on_now = [r for r, v in (("linear", S._LINEAR),
                                 ("norm", S._NORM if not getattr(S, "_NORM_PAUSED", False) else None))
                  if v is not None]
        _write(f"{state} on={','.join(on_now) or '-'}")
    except Exception as e:  # noqa: BLE001 -- said in the state file, checked by the driver
        _write(f"error {type(e).__name__}: {e}")


if _DIR:
    try:
        os.makedirs(_DIR, exist_ok=True)
        signal.signal(signal.SIGRTMIN + 3, lambda *a: _set(False))
        signal.signal(signal.SIGRTMIN + 4, lambda *a: _set(True))
        signal.signal(signal.SIGRTMIN + 5, lambda *a: _write("dump"))
        _write("started")
    except (OSError, ValueError):     # ValueError: not the main thread
        pass
