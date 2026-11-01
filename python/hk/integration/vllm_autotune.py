"""Measure, as vLLM loads a model, the linear shapes no routing table has.

The linear-layer patch routes by a measured table, and a (N, K) the table
never measured takes vLLM's own path -- so a model nobody tuned for got
nothing from it. This closes that: once the weights are loaded, and before
vLLM's profile run sizes the KV cache or any CUDA graph is captured, every
unquantized bf16 linear of the loaded model is looked up, and the shapes the
table does not have are measured (`hk.autotune.routes.tune_missing`) and
written to this machine's records. The routes are then reloaded, so this
start already runs on them; the next start of the same model finds every
shape and measures nothing.

Why there, and nowhere earlier or later:
* **After the weights.** The shapes come from the model vLLM built -- every
  architecture it supports, sharded for this rank -- not from a config
  parser that knows five layer names.
* **Before the profile run.** vLLM sizes the KV cache from the peak memory
  of a dummy forward. The tuner's buffers are freed, and the allocator's
  cache emptied, before it; the profile resets the peak itself.
* **Before capture.** A graph captured on one route replays it whatever the
  table says afterwards.
* **Once per machine.** Each tensor-parallel rank is a process, and so is
  every other server: the records are a file, written under a lock. The
  first process measures; the others wait, then read what it wrote.

**A budget.** `HK_AUTOTUNE_BUDGET` seconds (default 60) bound the
measuring. Decode-sized Ms (up to 64 rows) come first for every shape --
they are where a decode step's time is, and the cheap points -- then the
prefill grid. A shape cut short answers only up to the largest M measured,
and the next start carries on from there. The patch's report line says what
was measured, and which shapes stayed on vLLM's path. `HK_AUTOTUNE_BUDGET=0`
measures nothing; `inf`, without a limit. `HK_AUTOTUNE=0` ignores every
record and measures nothing. A table named by `$HK_VLLM_ROUTES` is the table,
and nothing is measured against it.

Noise. A server's node is shared, and a point measured at start-up is
measured once. The tuner re-measures close calls and a kernel takes an M
only by 3%; `python -m hk.integration tune --model X` measures the same thing
before the server starts, on a quiet card, and `--force` measures again.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Dict, List, Optional, Tuple

BUDGET_ENV = "HK_AUTOTUNE_BUDGET"
DEFAULT_BUDGET = 60.0

#: What the last load measured, for `ab` and the status line.
REPORT: Dict[str, Any] = {}

_ORIG_LOAD: list = []


def budget() -> Optional[float]:
    """Seconds the tuning at load may take: 0 for none, None for no limit."""
    raw = os.environ.get(BUDGET_ENV, "").strip().lower()
    if raw in ("", "default"):
        return DEFAULT_BUDGET
    if raw in ("0", "off", "no", "false"):
        return 0.0
    if raw in ("inf", "none", "unlimited"):
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_BUDGET


def model_shapes(model) -> Tuple[List[Tuple[str, int, int, int]], List[Tuple[str, int, int]]]:
    """(shapes worth tuning, shapes every call of which has a bias) of a
    loaded vLLM model: (name, N, K, layers) per distinct weight shape of its
    unquantized bf16 linear layers and LM head, as this rank holds them. A
    biased call falls back (`vllm_gemm.route_for`), so a shape only biased
    calls reach is not measured."""
    import torch  # noqa: PLC0415
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: PLC0415
    from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: PLC0415
        ParallelLMHead,
    )

    seen: Dict[Tuple[int, int], List[Any]] = {}
    # With tied embeddings the LM head is the input embedding module, which
    # named_modules lists once, under the embedding's name.
    head = getattr(model, "lm_head", None)
    for prefix, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
            continue
        qm = getattr(mod, "quant_method", None)
        if isinstance(mod, ParallelLMHead) or (head is not None and mod is head):
            if type(qm).__name__ != "UnquantizedEmbeddingMethod":
                continue
            name, bias = "lm_head", False
        elif isinstance(qm, UnquantizedLinearMethod):
            name = prefix.rsplit(".", 1)[-1]
            bias = getattr(mod, "bias", None) is not None and not getattr(
                mod, "skip_bias_add", False)
        else:
            continue
        n, k = int(w.shape[0]), int(w.shape[1])
        rec = seen.setdefault((n, k), [name, 0, 0])
        rec[1] += 1
        rec[2] += 0 if bias else 1
    tune = [(name, n, k, calls) for (n, k), (name, calls, plain) in seen.items() if plain]
    biased = [(name, n, k) for (n, k), (name, calls, plain) in seen.items() if not plain]
    return tune, biased


def _print(s: str) -> None:
    print(s, flush=True)


def at_load(model, device=None, log=_print) -> Dict[str, Any]:
    """Measure the model's shapes the table lacks, within the budget, and
    reload the routes. Returns (and keeps, in REPORT) what happened."""
    from ..autotune import enabled  # noqa: PLC0415
    from ..autotune import routes as _routes  # noqa: PLC0415
    from ..target import current_arch  # noqa: PLC0415
    from . import vllm_gemm  # noqa: PLC0415

    REPORT.clear()
    t0 = time.monotonic()
    b = budget()
    shapes, biased = model_shapes(model)
    REPORT.update(shapes=[list(s) for s in shapes], biased=[list(s) for s in biased],
                  budget=b)
    table = vllm_gemm.routes()
    have = table.complete() if table is not None else set()
    missing = [s for s in shapes if (s[1], s[2]) not in have]
    why = None
    if not missing:
        why = f"all {len(shapes)} shapes in the table"
    elif os.environ.get(vllm_gemm.ROUTES_ENV):
        why = f"routes pinned by ${vllm_gemm.ROUTES_ENV}; nothing measured"
    elif not enabled():
        why = "HK_AUTOTUNE=0; nothing measured"
    elif b == 0:
        why = f"{BUDGET_ENV}=0; nothing measured"
    if why is not None:
        REPORT.update(line=why + _biased_note(biased), seconds=0.0, missing=len(missing))
        return REPORT
    engine, engine_name = _routes.engine_path("vllm")
    # Every layer, biased ones too, is the context the missing ones are timed in, each on the
    # table's route.
    rep = _routes.tune_missing(
        "vllm_linear", current_arch(), missing, engine=engine, engine_name=engine_name,
        complete=have, budget=b, by="vllm startup", device=device,
        log=lambda s: log(f"[hk] {s}"),
        model=[s[:3] for s in shapes] + [tuple(s) for s in biased],
        route_of=lambda m, n, k: table.lookup(m, n, k) if (n, k) in have else "engine")
    vllm_gemm.set_routes(None)
    vllm_gemm.routes()
    REPORT.update(rep.as_dict(), line=rep.line() + _biased_note(biased),
                  seconds=round(time.monotonic() - t0, 1), missing=len(missing))
    return REPORT


def _biased_note(biased) -> str:
    if not biased:
        return ""
    return ("; " + ", ".join(f"{n} {a}x{b}" for n, a, b in biased)
            + " carr" + ("ies" if len(biased) == 1 else "y") + " a bias and stay"
            + ("s" if len(biased) == 1 else "") + " on vLLM's path")


#: The module whose worker loads the model, patched when it is imported.
WORKER_MODULE = "vllm.v1.worker.gpu_worker"


def _wrap(worker) -> None:
    """Wrap `worker.load_model` to measure after it. Once."""
    if _ORIG_LOAD:
        return
    orig = worker.load_model
    _ORIG_LOAD.append((worker, orig))

    @functools.wraps(orig)
    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            rep = at_load(self.model_runner.get_model(), getattr(self, "device", None))
            print(f"[hk] linear routes at load: {rep['line']} (pid {os.getpid()})",
                  flush=True)
        except BaseException as e:  # noqa: BLE001 -- a slower server, not a broken one
            REPORT.update(line=f"tuning at load failed ({type(e).__name__}: {e})")
            print(f"[hk] linear routes at load: {REPORT['line']}", flush=True)
        try:
            from . import vllm_gemm  # noqa: PLC0415

            line = vllm_gemm.decide_layers(self.model_runner.get_model())
            REPORT["layers"] = line
            print(f"[hk] linear layers at load: {line} (pid {os.getpid()})", flush=True)
        except BaseException as e:  # noqa: BLE001 -- layers then decide on their first call
            print(f"[hk] linear layers at load: not decided ({type(e).__name__}: {e}); "
                  f"each decides on its first call", flush=True)
        return out

    worker.load_model = load_model


class _OnImport:
    """A finder that does nothing but wrap the worker once its module has
    run. Importing that module costs seconds, and vLLM's API server, which
    loads the plugin too, never builds a worker; so it is not imported here."""

    def __init__(self, name: str, then):
        self.name, self.then = name, then

    def find_spec(self, name, path=None, target=None):
        if name != self.name:
            return None
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            self.then(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> str:
    """Measure missing shapes when vLLM's GPU worker loads a model: wrap its
    `load_model` now if its module is imported, else as soon as it is.
    Idempotent; reports in one phrase what it will do."""
    import sys  # noqa: PLC0415

    b = budget()
    if b == 0:
        return f"no tuning at load ({BUDGET_ENV}=0)"
    mod = sys.modules.get(WORKER_MODULE)
    if mod is not None:
        _wrap(mod.Worker)
    elif not _ORIG_LOAD and not any(isinstance(f, _OnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _OnImport(WORKER_MODULE, lambda m: _wrap(m.Worker)))
    lim = "no limit" if b is None else f"{b:.0f} s"
    return f"shapes the table lacks are measured at load ({lim})"


def uninstall() -> None:
    import sys  # noqa: PLC0415

    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _OnImport)]
    if _ORIG_LOAD:
        worker, orig = _ORIG_LOAD.pop()
        worker.load_model = orig
