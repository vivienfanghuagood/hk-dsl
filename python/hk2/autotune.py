"""``@hk2.autotune``: pick the fastest of several configurations per problem, measured on the GPU it runs on.

    @hk.autotune(configs=[hk.Config(BM=128, BN=64, num_warps=8), hk.Config(BM=64, BN=64, num_warps=4)],
                 key=["S", "D"])
    @hk.jit
    def kernel(q, k, v, o, S, D: hk.constexpr, BM: hk.constexpr = 128, BN: hk.constexpr = 64): ...

    kernel[lambda m: (hk.cdiv(m["S"], m["BM"]),)](q, k, v, o, S, D=128)

The first launch for a key (the named arguments' values; tensors contribute their shape and dtype) times
every configuration that compiles (median of a few batches of launches, in one process) and keeps the
fastest. Choices persist per GPU in ``$HK2_CACHE_DIR/tune/<kernel>.json``, so later processes start tuned.
Launches during tuning write the kernel's outputs repeatedly: tune kernels whose outputs are pure functions
of their inputs.
"""
from __future__ import annotations

import inspect
import json
import os
import statistics
import threading
from typing import Any, Dict, List, Optional, Sequence

from . import runtime as rt
from .jit import Kernel

__all__ = ["Config", "autotune", "Autotuner"]


class Config:
    """Compile-time arguments (and ``num_warps``) for one candidate."""

    def __init__(self, num_warps: Optional[int] = None, **kwargs):
        self.kwargs = kwargs
        self.num_warps = num_warps

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.kwargs)
        if self.num_warps:
            d["num_warps"] = self.num_warps
        return d

    def __repr__(self):
        return "Config(" + ", ".join(f"{k}={v}" for k, v in self.as_dict().items()) + ")"


class Autotuner:
    def __init__(self, kernel: Kernel, configs: Sequence[Config], key: Sequence[str], warmup: int = 2,
                 rounds: int = 5, reps: int = 10):
        self.kernel, self.configs, self.key = kernel, list(configs), list(key)
        self.warmup, self.rounds, self.reps = warmup, rounds, reps
        self.best: Dict[str, Config] = {}
        self.timings: Dict[str, Dict[str, float]] = {}
        self._loaded = set()
        self._lock = threading.Lock()
        self.name = kernel.name
        self.sig = kernel.sig

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.run(grid, *args, **kwargs)
        return launch

    # ------------------------------------------------------------------ keys and persistence
    def _key(self, args, kwargs, arch: str) -> str:
        import torch
        ba = self.sig.bind_partial(*args, **{k: v for k, v in kwargs.items() if k in self.sig.parameters})
        parts = [arch]
        for name in self.key:
            v = ba.arguments.get(name, self.sig.parameters[name].default if name in self.sig.parameters else None)
            if isinstance(v, torch.Tensor):
                v = (tuple(v.shape), str(v.dtype).replace("torch.", ""))
            parts.append(f"{name}={v}")
        return "|".join(parts)

    def _path(self) -> str:
        d = os.path.join(rt.cache_dir(), "tune")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.name}.json")

    def _load(self, arch: str):
        if arch in self._loaded:
            return
        self._loaded.add(arch)
        try:
            data = json.load(open(self._path()))
        except (OSError, ValueError):
            return
        for k, d in data.items():
            if k.startswith(arch + "|") and k not in self.best:
                d = dict(d)
                nw = d.pop("num_warps", None)
                self.best[k] = Config(num_warps=nw, **d)

    def _save(self):
        path = self._path()
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            data = {}
        data.update({k: c.as_dict() for k, c in self.best.items()})
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, path)

    # ------------------------------------------------------------------ tuning
    def _time(self, grid, args, kwargs, cfg: Config) -> float:
        import torch
        kw = dict(kwargs, **cfg.as_dict())
        for _ in range(self.warmup):
            self.kernel.run(grid, *args, **kw)
        times = []
        for _ in range(self.rounds):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(self.reps):
                self.kernel.run(grid, *args, **kw)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) / self.reps)
        return statistics.median(times)

    def tune(self, grid, args, kwargs, key: str) -> Config:
        import warnings
        res = {}
        for cfg in self.configs:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    self.kernel.run(grid, *args, **dict(kwargs, **cfg.as_dict()))
                spills = any("spills" in str(x.message) for x in w)
                t = self._time(grid, args, kwargs, cfg)
                res[repr(cfg)] = t * (1.05 if spills else 1.0)  # a spilling plan must win clearly
            except Exception as e:  # noqa: BLE001 -- configurations that do not compile here are skipped
                res[repr(cfg)] = float("inf")
                if os.environ.get("HK2_TUNE_VERBOSE"):
                    print(f"[hk2 autotune] {self.name} {cfg}: {type(e).__name__}: {str(e)[:120]}")
        best = min(self.configs, key=lambda c: res[repr(c)])
        if res[repr(best)] == float("inf"):
            raise RuntimeError(f"hk2.autotune: no configuration of {self.name} compiles for {key}")
        self.timings[key] = res
        if os.environ.get("HK2_TUNE_VERBOSE"):
            print(f"[hk2 autotune] {self.name} {key}: {best} " +
                  " ".join(f"{k}={v * 1e3:.1f}us" for k, v in sorted(res.items(), key=lambda kv: kv[1])))
        return best

    def config_for(self, grid, args, kwargs) -> Config:
        arch = kwargs.get("arch") or rt.current_arch()
        key = self._key(args, kwargs, arch)
        with self._lock:
            self._load(arch)
            cfg = self.best.get(key)
            if cfg is None:
                cfg = self.best[key] = self.tune(grid, args, kwargs, key)
                try:
                    self._save()
                except OSError:
                    pass
        return cfg

    def run(self, grid, *args, **kwargs):
        if kwargs.get("interpret") or os.environ.get("HK2_INTERPRET", "0") not in ("", "0"):
            return self.kernel.run(grid, *args, **dict(kwargs, **self.configs[0].as_dict()))
        cfg = self.config_for(grid, args, kwargs)
        return self.kernel.run(grid, *args, **dict(kwargs, **cfg.as_dict()))


def autotune(configs: Sequence[Config], key: Sequence[str] = (), **opts):
    def wrap(kernel: Kernel) -> Autotuner:
        if not isinstance(kernel, Kernel):
            raise TypeError("@hk2.autotune goes above @hk2.jit")
        return Autotuner(kernel, configs, key, **opts)
    return wrap
