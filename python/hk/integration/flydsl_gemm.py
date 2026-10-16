"""FlyDSL's RDNA3 GEMM, as an optional path of the linear-layer routes.

FlyDSL (github.com/ROCm/FlyDSL) has a WMMA GEMM for gfx11 and a dispatcher,
`rdna3_gemm_dispatch`, that picks a tile from the shape (`pick_tile`) and the
options that go with it -- sched_group_barrier hints, a K rotation -- without
measuring anything. On the scoreboard (docs/history/26) it is 2-5% ahead of
hk's GEMM at prefill shapes, while hk's decode GEMV and attention have no
FlyDSL counterpart that beats them. hk-dsl sits beside FlyDSL, so where a
route table measured FlyDSL's GEMM fastest, the linear layers run it.

It is optional in every sense:
* hk does not depend on FlyDSL. The path exists on gfx11 when `flydsl` is
  importable and so is FlyDSL's `kernels` package, which is in its
  repository and not in its wheel: on sys.path already, or under
  `$FLYDSL_ROOT`. `HK_FLYDSL=0` turns it off.
* Where another `kernels` is installed (HuggingFace's, beside SGLang),
  FlyDSL's is imported from `$FLYDSL_ROOT` beside it, and `import kernels`
  still finds the other one (`_import_beside`).
* A table's FlyDSL rows are an overlay (`engine.RouteTable`). Where the path
  does not exist, or does not take a call, the call goes where the same table
  without those rows sends it -- hk's kernels or the engine's.

What it takes: bf16, contiguous, and an (M, N, K) that one of FlyDSL's tiles
divides. M is compiled into the kernel, so that means M a multiple of 32 at
the least. And only at the tile the table measured: FlyDSL's speed depends on
the tile M's divisibility leaves it (1024 rows run 128x128; 1056 rows only
32-row tiles), so `variant` names the tile the dispatcher would pick, and a
table's range holds for that tile alone.

What a server pays for it: each new M is a compile, 0.4-0.8 s cold and
0.04 s from FlyDSL's disk cache; and each call about 60 us of host time
(FlyDSL's launch path; hk's op takes 18). At prefill sizes, where FlyDSL
wins, both are behind the GEMM.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

NAME = "flydsl"
_OFF = ("0", "off", "no", "false")
MODULE = "kernels.gemm.rdna3_f16_gemm_autotune"

#: [FlyDSL's rdna3_f16_gemm_autotune module or None, why it is None].
_STATE: list = []
_VARIANT: Dict[Tuple[int, int, int], Optional[str]] = {}
#: FlyDSL's modules that could not keep their names beside another `kernels`.
_PRIVATE: Dict[str, object] = {}


def _is_flydsl(dirs) -> bool:
    return any(os.path.isfile(os.path.join(d, "gemm", "rdna3_f16_gemm.py")) for d in dirs)


def _kernels_dirs() -> list:
    """Where `import kernels` finds (or found) its package, without importing it."""
    mod = sys.modules.get("kernels")
    spec = getattr(mod, "__spec__", None) if mod is not None else None
    if spec is None and mod is None:
        try:
            spec = importlib.util.find_spec("kernels")
        except (ImportError, ValueError):
            spec = None
    return list(getattr(spec, "submodule_search_locations", None) or [])


def _import_beside(root: str, other: list):
    """FlyDSL's GEMM module, imported from `root`'s `kernels` package although
    another distribution's `kernels` (HuggingFace's, beside SGLang) is the one
    `import kernels` finds -- and leaving that one the one it finds.

    FlyDSL's modules import each other as `kernels.<...>`, so they are imported
    under those names, with `root` first on sys.path and the other package's
    modules out of sys.modules. Then sys.path and the other package's modules
    are put back, and `kernels` itself names the other package again. FlyDSL's
    submodules (`kernels.gemm.*`, `kernels.common.*`) keep their names, which
    FlyDSL's compile cache resolves its functions by; the other package has no
    such submodules, and one that did would keep its own (FlyDSL's is then
    held here only)."""
    def ours(k):
        return k == "kernels" or k.startswith("kernels.")

    theirs = {m.split(".")[0] for d in other for m in os.listdir(d)
              if os.path.isdir(os.path.join(d, m)) or m.endswith(".py")} if other else set()
    theirs = {m[:-3] if m.endswith(".py") else m for m in theirs}
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if ours(k)}
    path = list(sys.path)
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    try:
        return importlib.import_module(MODULE)
    finally:
        fly = {k: sys.modules.pop(k) for k in list(sys.modules) if ours(k)}
        sys.path[:] = path
        importlib.invalidate_caches()
        sys.modules.update(saved)
        for k, mod in fly.items():
            if k == "kernels" or k in saved or k.split(".")[1] in theirs:
                _PRIVATE[k] = mod
            else:
                sys.modules[k] = mod


def _load():
    if _STATE:
        return _STATE[0]
    from ..target import current_arch  # noqa: PLC0415

    fg, why = None, ""
    if os.environ.get("HK_FLYDSL", "1").strip().lower() in _OFF:
        why = "HK_FLYDSL is off"
    elif not current_arch().startswith("gfx11"):
        why = f"its RDNA3 GEMM is for gfx11, and this is {current_arch()}"
    elif importlib.util.find_spec("flydsl") is None:
        why = "flydsl is not importable"
    else:
        root = os.environ.get("FLYDSL_ROOT")
        other = _kernels_dirs()
        try:
            if other and not _is_flydsl(other) and root and _is_flydsl([os.path.join(root, "kernels")]):
                fg = _import_beside(root, other)
            else:
                if root and root not in sys.path:
                    sys.path.append(root)
                fg = importlib.import_module(MODULE)
        except Exception as e:  # noqa: BLE001 -- any failure means no path
            dirs = _kernels_dirs()
            if dirs and not _is_flydsl(dirs):
                why = (f"the 'kernels' package in {dirs[0]} is not FlyDSL's and "
                       f"shadows it ({type(e).__name__}: {e}); set FLYDSL_ROOT to a "
                       f"checkout of FlyDSL to import FlyDSL's beside it")
            else:
                why = (f"FlyDSL's kernels package is not importable ({type(e).__name__}: "
                       f"{e}); it is in FlyDSL's repository, not its wheel -- set "
                       f"FLYDSL_ROOT to a checkout of the installed version")
    _STATE[:] = [fg, why]
    return fg


def available() -> bool:
    return _load() is not None


def status() -> str:
    fg = _load()
    if fg is None:
        return f"FlyDSL not available: {_STATE[1]}"
    import flydsl  # noqa: PLC0415

    return f"FlyDSL {getattr(flydsl, '__version__', '?')} ({os.path.dirname(fg.__file__)})"


def tile_name(tile) -> str:
    """FlyDSL's (reg_m, reg_n, reg_k, waves_m, waves_n) as the block it
    computes and its wave count: (4, 4, 2, 2, 2) is "128x128x32w4"."""
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    return f"{16 * reg_m * waves_m}x{16 * reg_n * waves_n}x{16 * reg_k}w{waves_m * waves_n}"


def variant(m: int, n: int, k: int) -> Optional[str]:
    """The tile FlyDSL's dispatcher would run y = x @ W.T on, x (m, k) and
    W (n, k); None if FlyDSL is not available or none of its tiles divides
    the shape. Cached: it runs on every routed call."""
    key = (m, n, k)
    try:
        return _VARIANT[key]
    except KeyError:
        pass
    fg = _load()
    v = None
    if fg is not None and m > 0 and fg.feasible_tiles(m, n, k):
        v = tile_name(fg.pick_tile(m, n, k))
    _VARIANT[key] = v
    return v


def run(x, weight, out) -> None:
    """out = x @ weight.T through FlyDSL's dispatcher. x (m, k), weight (n, k)
    and out (m, n), contiguous bf16 on one device, and `variant` not None.
    The dispatcher launches on torch's current stream, so a CUDA graph being
    captured records it."""
    m, k = x.shape
    _load().rdna3_gemm_dispatch(out, x, weight, m, weight.shape[0], k)
