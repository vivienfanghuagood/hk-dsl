"""Load a compiled artifact.

Modules are loaded by explicit file path rather than by putting the cache
directory on sys.path, and each one is loaded once per process and kept: a
dlopen of a .so that is already mapped returns the same handle anyway, so
caching here just avoids re-walking importlib.

The .so is never overwritten in place -- compile.py always writes a new
content-hashed path. That is not tidiness: rebuilding a .so while a process has
it mapped corrupts that process's view of the code, and on a shared node it
takes down whoever else had it open.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_loaded: Dict[str, ModuleType] = {}


def load(so_path: Path, module_name: str) -> ModuleType:
    key = str(so_path)
    if key in _loaded:
        return _loaded[key]

    if not so_path.exists():
        raise FileNotFoundError(f"{so_path}: compiled module is missing")

    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {so_path} as {module_name!r}")
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec so that a module which imports itself (pybind does
    # not, but a future torch-library scaffold might) sees a consistent entry.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        sys.modules.pop(module_name, None)
        raise ImportError(
            f"{so_path} failed to import as {module_name!r}: {e}\n"
            f"If this says 'does not define module export function', the "
            f"PYBIND11_MODULE name and the .so basename disagree -- that is a "
            f"codegen bug, not a build problem."
        ) from e
    _loaded[key] = mod
    return mod
