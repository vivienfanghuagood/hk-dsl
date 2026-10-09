"""Shared fixtures and the tier gates.

Three tiers, by what they need:

  tests/hk/ir       pure Python -- no compiler, no GPU
  tests/hk/codegen  hipcc only  -- generates and compiles, never launches
  tests/hk/gpu      a Radeon + torch

The first two are the development loop and run on any machine. A tier that
cannot run is skipped with a reason, never silently passed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


def _hipcc_works() -> bool:
    exe = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    if not Path(exe).exists():
        return False
    try:
        return subprocess.run(
            [exe, "--version"], capture_output=True, timeout=120
        ).returncode == 0
    except Exception:
        return False


HAVE_HIPCC = _hipcc_works()


def _have_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


HAVE_GPU = _have_gpu()


def pytest_collection_modifyitems(config, items):
    no_hipcc = pytest.mark.skip(reason="no working hipcc on this machine")
    no_gpu = pytest.mark.skip(reason="no GPU (or no torch) on this machine")
    for item in items:
        path = str(item.fspath)
        if "/codegen/" in path or "hipcc" in item.keywords:
            if not HAVE_HIPCC:
                item.add_marker(no_hipcc)
        if "/gpu/" in path or "gpu" in item.keywords:
            if not HAVE_GPU:
                item.add_marker(no_gpu)


@pytest.fixture(scope="session")
def hk():
    import hk as _hk

    return _hk
