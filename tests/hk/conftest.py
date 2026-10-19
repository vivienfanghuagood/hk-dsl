"""Shared fixtures and the tier gates.

Three tiers, by what they need:

  tests/hk/ir       pure Python -- no compiler, no GPU
  tests/hk/codegen  hipcc only  -- generates and compiles, never launches
  tests/hk/gpu      a Radeon + torch

The first two are the development loop and run on any machine. A tier that
cannot run is skipped with a reason, never silently passed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

# The ir and codegen tiers assert gfx1100 facts -- the routing tables, the LDS
# plans, the derived waits were all measured on a W7900. Before gfx942/gfx950
# were targets, an Instinct box fell back to gfx1100 by not being recognised;
# now it is recognised, so the pin has to be said. HK_ARCH set by hand wins.
os.environ.setdefault("HK_ARCH", "gfx1100")

# Records this machine's tuning wrote -- routes for a model someone measured
# here -- would change what the shipped tables are tested to say. The tests
# read the shipped ones, and write only into a directory of their own.
if "HK_TUNE_DIR" not in os.environ:
    import tempfile

    os.environ["HK_TUNE_DIR"] = tempfile.mkdtemp(prefix="hk-tune-tests-")


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


def _gpu_arch() -> str:
    try:
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return ""


def pytest_collection_modifyitems(config, items):
    no_hipcc = pytest.mark.skip(reason="no working hipcc on this machine")
    no_gpu = pytest.mark.skip(reason="no GPU (or no torch) on this machine")
    gpu_arch = _gpu_arch() if HAVE_GPU else ""
    wrong_gpu = pytest.mark.skip(
        reason=f"the GPU is {gpu_arch} and the tests build for "
               f"{os.environ['HK_ARCH']}; set HK_ARCH={gpu_arch} to run them on it")
    for item in items:
        path = str(item.fspath)
        if "/codegen/" in path or "hipcc" in item.keywords:
            if not HAVE_HIPCC:
                item.add_marker(no_hipcc)
        if "/gpu/" in path or "gpu" in item.keywords:
            if not HAVE_GPU:
                item.add_marker(no_gpu)
            elif gpu_arch and gpu_arch != os.environ["HK_ARCH"]:
                item.add_marker(wrong_gpu)


@pytest.fixture(scope="session")
def hk():
    import hk as _hk

    return _hk
