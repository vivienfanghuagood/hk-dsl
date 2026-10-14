"""Every example kernel compiles without spilling: an example that spills
would teach a kernel that computes the wrong answer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import _catalog  # noqa: E402

KERNELS = _catalog.kernels()


@pytest.mark.parametrize("name, kernel", KERNELS, ids=[n for n, _ in KERNELS])
def test_it_compiles_without_spilling(name, kernel):
    built = kernel.build().kernels
    assert built, f"{name}: no resource remarks parsed"
    for k in built:
        assert k.scratch == 0 and not k.spills, f"{name}: {k.line()}"
