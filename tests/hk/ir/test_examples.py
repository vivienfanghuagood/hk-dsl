"""The examples trace and translate, and the GEMM example's waits are the
ones its docstring promises."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import _catalog  # noqa: E402

KERNELS = _catalog.kernels()


@pytest.mark.parametrize("name, kernel", KERNELS, ids=[n for n, _ in KERNELS])
def test_it_traces_and_translates(name, kernel):
    assert "func.func @" in kernel.trace().dump()
    assert "__global__" in kernel.source()


def test_the_gemm_waits_step_down():
    gemm = dict(KERNELS)["gemm"]
    counts = re.findall(r"hk\.lds_wait_for [^{]*\{n = (\d+)", gemm.trace().dump())
    assert counts == ["4", "0"]


def test_the_attention_rescale_is_a_runtime_branch():
    text = dict(KERNELS)["attention"].trace().dump()
    assert "hk.any_ne" in text and "scf.if" in text


def test_the_control_flow_example_carries_its_scalars():
    kernels = dict(KERNELS)
    assert "hk.carried_names = [\"col\"]" in kernels["prefix_sum_for"].trace().dump()
    assert "scf.while" in kernels["prefix_sum_while"].trace().dump()
