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
    """The second half of B is still in flight at the first wait: a 16 x 32
    bf16 tile, two ds_read_b128 a base tile on gfx1100 (each half-wave holds
    the whole K run, mirrored) and one on gfx1201 (the halves split it)."""
    from hk.target import current_arch, get_target

    gemm = dict(KERNELS)["gemm"]
    counts = re.findall(r"hk\.lds_wait_for [^{]*\{n = (\d+)", gemm.trace().dump())
    reads = max(1, get_target(current_arch()).mma["bf16"].operand_vgprs * 4 // 16)
    assert counts == [str(2 * reads), "0"]


def test_the_attention_rescale_is_a_runtime_branch():
    text = dict(KERNELS)["attention"].trace().dump()
    assert "hk.any_ne" in text and "scf.if" in text


def test_the_control_flow_example_carries_its_scalars():
    kernels = dict(KERNELS)
    assert "hk.carried_names = [\"col\"]" in kernels["prefix_sum_for"].trace().dump()
    assert "scf.while" in kernels["prefix_sum_while"].trace().dump()
