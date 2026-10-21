"""The DX suite (bench/dx) cannot rot: every task is complete, every solution
imports, and every solution is correct on its task's SMALL shapes.

The first test needs only Python; the rest are the GPU tier (torch on an AMD
GPU, hk built). Run: `python3 -m pytest tests/dx -q`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from bench.dx import common  # noqa: E402

TASKS = common.task_ids()


def _gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


GPU = _gpu()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs torch on a GPU")


def test_eight_tasks_complete():
    assert len(TASKS) == 8, TASKS
    for tid in TASKS:
        d = common.TASKS_DIR / tid
        assert (d / "task.md").is_file(), f"{tid}: no task.md"
        src = (d / "reference.py").read_text()
        for name in ("SHAPES", "QUICK", "SMALL", "def make_inputs", "def reference",
                     "def torch_impl", "def check"):
            assert name in src, f"{tid}: reference.py has no {name}"


def _pairs():
    if not GPU:
        return []
    out = []
    for tid in TASKS:
        task = common.load_task(tid)
        for sol in common.solutions_for(tid, task):
            for shape in task.SMALL:
                out.append(pytest.param(tid, sol.name, shape["name"],
                                        id=f"{tid}-{sol.name}-{shape['name']}"))
    return out


@needs_gpu
def test_every_solution_declares_a_kind():
    for tid in TASKS:
        task = common.load_task(tid)
        for sol in common.solutions_for(tid, task):
            assert sol.kind in common.KINDS, (tid, sol.name, sol.kind)
            if sol.kind == "proxy":
                assert sol.proxy, f"{tid}/{sol.name}: a proxy must say what it computes instead"


@needs_gpu
@pytest.mark.parametrize("tid,sol_name,shape_name", _pairs())
def test_solution_small(tid, sol_name, shape_name):
    import torch

    task = common.load_task(tid)
    sol = next(s for s in common.solutions_for(tid, task) if s.name == sol_name)
    shape = next(s for s in task.SMALL if s["name"] == shape_name)
    inputs = task.make_inputs(shape, device="cuda", seed=0)
    sup = sol.supports(inputs)
    if sup is not True:
        pytest.skip(f"unsupported: {sup}")
    ref = task.reference(**inputs)
    ins = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    out = sol.prepare(ins)()
    torch.cuda.synchronize()
    if sol.kind == "proxy":
        return
    task.check(out, ref, inputs)
