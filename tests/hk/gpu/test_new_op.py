"""`python -m hk.integration new-op` on the card: the op it writes registers,
routes, gives the exact answer bit for bit and agrees with its reference at
every shape it lists, before anyone has changed a line of it."""

import importlib.util

import pytest

from hk.integration import newop

torch = pytest.importorskip("torch")


def test_the_generated_op_checks_out(tmp_path):
    assert newop.main(["gen_op_check", "--dir", str(tmp_path)]) == 0
    spec = importlib.util.spec_from_file_location("hk_newop_gpu", tmp_path / "gen_op_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() <= 2e-2
    op = mod.registered()
    assert op.stats["hk"] == 1 + len(mod.SHAPES)
    assert op.stats["fallback"] == 0
    # A call the route declines runs the reference, and is counted as such.
    a = torch.randn(4, 100, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(op(a, a), a + a)
    assert op.stats["fallback"] == 1
