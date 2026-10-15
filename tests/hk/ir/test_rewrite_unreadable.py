"""A kernel whose source hk cannot read is traced without its `for`, `if` and
`while` rewritten. The error its device control flow then raises has to name
that cause: the generic advice -- "branch with a plain `if`" -- tells the
author to change code that was never the problem. A stale __pycache__
compiled at another path is how this happens in practice."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import bf16, fp32  # noqa: E402

SRC = """
def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
    if x.cols > 16:
        hk.arch.barrier()
"""


def _unreadable():
    ns = {"hk": hk, "bf16": bf16, "fp32": fp32}
    exec(compile(SRC, "/nonexistent/kernels.py", "exec"), ns)  # noqa: S102
    return ns["body"]


def test_an_unreadable_body_says_its_source_was_not_read():
    k = hk.kernel(_unreadable(), arch="gfx1100", warps=1, name="unreadable_if",
                  grid=lambda p: (1, 1, 1))
    with pytest.raises(TypeError, match="could not read the source of `body`") as e:
        k.trace()
    assert "/nonexistent/kernels.py" in str(e.value)


def test_a_readable_body_gets_the_plain_advice_alone():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        if x.cols > 16 and x.rows > 16:
            hk.arch.barrier()

    k = hk.kernel(body, arch="gfx1100", warps=1, name="readable_and",
                  grid=lambda p: (1, 1, 1))
    with pytest.raises(TypeError, match="Combine conditions with &") as e:
        k.trace()
    assert "could not read the source" not in str(e.value)


def test_a_rewritten_helper_inside_an_unreadable_body_is_not_blamed_on_it():
    @hk.func
    def helper(x):
        if x.cols > 16 and x.rows > 16:
            hk.arch.barrier()

    ns = {"hk": hk, "bf16": bf16, "fp32": fp32, "helper": helper}
    exec(compile("def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):\n    helper(x)\n",
                 "/nonexistent/outer.py", "exec"), ns)  # noqa: S102
    k = hk.kernel(ns["body"], arch="gfx1100", warps=1, name="helper_and",
                  grid=lambda p: (1, 1, 1))
    with pytest.raises(TypeError, match="Combine conditions with &") as e:
        k.trace()
    assert "could not read the source" not in str(e.value)
