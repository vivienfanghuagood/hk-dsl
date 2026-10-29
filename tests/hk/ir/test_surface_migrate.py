"""tools/surface/migrate.py keeps a kernel the same kernel (V0: byte-identical generated C++).

The trap it exists for: a kernel is its trace, and the trace is the order ops are emitted in.
`hk.mma_ABt_mx(a, b, acc, sa, sb)` evaluates its arguments a, b, acc, sa, sb; the new spelling
`hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb), acc)` evaluates a, sa, b, sb, acc. With inline
loads in the arguments, writing the new spelling in place issues the loads in another order --
another kernel. The rule (docs/design/dsl-surface.md §5.1, docs/migrating.md): bind the
arguments that emit ops to locals in the old order first."""

from __future__ import annotations

import importlib.util
import re
import textwrap
import warnings
from pathlib import Path

import hk
import pytest
from hk import fp8e4m3, fp32, i8

_spec = importlib.util.spec_from_file_location(
    "surface_migrate", Path(__file__).resolve().parents[3] / "tools" / "surface" / "migrate.py")
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

K = 256
ANN = {"a": hk.Tensor[fp8e4m3], "b": hk.Tensor[fp8e4m3], "sa": hk.Tensor[i8],
       "sb": hk.Tensor[i8], "c": hk.Tensor[fp32]}

# The decode MoE's shared-down step, as it was written: four inline loads in mma_ABt_mx's
# arguments (weights, x, the weights' scales, x's scales).
INLINE = """
def body(a, b, sa, sb, c):
    acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
    for k in range(2):
        hk.mma_ABt_mx(a.at[0, k * K].load(hk.RT(fp8e4m3, 16, K)),
                      b.at[0, k * K].load(hk.RT(fp8e4m3, 32, K)), acc,
                      hk.load_mx_scales(sa, 16, K // 32, 0, k * K // 32),
                      hk.load_mx_scales(sb, 32, K // 32, 0, k * K // 32), into=acc)
    c.at[0, 0] = acc
"""

# The accumulator and both scales made inline, as a returned value.
INLINE_ACC = """
def body(a, b, sa, sb, c):
    ta = a.at[0, 0].load(hk.RT(fp8e4m3, 16, K))
    tb = b.at[0, 0].load(hk.RT(fp8e4m3, 32, K))
    acc = hk.mma_ABt_mx(ta, tb, hk.zeros(hk.RT(fp32, 16, 32, "col")),
                        hk.load_mx_scales(sa, 16, K // 32, 0, 0),
                        hk.load_mx_scales(sb, 32, K // 32, 0, 0), b_fmt="fp8")
    c.at[0, 0] = acc
"""

# copy_mx_scales(dst, src) evaluates dst first; src.copy(into=dst) evaluates src first.
INLINE_COPY = """
def body(a, b, sa, sb, c):
    ta = a.at[0, 0].load(hk.RT(fp8e4m3, 16, K))
    tb = b.at[0, 0].load(hk.RT(fp8e4m3, 32, K))
    s_a = hk.load_mx_scales(sa, 16, K // 32, 0, 0)
    s_b = hk.load_mx_scales(sb, 32, K // 32, 0, 0)
    hk.copy_mx_scales(hk.load_mx_scales(sb, 32, K // 32, 0, 0), hk.load_mx_scales(sb, 32, K // 32, 32, 0))
    c.at[0, 0] = hk.mma("mma_ABt", ta.scaled(s_a), tb.scaled(s_b), hk.zeros(hk.RT(fp32, 16, 32, "col")))
"""

# What a hand migration in place gives for INLINE: the scales' loads move between the operands'.
NAIVE = """
def body(a, b, sa, sb, c):
    acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
    for k in range(2):
        hk.mma("mma_ABt", a.at[0, k * K].load(hk.RT(fp8e4m3, 16, K)).scaled(
                   hk.load_mx_scales(sa, 16, K // 32, 0, k * K // 32)),
               b.at[0, k * K].load(hk.RT(fp8e4m3, 32, K)).scaled(
                   hk.load_mx_scales(sb, 32, K // 32, 0, k * K // 32)), acc, into=acc)
    c.at[0, 0] = acc
"""


def _kernel(src: str, name: str):
    ns = {"hk": hk, "fp8e4m3": fp8e4m3, "fp32": fp32, "K": K}
    exec(compile(textwrap.dedent(src), f"<{name}>", "exec"), ns)  # noqa: S102 -- the test kernels are sources
    body = ns["body"]
    body.__annotations__ = ANN
    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=1, name="migrate", grid=lambda p: (1, 1, 1))


def _build(src: str):
    """(generated C++, IR, the DeprecationWarnings) -- one kernel name, so equal C++ means equal."""
    k = _kernel(src, "migrate_v0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cpp, ir = k.source(), k.trace().dump()
    return cpp, ir, [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]


def _loads(ir: str):
    """The trace's loads in order, each with its operands (which global, which coordinate)."""
    return [ln.split("=", 1)[1].strip() for ln in ir.splitlines() if re.search(r"\bhk\.\w*load\w*\b", ln)]


@pytest.mark.parametrize("src", [INLINE, INLINE_ACC, INLINE_COPY], ids=["loads", "acc", "copy"])
def test_the_migrated_kernel_is_byte_identical(src):
    new_src, manual = migrate.rewrite_source(textwrap.dedent(src))
    assert not manual
    assert "mma_ABt_mx" not in new_src and "copy_mx_scales" not in new_src
    old_cpp, old_ir, warned = _build(src)
    new_cpp, new_ir, none = _build(new_src)
    assert warned and not none
    assert new_ir == old_ir
    assert new_cpp == old_cpp


def test_the_loads_stay_in_the_old_order():
    old_ir = _build(INLINE)[1]
    new_ir = _build(migrate.rewrite_source(textwrap.dedent(INLINE))[0])[1]
    naive_ir = _build(NAIVE)[1]
    old, new, naive = _loads(old_ir), _loads(new_ir), _loads(naive_ir)
    assert len(old) == 8                      # 2 K steps x (weights, x, their scales)
    assert new == old
    # the trap, kept honest: the in-place spelling issues the same loads in another order
    assert sorted(naive) == sorted(old) and naive != old
    assert _build(NAIVE)[0] != _build(INLINE)[0]


def test_the_rewriter_binds_in_the_old_order_and_only_when_the_order_moves():
    new = migrate.rewrite_source(textwrap.dedent(INLINE))[0]
    body = [ln.strip() for ln in new.splitlines()]
    i = next(n for n, ln in enumerate(body) if ln.startswith("_a"))
    assert [ln.split(" = ")[0][:2] for ln in body[i:i + 4]] == ["_a", "_b", "_s", "_s"]
    assert body[i + 2].startswith("_sa") and body[i + 3].startswith("_sb")
    # one emitting argument (or several in an order the new spelling keeps): written in place
    kept = "hk.mma_ABt_mx(ta, b.at[0, 0].load(T), hk.zeros(T_C), s_a, s_b)\n"
    assert migrate.rewrite_source(kept)[0] == \
        'hk.mma("mma_ABt", ta.scaled(s_a), b.at[0, 0].load(T).scaled(s_b), hk.zeros(T_C))\n'
    # a list subscript is Python's; --strict treats it as emitting too
    lists = "hk.mma_ABt_mx(ws[k], xs[k][0], acc, ss[k], xs[k][1])\n"
    assert "_a" not in migrate.rewrite_source(lists)[0]
    assert migrate.rewrite_source(lists, strict=True)[0].startswith("# (bound")


def test_a_nested_call_it_cannot_reorder_is_left_and_reported():
    src = "o.store(hk.mma_ABt_mx(t, t, hk.zeros(T), hk.load_mx_scales(s, 16, 4, 0, 0), sb))\n"
    new, manual = migrate.rewrite_source(src)
    assert new == src and manual and manual[0][0] == 1


def test_the_old_spelling_keeps_its_order():
    # the shim takes its arguments as Python evaluated them at the call -- the old order
    old_ir = _build(INLINE)[1]
    hoisted = _build(textwrap.dedent("""
    def body(a, b, sa, sb, c):
        acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
        for k in range(2):
            w = a.at[0, k * K].load(hk.RT(fp8e4m3, 16, K))
            x = b.at[0, k * K].load(hk.RT(fp8e4m3, 32, K))
            ws = hk.load_mx_scales(sa, 16, K // 32, 0, k * K // 32)
            xs = hk.load_mx_scales(sb, 32, K // 32, 0, k * K // 32)
            hk.mma("mma_ABt", w.scaled(ws), x.scaled(xs), acc, into=acc)
        c.at[0, 0] = acc
    """))[1]
    assert old_ir == hoisted
