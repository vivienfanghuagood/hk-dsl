"""Kernel names without boilerplate: a factory's kernel needs no `body.__name__ = name`
(docs/language.md, "Kernels made by a factory")."""

import subprocess
import sys
import textwrap

import hk


def _factory(dtype, warps, op=None):
    def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype]):
        t = hk.RT(dtype, 16, 64)
        v = x[hk.tile_coord(0, 0, 0, 0)].load(t)
        o[hk.tile_coord(0, 0, 0, 0)] = v if op is None else op(v)

    return hk.kernel(body, warps=warps, grid=lambda p: (1, 1, 1))


def test_top_level_kernel_keeps_its_name():
    @hk.kernel(grid=lambda p: (1, 1, 1))
    def add(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o[hk.tile_coord(0, 0, 0, 0)] = a[hk.tile_coord(0, 0, 0, 0)].load(hk.RT(hk.bf16, 16, 64))

    # defined inside a test function, but it reads no factory value: its own name, as before
    assert add.name == "add" and add.__name__ == "add"


def test_factory_kernel_is_named_by_its_values():
    k = _factory(hk.bf16, 4)
    assert k.name == "_factory_body__dtypebf16_opNone_warps4", k.name
    assert k.__name__ == k.__qualname__ == k.name          # what hk.profile and tracebacks show
    assert _factory(hk.fp16, 4).name != k.name
    assert _factory(hk.bf16, 8).name != k.name
    assert f"{k.name}(" in k.source() or k.name in k.source()


def _double(v):
    return v * 2


def _inc(v):
    return v + 1


def test_callables_in_the_closure_hash_stably():
    a = _factory(hk.bf16, 4, op=_double)
    b = _factory(hk.bf16, 4, op=_inc)
    assert a.name != b.name and a.name.startswith("_factory_body__dtypebf16_warps4_")
    assert a.name == _factory(hk.bf16, 4, op=_double).name


def test_explicit_names_win():
    def f():
        def body(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
            pass
        return body

    assert hk.kernel(f(), name="mine", grid=lambda p: (1, 1, 1)).name == "mine"
    b = f()
    b.__name__ = "by_hand"                                   # the old spelling still works
    k = hk.kernel(b, grid=lambda p: (1, 1, 1))
    assert k.name == "by_hand" and k.__name__ == "by_hand"


def test_specialize_memoizes_and_names():
    calls = []

    @hk.specialize
    def silu(*, dtype=hk.bf16, warps=4):
        calls.append((dtype, warps))

        @hk.kernel(warps=warps, grid=lambda p: (1, 1, 1))
        def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype]):
            pass

        return body

    k = silu()
    assert k is silu(warps=4) is silu(dtype=hk.bf16, warps=4)   # defaults are part of the key
    assert k.name == "silu__dtypebf16_warps4" == k.__name__
    assert silu(dtype=hk.fp16).name == "silu__dtypefp16_warps4"
    assert len(calls) == 2


def test_specialize_keeps_an_explicit_name():
    @hk.specialize
    def g(*, n=1):
        return hk.kernel(_named(n), name=f"g_{n}", grid=lambda p: (1, 1, 1))

    assert g(n=3).name == "g_3"


def _named(n):
    def body(x: hk.Tensor[hk.bf16]):
        pass
    return body


def test_names_are_the_same_in_another_process():
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, "tests/hk/ir")
        import hk, test_kernel_names as t
        print(t._factory(hk.bf16, 4).name, t._factory(hk.bf16, 4, op=t._double).name)
    """)
    here = f"{_factory(hk.bf16, 4).name} {_factory(hk.bf16, 4, op=_double).name}"
    for seed in ("0", "123"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**__import__("os").environ, "PYTHONHASHSEED": seed}, check=True)
        assert out.stdout.split() == here.split()


def test_a_dtype_only_in_annotations_still_names_the_variant():
    def f(dtype):
        def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype]):
            pass
        return hk.kernel(body, grid=lambda p: (1, 1, 1))

    def h(dtype, n):
        def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype]):
            hk.cdiv(n, 2)
        return hk.kernel(body, grid=lambda p: (1, 1, 1))

    # a closure value (n): the annotation's dtype names the variant too
    assert h(hk.bf16, 2).name == "h_body__n2_Tbf16_warps1"

    # no closure value: the nested function keeps its name, as before; hk.specialize names it
    assert f(hk.bf16).name == "body"
    g = hk.specialize(lambda *, dtype: f(dtype))
    assert g(dtype=hk.bf16).name != g(dtype=hk.fp16).name
