"""Numerics: does a generated kernel compute what torch computes.

Needs a Radeon and torch. This is the tier that actually launches, and the only
one that can catch an indexing error -- a kernel that reads the wrong tile
compiles perfectly and passes every gate.

Tolerances are per dtype rather than global: bf16 has 8 mantissa bits, so
`assert_close`'s float32 defaults would fail on correct output.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

DTYPES = [torch.bfloat16, torch.float16, torch.float32]

#: (rtol, atol) per dtype. bf16 carries ~3 decimal digits; demanding more of it
#: tests the format, not the kernel.
TOL = {
    torch.bfloat16: (3e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-5, 1e-6),
}


def _rand(shape, dtype):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _close(got, want, dtype, msg=""):
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(got.float(), want.float(), rtol=rtol, atol=atol, msg=msg)


# -- the Phase 1 gate ---------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_add_matches_torch(dtype):
    """`hk.ops.add(a, b)` elementwise-equal to torch. This is the gate."""
    a, b = _rand((256, 512), dtype), _rand((256, 512), dtype)
    _close(hk.ops.add(a, b), a + b, dtype)


# -- the rest of the op set ---------------------------------------------------

BINARY = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("maximum", torch.maximum),
    ("minimum", torch.minimum),
]

UNARY = [
    ("exp", torch.exp),
    ("relu", torch.relu),
    ("neg", torch.neg),
    ("abs", torch.abs),
]


@pytest.mark.parametrize("name,ref", BINARY, ids=[n for n, _ in BINARY])
@pytest.mark.parametrize("dtype", DTYPES)
def test_binary(name, ref, dtype):
    a, b = _rand((128, 256), dtype), _rand((128, 256), dtype)
    _close(getattr(hk.ops, name)(a, b), ref(a, b), dtype)


@pytest.mark.parametrize("name,ref", UNARY, ids=[n for n, _ in UNARY])
@pytest.mark.parametrize("dtype", DTYPES)
def test_unary(name, ref, dtype):
    a = _rand((128, 256), dtype)
    _close(getattr(hk.ops, name)(a), ref(a), dtype)


def test_gelu_is_the_tanh_approximation():
    """base_ops::gelu is x*(0.5 + 0.5*fast_tanh(...)), so the erf form is the
    wrong reference -- it would disagree by ~1e-3 and look like a kernel bug.
    fast_tanh is a rational approximation, hence the loosened tolerance."""
    a = _rand((128, 256), torch.float32)
    want = torch.nn.functional.gelu(a, approximate="tanh")
    torch.testing.assert_close(hk.ops.gelu(a), want, rtol=1e-3, atol=1e-3)


# -- shapes -------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (16, 64),  # exactly one tile
        (16, 128),  # one tile row, two columns
        (64, 64),  # a grid in y
        (1024, 1024),  # many workgroups
        (4, 32, 128),  # 3D, last axis tiles
        (8, 16, 16, 64),  # 4D
        (4096,),  # 1D, flattened fallback
    ],
)
def test_shapes(shape):
    a, b = _rand(shape, torch.bfloat16), _rand(shape, torch.bfloat16)
    out = hk.ops.add(a, b)
    assert out.shape == a.shape
    _close(out, a + b, torch.bfloat16, msg=f"shape {shape}")


def test_out_parameter_is_written_in_place():
    a, b = _rand((64, 128), torch.float32), _rand((64, 128), torch.float32)
    out = torch.empty_like(a)
    got = hk.ops.add(a, b, out=out)
    assert got.data_ptr() == out.data_ptr()
    _close(out, a + b, torch.float32)


def test_a_shape_that_does_not_tile_is_refused():
    """A partial tile reads past the end of the tensor. That returns whatever
    is next in memory rather than failing, so it has to be refused up front."""
    a = torch.randn(7, 13, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not tile"):
        hk.ops.add(a, a)


def test_non_contiguous_input_is_refused():
    a = _rand((128, 256), torch.bfloat16).t()
    with pytest.raises(ValueError, match="contiguous"):
        hk.ops.add(a, a)


def test_mismatched_operands_are_refused():
    a = _rand((128, 256), torch.bfloat16)
    b = _rand((128, 64), torch.bfloat16)
    with pytest.raises(ValueError, match="do not broadcast"):
        hk.ops.add(a, b)


def test_an_unsupported_dtype_names_what_is_supported():
    a = _rand((128, 256), torch.bfloat16)
    with pytest.raises(TypeError, match="no bf16 gelu kernel"):
        hk.ops.gelu(a)


# -- the compile path ---------------------------------------------------------


def test_the_kernel_is_compiled_once_and_reused():
    """The second call must not shell out to hipcc; if it did, a decode step
    calling this in a loop would spend its time in the compiler."""
    k = hk.ops.elementwise.KERNELS["add_fp32"]
    a = _rand((64, 64), torch.float32)
    hk.ops.add(a, a)
    assert k.build().cached or k.build() is k.build()
    assert k.compile() is k.compile()
