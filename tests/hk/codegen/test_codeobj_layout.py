"""The code object path's host half, which needs hipcc and no GPU: the
device scaffold compiles, the code object carries its kernel under the
symbol the launcher asks for, and its layout table describes `globals`."""

from __future__ import annotations

import struct

import pytest

from hk.codegen import scaffold as sc
from hk.ops import norm
from hk.runtime import codeobj


@pytest.fixture(scope="module")
def built():
    k = norm.KERNELS["rmsw_bf16_w4"].specialize(TPW=1, FOLD=16, EXACT=1)
    return codeobj.load(k)


def test_the_kernel_is_under_its_mangled_name(built):
    assert built.symbol.encode() in built.image
    assert built.symbol == "_Z19rmsw_bf16_w4_kernel7globals"


def test_the_layout_covers_every_tensor(built):
    # x, w, o: a pointer, four extents and a stride each, laid out in order.
    assert [o[0] for o in built.offsets] == sorted(o[0] for o in built.offsets)
    assert built.size >= 3 * (8 + 4 * 4 + 4)
    for ptr, dims, row in built.offsets:
        assert ptr % 8 == 0 and all(ptr < d < row for d in dims)


def test_the_gate_reads_the_code_objects_remarks(built):
    assert built.build.kernels and not built.build.kernels[0].spills


def test_the_grid_is_evaluated_from_extents(built):
    shapes = [(1, 1, 16 * 300, 256), (1, 1, 16, 256), (1, 1, 16 * 300, 256)]
    assert built.grid_for(shapes) == (1, 300, 1)


def test_a_packed_argument_holds_what_was_packed(built):
    shapes = [(1, 2, 48, 256), (1, 1, 16, 256), (1, 1, 48, 512)]
    raw = built.pack(shapes, [0x1000, 0x2000, 0x3000], [256, 256, 768])
    (ptr, dims, row) = built.offsets[2]
    assert struct.unpack_from("<Q", raw, ptr)[0] == 0x3000
    assert [struct.unpack_from("<i", raw, o)[0] for o in dims] == [1, 1, 48, 512]
    assert struct.unpack_from("<i", raw, row)[0] == 768


def test_the_device_scaffold_names_the_layout_symbol():
    k = norm.KERNELS["rmsw_bf16_w4"].specialize(TPW=1, FOLD=16, EXACT=1)
    src = sc.device(k.trace())
    assert sc.LAYOUT_SYMBOL in src and "--offload" not in src
