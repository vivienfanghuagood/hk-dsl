"""Strided globals (`hk.Tensor[dtype, "strided"]`, kittens::gl_strided): the type, its C++,
the layout table the launchers pack by, and the refusals. GPU behaviour: tests/hk/gpu/test_strided_gl.py."""

import pytest

import hk
from hk.codegen import scaffold
from hk.types import GlobalType


def _stage_kernel(arch="gfx1100", strided=True):
    st = hk.ST(hk.bf16, 16, 128)
    ty = hk.RT(hk.bf16, 16, 128)
    src_t = hk.Tensor[hk.bf16, "strided"] if strided else hk.Tensor[hk.bf16]

    @hk.kernel(arch=arch, warps=1, grid=lambda p: (1, 1, 1))
    def k(src: src_t, o: hk.Tensor[hk.bf16]):
        g = hk.group(1)
        s = hk.lds.alloc(st)
        buf = g.stage_buffer(s)
        g.stage(buf, src[hk.elem_coord(0, 0, 0, 0)])
        hk.arch.vm_wait(0)
        g.commit(s, buf, wait=True)
        hk.arch.barrier()
        o[hk.tile_coord(0, 0, 0, 0)] = s.load(ty)

    return k


def test_annotation():
    assert hk.Tensor[hk.bf16, "strided"] == GlobalType(hk.bf16, "", True)
    assert repr(hk.Tensor[hk.bf16, "strided"]) == "gl<bf16, strided>"
    with pytest.raises(TypeError, match="scoped tensor is not strided"):
        hk.Tensor[hk.bf16, "system", "strided"]
    with pytest.raises(TypeError, match="each once"):
        hk.Tensor[hk.bf16, "strided", "strided"]


def test_stage_through_every_stride():
    src = scaffold.device(_stage_kernel().trace())
    assert "kittens::gl_strided<bf16> src;" in src
    assert "kittens::stage_load_strided<32>(" in src
    assert '#include "hk/rdna/strided.cuh"' in src
    # the strided table: one tensor, index 0, three offsets
    assert "hk_layout_strided[]" in src
    for f in scaffold.STRIDED_FIELDS:
        assert f"offsetof(decltype(globals::src), {f})" in src


def test_plain_kernels_are_unchanged():
    src = scaffold.device(_stage_kernel(strided=False).trace())
    assert "gl_strided" not in src and "hk/rdna/strided.cuh" not in src
    assert "hk_layout_strided" not in src


def test_other_ops_are_refused():
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch="gfx1100", warps=1, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16, "strided"], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        o[idx] = a[idx].load(t)

    with pytest.raises(Exception, match="read by g.stage only so far"):
        scaffold.bare(k.trace())


def test_cdna_refuses():
    with pytest.raises(Exception, match="RDNA-only"):
        scaffold.bare(_stage_kernel(arch="gfx950").trace())


def test_pybind_and_library_paths_refuse():
    ir = _stage_kernel().trace()
    with pytest.raises(ValueError, match="strided tensor"):
        scaffold.pybind_module(ir)
    with pytest.raises(ValueError, match="strided tensor"):
        scaffold.torch_library(ir)


def test_strided_args_shapes():
    torch = pytest.importorskip("torch")
    from hk.runtime.codeobj import strided_args

    class T:   # a CPU stand-in with the attributes strided_args reads
        def __init__(self, t):
            self.t = t

        def __getattr__(self, n):
            return getattr(self.t, n)

    # vLLM ROCM_ATTN K: (blocks, heads, d/8, block, 8) -> (blocks, heads, block, d/8, 8)
    k = torch.zeros(5, 2, 16, 16, 8, dtype=torch.bfloat16).permute(0, 1, 3, 2, 4)
    shape, st = strided_args(T(k), "k", "op")
    assert shape == (5, 2, 16, 128)
    assert st == (8, 2048, 4096, 128)          # row, depth, batch, granule
    # flash layout (blocks, block, heads, d) -> (blocks, heads, block, d)
    f = torch.zeros(5, 16, 2, 128, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    shape, st = strided_args(T(f), "f", "op")
    assert shape == (5, 2, 16, 128) and st == (256, 128, 4096, 8)
    # 2-D contiguous
    shape, st = strided_args(T(torch.zeros(3, 64, dtype=torch.bfloat16)), "x", "op")
    assert shape == (1, 1, 3, 64) and st == (64, 0, 0, 8)
    with pytest.raises(RuntimeError, match="one contiguous"):
        strided_args(T(torch.zeros(2, 2, 2, 2, 4, dtype=torch.bfloat16)), "y", "op")
