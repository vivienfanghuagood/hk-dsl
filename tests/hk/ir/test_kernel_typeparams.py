"""dtype parameters (`hk.Tensor[hk.T]`) and the `warps` argument: one kernel definition for every
dtype and warp count (docs/language.md, "One kernel for every dtype and warp count")."""

import re

import pytest

import hk
from hk import T


def _grid(p):
    return (hk.cdiv(p.o.cols, 64 * p.warps), hk.cdiv(p.o.rows, 16), 1)


@hk.kernel(grid=_grid)
def scale(x: hk.Tensor[T], o: hk.Tensor[T], *, warps=1, K=2):
    t = hk.RT(hk.fp32, 16, 64)
    col = (hk.arch.block_idx.x * warps + hk.arch.warp_id()) * 64 if warps > 1 \
        else hk.arch.block_idx.x * 64
    idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16, col)
    v = x[idx].load(t)
    v *= float(K)
    o[idx] = v.to(T)


def _concrete(dtype, warps):
    """The same body written the old way: a factory, one kernel per (dtype, warps)."""
    def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype], *, K=2):
        t = hk.RT(hk.fp32, 16, 64)
        col = (hk.arch.block_idx.x * warps + hk.arch.warp_id()) * 64 if warps > 1 \
            else hk.arch.block_idx.x * 64
        idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16, col)
        v = x[idx].load(t)
        v *= float(K)
        o[idx] = v.to(dtype)

    return hk.kernel(body, warps=warps, name="scale_ref", grid=_grid)   # p.warps: the launch's


def _body_only(src):
    """The kernel's code, without its name and the header comment's constexpr line."""
    return [ln for ln in src.splitlines() if not ln.startswith("// ")]


@pytest.mark.parametrize("dtype", [hk.bf16, hk.fp16, hk.fp32])
@pytest.mark.parametrize("warps", [1, 4])
def test_each_specialization_is_the_factory_kernel(dtype, warps):
    got = scale.source(T=dtype, warps=warps)
    want = _concrete(dtype, warps).source()
    name = f"scale__T{dtype.name}_warps{warps}"
    assert _body_only(got) == _body_only(want.replace("scale_ref", name))


def test_names_and_variants():
    assert isinstance(scale, hk.GenericKernel) and scale.name == "scale"
    v = scale.variant(T=hk.fp16, warps=8)
    assert v.name == "scale__Tfp16_warps8" and v.warps == 8 and not isinstance(v, hk.GenericKernel)
    assert scale.variant(T=hk.fp16, warps=8) is v                    # memoized
    assert scale.variant(T=hk.fp16).name == "scale__Tfp16_warps1"    # the default warps
    assert scale.trace(T=hk.bf16, warps=2).name == "scale__Tbf16_warps2"
    assert scale.specialize(T=hk.bf16, K=3)._defaults["K"] == 3
    assert "scale[T]" in repr(scale)


def test_trace_without_tensors_needs_T():
    with pytest.raises(TypeError, match=r"T=hk.bf16"):
        scale.trace()


def test_constexprs_still_vary_per_call():
    assert scale.source(T=hk.bf16, K=2) != scale.source(T=hk.bf16, K=5)


torch = pytest.importorskip("torch")


def test_T_is_read_from_the_tensors():
    x = torch.zeros(16, 64, dtype=torch.float16)
    o = torch.zeros(16, 64, dtype=torch.float16)
    key, rest = scale._split({}, (x, o))
    assert key == (hk.fp16, 1) and rest == {}
    key, rest = scale._split({"T": hk.bf16, "warps": 4, "K": 3}, (x, o))   # explicit wins, no read
    assert key == (hk.bf16, 4) and rest == {"K": 3}


def test_tensors_sharing_T_must_agree():
    x = torch.zeros(16, 64, dtype=torch.float16)
    o = torch.zeros(16, 64, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match=r"disagree on T: fp16: x; bf16: o"):
        scale._split({}, (x, o))


def test_a_dtype_hk_does_not_have_is_refused():
    x = torch.zeros(16, 64, dtype=torch.float64)
    with pytest.raises(TypeError, match=r"no hk dtype"):
        scale._split({}, (x, x))


def test_two_dtype_parameters():
    U = hk.TypeVar("U")

    @hk.kernel(grid=lambda p: (1, 1, 1))
    def cast(x: hk.Tensor[T], o: hk.Tensor[U]):
        t = hk.RT(hk.fp32, 16, 64)
        o[hk.tile_coord(0, 0, 0, 0)] = x[hk.tile_coord(0, 0, 0, 0)].load(t).to(U)

    assert list(cast._typevars) == ["T", "U"]
    x = torch.zeros(16, 64, dtype=torch.bfloat16)
    o = torch.zeros(16, 64, dtype=torch.float32)
    key, _ = cast._split({}, (x, o))
    assert key == (hk.bf16, hk.fp32)
    assert cast.variant(T=hk.bf16, U=hk.fp32).name == "cast__Tbf16_Ufp32"   # no warps argument
    assert "fp32" in cast.source(T=hk.bf16, U=hk.fp32)


def test_options_ride_along():
    @hk.kernel(grid=lambda p: (1, 1, 1))
    def peek(x: hk.Tensor[T, "strided"], o: hk.Tensor[T]):
        pass

    assert peek.variant(T=hk.bf16)._tensors[0][1] == hk.Tensor[hk.bf16, "strided"]


def test_warps_alone_makes_a_generic_kernel():
    @hk.kernel(grid=lambda p: (p.warps, 1, 1))
    def fill(o: hk.Tensor[hk.fp32], *, warps=2):
        o[hk.tile_coord(0, 0, 0, hk.arch.block_idx.x)] = hk.zeros(hk.RT(hk.fp32, 16, 64))

    assert isinstance(fill, hk.GenericKernel) and fill.warps == 2
    assert fill.variant(warps=4).name == "fill__warps4"
    assert fill.trace(warps=4).threads == 4 * hk.get_target(fill.variant().arch).wave


def test_warps_twice_is_refused():
    with pytest.raises(TypeError, match=r"say it twice"):
        @hk.kernel(warps=4, grid=lambda p: (1, 1, 1))
        def k(o: hk.Tensor[T], *, warps=4):
            pass


def test_warps_needs_a_positive_int_default():
    with pytest.raises(TypeError, match=r"positive int default"):
        @hk.kernel(grid=lambda p: (1, 1, 1))
        def k(o: hk.Tensor[T], *, warps=None):
            pass


def test_a_global_of_the_same_name_is_refused():
    ns = {"hk": hk, "T": hk.RT(hk.fp32, 16, 64), "Tv": hk.T}
    src = ("def k(o: hk.Tensor[Tv]):\n"
           "    o[hk.tile_coord(0, 0, 0, 0)] = hk.zeros(T)\n")
    exec(src, ns)    # noqa: S102 -- a kernel whose module's `T` is a tile, not the dtype parameter
    with pytest.raises(TypeError, match=r"also defines a global `T`"):
        hk.kernel(ns["k"], grid=lambda p: (1, 1, 1))


def test_plain_kernels_are_untouched():
    @hk.kernel(warps=4, grid=lambda p: (p.warps, 1, 1))          # p.warps: the launch's
    def plain(o: hk.Tensor[hk.bf16], *, K=1):
        o[hk.tile_coord(0, 0, 0, 0)] = hk.zeros(hk.RT(hk.bf16, 16, 64))

    assert type(plain) is hk.Kernel and plain.name == "plain"
    assert re.search(r"void plain_kernel", plain.source())


def test_for_arch_keeps_a_specialization_bound():
    v = scale.variant(T=hk.fp16, warps=4)
    on = v.for_arch("gfx950")
    assert on.arch == "gfx950" and on.name == v.name and on.warps == 4
    assert on.trace().arch == "gfx950" and on.source() != v.source()
    g = scale.for_arch("gfx950")
    assert isinstance(g, hk.GenericKernel) and g.variant(T=hk.bf16).arch == "gfx950"
