"""hk.templates.gemm without a GPU: which launches a shape gets and which of
them carries the epilogue, and that the epilogue traces into every kernel
the routes use, on both architectures.

The property the plans are held to: every column of C is written by exactly
one launch, that launch runs the epilogue, and a launch that stores split-K
shares never does -- so an epilogue that is not linear (a GELU, a bias) runs
once per element, on the whole sum.
"""

from __future__ import annotations

import importlib
import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.ops import gemm as rdna  # noqa: E402
from hk.ops import gemm_cdna, gemm_cdna_dp  # noqa: E402
from hk.templates import gemm as T  # noqa: E402
# The module (the name hk.templates.gemm is the template itself).
TG = importlib.import_module("hk.templates.gemm")

BF16 = hk.bf16

# Qwen3-8B / Llama-3-8B linear layers (N, K), and the M of a decode step, a
# small and a large prefill.
LAYERS = [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288), (28672, 4096),
          (4096, 14336), (151936, 4096)]
MS = [1, 4, 16, 33, 128, 300, 1024, 2048, 4096, 8192]


def _ep(fn, kinds=None, dtypes=None, consts=None, out=BF16):
    kinds = kinds or {}
    dtypes = dtypes or {n: BF16 for n, k in kinds.items() if k != "const"}
    return TG.Epilogue(fn, kinds, dtypes, consts or {}, out)


def test_the_template_is_what_hk_templates_gemm_names_every_time():
    """Importing hk.templates.gemm binds the package attribute to the module;
    either way the name calls the template and carries its epilogues."""
    for _ in range(2):
        g = hk.templates.gemm
        assert callable(g) and g.bias_gelu is TG.bias_gelu
    import hk.templates.gemm as g2  # noqa: PLC0415

    assert callable(g2) and g2.residual is TG.residual
    assert callable(TG) and TG.bias is TG.gemm.bias


@pytest.mark.parametrize("n,k", LAYERS)
def test_every_column_is_written_once_by_a_launch_that_runs_the_epilogue(n, k):
    for m in MS:
        steps = T.plan(m, n, k, arch="gfx950")
        cover = [0] * (n // 128)
        for st in steps:
            lo, hi = st.cols
            if st.into == "out":
                assert st.epilogue, f"{m}x{n}x{k}: {st} writes C without the epilogue"
                for c in range(lo // 128, hi // 128):
                    cover[c] += 1
            else:
                assert not st.epilogue, f"{m}x{n}x{k}: {st} runs it on a share"
        assert cover == [1] * len(cover), f"{m}x{n}x{k}: {[str(s) for s in steps]}"


def test_a_split_k_route_runs_the_epilogue_once_after_the_reduction():
    # gfx950's table: 4096x4096 at M=128 is dp64k4, 4096x12288 at M=1024 ppk4.
    assert gemm_cdna.route(128, 4096, 4096) == "dp64k4"
    s = T.plan(128, 4096, 4096, arch="gfx950")
    assert [(x.kernel, x.into, x.epilogue) for x in s] == [
        ("dp", "scratch", False), ("reduce", "out", True)]
    assert s[0].sched["splits"] == s[1].sched["splits"] == 4
    assert gemm_cdna.route(1024, 4096, 12288) == "ppk4"
    s = T.plan(1024, 4096, 12288, arch="gfx950")
    assert [(x.kernel, x.into, x.epilogue) for x in s] == [
        ("pp", "scratch", False), ("reduce", "out", True)]


def test_the_routes_are_hk_ops_matmuls():
    """The same kernel and schedule hk.ops.matmul runs, launch for launch."""
    for (n, k) in LAYERS[:6]:
        for m in MS:
            name = gemm_cdna.route(m, n, k)
            assert all(st.route in (name, "", *gemm_cdna.CANDIDATES)
                       for st in T.plan(m, n, k, arch="gfx950"))
    s = T.plan(1024, 4096, 4096, arch="gfx1100")
    assert len(s) == 1 and s[0].kernel == "rdna" and s[0].epilogue
    assert s[0].sched == rdna.schedule_for(1024, 4096, 4096)
    assert T.plan(48, 4096, 4096, arch="gfx1100")[0].sched == rdna.schedule_for(
        48, 4096, 4096)
    # Decode batches go to the GEMV on gfx1100 too, as hk's vLLM routes do
    # (hk.ops.matmul itself runs THIN there): 1.6-2.7x it on a W7900D.
    s = T.plan(8, 4096, 4096, arch="gfx1100")
    assert [(x.kernel, x.epilogue) for x in s] == [("decode", True)]


def test_a_wave_split_and_a_128_column_tail_each_carry_the_epilogue():
    s = T.plan(2048, 12288, 4096, arch="gfx950")          # ppw
    assert [(x.kernel, x.cols, x.epilogue) for x in s] == [
        ("pp", (0, 8192), True), ("pp", (8192, 12288), True)]
    s = T.plan(4096, 4224, 1024, arch="gfx950")           # 256 does not divide N
    assert s[0].cols == (0, 4096) and s[0].epilogue
    assert s[-1].cols == (4096, 4224) and s[-1].epilogue


def bias_gelu_own(acc, ctx):
    return (acc + ctx.bias).gelu()


def _steps():
    """A step of every kernel kind, both arches."""
    return [
        ("gfx1100", T.plan(1024, 4096, 4096, arch="gfx1100")[0]),
        ("gfx1100", T.plan(8, 4096, 4096, arch="gfx1100")[0]),
        ("gfx950", TG.Step("pp", gemm_cdna.schedule_for(4096, 4096, 4096), (0, 4096), True)),
        ("gfx950", TG.Step("dp", {**gemm_cdna_dp.DEFAULT, "block_m": 64,
                                  "xcd_chunk": 0}, (0, 4096), True)),
        ("gfx950", TG.Step("dp", {**gemm_cdna_dp.DEFAULT, **gemm_cdna.CANDIDATES[
            "dp32x96nt"][1]}, (0, 6144), True)),
        ("gfx950", TG.Step("decode", dict(tokens=1, slices=4, warps=16, combine_depth=0,
                                          resident=2, splits=1), (0, 4096), True)),
        ("gfx950", TG.Step("decode", dict(tokens=1, slices=4, warps=16, combine_depth=0,
                                          resident=2, splits=2), (0, 4096), True)),
    ]


@pytest.mark.parametrize("arch,step", _steps(), ids=lambda x: str(x).split(" ")[0])
@pytest.mark.parametrize("which", ["bias_gelu", "residual", "rowscale", "own"])
def test_the_epilogue_traces_into_every_kernel(arch, step, which):
    ep = {
        "bias_gelu": lambda: _ep(T.bias_gelu, {"bias": "row"}),
        "residual": lambda: _ep(T.residual, {"residual": "tile"}),
        "rowscale": lambda: _ep(T.scale, {"scale": "col"}, {"scale": hk.fp32}, out=hk.fp32),
        "own": lambda: _ep(bias_gelu_own, {"bias": "row"}),
    }[which]()
    k = T.kernel_for(step, ep, arch=arch)
    src = k.source()
    assert ep.tag in src
    names = [p.name for p in k.trace().tensors()]
    assert names[-len(ep.tensors):] == ["ep_" + n for n in ep.tensors]
    if which == "bias_gelu":
        # gemm.gelu: one exp2 and one rsqrt an element, not Tile.gelu()'s
        assert "kittens::exp2(" in src and "kittens::rsqrt(" in src
        assert "kittens::gelu(" not in src
    if which == "own":
        assert "kittens::gelu(" in src


def test_init_from_loads_the_accumulator_before_the_k_loop():
    """gemm.residual on a kernel that can: the accumulators are loaded from
    the residual before the first MMA, and nothing is added at the store."""
    ep = _ep(T.residual, {"residual": "tile"})
    for arch, step in _steps()[:4]:
        ir = T.kernel_for(step, ep, arch=arch).trace().dump()
        first_mma = min(i for i, l in enumerate(ir.splitlines()) if "hk.mma" in l)
        loads = [i for i, l in enumerate(ir.splitlines())
                 if re.search(r"hk.load .*%ep_residual", l)]
        assert loads and max(loads) < first_mma, (arch, step)
    # The decode GEMV sums its warps' partials, so the residual is added
    # once, at the store -- loaded before the K loop (Epilogue.early), its
    # latency under the weight stream.
    ir = T.kernel_for(_steps()[5][1], ep, arch="gfx950").trace().dump()
    lines = ir.splitlines()
    first_mma = min(i for i, l in enumerate(lines) if "hk.mma" in l)
    loads = [i for i, l in enumerate(lines) if re.search(r"hk.load .*%ep_residual", l)]
    adds = [i for i, l in enumerate(lines) if re.search(r"hk.add ", l)]
    assert loads and max(loads) < first_mma and max(adds) > first_mma


def test_a_split_kernel_takes_no_epilogue():
    ep = _ep(T.bias_gelu, {"bias": "row"})
    with pytest.raises(ValueError, match="reduction"):
        gemm_cdna.gemm_kernel("x", **{**gemm_cdna.DEFAULT, "splits": 2}, epilogue=ep,
                              arch="gfx950")
    with pytest.raises(ValueError, match="reduction"):
        gemm_cdna_dp.gemm_kernel("x", **{**gemm_cdna_dp.DEFAULT, "splits": 2},
                                 epilogue=ep, arch="gfx950")


def test_the_reduction_of_a_split_carries_the_epilogue():
    ep = _ep(T.bias_gelu, {"bias": "row"})
    k, consts = gemm_cdna_dp.reduce_epilogue_kernel(4, 4096, ep, arch="gfx950")
    src = k.source(**consts)
    assert ep.tag in src and "kittens::exp2(" in src
    assert [p.name for p in k.trace(**consts).tensors()] == ["ws", "c", "ep_bias"]


def test_an_operand_the_call_did_not_pass_is_named_at_trace():
    def wants_scale(acc, ctx):
        return acc * ctx.scale

    ep = _ep(wants_scale, {"bias": "row"})
    step = T.plan(1024, 4096, 4096, arch="gfx1100")[0]
    with pytest.raises(AttributeError, match=r"ctx.scale.*scale="):
        T.kernel_for(step, ep, arch="gfx1100").trace()


def test_an_epilogue_returns_a_tile_of_c():
    def reduces(acc, ctx):
        return acc.sum(axis=1)

    ep = _ep(reduces)
    step = T.plan(1024, 4096, 4096, arch="gfx1100")[0]
    with pytest.raises(TypeError, match="returned"):
        T.kernel_for(step, ep, arch="gfx1100").trace()


def test_coordinates_are_c_s_on_a_column_slice():
    """A launch on columns lo.. of C passes lo to an epilogue that reads
    ctx.col -- and only to one that does."""
    def mask(acc, ctx):
        return acc.right_fill_(1000 - ctx.col, 0.0)

    ep = _ep(mask)
    step = TG.Step("pp", gemm_cdna.schedule_for(4096, 4096, 1024), (0, 4096), True)
    k = T.kernel_for(step, ep, arch="gfx950")
    k.trace()
    assert ep.uses_coords
    assert "EP_COL0" in [p.name for p in k.trace().params]
    plain = _ep(T.bias_gelu, {"bias": "row"})
    T.kernel_for(step, plain, arch="gfx950").trace()
    assert not plain.uses_coords


def test_operand_kinds_follow_torch_broadcasting():
    torch = pytest.importorskip("torch")
    m, n = 64, 256
    assert TG._kind("b", torch.empty(n), m, n) == "row"
    assert TG._kind("b", torch.empty(1, n), m, n) == "row"
    assert TG._kind("s", torch.empty(m, 1), m, n) == "col"
    assert TG._kind("r", torch.empty(m, n), m, n) == "tile"
    assert TG._kind("r", torch.empty(1, n), 1, n) == "tile"   # C's own shape at M=1
    assert TG._kind("a", 0.5, m, n) == "const"
    with pytest.raises(ValueError, match="broadcast"):
        TG._kind("x", torch.empty(m), m, n)


def test_a_binding_is_named_the_same_in_every_process():
    a = _ep(T.bias_gelu, {"bias": "row"})
    TG._TAGS.pop(a.tag)
    b = _ep(T.bias_gelu, {"bias": "row"})
    assert a.tag == b.tag
    assert _ep(T.bias_gelu, {"bias": "row"}, out=hk.fp32).tag != b.tag

    def scaled(k):
        def f(acc):
            return acc * k
        return f

    assert _ep(scaled(2.0)).tag != _ep(scaled(3.0)).tag
    assert _ep(T.chain(T.bias, T.gelu), {"bias": "row"}).tag != \
        _ep(T.chain(T.bias, T.silu), {"bias": "row"}).tag
