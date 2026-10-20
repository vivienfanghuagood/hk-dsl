"""hk.templates.gemm on the GPU, against fp32 torch: every ready-made
epilogue and the operand kinds, through every kernel the routes use on this
GPU -- including the ones a split-K, a wave split and a 128-column tail
take, where the epilogue has to run once on the sum and see C's columns.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

import hk  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = pytest.mark.gpu

T = hk.templates.gemm
TG = importlib.import_module("hk.templates.gemm")
ARCH = current_arch()


def _data(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return a, b, a.float() @ b.float().t()


def _close(got, ref, rel=1e-2):
    err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < rel, f"rel err {err:.3g}"


def mask_cols(acc, ctx):
    """C[:, 1000:] = 0, through the tile's coordinates."""
    return acc.right_fill_(1000 - ctx.col, 0.0)


def _cases(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(7)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    res = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)
    rs = torch.rand(m, 1, device="cuda", generator=g) + 0.5
    gelu = lambda x: F.gelu(x, approximate="tanh")  # noqa: E731
    return {
        "bias": (T.bias, dict(bias=bias), lambda c: c + bias.float()),
        "bias_gelu": (T.bias_gelu, dict(bias=bias), lambda c: gelu(c + bias.float())),
        "bias_silu": (T.bias_silu, dict(bias=bias), lambda c: F.silu(c + bias.float())),
        "residual": (T.residual, dict(residual=res), lambda c: c + res.float()),
        "bias_residual": (T.bias_residual, dict(bias=bias, residual=res),
                          lambda c: c + bias.float() + res.float()),
        "row_scale": (T.scale, dict(scale=rs), lambda c: c * rs),
        "number": (T.scale, dict(scale=0.25), lambda c: c * 0.25),
        "own_gelu": (lambda acc, ctx: (acc + ctx.bias).gelu(), dict(bias=bias),
                     lambda c: gelu(c + bias.float())),
        "coords": (mask_cols, {}, lambda c: torch.cat([c[:, :1000], c[:, 1000:] * 0], 1)),
        "chain": (T.chain(T.residual, T.relu), dict(residual=res),
                  lambda c: F.relu(c + res.float())),
        # a tile read at the store (not init_from): loaded a band ahead
        "tile_mul": (lambda acc, ctx: acc * ctx.gate, dict(gate=res),
                     lambda c: c * res.float()),
    }


NAMES = ["bias", "bias_gelu", "bias_silu", "residual", "bias_residual", "row_scale",
         "number", "own_gelu", "coords", "chain", "tile_mul"]

SHAPES = [(1, 4096, 4096), (33, 6144, 4096), (128, 4096, 4096), (200, 1408, 640),
          (1024, 4096, 4096), (2048, 12288, 4096), (4096, 4224, 1024)]


@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("name", NAMES)
def test_the_routed_plan_is_right(m, n, k, name):
    if ARCH != "gfx950" and n % 128:
        pytest.skip("N a multiple of 128")
    a, b, acc = _data(m, n, k)
    fn, ops, ref = _cases(m, n, k)[name]
    _close(T(a, b, epilogue=fn, **ops), ref(acc))


def _candidates():
    if ARCH != "gfx950":
        return []
    from hk.ops import gemm_cdna  # noqa: PLC0415

    return sorted(gemm_cdna.CANDIDATES)


@pytest.mark.skipif(ARCH != "gfx950", reason="gfx950's route candidates")
@pytest.mark.parametrize("cand", _candidates())
@pytest.mark.parametrize("name", ["bias_gelu", "residual", "coords", "tile_mul"])
def test_every_gfx950_candidate_carries_the_epilogue(cand, name, monkeypatch):
    """Each candidate the route table can name, forced: the epilogue in its
    store, or in the reduction of its split, right against torch."""
    from hk.ops import gemm_cdna  # noqa: PLC0415

    m, n, k = (300, 1408, 1024) if cand in ("ppw", "ppwd") else (257, 1536, 2048)
    if cand in ("ppw", "ppwd"):
        monkeypatch.setattr(gemm_cdna, "CUS", 2)
    if not gemm_cdna.fits(cand, m, n, k):
        m, n, k = 64, 1536, 4096
    if not gemm_cdna.fits(cand, m, n, k):
        pytest.skip(f"{cand} takes neither shape")
    monkeypatch.setattr(gemm_cdna, "route", lambda *_: cand)
    TG._LAUNCH.clear()
    a, b, acc = _data(m, n, k)
    fn, ops, ref = _cases(m, n, k)[name]
    try:
        _close(T(a, b, epilogue=fn, **ops), ref(acc))
    finally:
        TG._LAUNCH.clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_out_dtypes(dtype):
    a, b, acc = _data(300, 4096, 1024)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32)
    got = T(a, b, epilogue=T.bias, out_dtype=dtype, bias=bias)
    assert got.dtype == dtype
    _close(got, acc + bias, rel=1e-2 if dtype != torch.float32 else 1e-5)


@pytest.mark.skipif(ARCH != "gfx950", reason="an int8 C is gfx950's")
def test_an_int8_quantizer_rounds_and_saturates():
    a, b, acc = _data(128, 4096, 4096)
    s = torch.full((4096,), 40.0, device="cuda")
    got = T(a, b, epilogue=T.scale, out_dtype=torch.int8, scale=s)
    ref = (acc * 40.0).round().clamp(-128, 127)
    assert got.dtype == torch.int8
    assert (got.float() - ref).abs().max().item() <= 1.0     # bf16 vs fp32 GEMM


@pytest.mark.skipif(ARCH != "gfx950", reason="an fp8 C is gfx950's")
def test_an_fp8_quantizer_rounds_and_saturates():
    a, b, acc = _data(300, 4096, 1024)
    s = torch.rand(4096, device="cuda") * 900          # some past 448
    got = T(a, b, epilogue=T.scale, out_dtype=torch.float8_e4m3fn, scale=s)
    assert got.dtype == torch.float8_e4m3fn
    want = (acc * s).clamp(-448, 448)
    big = want.abs() > 16
    rel = ((got.float() - want).abs() / want.abs())[big]
    assert rel.max().item() < 0.07                       # e4m3: 3 mantissa bits
    assert got.float().abs().max().item() == 448.0


def test_out_is_written_in_place_and_nothing_past_m():
    a, b, acc = _data(100, 4096, 1024)
    room = torch.full((108, 4096), 7.0, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(100, 4096, device="cuda", dtype=torch.bfloat16)
    got = T(a, b, epilogue=T.residual, out=room[:100], residual=res)
    assert got.data_ptr() == room.data_ptr()
    _close(room[:100], acc + res.float())
    assert torch.equal(room[100:], torch.full_like(room[100:], 7.0))


def test_the_bound_form_and_an_in_place_residual():
    """gemm(epilogue=...) without tensors binds for later; out may be the
    residual itself (each tile is read before it is written)."""
    a, b, acc = _data(512, 4096, 1024)
    lin = T(epilogue=T.residual)
    res = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
    want = acc + res.float()
    got = lin(a, b, out=res, residual=res)
    _close(got, want)


def test_a_column_operand_at_a_ragged_m():
    """A per-row vector is padded to whole blocks for the vector load."""
    for m in (1, 17, 255, 257):
        a, b, acc = _data(m, 4096, 1024)
        s = torch.rand(m, 1, device="cuda") + 0.5
        _close(T(a, b, epilogue=T.scale, scale=s), acc * s)


def heavy(acc, ctx):
    """Four tile operands and a GELU each: more than a 64x64 pingpong
    quadrant or the decode GEMV's combine has registers for."""
    for t in (ctx.r0, ctx.r1, ctx.r2, ctx.r3):
        acc += T.gelu_tanh(acc * t)
    return acc


def test_an_epilogue_that_spills_runs_in_a_pass_of_its_own():
    """The build refuses a spill; the template then stores fp32 and runs the
    epilogue in a separate pass -- slower, right, and said."""
    import warnings  # noqa: PLC0415

    gelu = lambda x: F.gelu(x, approximate="tanh")  # noqa: E731
    for (m, n, k) in [(4, 4096, 4096), (4096, 4096, 1024)]:
        a, b, acc = _data(m, n, k)
        rs = [torch.randn(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            got = T(a, b, epilogue=heavy, r0=rs[0], r1=rs[1], r2=rs[2], r3=rs[3])
        want = acc
        for t in rs:
            want = want + gelu(want * t.float())
        _close(got, want, rel=2e-2)


def test_misuse_is_named():
    a, b, _ = _data(64, 4096, 1024)
    with pytest.raises(ValueError, match="broadcast"):
        T(a, b, epilogue=T.bias, bias=torch.zeros(64, device="cuda"))
    with pytest.raises(AttributeError, match="ctx.bias"):
        T(a, b, epilogue=T.bias, scale=1.0)
    with pytest.raises(TypeError, match="bf16"):
        T(a.float(), b, epilogue=T.bias, bias=torch.zeros(4096, device="cuda"))
