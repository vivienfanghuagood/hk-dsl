"""hk.templates.elementwise and hk.templates.rowwise without a GPU: the plan
each picks on gfx1100 and gfx950 (hk.ops' measured plans, reached, not
copied), what the hooks trace to, the operand kinds a call infers, and what
the templates refuse with a reason.

No GPU, no hipcc: operands are hk.templates.meta stand-ins.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import templates as T  # noqa: E402
from hk.ops import elementwise as ew, fused, norm, quant  # noqa: E402

B = hk.bf16
ARCHS = ("gfx1100", "gfx950")


def gelu_mul(arch):
    return T.elementwise(lambda g, u: hk.math.gelu(g) * u, arch=arch)


def halves(m, i, dtype=B):
    return (T.meta((m, i), dtype, (2 * i, 1)), T.meta((m, i), dtype, (2 * i, 1)))


def rms_norm(arch):
    @T.rowwise(arch=arch)
    def rms(x, w, *, EPS=1e-6):
        return x * (T.row_mean(x * x) + EPS).rsqrt() * w
    return rms


def softmax(arch, **kw):
    @T.rowwise(arch=arch, **kw)
    def sm(x):
        e = (x - T.row_max(x)).exp()
        return e / T.row_sum(e)
    return sm


# ---------------------------------------------------------------- the package


def test_a_template_name_stays_the_template_after_its_module_is_imported():
    # Importing hk.templates.rowwise binds the module to the package's
    # `rowwise` attribute; the template has to win on every later lookup.
    assert callable(T.rowwise) and callable(T.rowwise)
    assert callable(T.elementwise) and callable(T.elementwise)
    assert T.row_sum is T.row_sum and callable(hk.templates.per_col)


# ---------------------------------------------------------------- elementwise plans


def test_elementwise_on_halves_takes_silu_muls_plan_on_gfx950():
    p = gelu_mul("gfx950").plan(*halves(4096, 14336))
    assert not p.flat and (p.warps, p.chunks) == fused.STRIP_PLAN and p.run == 8
    assert p.exact and p.grid == (4096 * 14336 // 512, 1, 1)


def test_elementwise_on_contiguous_operands_is_one_flat_row_on_gfx950():
    a = T.meta((8192, 8192), B)
    p = gelu_mul("gfx950").plan(a, a)
    moved = 8192 * 8192 * 2 * 3
    assert p.flat and p.rows == 1 and p.cols == 8192 * 8192
    assert (p.warps, p.chunks) == ew.flat_strip_plan(moved, True)


def test_a_ragged_problem_is_bounded_and_keeps_its_runs_aligned():
    p = gelu_mul("gfx950").plan(*halves(3, 1001))
    assert not p.flat and not p.exact and p.run == 1      # 1001 is odd: runs of one
    p = gelu_mul("gfx1100").plan(*halves(5, 4100))
    assert not p.exact and p.run == 4                     # 4100 = 4 x 1025


@pytest.mark.parametrize("arch", ARCHS)
def test_the_plan_can_be_overridden(arch):
    k = gelu_mul(arch).with_plan(warps=2, chunks=4)
    p = k.plan(*halves(64, 8192))
    assert (p.warps, p.chunks) == (2, 4)
    with pytest.raises(TypeError, match="overrides"):
        T.elementwise(lambda a: a, plan={"tiles": 2})


@pytest.mark.parametrize("arch", ARCHS)
def test_elementwise_traces_a_strip_kernel_for_each_arch(arch):
    t = gelu_mul(arch)
    k, c = t.kernel(*halves(37, 9216))
    src = k.source(**c)
    wave = 64 if arch == "gfx950" else 32
    p = t.plan(*halves(37, 9216))
    assert f"rs_fp32_1x{wave * 8}_r8" in src and k.warps == p.warps
    assert c["T_CW"] == wave * 8 and c["T_EXACT"] == int(9216 % (p.warps * c["T_CW"]) == 0)
    assert (p.warps, p.chunks, p.reserve) == fused.silu_strip_plan(arch, 37 * 9216 * 6)


def test_operand_kinds_follow_torch_broadcasting():
    k = T.elementwise(lambda x, b, s, t: x * s + b * t, arch="gfx950")
    ops, shape, outs = k._resolve((T.meta((2, 3, 4096), B), T.meta((4096,), B),
                                   T.meta((2, 3, 1), hk.fp32), T.meta((1,), hk.fp32)))
    assert [o.kind for o in ops] == ["full", "per_col", "per_row", "scalar"]
    assert shape == (2, 3, 4096) and outs == (B,)
    kk, c = k.kernel(T.meta((6, 4096), B), T.meta((4096,), B), T.meta((6,), hk.fp32),
                     T.meta((), hk.fp32))
    assert "rs_fp32" in kk.source(**c)


def test_declared_operands_are_checked():
    k = T.elementwise(lambda x, b: x + b, inputs=(B, T.per_col(B)), arch="gfx950")
    with pytest.raises(TypeError, match="declared bf16"):
        k.plan(T.meta((4, 64), hk.fp32), T.meta((64,), B))
    with pytest.raises(TypeError, match="takes 2 operand"):
        k.plan(T.meta((4, 64), B))


def test_an_operand_must_be_a_rows_view():
    k = gelu_mul("gfx950")
    with pytest.raises(ValueError, match="last axis must be contiguous"):
        k.plan(T.meta((4, 64), B, (1, 4)), T.meta((4, 64), B))
    with pytest.raises(ValueError, match="one row stride"):
        k.plan(T.meta((2, 3, 64), B, (1000, 128, 1)), T.meta((2, 3, 64), B))


def test_an_elementwise_hook_may_not_reduce():
    k = T.elementwise(lambda x: x * x.sum(axis=1), arch="gfx950")
    kk, c = k.kernel(T.meta((4, 512), B))
    with pytest.raises(TypeError, match="row_sum"):
        kk.trace(**c)


def test_several_outputs_and_their_dtypes():
    k = T.elementwise(lambda x: (x * 2.0, x + 1.0), outputs=(hk.fp32, hk.i8), arch="gfx950")
    kk, c = k.kernel(T.meta((4, 512), B))
    src = kk.source(**c)
    assert src.count("store") >= 2
    two = T.elementwise(lambda x: (x, x), outputs=2, arch="gfx950")
    assert two._resolve((T.meta((4, 512), B),))[2] == (B, B)


# ---------------------------------------------------------------- rowwise plans


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("rows,cols", [(1, 4096), (2048, 4096), (8192, 8192), (7, 4100)])
def test_rms_norm_takes_hk_ops_plan(arch, rows, cols):
    p = rms_norm(arch).plan(T.meta((rows, cols), B), T.meta((cols,), B))
    assert p.like == "rmsw" and not p.steps
    assert (p.warps, p.chunks, p.run, int(p.exact), p.reserve) == \
        norm.strip_plan_for("rmsw", 2, cols, rows, arch)


@pytest.mark.parametrize("arch", ARCHS)
def test_like_follows_the_hook(arch):
    x = T.meta((2048, 4096), B)
    assert softmax(arch).plan(x).like == "softmax"            # a row max

    @T.rowwise(arch=arch)
    def layer(x, *, EPS=1e-5):
        xc = x - T.row_mean(x)
        return xc * (T.row_mean(xc * xc) + EPS).rsqrt()
    assert layer.plan(x).like == "layer"                        # two sums

    @T.rowwise(arch=arch, outputs=2)
    def add_rms(x, r, w):
        h = x + r
        return h * T.row_mean(h * h).rsqrt() * w, h
    assert add_rms.plan(x, x, T.meta((4096,), B)).like == "addrmsw"

    @T.rowwise(arch=arch, outputs=(hk.i8, T.per_row(hk.fp32)))
    def quantize(x):
        amax = T.row_max(abs(x)).maximum(1e-12)
        return x * (127.0 / amax), amax / 127.0
    p = quantize.plan(x)
    assert p.like == "quantize"
    assert (p.warps, p.chunks, p.run, int(p.exact)) == quant.strip_plan_for(2, 4096, 2048, arch)


@pytest.mark.parametrize("arch", ARCHS)
def test_a_row_too_wide_to_hold_streams(arch):
    x = T.meta((64, 151936), B)
    p = softmax(arch).plan(x)
    wave = 64 if arch == "gfx950" else 32
    assert p.steps == -(-151936 // (p.warps * p.chunks * wave * p.run)) and not p.exact
    k, c = softmax(arch).kernel(x)
    src = k.source(**c)
    assert src.count("for (") >= 3        # a max pass, a sum pass, a write pass


@pytest.mark.parametrize("arch", ARCHS)
def test_cross_entropy_picks_its_label(arch):
    @T.rowwise(arch=arch, outputs=T.per_row(hk.fp32))
    def ce(logits, label):
        m = T.row_max(logits)
        lse = m + T.row_sum((logits - m).exp()).log()
        return (lse - T.pick(logits, label)) * (label != -100).to_int()

    for v in (4096, 151936):
        k, c = ce.kernel(T.meta((64, v), B), T.meta((64,), hk.i32))
        assert "store_scalar" in k.source(**c) or "store" in k.source(**c)
    ops, _, outs = ce._resolve((T.meta((8, 4096), B), T.meta((8,), "torch.int64")))
    assert ops[1].wide and ops[1].dtype is hk.i32 and outs[0].kind == "per_row"


def test_a_rowwise_hook_reduces_through_the_template():
    @T.rowwise(arch="gfx950")
    def bad(x):
        return x * x.sum(axis=1)
    with pytest.raises(TypeError, match="row_sum"):
        bad.kernel(T.meta((4, 4096), B))
    with pytest.raises(RuntimeError, match="inside a"):
        T.row_sum(None)


def test_a_per_row_output_must_be_declared():
    @T.rowwise(arch="gfx950")
    def lse(x):
        return T.row_max(x)
    with pytest.raises(TypeError, match="per_row"):
        lse.kernel(T.meta((4, 4096), B))


def test_the_rowwise_plan_can_be_overridden():
    p = rms_norm("gfx950").with_plan(warps=8, chunks=1).plan(T.meta((64, 4096), B),
                                                            T.meta((4096,), B))
    assert (p.warps, p.chunks, p.steps) == (8, 1, 0)
    p = softmax("gfx950").with_plan(steps=4).plan(T.meta((64, 4096), B))
    assert p.steps == 4
    with pytest.raises(ValueError, match="do not hold"):
        rms_norm("gfx950").with_plan(warps=1, chunks=1).plan(T.meta((4, 4096), B),
                                                             T.meta((4096,), B))


def test_rowwise_refuses_a_target_without_strips():
    with pytest.raises(NotImplementedError, match="strips"):
        rms_norm("gfx942").plan(T.meta((4, 4096), B), T.meta((4096,), B))


# ---------------------------------------------------------------- the operators


def test_a_row_value_broadcasts_on_either_side():
    @T.rowwise(arch="gfx950")
    def f(x):
        m = T.row_max(x)
        s = T.row_sum(x)
        return (m - x) + s * x + 1.0 - x / s + 2.0 / (abs(x) + 1.0)
    k, c = f.kernel(T.meta((4, 4096), B))
    src = k.source(**c)
    assert "rs_fp32" in src


@pytest.mark.parametrize("arch", ARCHS)
def test_a_per_row_output_of_k_values(arch):
    k = 4

    @T.rowwise(arch=arch, outputs=(T.per_row(hk.i32, k), T.per_row(hk.fp32, k)))
    def topk(logits, col):
        p = (logits - T.row_logsumexp(logits)).exp()
        ids, ws = [], []
        for _ in range(k):
            m = T.row_max(p)
            i = T.row_min(col + (m - p).step() * 1e9)
            ids.append(i)
            ws.append(m)
            p = p - (1.0 - (col - i).abs().step()) * 2.0
        return ids, ws

    args = (T.meta((16, 128), B), T.meta((128,), hk.fp32))
    _, _, outs = topk._resolve(args)
    assert [o.width for o in outs] == [k, k]
    kern, c = topk.kernel(*args)
    assert kern.source(**c).count("store_scalar") >= 2 * k
    with pytest.raises(ValueError, match="at least one"):
        T.per_row(hk.fp32, 0)
