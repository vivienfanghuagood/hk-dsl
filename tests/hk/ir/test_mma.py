"""The four WMMA variants: what they accept, and what they refuse.

gfx1100 WMMA takes its operands in fixed fragment layouts, so the choice among
the four names is not a convenience -- it is how a kernel gets a transpose for
free, and getting it wrong is either a wrong answer or eight times the LDS
instructions. Every rule asserted here is transcribed from the static_asserts
in include/rdna3/ops/warp/register/tile/mma.cuh; the point of checking them at
trace time is that the C++ ones fail forty lines deep in a template that names
`rt_base<...>` and not the line that wrote it.
"""

import pytest

import hk
from hk import bf16, fp16, fp32
from hk.lang import ops


def _trace(body, **kw):
    k = hk.kernel(body, arch="gfx1100", warps=1,
                  grid=lambda p: (1, 1, 1), name=body.__name__)
    return k.trace(**kw)


def _acc(rows, cols):
    return ops.zeros(ops.rt(fp32, rows, cols, "col"))


# -- the shapes each variant produces ----------------------------------------


@pytest.mark.parametrize(
    "variant,a,b,want",
    [
        # A is (M,K) row, B is (K,N) col  -> (M,N)
        ("mma_AB", (64, 32, "row"), (32, 128, "col"), (64, 128)),
        # A is (M,K) row, B is (N,K) row  -> (M,N).  The GEMM in
        # kernels/rdna3/gemm passes B pre-transposed for exactly this reason.
        ("mma_ABt", (64, 32, "row"), (128, 32, "row"), (64, 128)),
        # A is (K,M) col, B is (K,N) col  -> (M,N)
        ("mma_AtB", (32, 64, "col"), (32, 128, "col"), (64, 128)),
        # A is (K,M) col, B is (N,K) row  -> (M,N).  attn.cpp's second mma.
        ("mma_AtBt", (32, 64, "col"), (128, 32, "row"), (64, 128)),
    ],
)
def test_each_variant_produces_the_shape_its_asserts_promise(variant, a, b, want):
    fn = getattr(ops, variant)

    def body(o: hk.GL[fp32]):
        at = ops.zeros(ops.rt(bf16, *a))
        bt = ops.zeros(ops.rt(bf16, *b))
        d = fn(at, bt, _acc(*want))
        assert d.type == ops.rt(fp32, *want, "col"), d.type

    body.__name__ = variant
    _trace(body)


def test_the_variant_is_chosen_by_the_layouts_you_already_have():
    """The error for a layout mismatch names the variant that does apply.

    That is the whole content of the four-way choice: a kernel does not decide
    to transpose, it decides how to stage, and the staging picks the name.
    """
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(bf16, 128, 32, "row"))   # row, so ABt applies
        ops.mma_AB(a, b, _acc(64, 128))

    with pytest.raises(TypeError, match="mma_ABt"):
        _trace(body)


# -- what WMMA on this architecture cannot do --------------------------------


def test_an_fp32_operand_is_refused():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(fp32, 64, 32, "row"))
        b = ops.zeros(ops.rt(fp32, 128, 32, "row"))
        ops.mma_ABt(a, b, _acc(64, 128))

    with pytest.raises(TypeError, match="bf16 or"):
        _trace(body)


def test_a_mixed_operand_pair_is_refused():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(fp16, 128, 32, "row"))
        ops.mma_ABt(a, b, _acc(64, 128))

    with pytest.raises(TypeError, match="one operand type"):
        _trace(body)


def test_a_row_layout_accumulator_is_refused():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(bf16, 128, 32, "row"))
        ops.mma_ABt(a, b, ops.zeros(ops.rt(fp32, 64, 128, "row")))

    with pytest.raises(TypeError, match="col"):
        _trace(body)


def test_a_disagreeing_reduction_extent_is_refused():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(bf16, 128, 48, "row"))
        ops.mma_ABt(a, b, _acc(64, 128))

    with pytest.raises(TypeError, match="reduction extents"):
        _trace(body)


def test_an_accumulator_of_the_wrong_shape_is_refused():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(bf16, 128, 32, "row"))
        ops.mma_ABt(a, b, _acc(64, 64))

    with pytest.raises(TypeError, match="accumulator is"):
        _trace(body)


# -- codegen -----------------------------------------------------------------


def test_the_inplace_form_passes_the_accumulator_as_both_d_and_c():
    """`out=acc` is the GEMM inner loop, and it must not re-declare acc.

    A fresh declaration inside a loop body would shadow the loop-carried
    accumulator and silently keep only the last iteration -- the exact class of
    bug destination-passing form exists to prevent.
    """
    def gemm_inner(o: hk.GL[fp32]):
        acc = _acc(64, 128)
        for _ in hk.range(4):
            a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
            b = ops.zeros(ops.rt(bf16, 128, 32, "row"))
            ops.mma_ABt(a, b, acc, out=acc)

    src = hk.kernel(gemm_inner, arch="gfx1100", warps=1,
                    grid=lambda p: (1, 1, 1), name="gemm_inner").source()
    call = [ln.strip() for ln in src.splitlines() if "mma_ABt" in ln]
    assert len(call) == 1, src
    # d and c are the same name, and the name is declared outside the loop.
    d, a, b, c = call[0].split("(")[1].split(")")[0].split(", ")
    assert d == c and d != a and d != b, call[0]
    decl = [ln for ln in src.splitlines() if ln.strip().endswith(f" {d};")]
    assert len(decl) == 1, f"{d} declared {len(decl)} times"


def test_the_value_form_declares_a_fresh_destination():
    def body(o: hk.GL[fp32]):
        a = ops.zeros(ops.rt(bf16, 64, 32, "row"))
        b = ops.zeros(ops.rt(bf16, 128, 32, "row"))
        ops.mma_ABt(a, b, _acc(64, 128))

    src = hk.kernel(body, arch="gfx1100", warps=1,
                    grid=lambda p: (1, 1, 1), name="mma_value").source()
    call = next(ln for ln in src.splitlines() if "mma_ABt" in ln)
    d, _, _, c = call.split("(")[1].split(")")[0].split(", ")
    assert d != c, call
