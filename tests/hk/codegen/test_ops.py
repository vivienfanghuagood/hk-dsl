"""Every shipped Phase 2 kernel: does it compile, and does it spill.

Needs hipcc, not a GPU. This is the tier that would have caught every failure
Phase 2 actually produced -- an undeclared `inf`, a `gl<int8>` that did not
instantiate, a grid expression naming a parameter the kernel does not have.
None of those needs hardware to find, and all of them cost a pod round trip if
they are found there instead.

Cold, this is a couple of minutes of hipcc for ~20 kernels. Warm it is free:
the content-hash cache keys on the include tree, so it only pays again when
include/rdna3 changes.
"""

from __future__ import annotations

import pytest

from hk.ops import fused, norm, quant

#: name -> kernel, across the three Phase 2 modules. Parametrizing over the
#: union rather than per module means a kernel added to any of them is covered
#: without touching this file.
KERNELS = {
    **{f"norm.{n}": k for n, k in norm.KERNELS.items()},
    **{f"fused.{n}": k for n, k in fused.KERNELS.items()},
    **{f"quant.{n}": k for n, k in quant.KERNELS.items()},
}


def test_the_sweep_is_not_empty():
    """A parametrization over an accidentally-empty dict is a green run that
    tested nothing."""
    assert len(KERNELS) >= 15


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_it_compiles_without_spilling(name):
    """Spilling is not slowness here. include/rdna3 places `s_waitcnt` by hand
    and scratch traffic reorders against it, so a spilling kernel computes the
    wrong answer at full speed. hk.build() is supposed to refuse; this asserts
    it has nothing to refuse."""
    b = KERNELS[name].build()
    assert b.kernels, f"{name}: no resource remarks parsed -- the gate is vacuous"
    for k in b.kernels:
        assert k.scratch == 0 and not k.spills, f"{name}: {k.line()}"


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_it_keeps_a_workable_occupancy(name):
    """These are bandwidth-bound passes: they need enough waves in flight to
    cover a global load's latency. 4/16 is well under anything that would be
    worth shipping, so it is a floor, not a target."""
    for k in KERNELS[name].build().kernels:
        assert (k.occ_by_reg or k.occupancy) >= 4, f"{name}: {k.line()}"


#: The kernels that can be asked for norm.MAX_TPW live tiles. `plan` only ever
#: emits the top of the range at WARPS=16 -- a row of 65 to 80 blocks split
#: sixteen ways -- so that is what has to build, and the name carries the warp
#: count.
TOP_TPW = sorted(n for n, k in KERNELS.items()
                 if n.endswith("_w16") and "TPW" in k.signature.parameters)


def test_the_top_of_the_tpw_range_is_a_real_kernel():
    """An empty parametrization would make the next test vacuous, and the
    filter it depends on is a string match on a kernel name."""
    assert len(TOP_TPW) >= 9   # rms/layer/softmax/quantize x bf16/fp16/fp32


@pytest.mark.parametrize("name", TOP_TPW)
def test_the_top_of_the_tpw_range_does_not_spill(name):
    """What makes norm.MAX_TPW a measurement rather than a hope.

    The constant is the number of live fp32 tiles a warp may hold. Nothing
    else in the suite asks for it: every other build here takes the default
    TPW=0, so a cap set one too high would compile fine everywhere and then be
    refused at run time by the one shape that asks for the top of the range.

    **EXACT=0 is the point of this test.** The exact variant has had its clamp
    and its mask deleted at trace time and is the smaller of the two by enough
    to matter -- at TPW=5 the exact softmax builds and the inexact one does
    not. plan() emits TPW=MAX_TPW at widths that are not a whole number of
    blocks, so the inexact variant is the one that has to hold, and a version
    of this test written with EXACT=1 passed while the sweep was skipping the
    same kernel as unbuildable.
    """
    b = KERNELS[name].build(TPW=norm.MAX_TPW, FOLD=1, EXACT=0)
    assert b.kernels, f"{name}: no resource remarks parsed"
    for k in b.kernels:
        assert k.scratch == 0 and not k.spills, f"{name}: {k.line()}"


def test_a_norm_kernel_is_one_workgroup_per_row_block():
    """The grid is the contract with the tail idiom: one block per ROWS rows,
    each looping the full width. A y-extent computed from the wrong tensor is
    silently a partial result."""
    src = norm.KERNELS["rms_bf16_w1"].source()
    assert f"g.x.rows() + {norm.ROWS}) - 1) / {norm.ROWS}" in src


def test_the_quantized_store_goes_through_the_narrowing_convertor():
    """Nothing in the kernel rounds or clamps -- convertor<int8, float> does,
    inside kittens::store. If the tile were int8 instead, the C cast would
    truncate toward zero and wrap on an outlier."""
    src = quant.KERNELS["quantize_bf16_w1"].source()
    assert "kittens::store(g.q," in src
    assert "kittens::rt<int8" not in src


def _store_histogram(kernel) -> dict:
    """Which global store instructions the kernel actually ends up with.

    Costs one extra hipcc invocation (-S, device-only), which is why it is not
    part of the per-kernel sweep above. It is the only way to see the thing it
    checks: whether a store merged is a property of the emitted ISA, invisible
    in the C++ and invisible to the resource remarks.
    """
    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from hk.runtime.compile import flags, hipcc
    from hk.target import gfx1100

    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "k.cpp"
        asm = Path(d) / "k.s"
        cpp.write_text(kernel.source(scaffold="bare"))
        cmd = [
            hipcc(),
            *[f for f in flags(gfx1100.GFX1100, "probe")
              if f not in ("-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage")],
            "--offload-device-only", "-S", str(cpp), "-o", str(asm),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        text = asm.read_text()
    hist = {}
    for m in re.finditer(r"\bglobal_store_(\w+)", text):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    return hist


def test_a_narrowing_store_does_not_go_out_one_byte_at_a_time():
    """The f32 accumulator layout interleaves the two wave halves along the
    element axis -- lane l holds columns 2e, lane l+16 holds 2e+1 -- so no lane
    owns a contiguous run and nothing merges. Storing that tile into an int8
    global came out as 32 `global_store_b8` per tile and ran 3.8x slower than
    the same kernel writing twice as many bytes as bf16 (16384x4096: 2.25 ms vs
    0.59 ms). store_at's `packable` path does the interleave in registers
    instead; this asserts it is still being taken.
    """
    hist = _store_histogram(quant.KERNELS["quantize_bf16_w1"])
    assert hist.get("b128", 0) >= 4, hist
    # The b8 fallback is still in the binary -- alignment is a runtime property
    # of row_stride, so both arms are emitted -- but it must not be alone.
    assert hist.get("b128", 0) * 16 >= hist.get("b8", 0), hist


def test_a_same_width_store_is_left_alone():
    """`packable` is gated on the destination being *narrower*. At equal width
    the register interleave would cost 16 shuffles to replace 8 plain dword
    stores, so the fp32 path must stay on the elementwise loop."""
    hist = _store_histogram(norm.KERNELS["rms_fp32_w1"])
    assert hist.get("b32", 0) > 0 and "b128" not in hist, hist
