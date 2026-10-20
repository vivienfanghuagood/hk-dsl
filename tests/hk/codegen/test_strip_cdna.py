"""The strip kernels on gfx950, through hipcc: hk's wave64 port of kittens::rs
(include/hk/cdna.cuh) compiles every shipped strip kernel without spilling,
and the ISA is the one the layout exists for -- 16-byte accesses, a row
reduction of DPP and the two permlane swaps, no ds_bpermute.

Needs hipcc, not a GPU.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.ops import elementwise, fused, norm, quant  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.target import gfx950  # noqa: E402


def _for_950(k):
    return hk.kernel(k.fn, arch="gfx950", warps=k.warps, grid=k.grid_fn,
                     name=k.name + "_950", defines=getattr(k, "defines", None),
                     noalias=k.noalias, preload=k.preload)


#: (kernel, constexprs): each kind at its 4096-column bf16 plan, a bounded
#: ragged width, and the pointwise and RoPE kernels hk.ops runs on gfx950.
CASES = [(norm.STRIP_KERNELS[f"strip_{k}_bf16_w2"], dict(CHUNKS=4, RUN=8, EXACT=1))
         for k in ("rms", "layer", "softmax", "rmsw", "addrmsw")]
CASES += [(norm.STRIP_KERNELS["strip_addrmsw_bf16_w3"], dict(CHUNKS=6, RUN=4, EXACT=0)),
          (norm.STRIP_KERNELS["strip_layer_fp32_w2"], dict(CHUNKS=8, RUN=4, EXACT=1)),
          (quant.STRIP_KERNELS["strip_quantize_bf16_w2"], dict(CHUNKS=4, RUN=8, EXACT=1)),
          (quant.STRIP_KERNELS["strip_quantize_bf16_w3"], dict(CHUNKS=6, RUN=4, EXACT=0)),
          (elementwise.STRIP_KERNELS["strip_add_bf16_w8"], dict(CW=4096, RUN=8, EXACT=1)),
          (elementwise.STRIP_KERNELS["strip_exp_bf16_w1"], dict(CW=1024, RUN=8, EXACT=0)),
          (fused.STRIP_KERNELS["strip_silu_mul_bf16_w1"], dict(CW=512, RUN=8, EXACT=0)),
          (fused.ROTARY_KERNELS["rotary_bf16_w8_pos"], dict(H=32, D=128, RUN=2))]


@pytest.mark.hipcc
@pytest.mark.parametrize("k,consts", CASES, ids=[c[0].name for c in CASES])
def test_a_strip_kernel_builds_for_gfx950_without_spilling(k, consts):
    """Six waves a SIMD or more: 32 elements a lane is 68 VGPRs for the
    weighted norm (7 waves) and 76-78 for the fused add (6), the plans
    measured fastest (norm._STRIP_PER_LANE_WAVE64)."""
    k = _for_950(k)
    b = hc.build(k.with_defines(scaffold.device(k.trace(**consts))), "gfx950",
                 name=k.name, kind="co")
    (r,) = b.kernels
    assert r.scratch == 0 and not r.spills, r.line()
    assert r.occ_by_reg >= 6, r.line()


def _asm(k, **consts) -> str:
    """The whole assembly of `k` as gfx950 builds it, its own flags included."""
    with tempfile.TemporaryDirectory() as d:
        cpp, asm = Path(d) / "k.cpp", Path(d) / "k.s"
        cpp.write_text(k.source(scaffold="bare", **consts))
        cmd = [hc.hipcc(), *[f for f in hc.flags(gfx950.GFX950, "probe",
                                                  k.build_flags(k.trace(**consts)))
                             if f not in ("-shared", "-fPIC",
                                          "-Rpass-analysis=kernel-resource-usage")],
               "--offload-device-only", "-S", str(cpp), "-o", str(asm)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return asm.read_text()


def _isa(k, **consts) -> str:
    """The kernel's body, entry to s_endpgm: `<name>_kernel(const globals)`,
    or a flat kernel's `hk_flat::<name>_kernel(...)`."""
    text = _asm(_for_950(k), **consts)
    at = re.search(r"^_Z\w*_kernel\w*:", text, re.M).start()
    return text[at:text.index("s_endpgm", at)]


@pytest.mark.hipcc
def test_the_weighted_strip_moves_each_run_once_in_16_bytes():
    """4096 bf16 columns over 2 warps of 4 chunks: four runs a lane of the
    row, four of the weight, four stored; the sum of squares one butterfly."""
    text = _isa(norm.STRIP_KERNELS["strip_rmsw_bf16_w2"], CHUNKS=4, RUN=8, EXACT=1)
    loads = re.findall(r"\bglobal_load_(\w+)", text)
    stores = re.findall(r"\bglobal_store_(\w+)", text)
    assert loads == ["dwordx4"] * 8, loads
    assert stores == ["dwordx4"] * 4, stores
    assert len(re.findall(r"\bv_add_f32_dpp\b", text)) == 4
    assert "v_permlane16_swap_b32" in text and "v_permlane32_swap_b32" in text
    assert "ds_bpermute" not in text and "ds_swizzle" not in text


@pytest.mark.hipcc
def test_rope_turns_the_halves_with_one_swap_a_dword():
    """A head is one 128-column strip row, two bf16 a lane: x and its other
    half meet by v_permlane32_swap, not through LDS."""
    text = _isa(fused.ROTARY_KERNELS["rotary_bf16_w8_pos"], H=32, D=128, RUN=2)
    assert "v_permlane32_swap_b32" in text
    assert "ds_bpermute" not in text and "ds_swizzle" not in text
    assert re.findall(r"\bglobal_store_(\w+)", text) == ["dword"] * 4


@pytest.mark.hipcc
def test_the_preloaded_twin_has_its_pointers_in_sgprs_at_entry():
    """A flat kernel's arguments are scalars, so LLVM can preload them: the
    three pointers and two of the row strides of the pointwise add (16 user
    SGPRs less the kernarg pointer), and none without the flag."""
    k = _for_950(elementwise.STRIP_KERNELS["strip_add_bf16_w4"])
    consts = dict(CW=512, RUN=8, EXACT=1)
    pre = re.compile(r"\.amdhsa_user_sgpr_kernarg_preload_length (\d+)")
    assert pre.search(_asm(k, **consts)).group(1) == "0"
    assert int(pre.search(_asm(k.preloaded(), **consts)).group(1)) >= 6
