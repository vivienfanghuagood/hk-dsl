"""The MFMA hazard check (hk.runtime.mfmahaz): the wait states an MFMA result
needs before it is read or overwritten, over every path of the control flow.

Pure Python on assembly text: the windows against LLVM's tables, the walk on
straight-line code, across a branch, around a loop, and the case LLVM before
llvm/llvm-project#225989 got wrong (the reader's block reached the long way
first); the padding `fix_asm` adds; and the compiler probe, with a stand-in
llc. The compiled cases (the decode kernel, the gate in hk's build, the real
llc) are tests/hk/codegen/test_mfma_hazard_gate.py.
"""

from __future__ import annotations

import os
import stat

import pytest

from hk.runtime import mfmahaz as M

BF16_16 = "v_mfma_f32_16x16x32_bf16"
BF16_32 = "v_mfma_f32_32x32x16_bf16"


def _asm(body: str, name: str = "k") -> str:
    return f"{name}:\n" + "\n".join(
        line if line.endswith(":") else "\t" + line.strip()
        for line in body.strip().splitlines()) + "\n"


def _check(body: str, arch: str = "gfx950"):
    prog = [i for _, i in M.asm_functions(_asm(body))["k"]]
    return prog, M.deficits(prog, arch)


# ---------------------------------------------------------------- windows


@pytest.mark.parametrize("mnem,arch,passes,xdl", [
    (BF16_16, "gfx950", 4, True),
    (BF16_32, "gfx950", 8, True),
    ("v_mfma_scale_f32_16x16x128_f8f6f4", "gfx950", 4, True),
    ("v_mfma_scale_f32_32x32x64_f8f6f4", "gfx950", 8, True),
    ("v_mfma_f32_4x4x4_16b_f16", "gfx950", 2, True),
    ("v_mfma_f32_16x16x16_f16", "gfx942", 4, True),
    ("v_mfma_f32_16x16x8_xf32", "gfx942", 4, True),
    ("v_mfma_f32_32x32x2_f32", "gfx942", 16, False),
    ("v_mfma_f32_16x16x4_f32", "gfx950", 8, False),
    ("v_mfma_f64_16x16x4_f64", "gfx950", 16, False),
    ("v_mfma_f64_16x16x4_f64", "gfx942", 8, False),
])
def test_pass_counts_are_the_scheduling_models(mnem, arch, passes, xdl):
    """llvm-mca's throughput column on gfx942 / gfx950 (the pass count LLVM's
    recognizer reads), and XDL as VOP3PInstructions.td marks it: every MFMA
    but the FP32-input and f64 ones."""
    info = M.mfma_info(mnem, arch)
    assert (info.passes, info.xdl) == (passes, xdl)


def test_windows_are_llvms():
    """GCNHazardRecognizer's tables, at the shapes hk uses."""
    m16, m32 = M.mfma_info(BF16_16, "gfx950"), M.mfma_info(BF16_32, "gfx950")
    assert M.read_window(m16, "gfx950", memory=False) == 8      # 4 + 3 + 1
    assert M.read_window(m32, "gfx950", memory=True) == 12      # 8 + 3 + 1
    assert M.read_window(M.mfma_info(BF16_16, "gfx942"), "gfx942", memory=False) == 7
    assert M.srcab_window(m16, "gfx950") == 8
    assert M.srcc_window(m16, m16, "gfx950", full=True) == 0     # the same accumulator
    assert M.srcc_window(m16, m16, "gfx950", full=False) == 6    # an overlapping one
    assert M.waw_window(m32, "gfx950") == 12
    assert M.war_window(m16) == 3 and M.war_window(m32) == 7
    four = M.mfma_info("v_mfma_f32_4x4x4_16b_f16", "gfx950")
    assert M.read_window(four, "gfx950", memory=False) == 5     # 2-pass: no gfx950 extra
    f32 = M.mfma_info("v_mfma_f32_32x32x2_f32", "gfx942")
    assert M.read_window(f32, "gfx942", memory=False) == 18     # not XDL: passes + 2


def test_operands_keep_modifiers_whole():
    assert M.operands("a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 op_sel_hi:[0,0,0] cbsz:4") == [
        "a[0:3]", "v[0:7]", "v[8:15]", "a[0:3]", "v16", "v17"]


# ---------------------------------------------------------------- the walk


@pytest.mark.parametrize("nop,missing", [(7, 0), (6, 1), (0, 7)])
def test_straight_line(nop, missing):
    _, found = _check(f"""
        {BF16_16} a[0:3], v[0:3], v[4:7], a[0:3]
        s_nop {nop}
        v_accvgpr_read_b32 v8, a3
        s_endpgm""")
    assert [d.missing for d in found] == ([missing] if missing else [])


def test_a_chain_on_one_accumulator_needs_no_wait():
    _, found = _check(f"""
        {BF16_16} v[8:11], v[0:3], v[4:7], v[8:11]
        {BF16_16} v[8:11], v[12:15], v[4:7], v[8:11]
        s_endpgm""")
    assert not found


def test_an_overlapping_accumulator_does():
    """srcC overlapping a result it is not exactly: LLVM 22 left this at one
    wait state in a gfx950 decode; the window is 6."""
    _, found = _check(f"""
        {BF16_16} v[134:137], v[138:141], v[2:5], 0
        {BF16_16} v[136:139], v[154:157], v[6:9], v[134:137]
        {BF16_16} v[136:139], v[142:145], v[10:13], v[136:139]
        s_endpgm""")
    assert [(d.consumer, d.producer, d.need, d.have) for d in found] == [(2, 0, 6, 1)]


#: The recognizer's case as LLVM 22 compiles it (PROBE_MIR through llc
#: -start-before=post-RA-hazard-rec): the read's block has the MFMA one branch
#: away and a longer path from it, and got padding for the long one.
TWO_PATHS = f"""
        s_branch .LBB0_2
.LBB0_1:
        v_mov_b32_e32 v10, 0
        v_mov_b32_e32 v11, 0
        v_mov_b32_e32 v12, 0
        v_mov_b32_e32 v13, 0
        s_branch .LBB0_3
.LBB0_2:
        s_cmp_eq_u32 s0, 0
        {BF16_16} a[0:3], v[0:3], v[4:7], a[0:3]
        s_cbranch_scc1 .LBB0_3
        s_branch .LBB0_1
.LBB0_3:
        s_nop {{nop}}
        v_accvgpr_read_b32 v8, a3
        s_endpgm"""


def test_the_shortest_path_into_a_block_counts():
    """LLVM 22's s_nop 0 is 2 wait states on the taken branch (the branch and
    the nop); the window is 8. The fixed recognizer's s_nop 6 is enough."""
    prog, found = _check(TWO_PATHS.format(nop=0))
    assert [(d.need, d.have, d.missing) for d in found] == [(8, 2, 6)]
    assert "v_accvgpr_read_b32 v8, a3" in M.report(prog, found)[0]
    assert not _check(TWO_PATHS.format(nop=6))[1]


def test_a_loop_carries_the_result_to_its_top():
    _, found = _check(f"""
.LBB0_1:
        v_accvgpr_read_b32 v8, a3
        s_nop 3
        {BF16_16} a[0:3], v[0:3], v[4:7], a[0:3]
        s_cbranch_scc1 .LBB0_1
        s_endpgm""")
    # around the back edge: the branch, then the read
    assert [(d.need, d.have) for d in found] == [(8, 1)]


def test_a_vgpr_result_read_by_a_store_and_a_load_overwriting_it():
    """A store reads its data at issue: the window applies. A load writes its
    destination when the data returns, after any window."""
    _, found = _check(f"""
        {BF16_16} v[8:11], v[0:3], v[4:7], v[8:11]
        global_store_dwordx4 v[20:21], v[8:11], off
        ds_read_b128 v[8:11], v12
        s_endpgm""")
    assert [(d.consumer, d.what) for d in found] == [(1, "read of its result")]


def test_no_mfmas_no_findings():
    assert not _check("v_add_f32_e32 v0, v1, v2\ns_endpgm")[1]
    assert not _check(f"{BF16_16} a[0:3], v[0:3], v[4:7], a[0:3]\nv_accvgpr_read_b32 v8, a3",
                      arch="gfx1100")[1]


# ---------------------------------------------------------------- padding


def test_fix_asm_pads_the_short_path_and_nothing_else():
    text = _asm(TWO_PATHS.format(nop=0))
    fixed, said = M.fix_asm(text, "gfx950")
    assert len(said) == 1 and "6 more" in said[0]
    prog = [i for _, i in M.asm_functions(fixed)["k"]]
    assert not M.deficits(prog, "gfx950")
    added = [i.text for i in prog if i.text.startswith("s_nop")]
    assert added == ["s_nop 0", "s_nop 5"]        # LLVM's own, then 6 wait states
    assert fixed.count("\n") == text.count("\n") + 1


def test_fix_asm_splits_long_padding():
    text = _asm(f"""
        {BF16_32} v[0:15], v[16:19], v[20:23], v[0:15]
        v_max_f32_e32 v30, v0, v1
        s_endpgm""")
    fixed, _ = M.fix_asm(text, "gfx950")
    assert "s_nop 7" in fixed and "s_nop 3" in fixed          # 12 = 8 + 4
    assert not M.deficits([i for _, i in M.asm_functions(fixed)["k"]], "gfx950")


# ---------------------------------------------------------------- the compiler


def _fake_llc(tmp_path, asm: str):
    out = tmp_path / "probe-out.s"
    out.write_text(asm)
    llc = tmp_path / "llc"
    llc.write_text(f'#!/bin/sh\nfor a; do case "$prev" in -o) cp {out} "$a";; esac; prev=$a; done\n')
    llc.chmod(llc.stat().st_mode | stat.S_IEXEC)
    return str(llc)


@pytest.mark.parametrize("nop,fixed", [(0, False), (6, True)])
def test_the_probe_reads_the_compilers_padding(tmp_path, nop, fixed):
    """The gate is skipped when the compiler pads the probe for its short
    path, and runs when it does not, or when there is no llc to ask."""
    body = TWO_PATHS.format(nop=nop)
    llc = _fake_llc(tmp_path, _asm(body, name="mfma_read_two_paths"))
    M._FIXED.pop(llc, None)
    assert M.compiler_fixed(llc) is fixed
    assert M.compiler_fixed(os.path.join(tmp_path, "no-llc")) is None
