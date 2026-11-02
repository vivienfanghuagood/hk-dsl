"""The VOPD gate (hk.runtime.compile, `_vopd_found`), on real builds.

gfx11.0 / gfx11.5 have a VOPD interlock hazard: when X's and Y's SRC0, or
their VSRC1, are distinct VGPRs of the same parity, the destination-buffer
forwarding drops the interlock and an operand written just before is read
stale. LLVM before llvm/llvm-project#220348 (every ROCm release so far)
requires only different banks there. gfx1100's bounded strip RMSNorm with
the fused add (strip_addrmsw, 16 warps, EXACT=0), built by AMD clang 23
(TheRock 7.14), had `v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53,
16, v0` (VSRC1 v22 / v0) and was wrong in 25-65% of launches.
- by default (HK_VOPD_GATE=fix) such a kernel is rebuilt with
  -mllvm -amdgpu-enable-vopd=0 and has no hazardous VOPD left; the build
  log, VOPDHAZ and `hk.integration --status` say so;
- HK_VOPD_GATE=refuse refuses it, naming the VOPD; report builds it;
- a cached build from before the gate is rebuilt on its first hit.
Compile-only: hipcc and the ROCm LLVM tools.
"""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.ops import norm  # noqa: E402
from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.runtime.resources import ResourceError  # noqa: E402

#: The failing plan: 1024 x 5120 bf16 rows on 16 warps, bounded (EXACT=0).
CONSTS = dict(CHUNKS=2, RUN=8, EXACT=0, RESERVE=0)


def test_the_hazard_is_same_parity_sources_only():
    def hz(t):
        return cocheck.vopd_parity_hazard(cocheck.Inst(0, t))

    # the strip RMSNorm's VOPD: VSRC1 v22 / v0
    assert "VSRC1" in hz("v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0")
    # the same with the parities made different: no hazard, whatever OPY reads
    assert hz("v_dual_add_f32 v53, v14, v22 :: v_dual_lshlrev_b32 v0, 16, v53") is None
    # no dependency between X and Y at all, same parity: the hazard
    assert hz("v_dual_add_f32 v14, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0")
    assert "SRC0" in hz("v_dual_mov_b32 v21, v172 :: v_dual_add_nc_u32 v166, v6, v145")
    # the same VGPR on both sides is allowed; constants, literals and SGPRs never count
    assert hz("v_dual_add_f32 v1, v4, v6 :: v_dual_add_f32 v2, v4, v7") is None
    assert hz("v_dual_add_f32 v1, 1.0, v6 :: v_dual_add_nc_u32 v2, s4, v9") is None
    assert hz("v_dual_cndmask_b32 v1, 0, v10 :: v_dual_lshlrev_b32 v2, 1, v3") is None
    # fmamk carries its literal between SRC0 and VSRC1
    assert hz("v_dual_fmamk_f32 v1, v2, 0x3f800000, v4 :: v_dual_add_f32 v3, v5, v8")
    assert hz("v_dual_fmamk_f32 v1, v2, 0x3f800000, v4 :: v_dual_add_f32 v3, v5, v9") is None
    assert hz("v_add_f32_e32 v0, v1, v3") is None


def _source():
    k = norm.STRIP_KERNELS["strip_addrmsw_bf16_w16"]
    tr = k.trace(**CONSTS)
    if not tr.arch.startswith(cocheck.VOPD_HAZARD_ARCHS):
        pytest.skip(f"{tr.arch} has no VOPD interlock hazard")
    return tr.name, tr.arch, scaffold.device(tr)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("HK_VOPD_GATE", raising=False)
    return tmp_path / "cache"


def _pairs_without_gate(monkeypatch):
    name, arch, src = _source()
    monkeypatch.setenv("HK_VOPD_GATE", "0")
    b = hc.build(src, arch, name=name, kind="co")
    monkeypatch.delenv("HK_VOPD_GATE")
    found = cocheck.vopd_hazards(b.so_path, bundled=False)
    if not found:
        pytest.skip("this compiler builds the strip kernel with no hazardous VOPD")
    return name, arch, src, b, found


def test_default_rebuilds_without_vopd(cache, monkeypatch, capfd):
    name, arch, src, _, found = _pairs_without_gate(monkeypatch)
    shutil.rmtree(cache)  # the gate-off build is cached under the same key: start over
    b = hc.build(src, arch, name=name, kind="co")
    assert cocheck.vopd_hazards(b.so_path, bundled=False) == []
    mark = (b.so_path.parent / hc._VOPD_MARK).read_text()
    assert mark.startswith(f"rebuilt {len(found)}\n")
    assert "-amdgpu-enable-vopd=0" in (b.so_path.parent / "command").read_text()
    assert "HK_VOPD_GATE=fix rebuilt" in (b.so_path.parent / "build.log").read_text()
    assert "HK_VOPD_GATE=fix rebuilt" in capfd.readouterr().err
    status = "\n".join(hc.vopd_gate_status())
    assert "1 cached build(s) rebuilt without VOPD" in status and name in status


def test_refuse_names_the_pair(cache, monkeypatch):
    name, arch, src, _, found = _pairs_without_gate(monkeypatch)
    shutil.rmtree(cache)
    monkeypatch.setenv("HK_VOPD_GATE", "refuse")
    with pytest.raises(ResourceError, match="interlock hazard") as e:
        hc.build(src, arch, name=name, kind="co")
    assert found[0].split(": ", 1)[1].split("   <-")[0] in str(e.value)


def test_report_builds_it(cache, monkeypatch, capfd):
    name, arch, src, _, found = _pairs_without_gate(monkeypatch)
    shutil.rmtree(cache)
    monkeypatch.setenv("HK_VOPD_GATE", "report")
    b = hc.build(src, arch, name=name, kind="co")
    assert len(cocheck.vopd_hazards(b.so_path, bundled=False)) == len(found)
    assert "interlock hazard" in capfd.readouterr().err
    assert not (b.so_path.parent / hc._VOPD_MARK).exists()


def test_cached_build_from_before_the_gate_is_rebuilt(cache, monkeypatch):
    name, arch, src, old, _ = _pairs_without_gate(monkeypatch)
    # the gate-off build is in the cache with no VOPDHAZ, as one from before the gate
    assert not (old.so_path.parent / hc._VOPD_MARK).exists()
    b = hc.build(src, arch, name=name, kind="co")
    assert not b.cached
    assert cocheck.vopd_hazards(b.so_path, bundled=False) == []
    assert (b.so_path.parent / hc._VOPD_MARK).read_text().startswith("rebuilt ")
