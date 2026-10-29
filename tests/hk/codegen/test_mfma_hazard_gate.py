"""The MFMA hazard gate (hk.runtime.compile, `_mfma_found`), on real builds.

The kernel it exists for: gfx950's paged decode at two pages a step, without
the QK fence. LLVM before llvm/llvm-project#225989 reads the scores 2 wait
states after the last QK MFMA on the taken path of a branch, where 8 are
needed, and the decode computed some scores wrong (ROCm/llvm-project#5153).
- by default (HK_MFMA_GATE=fix) rebuilt with the padding the fixed compiler
  emits, and the padded code object is clean; the build log and
  `hk.integration --status` say which kernel and how many places;
- HK_MFMA_GATE=refuse refuses it, with where and how many wait states;
- HK_MFMA_GATE=report builds it and says so; a compiler with the fix skips
  the check (its probe, PROBE_MIR through this toolchain's llc);
- this toolchain's llc is told apart: every ROCm release so far lacks the fix.
Compile-only: hipcc and the ROCm LLVM tools. The padded decode's numbers are
tests/hk/gpu/test_mfma_hazard_gate.py.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.ops import attn_cdna, paged_cdna  # noqa: E402
from hk.runtime import cocheck, mfmahaz  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.runtime.resources import ResourceError  # noqa: E402


def _llc() -> str:
    return cocheck._tool("llc")


_HITS: dict = {}


def _unfixed():
    """Skip unless this compiler leaves the unfenced decode short somewhere; the
    places it does (ROCm 7.2.3: 7, after the QK MFMAs on the taken path)."""
    if mfmahaz.compiler_fixed(_llc()) is True:
        pytest.skip("this compiler pads every path (llvm/llvm-project#225989)")
    if "n" not in _HITS:
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        name, src = _decode_source(fence=False)
        old = {k: os.environ.get(k) for k in ("HK_CACHE_DIR", "HK_MFMA_GATE")}
        with tempfile.TemporaryDirectory() as d:
            os.environ.update(HK_CACHE_DIR=d, HK_MFMA_GATE="0")
            try:
                b = hc.build(src, "gfx950", name=name, kind="co")
                _HITS["n"] = len(cocheck.mfma_hazards(b.so_path, bundled=False))
            finally:
                for k, v in old.items():
                    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    if not _HITS["n"]:
        pytest.skip("this compiler's unfenced decode is not short anywhere")
    return _HITS["n"]


def _decode_source(fence: bool):
    k = paged_cdna.decode_kernel(f"mfmahaz_decode_{int(fence)}", head_dim=128, splits=1,
                                 mode="fused", ppstep=2, fence_qk=fence, arch="gfx950")
    tr = k.trace()
    return tr.name, scaffold.device(tr)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("HK_MFMA_GATE", raising=False)
    return tmp_path / "cache"


def test_this_toolchains_llc_answers_the_probe():
    verdict = mfmahaz.compiler_fixed(_llc())
    if verdict is None:
        pytest.skip("no llc beside this hipcc to probe")
    major = int(re.search(r"clang version (\d+)", hc._hipcc_version()).group(1))
    if major <= 24:
        # ROCm 7.1-7.2 (22), the ROCm 10.0 / 10.1 images (23, 24): all from before the fix
        assert verdict is False


def test_refuse_names_where_and_how_much(cache, monkeypatch):
    _unfixed()
    monkeypatch.setenv("HK_MFMA_GATE", "refuse")
    name, src = _decode_source(fence=False)
    with pytest.raises(ResourceError) as e:
        hc.build(src, "gfx950", name=name, kind="co")
    msg = str(e.value)
    assert "MFMA result(s) read or overwritten too few wait states" in msg
    assert re.search(r"` comes \d+ wait state\(s\) after `v_mfma_\w+ .*needs \d+: \d+ more", msg)
    assert not list(cache.glob("*/OK"))           # nothing usable left behind


def test_the_default_pads_it_and_the_padded_object_is_clean(cache, monkeypatch, capsys):
    n = _unfixed()
    name, src = _decode_source(fence=False)
    b = hc.build(src, "gfx950", name=name, kind="co")
    assert cocheck.mfma_hazards(b.so_path, bundled=False) == []
    assert f"HK_MFMA_GATE=fix padded {n} MFMA hazard(s)" in b.log_path.read_text()
    assert "padded" in capsys.readouterr().err
    (ck,) = cocheck.check(b.so_path)
    assert ck.ok, ck.line()
    # where it went: the marker, and the status line
    assert (b.so_path.parent / hc._MFMA_MARK).read_text().startswith(f"padded {n}\n")
    status = "\n".join(hc.mfma_gate_status())
    assert re.search(rf"1 cached build\(s\) padded:\n  \w+ _Z\w*mfmahaz_decode_0\w*: {n} place", status)
    # and a refuse-mode hit on the padded entry passes: it is marked checked
    monkeypatch.setenv("HK_MFMA_GATE", "refuse")
    assert hc.build(src, "gfx950", name=name, kind="co").cached


def test_an_unpadded_cache_entry_is_rebuilt_padded(cache, monkeypatch):
    _unfixed()
    name, src = _decode_source(fence=False)
    monkeypatch.setenv("HK_MFMA_GATE", "0")
    first = hc.build(src, "gfx950", name=name, kind="co")
    assert cocheck.mfma_hazards(first.so_path, bundled=False)
    monkeypatch.delenv("HK_MFMA_GATE")
    again = hc.build(src, "gfx950", name=name, kind="co")
    assert not again.cached and cocheck.mfma_hazards(again.so_path, bundled=False) == []


def test_report_builds_and_says_so(cache, monkeypatch, capsys):
    _unfixed()
    monkeypatch.setenv("HK_MFMA_GATE", "report")
    name, src = _decode_source(fence=False)
    b = hc.build(src, "gfx950", name=name, kind="co")
    assert "MFMA result(s) read or overwritten" in capsys.readouterr().err
    assert cocheck.mfma_hazards(b.so_path, bundled=False)
    # unmarked: a refusing build refuses it from the cache (the default rebuilds it padded)
    monkeypatch.setenv("HK_MFMA_GATE", "refuse")
    with pytest.raises(ResourceError, match="Cached build"):
        hc.build(src, "gfx950", name=name, kind="co")


def test_a_compiler_with_the_fix_is_not_checked(cache, monkeypatch):
    monkeypatch.setattr(mfmahaz, "compiler_fixed", lambda llc: True)
    monkeypatch.setattr(cocheck, "mfma_hazards", lambda *a, **k: pytest.fail("checked anyway"))
    name, src = _decode_source(fence=False)
    b = hc.build(src, "gfx950", name=name, kind="co")
    assert (b.so_path.parent / hc._MFMA_MARK).exists()


def test_the_fenced_decode_passes(cache):
    name, src = _decode_source(fence=True)
    hc.build(src, "gfx950", name=name, kind="co")


def test_off_builds_anything(cache, monkeypatch):
    monkeypatch.setenv("HK_MFMA_GATE", "0")
    name, src = _decode_source(fence=False)
    b = hc.build(src, "gfx950", name=name, kind="co")
    assert not (b.so_path.parent / hc._MFMA_MARK).exists()


def test_the_dense_attention_hits_under_rocm_7_2_3(cache, monkeypatch):
    """A kernel that ships: gfx950's dense attention, head_dim 128, not causal.
    ROCm 7.2.3's LLVM reads S for the softmax's max 3-4 wait states after the
    last QK MFMA on the taken path of a branch; a 32x32x16 MFMA needs 12. Padded
    by default (its numbers: tests/hk/gpu/test_mfma_hazard_gate.py)."""
    _unfixed()
    if "roc-7.2.3" not in hc._hipcc_version():
        pytest.skip("measured under ROCm 7.2.3's hipcc")
    k = attn_cdna.attn_kernel("mfmahaz_attn_d128", head_dim=128, arch="gfx950")
    tr = k.trace()
    src = scaffold.device(tr)
    monkeypatch.setenv("HK_MFMA_GATE", "refuse")
    with pytest.raises(ResourceError) as e:
        hc.build(src, "gfx950", name=tr.name, kind="co")
    assert re.search(r"`v_max_f32_e32 v\d+, v\d+, v\d+` comes \d wait state\(s\) after "
                     r"`v_mfma_f32_32x32x16_bf16 .*needs 12", str(e.value))
    monkeypatch.delenv("HK_MFMA_GATE")
    b = hc.build(src, "gfx950", name=tr.name, kind="co")
    assert "padded 2 MFMA hazard(s)" in b.log_path.read_text()
    assert cocheck.mfma_hazards(b.so_path, bundled=False) == []


def test_the_padding_rebuild_finds_the_device_assembly_of_a_module(tmp_path, monkeypatch):
    """A host+device .so, as kind="so" builds one. clang 24's new offload driver generates its
    device code at link time (clang-linker-wrapper), with no device assembly step in
    `hipcc -###` to pad; the rebuild plans it under --no-offload-new-driver instead. Before,
    it ran the pipeline, padded nothing, and the refusal that followed named a hazard the
    default mode should have fixed."""
    src = tmp_path / "k.cpp"
    src.write_text("#include <hip/hip_runtime.h>\n"
                   "__global__ void k(float* x) { x[threadIdx.x] *= 2.0f; }\n")
    out = tmp_path / "k.so"
    cmd = [hc.hipcc(), "--offload-arch=gfx950", "-O3", "-w", "-shared", "-fPIC", str(src),
           "-o", str(out)]
    seen = []
    fix_asm = mfmahaz.fix_asm
    monkeypatch.setattr(mfmahaz, "fix_asm", lambda text, arch: seen.append(arch) or fix_asm(text, arch))
    assert hc._pad_mfma_hazards(cmd, tmp_path, "gfx950") == []
    assert seen == ["gfx950"], "the device assembly was never handed to the padder"
    assert out.exists() and cocheck.mfma_hazards(out, bundled=True) == []
