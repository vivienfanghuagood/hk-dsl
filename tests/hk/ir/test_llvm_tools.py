"""hk finds ROCm's LLVM tools where hipcc's clang is, not on PATH.

TheRock's pip layout (rocm/pytorch 10.1) puts a hipcc launcher in /usr/local/bin and
the toolchain under site-packages/_rocm_sdk_devel/lib/llvm/bin; no /opt/rocm and no
llvm-objdump on PATH. HK_LAUNCH=codeobj then died on FileNotFoundError: 'llvm-objdump'.
These tests build that layout out of empty scripts.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hk.runtime import compile as C


def _exe(p: Path, body: str = "exit 0") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"#!/bin/sh\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


@pytest.fixture
def therock(tmp_path, monkeypatch):
    """bin/hipcc (a launcher, as TheRock installs it) and sdk/_rocm_sdk_devel with
    include/hip and lib/llvm/bin/{clang,llvm-objdump,llvm-readelf}."""
    sdk = tmp_path / "site" / "_rocm_sdk_devel"
    llvm = sdk / "lib" / "llvm" / "bin"
    (sdk / "include" / "hip").mkdir(parents=True)
    for t in ("clang", "llvm-objdump", "llvm-readelf"):
        _exe(llvm / t)
    hipcc = _exe(tmp_path / "usrlocal" / "bin" / "hipcc",
                 "echo 'HIP version: 7.16.0'\n"
                 "echo 'AMD clang version 24.0.0git (https://example.invalid/llvm-project.git x)'\n"
                 "echo 'Target: x86_64-unknown-linux-gnu'\n"
                 f"echo 'InstalledDir: {llvm}'")
    monkeypatch.setenv("HK_HIPCC", str(hipcc))
    for v in ("ROCM_PATH", "HIP_CLANG_PATH"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(C, "_toolchain_cache", {})
    return sdk


def test_tools_are_found_beside_hipcc_s_clang(therock):
    llvm = therock / "lib" / "llvm" / "bin"
    assert C.clang_dir() == llvm
    assert C.llvm_tool("llvm-objdump") == str(llvm / "llvm-objdump")
    assert C.compiler_version().startswith("AMD clang version 24.0.0git")


def test_rocm_path_is_the_sdk_not_opt_rocm(therock):
    assert C.rocm_path() == therock


def test_a_missing_tool_names_where_it_looked(therock):
    assert C.llvm_tool("llc", required=False) is None
    with pytest.raises(FileNotFoundError) as e:
        C.llvm_tool("llc")
    assert str(therock / "lib" / "llvm" / "bin") in str(e.value)
    assert "ROCM_PATH" in str(e.value)


def test_cocheck_and_the_profiler_use_it(therock):
    from hk.profile import isa
    from hk.runtime import cocheck

    llvm = therock / "lib" / "llvm" / "bin"
    assert cocheck._tool("llvm-readelf") == str(llvm / "llvm-readelf")
    assert cocheck._tool("llc") == "llc"  # probed and reported missing by its callers
    assert isa.objdump() == str(llvm / "llvm-objdump")


def test_rocm_path_and_hip_clang_path_win(therock, tmp_path, monkeypatch):
    other = tmp_path / "rocm"
    _exe(other / "llvm" / "bin" / "llvm-objdump")
    monkeypatch.setenv("ROCM_PATH", str(other))
    assert C.rocm_path() == other
    # beside hipcc's clang still comes first: it is the toolchain that built the code
    assert C.llvm_tool("llvm-objdump") == str(therock / "lib" / "llvm" / "bin" / "llvm-objdump")
    monkeypatch.setenv("HIP_CLANG_PATH", str(other / "llvm" / "bin"))
    assert C.llvm_tool("llvm-objdump") == str(other / "llvm" / "bin" / "llvm-objdump")
    assert os.path.isfile(C.llvm_tool("llvm-objdump"))
