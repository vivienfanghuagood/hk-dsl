"""scripts/check_wheel_glibc.py: a wheel whose binaries need a newer glibc than
its manylinux tag promises fails; one tagged at or above what they need passes.
Wheels are made here from a system binary, so no build is needed."""

from __future__ import annotations

import importlib.util
import shutil
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "check_wheel_glibc", REPO / "scripts" / "check_wheel_glibc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wheel_tags():
    tags = _mod().wheel_tags
    assert tags("hk_dsl-0.7.0-cp312-cp312-manylinux_2_35_x86_64.whl") == [(2, 35)]
    assert tags("x-1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl") == [(2, 17), (2, 17)]
    assert tags("x-1-cp310-cp310-manylinux2010_x86_64.whl") == [(2, 12)]
    assert tags("hk_dsl-0.7.0-cp310-cp310-linux_x86_64.whl") == []
    with pytest.raises(ValueError):
        tags("x-1-cp310-cp310-manylinux2099_x86_64.whl")


@pytest.mark.skipif(shutil.which("readelf") is None or shutil.which("ls") is None,
                    reason="needs readelf (binutils) and an ELF to test with")
def test_glibc_against_tag(tmp_path):
    mod = _mod()
    elf = Path(shutil.which("ls")).resolve()
    if elf.read_bytes()[:4] != b"\x7fELF":
        pytest.skip(f"{elf} is not ELF")
    need = max(v for _, n in mod.version_needs(elf) if (v := mod._glibc(n)))

    def wheel(tag):
        path = tmp_path / f"t-1-cp312-cp312-{tag}_x86_64.whl"
        with zipfile.ZipFile(path, "w") as z:
            z.write(elf, "t/_lib.so")
            z.writestr("t/__init__.py", "")
        return str(path)

    ok = wheel(f"manylinux_{need[0]}_{need[1]}")
    newer = wheel(f"manylinux_{need[0]}_{need[1] + 1}")
    old = wheel(f"manylinux_{need[0]}_{need[1] - 1}")
    plain = wheel("linux")
    assert mod.check(ok) == []
    assert mod.check(newer) == []
    assert mod.check(plain) == []
    problems = mod.check(old)
    assert problems and f"GLIBC_{need[0]}.{need[1]}" in problems[0], problems
    assert mod.main([ok, newer]) == 0
    assert mod.main([ok, old]) == 1
