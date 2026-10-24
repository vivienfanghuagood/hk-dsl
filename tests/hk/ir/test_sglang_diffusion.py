"""hk.integration.sglang_diffusion's install: the .pth boot reaches every
process of the environment, does nothing without HK_SGLD, and applies the
patch when SGLang-Diffusion's SDPA backend is imported. No GPU, no SGLang: a
stand-in package of the same module path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hk.integration import sglang_diffusion as sgld

REPO = Path(__file__).resolve().parents[3]


def _env(tmp_path, on):
    pkg = tmp_path / "pkgs" / Path(*sgld.TARGET.split(".")[:-1])
    pkg.mkdir(parents=True)
    for d in [tmp_path / "pkgs" / Path(*sgld.TARGET.split(".")[:i])
              for i in range(1, len(sgld.TARGET.split(".")))]:
        (d / "__init__.py").write_text("")
    (pkg / "sdpa.py").write_text("LOADED = True\n")
    site = tmp_path / "site"
    site.mkdir()
    sgld.install(site)
    env = dict(os.environ, PYTHONPATH=f"{tmp_path / 'pkgs'}{os.pathsep}{REPO / 'python'}")
    env.pop("HK_SGLD", None)
    if on:
        env["HK_SGLD"] = "1"
    code = (f"import site; site.addsitedir({str(site)!r}); import {sgld.TARGET} as m; "
            f"print('loaded', m.LOADED)")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)


def test_the_boot_patches_on_import_with_hk_sgld(tmp_path):
    r = _env(tmp_path, True)
    assert r.returncode == 0, r.stderr
    assert "[hk] sglang-diffusion:" in r.stdout and "loaded True" in r.stdout


def test_the_boot_does_nothing_without_hk_sgld(tmp_path):
    r = _env(tmp_path, False)
    assert r.returncode == 0, r.stderr
    assert "[hk]" not in r.stdout and "loaded True" in r.stdout


def test_uninstall_removes_both_files(tmp_path):
    written = sgld.install(tmp_path)
    assert all(p.exists() for p in written)
    assert sorted(sgld.uninstall(tmp_path)) == sorted(written)
    assert not any(p.exists() for p in written)
