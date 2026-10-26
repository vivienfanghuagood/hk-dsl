"""hk.integration.sglang_diffusion's install: the .pth boot reaches every
process of the environment, applies the routes when SGLang-Diffusion's SDPA
backend is imported (hk is on by default), and does nothing with HK_DISABLE=1
or HK_SGLD=0. No GPU, no SGLang: a stand-in package of the same module path --
whose engine contract is broken, so the routes are checked and refused."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hk.integration import sglang_diffusion as sgld

REPO = Path(__file__).resolve().parents[3]


def _env(tmp_path, env_set):
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
    for k in ("HK_SGLD", "HK_DISABLE", "HK_SGLD_ATTN", "HK_SGLD_LINEAR", "HK_SGLD_CONVROT"):
        env.pop(k, None)
    env["HK_ARCH"] = "gfx1100"
    env.update(env_set)
    code = (f"import site; site.addsitedir({str(site)!r}); import {sgld.TARGET} as m; "
            f"print('loaded', m.LOADED)")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.mark.parametrize("env_set", [{}, {"HK_SGLD": "1"}, {"HK_SGLD": "attn"}])
def test_the_boot_applies_on_import_by_default_and_checks_the_contract(tmp_path, env_set):
    r = _env(tmp_path, env_set)
    assert r.returncode == 0, r.stderr
    assert "loaded True" in r.stdout
    # The stand-in is not SGLang-Diffusion: its contract is broken, so nothing is patched and
    # one warning line says so.
    assert "[hk] sglang-diffusion: engine contract failed: nothing patched, stock" in r.stdout
    warn = [x for x in r.stderr.splitlines() if x.startswith("[hk] WARNING")]
    assert len(warn) == 1 and "sglang_diffusion: engine contract FAILED" in warn[0]
    assert "sgld_attn: " in warn[0]


@pytest.mark.parametrize("env_set", [{"HK_DISABLE": "1"}, {"HK_SGLD": "0"}, {"HK_SGLD": "off"}])
def test_the_boot_does_nothing_when_turned_off(tmp_path, env_set):
    r = _env(tmp_path, env_set)
    assert r.returncode == 0, r.stderr
    assert "[hk]" not in r.stdout + r.stderr and "loaded True" in r.stdout


def test_uninstall_removes_both_files(tmp_path):
    written = sgld.install(tmp_path)
    assert all(p.exists() for p in written)
    assert sorted(sgld.uninstall(tmp_path)) == sorted(written)
    assert not any(p.exists() for p in written)
