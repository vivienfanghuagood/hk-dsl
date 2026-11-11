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
    for k in ("HK_SGLD", "HK_DISABLE", "HK_SGLD_ATTN", "HK_SGLD_LINEAR", "HK_SGLD_CONVROT",
              "HK_SGLD_VAECONV", "HK_SGLD_ROPE", "HK_SGLD_QKNORM", "HK_SGLD_DITNORM",
              "HK_SGLD_H3VAE", "HK_SGLD_PATCHEMBED"):
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
    # The stand-in is not SGLang-Diffusion: every route's contract is broken, so nothing is
    # patched and each route on says so in one warning line of its own.
    assert "[hk] sglang-diffusion: no route applied (" in r.stdout and "): stock" in r.stdout
    warn = [x for x in r.stderr.splitlines() if x.startswith("[hk] WARNING")]
    want = {"attn": ["sgld_attn"], "1": ["sgld_attn", "sgld_linear"]}.get(
        env_set.get("HK_SGLD"), ["sgld_attn", "sgld_linear", "sgld_vaeconv", "sgld_rope", "sgld_qknorm", "sgld_ditnorm"])
    assert len(warn) == len(want)
    for route, line in zip(want, warn):
        assert f"sglang_diffusion: route {route} OFF -- engine contract FAILED" in line


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


@pytest.mark.parametrize("src,ok", [
    # SGLang up to 0.5.22: the fallback goes through apply_qk_norm (norm(x) off CUDA)
    ("    out = apply_rope(apply_qk_norm(x, norm), rope)\n    return out\n", True),
    # SGLang 0.5.23: norm(x) directly, after two CUDA-only fusion gates
    ("    if _is_cuda:\n        return fused\n    out = apply_rope(norm(x), rope)\n", True),
    ("    return fused_only(x, norm, rope)\n", False),
])
def test_the_qknorm_contract_reads_both_forms_of_the_fallback(src, ok):
    from hk.integration import contract

    assert contract.norm_then_rope(src) is ok


H3VAE = {"conv3d": True, "forward": ["self", "x"], "sets": True, "one_frame_causal_zeros": True,
         "spatial_pad": True, "conv": True}


@pytest.mark.parametrize("change,ok", [
    ({}, True),
    ({"forward": ["self", "x", "pre_padded"]}, True),        # SGLang's newer BaseConv3d
    ({"forward": ["self", "x", "cache"]}, False),
    ({"one_frame_causal_zeros": False}, False),             # one frame no longer behind zeros
    ({"spatial_pad": False}, False),
    ({"conv": False}, False),
])
def test_the_h3vae_contract(change, ok):
    from hk.integration import contract  # noqa: PLC0415

    assert (contract.judge("sgld_h3vae", dict(H3VAE, **change)) == []) is ok


def test_the_patchembed_contract_wants_c_t_p_p_patches():
    from hk.integration import contract  # noqa: PLC0415

    f = {"forward": ["self", "hidden_states"], "sets": True, "kernel_is_stride": True,
         "patch_order": True}
    assert contract.judge("sgld_patchembed", f) == []
    assert contract.judge("sgld_patchembed", dict(f, patch_order=False))
    assert contract.judge("sgld_patchembed", dict(f, kernel_is_stride=False))


def test_an_absent_engine_module_is_nothing_to_route_but_a_broken_one_is_an_error(tmp_path,
                                                                                  monkeypatch):
    from hk.integration import contract  # noqa: PLC0415

    pkg = tmp_path / "hkfake_engine"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("import hkfake_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert contract._module_or_none("hkfake_engine.minimax_h3") is None
    assert contract._module_or_none("hkfake_absent_engine.x.y") is None
    with pytest.raises(ModuleNotFoundError):
        contract._module_or_none("hkfake_engine.broken")


def test_flat_source_ignores_wrapping_and_trailing_commas():
    from hk.integration import contract  # noqa: PLC0415

    assert contract._flat("F.pad(\n    x,\n    (0, 0, 0, 0, self.kernel_size[0] - 1, 0,),\n)") \
        == "F.pad(x,(0,0,0,0,self.kernel_size[0]-1,0))"
