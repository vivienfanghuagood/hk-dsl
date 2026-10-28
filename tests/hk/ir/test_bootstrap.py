"""A fresh clone: the examples run against the installed hk-dsl when the checkout's
bindings are not built, and `import hk` from an unbuilt checkout says what to do.
No GPU, no compiler: the packages are copies without hk._mlir and a stand-in."""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _clone_without_bindings(tmp: Path) -> Path:
    """examples/_common.py and python/hk as a fresh clone has them: no hk._mlir."""
    clone = tmp / "clone"
    (clone / "examples").mkdir(parents=True)
    shutil.copy(REPO / "examples" / "_common.py", clone / "examples")
    shutil.copy(REPO / "pyproject.toml", clone)
    shutil.copytree(REPO / "python" / "hk", clone / "python" / "hk",
                    ignore=shutil.ignore_patterns("_mlir", "__pycache__", "include"))
    return clone


def _run(code: str, cwd: Path, pythonpath: str = "") -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath, "PYTHONNOUSERSITE": "1"}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


def test_unbuilt_checkout_says_how_to_get_bindings(tmp_path):
    clone = _clone_without_bindings(tmp_path)
    r = _run("import hk", clone, str(clone / "python"))
    assert r.returncode != 0
    assert "pip install" in r.stderr and "hk-dsl" in r.stderr
    assert "scripts/build_hk_mlir.sh" in r.stderr
    assert "ModuleNotFoundError: No module named 'hk._mlir'" not in r.stderr.splitlines()[-1]


def test_examples_use_the_installed_release(tmp_path):
    clone = _clone_without_bindings(tmp_path)
    site = tmp_path / "site" / "hk"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text('__version__ = "installed"\n')
    r = _run(textwrap.dedent("""
        import _common, hk
        print(hk.__version__)
    """), clone / "examples", str(tmp_path / "site"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "installed"


def test_examples_without_any_hk_say_what_to_install(tmp_path):
    clone = _clone_without_bindings(tmp_path)
    r = _run("import _common", clone / "examples")
    assert r.returncode != 0
    assert "pip install hk-dsl" in r.stderr
    assert "scripts/build_mlir.sh && scripts/build_hk_mlir.sh" in r.stderr
# merge-train test D: conflicts with test A (throwaway PR, never merged)
