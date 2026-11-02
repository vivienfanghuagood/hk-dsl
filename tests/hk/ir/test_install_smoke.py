"""The install matrix (.github/workflows/install.yml) runs README.md's quick start as written, read by
scripts/ci/install_smoke.py. That job runs nightly, on GPUs; this keeps a README edit that would break
the extraction from waiting for it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _smoke():
    spec = importlib.util.spec_from_file_location("install_smoke", REPO / "scripts" / "ci" / "install_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_readme_quick_start_is_what_the_install_matrix_runs():
    code, cmd = _smoke().quick_start(str(REPO / "README.md"))
    assert "import hk" in code and "def rms_norm" in code
    compile(code, "my_kernels.py", "exec")
    assert cmd.startswith("python3 -m hk.check my_kernels.py:") and "\n" not in cmd
    assert "--ref" in cmd and "--inputs" in cmd
