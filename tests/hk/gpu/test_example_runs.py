"""Every example runs to completion on the GPU, its check against torch
included, and validate.py passes on the quick shapes. An example pinned to
some GPUs (its ARCHES) is skipped elsewhere, with its reason."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
import _catalog  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("script", _catalog.SCRIPTS, ids=[" ".join(s) for s in _catalog.SCRIPTS])
def test_it_runs_and_its_check_passes(script):
    arches, why = _catalog.arches(script[0])
    if arches is not None:
        import torch  # noqa: PLC0415

        from hk.runtime.arch import device_arch  # noqa: PLC0415

        here = device_arch(torch.cuda.current_device())
        if here not in arches:
            # a skip, not a pass: the script would only print why it does not run here
            pytest.skip(f"{script[0]} runs on {' / '.join(arches)} only ({here} here): {why}")
    r = subprocess.run([sys.executable, str(EXAMPLES / script[0]), *script[1:], "--no-bench"],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"{' '.join(script)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
