"""The schedules hk ships for an RDNA arch (python/hk/tuned/routes_gemm_<arch>.json),
through hk.ops.gemm.matmul, against fp32 at both ends of the M bucket each answers.

A shipped record was tuned at one M and answers every M of its power-of-two
bucket, down to half of it plus one row; a 32-row tile at the bottom of a
bucket meets a last block that is mostly past M, which the tuned M never did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hk.ops import gemm  # noqa: E402
from hk.target import current_arch  # noqa: E402
from hk.tune import records  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a Radeon")

_TABLE = Path(records.__file__).resolve().parents[1] / "tuned" / f"routes_gemm_{current_arch()}.json"
_CASES = []
if current_arch() != "gfx950" and _TABLE.exists():
    for nk, by_m in json.loads(_TABLE.read_text())["routes"].items():
        n, k = map(int, nk.split("x"))
        for ms in by_m:
            _CASES += [(int(ms) // 2 + 1, n, k), (int(ms), n, k)]


@pytest.fixture(autouse=True)
def _no_machine_records(tmp_path, monkeypatch):
    # The shipped record, not one this machine's hk.tune wrote.
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.delenv("HK_AUTOTUNE", raising=False)
    records.forget()
    gemm._USER.clear()
    yield
    records.forget()
    gemm._USER.clear()


@pytest.mark.skipif(not _CASES, reason="no shipped records for this arch")
@pytest.mark.parametrize("m,n,k", _CASES, ids=[f"{m}x{n}x{k}" for m, n, k in _CASES] or None)
def test_a_shipped_schedule_matches_fp32(m, n, k):
    assert gemm.user_route(m, n, k) is not None, "the shipped record should answer"
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    out = gemm.matmul(a, b)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
