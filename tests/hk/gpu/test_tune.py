"""hk.tune end to end on the GPU it runs on: a small search writes a record
that hk.ops.matmul and hk.templates.gemm then run, correctly.

Small on purpose (a handful of candidates, a few GPU-seconds): the claim
under test is that the path works -- search, acceptance, record, lookup,
launch -- not that a particular schedule wins. The search's logic and the
acceptance rule are tested on synthetic timings in tests/hk/ir/test_tune.py.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
import hk.schedules as S  # noqa: E402
from hk.target import current_arch  # noqa: E402
from hk.tune import records  # noqa: E402

pytestmark = pytest.mark.gpu

ARCH = current_arch()
if ARCH not in ("gfx950", "gfx1100"):
    pytest.skip(f"hk.tune's GEMM spaces are gfx950's and gfx1100's, not {ARCH}",
                allow_module_level=True)


@pytest.fixture
def tunedir(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.delenv("HK_AUTOTUNE", raising=False)
    records.forget()
    yield tmp_path
    records.forget()


def _check_matmul(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    ref = a.float() @ b.float().t()
    for out in (hk.ops.matmul(a, b),
                hk.templates.gemm(a, b)):
        err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 2e-2, err


def test_a_small_search_end_to_end(tunedir):
    import hk.tune

    m, n, k = (256, 2048, 1024) if ARCH == "gfx950" else (256, 1024, 1024)
    res = hk.tune.tune(m, n, k, budget=10, log=None, workers=8, pool_per_kind=1, explore=0,
                       max_seeds=3, local_iters=1, local_moves=2, rung_rounds=(3, 5),
                       decide_sets=2, decide_rounds=5, confirm_sets=1)
    assert res.decision is not None and res.gpu_seconds > 0
    assert res.route_ms and res.baselines["torch"]["ms"]
    d = res.as_dict()
    json.dumps(d)
    path = records.path(ARCH)
    if res.verdict == "win":
        rec = json.loads(path.read_text())
        assert rec["routes"][f"{n}x{k}"] == {str(m): "u:" + res.champion.label}
        name = records.user_route(ARCH, m, n, k)[0]
        assert name == "u:" + res.champion.label
    else:
        assert not path.exists() or f"{n}x{k}" not in json.loads(path.read_text())["routes"]
    _check_matmul(m, n, k)


def test_a_record_is_what_matmul_and_the_template_run(tunedir):
    """A hand-written record of a legal schedule other than the route: both
    entry points take it for its M bucket and stay correct."""
    if ARCH == "gfx950":
        from hk.ops import gemm_cdna

        m, n, k = 300, 1536, 2048
        s = S.Dp(block_m=64, block_n=96, stages=4, xcd_chunk=0)
        records.save(ARCH, m, n, k, "u:test", s.to_record(), {"m": m, "n": n, "k": k})
        assert gemm_cdna.route(m, n, k) == "u:test"
        assert gemm_cdna.route(m, n, k, user=False) != "u:test"
        assert gemm_cdna.route(1024, n, k) != "u:test"           # another M bucket
    else:
        from hk.ops import gemm

        m, n, k = 300, 1536, 2048
        s = S.Rdna(block_m=64, block_n=128, warp_rows=2, wgm=4)
        records.save(ARCH, m, n, k, "u:test", s.to_record(), {"m": m, "n": n, "k": k})
        assert gemm.schedule_for(m, n, k)["block_m"] == 64
        assert gemm.schedule_for(m, n, k, user=False) == gemm.DEFAULT
    _check_matmul(m, n, k)
    _check_matmul(260, n, k)                                    # same bucket, ragged M
