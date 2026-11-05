"""docs/language.md's launchable kernels give torch's answer. row_sumsq once stored its row sums with
o.at[row, warp] -- along a row, not down a column -- and the page's first example was wrong for months."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _doc_blocks  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    return _doc_blocks.load(tmp_path_factory.mktemp("doc"))


@pytest.mark.parametrize("rows,cols", [(64, 1024), (128, 4096), (48, 320)])
def test_row_sumsq(doc, rows, cols):
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    o = torch.zeros(4, rows, device="cuda")
    doc.row_sumsq(x, o)
    torch.testing.assert_close(o.sum(0), x.float().pow(2).sum(1), rtol=1e-3, atol=1e-2)


def test_scale_pow2(doc):
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    doc.scale_pow2(x, o, N=3)
    torch.testing.assert_close(o, x * 8, rtol=0, atol=0)


def test_silu_mul(doc):
    import hk  # noqa: PLC0415

    a = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    o = torch.empty_like(a)
    doc.silu_mul(dtype=hk.fp16)(a, b, o)
    ref = (torch.nn.functional.silu(a.float()) * b.float()).half()
    torch.testing.assert_close(o, ref, rtol=2e-2, atol=2e-2)
