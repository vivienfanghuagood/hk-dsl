"""On RDNA, `k(...)` launches on the null stream: the pybind entry point takes
no stream (`scaffold.takes_stream`), and the null stream is ordered against
nothing torch queued on a stream of its own. Called under a non-default torch
stream, such a launch can read its inputs before they are written -- SGLang's
norm route returned NaN that way. So the launch checks torch's current stream:
on the default one it is silent; on any other it warns once for that kernel,
and with HK_STRICT_STREAM=1 raises. The warning does not change the launch:
with the producer's stream synchronized first, the result is the default
stream's, bit for bit.
"""

from __future__ import annotations

import importlib
import os
import warnings

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch, get_target  # noqa: E402

if get_target(current_arch()).wave != 32:
    pytest.skip("CDNA's entry point takes torch's stream (test_stream_cdna.py)",
                allow_module_level=True)
if os.environ.get("HK_LAUNCH") == "codeobj":
    pytest.skip("HK_LAUNCH=codeobj launches on torch's stream", allow_module_level=True)

K = importlib.import_module("hk.dsl.kernel")   # hk.dsl.kernel is also the decorator's name
from hk.ops import norm  # noqa: E402

ROWS, COLS = 256, 4096


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    K._NULL_STREAM_WARNED.clear()
    monkeypatch.delenv("HK_STRICT_STREAM", raising=False)


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.float16, generator=g)
    w = torch.randn(COLS, device="cuda", dtype=torch.float16, generator=g)
    return x, w


def _on_default_stream(x, w):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = norm.rms_norm(x, w)
    torch.cuda.synchronize()
    return y


def test_the_default_stream_is_silent():
    x, w = _inputs()
    y = _on_default_stream(x, w)
    assert torch.isfinite(y).all()


def test_a_side_stream_warns_once_and_a_synchronized_producer_is_exact():
    x0, w = _inputs()
    want = _on_default_stream(x0, w)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # the producer: busy work on the side stream, then x
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        for _ in range(8):
            a = a @ a * 2048 ** -0.5
        x = x0 + a[:1, :1] * 0
        side.synchronize()
        with pytest.warns(RuntimeWarning, match="null stream"):
            y = norm.rms_norm(x, w)
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # once a kernel: the second call is silent
            y2 = norm.rms_norm(x, w)
    torch.cuda.synchronize()
    assert torch.equal(y, want)
    assert torch.equal(y2, want)


def test_hk_strict_stream_refuses(monkeypatch):
    x, w = _inputs()
    _on_default_stream(x, w)                     # built and loaded
    monkeypatch.setenv("HK_STRICT_STREAM", "1")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with pytest.raises(RuntimeError, match="HK_STRICT_STREAM=1"):
            norm.rms_norm(x, w)
    torch.cuda.synchronize()
