"""SGLang's linear patch, without a GPU or SGLang: which routing table it
uses, and that revert puts back both SGLang's apply and vLLM's table."""

import importlib.machinery
import json
import sys
import types

import pytest

from hk.integration import engine, vllm_gemm
from hk.integration import sglang as hks

QKV = (6144, 4096)
HOME = "sglang.srt.layers.quantization.unquant"


class Unquantized:
    def apply(self, layer, x, bias=None):
        return "sglang"


@pytest.fixture
def fake_sglang(monkeypatch):
    parts = HOME.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules[HOME].UnquantizedLinearMethod = Unquantized
    monkeypatch.setattr(vllm_gemm, "register_op", lambda: None)  # needs torch
    orig = Unquantized.apply
    vllm_gemm.set_routes(None)
    yield
    hks.revert()
    Unquantized.apply = orig
    vllm_gemm.set_routes(None)


def test_a_table_measured_against_f_linear_is_preferred(fake_sglang, tmp_path, monkeypatch):
    table = dict(kind="sglang_linear", arch="gfx1100", engine="sglang F.linear",
                 shapes={"6144x4096": dict(name="qkv", runs=[[1, 64, "hk_decode"]],
                                           above="hk")})
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(table))
    monkeypatch.setenv(hks.ROUTES_ENV, str(path))
    orig = Unquantized.apply

    assert "sglang F.linear" in hks.apply_linear()
    assert Unquantized.apply is not orig
    # vLLM's table keeps M=1 on vLLM's skinny GEMM; this one sends it to hk.
    assert vllm_gemm.route_for(1, *QKV) == "hk_decode"

    assert "linear -> sglang" in hks.revert()
    assert Unquantized.apply is orig
    assert vllm_gemm.route_for(1, *QKV) == "fallback"


def test_without_one_it_routes_by_vllms(fake_sglang, monkeypatch):
    real = engine.shipped_table
    monkeypatch.setattr(engine, "shipped_table",
                        lambda kind, arch, env: None if kind == "sglang_linear"
                        else real(kind, arch, env))

    assert "vllm 0.31.0" in hks.apply_linear()
    assert vllm_gemm.route_for(1, *QKV) == "fallback"


def test_a_missing_override_is_an_error_not_a_silent_fallback(fake_sglang, tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv(hks.ROUTES_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        hks.apply_linear()


SHAPES = (QKV, (4096, 4096), (24576, 4096), (4096, 12288), (151936, 4096))


@pytest.mark.parametrize("nk", SHAPES)
def test_the_shipped_table_gives_every_decode_batch_to_hk(fake_sglang, monkeypatch, nk):
    # Measured against SGLang's F.linear (torch 2.9.1, hipBLASLt), which runs
    # Qwen3-8B's decode GEMVs at 40-80% of the generated GEMV's speed -- so
    # unlike vLLM's table, M=1 is hk's too.
    monkeypatch.delenv(hks.ROUTES_ENV, raising=False)
    assert "sglang F.linear" in hks.apply_linear()
    assert {vllm_gemm.route_for(m, *nk) for m in (1, 2, 3, 8, 32, 64)} == {"hk_decode"}


def test_the_plugin_does_nothing_unless_asked(fake_sglang, monkeypatch, capsys):
    from hk.integration import _sglang_plugin

    monkeypatch.delenv("HK_SGLANG_LINEAR", raising=False)
    monkeypatch.delenv("HK_SGLANG_MOE", raising=False)
    orig = Unquantized.apply
    _sglang_plugin.register()
    assert Unquantized.apply is orig
    assert capsys.readouterr().out == ""


def test_the_plugin_patches_when_asked_and_never_raises(fake_sglang, monkeypatch, capsys):
    from hk.integration import _sglang_plugin

    monkeypatch.setenv("HK_SGLANG_LINEAR", "1")
    _sglang_plugin.register()
    assert "linear -> hk" in capsys.readouterr().out

    def boom():
        raise RuntimeError("no table")

    hks.revert()
    monkeypatch.setattr(hks, "apply_linear", boom)
    _sglang_plugin.register()
    assert "linear not patched (RuntimeError: no table)" in capsys.readouterr().out
