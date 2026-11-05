"""SGLang's GemmaRMSNorm route without a GPU or SGLang: where it applies, that it leaves
SGLang's own fused paths alone, what its engine contract checks, and that revert puts the
forward back."""

import importlib.machinery
import sys
import types

import pytest

from hk.integration import contract
from hk.integration import sglang as hks


class GemmaRMSNorm:
    def forward_hip(self, x, residual=None, post_residual_addition=None):
        return "sglang"


@pytest.fixture
def fake_layernorm(monkeypatch):
    parts = hks._NORM_HOME.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)
    L = sys.modules[hks._NORM_HOME]
    L.GemmaRMSNorm, L._use_aiter, L._has_vllm_rms_norm = GemmaRMSNorm, False, False
    orig = GemmaRMSNorm.forward_hip
    yield L
    hks.revert()
    GemmaRMSNorm.forward_hip = orig


def test_it_takes_the_native_path_and_reverts(fake_layernorm):
    orig = GemmaRMSNorm.forward_hip
    assert "norm -> hk" in hks.apply_norm()
    assert GemmaRMSNorm.forward_hip is not orig
    assert hks.apply_norm() == "norm: already patched"
    assert "norm -> sglang" in hks.revert()
    assert GemmaRMSNorm.forward_hip is orig


def test_pause_reaches_a_layer_that_cached_the_bound_forward(fake_layernorm, monkeypatch):
    """A layer built while the route was on keeps hk's bound forward_hip; pause_norm makes that
    method run SGLang's, and unpausing gives hk's back."""
    monkeypatch.setattr(hks, "gemma_rms_norm", lambda *a: "hk")
    monkeypatch.setattr(hks, "_norm_refusal", lambda *a: "")
    hks.apply_norm()
    layer = GemmaRMSNorm()
    layer.weight, layer.variance_epsilon = None, 1e-6
    bound = layer.forward_hip                         # what SGLang's CustomOp keeps
    assert bound("x") == "hk"
    hks.pause_norm(True)
    try:
        assert bound("x") == "sglang"
    finally:
        hks.pause_norm(False)
    assert bound("x") == "hk"


@pytest.mark.parametrize("flag", ["_use_aiter", "_has_vllm_rms_norm"])
def test_sglangs_own_fused_paths_are_left_alone(fake_layernorm, flag):
    setattr(fake_layernorm, flag, True)
    orig = GemmaRMSNorm.forward_hip
    assert "not patched" in hks.apply_norm()
    assert GemmaRMSNorm.forward_hip is orig


def test_the_contract_holds_on_the_stand_in_and_breaks_on_a_new_signature(fake_layernorm):
    assert contract.CHECKS["sglang_norm"]() == []

    class Moved:
        def forward_hip(self, x, residual=None):
            return "sglang"

    fake_layernorm.GemmaRMSNorm = Moved
    assert contract.CHECKS["sglang_norm"]()
