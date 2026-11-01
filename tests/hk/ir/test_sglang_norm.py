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
