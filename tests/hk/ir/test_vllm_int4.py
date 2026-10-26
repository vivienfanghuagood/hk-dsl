"""The 4-bit patch, without a GPU or vLLM."""

import pytest

from hk.integration import _common, vllm_int4
from hk.ops import w4a16


def test_the_op_declares_itself_graph_safe():
    assert vllm_int4.PATCH.graph == "safe"
    assert set(vllm_int4.PATCH.paths) == {"hk", "fallback"}


def test_the_fake_is_tokens_by_output_features():
    torch = pytest.importorskip("torch")
    x = torch.empty(5, 4096, device="meta", dtype=torch.bfloat16)
    qw = torch.empty(6144, 512, device="meta", dtype=torch.int32)
    out = vllm_int4._fake(x, None, None, None, qw, None, None, 128)
    assert out.shape == (5, 6144)


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply_patch()")
    assert "not importable" in vllm_int4.apply_patch()
    assert vllm_int4.status() == "not patched"


def test_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract

    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setattr(contract, "check", lambda routes: [])   # no vLLM here
    monkeypatch.setenv("HK_PATCH_VLLM", "1")
    monkeypatch.setenv("HK_VLLM_INT4", "1")
    monkeypatch.setattr(vllm_int4, "apply_patch", lambda: seen.append("int4") or "int4: test")
    _vllm_plugin.register()
    assert seen == ["int4"]


@pytest.mark.parametrize("m,want", [(1, True), (16, True), (17, True), (64, True),
                                    (65, False), (256, False)])
def test_the_decode_kernel_takes_batches_to_64(m, want):
    assert w4a16.takes(m, 4096, 4096, 128) is want


def test_a_group_narrower_than_the_load_is_not_taken():
    assert not w4a16.takes(1, 4096, 4096, 32)      # one token loads 64 columns
    assert w4a16.takes(17, 4096, 4096, 32)         # two load 32
