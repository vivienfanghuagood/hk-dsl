"""The 4-bit MoE patch (moe_wna16), without a GPU or vLLM: its schedules, what it refuses and why,
the config it asks vLLM for, and its switch."""

import pytest

from hk.integration import _common, switches, vllm_moe_wna16 as r

torch = pytest.importorskip("torch")


def test_the_patch_declares_itself_graph_safe():
    assert r.PATCH.graph == "safe"
    assert r.STATS is r.PATCH.stats
    assert set(r.PATCH.paths) == {"hk", "fallback"}


def test_it_is_off_until_its_gate():
    route = switches.ROUTES["vllm_moe_wna16"]
    assert route.env == "HK_VLLM_MOE_WNA16" and not route.default and route.arches == ()


@pytest.mark.parametrize("k,want", [(2048, (8, 8)), (512, (8, 2)), (4096, (8, 16)), (768, (8, 2)), (1536, (8, 4)),
                                    (128, None)])
def test_schedules_are_two_eight_slice_loads_a_warp(k, want):
    assert r.schedule(k) == want


def _call(tokens=4, k=2048, n=1024, dtype=torch.float16, zp=True, bm=16, int4=True, bs=(0, 32), e=256):
    a = torch.empty(tokens, k, dtype=dtype)
    b = torch.empty(e, n, k // 2, dtype=torch.uint8)
    s = torch.empty(e, n, k // 32, dtype=torch.float16)
    z = torch.empty(e, n // 2, k // 32, dtype=torch.uint8) if zp else None
    c = torch.empty(tokens, 8, n, dtype=torch.float16)
    return r._why_not(a, b, c, s, z, tokens, {"BLOCK_SIZE_M": bm}, False, int4, list(bs))


def test_what_it_refuses_and_why():
    assert _call() == "dtype"                       # a CPU tensor: not is_cuda
    assert _call(zp=False) == "no zero points"
    assert _call(int4=False) == "not int4"
    assert _call(bm=32) == "block_m not 16"
    assert _call(tokens=33) == "tokens"
    assert _call(bs=(128, 128)) == "block shape"


def test_only_the_validated_shapes_unless_forced(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    assert _call() is None and _call(k=512, n=2048) is None       # Qwen3.6's w13 and w2
    assert _call(e=128) == "not validated" and _call(n=768, k=2048) == "not validated"
    monkeypatch.setattr(r, "_FORCE", True)
    assert _call(e=128) is None


def test_the_config_asks_for_16_row_blocks_at_small_batches(monkeypatch):
    monkeypatch.setattr(r, "_ORIG_CONFIG", lambda *a: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64})
    w1, w2 = (256, 1024, 1024), (256, 2048, 256)
    assert r.try_get_optimal_moe_config(w1, w2, 8, "int4_w4a16", 8, [0, 32])["BLOCK_SIZE_M"] == 16
    assert r.try_get_optimal_moe_config(w1, w2, 8, "int4_w4a16", 64, [0, 32])["BLOCK_SIZE_M"] == 32
    assert r.try_get_optimal_moe_config(w1, w2, 8, None, 8, None)["BLOCK_SIZE_M"] == 32


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply()")
    assert "not importable" in r.apply()
    assert r.status() == "not patched"
