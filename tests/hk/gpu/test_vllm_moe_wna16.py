"""The 4-bit MoE route in vLLM's own call: hk's GEMM against vLLM's moe_wna16 kernel on the same
tensors, the batches it takes and the ones it leaves. Needs vLLM (the Radeon vLLM images)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from hk.integration import vllm_moe_wna16 as r  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx1" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="hk.load_w4(zp=) is RDNA's")


def _layer(e, n, k, g):
    q = torch.randint(0, 256, (e, n, k // 2), device="cuda", generator=g, dtype=torch.uint8)
    s = (torch.rand(e, n, k // 32, device="cuda", generator=g) * 0.02 + 0.001).half()
    z = torch.randint(0, 256, (e, n // 2, k // 32), device="cuda", generator=g, dtype=torch.uint8)
    return q, s, z


@pytest.mark.parametrize("tokens", [1, 7, 16, 32, 40])
@pytest.mark.parametrize("second", [False, True], ids=["w13", "w2"])
def test_the_route_matches_vllms_kernel_and_takes_up_to_32_tokens(tokens, second, monkeypatch):
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    monkeypatch.setattr(r, "_ORIG", fm.invoke_fused_moe_wna16_triton_kernel)
    monkeypatch.setattr(r, "_FORCE", True)          # a small layer, not one of VALIDATED
    g = torch.Generator(device="cuda").manual_seed(tokens + 100 * second)
    e, top_k = 64, 8
    n, k = (512, 1024) if not second else (1024, 256)
    q, s, z = _layer(e, n, k, g)
    topk_ids = torch.stack([torch.randperm(e, device="cuda", generator=g)[:top_k]
                            for _ in range(tokens)]).to(torch.int32)
    topk_w = torch.rand(tokens, top_k, device="cuda", generator=g)
    sid, eid, ntp = moe_align_block_size(topk_ids, 16, e)
    a = torch.randn(tokens * (top_k if second else 1), k, device="cuda", generator=g).half()
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
    args = dict(compute_type=tl.float16, use_int8_w8a16=False, use_int4_w4a16=True, block_shape=[0, 32])
    kk = 1 if second else top_k
    want = torch.zeros(tokens, top_k, n, device="cuda", dtype=torch.float16)
    fm.invoke_fused_moe_wna16_triton_kernel(a, q, want, s, z, topk_w, sid, eid, ntp, second, kk, cfg, **args)
    got = torch.zeros_like(want)
    before = dict(r.STATS)
    r.invoke_fused_moe_wna16_triton_kernel(a, q, got, s, z, topk_w, sid, eid, ntp, second, kk, cfg, **args)
    took = r.STATS["hk"] - before["hk"]
    assert took == (1 if tokens <= r.MAX_TOKENS else 0)
    top = want.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)
    assert ((got.float() - want.float()).abs() <= 4e-3 * top).all()
