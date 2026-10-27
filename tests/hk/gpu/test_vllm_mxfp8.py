"""hk.integration.vllm_mxfp8: the patched block-scaled MXFP8 layer is vLLM's, bit for bit, on
shapes hk takes and on shapes it leaves to vLLM (needs vLLM's ROCm build: the vLLM image)."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled MFMA", allow_module_level=True)

from hk.integration import vllm_mxfp8 as P  # noqa: E402


class _Layer(torch.nn.Module):
    pass


@pytest.mark.parametrize("m,n,k", [(2048, 1792, 5120), (2048, 5120, 1152), (5120, 2304, 5120),
                                   (64, 5120, 1152), (2048, 768, 512),
                                   # MIN_M_NK: the indexer's wq_b from 2825, engram's wkv from
                                   # 1024 -- at, one under, and ragged above
                                   (2825, 4096, 1280), (2824, 4096, 1280), (3073, 4096, 1280),
                                   (1024, 12800, 6144), (1023, 12800, 6144),
                                   (1537, 12800, 6144)])
def test_patched_layer_matches_vllm(m, n, k):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm

    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    layer = _Layer()
    layer.weight = torch.nn.Parameter(torch.randn(n, k, device="cuda", generator=g)
                                      .to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(120, 135, (n // 32, k // 32), device="cuda", dtype=torch.uint8, generator=g),
        requires_grad=False)
    P._ORIG_PWAL = lambda self, layer: None
    P.PATCH.define(P.mxfp8_linear)
    P.process_weights_after_loading(None, layer)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    xq, xs = mxfp8_e4m3_quantize(x)
    ref = rocm_mxfp8_block32_gemm(xq, xs, layer.weight, layer.weight_scale, torch.bfloat16)
    before = dict(P.STATS)
    out = P.apply_weights(None, layer, x)
    took = {kk: P.STATS.get(kk, 0) - before.get(kk, 0) for kk in ("hk", "fallback")}
    assert took == ({"hk": 1, "fallback": 0} if P.schedule(m, n, k) else {"hk": 0, "fallback": 1})
    assert torch.equal(out, ref)
