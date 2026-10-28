"""vllm_moe's token limit by arch: hk's MoE GEMM takes a step's calls up to it, vLLM's
kernel past it (tools/hk-bench/moe_ab.py on each arch)."""

from __future__ import annotations

import hk.target
from hk.integration import vllm_moe


def test_gfx1100_keeps_its_limit_and_gfx1201_has_its_own(monkeypatch):
    monkeypatch.setattr(hk.target, "current_arch", lambda: "gfx1100")
    assert vllm_moe.block16_max() == vllm_moe.BLOCK16_MAX_TOKENS == 256
    monkeypatch.setattr(hk.target, "current_arch", lambda: "gfx1201")
    # Up to vLLM's unsorted, one-pair-a-block assignment only (2 tokens of OLMoE).
    assert vllm_moe.block16_max() == 2
