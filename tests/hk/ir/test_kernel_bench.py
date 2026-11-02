"""tools/hk-bench/kernel_bench.py without a GPU: the shapes it reads from a config, how a row is
summarised and drawn, and the table it renders from a saved run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

KB = Path(__file__).resolve().parents[3] / "tools" / "hk-bench" / "kernel_bench.py"


@pytest.fixture(scope="module")
def kb():
    spec = importlib.util.spec_from_file_location("kernel_bench", KB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(tmp_path, **text):
    base = dict(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, head_dim=128,
                num_hidden_layers=4, vocab_size=73440)
    p = tmp_path / "config.json"
    p.write_text(json.dumps({**base, **text}))
    return str(p)


def test_a_dense_models_shapes_and_a_moe_models_shared_expert(kb, tmp_path):
    dense = {name: (n, k) for name, n, k in kb.linear_shapes(_cfg(tmp_path, intermediate_size=6144), 1)}
    assert dense == {"qkv": (2560, 2048), "o": (2048, 2048), "gate_up": (12288, 2048),
                     "down": (2048, 6144), "lm_head": (73472, 2048)}  # vocab padded to 64, as vLLM does
    moe = [s[0] for s in kb.linear_shapes(_cfg(tmp_path, num_experts=256, num_experts_per_tok=8,
                                                moe_intermediate_size=512,
                                                shared_expert_intermediate_size=512), 2)]
    assert moe == ["qkv", "o", "shared_gate_up", "shared_down", "lm_head"]
    assert kb.moe_shape(_cfg(tmp_path, num_experts=256, num_experts_per_tok=8,
                             moe_intermediate_size=512), 2) == (256, 8, 2048, 256)


def test_a_row_names_hks_best_the_best_competitor_and_their_ratio(kb):
    row = kb.summarise({"shape": "o", "m": 8, "arms": {
        "hk decode_gemm": {"us": 16.0}, "hk gemm": {"us": 33.0}, "hipBLASLt": {"us": 24.0},
        "vLLM": {"us": 23.2}, "FlyDSL": {"wrong": 0.5}}})
    assert (row["hk"], row["best"], row["ratio"]) == ("hk decode_gemm", "vLLM", 1.45)


def test_the_table_marks_loose_and_wrong_arms(kb):
    row = kb.summarise({"shape": "qkv", "m": 1, "arms": {
        "hk w4a16": {"us": 77.5, "err": 0.004}, "vLLM gptq_gemm_rdna3": {"us": 61.7, "err": 0.032},
        "vLLM gptq_gemm (exllama)": {"wrong": 0.4}}})
    t = kb.table([row], ["hk w4a16", "vLLM gptq_gemm (exllama)", "vLLM gptq_gemm_rdna3"])
    assert "61.70 †" in t and "| wrong |" in t and "**0.80x**" in t


def test_a_saved_run_renders_with_each_arms_setup(kb):
    d = dict(op="moe", model="/srv/hf/hub/models--openbmb--MiniCPM-V-4.7-35B-A3B/snapshots/x",
             tp=2, rounds=5, rotate_mb=576, notes={"AITER": "n/a: CDNA-only"},
             env=dict(gpu="W7900D", arch="gfx1100", torch="2.13", hip="7.14", vllm="0.1",
                      triton="3.7", flydsl="0.4", aiter="n/a", hk="0.8"),
             rows=[kb.summarise({"shape": "E256", "m": m, "arms": {
                 "hk": {"us": 30.0 + m, "err": 0.0}, "vLLM Triton (searched config)": {"us": 44.0}},
                 "setup": {"hk": "BLOCK_M 16", "vLLM Triton (searched config)": f"config {m}"}})
                   for m in (1, 2)])
    md = kb.render(d)
    assert "### moe at openbmb/MiniCPM-V-4.7-35B-A3B (TP2)" in md
    assert "- hk: BLOCK_M 16" in md and "E256 M=2: config 2" in md and "AITER: n/a" in md
