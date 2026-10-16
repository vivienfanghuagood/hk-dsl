"""Every kernel the examples define, built the way the examples build them.

The test suite traces and compiles all of them (tests/hk/ir/test_examples.py,
tests/hk/codegen/test_example_builds.py), so a change to the language that
breaks an example fails there rather than in a reader's terminal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent


def load(name: str):
    """examples/<name>.py as a module (its file name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(f"hk_example_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kernels() -> List[Tuple[str, object]]:
    add = load("01_vector_add")
    rms = load("02_rmsnorm")
    sm = load("03_softmax")
    gemm = load("04_gemm")
    attn = load("05_attention")
    cf = load("06_control_flow")
    drop = load("07_dropout")
    emb = load("08_embedding")
    kv = load("09_kv_scatter")
    lnb = load("10_layernorm_backward")
    vi = load("11_vllm_integration")
    return [
        ("vector_add", add.add),
        ("rmsnorm", rms.rmsnorm_kernel()),
        ("rmsnorm_strip", rms.rmsnorm_strip_kernel()),
        ("softmax", sm.softmax_kernel()),
        ("gemm", gemm.gemm),
        ("attention", attn.attention_kernel()),
        ("attention_causal", attn.attention_kernel(causal=True)),
        ("prefix_sum_for", cf.prefix_sum_for),
        ("prefix_sum_while", cf.prefix_sum_while),
        ("dropout", drop.dropout_kernel()),
        ("embedding", emb.embedding),
        ("kv_scatter", kv.kv_scatter),
        ("ln_bwd_dx", lnb.ln_bwd_dx_kernel()),
        ("ln_bwd_dx_held", lnb.ln_bwd_dx_kernel(*lnb.plan(2048))),
        ("ln_bwd_dwdb", lnb.ln_bwd_dwdb),
        ("silu_mul_vllm", vi.silu_mul),
    ]


#: The scripts a GPU run executes, each with the arguments it runs under.
SCRIPTS = [
    ("01_vector_add.py",),
    ("02_rmsnorm.py",),
    ("03_softmax.py",),
    ("04_gemm.py",),
    ("05_attention.py",),
    ("05_attention.py", "--causal"),
    ("06_control_flow.py",),
    ("07_dropout.py",),
    ("08_embedding.py",),
    ("09_kv_scatter.py",),
    ("10_layernorm_backward.py",),
    ("11_vllm_integration.py",),
    ("validate.py", "--quick"),
]
