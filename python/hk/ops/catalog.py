"""The op catalog: every kernel family `hk.ops` ships, the GPUs it runs on, its dtypes, the engine
routes that use it and the GPU tests that run it. docs/ops.md is generated from this table
(scripts/gen_hardware_docs.py), and tests/hk/ir/test_ops_catalog.py holds it to the code: every
public name of `hk.ops` is in a family, every route is one of hk.integration.switches', every test
file exists, and a family behind `hk.ops._arch.require` lists the GPUs that check allows.

Plain data, with no imports: the generator reads it without importing hk.
"""

from __future__ import annotations

#: The GPU columns, in the hardware page's order.
ARCHES = ("gfx1100", "gfx1201", "gfx942", "gfx950")

#: The GPUs of `hk.ops._arch.RDNA_AND_GFX950`: GEMM, attention, paged attention, decode GEMM,
#: MoE and W4A16, whose RDNA kernels do not build for gfx942.
RDNA_AND_GFX950 = ("gfx1100", "gfx1201", "gfx950")
EVERY = ARCHES

#: name: what the family is; entries: its public entry points; arches: the GPUs it runs on;
#: dtypes: what it takes; routes: hk.integration.switches routes that call it; tests: the GPU
#: tests that run it (tests/hk/gpu); note: what the columns cannot say.
FAMILIES = [
    dict(name="elementwise", what="one op an element: arithmetic and activations",
         entries=["hk.ops.add", "hk.ops.sub", "hk.ops.mul", "hk.ops.maximum", "hk.ops.minimum",
                  "hk.ops.exp", "hk.ops.relu", "hk.ops.gelu", "hk.ops.neg", "hk.ops.abs"],
         arches=EVERY, dtypes="bf16, fp16, fp32 (the kernels HipKittens specialises per op)",
         routes=[], tests=["test_elementwise.py", "test_cdna_rows.py"],
         note="register strips on gfx1100, gfx1201 and gfx950; tiles on gfx942"),
    dict(name="norm", what="RMSNorm, the fused add + RMSNorm, LayerNorm, softmax, over the last axis",
         entries=["hk.ops.rmsnorm", "hk.ops.rms_norm", "hk.ops.fused_add_rms_norm",
                  "hk.ops.layernorm", "hk.ops.softmax"],
         arches=EVERY, dtypes="bf16, fp16, fp32", routes=["sglang_norm"],
         tests=["test_phase2.py", "test_strip.py", "test_rms_norm_weighted.py"]),
    dict(name="fused", what="SwiGLU's activation half (SiLU-and-mul) and RoPE",
         entries=["hk.ops.silu_mul", "hk.ops.rope", "hk.ops.rotary"],
         arches=EVERY, dtypes="bf16, fp16, fp32", routes=[],
         tests=["test_phase2.py", "test_cdna_rows.py"]),
    dict(name="quant", what="dynamic per-row int8 quantization and its inverse",
         entries=["hk.ops.quantize", "hk.ops.dequantize"],
         arches=EVERY, dtypes="bf16, fp16, fp32 to int8 and back", routes=[],
         tests=["test_phase2.py", "test_cdna_rows.py"]),
    dict(name="gemm", what="the GEMM, C = A B^T with fp32 accumulate; its gfx950 kernels are "
                           "`gemm_cdna` and `gemm_cdna_dp`",
         entries=["hk.ops.matmul"],
         arches=RDNA_AND_GFX950, dtypes="bf16",
         routes=["vllm_gemm", "sglang_linear", "sgld_linear"],
         tests=["test_gemm_shipped.py", "test_gemm_cdna.py", "test_vllm_gemm.py"]),
    dict(name="scaled_mm", what="an fp8 GEMM with per-tensor or per-token x per-channel scales (torch._scaled_mm's product)",
         entries=["hk.ops.scaled_mm"],
         arches=("gfx1201",), dtypes="fp8 (e4m3, e5m2) in, bf16 out", routes=[],
         tests=["test_scaled_mm.py"],
         note="gfx1100 has no fp8 WMMA; gfx950's fp8 GEMMs are the MXFP8 family"),
    dict(name="decode_gemm", what="the decode GEMV / skinny GEMM, M <= 64",
         entries=["hk.ops.decode_gemm.matmul"],
         arches=RDNA_AND_GFX950, dtypes="bf16",
         routes=["vllm_gemm", "sglang_linear"], tests=["test_decode_gemm.py"]),
    dict(name="w4a16", what="4-bit weights (AWQ, GPTQ) against 16-bit activations",
         entries=["hk.ops.w4a16.matmul"],
         arches=RDNA_AND_GFX950, dtypes="int4 weights, fp16 or bf16 activations",
         routes=["vllm_int4", "vllm_gptq"], tests=["test_w4a16.py"]),
    dict(name="moe", what="the expert GEMMs of a MoE layer, on vLLM's sorted token lists",
         entries=["hk.ops.moe.moe_gemm", "hk.ops.moe.moe_align"],
         arches=RDNA_AND_GFX950, dtypes="bf16",
         routes=["vllm_moe", "sglang_moe"], tests=["test_moe.py"]),
    dict(name="attention", what="flash attention, forward; its gfx950 kernel is `attn_cdna`",
         entries=["hk.ops.attention"],
         arches=RDNA_AND_GFX950, dtypes="bf16", routes=[], tests=["test_attn.py"]),
    dict(name="sdpa", what="`F.scaled_dot_product_attention`, on hk's attention where it fits "
                           "and torch's elsewhere",
         entries=["hk.ops.sdpa.scaled_dot_product_attention"],
         arches=RDNA_AND_GFX950, dtypes="bf16 (other calls fall back to torch)",
         routes=["vllm_sdpa", "sgld_attn"], tests=["test_sdpa.py"]),
    dict(name="attn_k3", what="non-causal attention at head_dim 128, the DiT kernel `hk.ops.sdpa` "
                              "takes on gfx1100",
         entries=["hk.ops.attn_k3.attention"],
         arches=("gfx1100",), dtypes="bf16",
         routes=["sdpa_k3", "sgld_attn"], tests=["test_attn_k3.py"]),
    dict(name="paged", what="decode attention over a paged KV cache: the split walk and its "
                            "merge; its gfx950 kernel is `paged_cdna`",
         entries=["hk.ops.paged.split_kernel", "hk.ops.paged.fused_kernel",
                  "hk.ops.paged.merge_kernel", "hk.ops.paged.plan_splits",
                  "hk.ops.paged.merge_splits"],
         arches=RDNA_AND_GFX950, dtypes="bf16",
         routes=["vllm_backend"], tests=["test_paged.py", "test_paged_fused.py"]),
    dict(name="paged_prefill", what="prefill attention over a paged KV cache (chunked, causal)",
         entries=["hk.ops.paged_prefill.kernel_for", "hk.ops.paged_prefill.work_list"],
         arches=RDNA_AND_GFX950, dtypes="bf16",
         routes=["vllm_prefill"], tests=["test_paged_prefill.py"]),
    dict(name="hipkittens", what="HipKittens' own CDNA4 kernels as hk ops: the bf16 GEMM and GQA "
                                 "attention",
         entries=["hk.ops.hipkittens.gemm_bf16", "hk.ops.hipkittens.gqa_fwd"],
         arches=("gfx950",), dtypes="bf16", routes=[], tests=["test_hipkittens.py"],
         note="a candidate in vLLM's gfx950 linear table, whose route is off by default there"),
    dict(name="mxfp8", what="the block-scaled MXFP8 GEMM and the activation quant (DeepSeek-V4.1)",
         entries=["hk.ops.mxfp8.gemm", "hk.ops.mxfp8.quant"],
         arches=("gfx950",), dtypes="MXFP8 (fp8 e4m3 with E8M0 scales) in, bf16 out",
         routes=["vllm_mxfp8"], tests=["test_mxfp8_gemm.py", "test_mxfp8_quant.py"]),
    dict(name="moe_a8w4", what="the a8w4 MoE: FP8 activations against MXFP4 expert weights "
                               "(DeepSeek-V4.1)",
         entries=["hk.ops.moe_a8w4.stage1_sorted", "hk.ops.moe_a8w4.stage2_sorted"],
         arches=("gfx950",), dtypes="fp8 activations, MXFP4 weights, bf16 out",
         routes=["vllm_moe_a8w4"], tests=["test_moe_a8w4.py"]),
    dict(name="sparse_mla", what="DeepSeek-V4.1's sparse MLA prefill attention",
         entries=["hk.ops.sparse_mla.prefill", "hk.ops.sparse_mla.prefill_mxfp8"],
         arches=("gfx950",), dtypes="bf16; MXFP8 output rows",
         routes=["vllm_sparse_mla"], tests=["test_sparse_mla.py"]),
    dict(name="dsv41_index", what="DeepSeek-V4.1's index chain for a decode step, one resident "
                                  "launch",
         entries=["hk.ops.dsv41_index.index_kernel"],
         arches=("gfx950",), dtypes="bf16 activations; MXFP8 (qr, wq_b) and MXFP4 (q, K cache) inside the chain",
         routes=["vllm_dsv41_index"], tests=["test_vllm_dsv41_index.py"]),
    dict(name="convrot", what="ConvRot W8A8 linear: a rotated int8 GEMM",
         entries=["hk.ops.convrot.linear"],
         arches=("gfx1100",), dtypes="bf16 in, int8 W8A8, bf16 out",
         routes=["sgld_convrot"], tests=["test_convrot.py"]),
]
