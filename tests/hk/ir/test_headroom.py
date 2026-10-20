"""`python -m hk.integration headroom` without a GPU: what a trace is
attributed to, what the ranking puts first and why, and the CLI on a trace
from elsewhere."""

import json

import pytest

from hk.integration import __main__ as cli
from hk.integration import headroom as hr

#: Kernel names from vLLM 0.31's traces on gfx1100 (eager and graphs), hk's
#: own, and Inductor's -- each with the family it belongs to.
NAMES = [
    ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_LDSB0", "linear"),
    ("kernel_paged_attention_2d.kd", "attention"),
    ("_fwd_kernel.kd", "attention"),
    ("kernel_unified_attention_2d", "attention"),
    ("void vllm::reshape_and_cache_kernel<__hip_bfloat16, __hip_bfloat16>", "kv_write"),
    ("void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*)",
     "rotary"),
    ("std::enable_if<(((8)>(0))), void>::type vllm::fused_add_rms_norm_kernel<c10::BFloat16>",
     "norm"),
    ("_ZN4vllm18act_and_mul_kernelIN3c108BFloat16E15__hip_bfloat162TnPFT_RKS4_fE", "activation"),
    ("triton_red_fused_fused_add_rms_norm_2.kd", "norm"),
    ("triton_poi_fused_mul_silu_slice_1.kd", "activation"),
    ("_gumbel_sample_kernel.kd", "sampling"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, "
     "at::native::ArgMaxOps<float>>>", "sampling"),
    ("void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16>",
     "embedding"),
    ("void at::native::elementwise_kernel_manual_unroll<128, 8>", "elementwise"),
    ("Memcpy HtoD (Host -> Device)", "copy"),
    ("__amd_rocclr_copyBuffer.kd", "copy"),
    ("gemm_bf16_128x128x64_w8r4_s2_d16_g8_p2_l1_kernel(globals)", "linear"),
    ("attn_fwd_d128_causal_kernel(globals)", "attention"),
    ("triton_red_fused_3.kd", "other"),
    # SGLang 0.5.18's Triton attention: its decode is two stages
    ("_fwd_grouped_kernel_stage1", "attention"),
    ("_fwd_kernel_stage2", "attention"),
    ("_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_E", "activation"),
    # vLLM on gfx950 with aiter on: aiter's tuned GEMMs (its Triton a16w16, asm
    # hgemm, the mono-tile kernel), the skinny GEMMs, hipBLASLt; ROCM_ATTN's
    # ll4mi decode and ROCM_AITER_FA's asm prefill; aiter's fused add + RMSNorm;
    # CK's MoE GEMMs with aiter's sorting and vLLM's gating.
    ("_gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1"
     "_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_EVEN_MN_1_cache_modifier", "linear"),
    ("hgemm_bf16_t16x16x128x5_ks1_w1x1x1_gm4_bias0_lnt_pft_0.kd", "linear"),
    ("_Z35gemm_a16w16_mono_tile_kernel_gfx950ILi256ELi256E", "linear"),
    ("void wvSplitKrc_<__hip_bfloat16, 64, 16, 4, 8, 1, 16, 1, 1, 1>(int, int, int)", "linear"),
    ("void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 8, 2, 1>(int, int, int)", "linear"),
    ("void paged_attention_ll4mi_QKV_mfma16_kernel<__hip_bfloat16, __hip_bfloat16, "
     "(vllm::Fp8KVCacheDataType)0, __hip_bfloat16, 32, 128, 256, false, 4>", "attention"),
    ("void paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 128, 128, 256, 1>",
     "attention"),
    ("_ZN5aiter32fmha_fwd_hd128_bf16_causal_groupE.kd", "attention"),
    ("_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi16ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4"
     "_S4_dbiiiiiiibb.kd", "norm"),
    ("void vllm::reshape_and_cache_flash_kernel<__hip_bfloat16, __hip_bfloat16, "
     "(vllm::Fp8KVCacheDataType)0>", "kv_write"),
    ("_ZN2ck15kernel_moe_gemmINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorENS3_11Column"
     "MajorE", "moe"),
    ("_ZN2ck20kernel_moe_gemm_2ldsINS_15GridwiseMoeGemmINS_13tensor_layout4gemm8RowMajorE", "moe"),
    ("void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, "
     "float, 2, true>>>", "moe"),
    ("void vllm::moe::topkGatingSoftmax<bool _Accum, int, ELi128ELi8ELi32E, true, 0>", "moe"),
]


def test_steps_are_counted_by_the_attention_kernel_a_layer_calls():
    sgl_decode = {"_fwd_grouped_kernel_stage1": [13142.0, 259], "_fwd_kernel_stage2": [1431.0, 259],
                  "Cijk_x": [283610.0, 1042],
                  "void at::native::indexSelectSmallIndex<c10::BFloat16>": [46.0, 8]}
    assert hr.layer_steps(sgl_decode, 36) == pytest.approx(7.19)
    assert hr.layer_steps(sgl_decode, None) is None
    assert hr.layer_steps({"Cijk_x": [1.0, 145]}, 36) is None


@pytest.mark.parametrize("name, fam", NAMES, ids=[f for _, f in NAMES])
def test_each_kernel_lands_in_its_family(name, fam):
    assert hr.family(name) == fam


def test_hk_kernels_are_told_from_the_engines():
    assert hr.provider("gemm_bf16_128x128x64_kernel(globals)") == "hk"
    assert hr.provider("_Z30gemm_bf16_128x128x64_kernel7globals") == "hk"
    assert hr.provider("Cijk_Alik_Bljk_BBS_BH") == "engine"
    # FlyDSL's GEMM, which hk's table routes some prefill rows to, is hk's choice.
    assert hr.provider("wmma_gemm_kernel_0.kd") == "flydsl"
    assert hr.family("wmma_gemm_kernel_0.kd") == "linear"
    f = hr.attribute({"kernels": {"wmma_gemm_kernel_0.kd": [3.0, 1],
                                  "gemm_bf16_kernel(globals)": [1.0, 1],
                                  "Cijk_x": [4.0, 1]}, "steps": 1})["linear"]
    assert (f["us"], f["hk_us"]) == (8.0, 4.0)


#: Qwen3-8B's config.json, the fields the rooflines read.
QWEN3 = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
         "head_dim": 128, "intermediate_size": 12288, "num_hidden_layers": 36,
         "vocab_size": 151936, "architectures": ["Qwen3ForCausalLM"]}


def test_the_rooflines_follow_from_the_model_and_the_step():
    dec = hr.floors(QWEN3, {"tokens_per_step": [16], "requests_per_step": [16]}, "decode",
                    batch=16, context=4096, arch="gfx1100")
    # Every linear weight read once at 864 GB/s, and 16 x 4096 tokens of KV.
    weights = 36 * (6144 + 4096 + 24576 + 12288) * 4096 * 2 + 151936 * 4096 * 2
    assert dec["linear"][0] == pytest.approx(weights / 864e9 * 1e3, rel=0.01)
    assert dec["attention"][0] == pytest.approx(16 * 4096 * 36 * 2 * 8 * 128 * 2 / 864e9 * 1e3)
    pre = hr.floors(QWEN3, {"tokens_per_step": [8192], "requests_per_step": [2]}, "prefill",
                    batch=16, context=4096, arch="gfx1100")
    assert pre["attention"][0] == pytest.approx(2 * 36 * 2 * 32 * 4096 ** 2 * 128 / 122.6e12 * 1e3)
    assert hr.floors(QWEN3, {}, "decode", batch=16, context=4096, arch="gfx942") == {}


def test_gfx950s_rooflines_are_the_measured_peaks():
    peak, bw, how = hr.PEAKS["gfx950"]
    assert (peak, bw) == (2.48e15, 6.6e12) and "peaks.hip" in how
    dec = hr.floors(QWEN3, {}, "decode", batch=1, context=2048, arch="gfx950")
    assert dec["attention"][0] == pytest.approx(2048 * 36 * 2 * 8 * 128 * 2 / 6.6e12 * 1e3)


#: Qwen3-30B-A3B's: 48 layers, every MLP is 128 experts.
QWEN3_MOE = {"hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
             "head_dim": 128, "intermediate_size": 6144, "moe_intermediate_size": 768,
             "num_experts": 128, "num_experts_per_tok": 8, "mlp_only_layers": [],
             "num_hidden_layers": 48, "vocab_size": 151936,
             "architectures": ["Qwen3MoeForCausalLM"]}


def test_an_moe_models_linear_floor_leaves_out_the_experts():
    dec = hr.floors(QWEN3_MOE, {}, "decode", batch=1, context=2048, arch="gfx950")
    # Attention's projections and the LM head; no dense MLP.
    weights = 48 * (5120 * 2048 + 2048 * 4096) * 2 + 151936 * 2048 * 2
    assert dec["linear"][0] == pytest.approx(weights / 6.6e12 * 1e3, rel=0.01)
    assert "0 dense layer(s) of 48" in dec["linear"][1]


def test_with_no_ratio_the_roofline_is_the_ceiling():
    roof = {"decode": {"attention": (11.0, "KV once")}}
    ph = _phase({"paged_decode_fused_d128_s8_kernel(globals)": (33000.0, 72)})
    e = hr.rank({"decode": ph}, engine_name="vllm", arch="gfx1100", roof=roof)[0]
    assert e["roofline"]["of_peak"] == pytest.approx(11 / 16.5, 1e-3)
    assert e["ceiling"] == pytest.approx(1 - 11 / 16.5, 1e-3)
    assert e["ceiling_source"].startswith("roofline, at most")


def _ev(name, ts, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


def test_busy_time_counts_overlap_once():
    dev = hr.device_events([_ev("a", 0, 10), _ev("b", 5, 10), _ev("c", 30, 5),
                            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 99}])
    assert [d[0] for d in dev] == ["a", "b", "c"]
    assert hr.busy_us(dev) == 20


def test_a_trace_is_summed_by_name_and_graph_launches_counted():
    s = hr.summarize_trace([_ev("k", 0, 4), _ev("k", 10, 6), _ev("Memset", 20, 1, "gpu_memset"),
                            {"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch",
                             "ts": 0, "dur": 1}])
    assert s["kernels"] == {"k": [10.0, 2], "Memset": [1.0, 1]}
    assert s["span_us"] == 21 and s["graph_launches"] == 1


def _phase(kernels, steps=2, step_ms=None):
    busy = sum(us for us, _ in kernels.values()) / 1e3 / steps
    return {"kernels": {k: list(v) for k, v in kernels.items()}, "steps": steps,
            "step_ms": step_ms, "busy_ms": busy,
            "idle_ms": (step_ms - busy) if step_ms else None}


DECODE = {"Cijk_Alik_Bljk": (40000.0, 290), "kernel_paged_attention_2d.kd": (28000.0, 72),
          "void vllm::fused_add_rms_norm_kernel<c10::BFloat16>": (600.0, 290)}


def _routes(engine_ms, path_ms):
    return [{"name": "qkv", "n": 6144, "k": 4096, "m": 16, "calls": 36, "path": "hk_decode",
             "engine_ms": engine_ms, "path_ms": path_ms}]


def test_the_rank_is_the_upper_bound_times_a_ceiling():
    ents = hr.rank({"decode": _phase(DECODE, step_ms=35.0)}, engine_name="vllm",
                   arch="gfx1100", routes={"decode": _routes(0.2, 0.1)})
    by = {e["family"]: e for e in ents}
    lin = by["linear"]
    assert lin["upper_bound_ms"] == 20.0 and lin["share_of_step"] == pytest.approx(20 / 35, 1e-3)
    assert lin["ratio"] == 2.0 and lin["ceiling"] == 0.5 and lin["score_ms"] == 10.0
    assert lin["ceiling_source"] == "measured: route table"
    # Nothing measured hk's attention against this backend's kernel, so its
    # bound is all there is -- and the record's ratio, which was against
    # TRITON_ATTN's kernel, is not borrowed.
    att = by["attention"]
    assert att["ratio"] is None and att["ceiling"] is None and att["score_ms"] == 14.0
    assert [e["family"] for e in ents][:2] == ["attention", "linear"]
    assert by["(GPU idle)"]["upper_bound_ms"] == pytest.approx(35 - 34.3)
    assert all(e["upper_bound"] for e in ents)


def test_the_record_applies_only_to_the_kernel_it_measured():
    ph = _phase({"kernel_unified_attention_2d": (1000.0, 36)})
    att = hr.rank({"decode": ph}, engine_name="vllm", arch="gfx1100")[0]
    assert att["ratio"] == 0.83 and att["ceiling"] == 0.0 and "chapter 20" in att["ratio_source"]
    assert hr.rank({"decode": ph}, engine_name="vllm", arch="gfx950")[0]["ratio"] is None


def test_another_arm_measures_the_family_and_comes_first():
    hk_arm = {"decode": _phase({"paged_decode_split_kernel(globals)": (14000.0, 72),
                                "Cijk_Alik_Bljk": (40000.0, 290)})}
    ents = hr.rank({"decode": _phase(DECODE)}, engine_name="vllm", arch="gfx1100",
                   routes={"decode": _routes(0.2, 0.1)}, compare={"hk": hk_arm})
    by = {e["family"]: e for e in ents}
    assert by["attention"]["ratio"] == 2.0
    assert by["attention"]["ceiling_source"] == "measured: arm hk"
    # The other arm ran the engine's GEMM too: no ratio from it, the table's stands.
    assert [r["kind"] for r in by["linear"]["ratios"]] == ["measured: route table"]


def test_a_family_hk_already_runs_says_so():
    ents = hr.rank({"decode": _phase({"gemm_bf16_kernel(globals)": (1000.0, 10)})},
                   engine_name="vllm", arch="gfx1100", patched=["gemm"])
    e = ents[0]
    assert e["hk"]["runs_hk_kernels"] and e["hk"]["on_in_this_run"]
    assert e["ceiling_source"].startswith("none: hk's kernels already run 100%")


def test_a_graph_trace_without_its_replays_is_blind():
    d = {"graph_launches": 8, "step_ms": 50.0, "busy_ms": 3.0}
    assert "did not see" in hr._blind(d, graphs=True)
    assert hr._blind({**d, "busy_ms": 49.0}, graphs=True) is None
    assert hr._blind(d, graphs=False) is None


def test_the_request_view_weights_decode_by_its_steps():
    v = hr.request_view({"prefill": _phase({"_fwd_kernel.kd": (900.0, 1)}, steps=1),
                         "decode": _phase({"Cijk_x": (100.0, 1)}, steps=1)}, tokens=2)
    assert v["shares"][0] == ("attention (prefill)", 0.9)


def test_linear_ratio_is_over_the_steps_calls():
    rs = _routes(0.2, 0.1) + [{"name": "lm_head", "calls": 1, "engine_ms": 2.0, "path_ms": 2.0,
                               "path": "fallback", "m": 16}]
    r, why = hr.linear_ratio(rs)
    assert r == pytest.approx((36 * 0.2 + 2) / (36 * 0.1 + 2)) and "lm_head fallback@16" in why
    assert hr.linear_ratio(rs + [{"name": "o", "calls": 36, "engine_ms": None}]) is None


def test_the_cli_ranks_a_trace_from_elsewhere(tmp_path, capsys):
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"traceEvents": [_ev("Cijk_x", 0, 500),
                                             _ev("_fwd_kernel.kd", 600, 300)]}))
    out = tmp_path / "o.json"
    assert cli.main(["headroom", "--trace", f"prefill={t}", "--steps", "1", "--arch", "gfx1100",
                     "--model", str(tmp_path / "none"), "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert [e["family"] for e in res["ranking"]] == ["linear", "attention"]
    assert "upper bound" in capsys.readouterr().out
    # --from ranks a result again from what it kept (here: nothing raw).
    with pytest.raises(KeyError):
        hr.rerank(out)
