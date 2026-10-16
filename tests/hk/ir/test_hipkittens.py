"""HipKittens' CDNA4 kernels as hk ops: what each takes, what it is built from,
and where the gfx950 routing table sends a linear layer -- no GPU, no hipcc.

The shape rules are the kernels' own, read off their source: a partial tile
or an odd K step is a wrong answer or a write past the end of C, not a slow
one. They are pinned here so a refactor cannot loosen one silently.
"""

import pytest

from hk.integration import engine, vllm_gemm
from hk.ops import hipkittens as hkk

QKV, O, GATE_UP, DOWN, LM_HEAD = ((6144, 4096), (4096, 4096), (24576, 4096),
                                  (4096, 12288), (151936, 4096))
LLAMA_GATE_UP, LLAMA_DOWN, LLAMA_LM_HEAD = (28672, 4096), (4096, 14336), (128256, 4096)


@pytest.mark.parametrize("m,n,k", [(256, 256, 128), (4096, 4096, 4096), (8192, 28672, 4096),
                                   (2048, 4096, 14336), (768, 4096, 8320)])
def test_the_gemm_takes_whole_tiles_and_even_k_steps(m, n, k):
    assert hkk.gemm_fits(m, n, k)


@pytest.mark.parametrize("m,n,k,why", [
    (1000, 4096, 4096, "multiples of 256"),     # a ragged M: the C store is unpredicated
    (4096, 151936, 4096, "multiples of 256"),   # Qwen3's LM head
    (4096, 4096, 4160, "multiple of 128"),      # 65 K steps: the loop takes two a trip
    (4096, 4096, 64, "multiple of 128"),
    (65536, 4096, 16384, "2**30"),              # M*K*2 bytes overflows the int descriptor size
    (0, 4096, 4096, "empty"),
])
def test_the_gemm_refuses_what_it_would_get_wrong(m, n, k, why):
    assert not hkk.gemm_fits(m, n, k)
    assert why in hkk.gemm_unsupported(m, n, k)


@pytest.mark.parametrize("b,n,h,h_kv,d", [(16, 1024, 64, 8, 128), (1, 4096, 32, 8, 128),
                                          (16, 8192, 64, 8, 128), (2, 256, 8, 8, 64)])
def test_gqa_takes_whole_query_blocks(b, n, h, h_kv, d):
    assert hkk.gqa_fits(b, n, h, h_kv, d)


@pytest.mark.parametrize("args,why", [
    ((1, 1000, 32, 8, 128), "multiple of 256"),
    ((1, 4096, 32, 8, 96), "head dims"),
    ((1, 4096, 32, 6, 128), "multiple of KV heads"),
    ((32, 8192, 64, 8, 128), "2**31"),
])
def test_gqa_refuses_what_it_would_get_wrong(args, why):
    assert not hkk.gqa_fits(*args)
    assert why in hkk.gqa_unsupported(*args)


def test_gqa_has_no_kv_cache():
    assert "no KV cache" in hkk.gqa_unsupported(1, 1024, 32, 8, 128, n_kv=2048)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942"])
def test_another_arch_is_refused_loudly(arch):
    with pytest.raises(hkk.WrongArch, match="gfx950"):
        hkk.gemm_build(arch=arch)
    with pytest.raises(hkk.WrongArch, match="gfx950"):
        hkk.gqa_build(32, 8, 1024, arch=arch)


def test_the_gemm_source_is_upstream_verbatim_plus_hks_entry_point():
    upstream = (hkk.kernels_dir() / hkk.GEMM_SRC).read_text()
    src = hkk.gemm_source()
    assert upstream in src
    # Upstream's own module is renamed out of the way, before its text.
    assert src.index("#define tk_kernel hk_hipkittens_upstream_module") < src.index(upstream)
    assert "PYBIND11_MODULE(HK_MODULE_NAME, mod)" in src
    # Launched on the caller's stream, not upstream's null stream.
    assert "reinterpret_cast<hipStream_t>(stream)" in src
    assert "micro_tk<<<g.grid(), g.block(), lds, g.stream>>>" in src


@pytest.mark.parametrize("causal,d,rel", [(False, 128, "attn/gqa/kernel.cpp"),
                                          (True, 128, "attn/gqa_causal/kernel.cpp"),
                                          (False, 64, "attn/gqa/kernel_d64.cpp"),
                                          (True, 64, "attn/gqa_causal/kernel_d64.cpp")])
def test_one_gqa_source_per_shape_and_none_per_batch(causal, d, rel):
    src = hkk.gqa_source(64, 8, 2048, d, causal)
    assert (hkk.kernels_dir() / rel).read_text() in src
    for define in ("#define ATTN_H 64", "#define ATTN_H_KV 8", "#define ATTN_N 2048",
                   f"#define ATTN_D {d}"):
        assert define in src
    # The batch is the launch's, so every batch size shares the build.
    assert "#define ATTN_B" not in src
    assert "dim3 grid(ATTN_H, (ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS, batch)" in src


def test_a_gqa_source_for_a_shape_it_cannot_take_is_refused():
    with pytest.raises(hkk.Unsupported, match="multiple of 256"):
        hkk.gqa_source(32, 8, 1000)


def test_the_ops_are_graph_safe_with_a_fallback():
    for op, name in ((hkk.GEMM_OP, "hk_cdna::gemm_bf16"), (hkk.GQA_OP, "hk_cdna::gqa_fwd")):
        assert op.name == name and op.graph == "safe"
        assert op.paths == ("hipkittens", "fallback")
        assert engine.PATCHES[name] is op


# ---------------------------------------------------------------- the gfx950 table


@pytest.fixture
def gfx950():
    t = engine.shipped_table("vllm_linear", "gfx950", "HK_TEST_NO_SUCH_ENV")
    vllm_gemm.set_routes(t)
    yield t
    vllm_gemm.set_routes(None)


def test_the_gfx950_table_is_measured_on_gfx950_against_f_linear(gfx950):
    assert gfx950 is not None and gfx950.meta["arch"] == "gfx950"
    assert gfx950.meta["engine"] == "F.linear"
    assert gfx950.paths() <= set(vllm_gemm.PATCH.paths)
    assert gfx950.paths() <= {"hipkittens", "fallback"}
    for nk in (QKV, O, GATE_UP, DOWN, LM_HEAD, LLAMA_GATE_UP, LLAMA_DOWN, LLAMA_LM_HEAD):
        assert nk in set(gfx950.shapes())


def test_the_gfx1100_table_is_not_loaded_as_gfx950(tmp_path, monkeypatch):
    from pathlib import Path

    tuned = Path(vllm_gemm.__file__).resolve().parents[1] / "tuned"
    p = tmp_path / "routes.json"
    p.write_text((tuned / "routes_vllm_linear_gfx1100.json").read_text())
    monkeypatch.setenv("HK_TEST_GFX950_ROUTES", str(p))
    with pytest.raises(ValueError, match="measured on gfx1100, not gfx950"):
        engine.shipped_table("vllm_linear", "gfx950", "HK_TEST_GFX950_ROUTES")


def test_small_batches_stay_on_hipblaslt(gfx950):
    # 256x256 tiles over 256 CUs: at M <= 1024 the GEMM ran at 0.2-0.7x hipBLASLt.
    for nk in (QKV, O, GATE_UP, DOWN):
        for m in (1, 16, 64, 128, 256, 512, 1024):
            assert vllm_gemm.route_for(m, *nk) == "fallback"


def test_a_ragged_m_inside_a_hipkittens_run_falls_back(gfx950):
    # o_proj at M=2048 measured 1.27x hipBLASLt; 2047 and 2049 are not whole tiles.
    assert vllm_gemm.route_for(2048, *O) == "hipkittens"
    assert vllm_gemm.route_for(2047, *O) == "fallback"
    assert vllm_gemm.route_for(2049, *O) == "fallback"


def test_the_wins_are_teeth_not_a_threshold(gfx950):
    # hipBLASLt's pick is the comb: HipKittens wins at Ms where it picks a slow
    # kernel and loses 5-10% at the powers of two either side.
    r = [vllm_gemm.route_for(m, *O) for m in (1792, 2048, 2304, 3840, 4096)]
    assert r == ["hipkittens", "hipkittens", "fallback", "hipkittens", "fallback"]
    for nk in (QKV, O, GATE_UP, DOWN, LLAMA_GATE_UP, LLAMA_DOWN):
        assert vllm_gemm.route_for(8192, *nk) == "fallback"
        assert vllm_gemm.route_for(7680, *nk) == "hipkittens"


def test_lm_heads_never_take_the_hipkittens_gemm(gfx950):
    for m in range(1, 300, 7):
        assert vllm_gemm.route_for(m, *LM_HEAD) == "fallback"
        assert vllm_gemm.route_for(m, *LLAMA_LM_HEAD) == "fallback"


def test_a_table_cannot_send_a_shape_to_hipkittens_that_it_refuses():
    vllm_gemm.set_routes(engine.RouteTable({(6100, 4096): {"runs": [[1, 99999, "hipkittens"]]},
                                            (6144, 4160): {"runs": [[1, 99999, "hipkittens"]]}}))
    try:
        assert vllm_gemm.route_for(4096, 6100, 4096) == "fallback"
        assert vllm_gemm.route_for(4096, 6144, 4160) == "fallback"
        assert vllm_gemm.route_for(4096, 6100, 4096, bias=True) == "fallback"
    finally:
        vllm_gemm.set_routes(None)
