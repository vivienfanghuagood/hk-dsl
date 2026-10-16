"""HipKittens' CDNA4 kernels as hk ops: what each takes, and what it is built
from -- no GPU, no hipcc.

The shape rules are the kernels' own, read off their source: a partial tile
or an odd K step is a wrong answer or a write past the end of C, not a slow
one. They are pinned here so a refactor cannot loosen one silently.
"""

import pytest

from hk.integration import engine
from hk.ops import hipkittens as hkk


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
