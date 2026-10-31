"""Every route that is on by default has a GPU test on every arch it is on by default on, and CI
runs that test there: a route a user gets without asking is the last place a regression should
pass silently. A new default-on (route, arch) fails here until it has one."""

from __future__ import annotations

import re
from pathlib import Path

from hk.integration import switches

REPO = Path(__file__).resolve().parents[3]
GPU = REPO / "tests" / "hk" / "gpu"

#: The test files that need vLLM (and AITER) run in the gfx950-vllm job only.
VLLM_JOB = "gfx950-vllm"

#: (route, arch) -> "file::test" in tests/hk/gpu: the GPU test of that route on that arch.
COVERAGE = {
    ("vllm_mxfp8", "gfx950"): "test_vllm_mxfp8.py::test_patched_layer_matches_vllm",
    ("vllm_moe_a8w4", "gfx950"): "test_vllm_moe_a8w4.py::test_patched_fused_moe_routes_and_agrees",
    ("vllm_dsv41_index", "gfx950"): "test_vllm_dsv41_index.py::test_the_launch_is_vllms_chain",
    ("vllm_sparse_mla", "gfx950"):
        "test_vllm_sparse_mla.py::test_the_route_takes_a_large_call_and_matches_opus",
    ("vllm_int4", "gfx1100"):
        "test_route_paths.py::test_vllm_int4_takes_up_to_this_archs_limit_and_leaves_the_rest",
    ("vllm_int4", "gfx1201"):
        "test_route_paths.py::test_vllm_int4_takes_up_to_this_archs_limit_and_leaves_the_rest",
    ("vllm_int4", "gfx950"):
        "test_route_paths.py::test_vllm_int4_takes_up_to_this_archs_limit_and_leaves_the_rest",
    ("vllm_gptq", "gfx1100"):
        "test_route_paths.py::test_vllm_gptq_takes_min_m_to_64_and_leaves_the_rest",
    ("vllm_gemm", "gfx1100"): "test_vllm_gemm.py::test_the_patched_layer_function_matches_f_linear",
    ("vllm_gemm", "gfx1201"): "test_vllm_gemm.py::test_the_patched_layer_function_matches_f_linear",
    ("vllm_moe", "gfx1100"):
        "test_route_paths.py::test_vllm_moe_takes_a_decode_batch_and_writes_c_in_place",
    ("vllm_moe", "gfx1201"):
        "test_route_paths.py::test_vllm_moe_takes_a_decode_batch_and_writes_c_in_place",
    ("vllm_sdpa", "gfx1100"): "test_sdpa.py::test_patch_and_unpatch_round_trip",
    ("sglang_linear", "gfx1100"):
        "test_route_paths.py::test_sglang_linear_routes_its_apply_through_hk_and_reverts",
    ("sglang_moe", "gfx1100"):
        "test_route_paths.py::test_sglang_moe_takes_a_decode_batch_and_its_top_k_matches_torch",
    # the route is hk.ops.sdpa.patch(), as vllm_sdpa's
    ("sgld_attn", "gfx1100"): "test_sdpa.py::test_patch_and_unpatch_round_trip",
    ("sgld_linear", "gfx1100"):
        "test_route_paths.py::test_sgld_linear_takes_a_dit_projection_through_hk",
    ("sdpa_k3", "gfx1100"): "test_attn_k3.py::test_sdpa_routes_non_causal_d128_to_k3_when_enabled",
}


def _default_on():
    return {(r.name, a) for r in switches.ROUTES.values() if r.default for a in r.arches}


def _vllm_job_tests():
    text = (REPO / ".github" / "workflows" / "gpu.yml").read_text(encoding="utf-8")
    job = text.split(f"  {VLLM_JOB}:", 1)[1]
    block = job.split("HK_CI_TESTS:", 1)[1].split("steps:", 1)[0]
    return set(re.findall(r"tests/hk/gpu/(test_\w+\.py)", block))


def test_every_default_on_route_has_a_gpu_test_on_every_arch_it_is_on():
    missing = sorted(_default_on() - set(COVERAGE))
    assert not missing, f"default-on with no GPU test: {missing} (add one, then a COVERAGE row)"
    stale = sorted(set(COVERAGE) - _default_on())
    assert not stale, f"COVERAGE rows for (route, arch) pairs no longer on by default: {stale}"


def test_each_named_test_exists_and_ci_runs_it_on_that_arch():
    vllm_job = _vllm_job_tests()
    for (route, arch), ref in COVERAGE.items():
        file, _, name = ref.partition("::")
        path = GPU / file
        assert path.is_file(), f"{route} on {arch}: no {path}"
        text = path.read_text(encoding="utf-8")
        if name:
            assert re.search(rf"^def {name}\(", text, re.M), f"{route} on {arch}: no {ref}"
        if re.search(r'importorskip\("(vllm|aiter)', text):
            # skips in every other job: the vLLM image is gfx950's
            assert arch == "gfx950" and file in vllm_job, \
                f"{route} on {arch}: {file} needs vLLM, and gpu.yml's {VLLM_JOB} does not run it"
