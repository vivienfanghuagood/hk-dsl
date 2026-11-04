"""tools/hk-bench/serving/sglang_toggle_ab.py and its in-server toggle, without SGLang or a GPU: the
toggle swaps hk's SGLang methods for SGLang's own and back on its signals and says so in its state
file; the driver refuses CUDA graphs and calls a ratio only when the arms' ranges are disjoint."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parents[3] / "tools" / "hk-bench" / "serving"


def _driver():
    spec = importlib.util.spec_from_file_location("sglang_toggle_ab", SERVING / "sglang_toggle_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_toggle_swaps_the_methods_and_reports_it(tmp_path):
    """A process with fake hk/SGLang modules: stock puts SGLang's methods back, hk re-applies hk's,
    and every switch is in the state file with the routes on."""
    script = textwrap.dedent("""
        import os, signal, sys, time, types
        class Linear:            # SGLang's UnquantizedLinearMethod / GemmaRMSNorm stand-ins
            def apply(self): return "sglang"
        class Norm:
            def forward_hip(self): return "sglang"
        def hk_apply(self): return "hk"
        def hk_norm(self): return "hk"
        S = types.ModuleType("hk.integration.sglang")
        def apply_linear():
            S._LINEAR = (Linear, Linear.apply); Linear.apply = hk_apply
        def apply_norm():
            S._NORM = (Norm, Norm.forward_hip); Norm.forward_hip = hk_norm
        S.apply_linear, S.apply_norm, S._LINEAR, S._NORM = apply_linear, apply_norm, None, None
        S.norm_stats = lambda: "{}"
        G = types.ModuleType("hk.integration.vllm_gemm")
        G.PATCH = types.SimpleNamespace(stats={"hk": 0}); G.set_routes = lambda t: None
        hk = types.ModuleType("hk"); I = types.ModuleType("hk.integration"); I.vllm_gemm = G; I.sglang = S
        sys.modules.update({"hk": hk, "hk.integration": I, "hk.integration.sglang": S,
                            "hk.integration.vllm_gemm": G})
        apply_linear(); apply_norm()                      # the plugin applied both at start
        d = os.environ["HK_SGLANG_TOGGLE_DIR"]; me = os.path.join(d, str(os.getpid()))
        def state(): return open(me).read().splitlines()[0]
        seen = [state()]
        os.kill(os.getpid(), signal.SIGRTMIN + 3); time.sleep(0.2)
        seen += [state(), Linear().apply(), Norm().forward_hip()]
        os.kill(os.getpid(), signal.SIGRTMIN + 4); time.sleep(0.2)
        seen += [state(), Linear().apply(), Norm().forward_hip()]
        print("|".join(seen))
    """)
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(tmp_path),
               PYTHONPATH=str(SERVING / "sglang_toggle"))
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split("|") == [
        "started", "stock on=-", "sglang", "sglang", "hk on=linear,norm", "hk", "hk"]


def test_a_process_without_hk_says_none(tmp_path):
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(tmp_path), PYTHONPATH=str(SERVING / "sglang_toggle"))
    script = ("import os, signal, time; os.kill(os.getpid(), signal.SIGRTMIN + 3); time.sleep(0.2); "
              "print(open(os.path.join(os.environ['HK_SGLANG_TOGGLE_DIR'], str(os.getpid()))).read().split()[0])")
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "none", r.stderr


def test_the_driver_refuses_cuda_graphs(tmp_path):
    with pytest.raises(SystemExit, match="disable-cuda-graph"):
        _driver().main(["--model", "m", "--out", str(tmp_path), "--", "--tp", "2"])


@pytest.mark.parametrize("stock, hk, verdict", [
    ([61.0, 60.8, 61.2], [55.4, 55.6, 55.3], "hk faster, disjoint"),
    ([61.0, 56.0, 61.2], [55.4, 57.0, 55.3], "ranges overlap: no result"),
    ([55.0, 55.2, 55.1], [58.0, 58.3, 58.1], "hk slower, disjoint"),
])
def test_a_ratio_is_a_result_only_when_the_ranges_are_disjoint(stock, hk, verdict):
    rows = ([{"spec": "1024/128/1/4", "arm": "stock", "median_tpot_ms": v} for v in stock]
            + [{"spec": "1024/128/1/4", "arm": "hk", "median_tpot_ms": v} for v in hk])
    for r in rows:
        r.update({k: None for k in ("median_ttft_ms", "output_throughput", "median_itl_ms")})
    (s,) = _driver().summarize(rows, ["1024/128/1/4"])
    assert s["metric"] == "median_tpot_ms" and s["verdict"] == verdict
