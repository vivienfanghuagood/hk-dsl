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


def test_stock_reaches_layers_that_cached_the_bound_forward(tmp_path):
    """SGLang's CustomOp caches a layer's forward_hip as a bound method at construction, so
    restoring the class attribute does not reach a layer already built: the toggle pauses the
    norm route in place (`pause_norm`), and stock is stock on such a layer too."""
    script = textwrap.dedent("""
        import os, signal, sys, time, types
        class Norm:
            def forward_hip(self): return "sglang"
        class Layer:             # what SGLang's CustomOp does at construction
            def __init__(self): self._forward_method = Norm.forward_hip.__get__(self)
            def __call__(self): return self._forward_method()
        S = types.ModuleType("hk.integration.sglang")
        S._LINEAR, S._NORM, S._NORM_PAUSED = None, None, False
        def pause_norm(p): S._NORM_PAUSED = bool(p)
        def apply_norm():
            orig = Norm.forward_hip
            def hk_norm(self):
                return orig(self) if S._NORM_PAUSED else "hk"
            S._NORM = (Norm, orig); Norm.forward_hip = hk_norm
        S.apply_linear, S.apply_norm, S.pause_norm = (lambda: None), apply_norm, pause_norm
        S.norm_stats = lambda: "{}"
        hk = types.ModuleType("hk"); I = types.ModuleType("hk.integration"); I.sglang = S
        sys.modules.update({"hk": hk, "hk.integration": I, "hk.integration.sglang": S})
        apply_norm()
        layer = Layer()                                   # built with hk's forward bound
        d = os.environ["HK_SGLANG_TOGGLE_DIR"]; me = os.path.join(d, str(os.getpid()))
        def state(): return open(me).read().splitlines()[0]
        seen = [layer()]
        os.kill(os.getpid(), signal.SIGRTMIN + 3); time.sleep(0.2)
        seen += [state(), layer()]
        os.kill(os.getpid(), signal.SIGRTMIN + 4); time.sleep(0.2)
        seen += [state(), layer()]
        print("|".join(seen))
    """)
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(tmp_path),
               PYTHONPATH=str(SERVING / "sglang_toggle"))
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split("|") == ["hk", "stock on=-", "sglang", "hk on=norm", "hk"]


def test_the_toggle_switches_sglang_diffusions_routes(tmp_path):
    """SGLang-Diffusion: stock unpatches hk's SDPA and reverts every applied sgld_* route (and only
    those); hk applies them again; the counters are SGLang-Diffusion's stats()."""
    script = textwrap.dedent("""
        import os, signal, sys, time, types
        P = types.ModuleType("hk.ops.sdpa"); P.on = True
        P.patched = lambda: P.on
        def _patch(): P.on = True
        def _unpatch(): P.on = False
        P.patch, P.unpatch = _patch, _unpatch
        D = types.ModuleType("hk.integration.sglang_diffusion")
        D._LINEAR_ORIG, D._VAE_ORIG, D._ROPE_ORIG = [1], [1], []      # linear and vaeconv applied
        calls = []
        for n, o in (("linear", "_LINEAR_ORIG"), ("vaeconv", "_VAE_ORIG"), ("rope", "_ROPE_ORIG")):
            def ap(o=o, n=n): setattr(D, o, [1]); calls.append("apply_" + n)
            def rv(o=o, n=n): setattr(D, o, []); calls.append("revert_" + n)
            setattr(D, "apply_" + n, ap); setattr(D, "revert_" + n, rv)
        D.stats = lambda: "attention: hk 3"
        hk = types.ModuleType("hk"); I = types.ModuleType("hk.integration"); O = types.ModuleType("hk.ops")
        I.sglang_diffusion = D; O.sdpa = P
        sys.modules.update({"hk": hk, "hk.integration": I, "hk.ops": O, "hk.ops.sdpa": P,
                            "hk.integration.sglang_diffusion": D})
        d = os.environ["HK_SGLANG_TOGGLE_DIR"]; me = os.path.join(d, str(os.getpid()))
        def state(): return open(me).read().splitlines()
        os.kill(os.getpid(), signal.SIGRTMIN + 3); time.sleep(0.2)
        seen = [state()[0], str(P.on), ",".join(calls)]
        calls.clear()
        os.kill(os.getpid(), signal.SIGRTMIN + 4); time.sleep(0.2)
        seen += [state()[0], str(P.on), ",".join(calls), state()[1]]
        print("|".join(seen))
    """)
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(tmp_path),
               PYTHONPATH=str(SERVING / "sglang_toggle"))
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split("|") == [
        "stock on=-", "False", "revert_linear,revert_vaeconv",
        "hk on=attn,linear,vaeconv", "True", "apply_linear,apply_vaeconv", "sgld attention: hk 3"]


def test_the_stock_arm_can_run_without_cudnn(tmp_path):
    """HK_TOGGLE_STOCK_NO_CUDNN=1: stock turns torch's cudnn (MIOpen) off, hk turns it back on."""
    script = textwrap.dedent("""
        import os, signal, sys, time, types
        torch = types.ModuleType("torch"); torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True))
        sys.modules["torch"] = torch
        P = types.ModuleType("hk.ops.sdpa"); P.patched = lambda: False; P.patch = P.unpatch = lambda: None
        D = types.ModuleType("hk.integration.sglang_diffusion"); D.stats = lambda: "-"
        hk = types.ModuleType("hk"); I = types.ModuleType("hk.integration"); O = types.ModuleType("hk.ops")
        I.sglang_diffusion = D; O.sdpa = P
        sys.modules.update({"hk": hk, "hk.integration": I, "hk.ops": O, "hk.ops.sdpa": P,
                            "hk.integration.sglang_diffusion": D})
        me = os.path.join(os.environ["HK_SGLANG_TOGGLE_DIR"], str(os.getpid()))
        os.kill(os.getpid(), signal.SIGRTMIN + 3); time.sleep(0.2)
        seen = [open(me).read().splitlines()[0], str(torch.backends.cudnn.enabled)]
        os.kill(os.getpid(), signal.SIGRTMIN + 4); time.sleep(0.2)
        seen += [open(me).read().splitlines()[0], str(torch.backends.cudnn.enabled)]
        print("|".join(seen))
    """)
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(tmp_path), HK_TOGGLE_STOCK_NO_CUDNN="1",
               PYTHONPATH=str(SERVING / "sglang_toggle"))
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().split("|") == ["stock on=- cudnn=off", "False", "hk on=- cudnn=on", "True"]


def test_a_diffusion_run_reports_seconds_lower_better():
    rows = ([{"spec": "video", "arm": "stock", "seconds": v} for v in (200.0, 201.0, 199.5)]
            + [{"spec": "video", "arm": "hk", "seconds": v} for v in (190.0, 191.0, 190.5)])
    d = _driver()
    (s,) = d.summarize(rows, ["video"], d.DIFF_METRICS)
    assert s["metric"] == "seconds" and s["verdict"] == "hk faster, disjoint"
    assert s["ratio_hk_over_stock"] > 1


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
