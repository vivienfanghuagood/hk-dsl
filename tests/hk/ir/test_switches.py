"""hk is on by default: which routes are on, the switches that turn them off, the engine contract
that leaves an engine stock when it is broken, and the JIT notice. No GPU, no engine: the
plugins' route applies and the contract checks are stand-ins, except where a stand-in aiter
breaks the real check (`test_a_real_signature_mismatch_...`)."""

from __future__ import annotations

import importlib.machinery
import subprocess
import sys
import types

import pytest

from hk.integration import _sglang_plugin, _vllm_plugin, contract, switches
from hk.runtime import compile as C

SWITCHES = ["HK_DISABLE", *[e for e in switches.ENGINES.values() if e],
            *[r.env for r in switches.ROUTES.values()], "HK_VLLM_USER", "HK_STACK_CHECK"]


@pytest.fixture
def clean(monkeypatch):
    for e in SWITCHES:
        monkeypatch.delenv(e, raising=False)
    return monkeypatch


def on(env=None, gpu="gfx950", engine=None):
    env = {} if env is None else env
    return {n for n, r in switches.ROUTES.items()
            if (engine is None or r.engine == engine) and switches.decide(n, env, gpu).on}


# -- the defaults ------------------------------------------------------------------------------


def test_on_by_default_on_gfx950_is_k1_k2_and_int4():
    assert on() == {"vllm_mxfp8", "vllm_moe_a8w4", "vllm_int4"}


def test_on_by_default_on_gfx1100_is_every_route_validated_there():
    assert on(gpu="gfx1100") == {"vllm_int4", "vllm_gemm", "vllm_moe", "vllm_sdpa",
                                 "sglang_linear", "sglang_moe", "sgld_attn", "sgld_linear",
                                 "sdpa_k3"}


@pytest.mark.parametrize("gpu", ["gfx942", "gfx1201", None])
def test_nothing_is_on_by_default_where_nothing_was_validated(gpu):
    assert on(gpu=gpu) == set()
    assert "validated on" in switches.decide("vllm_mxfp8", {}, gpu).why


@pytest.mark.parametrize("gpu", ["gfx950", "gfx1100"])
def test_lossy_and_unvalidated_routes_stay_off_by_default(gpu):
    for name in ("sgld_convrot", "vllm_backend"):
        d = switches.decide(name, {}, gpu)
        assert not d.on and d.why == "default off"


def test_every_default_on_route_names_its_evidence_and_an_arch():
    for r in switches.ROUTES.values():
        assert r.evidence and r.env.startswith("HK_")
        assert bool(r.arches) == r.default, r.name


# -- HK_DISABLE, HK_<ROUTE>=0, and the precedence ----------------------------------------------


EVERYTHING_ON = {**{r.env: "1" for r in switches.ROUTES.values()}, "HK_PATCH_VLLM": "1",
                 "HK_SGLD": "attn,linear,convrot", "HK_PATCH_SGLANG": "1"}


@pytest.mark.parametrize("v", ["1", "yes", "on", "true", "anything"])
def test_hk_disable_beats_everything(v):
    env = dict(EVERYTHING_ON, HK_DISABLE=v)
    for gpu in ("gfx950", "gfx1100"):
        assert on(env, gpu) == set()
    assert switches.decide("vllm_mxfp8", env, "gfx950").why == f"HK_DISABLE={v}"
    assert switches.disabled(env)


@pytest.mark.parametrize("v", ["0", "", "off", "no", "false"])
def test_hk_disable_off_or_empty_is_not_disable(v):
    assert not switches.disabled({"HK_DISABLE": v})
    assert on({"HK_DISABLE": v}) == on()


@pytest.mark.parametrize("name", sorted(switches.ROUTES))
def test_a_route_switch_at_0_turns_off_that_route_only(name):
    r = switches.ROUTES[name]
    gpu = r.arches[0] if r.arches else "gfx1100"
    for v in ("0", "off", "no", "FALSE"):
        got = on({r.env: v}, gpu)
        assert name not in got
        assert got == on(gpu=gpu) - {name}
        assert switches.decide(name, {r.env: v}, gpu).why == f"{r.env}={v}"


@pytest.mark.parametrize("engine,env", [("vllm", "HK_PATCH_VLLM"), ("sglang", "HK_PATCH_SGLANG"),
                                        ("sglang_diffusion", "HK_SGLD")])
def test_an_engine_switch_at_0_beats_its_routes_explicit_values(engine, env):
    every = {r.env: "1" for r in switches.routes(engine)}
    assert on(dict(every, **{env: "0"}), "gfx1100", engine) == set()
    assert on(dict(every, **{env: "0"}), "gfx950", engine) == set()
    others = on(dict(every, **{env: "0"}), "gfx1100") - {r.name for r in switches.routes(engine)}
    assert others == on(gpu="gfx1100") - {r.name for r in switches.routes(engine)}


def test_an_explicit_route_value_beats_the_default_and_the_arch():
    # HK_VLLM_MXFP8=1 on a gfx1100: asked for, so on (its apply then says it needs gfx950).
    assert switches.decide("vllm_mxfp8", {"HK_VLLM_MXFP8": "1"}, "gfx1100").on
    assert switches.decide("sgld_convrot", {"HK_SGLD_CONVROT": "1"}, "gfx1100").on
    assert switches.decide("vllm_backend", {"HK_VLLM_BACKEND": "1"}, None).on


def test_an_empty_value_is_unset():
    assert on({"HK_VLLM_MXFP8": "", "HK_PATCH_VLLM": "", "HK_SGLD": ""}) == on()


# -- the legacy HK_*=1 switches ----------------------------------------------------------------


def test_legacy_opt_ins_still_turn_their_routes_on():
    env = {"HK_PATCH_VLLM": "1", "HK_VLLM_MOE_A8W4": "1", "HK_VLLM_MXFP8": "1"}
    assert {"vllm_mxfp8", "vllm_moe_a8w4"} <= on(env, "gfx950")
    # HK_PATCH_VLLM=1 always applied the encoder SDPA patch, on any GPU.
    assert "vllm_sdpa" in on({"HK_PATCH_VLLM": "1"}, "gfx950")
    assert switches.value("vllm_backend", {"HK_VLLM_BACKEND": "FLEX_ATTENTION"}) == "FLEX_ATTENTION"
    assert switches.value("vllm_backend", {"HK_VLLM_BACKEND": "1"}) == ""
    assert switches.value("sglang_moe", {"HK_SGLANG_MOE": "topk"}) == "topk"


@pytest.mark.parametrize("v,want", [("1", {"sgld_attn", "sgld_linear"}),
                                    ("all", {"sgld_attn", "sgld_linear"}),
                                    ("attn", {"sgld_attn"}),
                                    ("linear", {"sgld_linear"}),
                                    ("attn,linear,convrot",
                                     {"sgld_attn", "sgld_linear", "sgld_convrot"}),
                                    ("0", set())])
def test_hk_sgld_lists_name_exactly_their_parts_on_any_gpu(v, want):
    for gpu in ("gfx1100", "gfx950"):
        assert on({"HK_SGLD": v}, gpu, "sglang_diffusion") == want


def test_a_part_switch_beats_the_hk_sgld_list():
    assert on({"HK_SGLD": "1", "HK_SGLD_LINEAR": "0"}, "gfx1100",
              "sglang_diffusion") == {"sgld_attn"}


# -- the vLLM plugin -------------------------------------------------------------------------


@pytest.fixture
def plugin(clean):
    """The vLLM plugin with stand-in routes: (applied, reverted) lists."""
    applied, reverted = [], []

    def fake(name):
        def apply():
            applied.append(name)
            return f"{name}: applied", lambda: reverted.append(name)
        return apply

    clean.setattr(_vllm_plugin, "APPLY", {n: fake(n) for n in _vllm_plugin.APPLY})
    clean.setattr(contract, "check", lambda routes: [])
    from hk.integration import stack

    clean.setattr(stack, "enforce", lambda *a, **k: "stack: test")
    clean.setattr(switches, "arch", lambda: "gfx950")
    return applied, reverted


def test_the_vllm_plugin_applies_the_defaults_with_nothing_set(plugin, capsys):
    applied, _ = plugin
    _vllm_plugin.register()
    assert applied == ["vllm_int4", "vllm_moe_a8w4", "vllm_mxfp8"]
    out = capsys.readouterr().out
    assert "[hk] vLLM routes on: " in out and "default on (gfx950)" in out


def test_the_vllm_plugin_with_hk_disable_applies_nothing(plugin, monkeypatch, capsys):
    applied, _ = plugin
    for e, v in dict(EVERYTHING_ON, HK_DISABLE="1", HK_VLLM_USER="some_module").items():
        monkeypatch.setenv(e, v)
    _vllm_plugin.register()
    assert applied == []
    assert "off (HK_DISABLE=1), stock" in capsys.readouterr().out


def test_the_vllm_plugin_leaves_out_a_route_at_0(plugin, monkeypatch):
    applied, _ = plugin
    monkeypatch.setenv("HK_VLLM_MOE_A8W4", "0")
    _vllm_plugin.register()
    assert applied == ["vllm_int4", "vllm_mxfp8"]


def test_the_vllm_plugin_with_hk_patch_vllm_0_applies_nothing(plugin, monkeypatch):
    applied, _ = plugin
    monkeypatch.setenv("HK_PATCH_VLLM", "0")
    monkeypatch.setenv("HK_VLLM_MXFP8", "1")
    monkeypatch.setenv("HK_VLLM_USER", "some_module")
    _vllm_plugin.register()
    assert applied == []


def test_a_broken_contract_leaves_vllm_stock_with_one_warning(plugin, capsys, monkeypatch):
    applied, _ = plugin
    seen = []

    def check(routes):
        seen.append(sorted(routes))
        return ["vllm_moe_a8w4: aiter.fused_moe.fused_moe lost ['gate_mode']"]

    monkeypatch.setattr(contract, "check", check)
    _vllm_plugin.register()
    assert applied == []                         # all or nothing: K2 is not applied either
    assert seen == [["vllm_int4", "vllm_moe_a8w4", "vllm_mxfp8"]]
    cap = capsys.readouterr()
    lines = [x for x in cap.err.splitlines() if x.strip()]
    assert len(lines) == 1 and lines[0].startswith("[hk] WARNING: vllm: engine contract FAILED")
    assert "patched NOTHING" in lines[0] and "gate_mode" in lines[0]
    assert "routes on" not in cap.out


def test_a_route_that_raises_reverts_the_ones_before_it(plugin, capsys, monkeypatch):
    applied, reverted = plugin

    def boom():
        raise RuntimeError("moved")

    monkeypatch.setitem(_vllm_plugin.APPLY, "vllm_mxfp8", boom)
    _vllm_plugin.register()
    assert applied == ["vllm_int4", "vllm_moe_a8w4"]
    assert reverted == ["vllm_moe_a8w4", "vllm_int4"]          # undone, last first
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "a route failed to apply" in err[0]
    assert "vllm_mxfp8 raised RuntimeError: moved" in err[0]


def _fake_aiter(monkeypatch, with_gate_mode: bool):
    """aiter's fused_moe as a module tree, its signature with or without gate_mode."""
    from hk.integration import vllm_moe_a8w4 as PM

    params = [p for p in PM.AITER_PARAMS if with_gate_mode or p != "gate_mode"]
    src = (f"def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, "
           f"{', '.join(p + '=None' for p in params)}): pass\n"
           f"def moe_sorting(topk_ids, topk_weights, e, d, dtype, block_size=32): pass\n")
    mods = {}
    for name in ("aiter", "aiter.fused_moe", "aiter.ops", "aiter.ops.quant"):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mods[name] = m
        monkeypatch.setitem(sys.modules, name, m)
    exec(src, mods["aiter.fused_moe"].__dict__)  # noqa: S102
    exec("def fused_dynamic_mxfp8_quant_moe_sort(x, sid, nv, m, topk, bm): pass\n",  # noqa: S102
         mods["aiter.ops.quant"].__dict__)
    mods["aiter"].fused_moe = mods["aiter.fused_moe"]
    mods["aiter"].ops = mods["aiter.ops"]
    mods["aiter.ops"].quant = mods["aiter.ops.quant"]
    return mods["aiter.fused_moe"]


def test_a_real_signature_mismatch_is_found_by_the_real_check(clean):
    clean.setattr(switches, "arch", lambda: "gfx950")
    _fake_aiter(clean, with_gate_mode=False)
    problems = contract.check(["vllm_moe_a8w4"])
    assert problems == ["vllm_moe_a8w4: aiter.fused_moe.fused_moe lost ['gate_mode']"]
    _fake_aiter(clean, with_gate_mode=True)
    assert contract.check(["vllm_moe_a8w4"]) == []


def test_a_real_mismatch_leaves_aiter_unwrapped(clean, capsys):
    clean.setattr(switches, "arch", lambda: "gfx950")
    afm = _fake_aiter(clean, with_gate_mode=False)
    orig = afm.fused_moe
    from hk.integration import stack

    clean.setattr(stack, "enforce", lambda *a, **k: "stack: test")
    clean.setenv("HK_VLLM_MXFP8", "0")            # vLLM is not importable here: K1 alone
    clean.setenv("HK_VLLM_INT4", "0")
    _vllm_plugin.register()
    assert afm.fused_moe is orig
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "lost ['gate_mode']" in err[0]


def test_the_contract_check_reports_a_missing_engine_as_broken(clean):
    clean.setattr(switches, "arch", lambda: "gfx1100")
    clean.setitem(sys.modules, "vllm", None)      # import vllm raises
    problems = contract.check(["vllm_gemm"])
    assert len(problems) == 1 and problems[0].startswith("vllm_gemm: ")


def test_gfx950_only_contracts_do_not_apply_elsewhere(clean):
    clean.setattr(switches, "arch", lambda: "gfx1100")
    clean.setitem(sys.modules, "vllm", None)
    assert contract.check(["vllm_mxfp8", "vllm_moe_a8w4"]) == []


# -- the SGLang plugin -------------------------------------------------------------------------


def test_the_sglang_plugin_is_on_by_default_and_off_with_hk_disable(clean, capsys):
    from hk.integration import sglang as hks
    from hk.integration import sglang_moe

    seen = []
    clean.setattr(switches, "arch", lambda: "gfx1100")
    clean.setattr(contract, "check", lambda routes: [])
    clean.setattr(hks, "apply_linear", lambda: seen.append("linear") or "linear: test")
    clean.setattr(sglang_moe, "apply", lambda: seen.append("moe") or "moe: test")
    _sglang_plugin.register()
    assert seen == ["linear", "moe"]
    seen.clear()
    clean.setenv("HK_DISABLE", "1")
    _sglang_plugin.register()
    assert seen == []
    assert "SGLang: off (HK_DISABLE=1)" in capsys.readouterr().out


# -- the JIT notice ----------------------------------------------------------------------------


@pytest.fixture
def jit(monkeypatch, tmp_path):
    """compile.build with hipcc a stand-in that fails: what is printed before it runs."""
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(C, "_JIT_NOTICE", [])
    monkeypatch.setattr(C, "_JIT_SAID", [])
    monkeypatch.setattr(C, "hipcc", lambda: "/bin/false")
    monkeypatch.setattr(C, "_toolchain_cache", {})
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="hipcc stand-in", stderr=""))

    def build(i):
        with pytest.raises(C.CompileError):
            C.build(f"// kernel {i}\n", "gfx1100", name=f"k{i}")
    return build


def test_the_first_jit_compile_after_a_plugin_applied_routes_says_so_once(jit, capsys):
    C.arm_jit_notice("vLLM plugin")
    jit(1)
    jit(2)
    err = [x for x in capsys.readouterr().err.splitlines() if x.startswith("[hk] JIT")]
    assert len(err) == 1
    assert "compiling kernel 'k1'" in err[0] and "not pre-warmed" in err[0]
    assert "python3 -m hk.integration warm-routes" in err[0]


def test_no_jit_notice_unless_a_plugin_armed_it(jit, capsys):
    jit(3)
    assert "[hk] JIT" not in capsys.readouterr().err


def test_the_plugin_arms_the_jit_notice(plugin, monkeypatch):
    monkeypatch.setattr(C, "_JIT_NOTICE", [])
    _vllm_plugin.register()
    assert C._JIT_NOTICE == ["vLLM plugin"]
