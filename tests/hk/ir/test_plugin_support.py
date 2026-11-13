"""The vLLM plugin's support check and startup summary (owner decision D-20261113-07).

An unsupported stack: one line saying what is unsupported and what is, hk fully off, nothing of
hk's routes or contracts loaded. A supported one: a summary with hk's version, the routes on, and
how to turn all of hk or one route off.
"""
import pytest

from hk.integration import _vllm_plugin, contract, support, switches


def _versions(monkeypatch, **v):
    names = {"vllm": "vllm", "torch": "torch"}
    monkeypatch.setattr(support, "_version", lambda d: v.get(names.get(d, d)))
    monkeypatch.setattr(support, "rocm_version", lambda: v.get("rocm"))


@pytest.mark.parametrize("vllm,torch,rocm,arch,bad", [
    ("0.33.0+rocm72", "2.13.0", "7.2.3", "gfx1100", []),
    ("0.1.dev1+g3775d5fca.rocm714", "2.13.0a0", "7.14.0", "gfx1100", []),   # source build
    ("0.16.1.dev0+g89a77b108.rocm721", "2.9.1+gitff65f5b", "7.2.1", "gfx1100", ["vllm 0.16.1"]),
    ("0.32.1rc1.dev36", "2.8.0", "7.2.3", "gfx950", ["torch 2.8.0"]),
    ("0.32.1", "2.13.0", "7.1.0", "gfx1201", ["ROCm 7.1.0"]),
    ("0.32.1", "2.13.0", "7.2.3", "gfx942", ["GPU gfx942"]),
])
def test_the_support_check(monkeypatch, vllm, torch, rocm, arch, bad):
    _versions(monkeypatch, vllm=vllm, torch=torch, rocm=rocm)
    probs, seen = support.check("vllm", arch)
    assert len(probs) == len(bad) and all(p.startswith(b) for p, b in zip(probs, bad)), probs
    assert all("supported:" in p for p in probs) and seen["arch"] == arch


def test_an_unsupported_vllm_leaves_hk_fully_off_in_one_line(monkeypatch, capsys):
    for e in ("HK_DISABLE", "HK_PATCH_VLLM"):
        monkeypatch.delenv(e, raising=False)
    _versions(monkeypatch, vllm="0.16.1.dev0+g89a77b108.d20260317.rocm721", torch="2.9.1", rocm="7.2.1")
    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")

    def boom(*a, **k):
        raise AssertionError("routes or contracts touched on an unsupported stack")
    monkeypatch.setattr(contract, "apply_routes", boom)
    monkeypatch.setattr(contract, "check_each", boom)
    _vllm_plugin.register()
    cap = capsys.readouterr()
    lines = [x for x in (cap.out + cap.err).splitlines() if x.strip()]
    assert len(lines) == 1, lines
    assert lines[0].startswith("[hk] vLLM: hk is OFF, this stack is not supported -- vllm 0.16.1")
    assert "supported: vllm >= 0.19" in lines[0] and "runs as stock" in lines[0]


def test_the_startup_summary_says_what_runs_and_how_to_turn_it_off(monkeypatch, capsys):
    for e in ("HK_DISABLE", "HK_PATCH_VLLM", "HK_VLLM_USER"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    monkeypatch.setattr(support, "check",
                        lambda engine, arch: ([], {"vllm": "0.33.0", "torch": "2.13.0",
                                                   "ROCm": "7.2.3", "arch": arch}))
    on = list(switches.on("vllm"))
    assert on, "no vLLM route on by default on gfx1100?"
    monkeypatch.setattr(_vllm_plugin, "APPLY", {n: (lambda n=n: (f"{n} ok", None)) for n in on})
    monkeypatch.setattr(contract, "apply_routes",
                        lambda engine, routes, appliers: ([f"{n} ok" for n in routes], list(routes), {}))
    from hk.integration import stack
    monkeypatch.setattr(stack, "enforce", lambda *a, **k: "stack: test")
    from hk.runtime import compile as rc
    monkeypatch.setattr(rc, "arm_jit_notice", lambda *a, **k: None)
    _vllm_plugin.register()
    out = capsys.readouterr().out.splitlines()
    import hk
    head = next(x for x in out if x.startswith("[hk] ====="))
    assert f"hk-dsl {hk.__version__}" in head and "vllm 0.33.0" in head and "arch gfx1100" in head
    assert any(x.startswith("[hk] vLLM routes on: ") for x in out)
    off = next(x for x in out if x.startswith("[hk] turn hk off: HK_DISABLE=1"))
    for n in on:
        assert f"{switches.ROUTES[n].env}=0" in off



def test_the_summary_lists_the_routes_held_off_with_their_reason():
    lines = support.summary("vLLM", "0.0", {"arch": "gfx1100"}, ["vllm_gemm"], [], {"vllm_gemm": "HK_VLLM_GEMM"}, 1,
                            [("vllm_int4", "off by default on gfx1100: holds a second copy of the 4-bit weights "
                                           "(~12 GB on a 27B model, taken from the KV cache); HK_VLLM_INT4=1 turns it on")])
    held = [x for x in lines if x.startswith("[hk] off by default here: ")]
    assert len(held) == 1 and "vllm_int4 (" in held[0] and "second copy" in held[0] and "HK_VLLM_INT4=1" in held[0]
