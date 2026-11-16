"""hk.integration.sglang_diffusion's install: the .pth boot reaches every
process of the environment, applies the routes when SGLang-Diffusion's SDPA
backend is imported (hk is on by default), and does nothing with HK_DISABLE=1
or HK_SGLD=0. No GPU, no SGLang: a stand-in package of the same module path --
whose engine contract is broken, so the routes are checked and refused."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hk.integration import sglang_diffusion as sgld

REPO = Path(__file__).resolve().parents[3]


def _env(tmp_path, env_set):
    pkg = tmp_path / "pkgs" / Path(*sgld.TARGET.split(".")[:-1])
    pkg.mkdir(parents=True)
    for d in [tmp_path / "pkgs" / Path(*sgld.TARGET.split(".")[:i])
              for i in range(1, len(sgld.TARGET.split(".")))]:
        (d / "__init__.py").write_text("")
    (pkg / "sdpa.py").write_text("LOADED = True\n")
    site = tmp_path / "site"
    site.mkdir()
    sgld.install(site)
    env = dict(os.environ, PYTHONPATH=f"{tmp_path / 'pkgs'}{os.pathsep}{REPO / 'python'}")
    for k in ("HK_SGLD", "HK_DISABLE", "HK_SGLD_ATTN", "HK_SGLD_LINEAR", "HK_SGLD_CONVROT",
              "HK_SGLD_VAECONV", "HK_SGLD_ROPE", "HK_SGLD_QKNORM", "HK_SGLD_DITNORM",
              "HK_SGLD_H3VAE", "HK_SGLD_PATCHEMBED"):
        env.pop(k, None)
    env["HK_ARCH"] = "gfx1100"
    env.update(env_set)
    code = (f"import site; site.addsitedir({str(site)!r}); import {sgld.TARGET} as m; "
            f"print('loaded', m.LOADED)")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.mark.parametrize("env_set", [{}, {"HK_SGLD": "1"}, {"HK_SGLD": "attn"}])
def test_the_boot_applies_on_import_by_default_and_checks_the_contract(tmp_path, env_set):
    r = _env(tmp_path, env_set)
    assert r.returncode == 0, r.stderr
    assert "loaded True" in r.stdout
    # The stand-in is not SGLang-Diffusion: every route's contract is broken, so nothing is
    # patched and each route on says so in one warning line of its own.
    assert "[hk] sglang-diffusion: no route applied (" in r.stdout and "): stock" in r.stdout
    warn = [x for x in r.stderr.splitlines() if x.startswith("[hk] WARNING")]
    want = {"attn": ["sgld_attn"], "1": ["sgld_attn", "sgld_linear"]}.get(
        env_set.get("HK_SGLD"), ["sgld_attn", "sgld_linear", "sgld_vaeconv", "sgld_rope", "sgld_qknorm", "sgld_ditnorm"])
    assert len(warn) == len(want)
    for route, line in zip(want, warn):
        assert f"sglang_diffusion: route {route} OFF -- engine contract FAILED" in line


@pytest.mark.parametrize("env_set", [{"HK_DISABLE": "1"}, {"HK_SGLD": "0"}, {"HK_SGLD": "off"}])
def test_the_boot_does_nothing_when_turned_off(tmp_path, env_set):
    r = _env(tmp_path, env_set)
    assert r.returncode == 0, r.stderr
    assert "[hk]" not in r.stdout + r.stderr and "loaded True" in r.stdout


def test_uninstall_removes_both_files(tmp_path):
    written = sgld.install(tmp_path)
    assert all(p.exists() for p in written)
    assert sorted(sgld.uninstall(tmp_path)) == sorted(written)
    assert not any(p.exists() for p in written)


@pytest.mark.parametrize("src,ok", [
    # SGLang up to 0.5.22: the fallback goes through apply_qk_norm (norm(x) off CUDA)
    ("    out = apply_rope(apply_qk_norm(x, norm), rope)\n    return out\n", True),
    # SGLang 0.5.23: norm(x) directly, after two CUDA-only fusion gates
    ("    if _is_cuda:\n        return fused\n    out = apply_rope(norm(x), rope)\n", True),
    ("    return fused_only(x, norm, rope)\n", False),
])
def test_the_qknorm_contract_reads_both_forms_of_the_fallback(src, ok):
    from hk.integration import contract

    assert contract.norm_then_rope(src) is ok


def test_a_linear_decision_is_pinned_so_a_near_tie_cannot_flip_between_processes(tmp_path, monkeypatch):
    from hk.integration import sglang_diffusion as sd

    f = tmp_path / "sgld_linear_gfx1100_w7900d.json"
    monkeypatch.setattr(sd, "decisions_file", lambda: f)
    monkeypatch.setattr(sd, "DECISIONS", {})
    key = (2048, 256, 5120, False)
    timed = []
    assert sd._decide(key, lambda: timed.append(1) or (True, "hk (1.03x)"))[0] is True
    # a second process: its own timing of the near-tie would say "engine" -- the pin wins, untimed
    monkeypatch.setattr(sd, "DECISIONS", {})
    got = sd._decide(key, lambda: timed.append(2) or (False, "engine (1.02x)"))
    assert got[0] is True and "pinned" in got[1] and timed == [1]
    # another shape is decided (and pinned) on its own
    assert sd._decide((4096, 4096, 4096, True), lambda: (False, "engine"))[0] is False
    assert sorted(__import__("json").loads(f.read_text())) == ["2048,256,5120,0", "4096,4096,4096,1"]


def test_a_torn_decisions_file_is_decided_again(tmp_path, monkeypatch):
    from hk.integration import sglang_diffusion as sd

    f = tmp_path / "d.json"
    f.write_text("{not json")
    monkeypatch.setattr(sd, "decisions_file", lambda: f)
    monkeypatch.setattr(sd, "DECISIONS", {})
    assert sd._decide((1024, 64, 64, False), lambda: (True, "hk"))[0] is True
    assert "1024,64,64,0" in __import__("json").loads(f.read_text())


# -- more than one rank: SGLang-Diffusion's sequence / tensor parallel workers ---------------------


def test_with_ranks_nothing_is_timed_and_unpinned_shapes_keep_the_engine(tmp_path, monkeypatch):
    """A 4-GPU SP Qwen-Image replica deadlocked: a rank timed a new shape under the decisions
    file's lock while its stream waited on a collective the other ranks, blocked on that lock,
    never reached. With ranks a decision is the pinned one, or the engine's GEMM -- never timed."""
    import json

    from hk.integration import sglang_diffusion as sd

    f = tmp_path / "decisions.json"
    f.write_text(json.dumps({"4096,3072,3072,0": [True, "hk (1.27x)"]}))
    monkeypatch.setattr(sd, "decisions_file", lambda: f)
    monkeypatch.setattr(sd, "DECISIONS", {})
    monkeypatch.setattr(sd, "_PINNED", [])
    monkeypatch.setattr(sd, "world_size", lambda: 4)

    def never():
        raise AssertionError("timed a shape with 4 ranks")

    assert sd._decision((4096, 3072, 3072, False), never)[0] is True
    assert sd._decision((2048, 1024, 4096, False), never) == (
        False, "engine (unpinned; 4 ranks decide nothing at run time)")
    assert json.loads(f.read_text()) == {"4096,3072,3072,0": [True, "hk (1.27x)"]}   # nothing written


def test_a_held_lock_raises_instead_of_waiting(tmp_path, monkeypatch):
    import fcntl

    from hk.integration import sglang_diffusion as sd

    f = tmp_path / "decisions.json"
    monkeypatch.setattr(sd, "decisions_file", lambda: f)
    monkeypatch.setattr(sd, "DECISIONS", {})
    monkeypatch.setenv("HK_SGLD_LOCK_S", "0.2")
    with open(f"{f}.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="held by another process"):
            sd._decide((1024, 64, 64, False), lambda: (True, "hk"))


_RANK = r"""
import json, os, sys
from pathlib import Path
import torch.distributed as dist
from hk.integration import sglang_diffusion as sd

rank, world, table, port = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
sd.decisions_file = lambda: table
timed = []

def measure():
    # what the old route did inside the forward: wait for the other ranks (their collective)
    timed.append(1)
    dist.barrier()
    return True, "hk (timed)"

for key in [(2048, 1024, 4096, False), (1024, 3072, 3072, False), (2048, 256, 4096, False)]:
    sd._decision(key, measure)
    dist.barrier()                       # the next collective of the forward: every rank reaches it
print(json.dumps({"rank": rank, "timed": len(timed),
                  "decisions": {",".join(map(str, k)): v[0] for k, v in sorted(sd.DECISIONS.items())}}))
dist.destroy_process_group()
"""


@pytest.mark.parametrize("world", [2, 4])
def test_sp_ranks_decide_identically_and_never_deadlock(tmp_path, world):
    """The SP regression, with real ranks (gloo on CPU): each rank meets new GEMM shapes between
    collectives, as SGLang-Diffusion's SP workers do, with one shape pinned in the shared file.
    Every rank must finish, time nothing, and decide exactly what the others decide."""
    pytest.importorskip("torch.distributed")
    import json
    import socket

    table = tmp_path / "sgld_linear.json"
    table.write_text(json.dumps({"1024,3072,3072,0": [True, "hk (1.27x)"]}))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    script = tmp_path / "rank.py"
    script.write_text(_RANK)
    env = dict(os.environ, WORLD_SIZE=str(world))
    procs = [subprocess.Popen([sys.executable, str(script), str(r), str(world), str(table), port],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
             for r in range(world)]
    outs = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            pytest.fail(f"{world} ranks hung deciding GEMM shapes (the SP deadlock)")
        assert p.returncode == 0, err[-2000:]
        outs.append(json.loads(out.strip().splitlines()[-1]))
    assert all(o["timed"] == 0 for o in outs)
    assert all(o["decisions"] == outs[0]["decisions"] for o in outs)
    assert outs[0]["decisions"] == {"1024,3072,3072,False": True, "2048,1024,4096,False": False,
                                    "2048,256,4096,False": False}
