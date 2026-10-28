"""hk.integration.vllm_dsv41_index without a GPU or vLLM: its contract (read from the source, not
imported), the import hook that patches mono_decode when vLLM imports it, the steps it leaves to
vLLM's chain, and the launch's variants tracing. A stand-in vLLM tree on sys.path plays the
engine."""

from __future__ import annotations

import sys
import types

import pytest

from hk.integration import contract, switches
from hk.integration import vllm_dsv41_index as PI

MONO = '''
_runner = None
CALLS = []


def _mono_runner(device, draft=None):
    return _runner


def _select(layer, positions, normed, qr):
    CALLS.append(normed.shape[0])


def run(layer, positions, normed, qr):
    return _select(layer, positions, normed, qr)
'''
LAYOUT = "def rocm_paged_mxfp4_cache_layout(n_head, head_dim, block): pass\n"


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    """A vLLM tree with mono_decode and the indexer layout module; returns a writer for
    mono_decode's source."""
    root = tmp_path / "vllm"
    for d in ("models/deepseek_v41/amd", "v1/attention/ops"):
        (root / d).mkdir(parents=True)
    for d in ("", "models", "models/deepseek_v41", "models/deepseek_v41/amd", "v1", "v1/attention",
              "v1/attention/ops"):
        (root / d / "__init__.py").write_text("")
    (root / "v1/attention/ops/rocm_paged_mxfp4_indexer.py").write_text(LAYOUT)
    mono = root / "models/deepseek_v41/amd/mono_decode.py"
    mono.write_text(MONO)
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(switches, "arch", lambda: "gfx950")
    import hk.target

    monkeypatch.setattr(hk.target, "current_arch", lambda: "gfx950")
    yield mono.write_text
    PI.revert()
    PI.STOCK_LAYERS.clear()
    PI._LAYERS.clear()


def test_the_contract_holds_on_the_measured_shape(fake_vllm):
    assert contract.check(["vllm_dsv41_index"]) == []


def test_the_contract_does_not_import_the_model_package(fake_vllm):
    contract.check(["vllm_dsv41_index"])
    assert PI.MODULE not in sys.modules
    assert "vllm.models" not in sys.modules


def test_a_renamed_parameter_breaks_the_contract(fake_vllm):
    fake_vllm(MONO.replace("def _select(layer, positions, normed, qr)",
                           "def _select(layer, positions, x, qr)"))
    (p,) = contract.check(["vllm_dsv41_index"])
    assert p.startswith("vllm_dsv41_index: ") and "'x'" in p


def test_a_missing_runner_breaks_the_contract(fake_vllm):
    fake_vllm(MONO.replace("_mono_runner", "_runner_of"))
    (p,) = contract.check(["vllm_dsv41_index"])
    assert "_mono_runner" in p


def test_a_vllm_without_the_mono_path_is_not_a_broken_contract(fake_vllm, tmp_path):
    (tmp_path / "vllm/models/deepseek_v41/amd/mono_decode.py").unlink()
    assert contract.check(["vllm_dsv41_index"]) == []
    assert "no DeepSeek-V4.1 mono decode path" in PI.apply_patch()


def test_off_gfx950_the_contract_does_not_apply(fake_vllm, monkeypatch):
    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    fake_vllm("")
    assert contract.check(["vllm_dsv41_index"]) == []


def test_it_patches_mono_decode_when_vllm_imports_it(fake_vllm):
    report = PI.apply_patch()
    assert "when vLLM imports it" in report
    assert PI.MODULE not in sys.modules            # nothing imported by the patch
    import importlib

    m = importlib.import_module(PI.MODULE)
    assert m._select is not PI._ORIG and m._select.__wrapped__ is PI._select
    assert PI.status().startswith("hk_vllm::dsv41_index: ")
    orig = PI._ORIG
    assert PI.revert() == "dsv41_index -> vllm"
    assert m._select is orig and not hasattr(m._select, "__wrapped__")


def test_it_patches_an_already_imported_module_at_once(fake_vllm):
    import importlib

    m = importlib.import_module(PI.MODULE)
    orig = m._select
    assert PI.apply_patch().startswith("DeepSeek-V4.1 index layers")
    assert m._select.__wrapped__ is PI._select
    PI.revert()
    assert m._select is orig


def test_a_changed_signature_at_import_leaves_the_module_alone(fake_vllm, capsys):
    fake_vllm(MONO.replace("def _select(layer, positions, normed, qr)",
                           "def _select(layer, positions, normed, qr, extra)"))
    PI.apply_patch()
    import importlib

    m = importlib.import_module(PI.MODULE)
    assert not hasattr(m._select, "__wrapped__")
    assert "not the one measured" in capsys.readouterr().out


def _step(rows):
    return types.SimpleNamespace(shape=(rows, 5120))


def _layer(lid, draft=False):
    return types.SimpleNamespace(attn=types.SimpleNamespace(layer_id=lid),
                                 mono=types.SimpleNamespace(draft=draft))


def test_steps_it_does_not_take_run_vllms_chain(fake_vllm, monkeypatch, capsys):
    import importlib

    PI.apply_patch()
    m = importlib.import_module(PI.MODULE)
    before = dict(PI.STATS)
    m.run(_layer(2), None, _step(9), None)         # more rows than the launch takes
    m.run(_layer(61, draft=True), None, _step(1), None)
    monkeypatch.setattr(PI, "ENABLED", False)
    m.run(_layer(2), None, _step(1), None)
    assert m.CALLS == [9, 1, 1]
    assert PI.STATS["stock"] - before["stock"] == 3 and PI.STATS["hk"] == before["hk"]
    assert "draft" in PI.STOCK_LAYERS[61]
    assert "layer 61 stays on vLLM's chain" in capsys.readouterr().out


def test_the_route_and_the_launch_agree_on_rows():
    pytest.importorskip("hk._mlir")
    from hk.ops import dsv41_index as X

    assert PI.ROWS == X.ROWS


@pytest.mark.parametrize("consts", [
    dict(RATIO=2, HASKV=1, MC=8, SEL=1, WIDTH=12288),              # a dense index source
    dict(RATIO=1, HASKV=1, MC=1, SEL=1, CAND=1, WIDTH=8192),       # the candidate source
    dict(RATIO=1, HASKV=0, MC=4, SEL=1, WIDTH=4096),               # a consumer
    dict(RATIO=2, HASKV=1, MC=3, SEL=0)])                          # logits / top-k left to vLLM
def test_the_launch_traces(consts):
    pytest.importorskip("hk._mlir")
    from hk.ops import dsv41_index as X

    t = X.index_kernel(0.0884).trace(**consts)
    src = t.cpp(['"kittens.cuh"'])
    assert "dsv41_index" in src
