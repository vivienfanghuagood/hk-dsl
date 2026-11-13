"""The vllm_quark route must not import aiter on its own (README failure, 2026-11-13).

vLLM's quark.schemes package imports the OCP MX scheme, which imports vllm._aiter_ops and with it
aiter; aiter JIT-builds kernels at import, which on RDNA with an image's ROCm clang fails loudly.
Stock vLLM imports that package only for a Quark model, so hk must not import it at plugin load:
the contract is read from the source file and the patch waits for vLLM's own import.
"""
import sys
import textwrap

import pytest

from hk.integration import contract, vllm_quark

SCHEME = '''
class QuarkW4A16Int4:
    def __init__(self, cfg):
        self.is_symmetric = True
        self.group_size = 128
        self.kernel = None
        self.pack_reorder = False

    def process_weights_after_loading(self, layer):
        return "orig-pwal"

    def apply_weights(self, layer, x, bias=None):
        return "orig-apply"
'''


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    """A vllm package whose quark.schemes __init__ imports aiter, like vLLM 0.16's."""
    root = tmp_path / "site"
    q = root / "vllm" / "model_executor" / "layers" / "quantization" / "quark" / "schemes"
    q.mkdir(parents=True)
    for d in [root / "vllm", root / "vllm" / "model_executor", root / "vllm" / "model_executor" / "layers",
              root / "vllm" / "model_executor" / "layers" / "quantization", q.parent]:
        (d / "__init__.py").write_text("")
    (q / "__init__.py").write_text("import aiter  # what quark_ocp_mx -> vllm._aiter_ops does\n")
    (q / "quark_w4a16_int4.py").write_text(textwrap.dedent(SCHEME))
    (root / "aiter.py").write_text("raise RuntimeError('aiter imported')\n")
    monkeypatch.syspath_prepend(str(root))
    for m in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.") or m == "aiter"]:
        monkeypatch.delitem(sys.modules, m)
    yield root
    vllm_quark._unhook()
    for m in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.") or m == "aiter"]:
        sys.modules.pop(m, None)


def test_the_contract_is_read_without_importing_the_scheme_or_aiter(fake_vllm):
    facts = contract.gather_vllm_quark()
    assert "aiter" not in sys.modules and vllm_quark.MODULE not in sys.modules
    assert facts == {"pwal": ["self", "layer"], "apply": ["self", "layer", "x", "bias"],
                     "class_has": {a: True for a in contract._QUARK_HAS}}
    assert contract.judge("vllm_quark", facts) == []


def test_the_file_and_the_loaded_class_give_the_same_facts(fake_vllm, monkeypatch):
    from_file = contract.gather_vllm_quark()
    (fake_vllm / "vllm" / "model_executor" / "layers" / "quantization" / "quark" / "schemes"
     / "__init__.py").write_text("")
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(vllm_quark.MODULE)
    assert contract.quark_facts_of(mod.QuarkW4A16Int4) == from_file


def test_the_patch_waits_for_vllms_own_import(fake_vllm, monkeypatch):
    (fake_vllm / "vllm" / "model_executor" / "layers" / "quantization" / "quark" / "schemes"
     / "__init__.py").write_text("")
    patched = []
    monkeypatch.setattr(vllm_quark, "_patch_module", lambda m: patched.append(m) or "patched")
    vllm_quark._FINDER = vllm_quark._Finder()
    sys.meta_path.insert(0, vllm_quark._FINDER)
    assert not patched and vllm_quark.MODULE not in sys.modules
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(vllm_quark.MODULE)       # vLLM loading a Quark model
    assert patched == [mod] and vllm_quark._FINDER is None
