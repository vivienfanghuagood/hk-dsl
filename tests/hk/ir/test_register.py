"""The registration entry point, without a GPU: arity, patching, loading."""

import sys
import types

import pytest

from hk.integration import ab, register


def test_patch_rebinds_and_revert_restores():
    mod = types.ModuleType("hk_test_target")
    mod.Thing = type("Thing", (), {"forward": lambda self, x: x})
    sys.modules["hk_test_target"] = mod
    try:
        orig = mod.Thing.forward
        register.patch("hk_test_target.Thing.forward", lambda self, x: x + 1)
        assert mod.Thing().forward(1) == 2
        assert register.revert() == 1
        assert mod.Thing.forward is orig
    finally:
        del sys.modules["hk_test_target"]


def test_patch_names_an_importable_module():
    with pytest.raises(ImportError):
        register.patch("no_such_module_anywhere.f", lambda: None)


def test_load_user_imports_files_and_modules(tmp_path):
    f = tmp_path / "my-kernels.py"
    f.write_text("LOADED = True\n")
    names = register.load_user(f"{f}, json")
    assert names == ["hk_user_my_kernels", "json"]


def test_the_schema_names_inputs_and_outputs():
    assert register._schema("f", 2, 1) == "f(Tensor x0, Tensor x1) -> Tensor"
    assert register._schema("g", 1, 2) == "g(Tensor x0) -> (Tensor, Tensor)"


def test_outputs_are_counted_from_the_allocator():
    torch = pytest.importorskip("torch")
    assert register._count_outputs(lambda x: x.new_empty(4, 4), 2) == 1
    assert register._count_outputs(lambda x, y: (x.new_empty(1), y.new_empty(1)), 4) == 2
    del torch


def test_ab_insists_the_short_run_covers_the_batch():
    with pytest.raises(SystemExit):
        ab.main(["--batch", "32", "--short", "16", "--tokens", "72"])


def test_python_m_hk_integration_ab_dispatches(monkeypatch):
    from hk.integration import __main__ as cli

    seen = []
    monkeypatch.setattr(ab, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["ab", "--arms", "user"]) == 0
    assert seen == [["--arms", "user"]]
