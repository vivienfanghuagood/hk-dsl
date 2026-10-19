"""`python -m hk.integration new-op` without a GPU: the files it writes trace
and translate, their own no-GPU tests pass, and the scoreboard finds the row."""

import importlib.util
import sys
from pathlib import Path

import pytest

from hk.integration import __main__ as cli
from hk.integration import newop

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "hk-bench"))
import _cases  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def generated(tmp_path):
    assert cli.main(["new-op", "my_op", "--dir", str(tmp_path)]) == 0
    return tmp_path


def test_it_writes_the_op_and_its_tests(generated):
    assert sorted(p.name for p in generated.iterdir()) == ["my_op.py", "test_my_op.py"]
    for p in generated.iterdir():
        assert "__NAME__" not in p.read_text() and "__PATH__" not in p.read_text()


def test_the_generated_kernel_traces_and_translates(generated):
    pytest.importorskip("hk._mlir")
    mod = _load(generated / "my_op.py", "hk_newop_ir_my_op")
    assert mod.NAME == "my_op"
    assert "func.func @" in mod.kernel.trace().dump()
    assert "__global__" in mod.kernel.source()


def test_the_generated_tests_pass_without_a_gpu(generated):
    pytest.importorskip("hk._mlir")
    t = _load(generated / "test_my_op.py", "hk_newop_ir_test_my_op")
    t.test_it_traces_and_translates()
    if importlib.util.find_spec("torch"):
        t.test_out_is_the_shape_of_the_reference()


def test_the_scoreboard_finds_its_row(generated, monkeypatch):
    monkeypatch.delenv(_cases.ENV, raising=False)
    files = _cases.paths(str(generated))
    assert files == [generated / "my_op.py"]       # the test file is not a case
    assert list(_cases.load(files)) == ["my_op"]
    monkeypatch.setenv(_cases.ENV, str(generated / "my_op.py"))
    assert _cases.paths("") == [generated / "my_op.py"]
    with pytest.raises(FileNotFoundError):
        _cases.paths(str(generated / "missing.py"))


def test_it_refuses_bad_names_and_existing_files(generated, capsys):
    assert newop.valid("my_op") == ""
    for bad in ("MyOp", "my-op", "1op", "class"):
        assert newop.valid(bad)
    with pytest.raises(SystemExit):
        newop.main(["My-Op", "--dir", str(generated)])
    assert newop.main(["my_op", "--dir", str(generated)]) == 1
    assert "not overwriting" in capsys.readouterr().out
    assert newop.main(["my_op", "--dir", str(generated), "--force"]) == 0
