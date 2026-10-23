"""hk.integration.stack: which Triton a vLLM runs with, without vLLM.

The check reads versions; the cases are the stacks the engines' pod has had
(chapter 44): the matched venv, the venv where xgrammar's unversioned
`triton` 3.8.0 went over triton-rocm 3.7.1, and a vLLM no table lists.
"""

import pytest

from hk.integration import ab, stack

MATCHED = {"vllm": "0.31.0+rocm721", "torch": "2.13.0+rocm7.2", "triton": "3.7.1",
           "triton_dists": {"triton-rocm": "3.7.1"}, "python": "/v/bin/python"}
OVERWRITTEN = {**MATCHED, "triton": "3.8.0",
               "triton_dists": {"triton": "3.8.0", "triton-rocm": "3.7.1"}}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("HK_STACK_CHECK", raising=False)
    monkeypatch.setattr(stack, "_WARNED", False)


def test_the_matched_stack_passes():
    assert stack.check(MATCHED) == []
    assert stack.enforce("t", default="refuse", found=MATCHED).startswith("stack matched")


def test_the_overwritten_venv_is_named_twice():
    problems = stack.check(OVERWRITTEN)
    assert len(problems) == 2
    assert "Triton 3.8.0" in problems[0] and "triton-rocm 3.7.1" in problems[0]
    assert "triton 3.8.0, triton-rocm 3.7.1" in problems[1]


def test_a_lone_wrong_triton_and_a_wrong_torch():
    assert len(stack.check({**MATCHED, "triton": "3.8.0",
                            "triton_dists": {"triton": "3.8.0"}})) == 1
    assert "torch 2.10.0" in stack.check({**MATCHED, "torch": "2.10.0"})[0]


def test_a_local_version_suffix_is_the_same_triton():
    assert stack.check({**MATCHED, "triton": "3.7.1+gitabc"}) == []


def test_no_triton_at_all_is_a_mismatch():
    assert "imports Triton nothing" in stack.check({**MATCHED, "triton": None,
                                                    "triton_dists": {}})[0]


@pytest.mark.parametrize("vllm", [None, "0.16.1", "0.32.0"])
def test_a_vllm_without_a_measured_stack_is_not_held_to_one(vllm):
    found = {**OVERWRITTEN, "vllm": vllm}
    assert stack.check(found) == []
    assert stack.enforce("t", default="refuse", found=found).startswith("stack not checked")


def test_refuse_raises_with_the_fix(monkeypatch):
    with pytest.raises(stack.StackMismatch) as e:
        stack.enforce("t", default="refuse", found=OVERWRITTEN)
    msg = str(e.value)
    assert "/v/bin/python -m pip uninstall -y triton\n" in msg
    assert "--no-deps --force-reinstall 'triton-rocm==3.7.1'" in msg


def test_warn_prints_once_and_carries_on(capsys):
    note = stack.enforce("t", default="warn", found=OVERWRITTEN)
    assert note.startswith("STACK MISMATCH")
    stack.enforce("t", default="warn", found=OVERWRITTEN)
    assert capsys.readouterr().err.count("not the stack vLLM") == 1


@pytest.mark.parametrize("env, out", [("off", "stack not checked"), ("0", "stack not checked"),
                                      ("warn", "STACK MISMATCH")])
def test_the_environment_overrides_the_default(monkeypatch, env, out):
    monkeypatch.setenv("HK_STACK_CHECK", env)
    assert stack.enforce("t", default="refuse", found=OVERWRITTEN).startswith(out)


def test_an_unknown_policy_is_an_error(monkeypatch):
    monkeypatch.setenv("HK_STACK_CHECK", "maybe")
    with pytest.raises(ValueError):
        stack.enforce("t", found=MATCHED)


def test_series():
    assert stack.series("0.31.0+rocm721") == "0.31"
    assert stack.series("0.31.1") == "0.31"
    assert stack.series(None) is None


def test_installed_reads_this_python_without_importing_triton():
    found = stack.installed()
    assert set(found) == {"vllm", "torch", "triton", "triton_dists", "python"}


def test_ab_refuses_before_starting_an_engine(monkeypatch, capsys):
    monkeypatch.setattr(stack, "installed", lambda: OVERWRITTEN)
    ran = []
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: ran.append(a))
    assert ab.main(["--batch", "1", "--short", "4", "--tokens", "8"]) == 2
    assert ran == []
    assert "pip uninstall -y triton" in capsys.readouterr().err
