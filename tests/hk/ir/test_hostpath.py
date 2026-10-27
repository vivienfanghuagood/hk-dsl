"""The host fast paths' pure-Python parts (hk.runtime.hostpath), without a GPU or torch:
the switch, and K1's argument mapping, which must say what inspect's bind() says."""

from __future__ import annotations

import inspect

import pytest

from hk.integration import vllm_moe_a8w4 as P
from hk.runtime import hostpath


def test_the_switch_flips_and_reports_what_it_was():
    prev = hostpath.set_enabled(False)
    try:
        assert hostpath.ENABLED is False
        assert hostpath.set_enabled(True) is False
        assert hostpath.ENABLED is True
    finally:
        hostpath.set_enabled(prev)


def _aiter_like(hidden_states, w1, w2, topk_weight, topk_ids, expert_mask=None, activation=None,
                quant_type=None, doweight_stage1=False, w1_scale=None, w2_scale=None,
                a1_scale=None, a2_scale=None, num_local_tokens=None, dtype=None, hidden_pad=0,
                intermediate_pad=0, bias1=None, bias2=None, gate_mode="", swiglu_limit=0.0,
                **kwargs):
    raise AssertionError("never called")


def _strict(hidden_states, w1, w2, topk_weight, topk_ids, expert_mask=None, *, gate_mode=""):
    raise AssertionError("never called")


def _bound(f, args, kwargs):
    kw = inspect.signature(f).bind("h", "w1", "w2", "tw", "ti", *args, **kwargs).arguments
    kw.update(kw.pop("kwargs", {}))
    return dict(kw)


FIVE = ("h", "w1", "w2", "tw", "ti")

CALLS = [
    # vLLM's: eight positionals after the five, then keywords (vllm/_aiter_ops.py)
    (("em", "act", "qt", False, "s1", "s2", None, None),
     dict(num_local_tokens=None, dtype="bf16", hidden_pad=0, intermediate_pad=0, bias1=None,
          bias2=None, swiglu_limit=7.0, gate_mode="interleave")),
    ((), dict(activation="act", w1_scale="s1")),
    ((), {}),
    (("em",), dict(moe_sorting_dispatch_policy=0)),        # an unknown name, into **kwargs
]


@pytest.mark.parametrize("args,kwargs", CALLS)
def test_k1s_argument_map_is_binds(monkeypatch, args, kwargs):
    monkeypatch.setattr(P, "_ORIG", _aiter_like)
    assert P._arguments(args, kwargs, FIVE) == _bound(_aiter_like, args, kwargs)


@pytest.mark.parametrize("args,kwargs", [
    (("em",), dict(expert_mask="again")),                  # a name given twice
    (tuple(range(30)), {}),                                # more positionals than parameters
    ((), dict(w1="again")),                                # one of the five, again
])
def test_what_bind_refuses_the_map_refuses_the_same_way(monkeypatch, args, kwargs):
    monkeypatch.setattr(P, "_ORIG", _aiter_like)
    with pytest.raises(TypeError) as theirs:
        _bound(_aiter_like, args, kwargs)
    with pytest.raises(TypeError) as ours:
        P._arguments(args, kwargs, FIVE)
    assert str(ours.value) == str(theirs.value)


def test_without_var_kwargs_an_unknown_name_is_binds_error(monkeypatch):
    monkeypatch.setattr(P, "_ORIG", _strict)
    with pytest.raises(TypeError, match="unexpected keyword"):
        P._arguments((), dict(swiglu_limit=1.0), FIVE)
    assert P._arguments(("em",), dict(gate_mode="x"), FIVE) == _bound(_strict, ("em",),
                                                                      dict(gate_mode="x"))


def test_a_new_original_is_read_again(monkeypatch):
    monkeypatch.setattr(P, "_ORIG", _aiter_like)
    P._arguments((), {}, FIVE)
    monkeypatch.setattr(P, "_ORIG", _strict)
    with pytest.raises(TypeError):
        P._arguments((), dict(swiglu_limit=1.0), FIVE)
