"""The framework patch, minus the framework.

What is worth testing here without vLLM or SGLang installed is the one piece
that is subtle: a module that did `from torch.nn.functional import
scaled_dot_product_attention` holds the *original function object*, and
rebinding the attribute on `torch.nn.functional` does nothing for it. The
rebind-by-identity walk is what covers that case, and it is pure Python.
"""

import sys
import types

import pytest

from hk.integration import _common


def _fake(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture
def fake_modules():
    made = []

    def make(name, **attrs):
        made.append(name)
        return _fake(name, **attrs)

    yield make
    for n in made:
        sys.modules.pop(n, None)


def test_a_direct_import_is_rebound(fake_modules):
    def original():
        return "torch"

    def replacement():
        return "hk"

    mod = fake_modules("fakefw.attention", sdpa=original)
    hits = _common.rebind(("fakefw",), original, replacement)
    assert hits == ["fakefw.attention.sdpa"]
    assert mod.sdpa() == "hk"


def test_rebinding_is_by_identity_not_by_name(fake_modules):
    def original():
        pass

    def impostor():
        pass

    mod = fake_modules("fakefw.other", scaled_dot_product_attention=impostor)
    assert _common.rebind(("fakefw",), original, lambda: None) == []
    assert mod.scaled_dot_product_attention is impostor


def test_only_the_named_prefixes_are_touched(fake_modules):
    def original():
        pass

    mine = fake_modules("fakefw.a", f=original)
    theirs = fake_modules("otherpkg.b", f=original)
    _common.rebind(("fakefw",), original, lambda: None)
    assert mine.f is not original
    assert theirs.f is original


def test_every_name_bound_to_it_in_one_module_is_rebound(fake_modules):
    def original():
        pass

    def replacement():
        pass

    mod = fake_modules("fakefw.aliases", sdpa=original, _sdpa=original)
    hits = _common.rebind(("fakefw",), original, replacement)
    assert sorted(hits) == ["fakefw.aliases._sdpa", "fakefw.aliases.sdpa"]
    assert mod.sdpa is replacement and mod._sdpa is replacement


def test_an_absent_framework_reports_rather_than_raises():
    import hk.integration as hki

    out = hki.apply(["vllm", "sglang"])
    assert set(out) == {"vllm", "sglang"}
    assert all("not importable" in v for v in out.values())


def test_an_unknown_backend_is_named_not_ignored():
    import hk.integration as hki

    assert "unknown backend" in hki.apply(["nope"])["nope"]
