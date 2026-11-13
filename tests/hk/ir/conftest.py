"""Fixtures for the IR tests."""
import pytest


@pytest.fixture(autouse=True)
def _supported_stack(request, monkeypatch):
    """IR tests run without a GPU stack (CI's IR image has ROCm 7.1, no vLLM): the plugins' support
    check (hk.integration.support) passes here unless a test is about it."""
    if request.module.__name__.endswith("test_plugin_support"):
        return
    try:
        from hk.integration import support
    except ImportError:                 # a test that runs without hk importable
        return
    monkeypatch.setattr(support, "check", lambda engine, arch: ([], {"arch": arch or "test"}))
