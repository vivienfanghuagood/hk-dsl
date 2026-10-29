"""scripts/check_public_hygiene.py: each built-in rule catches what it is for and
leaves the look-alikes alone; site patterns load from a file; the allowlist
exempts by path and whole match; the repository's allowlist parses."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "check_public_hygiene", REPO / "scripts" / "check_public_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _mod()


def _hits(rule: str, text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(MOD.RULES[rule], text)]


@pytest.mark.parametrize("rule,text,want", [
    ("private-ipv4", "ssh 10.1.2.3 and 192.168.0.7", ["10.1.2.3", "192.168.0.7"]),
    ("private-ipv4", "flydsl 0.3.4.1, MASTER_ADDR=127.0.0.1, 8.8.8.8", []),
    ("pci-address", "the lock /locks/gpu/0000:c3:00.0.lock", ["0000:c3:00.0"]),
    ("pci-address", "card 0000:03:00.0 and DDDD:BB:DD.F", ["0000:03:00.0"]),
    ("pci-address", "at 12:34:56.7, sha 1a2b:3c, v1.2.3", []),
    ("workspace-path", "cd /workspace/someone/hk-dsl-tune", ["/workspace/someone"]),
    ("workspace-path", "<workspace>/kda and a/workspace/x and /workspaces", []),
    ("home-dir", "state in ~/gha and ~/my-checkout/x", ["~/gha", "~/my-checkout"]),
    ("home-dir", "~/.cache/hk, $HOME/x, a~/b", []),
    ("container-host", '"host": "f2f887132f5d", "hostname": "0123456789ab"',
     ['"host": "f2f887132f5d"', '"hostname": "0123456789ab"']),
    ("container-host", '"host": "<host>", "host": "f2f887132f5", "sha": "0123456789ab"', []),
    ("home-path", "/home/alice/x but not $HOME or ~/.cache", ["/home/alice"]),
])
def test_rules(rule, text, want):
    assert _hits(rule, text) == want


def test_patterns_file(tmp_path):
    f = tmp_path / "site.txt"
    f.write_text("# a comment\n\n\\bmybox\\d+\\b\n(?i)SecretWiki\n")
    rules = MOD.load_patterns([str(f)])
    assert list(rules.values()) == ["\\bmybox\\d+\\b", "(?i)SecretWiki"]
    # A hit is reported by the file and line, never by the pattern.
    assert list(rules) == ["local:site.txt:3", "local:site.txt:4"]


def test_scan_and_allow(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    a = tmp_path / "scripts" / "run.sh"
    a.write_text("STATE=~/gha\nLOCKS=/workspace/locks\n")
    b = tmp_path / "doc.md"
    b.write_text("in /workspace/someone and ~/gha\n")
    allow_file = tmp_path / "allow.txt"
    allow_file.write_text("# why\nscripts/*.sh   ~/(?:gha|hk-ci)\nscripts/*.sh   /workspace/locks\n")
    allow = MOD.load_allow(allow_file)
    hits, allowed = MOD.scan([a, b], MOD.RULES, allow)
    assert sorted((r, t) for _, _, r, t in allowed) == [
        ("home-dir", "~/gha"), ("workspace-path", "/workspace/locks")]
    # The same names outside the allowed path are hits.
    assert sorted((p, r, t) for p, _, r, t in hits) == [
        ("doc.md", "home-dir", "~/gha"), ("doc.md", "workspace-path", "/workspace/someone")]


def test_repository_allowlist_parses():
    allow = MOD.load_allow(MOD.ALLOW)
    assert allow, "scripts/public_hygiene_allow.txt has no entries"
    for glob, rx in allow:
        assert glob and rx.pattern
