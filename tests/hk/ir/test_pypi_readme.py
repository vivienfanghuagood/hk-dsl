"""The package index shows setup.py's pypi_readme(README.md): every link in it
must be absolute, and every relative link in README.md must name a file or
directory that exists, so the absolute one resolves at the release's tag."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _setup():
    # setup.py only calls setup() as __main__, so loading it under another name is safe
    spec = importlib.util.spec_from_file_location("hk_setup", REPO / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _targets(text):
    return re.findall(r'!?\]\(([^)\s]+)', text)


def test_pypi_description_has_no_relative_links():
    mod = _setup()
    text = mod.pypi_readme((REPO / "README.md").read_text(encoding="utf-8"), mod._version())
    relative = [t for t in _targets(text) if not re.match(r"[a-z][a-z0-9+.-]*:", t, re.I)]
    assert relative == [], relative


def test_readme_relative_links_exist():
    missing = []
    for t in _targets((REPO / "README.md").read_text(encoding="utf-8")):
        if re.match(r"[a-z][a-z0-9+.-]*:", t, re.I) or t.startswith("#"):
            continue
        path = t.split("#", 1)[0]
        if not (REPO / path).exists():
            missing.append(t)
    assert missing == [], missing


def test_rewrite_forms():
    f = _setup().pypi_readme
    assert f("[a](docs/x.md#y)", "1.2.3") == (
        "[a](https://github.com/vivienfanghuagood/hk-dsl/blob/v1.2.3/docs/x.md#y)")
    assert f("![c](tools/c.png)", "1.2.3") == (
        "![c](https://raw.githubusercontent.com/vivienfanghuagood/hk-dsl/v1.2.3/tools/c.png)")
    assert f("[i](#install)", "1.2.3") == (
        "[i](https://github.com/vivienfanghuagood/hk-dsl/blob/v1.2.3/README.md#install)")
    assert f("[p](https://pypi.org/project/hk-dsl/)", "1.2.3") == (
        "[p](https://pypi.org/project/hk-dsl/)")
