"""docs/api and llms-full.txt are generated from the code: they must be
regenerated when a public signature or docstring changes, and every public
name must have a docstring for them to show."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_api_docs.py"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_api_docs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_api_docs_are_current():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, (
        "docs/api or llms-full.txt is stale; run `python3 scripts/gen_api_docs.py` "
        "and commit the result:\n" + r.stdout[-4000:] + r.stderr[-2000:])


def test_every_documented_name_has_a_docstring():
    empty = _gen().empty_docstrings()
    assert not empty, f"no docstring (the API reference shows them empty): {empty}"


def test_since_marks_against_the_pypi_wheel():
    gen = _gen()
    assert gen.PYPI == "0.8.0" and gen.BASELINE.name == "api_baseline_0.8.0.json"
    # 0.8.0's additions are in the 0.8.0 wheel: no mark left on them
    arch_md = (REPO / "docs" / "api" / "hk.arch.md").read_text()
    section = arch_md.split("### `hk.arch.xcd_count`", 1)[1].split("###", 1)[0]
    assert "after 0.8.0" not in section          # in it (#63)
    hk_md = (REPO / "docs" / "api" / "hk.md").read_text()
    for name in ("hk.Placement", "hk.map_cpp", "hk.iota", "hk.RS"):
        section = hk_md.split(f"### `{name}`", 1)[1].split("###", 1)[0]
        assert "after 0.8.0" not in section, name   # #53, #72 are in 0.8.0
    # hk.kernel: placement= is in 0.8.0; max_occupancy= came after it, and is the only mark
    section = hk_md.split("### `hk.kernel`", 1)[1].split("###", 1)[0]
    assert "*Changed after 0.8.0*: new parameter `max_occupancy`" in section
    assert "placement" not in section.split("after 0.8.0", 1)[1].split("\n", 1)[0]
    group_md = (REPO / "docs" / "api" / "hk.group.md").read_text()
    section = group_md.split("#### `Group.wait_for`", 1)[1].split("###", 1)[0]
    assert "after 0.8.0" not in section          # *tiles, exact: unchanged, not reordered
    ops_md = (REPO / "docs" / "api" / "hk.ops.md").read_text()
    section = ops_md.split("### `hk.ops.scaled_mm`", 1)[1].split("###", 1)[0]
    assert "*New after 0.8.0*" in section        # not in the 0.8.0 wheel (#76)
    for page in (REPO / "docs" / "api").glob("*.md"):
        assert "after 0.7.0" not in page.read_text(), page.name   # the old baseline's marks
    # what lands after 0.8.0 is marked against the 0.8.0 wheel
    base = gen.load_baseline()
    import hk

    base["hk"] = {n: v for n, v in base["hk"].items() if n != "map_cpp"}
    base["hk"]["kernel"] = {"params": [p for p in base["hk"]["kernel"]["params"]
                                       if p != "placement"]}
    since = gen.Since(base)
    assert since.mark("hk", "map_cpp", hk.map_cpp) == "*New after 0.8.0*"
    assert since.mark("hk", "kernel", hk.kernel) == \
        "*Changed after 0.8.0*: new parameters `max_occupancy`, `placement`"
    assert since.mark("hk", "iota", hk.iota) == ""
    index = (REPO / "docs" / "api" / "README.md").read_text()
    if gen.RELEASE:      # a release PR: the release that has the marked names
        assert f"{gen.RELEASE} does, and a checkout" in index
    else:                # between releases: main only
        assert "no release has it yet, only a checkout of main" in index


def test_math_page_says_what_gelu_is():
    text = (REPO / "docs" / "api" / "hk.math.md").read_text()
    assert 'approximate="tanh"' in text
    import hk

    for name in hk.math.__all__:
        assert f"### `hk.math.{name}`" in text


def test_baseline_reader(tmp_path):
    """The wheel reader resolves star imports and re-exports."""
    pkg = tmp_path / "hk"
    (pkg / "dsl").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .dsl import *\nfrom .dsl import f as g\n")
    (pkg / "dsl" / "kernel.py").write_text("def k(fn, *, warps=1): pass\n")
    (pkg / "dsl" / "__init__.py").write_text(
        "from .kernel import k\n"
        "__all__ = ['f', 'C', 'k']\n"
        "def f(a, b=1, *, c=2): pass\n"
        "class C:\n    def m(self, x): pass\n    def w(self, *t, exact=False): pass\n")
    base = _gen().record_baseline(tmp_path)
    assert base["hk"]["f"]["params"] == ["a", "b", "c"]
    assert base["hk"]["g"]["params"] == ["a", "b", "c"]
    # a re-export reached through a star import resolves in the module that made it
    assert base["hk"]["k"]["params"] == ["fn", "warps"]
    assert base["hk.dsl"]["C"]["methods"]["m"] == ["self", "x"]
    # inspect.signature's order, so an unchanged signature is not "reordered"
    assert base["hk.dsl"]["C"]["methods"]["w"] == ["self", "*t", "exact"]
