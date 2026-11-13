"""scripts/check_docs.py: a table's delimiter / rows must match its header; relative links must exist."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts" / "check_docs.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)


def _probs(tmp_path, text):
    p = tmp_path / "x.md"
    p.write_text(text)
    cd.ROOT = tmp_path
    return cd.check(p)


def test_a_delimiter_wider_than_its_header_is_caught(tmp_path):
    probs = _probs(tmp_path, "| a | b |\n|---|---|---|\n| 1 | 2 |\n")
    assert len(probs) == 1 and "delimiter row has 3 cells, header 2" in probs[0]


def test_a_short_row_and_a_missing_link_are_caught_and_code_is_not(tmp_path):
    (tmp_path / "there.md").write_text("x")
    text = ("| a | b |\n|---|---|\n| `x|y` | 2 |\n| 3 |\n\n[ok](there.md) [gone](missing.md#a) "
            "[web](https://example.com)\n```\n| not | a | table |\n|---|\n```\n")
    probs = _probs(tmp_path, text)
    assert len(probs) == 2
    assert "row has 1 cells, header 2" in probs[0] and "missing.md" in probs[1]


def test_the_repository_docs_are_clean():
    cd.ROOT = ROOT
    assert [x for p in cd.files([]) for x in cd.check(p)] == []
