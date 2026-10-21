"""llms.txt and the hk-kernel skill: every link resolves, and the kernel
llms.txt teaches traces for both GPUs (the docs' code must not rot)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = [REPO / "llms.txt", REPO / "skills" / "hk-kernel" / "SKILL.md", REPO / "docs" / "check.md"]
#: Written by the template streams; a link to them may land first.
PENDING = ("docs/templates",)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_links_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        if "://" in target:
            continue
        path = (doc.parent / target).resolve()
        rel = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        if not path.exists() and not rel.startswith(PENDING):
            missing.append(target)
    assert not missing, f"{doc.name}: links to nothing: {missing}"


def test_skill_frontmatter():
    text = (REPO / "skills" / "hk-kernel" / "SKILL.md").read_text(encoding="utf-8")
    head = text.split("---")[1]
    assert re.search(r"^name: hk-kernel$", head, re.M)
    assert re.search(r"^description: .{40,}", head, re.M)


def _first_kernel(doc: Path, tmp_path: Path, name: str):
    """The first ```python block of `doc` that defines an @hk.kernel, as a
    module (written to a file: a kernel body must be readable source)."""
    blocks = re.findall(r"```python\n(.*?)```", doc.read_text(encoding="utf-8"), re.S)
    src = next(b for b in blocks if "@hk.kernel" in b)
    f = tmp_path / f"{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
@pytest.mark.parametrize("doc", ["llms.txt", "docs/check.md"])
def test_the_docs_kernel_traces(tmp_path, doc, arch):
    pytest.importorskip("torch")       # the snippets import it, as a user's file does
    import hk

    mod = _first_kernel(REPO / doc, tmp_path, "doc_kernel_" + doc.replace("/", "_").replace(".", "_"))
    k = mod.gelu_mul
    on = hk.kernel(k.fn, arch=arch, grid=k.grid_fn, warps=k.warps, name=k.name)
    t = on.trace()
    assert t.arch == arch and t.written_tensors() == ["o"]
