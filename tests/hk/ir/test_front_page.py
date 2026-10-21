"""The front page and the docs index: every link resolves (anchors included),
and the code README.md shows traces for both GPUs, so the first thing a
reader copies cannot rot. The same code runs on the GPUs in the GPU tier's
examples (12, 13); here it needs no GPU and no hipcc."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = [REPO / "README.md", REPO / "docs" / "README.md", REPO / "docs" / "getting-started.md",
        REPO / "docs" / "language.md", REPO / "docs" / "performance.md",
        REPO / "examples" / "README.md"]


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    s = re.sub(r"[`*_]", "", heading.strip().lower())
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def _anchors(md: Path) -> set:
    seen, out = {}, set()
    in_code = False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code or not line.startswith("#"):
            continue
        s = _slug(line.lstrip("#"))
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_links_and_anchors_resolve(doc):
    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
    bad = []
    for target, anchor in re.findall(r"\]\(([^)#\s]*)(?:#([^)\s]*))?\)", text) + \
            re.findall(r'(?:src|srcset)="([^"#]+)()"', text):
        if "://" in target:
            continue
        path = (doc.parent / target).resolve() if target else doc
        if not path.exists():
            bad.append(target)
        elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
            bad.append(f"{target}#{anchor}")
    assert not bad, f"{doc.name}: links to nothing: {bad}"


def _blocks(doc: str):
    return re.findall(r"```python\n(.*?)```", (REPO / doc).read_text(encoding="utf-8"), re.S)


def _module(src: str, tmp_path: Path, name: str):
    """`src` as a module, from a file (a kernel body must be readable source);
    top-level print(...) lines -- what they print needs hipcc -- left out."""
    src = "\n".join(l for l in src.splitlines() if not l.startswith("print("))
    f = tmp_path / f"{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_the_front_page_templates_trace(tmp_path, arch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("hk._mlir")
    import hk
    from hk import templates as T

    # Another test may have imported hk.templates.elementwise / .rowwise as
    # modules first, which binds the module, not the template, to that name on
    # the package (hk.templates.gemm guards against it; these two do not yet).
    # This test is about the README's code, so it looks the templates up anew.
    import types
    for name in ("elementwise", "rowwise"):
        if isinstance(vars(T).get(name), types.ModuleType):
            delattr(T, name)
    src = next(b for b in _blocks("README.md") if "from hk import templates" in b)
    mod = _module(src, tmp_path, "readme_my_kernels")
    B = hk.bf16
    ew = T.elementwise(mod.gelu_and_mul.fn, arch=arch)
    k, c = ew.kernel(T.meta((4096, 14336), B, stride=(28672, 1)),
                     T.meta((4096, 14336), B, stride=(28672, 1)))
    assert k.trace(**c).arch == arch
    rw = T.rowwise(mod.rms_norm.fn, arch=arch)
    k, c = rw.kernel(T.meta((4096, 4096), B), T.meta((4096,), B))
    assert k.trace(**c).arch == arch
    bias = torch.empty(12288, dtype=torch.bfloat16, device="meta")
    launches = hk.templates.gemm.kernels(4096, 12288, 4096, epilogue=mod.bias_gelu,
                                         bias=bias, arch=arch)
    assert any(st.epilogue for st, _, _ in launches)
    for _, kern, consts in launches:
        assert kern.trace(**consts).arch == arch


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
@pytest.mark.parametrize("doc", ["README.md", "docs/getting-started.md"])
def test_the_raw_kernels_trace(tmp_path, doc, arch):
    pytest.importorskip("torch")
    pytest.importorskip("hk._mlir")
    import hk

    for i, src in enumerate(b for b in _blocks(doc) if "@hk.kernel" in b):
        mod = _module(src, tmp_path, f"doc_{Path(doc).stem}_{i}")
        for k in (v for v in vars(mod).values() if isinstance(v, hk.Kernel)):
            on = hk.kernel(k.fn, arch=arch, grid=k.grid_fn, warps=k.warps, name=k.name)
            assert on.trace().arch == arch
