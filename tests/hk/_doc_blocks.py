"""The kernels in docs/language.md's code blocks, imported as a module.

A block that defines a kernel (`@hk.kernel` or `@hk.specialize` at the start of a line) is written to a
file with the others, so that tracing can read each kernel's source. Body fragments are left out.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

LANGUAGE = Path(__file__).resolve().parents[2] / "docs" / "language.md"


def kernel_blocks(page: Path = LANGUAGE) -> list[str]:
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    return [b for b in blocks if re.search(r"^@hk\.(kernel|specialize)\b", b, re.M)]


def load(tmp_path: Path, page: Path = LANGUAGE):
    src = "import hk\n\n" + "\n\n".join(kernel_blocks(page))
    path = tmp_path / "doc_language_kernels.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("doc_language_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(), re.S)


def run(page: Path, marker: str, ns: dict) -> dict:
    """Execute, in `ns`, the one python block of `page` that contains `marker`. The source is
    registered with linecache, so whatever traces a function from it can read its source."""
    import linecache  # noqa: PLC0415

    found = [b for b in blocks(page) if marker in b]
    assert len(found) == 1, f"{page.name}: {len(found)} blocks contain {marker!r}"
    name = f"<{page.name}: {marker}>"
    linecache.cache[name] = (len(found[0]), None, found[0].splitlines(True), name)
    exec(compile(found[0], name, "exec"), ns)
    return ns
