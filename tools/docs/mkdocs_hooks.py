"""MkDocs hooks for the docs site (mkdocs.yml).

- A link from a page in docs/ to a file outside docs/ -- llms.txt, examples/,
  bench/dx/, skills/, the README -- becomes a link to that file on GitHub,
  which the site cannot serve.
- docs/templates/ is content: MkDocs leaves any templates/ directory out by
  default (a theme override's place), and no exclude_docs setting undoes that.
- Each chapter of the development record (docs/history/NN-*.md) opens with the
  archive's banner on the site: a dated record, not current documentation.
"""

from __future__ import annotations

import posixpath
import re

REPO = "https://github.com/vivienfanghuagood/hk-dsl/blob/main/"
TREE = "https://github.com/vivienfanghuagood/hk-dsl/tree/main/"
# ](target) or ](target#anchor), not inside code; images too.
LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")
FENCE = re.compile(r"^(```|~~~)")


def _outside(page_path: str, target: str):
    """The repository path `target` names, if it leaves docs/; else None."""
    if "://" in target or target.startswith(("mailto:", "/")):
        return None
    joined = posixpath.normpath(posixpath.join("docs", posixpath.dirname(page_path), target))
    return None if joined == "docs" or joined.startswith("docs/") else joined


def on_files(files, config):  # noqa: ARG001
    from mkdocs.structure.files import InclusionLevel  # noqa: PLC0415

    for f in files:
        if f.src_uri.startswith("templates/") and f.src_uri.endswith(".md"):
            f.inclusion = InclusionLevel.INCLUDED
    return files


HISTORY_BANNER = (
    '!!! note "Archived: a dated record"\n'
    "    This chapter is the state of the work when it was written: its API, numbers and "
    "baselines are of that time. Current documentation starts at [the docs home](../README.md); "
    "the numbers the project quotes are on [the scorecard](../performance/scorecard.md).\n\n")


def on_page_markdown(markdown, page, config, files):  # noqa: ARG001 (MkDocs' signature)
    if re.match(r"history/\d", page.file.src_uri):
        title, _, rest = markdown.partition("\n")
        markdown = f"{title}\n\n{HISTORY_BANNER}{rest.lstrip()}"
    out, fenced = [], False
    for line in markdown.split("\n"):
        if FENCE.match(line.lstrip()):
            fenced = not fenced
        if not fenced:
            def repl(m):
                path = _outside(page.file.src_uri, m.group(2))
                if path is None:
                    return m.group(0)
                base = TREE if m.group(2).endswith("/") or "." not in posixpath.basename(path) else REPO
                return f"{m.group(1)}{base}{path}{m.group(3) or ''}{m.group(4)}"
            line = LINK.sub(repl, line)
        out.append(line)
    return "\n".join(out)
