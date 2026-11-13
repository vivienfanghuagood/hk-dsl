#!/usr/bin/env python3
"""check_docs.py -- fail on Markdown that GitHub renders broken.

1. Tables: every row of a pipe table must have as many cells as its header, and the delimiter row
   (|---|---|) too. GitHub drops the whole table to plain text when the delimiter count differs.
2. Relative links: [text](path) and [text](path#anchor) to a file in the repository must exist.

Checks README.md, CONTRIBUTING.md and docs/**/*.md (not docs/history or docs/archive: dated records).
Usage: python3 scripts/check_docs.py [FILES...]   (exit 1 and one line per problem)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
DELIM = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def cells(line: str) -> int:
    s = line.strip()
    s = re.sub(r"\\\|", "", s)                       # escaped pipes are text
    s = re.sub(r"`[^`]*`", "", s)                    # so are pipes in code spans
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.count("|") + 1


def files(args):
    if args:
        return [pathlib.Path(a) for a in args]
    out = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    for p in sorted((ROOT / "docs").rglob("*.md")):
        rel = p.relative_to(ROOT / "docs").parts
        if rel and rel[0] in ("history", "archive"):
            continue
        out.append(p)
    return [p for p in out if p.exists()]


def check(p: pathlib.Path):
    probs = []
    lines = p.read_text(encoding="utf-8").split("\n")
    fence = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if not fence and "|" in ln and i + 1 < len(lines) and DELIM.match(lines[i + 1]) and "-" in lines[i + 1]:
            want = cells(ln)
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|") or (j == i + 1):
                n = cells(lines[j])
                if n != want:
                    kind = "delimiter row" if j == i + 1 else "row"
                    probs.append(f"{p.relative_to(ROOT)}:{j + 1}: table {kind} has {n} cells, header {want}")
                j += 1
            i = j
            continue
        if not fence:
            for m in LINK.finditer(ln):
                target = m.group(1) or m.group(2)
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if not path:
                    continue
                if not (p.parent / path).exists():
                    probs.append(f"{p.relative_to(ROOT)}:{i + 1}: broken relative link {target}")
        i += 1
    return probs


def main(argv):
    probs = [x for p in files(argv[1:]) for x in check(p)]
    for x in probs:
        print(x)
    print(f"check_docs: {len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
