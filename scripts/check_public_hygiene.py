#!/usr/bin/env python3
"""Check the tracked files for internal addresses, host names and user paths.

The repository is public. Private IPv4 addresses, internal host names, user
home directories and paths in someone's workspace belong to the machines'
configuration (a runner's .env, a local note), not to it. This scans every
file git tracks (the submodule excluded) and fails on a hit that the
allowlist does not name:

    python3 scripts/check_public_hygiene.py                 # exit 1 on a hit
    python3 scripts/check_public_hygiene.py --show-allowed  # also list the allowed hits
    python3 scripts/check_public_hygiene.py --patterns FILE # add rules of your own

The built-in rules are generic (they name no machine): private and shared
IPv4 ranges, /home/<user>, /root/, /dev/shm/<name>, internal-only domain
suffixes. The names of a particular site cannot be listed here without
publishing them, so a maintainer keeps them in a local file, one regular
expression per line (a line starting with '#' is a comment), passed with --patterns or in
HK_HYGIENE_PATTERNS (paths separated by ':'). CI runs the built-in rules;
the release gate (docs/releasing.md) runs them with the local file too.

The allowlist (scripts/public_hygiene_allow.txt) has one entry per line:
a path glob, whitespace, and a regular expression the whole matched text
must match; a line starting with '#' is a comment. Keep it short and say why in a comment.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = ROOT / "scripts" / "public_hygiene_allow.txt"
SELF = {"scripts/check_public_hygiene.py", "scripts/public_hygiene_allow.txt"}  # rules, not hits

_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
RULES: dict[str, str] = {
    # RFC 1918 and the shared (CGNAT) range; loopback and public addresses are fine.
    "private-ipv4": (
        r"(?<![\d.])(?:10\.{o}|172\.(?:1[6-9]|2\d|3[01])|192\.168|"
        r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.{o}\.{o}(?![\d])"
        .replace("{o}", _OCT)
    ),
    "home-path": r"(?<![\w.])/home/[A-Za-z0-9_.-]+",
    "root-path": r"(?<![\w.])/root/[^\s`'\")]+",
    "shm-path": r"(?<![\w.])/dev/shm/[A-Za-z0-9_.-]+",
    "internal-domain": r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:internal|corp|intranet|localdomain|lan)\b",
}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True,
                         capture_output=True).stdout
    files = []
    for name in out.decode().split("\0"):
        p = ROOT / name
        if name and name not in SELF and p.is_file() and not p.is_symlink():   # a submodule is a directory
            files.append(p)
    return files


def load_patterns(paths: list[str]) -> dict[str, str]:
    rules = {}
    for path in paths:
        for n, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.strip()
            if line and not line.startswith("#"):
                rules[f"local:{Path(path).name}:{n}"] = line
    return rules


def load_allow(path: Path) -> list[tuple[str, re.Pattern]]:
    allow = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            sys.exit(f"{path}:{n}: want 'GLOB REGEX'")
        allow.append((parts[0], re.compile(parts[1])))
    return allow


def scan(files, rules, allow):
    compiled = {k: re.compile(v) for k, v in rules.items()}
    hits, allowed = [], []
    for p in files:
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for rule, rx in compiled.items():
                for m in rx.finditer(line):
                    text = m.group(0)
                    ok = any(fnmatch.fnmatch(rel, g) and r.fullmatch(text) for g, r in allow)
                    (allowed if ok else hits).append((rel, lineno, rule, text))
    return hits, allowed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--patterns", action="append", default=[],
                    help="a file of extra regular expressions (one a line)")
    ap.add_argument("--allow", default=str(ALLOW), help="the allowlist (default: %(default)s)")
    ap.add_argument("--show-allowed", action="store_true", help="list the allowed hits too")
    args = ap.parse_args(argv)
    extra = args.patterns + [p for p in os.environ.get("HK_HYGIENE_PATTERNS", "").split(":") if p]
    rules = dict(RULES)
    rules.update(load_patterns(extra))
    hits, allowed = scan(tracked_files(), rules, load_allow(Path(args.allow)))
    if args.show_allowed:
        for rel, n, rule, text in allowed:
            print(f"allowed {rel}:{n}: [{rule}] {text}")
    for rel, n, rule, text in hits:
        print(f"{rel}:{n}: [{rule}] {text}")
    local = f", {len(rules) - len(RULES)} local" if extra else ""
    print(f"public hygiene: {len(hits)} hit(s), {len(allowed)} allowed "
          f"({len(RULES)} built-in rules{local})", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
