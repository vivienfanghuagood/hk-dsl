#!/usr/bin/env python3
"""Check that a wheel's binaries run on the glibc its manylinux tag promises.

    python3 scripts/check_wheel_glibc.py dist/manylinux/*.whl

For each wheel: the symbol versions every ELF file in it requires (the version
needs, as `readelf -V` lists them), the highest GLIBC_2.N among them, and the
manylinux_2_M tags in the wheel's file name, which is what pip goes by. It fails
when N > M: such a wheel installs on a glibc-2.M system and then fails at import
(0.3.1-0.7.0's cp312 wheels needed GLIBC_2.38 and were rightly tagged 2_38, so
they did not install in the vLLM images, glibc 2.35; a wrong tag would have
installed and broken). A wheel tagged only linux_* promises nothing and passes
with its requirement printed.

When auditwheel is importable (with the same interpreter), every other required
version of a family its policy lists -- GLIBCXX, CXXABI, GCC -- is checked
against the tag's policy too: GLIBCXX_3.4.32 is gcc 13's libstdc++ and not in
manylinux_2_35's, whatever glibc says. Exit status 1 if any wheel fails, 2 on
usage errors. Needs readelf (binutils).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# the legacy names, as PEP 600 maps them
_LEGACY = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}


def wheel_tags(name: str) -> list[tuple[int, int]]:
    """The (2, M) of each manylinux tag in a wheel file name; [] for none."""
    stem = Path(name).name
    if not stem.endswith(".whl"):
        raise ValueError(f"{name}: not a .whl")
    platform = stem[: -len(".whl")].split("-")[-1]
    tags = []
    for tag in platform.split("."):
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_\w+", tag)
        if m:
            tags.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.fullmatch(r"(manylinux\d+)_\w+", tag)
        if m:
            if m.group(1) not in _LEGACY:
                raise ValueError(f"{name}: unknown manylinux tag {tag}")
            tags.append(_LEGACY[m.group(1)])
    return tags


def version_needs(path: Path) -> set[tuple[str, str]]:
    """(library, version name) pairs an ELF file requires, from .gnu.version_r."""
    out = subprocess.run(["readelf", "-V", "-W", str(path)], check=True,
                         capture_output=True, text=True).stdout
    needs, in_needs, lib = set(), False, None
    for line in out.splitlines():
        if line.startswith("Version needs section"):
            in_needs = True
            continue
        if line.startswith("Version") and "section" in line:
            in_needs = False
            continue
        if not in_needs:
            continue
        m = re.search(r"File:\s*(\S+)", line)
        if m:
            lib = m.group(1)
            continue
        m = re.search(r"Name:\s*(\S+)", line)
        if m:
            needs.add((lib, m.group(1)))
    return needs


def _glibc(version: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"GLIBC_(\d+)\.(\d+)(?:\.\d+)?", version)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _policies() -> dict[tuple[int, int], dict[str, set[str]]] | None:
    """auditwheel's x86_64 symbol-version policy per manylinux_2_M, if it is installed."""
    try:
        import auditwheel
    except ImportError:
        return None
    path = Path(auditwheel.__file__).parent / "policy" / "manylinux-policy.json"
    if not path.is_file():
        return None
    out = {}
    for p in json.loads(path.read_text()):
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)", p["name"])
        if m and "x86_64" in p.get("symbol_versions", {}):
            out[(int(m.group(1)), int(m.group(2)))] = {
                fam: set(vers) for fam, vers in p["symbol_versions"]["x86_64"].items()}
    return out


def check(wheel: str, policies=None) -> list[str]:
    """The problems with one wheel ([] if none); prints what it found."""
    tags = wheel_tags(wheel)
    needs: dict[str, set[tuple[str, str]]] = {}
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        for info in z.infolist():
            if info.is_dir():
                continue
            with z.open(info) as f:
                if f.read(4) != b"\x7fELF":
                    continue
            path = Path(z.extract(info, tmp))
            needs[info.filename] = version_needs(path)
    if not needs:
        print(f"{Path(wheel).name}: no ELF files")
        return []
    every = set().union(*needs.values())
    glibc = [(v, f) for f, ns in needs.items() for _, n in ns if (v := _glibc(n))]
    top = max(glibc)[0] if glibc else (0, 0)
    who = sorted({f for v, f in glibc if v == top})
    print(f"{Path(wheel).name}: {len(needs)} ELF files, max GLIBC_{top[0]}.{top[1]} "
          f"({', '.join(Path(f).name for f in who)}); tags "
          + (", ".join(f"manylinux_{a}_{b}" for a, b in tags) or "none (linux_*)"))
    problems = []
    for tag in tags:
        if top > tag:
            problems.append(f"needs GLIBC_{top[0]}.{top[1]} but is tagged manylinux_{tag[0]}_{tag[1]}")
        if policies is None:
            continue
        allowed = policies.get(tag)
        if allowed is None:
            problems.append(f"auditwheel has no policy for manylinux_{tag[0]}_{tag[1]}")
            continue
        # numbered GLIBC versions are the check above; the rest by the policy
        bad = sorted({n for _, n in every
                      if "_" in n and not _glibc(n) and n.split("_", 1)[0] in allowed
                      and n.split("_", 1)[1] not in allowed[n.split("_", 1)[0]]})
        if bad:
            problems.append(f"manylinux_{tag[0]}_{tag[1]}'s policy does not allow {', '.join(bad)}")
    for p in problems:
        print(f"  FAIL: {p}")
    return problems


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    if shutil.which("readelf") is None:
        print("readelf not found (binutils)", file=sys.stderr)
        return 2
    policies = _policies()
    if policies is None:
        print("(auditwheel not importable: GLIBC only, no GLIBCXX / CXXABI / GCC policy check)")
    failed = [w for w in argv if check(w, policies)]
    if failed:
        print(f"{len(failed)} of {len(argv)} wheels FAIL")
        return 1
    print(f"all {len(argv)} wheels OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
