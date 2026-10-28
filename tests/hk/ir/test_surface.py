"""The surface registry (python/hk/dsl/surface.py; docs/design/dsl-surface.md §4) covers the
language exactly: every documented name, every dialect op and every C++ helper in include/hk/ is
registered with a kind, and every use of an escape hatch in hk.ops, the templates and the
examples is in a file scripts/escape_hatches_allow.txt lists, within the line's ceiling. A failure
here means something was added without the registry -- see CONTRIBUTING.md: a new kernel adds no
kernel-specific helper or one-off op, and a needed one is reviewed and registered as an escape
hatch first. Both ratchet: a removal (a registered name gone, uses under a ceiling) is a hint to
tighten them, never a failure, so a change that only removes never conflicts with them."""

from __future__ import annotations

import ast
import importlib.util
import re
import warnings
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOW = ("register it in python/hk/dsl/surface.py (a general primitive in PRIMITIVES / OPS / CPP; "
       "anything kernel-specific only after review, CONTRIBUTING.md)")


def _hint(msg: str) -> None:
    """A hint, not a failure: in pytest's warnings summary (and on stdout with -s)."""
    warnings.warn(msg, UserWarning, stacklevel=2)
    print(msg)


def _surface():
    spec = importlib.util.spec_from_file_location("hk_surface", REPO / "python/hk/dsl/surface.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _surface()

#: The API pages of the language (not the library: hk.ops, templates, schedules, tune, ...).
PAGES = ("hk.md", "hk.arch.md", "hk.lds.md", "hk.group.md", "hk.collective.md", "hk.comm.md",
         "hk.math.md", "hk.placement.md")


def documented():
    """{name: its signature block} for every heading of the language's API pages, and the
    names listed under "Also exported"."""
    out = {}
    for page in PAGES:
        text = (REPO / "docs" / "api" / page).read_text(encoding="utf-8")
        parts = re.split(r"^#{3,4} `([^`]+)`", text, flags=re.M)
        for name, body in zip(parts[1::2], parts[2::2]):
            out[name.replace(" (property)", "")] = body.split("\n#", 1)[0]
        also = re.search(r"^## Also exported\n\n(.*)$", text, flags=re.M)
        if also:
            out.update({n: "" for n in re.findall(r"`([^`]+)`", also.group(1))})
    return out


def test_every_documented_name_is_registered():
    names = documented()
    missing = sorted(n for n in names if S.kind_of(n) is None)
    assert not missing, f"documented but not in the surface registry: {missing}; {HOW}"


def test_the_registry_names_nothing_that_is_gone():
    names = documented()
    stale = sorted(n for n in S.PRIMITIVES if n not in names)
    for name in S.ENTRIES:
        base, _, kw = name.partition("(")
        if base not in names:
            stale.append(name)
        elif kw and not re.search(rf"\b{re.escape(kw.rstrip('=)'))}\b", names[base]):
            stale.append(name)           # the keyword is no longer in the signature
    # a removal is not a failure (the registry ratchets like the allowlist): a hint to drop them
    if stale:
        _hint(f"registered but not documented (removed? renamed?), drop them from "
              f"python/hk/dsl/surface.py: {sorted(stale)}")


def test_entries_are_well_formed():
    for name, e in S.ENTRIES.items():
        assert e.kind in S.KINDS, name
        assert name not in S.PRIMITIVES, f"{name}: both an ENTRY and a primitive"
        if e.kind in ("escape", "experiment", "format"):
            assert e.boundary, f"{name}: an {e.kind} states its boundary"
        if e.kind in ("merge", "delete", "deprecated"):
            assert e.target and e.step, f"{name}: a {e.kind} names its target and step"


def _dialect_ops():
    td = (REPO / "mlir/include/hk/Dialect/HK/IR/HKOps.td").read_text(encoding="utf-8")
    return re.findall(r'^def HK_\w+\s*:\s*HK_\w*Op<"([\w.]+)"', td, flags=re.M)


def test_every_dialect_op_is_registered():
    ops = _dialect_ops()
    assert len(ops) == len(set(ops))
    missing = sorted(set(ops) - set(S.OPS))
    stale = sorted(set(S.OPS) - set(ops))
    assert not missing, f"dialect ops not in surface.OPS: {missing}; {HOW}"
    if stale:
        _hint(f"surface.OPS names ops the dialect no longer has (drop them): {stale}")
    assert set(S.OPS.values()) <= set(S.KINDS)


def cpp_names(path: Path):
    """The functions a header defines: the name before the parameter list of each __device__
    definition (comments stripped). Macro-made names are seen through their macros' bodies."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return set(re.findall(r"__device__[^;{}()]*?\b([A-Za-z_]\w*)\s*\(", text))


def test_every_cpp_helper_is_registered():
    headers = sorted((REPO / "include" / "hk").rglob("*.cuh"))
    umbrella = {h for h in headers if not cpp_names(h)}     # only includes (hk/cdna.cuh)
    headers = [h for h in headers if h not in umbrella]
    have = {h.relative_to(REPO).as_posix() for h in headers}
    assert have <= set(S.CPP), f"headers in include/hk/ not in surface.CPP: {sorted(have - set(S.CPP))}"
    if set(S.CPP) - have:
        _hint(f"surface.CPP lists headers that are gone (drop them): {sorted(set(S.CPP) - have)}")
    for h in headers:
        rel = h.relative_to(REPO).as_posix()
        names, reg = cpp_names(h), S.CPP[rel]
        missing, stale = sorted(names - set(reg)), sorted(set(reg) - names)
        assert not missing, f"{rel}: C++ helpers not in surface.CPP: {missing}; {HOW}"
        if stale:
            _hint(f"{rel}: surface.CPP names helpers the header no longer has (drop them): {stale}")
        assert set(reg.values()) <= set(S.KINDS)


#: How a use of each escape hatch or experiment is recognized in source.
USES = {
    "hk.map_cpp": r"\bmap_cpp\(",
    "hk.Granules": r"\bhk\.Granules\b",
    "SharedTile.granules": r"\.granules\(",
    "Granules.load_frag": r"\.load_frag\(",
    "Granules.store_frag": r"\.store_frag\(",
    "hk.arch.agpr": r"\barch\.agpr\(",
    "Kernel.with_defines": r"\bwith_defines\b",
    "Kernel.build_flags": r"\bbuild_flags\b",
}


def _scanned():
    roots = (REPO / "python/hk/ops", REPO / "python/hk/templates", REPO / "examples")
    return sorted(p for r in roots for p in r.rglob("*.py"))


def _allowlist():
    allowed = {}
    for line in (REPO / "scripts/escape_hatches_allow.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path, name, ceiling, reason = line.split(None, 3)
        assert reason.strip(), f"no reason given: {line}"
        assert (path, name) not in allowed, f"listed twice: {path} {name}"
        allowed[(path, name)] = int(ceiling)
    return allowed


def _rewriting_subclasses(path: Path):
    """Classes in `path` that override Kernel.with_defines or Kernel.source."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef):
            methods = {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
            if methods & {"with_defines", "source"}:
                found.append(node.name)
    return found


def _uses():
    """{(file, escape hatch or rewriting Kernel subclass): uses} in hk.ops, the templates and
    the examples."""
    hatches = {n for n, e in S.ENTRIES.items() if e.kind in ("escape", "experiment")}
    assert hatches <= set(USES), f"no use pattern for {sorted(hatches - set(USES))}"
    seen = Counter()
    for path in _scanned():
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text(encoding="utf-8")
        for name in hatches:
            n = len(re.findall(USES[name], text))
            if n:
                seen[(rel, name)] = n
        for cls in _rewriting_subclasses(path):
            seen[(rel, cls)] += 1
    return seen


def check_ceilings(seen, allowed):
    """The allowlist is a ratchet: each (file, name) must be listed, and its uses may not exceed
    the line's ceiling. Returns (unlisted, over, below): the first two fail; `below` --
    {(file, name): (uses, ceiling)} for a line above its uses, or listed with none -- is a hint
    to tighten the line, not a failure, so a change that removes uses never conflicts with the
    table, and one that adds them always edits it (the review point)."""
    unlisted = {k: v for k, v in seen.items() if k not in allowed}
    over = {k: (v, allowed[k]) for k, v in seen.items() if k in allowed and v > allowed[k]}
    below = {k: (seen.get(k, 0), c) for k, c in allowed.items() if seen.get(k, 0) < c}
    return unlisted, over, below


def test_escape_hatch_uses_are_within_their_ceilings():
    unlisted, over, below = check_ceilings(_uses(), _allowlist())
    assert not unlisted, (f"escape-hatch uses in a file scripts/escape_hatches_allow.txt does not "
                          f"list: {unlisted}. Each needs review and a line with its reason and a "
                          f"ceiling (CONTRIBUTING.md).")
    assert not over, (f"escape-hatch uses above their line's ceiling, (uses, ceiling): {over}. "
                      f"Raising a ceiling is a reviewed change to scripts/escape_hatches_allow.txt "
                      f"(CONTRIBUTING.md).")
    for (path, name), (n, ceiling) in sorted(below.items()):
        _hint(f"escape_hatches_allow.txt: {path} {name} has {n} uses under its ceiling {ceiling}: "
              + ("delete the line" if n == 0 else f"tighten the ceiling to {n}"))


def test_the_ceiling_check_fails_on_new_files_and_growth_and_hints_on_a_drop():
    allowed = {("a.py", "hk.map_cpp"): 10, ("b.py", "hk.map_cpp"): 3, ("c.py", "Sub"): 1}
    same = {("a.py", "hk.map_cpp"): 10, ("b.py", "hk.map_cpp"): 3, ("c.py", "Sub"): 1}
    assert check_ceilings(same, allowed) == ({}, {}, {})
    # a file not listed: fails, at any count
    new_file = {**same, ("d.py", "hk.map_cpp"): 1}
    assert check_ceilings(new_file, allowed)[0] == {("d.py", "hk.map_cpp"): 1}
    # over the ceiling: fails
    grew = {**same, ("a.py", "hk.map_cpp"): 11}
    assert check_ceilings(grew, allowed)[:2] == ({}, {("a.py", "hk.map_cpp"): (11, 10)})
    # under it, or gone (a rewriting subclass removed too): passes, with a hint each
    dropped = {("a.py", "hk.map_cpp"): 7, ("b.py", "hk.map_cpp"): 3}
    assert check_ceilings(dropped, allowed) == (
        {}, {}, {("a.py", "hk.map_cpp"): (7, 10), ("c.py", "Sub"): (0, 1)})
