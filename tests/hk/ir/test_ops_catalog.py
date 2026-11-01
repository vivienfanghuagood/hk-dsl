"""The op catalog (python/hk/ops/catalog.py, rendered as docs/ops.md) is held to the code: every
public name of hk.ops is in a family, every entry point exists, every route is one the switches
know and every route that runs an hk.ops kernel is in some family, every test file exists, and a
family behind hk.ops._arch.require lists exactly the GPUs that check allows. Reads the sources
with ast: no hk._mlir, no GPU."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPS = REPO / "python" / "hk" / "ops"


def _catalog():
    spec = importlib.util.spec_from_file_location("hk_ops_catalog", OPS / "catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen():
    spec = importlib.util.spec_from_file_location("gen_hardware_docs", REPO / "scripts" / "gen_hardware_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _all(path: Path):
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            return [e.value for e in node.value.elts]
    raise AssertionError(f"no __all__ in {path}")


def _defined(module: str):
    """Top-level functions of hk.ops.<module>, and the names its __init__ re-exports."""
    path = OPS / f"{module}.py"
    return {n.name for n in _tree(path).body if isinstance(n, ast.FunctionDef)} if path.exists() else set()


def test_every_public_name_of_hk_ops_is_in_a_family():
    cat = _catalog()
    listed = {e for f in cat.FAMILIES for e in f["entries"]}
    modules = {p.stem for p in OPS.glob("*.py")}
    missing = [n for n in _all(OPS / "__init__.py") if n not in modules and f"hk.ops.{n}" not in listed]
    assert not missing, f"hk.ops names in no catalog family: {missing}"


def test_every_entry_point_exists():
    cat = _catalog()
    init = set(_all(OPS / "__init__.py"))
    bad = []
    for f in cat.FAMILIES:
        for e in f["entries"]:
            parts = e.split(".")
            assert parts[:2] == ["hk", "ops"], e
            if len(parts) == 3:
                ok = parts[2] in init
            else:
                ok = parts[3] in _defined(parts[2])
            if not ok:
                bad.append(e)
    assert not bad, f"catalog entries that do not exist: {bad}"


def test_routes_and_tests_exist_and_every_kernel_route_has_a_family():
    cat = _catalog()
    _, switches, _ = _gen()._load()
    used = set()
    for f in cat.FAMILIES:
        assert set(f["arches"]) <= set(cat.ARCHES), f["name"]
        for r in f["routes"]:
            assert r in switches.ROUTES, (f["name"], r)
            used.add(r)
        for t in f["tests"]:
            assert (REPO / "tests" / "hk" / "gpu" / t).exists(), (f["name"], t)
    # every route the switches know runs one of these families
    assert set(switches.ROUTES) == used, f"routes in no family: {sorted(set(switches.ROUTES) - used)}"


def test_a_family_behind_arch_require_lists_what_it_allows():
    cat = _catalog()
    tree = _tree(OPS / "_arch.py")
    allowed = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "RDNA_AND_GFX950")
    assert tuple(cat.RDNA_AND_GFX950) == tuple(allowed)
    # module -> family, for the modules whose entry points call _arch.require with its default
    behind = {"gemm": "gemm", "decode_gemm": "decode_gemm", "w4a16": "w4a16", "moe": "moe",
              "attn": "attention", "paged": "paged", "paged_prefill": "paged_prefill"}
    fam = {f["name"]: f for f in cat.FAMILIES}
    for module, family in behind.items():
        assert "_arch.require(" in (OPS / f"{module}.py").read_text(encoding="utf-8"), module
        assert tuple(fam[family]["arches"]) == tuple(allowed), family


def test_the_page_is_current():
    gen = _gen()
    assert gen.OPS_OUT.read_text(encoding="utf-8") == gen.ops_page(), \
        "docs/ops.md is stale: run `python3 scripts/gen_hardware_docs.py`"
