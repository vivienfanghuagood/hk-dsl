"""What the engine contract tests share: signatures compared the way a
stand-in depends on them, and the call sites an identity rebind has to reach."""

from __future__ import annotations

import dis
import inspect
import sys
from typing import Callable, List, Set


def params(fn: Callable, drop_self: bool = False) -> List[inspect.Parameter]:
    ps = [p for p in inspect.signature(fn).parameters.values()
          if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return ps[1:] if drop_self and ps and ps[0].name == "self" else ps


def names(fn: Callable, drop_self: bool = False) -> List[str]:
    return [p.name for p in params(fn, drop_self)]


def assert_same_parameters(engine_fn: Callable, standin: Callable, what: str) -> None:
    """The stand-in replaces `engine_fn` and forwards to it positionally, so
    the two take the same parameters in the same order, and every one the
    engine lets a caller omit, the stand-in does too, with the same default."""
    e, s = params(engine_fn), params(standin)
    assert [p.name for p in e] == [p.name for p in s], (
        f"{what}: the engine's parameters changed.\n  engine:   "
        f"{[p.name for p in e]}\n  stand-in: {[p.name for p in s]}")
    for pe, ps in zip(e, s):
        if pe.default is not pe.empty:
            assert ps.default is not ps.empty and ps.default == pe.default, (
                f"{what}: `{pe.name}` defaults to {pe.default!r} in the engine "
                f"and {ps.default!r} in the stand-in")


def _loads_global(code, name: str) -> bool:
    if name not in code.co_names and not any(inspect.iscode(c) for c in code.co_consts):
        return False
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and ins.argval == name:
            return True
    return any(_loads_global(c, name) for c in code.co_consts if inspect.iscode(c))


def call_sites(name: str, prefix: str) -> Set[str]:
    """Imported modules under `prefix` with a function or method -- nested
    ones included -- that loads `name` as a module global, which is a call
    an identity rebind of that module's attribute redirects."""
    hits = set()
    for modname, mod in list(sys.modules.items()):
        if mod is None or not modname.startswith(prefix):
            continue
        try:
            members = list(vars(mod).values())
        except Exception:  # noqa: BLE001,S112 -- a lazy module that will not load
            continue
        for v in members:
            try:
                fns = list(vars(v).values()) if inspect.isclass(v) else [v]
            except Exception:  # noqa: BLE001,S112
                continue
            for f in fns:
                try:
                    f = getattr(f, "__func__", f)
                    code = getattr(f, "__code__", None)
                    if (code is not None and getattr(f, "__module__", None) == modname
                            and _loads_global(code, name)):
                        hits.add(modname)
                except Exception:  # noqa: BLE001,S112
                    continue
    return hits


def assert_rebind_reaches_every_call_site(name: str, prefix: str, original) -> Set[str]:
    sites = call_sites(name, prefix)
    assert sites, f"nothing under {prefix} calls `{name}` as a global any more"
    stale = sorted(m for m in sites if vars(sys.modules[m]).get(name) is not original)
    assert not stale, (
        f"{stale} call `{name}` through a binding that is not the original "
        f"object, so a rebind by identity would miss them")
    return sites
