"""Every HK_* environment variable the package reads, and every route switch, has exactly one row in
the two reference tables: docs/integration.md (engines, routes, the ops' plans) or docs/runtime.md
(the runtime, autotune, hk.tune, hk.profile)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TABLES = [REPO / "docs" / "integration.md", REPO / "docs" / "runtime.md"]
#: HK_* strings in the code that are not environment variables: C++ macros the generated source
#: defines, and vLLM's name for hk's attention backend.
NOT_ENV = {"HK_BF16_ROUND", "HK_BF16_STORE_ROUND", "HK_RT_HALF_SPLIT", "HK_MODULE_NAME", "HK_ATTN"}


def _read_by_code() -> set:
    names = set()
    for f in (REPO / "python" / "hk").rglob("*.py"):
        names |= set(re.findall(r"""["'](HK_[A-Z0-9_]+)["']""", f.read_text(encoding="utf-8")))
    return names - NOT_ENV


def _rows(md: Path) -> list:
    out = []
    for line in md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| ((?:`[A-Z0-9_]+`(?:, )?)+) \|", line)
        if m:
            out += re.findall(r"`([A-Z0-9_]+)`", m.group(1))
    return out


def test_each_variable_has_exactly_one_row():
    rows = [v for md in TABLES for v in _rows(md)]
    missing = sorted(v for v in _read_by_code() if v not in rows)
    twice = sorted({v for v in rows if rows.count(v) > 1})
    assert not missing, f"HK_* variables with no row in {[t.name for t in TABLES]}: {missing}"
    assert not twice, f"variables with more than one row: {twice}"
