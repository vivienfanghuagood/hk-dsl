"""Scoreboard rows from outside the scoreboard: `--cases` and $HK_SCOREBOARD_CASES.

Each path is a .py file, or a directory of them, that defines
`scoreboard_cases(tune) -> [(name, work, unit, reference, arms, footprint)]`
in the scoreboard's case format (scoreboard.py, "arms"); its group is the
module's NAME, or the file's stem. `python -m hk.integration new-op` writes
such a file. Torch-free, so the no-GPU tests can check the discovery.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List

ENV = "HK_SCOREBOARD_CASES"


def paths(spec: str = "") -> List[Path]:
    """The .py files `spec` (comma-separated) and $HK_SCOREBOARD_CASES name."""
    out = []
    for item in [s for s in (spec + "," + os.environ.get(ENV, "")).split(",") if s.strip()]:
        p = Path(item.strip())
        if p.is_dir():
            out += sorted(f for f in p.glob("*.py") if not f.name.startswith(("_", "test_")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"--cases {item}: no such file or directory")
    return out


def load(files: Iterable[Path]) -> Dict[str, Callable]:
    """{group: scoreboard_cases} for every file that defines one."""
    groups = {}
    for f in files:
        spec = importlib.util.spec_from_file_location(f"hk_cases_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "scoreboard_cases", None)
        if fn is None:
            raise AttributeError(f"{f} defines no scoreboard_cases(tune)")
        groups[str(getattr(mod, "NAME", f.stem))] = fn
    return groups
