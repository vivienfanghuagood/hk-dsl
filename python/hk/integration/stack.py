"""Which Triton a vLLM runs with, checked before anything is measured.

    python3 -m hk.integration.stack      # exit 0 matched, 1 not; prints the versions

vLLM 0.31 is measured on one stack: torch 2.13.0+rocm7.2 and the Triton
that torch pins, triton-rocm 3.7.1 (chapter 44,
docs/history/44-matched-stack.md). pip can put another Triton over it without a word. xgrammar,
one of vLLM 0.31's requirements, requires `triton` with no version;
triton-rocm is a different distribution name, so installing vLLM's
requirements after torch puts PyPI's newest `triton` (3.8.0) over
triton-rocm's files in the same `triton` package. That is how the engines'
venv ran Triton 3.8.0 from chapter 24 to chapter 38: Triton's decode kernel
up to 1.9x slower in vLLM's allocation, stock's request 0-2.5% slower, and
the performance gate's Triton arms measured against the wrong kernel.
Nothing failed; every number just moved.

`check()` names what does not match, in sentences; `enforce()` acts on them
by `HK_STACK_CHECK`:

    refuse   raise StackMismatch (`python -m hk.integration ab`'s default:
             a measurement on the wrong stack is not a measurement)
    warn     print them to stderr, loudly, once a process, and carry on
             (the vLLM plugin's default: a plugin that raises takes the
             engine's startup with it)
    off      say nothing

A vLLM this table does not list is not checked: its matched stack has not
been measured, so there is nothing to hold it to.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import re
import sys
from typing import Dict, List, Optional

#: vLLM series -> the stack its numbers were measured on. The Triton is the
#: one the torch pins (`Requires-Dist: triton-rocm==3.7.1`), the one chapters
#: 20-22 and 44 measured. vLLM 0.31's own ROCm image builds a third Triton
#: (ROCm/triton release/internal/3.8.x); nothing here has measured it.
MATCHED: Dict[str, Dict[str, str]] = {
    "0.31": {"torch": "2.13.0+rocm7.2", "triton": "3.7.1",
             "triton_dist": "triton-rocm"},
}

#: Distributions that install the `triton` package. Two of them at once is
#: the overwrite's signature: the second one's files over the first's.
TRITON_DISTS = ("triton", "triton-rocm", "pytorch-triton-rocm")

POLICIES = ("refuse", "warn", "off")

_WARNED = False


class StackMismatch(RuntimeError):
    """vLLM is about to run on a stack its numbers were not measured on."""


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _triton_imported() -> Optional[str]:
    """The version of the `triton` package Python would import -- what
    runs, whichever distribution's files those are. Read from the module if
    it is loaded, else from its `__init__.py` without importing it."""
    mod = sys.modules.get("triton")
    if mod is not None and getattr(mod, "__version__", None):
        return str(mod.__version__)
    try:
        spec = importlib.util.find_spec("triton")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    try:
        with open(spec.origin, encoding="utf-8") as f:
            m = re.search(r"""^__version__\s*=\s*['"]([^'"]+)['"]""", f.read(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def installed() -> Dict[str, object]:
    """What is installed, without importing torch, Triton or vLLM."""
    return {
        "vllm": _version("vllm"),
        "torch": _version("torch"),
        "triton": _triton_imported(),
        "triton_dists": {d: v for d in TRITON_DISTS
                         if (v := _version(d)) is not None},
        "python": sys.executable,
    }


def _base(v: Optional[str]) -> str:
    return (v or "").split("+")[0]


def series(vllm_version: Optional[str]) -> Optional[str]:
    """'0.31.0+rocm721' -> '0.31'."""
    m = re.match(r"(\d+)\.(\d+)", vllm_version or "")
    return f"{m.group(1)}.{m.group(2)}" if m else None


def check(found: Optional[Dict[str, object]] = None) -> List[str]:
    """What does not match vLLM's measured stack, one sentence each; [] when
    it matches or when no vLLM in `MATCHED` is installed."""
    found = installed() if found is None else found
    want = MATCHED.get(series(found.get("vllm")) or "")
    if want is None:
        return []
    v = found["vllm"]
    out = []
    triton = found.get("triton")
    if _base(triton) != want["triton"]:
        out.append(
            f"vLLM {v} imports Triton {triton or 'nothing'}, not the "
            f"{want['triton_dist']} {want['triton']} its torch pins and its numbers "
            f"were measured on")
    dists = dict(found.get("triton_dists") or {})
    if len(dists) > 1:
        out.append(
            "more than one distribution installs the `triton` package ("
            + ", ".join(f"{d} {ver}" for d, ver in sorted(dists.items()))
            + "): the last one installed wrote its files over the other's")
    if found.get("torch") != want["torch"]:
        out.append(f"vLLM {v} runs on torch {found.get('torch')}, not the "
                   f"{want['torch']} it was built and measured against")
    return out


def fix(found: Optional[Dict[str, object]] = None) -> str:
    """The commands that restore the matched Triton in this venv."""
    found = installed() if found is None else found
    want = MATCHED.get(series(found.get("vllm")) or "") or MATCHED["0.31"]
    py = found.get("python") or "python"
    extra = [d for d in (found.get("triton_dists") or {}) if d != want["triton_dist"]]
    lines = [f"{py} -m pip uninstall -y {d}" for d in sorted(extra)]
    lines.append(
        f"{py} -m pip install --no-deps --force-reinstall "
        f"'{want['triton_dist']}=={want['triton']}' "
        f"--index-url https://download.pytorch.org/whl/rocm7.2")
    return "\n".join(lines)


def policy(default: str) -> str:
    p = os.environ.get("HK_STACK_CHECK", "").strip().lower() or default
    if p in ("0", "no", "false"):
        p = "off"
    if p not in POLICIES:
        raise ValueError(f"HK_STACK_CHECK={p!r}: one of {', '.join(POLICIES)}")
    return p


def report(problems: List[str], where: str,
           found: Optional[Dict[str, object]] = None) -> str:
    found = installed() if found is None else found
    return (f"[hk] {where}: this is not the stack vLLM {found.get('vllm')} was "
            f"measured on ({found.get('python')}):\n"
            + "".join(f"  - {p}\n" for p in problems)
            + "  Restore it:\n"
            + "".join(f"    {line}\n" for line in fix(found).splitlines())
            + "  (HK_STACK_CHECK=warn to run anyway, =off to say nothing; "
              "docs/integration.md#supported-versions-and-the-contract-tests)")


def enforce(where: str, default: str = "warn",
            found: Optional[Dict[str, object]] = None) -> str:
    """Check, then act by HK_STACK_CHECK (or `default`). Returns a one-line
    note for a log: 'stack matched', 'stack not checked', or the mismatch."""
    global _WARNED
    p = policy(default)
    if p == "off":
        return "stack not checked (HK_STACK_CHECK=off)"
    found = installed() if found is None else found
    problems = check(found)
    if not problems:
        if MATCHED.get(series(found.get("vllm")) or "") is None:
            return f"stack not checked (no measured stack for vLLM {found.get('vllm')})"
        return (f"stack matched: vLLM {found['vllm']}, torch {found['torch']}, "
                f"Triton {found['triton']}")
    msg = report(problems, where, found)
    if p == "refuse":
        raise StackMismatch(msg)
    if not _WARNED:
        _WARNED = True
        print("\n" + "!" * 78 + "\n" + msg + "!" * 78, file=sys.stderr, flush=True)
    return "STACK MISMATCH: " + "; ".join(problems)


def main(argv=None) -> int:
    found = installed()
    for k, v in found.items():
        print(f"{k:13s} {v}")
    problems = check(found)
    if problems:
        print(report(problems, "hk.integration.stack", found))
        return 1
    print(enforce("hk.integration.stack", default="warn", found=found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
