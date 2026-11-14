"""What hk's engine plugins support, checked before they touch the engine.

An engine hk was never measured on gets hk fully off: one line says what is unsupported and what
is, and the plugin imports nothing else (no route module, no engine module, no aiter), so the
engine starts exactly as stock. Decided from package metadata, the ROCm install's version file and
the GPU's arch from sysfs -- nothing here imports torch or the engine.

The ranges are the stacks hk's routes were measured on (docs/support-matrix.md). A vLLM built
from source without a release tag reports a version like 0.1.dev1+g<sha>; it has no number to
check, so it is allowed and said so (each route's engine contract still checks it).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

#: engine -> lowest supported release (inclusive). vLLM 0.19.2 is the oldest measured stack.
#: SGLang has none: the sglang-radeon build reports 0.0.0, so its routes' engine contracts decide.
ENGINE_MIN: Dict[str, Tuple[int, int]] = {"vllm": (0, 19)}
TORCH_MIN = (2, 9)
ROCM_MIN = (7, 2)
ARCHES = ("gfx1100", "gfx1201", "gfx950")


def _version(dist: str) -> Optional[str]:
    import importlib.metadata as md  # noqa: PLC0415

    try:
        return md.version(dist)
    except Exception:  # noqa: BLE001
        return None


def _tuple(v: Optional[str]) -> Optional[Tuple[int, int]]:
    m = re.match(r"(\d+)\.(\d+)", v or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def rocm_version() -> Optional[str]:
    """The ROCm install's version (ROCM_PATH, else /opt/rocm): its .info/version file."""
    root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    try:
        with open(os.path.join(root, ".info", "version")) as fh:
            return fh.read().strip().split("-")[0]
    except OSError:
        return None


def source_build(v: Optional[str]) -> bool:
    """A source build without a release tag: setuptools-scm's 0.1.devN+g<sha>."""
    return bool(v) and re.match(r"0\.1\.dev\d", v) is not None


def check(engine: str, arch: Optional[str]) -> Tuple[List[str], Dict[str, str]]:
    """(problems, versions). problems: one phrase per unsupported piece; [] = supported."""
    dist = {"vllm": "vllm", "sglang": "sglang"}.get(engine, engine)
    ev, tv, rv = _version(dist), _version("torch"), rocm_version()
    seen = {dist: ev or "not installed", "torch": tv or "not installed", "ROCm": rv or "unknown",
            "arch": arch or "no GPU"}
    probs = []
    want = ENGINE_MIN.get(engine)
    if want and ev and not source_build(ev) and (_tuple(ev) or (0, 0)) < want:
        probs.append(f"{dist} {ev} (supported: {dist} >= {want[0]}.{want[1]})")
    if tv and (_tuple(tv) or (0, 0)) < TORCH_MIN:
        probs.append(f"torch {tv} (supported: torch >= {TORCH_MIN[0]}.{TORCH_MIN[1]})")
    if rv and (_tuple(rv) or (0, 0)) < ROCM_MIN:
        probs.append(f"ROCm {rv} (supported: ROCm >= {ROCM_MIN[0]}.{ROCM_MIN[1]})")
    if arch not in ARCHES:
        probs.append(f"GPU {arch or 'none found'} (supported: {', '.join(ARCHES)})")
    return probs, seen


def off_line(engine_name: str, problems: List[str], pid: int) -> str:
    """The one line an unsupported stack gets."""
    return (f"[hk] {engine_name}: hk is OFF, this stack is not supported -- {'; '.join(problems)}. "
            f"The engine runs as stock; nothing of hk was loaded (pid {pid}). "
            f"Supported stacks: docs/support-matrix.md")


def summary(engine_name: str, version: str, seen: Dict[str, str], on: List[str],
            off: List[str], env_of: Dict[str, str], pid: int,
            held: Optional[List[Tuple[str, str]]] = None) -> List[str]:
    """The startup summary of an enabled plugin: what runs, and how to turn it off."""
    stack = ", ".join(f"{k} {v}" for k, v in seen.items())
    lines = [f"[hk] ===== hk-dsl {version} in {engine_name} ({stack}) (pid {pid}) =====",
             f"[hk] {engine_name} routes on: {', '.join(on) if on else 'none'}"
             + (f"; OFF: {', '.join(off)}" if off else "")]
    switches_ = ", ".join(f"{env_of[r]}=0" for r in on if env_of.get(r))
    lines.append(f"[hk] turn hk off: HK_DISABLE=1; one route off: {switches_ or 'n/a'}")
    if held:
        lines.append("[hk] off by default here: " + "; ".join(f"{n} ({why})" for n, why in held))
    return lines
