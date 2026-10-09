"""Which GPU are we compiling for.

Detection is best-effort and never guesses: if no device can be identified the
caller must say `arch=` explicitly. A wrong `--offload-arch` does not fail at
compile time -- it fails at load time with "no kernel image is available", or
worse, on a close-enough arch, runs and computes the wrong thing. Guessing is
not an acceptable default here.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from typing import List, Optional

from ..target import get_target

_GFX_RE = re.compile(r"\b(gfx\d{3,4}[a-z]*)\b")


def _from_env() -> Optional[str]:
    for var in ("HK_ARCH", "HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH"):
        v = os.environ.get(var, "").strip()
        if v:
            # PYTORCH_ROCM_ARCH may be a semicolon list; the first entry is the
            # one a single-GPU box is actually running.
            return re.split(r"[;,\s]+", v)[0]
    return None


def _from_torch() -> Optional[str]:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return None
    m = _GFX_RE.search(name or "")
    return m.group(1) if m else None


def _from_rocminfo() -> Optional[str]:
    exe = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if not os.path.exists(exe):
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    # rocminfo lists the CPU agent first; take the last gfx name, which is a
    # GPU agent on every machine we have seen. Multi-GPU boxes with mixed archs
    # would need HK_ARCH.
    found: List[str] = _GFX_RE.findall(out)
    return found[-1] if found else None


@functools.lru_cache(maxsize=1)
def detect() -> Optional[str]:
    """The local GPU's gfx name, or None. Cached: probing costs a subprocess
    and the answer cannot change within a process."""
    for probe in (_from_env, _from_torch, _from_rocminfo):
        arch = probe()
        if arch:
            return arch
    return None


def resolve(arch: Optional[str]) -> str:
    """Normalise an explicit arch, or detect one. Raises rather than defaulting."""
    if arch is None:
        arch = detect()
    if arch is None:
        raise RuntimeError(
            "no GPU detected and no arch given. Pass arch= to @hk.kernel (or set "
            "HK_ARCH) -- hk will not guess, because a wrong --offload-arch fails "
            "at load time or, on a near-enough arch, computes the wrong answer."
        )
    return get_target(arch).arch
