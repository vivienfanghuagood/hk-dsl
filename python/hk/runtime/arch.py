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


def _entries(var: str) -> List[str]:
    return [e for e in re.split(r"[;,\s]+", os.environ.get(var, "").strip()) if e]


def _from_env() -> Optional[str]:
    """HK_ARCH, or a build-arch variable that names exactly one arch. A list -- the vLLM ROCm
    image sets PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100;... -- says what a build targets,
    not which GPU is here: its first entry made an MI355X gfx90a, and hk then built for gfx1100.
    A list is only the last resort (`_from_env_list`), after the device."""
    hk = _entries("HK_ARCH")
    if hk:
        return hk[0]
    for var in ("HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH"):
        e = _entries(var)
        if len(e) == 1:
            return e[0]
    return None


def _from_env_list() -> Optional[str]:
    for var in ("HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH"):
        e = _entries(var)
        if e:
            return e[0]
    return None


#: Where the KFD driver lists its nodes; a parameter for the tests.
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _from_kfd() -> Optional[str]:
    """The first GPU node's gfx name from the KFD topology in sysfs. Unlike `_from_torch` this
    creates no HIP context -- an engine plugin decides in every process, an API server's too."""
    import glob  # noqa: PLC0415

    def node(path: str) -> int:
        d = os.path.basename(os.path.dirname(path))
        return int(d) if d.isdigit() else 1 << 30

    for p in sorted(glob.glob(os.path.join(KFD_NODES, "*", "properties")), key=node):
        try:
            with open(p) as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == "gfx_target_version" and int(v) > 0:
                        v = int(v)
                        return f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"
        except (OSError, ValueError):
            continue
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
    for probe in (_from_env, _from_kfd, _from_torch, _from_rocminfo, _from_env_list):
        arch = probe()
        if arch:
            return arch
    return None


@functools.lru_cache(maxsize=None)
def device_arch(device: int) -> Optional[str]:
    """The gfx name the runtime reports for HIP device `device`, or None
    with no torch or no GPU. Not `detect()`: HK_ARCH says what to compile
    for, and this is what the kernel is about to run on."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_properties(device).gcnArchName
    except Exception:
        return None
    m = _GFX_RE.search(name or "")
    return m.group(1) if m else None


class ArchMismatchError(RuntimeError):
    """A kernel built for one arch was about to be loaded onto another."""


def check_device(arch: str, what: str, device: Optional[int] = None) -> None:
    """Raise ArchMismatchError if `device` (default: torch's current one) is
    not `arch`. A code object for another arch does not fail cleanly at
    launch: a gfx1100 kernel launched on a gfx950 crashed the process. Free
    after the first call per device; does nothing with no GPU."""
    if device is None:
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return
            device = torch.cuda.current_device()
        except Exception:
            return
    have = device_arch(device)
    if have is None or have == arch:
        return
    raise ArchMismatchError(
        f"{what} was built for {arch}, but GPU {device} is {have}. Build it "
        f"for {have}: pass arch='{have}' to the kernel, or leave arch unset "
        f"and set HK_ARCH={have} (or unset HK_ARCH to detect it).")


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
