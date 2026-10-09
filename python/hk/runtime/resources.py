"""The spill gate: parse -Rpass-analysis=kernel-resource-usage and refuse.

This is a *correctness* gate, not a performance report. Every kernel in
include/rdna3 hand-manages `s_waitcnt`; when such a kernel spills, the scratch
traffic reorders against waits the compiler does not know are ordering
constraints and the kernel computes the wrong answer -- silently, with no
diagnostic, at full speed. So `hk.compile()` does not hand back a spilling
kernel with a warning attached. It raises.

The interactive analysis tool is .claude/skills/kernel-resource-check (tables,
headroom, before/after diffs). This module is deliberately not that: it is the
few lines that must keep working inside a wheel where .claude/ does not exist,
so it parses the remarks itself rather than shelling out to a script that may
not be installed. The remark format is stable compiler output; the two agree
because they read the same three fields.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..target import Target

_NAME_RE = re.compile(r"remark:\s*Function Name:\s*(\S+)")
_FIELD_RE = re.compile(r"remark:\s+(.+?):\s*(-?\d+|True|False)\s*\[")

_FIELDS = {
    "VGPRs": "vgpr",
    "TotalSGPRs": "sgpr",
    "ScratchSize [bytes/lane]": "scratch",
    "Occupancy [waves/SIMD]": "occupancy",
    "SGPRs Spill": "sgpr_spill",
    "VGPRs Spill": "vgpr_spill",
    "LDS Size [bytes/block]": "lds",
}


class ResourceError(Exception):
    """A kernel that must not be used. Raised, never warned about."""


@dataclass
class KernelResources:
    name: str
    mangled: str
    vgpr: int = 0
    sgpr: int = 0
    scratch: int = 0
    vgpr_spill: int = 0
    sgpr_spill: int = 0
    lds: int = 0
    occupancy: Optional[int] = None
    #: Filled in when a Target is known.
    vgpr_rounded: Optional[int] = None
    occ_by_reg: Optional[int] = None
    vgpr_headroom: Optional[int] = None

    @property
    def spills(self) -> bool:
        return bool(self.scratch or self.vgpr_spill or self.sgpr_spill)

    def line(self) -> str:
        vg = str(self.vgpr)
        if self.vgpr_rounded and self.vgpr_rounded != self.vgpr:
            vg = f"{self.vgpr}->{self.vgpr_rounded}"
        occ = self.occ_by_reg if self.occ_by_reg is not None else self.occupancy
        hdrm = "" if self.vgpr_headroom is None else f" hdrm={self.vgpr_headroom}"
        return (
            f"{self.name}: vgpr={vg} occ={occ} lds={self.lds} "
            f"scratch={self.scratch} spill={self.vgpr_spill + self.sgpr_spill}{hdrm}"
        )


def _demangle(names: List[str]) -> Dict[str, str]:
    exe = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not exe or not names:
        return {n: n for n in names}
    try:
        out = subprocess.run(
            [exe], input="\n".join(names), capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
    except Exception:
        return {n: n for n in names}
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def parse(log: str, target: Optional[Target] = None) -> List[KernelResources]:
    """Build log -> one record per kernel instantiation.

    An empty result is an error, not an empty answer: an up-to-date target
    emits no remarks at all, which is indistinguishable from a clean build.
    hk always compiles into a fresh directory, so reaching here with nothing
    means the flag did not take.
    """
    raw: Dict[str, Dict[str, int]] = {}
    cur = None
    for line in log.splitlines():
        m = _NAME_RE.search(line)
        if m:
            cur = m.group(1)
            raw.setdefault(cur, {})
            continue
        m = _FIELD_RE.search(line)
        if m and cur is not None:
            key = _FIELDS.get(m.group(1).strip())
            if key:
                val = m.group(2)
                if val in ("True", "False"):
                    raw[cur][key] = int(val == "True")
                else:
                    raw[cur][key] = int(val)

    if not raw:
        raise ResourceError(
            "the build emitted no kernel-resource-usage remarks.\n"
            "  hk cannot certify a kernel it has no numbers for, and a spilling\n"
            "  kernel silently computes the wrong answer, so this is fatal rather\n"
            "  than skipped. Check that -Rpass-analysis=kernel-resource-usage\n"
            "  survived into the hipcc command line."
        )

    pretty = _demangle(list(raw))
    out = []
    for mangled, fields in raw.items():
        r = KernelResources(name=pretty[mangled], mangled=mangled, **fields)
        if target is not None and r.vgpr:
            r.vgpr_rounded = target.vgpr_rounded(r.vgpr)
            r.occ_by_reg = target.occupancy(r.vgpr)
            r.vgpr_headroom = target.vgpr_headroom(r.vgpr)
        out.append(r)
    out.sort(key=lambda r: -r.vgpr)
    return out


def gate(
    kernels: List[KernelResources],
    *,
    allow_scratch: bool = False,
    max_vgprs: Optional[int] = None,
    min_occupancy: Optional[int] = None,
) -> None:
    """Raise ResourceError if any instantiation is unusable.

    Every instantiation, not just the one being benchmarked -- reading only the
    shipped shape's line is how a regression hides in a build nobody times.
    """
    fails = []
    for r in kernels:
        if r.spills and not allow_scratch:
            fails.append(
                f"{r.name}: ScratchSize={r.scratch} B/lane, "
                f"spill vgpr={r.vgpr_spill} sgpr={r.sgpr_spill}"
            )
            continue
        if max_vgprs and r.vgpr > max_vgprs:
            fails.append(f"{r.name}: vgpr={r.vgpr} > max_vgprs={max_vgprs}")
        occ = r.occ_by_reg if r.occ_by_reg is not None else r.occupancy
        if min_occupancy and occ is not None and occ < min_occupancy:
            fails.append(f"{r.name}: occupancy={occ} < min_occupancy={min_occupancy}")

    if not fails:
        return
    raise ResourceError(
        "generated kernel rejected:\n"
        + "\n".join(f"  {f}" for f in fails)
        + "\n\nAll instantiations:\n"
        + "\n".join(f"  {r.line()}" for r in kernels)
        + "\n\nA spilling kernel is not slow, it is wrong: scratch traffic\n"
        "reorders against the s_waitcnt this library places by hand. Shrink the\n"
        "tile, drop a live value, or lower the warp count."
    )


def report(kernels: List[KernelResources]) -> str:
    return "\n".join(r.line() for r in kernels)
