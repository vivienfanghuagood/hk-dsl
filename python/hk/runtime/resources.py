"""The spill gate: parse -Rpass-analysis=kernel-resource-usage and refuse.

This is a *correctness* gate, not a performance report. Every kernel in
include/rdna3 hand-manages `s_waitcnt`; when such a kernel spills, the scratch
traffic reorders against waits the compiler does not know are ordering
constraints and the kernel computes the wrong answer -- silently, with no
diagnostic, at full speed. So `hk.compile()` does not hand back a spilling
kernel with a warning attached. It raises.

The interactive resource analysis (tables, headroom, before/after diffs) is a
separate maintainers' tool. This module is deliberately not that: it is the few
lines that must keep working inside a wheel, where that tool does not exist, so
it parses the remarks itself rather than shelling out to a script that may not
be installed. The two agree because they read the same three fields.

The remarks come in two shapes, and `parse` reads both. A compile that runs the
device code generator itself -- every hipcc through clang 23, and clang 24's
device-only `--genco` -- prints clang diagnostics:

    k.cpp:3:1: remark: Function Name: _Z5scalePff [-Rpass-analysis=kernel-resource-usage]
    k.cpp:3:1: remark:     VGPRs: 2 [-Rpass-analysis=kernel-resource-usage]

clang 24 (TheRock, ROCm 10.1) builds a host+device `-shared` module with the new
offload driver: the device code is generated at link time, by
clang-linker-wrapper, which prints the same remarks bare:

    k.cpp:3:0: Function Name: _Z5scalePff
    k.cpp:3:0:     VGPRs: 2

A bare line is only taken as a remark when it starts with a `file:line:col:`
location and names one of the known keys, so no other diagnostic can be read
as one. Captured outputs of clang 22, 23 and 24 are in tests/hk/remark_fixtures.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..target import Target

_FIELDS = {
    "VGPRs": "vgpr",
    "AGPRs": "agpr",
    "TotalSGPRs": "sgpr",
    "ScratchSize [bytes/lane]": "scratch",
    "Occupancy [waves/SIMD]": "occupancy",
    "SGPRs Spill": "sgpr_spill",
    "VGPRs Spill": "vgpr_spill",
    "LDS Size [bytes/block]": "lds",
}

#: The remark's text after its location: `Function Name: <mangled>`, or one of
#: the kernel's fields (`Dynamic Stack` is printed but not kept).
_BODY = (r"\s*(?:Function Name:\s*(?P<name>\S+)"
         r"|(?P<key>" + "|".join(re.escape(k) for k in [*_FIELDS, "Dynamic Stack"]) + r")"
         r":\s*(?P<val>-?\d+|True|False))")
#: clang's diagnostic: `<loc>: remark: <body> [-Rpass-analysis=kernel-resource-usage]`.
_REMARK_RE = re.compile(r"remark:" + _BODY + r"\s*\[-Rpass-analysis[^\]]*\]\s*$")
#: clang-linker-wrapper's (clang 24, -shared): `<file>:<line>:<col>: <body>` and nothing after.
_BARE_RE = re.compile(r"^\S[^\n]*?:\d+:\d+:" + _BODY + r"\s*$")


class ResourceError(Exception):
    """A kernel that must not be used. Raised, never warned about."""


@dataclass
class KernelResources:
    name: str
    mangled: str
    vgpr: int = 0
    #: Accumulation registers. Zero on RDNA; on CDNA they come out of the
    #: VGPRs' file and are charged with them (Target.charged_vgprs).
    agpr: int = 0
    sgpr: int = 0
    scratch: int = 0
    vgpr_spill: int = 0
    sgpr_spill: int = 0
    #: Set by hk.runtime.compile when the code object shows a ScratchSize > 0 that nothing
    #: touches (cocheck.unused_scratch_frame): why the gate lets that scratch through.
    scratch_note: str = ""
    lds: int = 0
    occupancy: Optional[int] = None
    #: Filled in when a Target is known.
    vgpr_rounded: Optional[int] = None
    occ_by_reg: Optional[int] = None
    vgpr_headroom: Optional[int] = None

    @property
    def spills(self) -> bool:
        # An SGPR spill with ScratchSize 0 went to VGPR lanes (v_writelane / v_readlane): no
        # memory traffic, nothing for a hand-placed s_waitcnt to reorder against -- not a spill
        # in the sense that makes a kernel wrong. SGPR spills to scratch show up as scratch,
        # unless the code object shows the frame is reserved and never accessed (scratch_note).
        # A VGPR spill is refused whatever the code shows.
        return bool(self.vgpr_spill or (self.scratch and not self.scratch_note))

    def line(self) -> str:
        vg = str(self.vgpr)
        if self.agpr:
            vg = f"{self.vgpr}+{self.agpr}a"
        if self.vgpr_rounded and self.vgpr_rounded != self.vgpr + self.agpr:
            vg = f"{vg}->{self.vgpr_rounded}"
        occ = self.occ_by_reg if self.occ_by_reg is not None else self.occupancy
        hdrm = "" if self.vgpr_headroom is None else f" hdrm={self.vgpr_headroom}"
        return (
            f"{self.name}: vgpr={vg} occ={occ} lds={self.lds} "
            f"scratch={self.scratch} spill={self.vgpr_spill + self.sgpr_spill}{hdrm}"
            + (f" ({self.scratch_note})" if self.scratch_note else "")
        )


def _demangle(names: List[str]) -> Dict[str, str]:
    exe = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not exe:  # TheRock's pip layout: ROCm's LLVM is not on PATH
        try:
            from .compile import llvm_tool  # noqa: PLC0415 -- compile imports this module

            exe = llvm_tool("llvm-cxxfilt", required=False)
        except Exception:  # noqa: BLE001 -- no hipcc either: stay mangled
            exe = None
    if not exe or not names:
        return {n: n for n in names}
    try:
        out = subprocess.run(
            [exe], input="\n".join(names), capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
    except Exception:
        return {n: n for n in names}
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def remark(line: str):
    """One line of a build log -> ("name", mangled), (field, value), or None
    when it is not a kernel-resource-usage remark (either shape; module doc)."""
    m = _REMARK_RE.search(line) or _BARE_RE.match(line)
    if m is None:
        return None
    if m.group("name"):
        return "name", m.group("name")
    key = _FIELDS.get(m.group("key"))
    if key is None:
        return None
    val = m.group("val")
    return key, (int(val == "True") if val in ("True", "False") else int(val))


def parse(log: str, target: Optional[Target] = None, *, compiler: str = "") -> List[KernelResources]:
    """Build log -> one record per kernel instantiation.

    An empty result is an error, not an empty answer: an up-to-date target
    emits no remarks at all, which is indistinguishable from a clean build.
    hk always compiles into a fresh directory, so reaching here with nothing
    means the compiler printed them in a shape this does not read, or not at
    all. `compiler` (`hipcc --version`'s clang line) goes into that error.
    """
    raw: Dict[str, Dict[str, int]] = {}
    cur = None
    for line in log.splitlines():
        r = remark(line)
        if r is None:
            continue
        if r[0] == "name":
            cur = r[1]
            raw.setdefault(cur, {})
        elif cur is not None:
            raw[cur][r[0]] = r[1]

    if not raw:
        head = [l for l in log.splitlines() if l.strip()][:12]
        shown = ("\n".join(f"    {l[:200]}" for l in head) if head
                 else "    (nothing: the compiler printed no output at all)")
        raise ResourceError(
            "the build emitted no kernel-resource-usage remarks that hk can read.\n"
            "  hk cannot certify a kernel it has no numbers for, and a spilling\n"
            "  kernel silently computes the wrong answer, so this is fatal rather\n"
            "  than skipped. hk passed -Rpass-analysis=kernel-resource-usage; it reads\n"
            "  the remarks of clang 22, 23 and 24 (the README lists the tested toolchains).\n"
            f"  Compiler: {compiler or 'unknown'}\n"
            f"  First lines of its output:\n{shown}"
        )

    pretty = _demangle(list(raw))
    out = []
    for mangled, fields in raw.items():
        r = KernelResources(name=pretty[mangled], mangled=mangled, **fields)
        if target is not None and r.vgpr:
            charged = target.charged_vgprs(r.vgpr, r.agpr)
            r.vgpr_rounded = target.vgpr_rounded(charged)
            r.occ_by_reg = target.occupancy(charged)
            r.vgpr_headroom = target.vgpr_headroom(charged)
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
