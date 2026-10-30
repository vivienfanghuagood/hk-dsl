"""Read a compiled kernel's assembly: instruction mix per loop, for tuning and for tests that pin codegen."""
from __future__ import annotations

import collections
import re
from typing import Dict, List

__all__ = ["loops", "mix", "summary"]

_INSN = re.compile(r"^\s+([a-z_][a-z0-9_.]*)\b")
_LABEL = re.compile(r"^(\.LBB\d+_\d+):")
_BR = re.compile(r"^\s+s_cbranch_\w+\s+(\.LBB\d+_\d+)")


def _body(asm: str) -> List[str]:
    lines = asm.splitlines()
    out, on = [], False
    for ln in lines:
        if re.match(r"^[A-Za-z_][\w.]*:\s*(;.*)?$", ln) and not ln.startswith("."):
            on = True
            continue
        if on and ln.strip().startswith(".Lfunc_end"):
            break
        if on:
            out.append(ln)
    return out


def loops(asm: str) -> List[List[str]]:
    """Instruction lines of each innermost backward-branch loop."""
    body = _body(asm)
    pos = {}
    res = []
    for i, ln in enumerate(body):
        m = _LABEL.match(ln)
        if m:
            pos[m.group(1)] = i
        m = _BR.match(ln)
        if m and m.group(1) in pos:
            res.append([l for l in body[pos[m.group(1)]:i + 1] if _INSN.match(l)])
    return res


def mix(lines: List[str]) -> Dict[str, int]:
    c = collections.Counter()
    for ln in lines:
        m = _INSN.match(ln)
        if m:
            c[m.group(1)] += 1
    return dict(c.most_common())


def summary(asm: str) -> str:
    out = []
    for i, lp in enumerate(loops(asm)):
        m = mix(lp)
        groups = collections.Counter()
        for k, v in m.items():
            g = ("mfma/wmma" if ("mfma" in k or "wmma" in k) else "ds_read" if k.startswith("ds_read") or
                 k.startswith("ds_load") else "ds_write" if k.startswith("ds_write") or k.startswith("ds_store")
                 else "vmem" if k.startswith(("buffer_", "global_")) else "waitcnt" if "waitcnt" in k
                 else "barrier" if "barrier" in k else "salu" if k.startswith("s_") else "valu")
            groups[g] += v
        out.append(f"loop {i}: {len(lp)} insns " + " ".join(f"{k}={v}" for k, v in groups.most_common()))
        out.append("   " + " ".join(f"{k}={v}" for k, v in list(m.items())[:24]))
    return "\n".join(out)
