"""MFMA hazards over the control flow: the wait states an MFMA's result needs
before anything reads or overwrites it, measured on every path.

The hardware does not interlock on an MFMA's result. The compiler pads with
s_nop: an instruction that reads the result, overwrites it, or overwrites a
register the MFMA is still reading as srcC must come enough wait states
after it (a "window", by the MFMA's shape). Where the reader sits in a block
with several predecessors, the distance is the shortest over the paths into
it. LLVM's post-RA hazard recognizer before llvm/llvm-project#225989
(2026-10-06) walked predecessors with a visited set that kept no distances:
a block reached the long way first was not looked at again on the short
way, so the padding came out too small. On gfx950, a paged decode attention
read an MFMA result 2 wait states after it where 8 are needed, and computed
some scores wrong (ROCm/llvm-project#5153; ROCm 7.2.3, 10.0 and 10.1 do not
have the fix).

This checks a compiled kernel the way the fixed recognizer does
(WindowDeficitSearch): from each instruction that touches an MFMA's
registers, a shortest-distance walk backwards over the control flow, each
producer measured against its own window. It reads the disassembly of a code
object or the assembly text the compiler emitted, so it needs no LLVM
internals; `fix_asm` pads the assembly where a deficit is.

The windows are LLVM's, for CDNA3 and CDNA4 (gfx940+; GCNHazardRecognizer:
checkMAIHazards90A, checkMAIVALUHazards): XDL MFMAs of N passes need
N + 3 (+1 on gfx950 unless N = 2) before a VALU, memory or MFMA srcA/B read
or a VALU overwrite of the result, N + 1 (+1 on gfx950) before another
MFMA reads it as an overlapping srcC that is not the same MFMA's exact
accumulator; FP32-input MFMAs (not XDL) N + 2; f64 (DGEMM) a table. The
pass counts are the scheduling model's (llvm-mca on gfx942 / gfx950).
"""

from __future__ import annotations

import heapq
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

_REG = re.compile(r"\b([va])(?:\[(\d+):(\d+)\]|(\d+)(?!\d))")
_MFMA = re.compile(r"^v_(s)?mfma(c)?(_scale)?_([a-z]+\d+)_(\d+)x(\d+)x(\d+)(\w*)$")
_STORE = re.compile(r"^(buffer|global|flat|scratch)_store|^ds_(write|store)")
_ATOMIC = re.compile(r"^(buffer|global|flat)_atomic")
_DS_RET = re.compile(r"^ds_(read|load|bpermute|permute|swizzle)|^ds_\w*_rtn_")
_BRANCH = re.compile(r"^s_(c?branch)")

Regs = FrozenSet[Tuple[str, int]]
_NONE: Regs = frozenset()


def regs(operand: str) -> Regs:
    out = set()
    for m in _REG.finditer(operand):
        lo, hi = ((int(m.group(2)), int(m.group(3))) if m.group(2)
                  else (int(m.group(4)), int(m.group(4))))
        out |= {(m.group(1), r) for r in range(lo, hi + 1)}
    return frozenset(out)


def operands(rest: str) -> List[str]:
    """The operands of an instruction's text after the mnemonic: split at
    commas outside brackets (op_sel_hi:[0,0,0] is one modifier), each cut
    at the first space (the modifiers after the last one)."""
    out, depth, cur = [], 0, ""
    for ch in rest:
        depth += (ch == "[") - (ch == "]")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [o.strip().split(" ", 1)[0] for o in out if o.strip()]


@dataclass(frozen=True)
class Mfma:
    """What the windows depend on: the pass count, XDL or not, f64 or not,
    and the instruction's name (two MFMAs of one name share an accumulator
    path)."""

    name: str
    passes: int
    xdl: bool
    dgemm: bool


def mfma_info(mnem: str, arch: str) -> Optional[Mfma]:
    m = _MFMA.match(mnem)
    if not m:
        return None
    dtype, mdim, suffix = m.group(4), int(m.group(5)), m.group(8)
    dgemm = dtype == "f64"
    f32_in = bool(re.fullmatch(r"(_\d+b)?_?f32", suffix))
    if dgemm:
        passes = 4 if mdim == 4 else (16 if arch == "gfx950" else 8)
    else:
        passes = {4: 2, 16: 4, 32: 8}.get(mdim, 16)
        if f32_in and mdim != 4:
            passes *= 2
    return Mfma(mnem, passes, xdl=not (dgemm or f32_in), dgemm=dgemm)


# ---------------------------------------------------------------- windows


def max_window(arch: str) -> int:
    return 20 if arch == "gfx950" else 19


def _xdl_late(p: int, gfx950: bool) -> int:
    return p + 3 + (gfx950 and p != 2)


def read_window(p: Mfma, arch: str, memory: bool) -> int:
    """A VALU (memory, when `memory`) instruction reading p's result."""
    g = arch == "gfx950"
    if p.dgemm:
        if p.passes == 4:
            return 9 if memory else 6
        return 18 if memory else (19 if g else 11)
    return _xdl_late(p.passes, g) if p.xdl else p.passes + 2


def srcab_window(p: Mfma, arch: str) -> int:
    """An MFMA reading p's result as srcA or srcB."""
    g = arch == "gfx950"
    if p.dgemm:
        return 6 if p.passes == 4 else (19 if g else 11)
    return _xdl_late(p.passes, g) if p.xdl else p.passes + 2


def _overlapped_srcc(c: Mfma, p: Mfma, arch: str) -> int:
    g = arch == "gfx950"
    if c.xdl and not p.xdl:
        return 0
    if p.dgemm:
        return 4 if p.passes == 4 else (17 if g else 9)
    if not p.xdl:
        return p.passes
    if c.xdl:
        return p.passes + 1 + g
    return p.passes + 1 + (p.passes != 2 and g)


def srcc_window(c: Mfma, p: Mfma, arch: str, full: bool) -> int:
    """MFMA c reading p's result as srcC: exactly p's destination (`full`),
    or overlapping it."""
    if not full:
        return _overlapped_srcc(c, p, arch)
    need = 2 if p.passes == 2 else 0
    if c.name != p.name:
        # the zero-wait accumulator forwarding holds within one MFMA only
        need = max(need, _overlapped_srcc(c, p, arch))
    return need


def waw_window(p: Mfma, arch: str) -> int:
    """A VALU instruction overwriting p's result."""
    g = arch == "gfx950"
    if p.dgemm:
        return 6 if p.passes == 4 else 11
    return _xdl_late(p.passes, g) if p.xdl else p.passes + 2


def war_window(p: Mfma) -> Optional[int]:
    """A VALU instruction overwriting a register p reads as srcC."""
    if p.dgemm or not p.xdl:
        return None
    return {2: 1, 4: 3, 8: 7}.get(p.passes, 15)


# ---------------------------------------------------------------- programs


@dataclass
class Insn:
    text: str
    where: str            # an address or a label+offset, for the report
    target: Optional[int] = None   # a branch's destination, as an index

    @property
    def mnem(self) -> str:
        return self.text.partition(" ")[0]


@dataclass
class _Facts:
    weight: int
    reads: Regs
    writes: Regs
    mfma: Optional[Mfma] = None
    dst: Regs = _NONE
    srcab: Regs = _NONE
    srcc: Regs = _NONE
    memory: bool = False


def _facts(ins: Insn, arch: str) -> _Facts:
    mnem = ins.mnem
    rest = ins.text.partition(" ")[2]
    if mnem == "s_nop":
        try:
            n = int(rest.split()[0], 0)
        except (IndexError, ValueError):
            n = 0
        return _Facts(n + 1, _NONE, _NONE)
    if mnem.startswith("s_"):
        return _Facts(1, _NONE, _NONE)
    ops = operands(rest)
    info = mfma_info(mnem, arch)
    if info is not None and ops:
        dst = regs(ops[0])
        if info.name.startswith("v_smfmac"):
            ab = frozenset().union(*(regs(o) for o in ops[1:]))
            return _Facts(1, ab | dst, dst, info, dst, ab, dst)
        ab = frozenset().union(*(regs(o) for o in ops[1:3]), *(regs(o) for o in ops[4:]))
        c = regs(ops[3]) if len(ops) > 3 else _NONE
        return _Facts(1, ab | c, dst, info, dst, ab, c)
    memory = not mnem.startswith("v_")
    if _writes_no_vgpr(mnem, rest):
        return _Facts(1, frozenset().union(*(regs(o) for o in ops)) if ops else _NONE, _NONE,
                      memory=memory)
    reads = frozenset().union(*(regs(o) for o in ops[1:])) if len(ops) > 1 else _NONE
    return _Facts(1, reads, regs(ops[0]) if ops else _NONE, memory=memory)


def _writes_no_vgpr(mnem: str, rest: str) -> bool:
    """A store, a copy into LDS, an atomic without a return or an LDS op that
    returns nothing: every register operand is read."""
    if _STORE.match(mnem) or "_lds_" in mnem or re.search(r"(^|\s)lds(\s|$)", rest):
        return True
    if mnem.startswith("ds_"):
        return not _DS_RET.match(mnem)
    if _ATOMIC.match(mnem):
        return not re.search(r"\b(glc|sc0)\b", rest)
    return False


def from_objdump(insts) -> List[Insn]:
    """cocheck.Inst (address, text) as Insns, branch targets resolved."""
    from .cocheck import _branch_target  # noqa: PLC0415

    index = {ins.addr: i for i, ins in enumerate(insts)}
    out = []
    for i, ins in enumerate(insts):
        nxt = insts[i + 1].addr if i + 1 < len(insts) else ins.addr + 4
        t = _branch_target(ins, nxt)
        out.append(Insn(ins.text, f"0x{ins.addr:x}", index.get(t) if t is not None else None))
    return out


_LABEL = re.compile(r"^([.\w$]+):")


def asm_functions(text: str) -> Dict[str, List[Tuple[int, Insn]]]:
    """The assembly text the compiler emitted, per function: (line index,
    instruction), branch targets resolved through the labels."""
    funcs: Dict[str, List[Tuple[int, Insn]]] = {}
    cur: Optional[List[Tuple[int, Insn]]] = None
    labels: Dict[str, int] = {}
    pending: List[Tuple[List[Tuple[int, Insn]], Dict[str, int]]] = []
    last_label, off = "", 0
    for ln, raw in enumerate(text.splitlines()):
        line = raw.split(";", 1)[0].split("//", 1)[0].rstrip()
        if not line.strip():
            continue
        m = _LABEL.match(line)
        if m and not raw.startswith((" ", "\t")):
            name = m.group(1)
            if not name.startswith("."):
                cur = funcs.setdefault(name, [])
                labels = {}
                pending.append((cur, labels))
            if cur is not None:
                labels[name] = len(cur)
            last_label, off = name, 0
            continue
        s = line.strip()
        if cur is None or s.startswith(".") or not re.match(r"^[a-z]", s):
            continue
        cur.append((ln, Insn(re.sub(r"\s+", " ", s), f"{last_label}+{off}")))
        off += 1
    for body, labs in pending:
        for _, ins in body:
            if _BRANCH.match(ins.mnem):
                dest = ins.text.partition(" ")[2].strip().split(",")[0].strip()
                if dest in labs:
                    ins.target = labs[dest]
    return funcs


# ---------------------------------------------------------------- the walk


@dataclass
class Deficit:
    consumer: int
    producer: int
    need: int
    have: int
    what: str

    @property
    def missing(self) -> int:
        return self.need - self.have


def _blocks(prog: Sequence[Insn]):
    """Block starts, each index's block, and each block's predecessors (as
    the index of the predecessor's last instruction, which the edge runs)."""
    n = len(prog)
    lead = {0}
    for i, ins in enumerate(prog):
        m = ins.mnem
        if ins.target is not None:
            lead.add(ins.target)
        if _BRANCH.match(m) or m in ("s_endpgm", "s_setpc_b64", "s_trap"):
            lead.add(i + 1)
    starts = sorted(x for x in lead if x < n)
    block_of = [0] * n
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else n
        for i in range(s, e):
            block_of[i] = b
    preds: List[List[int]] = [[] for _ in starts]
    for b, s in enumerate(starts):
        e = (starts[b + 1] if b + 1 < len(starts) else n) - 1
        last = prog[e]
        m = last.mnem
        if last.target is not None:
            preds[block_of[last.target]].append(e)
        falls = not (m == "s_branch" or m in ("s_endpgm", "s_setpc_b64", "s_trap"))
        if falls and e + 1 < n:
            preds[b + 1].append(e)
    return starts, block_of, preds


def deficits(prog: Sequence[Insn], arch: str) -> List[Deficit]:
    """Every instruction that touches an MFMA's registers too soon after it on
    some path: the worst producer for each, as a Deficit. Empty when the
    padding holds on every path (and for a target without MFMAs)."""
    if not (arch.startswith("gfx94") or arch == "gfx950"):
        return []
    facts = [_facts(ins, arch) for ins in prog]
    touched = frozenset().union(*(f.dst | f.srcc for f in facts if f.mfma)) if any(
        f.mfma for f in facts) else _NONE
    if not touched:
        return []
    starts, block_of, preds = _blocks(prog)
    limit = max_window(arch)
    out: List[Deficit] = []

    def window(c: _Facts, p: _Facts) -> Optional[Tuple[int, str]]:
        need, what = 0, ""
        pm = p.mfma
        if c.mfma is not None:
            if c.srcab & p.dst:
                need, what = srcab_window(pm, arch), "MFMA srcA/B read of its result"
            if c.srcc & p.dst:
                w = srcc_window(c.mfma, pm, arch, full=(c.srcc == p.dst))
                if w > need:
                    need, what = w, "MFMA srcC read of its result"
            return (need, what) if need else None
        if c.reads & p.dst:
            w = read_window(pm, arch, c.memory)
            if w > need:
                need, what = w, "read of its result"
        if c.memory:
            # A load writes its destination when the data returns, tens of cycles after the
            # windows below close. LLVM pads these too; it never sees hk's inline-asm LDS
            # reads, whose destinations the allocator does put next to an MFMA's.
            return (need, what) if need else None
        if c.writes & p.dst:
            w = waw_window(pm, arch)
            if w > need:
                need, what = w, "overwrite of its result"
        if c.writes & p.srcc:
            w = war_window(pm)
            if w is not None and w > need:
                need, what = w, "overwrite of its srcC"
        return (need, what) if need else None

    for j, c in enumerate(facts):
        if not ((c.reads | c.writes) & touched):
            continue
        worst: Optional[Deficit] = None

        def scan(i: int, lo: int, dist: int) -> Optional[int]:
            nonlocal worst
            while i >= lo:
                p = facts[i]
                if p.mfma is not None:
                    w = window(c, p)
                    if w is not None and w[0] - dist > 0 and (
                            worst is None or w[0] - dist > worst.missing):
                        worst = Deficit(j, i, w[0], dist, w[1])
                dist += p.weight
                if dist >= limit:
                    return None
                i -= 1
            return dist

        b = block_of[j]
        d0 = scan(j - 1, starts[b], 0)
        if d0 is not None:
            best: Dict[int, int] = {}
            heap = [(d0, e) for e in preds[b]]
            heapq.heapify(heap)
            while heap:
                d, e = heapq.heappop(heap)
                pb = block_of[e]
                if best.get(pb, limit) <= d:
                    continue
                best[pb] = d
                dd = scan(e, starts[pb], d)
                if dd is not None:
                    for pe in preds[pb]:
                        heapq.heappush(heap, (dd, pe))
        if worst is not None:
            out.append(worst)
    return out


def report(prog: Sequence[Insn], found: Sequence[Deficit]) -> List[str]:
    """One line a deficit: where, the two instructions, the distance on the
    shortest path, the window, and the wait states to add before the reader."""
    lines = []
    for d in found:
        c, p = prog[d.consumer], prog[d.producer]
        lines.append(
            f"{c.where}: `{c.text}` comes {d.have} wait state(s) after `{p.text}` "
            f"({p.where}) on its shortest path; a {d.what} needs {d.need}: "
            f"{d.missing} more (s_nop) before it")
    return lines


def fix_asm(text: str, arch: str) -> Tuple[str, List[str]]:
    """`text` (compiler assembly) with s_nop in front of every instruction
    that is short of its window on some path, and what was padded. Padding
    only lengthens paths, so one pass settles it; the result is checked
    again by the caller."""
    lines = text.splitlines(keepends=True)
    pads: Dict[int, int] = {}
    said: List[str] = []
    for name, body in asm_functions(text).items():
        prog = [ins for _, ins in body]
        found = deficits(prog, arch)
        for d, line in zip(found, report(prog, found)):
            ln = body[d.consumer][0]
            pads[ln] = max(pads.get(ln, 0), d.missing)
            said.append(f"{name}: {line}")
    for ln in sorted(pads, reverse=True):
        indent = re.match(r"^\s*", lines[ln]).group(0) or "\t"
        k, nops = pads[ln], []
        while k > 0:
            nops.append(f"{indent}s_nop {min(k, 8) - 1}\n")
            k -= 8
        lines[ln:ln] = nops
    return "".join(lines), said


# ---------------------------------------------------------------- the compiler

#: The recognizer's case, as MIR: the reader's block has the MFMA's block one
#: branch away and, first in its predecessor list, a longer path from it. A
#: fixed LLVM pads the read for the short path (s_nop 6 on gfx950); LLVM
#: before llvm/llvm-project#225989 for the long one (s_nop 0).
PROBE_MIR = """\
---
name: mfma_read_two_paths
tracksRegLiveness: true
body: |
  bb.0:
    successors: %bb.2
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr4_vgpr5_vgpr6_vgpr7, $agpr0_agpr1_agpr2_agpr3, $sgpr0

    S_BRANCH %bb.2

  bb.1:
    successors: %bb.3
    liveins: $agpr0_agpr1_agpr2_agpr3

    $vgpr10 = V_MOV_B32_e32 0, implicit $exec
    $vgpr11 = V_MOV_B32_e32 0, implicit $exec
    $vgpr12 = V_MOV_B32_e32 0, implicit $exec
    $vgpr13 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.3

  bb.2:
    successors: %bb.3, %bb.1
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr4_vgpr5_vgpr6_vgpr7, $agpr0_agpr1_agpr2_agpr3, $sgpr0

    S_CMP_EQ_U32 $sgpr0, 0, implicit-def $scc
    $agpr0_agpr1_agpr2_agpr3 = V_MFMA_F32_16X16X32_BF16_e64 $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr4_vgpr5_vgpr6_vgpr7, $agpr0_agpr1_agpr2_agpr3, 0, 0, 0, implicit $mode, implicit $exec
    S_CBRANCH_SCC1 %bb.3, implicit killed $scc
    S_BRANCH %bb.1

  bb.3:
    liveins: $agpr0_agpr1_agpr2_agpr3

    $vgpr8 = V_ACCVGPR_READ_B32_e64 $agpr3, implicit $exec
    S_ENDPGM 0
...
"""

_FIXED: Dict[str, Optional[bool]] = {}


def compiler_fixed(llc: str) -> Optional[bool]:
    """Whether the LLVM of `llc` pads MFMA hazards over every path: its
    output for PROBE_MIR, checked here. None when it cannot say (no llc, or
    llc fails on the probe), and the gate then checks every kernel."""
    if llc in _FIXED:
        return _FIXED[llc]
    verdict: Optional[bool] = None
    if os.path.exists(llc):
        with tempfile.TemporaryDirectory(prefix="hk-mfmahaz-") as d:
            mir, out = os.path.join(d, "probe.mir"), os.path.join(d, "probe.s")
            with open(mir, "w") as f:
                f.write(PROBE_MIR)
            try:
                r = subprocess.run([llc, "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950",
                                    "-start-before=post-RA-hazard-rec", mir, "-o", out],
                                   capture_output=True, text=True, timeout=120)
                if r.returncode == 0:
                    with open(out) as f:
                        funcs = asm_functions(f.read())
                    body = funcs.get("mfma_read_two_paths")
                    if body and any(mfma_info(i.mnem, "gfx950") for _, i in body):
                        verdict = not deficits([i for _, i in body], "gfx950")
            except (OSError, subprocess.SubprocessError):
                verdict = None
    _FIXED[llc] = verdict
    return verdict
