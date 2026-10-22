"""The schedule space `hk.tune` searches, behind a small provider interface,
over hk.schedules (K1: one Schedule type per GEMM kernel family).

A provider answers, for one arch and one M x N x K:

    current(m, n, k)        the schedule the shape runs today (its route,
                            without this machine's tuning records)
    seeds(m, n, k)          schedules known to be good somewhere: every
                            shipped route candidate that takes the shape
    generate(m, n, k)       the families' seed spaces (hk.schedules.space)
    check(c, m, n, k)       "" if the schedule can run the shape, else why not
                            (Schedule.check: arithmetic, LDS, predicted
                            registers -- before anything compiles)
    neighbors(c, m, n, k)   one-knob moves (hk.schedules.space(around=c))
    jobs(c, m, n, k)        the kernels to compile for it (Schedule.kernels)
    runner(c, m, n, k)      fn(a, b, out) that runs it (Schedule.launch)
    spec(c) / from_spec     its record (Schedule.to_record / from_record)
    estimate(c, m, n, k)    the analytic model's guess (hk.tune.cost)
    structure / preference  what the model can tell apart, and which point
                            of a structure stands for it in the pool

A `Candidate` is the search's handle on a schedule: hashable by family and
record, carrying the schedule object (`obj`) and, for a shipped route
candidate, its name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import cost

__all__ = ["Candidate", "ScheduleSpace", "provider"]


def _hashable(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v


@dataclass(frozen=True)
class Candidate:
    """One point: a kernel family and its record (the knobs off the
    family's defaults); `obj` is the hk.schedules object, `name` the shipped
    candidate's name when it is one."""

    kind: str
    items: Tuple[Tuple[str, Any], ...]
    name: str = ""
    obj: Any = field(default=None, compare=False, hash=False, repr=False)

    @staticmethod
    def of(kind: str, sched: Dict[str, Any], name: str = "", obj: Any = None) -> "Candidate":
        return Candidate(kind, tuple(sorted((k, _hashable(v)) for k, v in sched.items())),
                         name, obj)

    @staticmethod
    def of_schedule(s, name: str = "") -> "Candidate":
        rec = dict(s.to_record())
        fam = rec.pop("family")
        return Candidate.of(fam, rec, name, s)

    @property
    def sched(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.items:
            if isinstance(v, str) and v[:1] in "{[":
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            out[k] = v
        return out

    @property
    def label(self) -> str:
        if self.kind == "cols":
            s = self.obj
            return f"cols:{s.cut}:{_short(s.head)}|{_short(s.tail)}" if s is not None else "cols"
        return _short_items(self.kind, self.items)

    def __str__(self) -> str:
        return self.name or self.label

    def __eq__(self, other):
        return isinstance(other, Candidate) and (self.kind, self.items) == (other.kind, other.items)

    def __hash__(self):
        return hash((self.kind, self.items))


_ABBREV = {"block_m": "", "block_n": "x", "k_step": "k", "stages": "s", "warps_m": "w",
           "warps_n": "x", "splits": "sk", "xcd_chunk": "c", "wgm": "g", "lds_shape": "l",
           "nt": "nt", "setprio": "sp", "pipelined": "pl", "early_store": "es", "late_b": "lb",
           "transposed_c": "tc", "slices": "sl", "warps": "w", "combine_depth": "cd",
           "resident": "r", "warp_rows": "wr", "n_split": "ns", "dot_slice": "d",
           "write_pos": "p", "lds_buffers": "b", "stream_a": "a", "sched": "i",
           "stagger": "t", "tokens": "tk"}


def _short_items(kind, items) -> str:
    parts = []
    for k, v in items:
        if isinstance(v, bool):
            v = int(v)
        if v is None:
            v = "-"
        parts.append(f"{_ABBREV.get(k, k)}{v}")
    return kind + ":" + "_".join(parts)


def _short(s) -> str:
    rec = dict(s.to_record())
    fam = rec.pop("family")
    return _short_items(fam, sorted(rec.items()))


def provider(arch: str) -> "ScheduleSpace":
    """The space provider for an arch."""
    if arch not in cost.CARDS:
        raise ValueError(f"hk.tune: no GEMM schedule space for {arch}")
    return ScheduleSpace(arch)


class ScheduleSpace:
    """hk's GEMMs on one arch, as hk.schedules spells them."""

    def __init__(self, arch: str):
        self.arch = arch

    @staticmethod
    def _S():
        from .. import schedules  # noqa: PLC0415

        return schedules

    def cand(self, s, name: str = "") -> Candidate:
        return Candidate.of_schedule(s, name)

    # -- what there is

    def current(self, m: int, n: int, k: int) -> Candidate:
        S = self._S()
        if self.arch == "gfx950":
            from ..ops import gemm_cdna  # noqa: PLC0415

            name = gemm_cdna.route(m, n, k, user=False)
            return self.cand(S.from_candidate(name, m, n, k), name)
        from ..ops import gemm as rdna  # noqa: PLC0415

        s = rdna.schedule_for(m, n, k, user=False)
        name = "thin" if s == rdna.THIN else ("default" if s == rdna.DEFAULT else "")
        return self.cand(S.Rdna(**s), name)

    def seeds(self, m: int, n: int, k: int) -> List[Candidate]:
        S = self._S()
        out = []
        if self.arch == "gfx950":
            from ..ops import gemm_cdna  # noqa: PLC0415

            for name in gemm_cdna.CANDIDATES:
                if gemm_cdna.fits(name, m, n, k):
                    try:
                        out.append(self.cand(S.from_candidate(name, m, n, k), name))
                    except (ValueError, KeyError):
                        continue
        else:
            from ..ops import gemm as rdna  # noqa: PLC0415

            out = [self.cand(S.Rdna(**rdna.DEFAULT), "default"),
                   self.cand(S.Rdna(**rdna.THIN), "thin"),
                   self.cand(S.Decode.base_for(m, n, k, self.arch), "decode")]
        return [c for c in out if not self.check(c, m, n, k)]

    #: Per family, axes the generated space takes beyond hk.schedules' seed
    #: values ({knob: [values]}): the RDNA GEMM's seed space holds its block
    #: and warp grid at the default, which leaves the model nothing to order.
    AXES = {
        # K splits that are not powers of two: at 1024x4608x18432 the 256x256
        # pingpong has 72 blocks, 144 with two shares and 288 with four --
        # 1.1 waves of 256 CUs -- and 216 with three.
        "pp": dict(splits=[1, 2, 3, 4, 6]),
        "dp": dict(block_m=[16, 32, 64, 128, 256], block_n=[32, 64, 96, 128, 192, 224, 256],
                   splits=[1, 2, 3, 4, 6, 8]),
        "rdna": dict(block_m=[32, 64, 128, 256], block_n=[64, 128, 256], k_step=[32, 64],
                     warps=[4, 8], warp_rows=[1, 2, 4, 8], n_split=[1, 2, 4],
                     lds_buffers=[1, 2], wgm=[8], write_pos=[2]),
    }

    def generate(self, m: int, n: int, k: int) -> List[Candidate]:
        S = self._S()
        out = []
        for fam in S.families_for(self.arch):
            ax = self.AXES.get(fam.family)
            got = fam.space(m, n, k, self.arch, axes=ax) if ax else fam.space(m, n, k, self.arch)
            out += [self.cand(s) for s in got]
        return out

    def check(self, c: Candidate, m: int, n: int, k: int) -> str:
        try:
            why = c.obj.check(m, n, k, self.arch)
        except Exception as e:  # noqa: BLE001 -- a check that raises is a refusal
            return f"{type(e).__name__}: {e}"
        return "; ".join(why)

    def neighbors(self, c: Candidate, m: int, n: int, k: int) -> List[Candidate]:
        S = self._S()
        try:
            got = list(S.space(m, n, k, self.arch, around=c.obj, radius=1))
        except Exception:  # noqa: BLE001
            return []
        return [self.cand(s) for s in got if s != c.obj]

    # -- the model's view

    def estimate(self, c: Candidate, m: int, n: int, k: int) -> float:
        s = c.obj
        if s.family == "cols":
            cut = s.cut_at(m, n)
            return (cost.estimate(s.head.family, s.head.knobs(), m, cut, k, self.arch)
                    + cost.estimate(s.tail.family, s.tail.knobs(), m, n - cut, k, self.arch))
        return cost.estimate(s.family, s.knobs(), m, n, k, self.arch)

    def structure(self, c: Candidate, m: int, n: int, k: int) -> tuple:
        """What the analytic model can tell apart: family, block, K split."""
        s = c.obj
        if s.family == "decode":
            return ("decode", s.splits, s.slices)
        if s.family == "cols":
            return ("cols", s.cut, self.structure(self.cand(s.tail), m, n, k))
        if s.family == "rdna":
            return ("rdna", s.block_m, s.block_n, s.lds_buffers)
        return (s.family, s.block_m, s.block_n, s.splits)

    def preference(self, c: Candidate) -> tuple:
        """Nearness to the family's defaults, for a structure's representative."""
        return (len(c.items), c.label)

    # -- running it

    def jobs(self, c: Candidate, m: int, n: int, k: int):
        try:
            return [(f"{c.label}#{i}", kern, consts)
                    for i, (_, kern, consts) in enumerate(c.obj.kernels(m, n, k))]
        except Exception:  # noqa: BLE001 -- the check after the build says why
            return []

    def runner(self, c: Candidate, m: int, n: int, k: int):
        s = c.obj

        def run(a, b, out):
            return s.launch(a, b, out)
        return run

    def spec(self, c: Candidate):
        return json.loads(json.dumps(c.obj.to_record()))

    def from_spec(self, spec, name: str = "") -> Candidate:
        return self.cand(self._S().from_record(spec), name)


def describe(c: Candidate) -> str:
    return json.dumps({"family": c.kind, **c.sched}, sort_keys=True)
