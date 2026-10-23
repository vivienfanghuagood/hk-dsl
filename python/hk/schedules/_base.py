"""The protocol every kernel family's schedule follows, and the machinery
the families share: knobs declared once as dataclass fields, records,
neighbourhoods and spaces.

A schedule is a frozen dataclass. Each field is a knob, declared with
`knob()`: its default, the values a search may give it (`values`, in the
order a neighbourhood steps through them), the subset the default space
takes (`seed`), and one line of what it does. Adding a knob to a family is
one field here and the parameter in the kernel factory it is passed to.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = ["knob", "Axis", "Schedule", "Illegal", "family", "families", "from_record"]


class Illegal(ValueError):
    """A schedule refused by check(): `reasons` lists why, one a line."""

    def __init__(self, schedule: "Schedule", reasons: Sequence[str], shape=None, arch=None):
        self.schedule = schedule
        self.reasons = list(reasons)
        where = ""
        if shape is not None:
            where = " at %dx%dx%d" % tuple(shape)
        if arch:
            where += f" on {arch}"
        super().__init__(f"{schedule}{where} is not legal:\n"
                         + "\n".join(f"  - {r}" for r in self.reasons))


def knob(default: Any, *, values: Sequence[Any] = (), seed: Optional[Sequence[Any]] = None,
         doc: str = "", gate: bool = False):
    """A schedule field.

    values  what a search may set it to, in neighbourhood order (empty: the
            axis is not searched; `space(axes=...)` can still set it)
    seed    the values the default space takes (None: held at the base)
    doc     one line: what it trades
    gate    a build gate (max_vgprs, min_occupancy), not code: it changes
            which builds are refused, never the C++
    """
    return field(default=default, metadata=dict(
        values=tuple(values), seed=None if seed is None else tuple(seed),
        doc=doc, gate=gate))


@dataclass(frozen=True)
class Axis:
    """One knob as a search sees it."""

    name: str
    default: Any
    values: Tuple[Any, ...]
    seed: Optional[Tuple[Any, ...]]
    doc: str
    gate: bool = False

    def near(self, value: Any, radius: int = 1) -> List[Any]:
        """The values within `radius` steps of `value` in `values` (all of
        them if `value` is not one of them), `value` excluded."""
        if not self.values:
            return []
        if value not in self.values:
            return [v for v in self.values if v != value]
        i = self.values.index(value)
        lo, hi = max(0, i - radius), min(len(self.values), i + radius + 1)
        return [v for v in self.values[lo:hi] if v != value]


_FAMILIES: Dict[str, type] = {}


def family(name: str) -> type:
    """The schedule class of a family ("rdna", "decode", "pp", "dp", "cols")."""
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ValueError(f"no schedule family {name!r}; have {sorted(_FAMILIES)}") from None


def families(arch: Optional[str] = None) -> List[type]:
    """Every family, or those that run on `arch`, in registration order."""
    return [c for c in _FAMILIES.values() if arch is None or arch in c.arches]


@dataclass(frozen=True)
class Schedule:
    """A point in a kernel family's schedule space. Subclasses are frozen
    dataclasses whose fields are the knobs (see `knob`)."""

    #: The family's name in records ("family": ...) and route candidates.
    family: ClassVar[str] = ""
    #: The archs the family's kernels build and run on.
    arches: ClassVar[Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.__dict__.get("family"):
            _FAMILIES[cls.family] = cls

    # ------------------------------------------------------------ knobs

    @classmethod
    def axes(cls) -> Dict[str, Axis]:
        """Every knob of the family, in declaration order."""
        out = {}
        for f in fields(cls):
            md = f.metadata
            out[f.name] = Axis(f.name, f.default, md.get("values", ()), md.get("seed"),
                               md.get("doc", ""), md.get("gate", False))
        return out

    def knobs(self) -> Dict[str, Any]:
        """{knob: value} for every knob."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def changed(self) -> Dict[str, Any]:
        """The knobs that differ from the family's defaults."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != f.default}

    def replace(self, **knobs: Any) -> "Schedule":
        """This schedule with some knobs changed."""
        unknown = set(knobs) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"{self.family} has no knob(s) {sorted(unknown)}; "
                            f"it has {[f.name for f in fields(self)]}")
        return dataclasses.replace(self, **knobs)

    def __str__(self) -> str:
        ch = self.changed()
        return f"{self.family}(" + ", ".join(f"{k}={v!r}" for k, v in ch.items()) + ")"

    # ------------------------------------------------------------ records

    def to_record(self, *, full: bool = False) -> Dict[str, Any]:
        """{"family": ..., knob: value, ...}: the knobs that differ from the
        defaults, or every knob with `full`. What from_record reads, and what
        a tuned record or route table stores."""
        return {"family": self.family, **(self.knobs() if full else self.changed())}

    @classmethod
    def from_record(cls, rec: Any, *, strict: bool = True) -> "Schedule":
        """A schedule from its record: {"family": f, knob: value, ...}, a
        route table's candidate [family, {knobs}], or {knobs} on a family's
        class. Knobs a record leaves out take the family's defaults. A knob
        the family does not have raises with `strict`, else is dropped (an
        old record should fall back toward the default, not fail a launch)."""
        return from_record(rec, strict=strict, family_cls=None if cls is Schedule else cls)

    @classmethod
    def _from_record_dict(cls, d: Mapping[str, Any], strict: bool) -> "Schedule":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown and strict:
            raise ValueError(f"{cls.family} has no knob(s) {sorted(unknown)}; it has "
                             f"{[f.name for f in fields(cls)]}")
        return cls(**{k: v for k, v in d.items() if k in names})

    # ------------------------------------------------------------ legality

    def check(self, m: int, n: int, k: int, arch: Optional[str] = None,
              dtype: Any = "bf16") -> List[str]:
        """Why this schedule cannot run C = A @ B.T at M x N x K on `arch`,
        before anything is compiled: [] if it can. Every reason is a sentence
        an agent can act on."""
        arch = _arch(arch)
        why = []
        if arch not in self.arches:
            why.append(f"{self.family}: {self._what()} builds for {', '.join(self.arches)}, "
                       f"not {arch}")
            return why
        if _dtype_name(dtype) != "bf16":
            why.append(f"{self.family}: A and B are bf16 (fp32 accumulate); "
                       f"{_dtype_name(dtype)} is not a GEMM hk has yet")
        if m < 1 or n < 1 or k < 1:
            why.append(f"M={m}, N={n}, K={k}: every extent must be at least 1")
            return why
        why += self.check_kernel(arch)
        why += self._check_shape(m, n, k, arch)
        return why

    def check_kernel(self, arch: Optional[str] = None) -> List[str]:
        """The shape-free half of check(): what decides whether the kernel
        traces and builds (warp grid, LDS, registers) -- a kernel is generic
        in M, N and K, so this is what `k.build()` answers."""
        arch = _arch(arch)
        if arch not in self.arches:
            return [f"{self.family}: {self._what()} builds for {', '.join(self.arches)}, "
                    f"not {arch}"]
        return self._check_knobs(arch)

    def legal(self, m: int, n: int, k: int, arch: Optional[str] = None) -> bool:
        """Whether check() has nothing against it."""
        return not self.check(m, n, k, arch)

    def require(self, m: int, n: int, k: int, arch: Optional[str] = None) -> "Schedule":
        """self, or Illegal (a ValueError) with check()'s reasons."""
        why = self.check(m, n, k, arch)
        if why:
            raise Illegal(self, why, (m, n, k), _arch(arch))
        return self

    def _check_knobs(self, arch: str) -> List[str]:
        return []

    def _check_shape(self, m: int, n: int, k: int, arch: str) -> List[str]:
        return []

    def _what(self) -> str:
        return self.family

    # ------------------------------------------------------------ spaces

    def neighbors(self, radius: int = 1, axes: Optional[Iterable[str]] = None
                  ) -> Iterator["Schedule"]:
        """Schedules that differ from this one in at most `radius` knobs, each
        moved one step along its axis' values -- this one first. `axes`
        limits which knobs move. Legality is not checked here (see space)."""
        ax = self.axes()
        names = [a for a in (axes if axes is not None else ax) if ax[a].values]
        seen = {self}
        yield self
        for r in range(1, radius + 1):
            for combo in itertools.combinations(names, r):
                choices = [ax[a].near(getattr(self, a)) for a in combo]
                for vals in itertools.product(*choices):
                    s = dataclasses.replace(self, **dict(zip(combo, vals)))
                    if s not in seen:
                        seen.add(s)
                        yield s

    @classmethod
    def base_for(cls, m: int, n: int, k: int, arch: str) -> "Schedule":
        """The family's starting point at a shape: its defaults, or what the
        shipped tables pick where that is this family."""
        return cls()

    @classmethod
    def space(cls, m: int, n: int, k: int, arch: Optional[str] = None, *,
              around: Optional["Schedule"] = None, radius: int = 1,
              axes: Optional[Mapping[str, Any]] = None,
              base: Optional["Schedule"] = None, legal: bool = True
              ) -> Iterator["Schedule"]:
        """The family's legal schedules for M x N x K on `arch`.

        By default the product of every knob's `seed` values, the other
        knobs at `base` (the family's base_for the shape). `axes` reshapes
        it: {knob: [values]} searches those values, {knob: "all"} the knob's
        full value list, {knob: value} pins it. `around=s` gives instead the
        neighbourhood of s: every schedule within `radius` knob changes of it
        (axes then names the knobs allowed to move, or gives their values).
        `legal=False` yields the refused points too (for testing check)."""
        arch = _arch(arch)
        if arch not in cls.arches:
            return
        ax = cls.axes()
        if axes:
            bad = set(axes) - set(ax)
            if bad:
                raise ValueError(f"{cls.family} has no knob(s) {sorted(bad)}; it has {list(ax)}")
        if around is not None:
            if not isinstance(around, cls):
                return
            if axes:
                ax2 = {}
                for name, a in ax.items():
                    v = axes.get(name, None) if name in axes else None
                    if name not in axes:
                        ax2[name] = dataclasses.replace(a, values=())
                    elif v == "all" or v is None:
                        ax2[name] = a
                    elif isinstance(v, (list, tuple)):
                        ax2[name] = dataclasses.replace(a, values=tuple(v))
                    else:
                        ax2[name] = dataclasses.replace(a, values=(v,))
                gen = _neighbors(around, ax2, radius)
            else:
                gen = around.neighbors(radius)
        else:
            b = base if base is not None else cls.base_for(m, n, k, arch)
            lists = {}
            for name, a in ax.items():
                if axes and name in axes:
                    v = axes[name]
                    lists[name] = (list(a.values) if v == "all" else
                                   list(v) if isinstance(v, (list, tuple)) else [v])
                elif a.seed is not None:
                    lists[name] = list(a.seed)
                else:
                    lists[name] = [getattr(b, name)]
            names = list(lists)
            gen = (cls(**dict(zip(names, combo)))
                   for combo in itertools.product(*(lists[n] for n in names)))
        seen = set()
        for s in gen:
            s = s.canonical()
            if s in seen:
                continue
            seen.add(s)
            if not legal or not s.check(m, n, k, arch):
                yield s

    def canonical(self) -> "Schedule":
        """The same kernel with knobs that do nothing at their value put
        back to the default (so a space holds each kernel once)."""
        return self


def _neighbors(s: Schedule, ax: Dict[str, Axis], radius: int) -> Iterator[Schedule]:
    names = [a for a in ax if ax[a].values]
    yield s
    for r in range(1, radius + 1):
        for combo in itertools.combinations(names, r):
            choices = [ax[a].near(getattr(s, a)) for a in combo]
            for vals in itertools.product(*choices):
                yield dataclasses.replace(s, **dict(zip(combo, vals)))


def from_record(rec: Any, *, strict: bool = True, family_cls: Optional[type] = None
                ) -> Schedule:
    """A Schedule from any of its record forms (see Schedule.from_record)."""
    if isinstance(rec, Schedule):
        return rec
    if isinstance(rec, (list, tuple)) and len(rec) == 2 and isinstance(rec[0], str):
        fam, knobs = rec
        cls = family(fam)
        return cls._from_record_dict(dict(knobs), strict)
    if isinstance(rec, Mapping):
        d = dict(rec)
        fam = d.pop("family", None)
        if fam is None:
            if family_cls is None:
                raise ValueError(f"a record without a family: {rec!r}; pass it to "
                                 f"the family's class (e.g. hk.schedules.Dp.from_record)")
            cls = family_cls
        else:
            cls = family(fam)
            if family_cls is not None and cls is not family_cls:
                raise ValueError(f"a {fam} record given to {family_cls.family}")
        return cls._from_record_dict(d, strict)
    raise TypeError(f"not a schedule record: {rec!r}")


def _arch(arch: Optional[str]) -> str:
    if arch:
        return arch
    from ..target import current_arch  # noqa: PLC0415

    return current_arch()


def _dtype_name(dtype: Any) -> str:
    if isinstance(dtype, str):
        return {"bfloat16": "bf16", "torch.bfloat16": "bf16"}.get(dtype, dtype)
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    s = str(dtype)
    return "bf16" if s == "torch.bfloat16" else s.replace("torch.", "")
