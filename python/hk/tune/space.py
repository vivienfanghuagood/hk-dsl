"""The schedule space `hk.tune` searches, behind a small provider interface.

A provider answers, for one arch and one M x N x K:

    current(m, n, k)        the candidate the shape runs today (its route,
                            without this machine's records)
    seeds(m, n, k)          candidates known to be good somewhere (the shipped
                            route table's names that take the shape)
    generate(m, n, k)       the space, enumerated over every kernel kind's knobs
    check(c, m, n, k)       "" if the candidate can be built for the shape,
                            else why not -- arithmetic only, no compile
    neighbors(c, m, n, k)   one-knob moves from a candidate, for local search
    jobs(c, m, n, k)        the kernels to compile for it (hk.runtime.warm)
    runner(c, m, n, k)      fn(a, b, out) that runs it
    spec(c) / from_spec     the record's [kind, schedule] form

A candidate is (kind, schedule): on gfx950 the kinds of hk.ops.gemm_cdna's
CANDIDATES -- "pp" (the pingpong, schedule merged over schedule_for's), "dp"
(gemm_cdna_dp's blocks), "decode" (decode_gemm, overrides of its schedule),
"ppw" (a wave-cut composite; seeds only) -- and on gfx1100 "rdna"
(hk.ops.gemm's kernel, the full knob set) and "decode". That is the record
format the routes already use, so a tuned candidate runs through the same
code (gemm_cdna.run_spec, gemm.matmul) as a shipped one.

Today the knobs are the kernels' keyword arguments; when hk.schedules lands
(K1: one Schedule type per kernel family, with check() and space()), a
provider over it replaces these two without the search changing.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import cost

__all__ = ["Candidate", "Gfx950Space", "RdnaSpace", "provider"]


def _norm(v):
    return list(v) if isinstance(v, tuple) else v


@dataclass(frozen=True)
class Candidate:
    """One point: a kernel kind and its schedule (as the owning module spells
    it). Hashable; `label` is short and stable; `name` is the shipped
    candidate's name when it is one."""

    kind: str
    items: Tuple[Tuple[str, Any], ...]
    name: str = ""

    @staticmethod
    def of(kind: str, sched: Dict[str, Any], name: str = "") -> "Candidate":
        return Candidate(kind, tuple(sorted((k, v) for k, v in sched.items())), name)

    @property
    def sched(self) -> Dict[str, Any]:
        return dict(self.items)

    @property
    def label(self) -> str:
        abbrev = {"block_m": "", "block_n": "x", "k_step": "k", "stages": "s",
                  "warps_m": "w", "warps_n": "x", "splits": "sk", "xcd_chunk": "c",
                  "wgm": "g", "lds_shape": "l", "nt": "nt", "setprio": "sp",
                  "pipelined": "pl", "early_store": "es", "late_b": "lb",
                  "transposed_c": "tc", "slices": "sl", "warps": "w", "combine_depth": "cd",
                  "resident": "r", "warp_rows": "wr", "n_split": "ns", "dot_slice": "d",
                  "write_pos": "p", "lds_buffers": "b", "stream_a": "a", "sched": "i",
                  "stagger": "t", "rem": "rem"}
        parts = []
        for k, v in self.items:
            if isinstance(v, bool):
                v = int(v)
            if v is None:
                v = "-"
            parts.append(f"{abbrev.get(k, k)}{v}")
        return self.kind + ":" + "_".join(parts)

    def __str__(self) -> str:
        return self.name or self.label

    def __eq__(self, other):
        return isinstance(other, Candidate) and (self.kind, self.items) == (other.kind, other.items)

    def __hash__(self):
        return hash((self.kind, self.items))


def provider(arch: str):
    """The space provider for an arch."""
    if arch == "gfx950":
        return Gfx950Space()
    if arch.startswith("gfx11") or arch.startswith("gfx12"):
        return RdnaSpace(arch)
    raise ValueError(f"hk.tune: no GEMM schedule space for {arch}")


def _top(cands: Sequence[Candidate], est: Callable[[Candidate], float], n: int) -> List[Candidate]:
    return sorted(cands, key=lambda c: (est(c), c.label))[:n]


# ---------------------------------------------------------------- gfx950


class Gfx950Space:
    """hk's gfx950 GEMMs: the pingpong, the four-warp blocks, the decode GEMV."""

    arch = "gfx950"
    #: Structural axes the space enumerates; the fine axes (swizzles,
    #: priorities, non-temporal copies) are local search's.
    DP_BLOCKS_M = (32, 64, 128, 256)
    DP_BLOCKS_N = (32, 48, 64, 96, 128, 192, 224, 256)
    DP_WARPS = ((2, 2), (2, 1), (1, 2), (4, 1), (1, 4), (2, 4), (4, 2))
    DP_KS = ((64, 2), (64, 3), (64, 4), (128, 2), (128, 3), (256, 2))
    SPLITS = (1, 2, 4, 8)
    LDS = 160000

    def _mods(self):
        from ..ops import decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

        return gemm_cdna, gemm_cdna_dp, decode_gemm

    # -- what there is

    def current(self, m: int, n: int, k: int) -> Candidate:
        gc, _, _ = self._mods()
        name = gc.route(m, n, k, user=False)
        kind, sched = gc.candidate(name)
        return Candidate.of(kind, sched, name)

    def seeds(self, m: int, n: int, k: int) -> List[Candidate]:
        gc, _, _ = self._mods()
        return [Candidate.of(kind, sched, name) for name, (kind, sched) in gc.CANDIDATES.items()
                if gc.fits(name, m, n, k)]

    def full(self, c: Candidate, m: int, n: int, k: int) -> Dict[str, Any]:
        """The candidate's schedule with every knob its kernel takes."""
        gc, dp, dec = self._mods()
        if c.kind == "pp":
            return {**gc.DEFAULT, **gc.schedule_for(m, n, k), **c.sched}
        if c.kind == "dp":
            return {**dp.DEFAULT, **c.sched}
        if c.kind == "decode":
            base = dec.schedule_for(m, k)
            return dec._settle({**base, **c.sched}) if base else dict(c.sched)
        return dict(c.sched)

    def estimate(self, c: Candidate, m: int, n: int, k: int) -> float:
        if c.kind == "ppw":
            # A wave-cut composite: the pingpong on the whole waves, and its
            # remainder candidate on the rest.
            gc, _, _ = self._mods()
            cut = gc.wave_cut(m, n)
            rem = Candidate.of(*gc.candidate(c.sched["rem"]))
            return (cost.estimate("pp", self.full(Candidate.of("pp", {}), m, cut, k), m, cut, k, self.arch)
                    + self.estimate(rem, m, n - cut, k))
        return cost.estimate(c.kind, self.full(c, m, n, k), m, n, k, self.arch)

    def check(self, c: Candidate, m: int, n: int, k: int) -> str:
        gc, dp, dec = self._mods()
        if not gc.fits_spec(c.kind, c.sched, m, n, k):
            return "does not take the shape (K, N or the split's scratch)"
        s = self.full(c, m, n, k)
        try:
            if c.kind == "pp":
                gc.kernel_for(s)
            elif c.kind == "dp":
                bm, bn = s["block_m"], s["block_n"]
                if s["stages"] * (bm + bn) * s["k_step"] * 2 > self.LDS:
                    return "LDS: stages x (block_m + block_n) x k_step x 2 bytes past 160 KB"
                acc = (bm // s["warps_m"]) * (bn // s["warps_n"]) // 64
                if acc > 128:
                    return f"{acc} accumulator registers a lane"
                dp.kernel_for(s)
            elif c.kind == "decode":
                if not s:
                    return "decode does not take this M"
                dec.kernel_for(**{x: s[x] for x in ("tokens", "slices", "warps",
                                                    "combine_depth", "resident", "splits")})
        except (ValueError, TypeError, KeyError) as e:
            return f"{type(e).__name__}: {str(e).splitlines()[0]}"
        return ""

    # -- the space

    def structure(self, c: Candidate, m: int, n: int, k: int) -> tuple:
        """What the analytic model can tell apart: kind, block, K split."""
        s = self.full(c, m, n, k)
        if c.kind == "decode":
            return ("decode", s.get("splits"), s.get("slices"))
        return (c.kind, s.get("block_m"), s.get("block_n"), s.get("splits", 1))

    def preference(self, c: Candidate) -> tuple:
        """Nearness to the kernel's defaults, for a structure's representative."""
        s = c.sched
        if c.kind == "dp":
            return ((s.get("warps_m", 2), s.get("warps_n", 2)) != (2, 2),
                    abs(s.get("stages", 3) - 3), s.get("k_step", 64) != 64, bool(s.get("nt")))
        if c.kind == "pp":
            return ((s.get("warps_m", 2), s.get("warps_n", 4)) != (2, 4), bool(s.get("nt")))
        if c.kind == "decode":
            return (s.get("warps", 16) != 16, s.get("combine_depth", 0), s.get("resident", 0))
        return ()

    def generate(self, m: int, n: int, k: int) -> List[Candidate]:
        out: List[Candidate] = []
        # The pingpong: its block, its warp grid, a K split, streamed weights.
        for bm, bn, (wm, wn), splits, nt in itertools.product(
                (128, 256), (128, 256), ((2, 4), (4, 2)), (1, 2, 4), ("", "b")):
            s = dict(block_m=bm, block_n=bn, warps_m=wm, warps_n=wn, splits=splits)
            if nt:
                s.update(nt="b", lds_shape="16x64")
            out.append(Candidate.of("pp", s))
        # The four-warp kernel's blocks (and its two- and eight-warp grids).
        for bm, bn, (wm, wn), (ks, st), splits in itertools.product(
                self.DP_BLOCKS_M, self.DP_BLOCKS_N, self.DP_WARPS, self.DP_KS, self.SPLITS):
            if (bm // wm) % 16 or (bn // wn) % 16 or n % bn:
                continue
            if st * (bm + bn) * ks * 2 > self.LDS:
                continue
            if (bm // wm) * (bn // wn) // 64 > 128:
                continue
            threads = 64 * wm * wn
            if bm * ks * 2 % (16 * threads) or bn * ks * 2 % (16 * threads):
                continue
            s = dict(block_m=bm, block_n=bn, warps_m=wm, warps_n=wn, k_step=ks, stages=st,
                     splits=splits, xcd_chunk=0)
            if ks >= 128 and bm <= 64:
                s.update(nt="b", lds_shape="16x64")
            out.append(Candidate.of("dp", s))
        _, _, dec = self._mods()
        if dec.schedule_for(m, k) is not None:
            for sl, w, cd, r, sp in itertools.product((1, 2, 4, 8), (4, 8, 16), (0, 4, 8),
                                                     (0, 1, 2), (1, 2, 4)):
                out.append(Candidate.of("decode", dict(slices=sl, warps=w, combine_depth=cd,
                                                       resident=r, splits=sp)))
        return out

    FINE = {
        "pp": dict(wgm=(1, 2, 4, 8, 16), xcd_chunk=(8, 16, 32, 64), setprio=(True, False),
                   early_store=(True, False)),
        "dp": dict(wgm=(1, 2, 4, 8, 16), xcd_chunk=(0, 8, 16, 32, 64), setprio=(False, True),
                   pipelined=(True, False), stages=(2, 3, 4), k_step=(64, 128, 256)),
        "decode": dict(slices=(1, 2, 4, 8), warps=(2, 4, 8, 16), combine_depth=(0, 2, 4, 8),
                       resident=(0, 1, 2, 4), splits=(1, 2, 4, 8)),
    }

    def neighbors(self, c: Candidate, m: int, n: int, k: int) -> List[Candidate]:
        axes = self.FINE.get(c.kind)
        if not axes:
            return []
        full = self.full(c, m, n, k)
        out = []
        for ax, vals in axes.items():
            for v in vals:
                if full.get(ax) == v:
                    continue
                s = {**c.sched, ax: v}
                if c.kind == "dp" and ax == "k_step" and v >= 128 and full["block_m"] <= 64:
                    s.update(nt="b", lds_shape="16x64")
                out.append(Candidate.of(c.kind, s))
        # The warp grid.
        if c.kind in ("pp", "dp"):
            grids = ((2, 4), (4, 2)) if c.kind == "pp" else self.DP_WARPS
            for wm, wn in grids:
                if (wm, wn) != (full.get("warps_m"), full.get("warps_n")):
                    out.append(Candidate.of(c.kind, {**c.sched, "warps_m": wm, "warps_n": wn}))
        # Streamed weights on or off.
        if c.kind in ("pp", "dp"):
            if full.get("nt"):
                s = {**c.sched, "nt": "", "lds_shape": None}
            else:
                s = {**c.sched, "nt": "b", "lds_shape": "16x64"}
            out.append(Candidate.of(c.kind, s))
        return out

    # -- running it

    def jobs(self, c: Candidate, m: int, n: int, k: int):
        gc, dp, dec = self._mods()
        s = self.full(c, m, n, k)
        if c.kind == "pp":
            return [(c.label, gc.kernel_for(s), {})]
        if c.kind == "dp":
            return [(c.label, dp.kernel_for(s), {})]
        if c.kind == "decode" and s:
            return [(c.label, dec.kernel_for(**{x: s[x] for x in (
                "tokens", "slices", "warps", "combine_depth", "resident", "splits")}), {})]
        return []

    def runner(self, c: Candidate, m: int, n: int, k: int):
        gc, _, _ = self._mods()
        kind, sched = c.kind, c.sched

        def run(a, b, out):
            return gc.run_spec(kind, sched, a, b, out)
        return run

    def spec(self, c: Candidate):
        return [c.kind, {k: _norm(v) for k, v in c.items}]

    def from_spec(self, spec, name: str = "") -> Candidate:
        return Candidate.of(spec[0], dict(spec[1]), name)


# ---------------------------------------------------------------- RDNA


class RdnaSpace:
    """hk's RDNA GEMM (hk.ops.gemm, every knob) and the decode GEMV."""

    def __init__(self, arch: str = "gfx1100"):
        self.arch = arch if arch in cost.CARDS else "gfx1100"
        self._arch = arch

    def _mods(self):
        from ..ops import decode_gemm, gemm  # noqa: PLC0415

        return gemm, decode_gemm

    def current(self, m: int, n: int, k: int) -> Candidate:
        g, _ = self._mods()
        s = g.schedule_for(m, n, k, user=False)
        name = "thin" if s == g.THIN else ("default" if s == g.DEFAULT else "")
        return Candidate.of("rdna", s, name)

    def seeds(self, m: int, n: int, k: int) -> List[Candidate]:
        g, dec = self._mods()
        out = [Candidate.of("rdna", dict(g.DEFAULT), "default"),
               Candidate.of("rdna", dict(g.THIN), "thin")]
        if dec.fits(m, n, k):
            out.append(Candidate.of("decode", {}, "decode"))
        return [c for c in out if not self.check(c, m, n, k)]

    def full(self, c: Candidate, m: int, n: int, k: int) -> Dict[str, Any]:
        g, dec = self._mods()
        if c.kind == "rdna":
            return {**g.DEFAULT, **c.sched}
        base = dec.schedule_for(m, k)
        return dec._settle({**base, **c.sched}) if base else dict(c.sched)

    def estimate(self, c: Candidate, m: int, n: int, k: int) -> float:
        return cost.estimate(c.kind, self.full(c, m, n, k), m, n, k, self.arch)

    def check(self, c: Candidate, m: int, n: int, k: int) -> str:
        g, dec = self._mods()
        s = self.full(c, m, n, k)
        try:
            if c.kind == "rdna":
                if k % s["k_step"] or n % s["block_n"]:
                    return f"K={k} or N={n} not whole k_step={s['k_step']} / block_n={s['block_n']}"
                reg_m = s["block_m"] // s["warp_rows"]
                reg_n = s["block_n"] // (s["warps"] // max(1, s["warp_rows"]))
                if reg_m < 16 or reg_m % 16:
                    return f"a {reg_m}-row warp tile"
                if reg_m * reg_n // 32 > 128:
                    return f"{reg_m * reg_n // 32} accumulator registers a lane"
                g._make(**s)
            elif c.kind == "decode":
                if not s or not dec.fits(m, n, k):
                    return "decode does not take this shape"
                if k % (16 * s["slices"] * s["warps"] * s["splits"]):
                    return "K is not whole rounds of every warp"
                dec.kernel_for(**{x: s[x] for x in ("tokens", "slices", "warps",
                                                    "combine_depth", "resident", "splits")})
            else:
                return f"no kind {c.kind} on {self._arch}"
        except (ValueError, TypeError, KeyError) as e:
            return f"{type(e).__name__}: {str(e).splitlines()[0]}"
        return ""

    def structure(self, c: Candidate, m: int, n: int, k: int) -> tuple:
        s = self.full(c, m, n, k)
        if c.kind == "decode":
            return ("decode", s.get("splits"), s.get("slices"))
        return ("rdna", s["block_m"], s["block_n"], s["lds_buffers"])

    def preference(self, c: Candidate) -> tuple:
        s = c.sched
        if c.kind == "decode":
            return (s.get("warps", 16) != 16, s.get("combine_depth", 0), s.get("resident", 0))
        return (s.get("warps", 8) != 8, s.get("k_step", 64) != 64, s.get("n_split", 2) != 2,
                s.get("warp_rows", 4) != 4)

    def generate(self, m: int, n: int, k: int) -> List[Candidate]:
        out: List[Candidate] = []
        for bm, bn, ks, warps, wr, ns, buf in itertools.product(
                (16, 32, 64, 128, 256), (64, 128, 256), (32, 64, 128), (4, 8),
                (1, 2, 4, 8), (1, 2, 4), (1, 2)):
            if wr > warps or warps % wr:
                continue
            reg_m, reg_n = bm // wr, bn // (warps // wr)
            if reg_m < 16 or reg_m % 16 or reg_n % (16 * ns) or n % bn or k % ks:
                continue
            if buf * (bm + bn) * ks * 2 > 65536 or reg_m * reg_n // 32 > 128:
                continue
            out.append(Candidate.of("rdna", dict(block_m=bm, block_n=bn, k_step=ks, warps=warps,
                                                 warp_rows=wr, n_split=ns, lds_buffers=buf,
                                                 write_pos=2, setprio=bm <= 32)))
        _, dec = self._mods()
        if dec.fits(m, n, k):
            for sl, w, cd, r, sp in itertools.product((1, 2, 4, 8), (4, 8, 16), (0, 4, 8),
                                                     (0, 1, 2), (1, 2, 4)):
                out.append(Candidate.of("decode", dict(slices=sl, warps=w, combine_depth=cd,
                                                       resident=r, splits=sp)))
        return out

    FINE = {
        "rdna": dict(wgm=(1, 2, 4, 8, 16), write_pos=(0, 1, 2), setprio=(False, True),
                     stream_a=(False, True), sched=(0, 1, 2, 4), dot_slice=(16, 32),
                     n_split=(1, 2, 4), k_step=(32, 64, 128), warps=(4, 8),
                     warp_rows=(1, 2, 4, 8)),
        "decode": Gfx950Space.FINE["decode"],
    }

    def neighbors(self, c: Candidate, m: int, n: int, k: int) -> List[Candidate]:
        full = self.full(c, m, n, k)
        out = []
        for ax, vals in self.FINE.get(c.kind, {}).items():
            for v in vals:
                if full.get(ax) != v:
                    out.append(Candidate.of(c.kind, {**c.sched, ax: v}))
        return out

    def jobs(self, c: Candidate, m: int, n: int, k: int):
        g, dec = self._mods()
        s = self.full(c, m, n, k)
        if c.kind == "rdna":
            return [(c.label, g.TUNER._kernel_for(s), {})]
        if c.kind == "decode" and s:
            return [(c.label, dec.kernel_for(**{x: s[x] for x in (
                "tokens", "slices", "warps", "combine_depth", "resident", "splits")}), {})]
        return []

    def runner(self, c: Candidate, m: int, n: int, k: int):
        g, dec = self._mods()
        if c.kind == "decode":
            sched = c.sched

            def run(a, b, out):
                return dec.matmul(a, b, out=out, **sched)
            return run
        kern = g.TUNER._kernel_for(self.full(c, m, n, k))

        def run(a, b, out):
            kern(a, b, out)
            return out
        return run

    def spec(self, c: Candidate):
        return [c.kind, {k: _norm(v) for k, v in c.items}]

    def from_spec(self, spec, name: str = "") -> Candidate:
        return Candidate.of(spec[0], dict(spec[1]), name)


def describe(c: Candidate) -> str:
    return json.dumps({"kind": c.kind, **c.sched}, sort_keys=True)
