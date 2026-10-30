"""hk.schedules round trip: everything hk ships goes through the schedule
types and comes out as the same kernels, byte for byte.

For every point of routes_gemm_gfx950.json, every gfx950 route candidate,
every gemm_bf16.json record, the decode GEMV's schedules and a grid of
shapes on both GPUs: the schedule `hk.schedules.lookup` / `from_candidate`
gives (and its record, read back) launches the kernels hk.ops does -- same
names, same columns, byte-identical C++ (`k.source()`) -- and
hk.templates.gemm's plan through the schedule is its plan without one, with
an epilogue's kernels identical too.

The reference for gfx950 is gemm_cdna.run itself, run on stand-in tensors
with the launches recorded rather than made; for gfx1100, hk.ops.gemm's
schedule_for and its tuner's kernels (matmul's two lines); for the
templates, kernel_for as it was before the schedule types made its kernels
(_legacy_kernel_for below). No GPU, no torch, no hipcc.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
import hk.schedules as S  # noqa: E402
from hk import target as _target  # noqa: E402
_kmod = importlib.import_module("hk.dsl.kernel")
from hk.ops import _epilogue, decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: E402
from hk.ops import gemm as rdna  # noqa: E402

TG = importlib.import_module("hk.templates.gemm")
SG = importlib.import_module("hk.schedules.gemm")
TUNED = Path(rdna.__file__).resolve().parents[1] / "tuned"
ROUTES = json.loads((TUNED / "routes_gemm_gfx950.json").read_text())

#: Llama-3-8B's and Qwen3-8B's linears (gemm_cdna_routes.py's grid), and Ms.
NK = [(4096, 4096), (6144, 4096), (12288, 4096), (14336, 4096), (24576, 4096),
      (28672, 4096), (4096, 12288), (4096, 14336), (128256, 4096), (151936, 4096)]
MS = [1, 4, 16, 32, 64, 77, 128, 256, 300, 512, 1024, 2048, 4096, 8192]


@contextlib.contextmanager
def on(arch: str):
    """Build for `arch` as a process on that GPU would: current_arch and every
    kernel cache hk.ops and hk.schedules keep, set aside and restored."""
    caches = [gemm_cdna._KERNELS, gemm_cdna_dp._KERNELS, decode_gemm._KERNELS,
              rdna.TUNER._kernels, SG._KERNELS, TG._KERNELS]
    saved = [dict(c) for c in caches]
    cur = list(_target._CURRENT)
    for c in caches:
        c.clear()
    _target._CURRENT[:] = [arch]
    try:
        yield
    finally:
        _target._CURRENT[:] = cur
        for c, s in zip(caches, saved):
            c.clear()
            c.update(s)


_SRC = {}


def src(kernel) -> str:
    key = (kernel.name, kernel.arch, id(kernel))
    if key not in _SRC:
        _SRC[key] = (kernel, kernel.source())
    return _SRC[key][1]


# ---------------------------------------------------------------- gfx950 run()


class Fake:
    """A stand-in for a tensor: a shape, slicing, a device."""

    device = "fake"

    def __init__(self, *shape, name="t"):
        self.shape = tuple(shape)
        self.name = name

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = list(self.shape)
        for d, sl in enumerate(idx):
            if isinstance(sl, slice):
                lo, hi, _ = sl.indices(shape[d])
                shape[d] = hi - lo
        f = Fake(*shape, name=self.name)
        f.col0 = getattr(self, "col0", 0) + (idx[1].indices(self.shape[1])[0]
                                             if len(idx) > 1 else 0)
        f.row0 = getattr(self, "row0", 0) + (idx[0].indices(self.shape[0])[0]
                                             if isinstance(idx[0], slice) else 0)
        return f


def run_launches(name: str, m: int, n: int, k: int, monkeypatch):
    """[(kernel, what, cols)]: what gemm_cdna.run(name) launches at M x N x K,
    recorded. `what` is "out", "scratch", "reduce" or "decode"."""
    rec = []

    def call(self, *ts, **kw):
        out = ts[2]
        if out.name == "ws":
            rec.append((self, "scratch", (out.c0, out.c0 + out.shape[2])))
        else:
            rec.append((self, "out", (getattr(out, "col0", 0),
                                      getattr(out, "col0", 0) + out.shape[1])))

    def scratch_for(device, splits, mm, nn):
        f = Fake(splits, mm, nn, name="ws")
        f.c0 = rec_cols[0]
        return f

    def reduce(ws, out, **kw):
        rec.append((None, "reduce", (getattr(out, "col0", 0),
                                     getattr(out, "col0", 0) + out.shape[1])))

    def decode(x, w, out=None):
        s = decode_gemm.schedule_for(x.shape[0], x.shape[1])
        rec.append((decode_gemm.kernel_for(**s), "out",
                    (getattr(out, "col0", 0), getattr(out, "col0", 0) + out.shape[1])))

    rec_cols = [0]

    def b_slice_tracker(fn):
        def wrapped(nm, a, b, out):
            rec_cols[0] = getattr(out, "col0", 0)
            return fn(nm, a, b, out)
        return wrapped

    monkeypatch.setattr(_kmod.Kernel, "__call__", call)
    monkeypatch.setattr(gemm_cdna_dp, "scratch_for", scratch_for)
    monkeypatch.setattr(gemm_cdna_dp, "reduce", reduce)
    monkeypatch.setattr(decode_gemm, "matmul", decode)
    monkeypatch.setattr(gemm_cdna, "run", b_slice_tracker(gemm_cdna.run))
    gemm_cdna.run(name, Fake(m, k, name="a"), Fake(n, k, name="b"), Fake(m, n, name="out"))
    monkeypatch.undo()
    return rec


def schedule_launches(s, m, n, k):
    """The same, from the schedule's steps: each kernel the one the schedule
    launches (the kernel hk.ops caches, by name) and, beside it, one made
    afresh from the schedule's knobs alone."""
    out = []
    for st in s.steps(m, n, k):
        if st.kernel == "reduce":
            out.append((None, "reduce", st.cols))
        else:
            assert st.schedule.kernel().name == st.schedule.name
            out.append((st.schedule._make(None, None),
                        "scratch" if st.into == "scratch" else "out", st.cols))
    return out


COMPARED = []


def same(old, new, what):
    assert [(w, c) for _, w, c in old] == [(w, c) for _, w, c in new], what
    for (ko, _, c), (kn, _, _) in zip(old, new):
        if ko is None:
            continue
        assert ko is not kn
        assert ko.name == kn.name, f"{what} cols {c}: {ko.name} != {kn.name}"
        assert src(ko) == src(kn), f"{what} cols {c}: {ko.name}: the C++ differs"
        COMPARED.append(ko.name)


def roundtrip(s):
    for full in (False, True):
        rec = json.loads(json.dumps(s.to_record(full=full)))
        assert S.from_record(rec) == s, rec
    if not isinstance(s, S.Columns):
        fam, knobs = s.family, s.to_record()
        knobs.pop("family")
        assert S.from_record([fam, knobs]) == s
        assert type(s).from_record(knobs) == s


def gfx950_shapes():
    pts = {(p["m"], p["n"], p["k"]) for p in ROUTES["points"]}
    pts |= {(m, n, k) for n, k in NK for m in MS}
    pts |= {(2048, 16384, 4096), (4096, 4096, 4096), (8192, 8192, 8192), (64, 151936, 4096)}
    return sorted(pts)


def test_every_gfx950_route_is_the_same_kernels(monkeypatch):
    with on("gfx950"):
        for m, n, k in gfx950_shapes():
            name = gemm_cdna.route(m, n, k)
            s = S.lookup(m, n, k, arch="gfx950")
            assert s.check(m, n, k, "gfx950") == [], (m, n, k, name, s)
            roundtrip(s)
            same(run_launches(name, m, n, k, monkeypatch), schedule_launches(s, m, n, k),
                 f"{m}x{n}x{k} [{name}] {s}")
    assert len(set(COMPARED)) >= 20, sorted(set(COMPARED))


@pytest.mark.parametrize("name", sorted(gemm_cdna.CANDIDATES))
def test_every_gfx950_candidate_is_the_same_kernels(name, monkeypatch):
    shapes = [(m, n, k) for m in (16, 256, 2048) for n, k in
              ((4096, 4096), (12288, 4096), (28672, 4096), (151936, 4096),
                                    (4096, 14336))]
    with on("gfx950"):
        tried = 0
        for m, n, k in shapes:
            if not gemm_cdna.fits(name, m, n, k):
                continue
            tried += 1
            s = S.from_candidate(name, m, n, k)
            assert s.check(m, n, k, "gfx950") == [], (m, n, k, name, s)
            roundtrip(s)
            same(run_launches(name, m, n, k, monkeypatch), schedule_launches(s, m, n, k),
                 f"{m}x{n}x{k} [{name}] {s}")
        assert tried, f"{name} takes none of the shapes tried"


def test_the_route_tables_candidates_are_the_codes():
    """The table's candidates section is what the code calls them, so
    from_record of it is from_candidate's schedule wherever the candidate's
    knobs are not shape-dependent (dp's are not)."""
    for nm, c in ROUTES["candidates"].items():
        assert tuple(c) == tuple(gemm_cdna.CANDIDATES[nm]) or list(c) == list(
            gemm_cdna.CANDIDATES[nm])
        if c[0] == "dp":
            assert S.from_record(c) == S.from_candidate(nm, 256, 4096, 4096)


# ---------------------------------------------------------------- gfx1100


def gfx1100_shapes():
    pts = {(m, n, k) for n, k in NK for m in MS}
    for key in json.loads((TUNED / "gemm_bf16.json").read_text()):
        pts.add(tuple(int(x) for x in key.split("x")))
    return sorted(pts)


def test_every_gfx1100_matmul_schedule_is_the_same_kernel():
    with on("gfx1100"):
        for m, n, k in gfx1100_shapes():
            sched = rdna.schedule_for(m, n, k)
            if k % sched["k_step"] or n % sched["block_n"]:
                continue
            old = rdna.TUNER._kernel_for(sched)
            s = S.lookup(m, n, k, arch="gfx1100")
            assert s.check(m, n, k, "gfx1100") == [], (m, n, k, s)
            roundtrip(s)
            kn = s.kernel()
            assert kn is old            # the very kernel matmul launches
            fresh = S.from_record(s.to_record()).kernel(arch="gfx1100")
            assert fresh.name == old.name and src(fresh) == src(old), (m, n, k)


def test_the_tuned_records_load():
    """gemm_bf16.json's records are Rdna records: every one reads back and
    is the schedule the tuner's lookup gives."""
    recs = json.loads((TUNED / "gemm_bf16.json").read_text())
    for key, r in recs.items():
        s = S.Rdna.from_record(r["schedule"], strict=False)
        assert s.legacy() == {**rdna.DEFAULT, **{k: v for k, v in r["schedule"].items()
                                                 if k in rdna.TUNER.knobs}}, key
    for sched in [rdna.DEFAULT, rdna.THIN] + rdna.SPACE:
        s = S.Rdna.from_record(sched)
        assert s.legacy() == {**rdna.DEFAULT, **sched}
        roundtrip(s)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_every_decode_schedule_is_the_same_kernel(arch):
    with on(arch):
        seen = set()
        for m in range(1, 16 * decode_gemm.MAX_TOKENS + 1):
            for k in (1024, 2048, 3072, 4096, 12288, 14336):
                s = decode_gemm.schedule_for(m, k)
                if s is None or tuple(sorted(s.items())) in seen:
                    continue
                seen.add(tuple(sorted(s.items())))
                d = S.Decode(**s)
                assert d.check(m, 4096, k, arch) == [], (m, k, d)
                roundtrip(d)
                old = decode_gemm.kernel_for(**s)
                assert d.kernel() is old
                fresh = d._make(None, arch)
                assert fresh.name == old.name and src(fresh) == src(old), (m, k)
        assert len(seen) >= 3


# ---------------------------------------------------------------- templates


def _legacy_kernel_for(step, ep, arch):
    """hk.templates.gemm.kernel_for before hk.schedules made its kernels."""
    e = ep if step.epilogue else None
    s = step.sched
    if step.kernel == "rdna":
        s = {**rdna.DEFAULT, **s}
        k = rdna.gemm_kernel(_epilogue.name(rdna._sched_name(s), e), **s, epilogue=e)
        if arch:
            k._arch = arch
    elif step.kernel == "pp":
        s = {**gemm_cdna.DEFAULT, **s}
        k = gemm_cdna.gemm_kernel(_epilogue.name(gemm_cdna._sched_name(s), e), **s,
                                  epilogue=e, arch=arch)
    elif step.kernel == "dp":
        s = {**gemm_cdna_dp.DEFAULT, **s}
        k = gemm_cdna_dp.gemm_kernel(_epilogue.name(gemm_cdna_dp._sched_name(s), e),
                                     **s, epilogue=e, arch=arch)
    else:
        nm = (f"decode_gemm_t{s['tokens']}_s{s['slices']}_w{s['warps']}"
              f"_c{s['combine_depth']}_r{s['resident']}"
              + (f"_k{s['splits']}" if s["splits"] > 1 else "")
              + (f"_o{s['occupancy']}" if s.get("occupancy") else ""))
        k = decode_gemm.decode_gemm_kernel(_epilogue.name(nm, e), **s, epilogue=e,
                                           arch=arch)
    return k


def _ep(fn, kinds):
    return TG.Epilogue(fn, kinds, {n: hk.bf16 for n in kinds}, {}, hk.bf16)


EPILOGUES = {
    "bias_gelu": lambda: _ep(TG.bias_gelu, {"bias": "row"}),
    "residual": lambda: _ep(TG.residual, {"residual": "tile"}),
}


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_the_templates_plan_is_the_schedules(arch):
    """plan() without a schedule and with lookup(op="template")'s are the
    same launches; every kernel they make, with an epilogue or without, is
    the C++ the template made before the schedule types."""
    shapes = gfx950_shapes() if arch == "gfx950" else gfx1100_shapes()
    with on(arch):
        done = set()
        for m, n, k in shapes:
            try:
                old = TG.plan(m, n, k, arch=arch)
            except ValueError:
                continue
            s = S.lookup(m, n, k, arch=arch, op="template")
            new = TG.plan(m, n, k, arch=arch, schedule=s)
            assert [(st.kernel, st.cols, st.epilogue, st.into) for st in old] == \
                [(st.kernel, st.cols, st.epilogue, st.into) for st in new], (m, n, k)
            for so, sn in zip(old, new):
                if so.kernel == "reduce":
                    assert so.sched == sn.sched
                    continue
                assert S.family(so.kernel).from_record(so.sched) == sn.schedule
                key = sn.schedule
                if key in done:
                    continue
                done.add(key)
                for ep in [None] + [f() for f in EPILOGUES.values()]:
                    if ep is not None and (not so.epilogue or (
                            ep.init and so.kernel == "pp" and so.sched.get("transposed_c"))):
                        continue
                    ko = _legacy_kernel_for(so, ep, arch)
                    kn = TG.kernel_for(sn, ep, arch=arch)
                    assert ko.name == kn.name, (m, n, k, ko.name, kn.name)
                    assert src(ko) == src(kn), (m, n, k, ko.name)
        assert done


def test_a_schedule_in_the_template_is_checked():
    bad = S.Dp(block_n=96)
    with pytest.raises(S.Illegal, match="block_n=96"):
        TG.plan(4096, 4096, 4096, arch="gfx950", schedule=bad)
    steps = TG.plan(4096, 4096, 4096, arch="gfx950", schedule=S.Dp(splits=4))
    assert [(st.kernel, st.into, st.epilogue) for st in steps] == [
        ("dp", "scratch", False), ("reduce", "out", True)]
