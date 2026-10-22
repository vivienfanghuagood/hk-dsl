"""hk.schedules' check() is honest: what it accepts builds, what it refuses
says why.

For every GEMM family on every arch it runs on, a fixed sample of its full
value product (every knob's `values`): each point check_kernel() accepts is
built (hipcc, no GPU) and must pass the passes and the resource gate --
trace, no spill, no SGPR spill; each point it refuses carries a reason. The
sample is small here (HK_SCHEDULES_SAMPLE points a family, default 6); the
measured rates come from tools/hk-bench/schedules_legality.py on thousands
of builds (docs/design/schedules.md: 1 false accept in 1,504 accepted, on
samples the register model was not fitted to, under both GPUs' hipcc).
"""

from __future__ import annotations

import os
import random
import zlib

import pytest

pytest.importorskip("hk._mlir")

import hk.schedules as S  # noqa: E402
from hk.runtime.warm import warm_cache  # noqa: E402

N = int(os.environ.get("HK_SCHEDULES_SAMPLE", "6"))
CASES = [(cls, arch) for cls in (S.Rdna, S.Decode, S.PingPong, S.Dp) for arch in cls.arches]


def _draw(cls, arch, n, seed):
    rng = random.Random(seed)
    acc, ref, seen = [], [], set()
    ax = [a for a in cls.axes().values() if a.values]
    tries = 0
    while len(acc) < n and tries < 100000:
        tries += 1
        s = cls(**{a.name: rng.choice(a.values) for a in ax}).canonical()
        if s in seen:
            continue
        seen.add(s)
        why = s.check_kernel(arch)
        (ref if why else acc).append((s, why))
    return acc, ref


@pytest.mark.parametrize("cls,arch", CASES, ids=[f"{c.family}-{a}" for c, a in CASES])
def test_what_check_accepts_builds(cls, arch):
    acc, ref = _draw(cls, arch, N, seed=zlib.crc32(f"{cls.family}@{arch}".encode()))
    assert len(acc) == N
    for s, why in ref:
        assert why and all(isinstance(r, str) and len(r) > 10 for r in why), s
    res = warm_cache([(str(s), s._make(None, arch), {}) for s, _ in acc])
    bad = {k: f"{type(e).__name__}: {str(e).strip().splitlines()[0]}"
           for k, e in res.errors.items()}
    assert not bad, f"{len(bad)} of {len(acc)} accepted {cls.family} schedules did not build on {arch}: {bad}"


def test_the_shipped_defaults_build_with_room():
    """The defaults and the predictions agree on the kernels hk runs most:
    each predicted at or above what it builds to."""
    from hk.schedules import _regs

    cases = [(S.Rdna(), "gfx1100"), (S.PingPong(), "gfx950"), (S.Dp(), "gfx950"),
             (S.Decode(), "gfx1100"), (S.Decode(slices=4, warps=16, resident=2), "gfx950")]
    res = warm_cache([(f"{s}@{a}", s._make(None, a), {}) for s, a in cases])
    assert not res.errors, res.errors
    for s, a in cases:
        r = res.built[f"{s}@{a}"].kernels[0]
        used = r.vgpr if a == "gfx1100" else -(-r.vgpr // 4) * 4 + r.agpr
        assert _regs.predict(s, a)["vgpr"] >= used, (s, a, used)


ATTN = [(S.AttnRdna, "gfx1100"), (S.AttnCdna, "gfx950")]


@pytest.mark.parametrize("cls,arch", ATTN, ids=[c.family for c, _ in ATTN])
def test_what_check_accepts_builds_attention(cls, arch):
    """The attention families' spaces are small enough that the tool builds
    them whole (sample-attn: 0 false accepts in 508); here a fixed few."""
    rng = random.Random(zlib.crc32(cls.family.encode()))
    ax = [a for a in cls.axes().values() if a.values]
    acc, seen = [], set()
    while len(acc) < N:
        s = cls(**{a.name: rng.choice(a.values) for a in ax})
        d, causal = rng.choice((64, 128)), rng.choice((False, True))
        if (s, d, causal) in seen:
            continue
        seen.add((s, d, causal))
        why = s.check_kernel(arch, head_dim=d, causal=causal)
        if why:
            assert all(isinstance(r, str) and len(r) > 10 for r in why), s
            continue
        acc.append((s, d, causal))
    res = warm_cache([(f"{s} d{d} c{int(c)}", s._make(d, c, None, arch), {})
                      for s, d, c in acc])
    assert not res.errors, {k: str(e).splitlines()[0] for k, e in res.errors.items()}
