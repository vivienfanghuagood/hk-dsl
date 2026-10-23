"""hk.schedules for attention: the same protocol over (n, head_dim, causal),
and the round trip -- what hk.attention launches today, through the
schedule types, is the same kernel, byte for byte. No GPU, no hipcc."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk.schedules as S  # noqa: E402
from hk import target as _target  # noqa: E402
from hk.ops import attn, attn_cdna  # noqa: E402

TUNED = Path(attn.__file__).resolve().parents[1] / "tuned"
SA = __import__("hk.schedules.attention", fromlist=["_KERNELS"])


@contextlib.contextmanager
def on(arch):
    caches = [attn_cdna._KERNELS, attn._SCALED, SA._KERNELS] + [
        t._kernels for t in attn.TUNERS.values()]
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


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
def test_what_hk_attention_runs_is_the_same_kernel(arch):
    seen = 0
    with on(arch):
        for d in attn.HEAD_DIMS:
            for causal in (False, True):
                for n in (192, 4096, 16384, 65536):
                    for scale in (None, 0.1):
                        old = attn._kernel_for(d, causal, scale, n)
                        s = S.lookup_attention(n, d, causal, arch, scale=scale)
                        assert s.check(n, d, arch, causal=causal) == [], (s, n, d)
                        assert S.from_record(json.loads(json.dumps(s.to_record()))) == s
                        new = s._make(d, causal, scale, None)
                        assert new is not old
                        assert new.name == old.name, (new.name, old.name)
                        assert new.source() == old.source(), old.name
                        assert s.kernel(d, causal, scale).name == old.name
                        seen += 1
    assert seen == 32


def test_the_tuned_attention_records_load():
    for f in ("attn_fwd_d128.json", "attn_fwd_d128_causal.json"):
        for key, rec in json.loads((TUNED / f).read_text()).items():
            s = S.AttnRdna.from_record(rec["schedule"])
            assert s.legacy() == {**attn.DEFAULT, **rec["schedule"]}, (f, key)
    legal = 0
    for sched in attn.SPACE:
        s = S.AttnRdna.from_record(sched)
        assert s.legacy() == sched
        if s.check_kernel("gfx1100", head_dim=128):
            continue        # e.g. pv_tiles=3: the factory refuses it too
        legal += 1
        assert s.name_for(128, False) == attn.TUNERS[(128, False)]._kernel_for(sched).name
    assert legal >= 6


def test_the_gfx950_ablations_are_the_same_kernels():
    with on("gfx950"):
        for opts in ({"lazy": 0.0}, {"sched": False}, {"setprio": False, "lpt": False},
                     {"skip": False}):
            old = attn_cdna.kernel_for(128, True, None, **opts)
            new = S.AttnCdna(**opts)._make(128, True, None, None)
            assert new.name == old.name and new.source() == old.source(), opts


def test_check_space_and_reasons():
    s = S.AttnRdna()
    assert s.check(4096, 128, "gfx1100") == []
    assert any("query tile" in w for w in s.check(100, 128, "gfx1100"))
    assert any("head_dim=96" in w for w in s.check(4096, 96, "gfx1100"))
    assert any("kv_block=64 at head_dim 128" in w
               for w in s.replace(kv_block=64).check(4096, 128, "gfx1100"))
    assert s.replace(kv_block=64).check(4096, 64, "gfx1100") == []
    assert any("cross attention" in w for w in s.check(4096, 128, "gfx1100", causal=True,
                                                      n_kv=2048))
    assert any("gfx950" in w for w in S.AttnCdna().check(4096, 128, "gfx1100"))
    rd = list(S.space_attention(4096, 128, "gfx1100"))
    assert rd and all(t.check(4096, 128, "gfx1100") == [] for t in rd)
    cd = list(S.space_attention(4096, 128, "gfx950", causal=True))
    assert len(cd) == 8 and len(set(cd)) == 8
    around = list(S.space_attention(4096, 64, "gfx1100", around=S.AttnRdna(), radius=1))
    assert around[0] == S.AttnRdna() and len(around) > 3
