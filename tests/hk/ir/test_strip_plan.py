"""hk.ops.norm's strip plan: the run, the chunks a warp holds, the warps, and
when a row goes to the tile kernels instead. No GPU, no hipcc."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.ops import norm  # noqa: E402


@pytest.mark.parametrize("cols,kind,nbytes,want", [
    (4096, "rms", 2, (8, 2, 8, 1)),      # two 16-byte runs a lane
    (4096, "addrmsw", 2, (16, 1, 8, 1)), # one for the in-place fused add
    (5120, "rmsw", 2, (10, 2, 8, 1)),    # any warp count, not a power of two
    (4096, "softmax", 4, (8, 4, 4, 1)),  # fp32: a 16-byte run is 4
    (4100, "rms", 2, (9, 4, 4, 0)),      # a run of 4 keeps the rows aligned; bounded
    (129, "layer", 2, (1, 5, 1, 0)),     # odd: a run of 1
    (64, "rms", 2, (1, 1, 8, 0)),        # narrower than a chunk
])
def test_the_strip_plan(cols, kind, nbytes, want):
    assert norm.strip_plan(cols, kind, nbytes) == want


@pytest.mark.parametrize("cols,kind,nbytes,want", [
    (4096, "rmsw", 2, (2, 4, 8, 1)),     # 32 a lane: 8 chunks as 2 warps of 4
    (4096, "addrmsw", 2, (2, 4, 8, 1)),  # the fused add too, on wave64
    (5120, "rms", 2, (2, 5, 8, 1)),      # 10 chunks: the exact split nearest 4 a warp
    (8192, "softmax", 2, (4, 4, 8, 1)),
    (4096, "layer", 4, (2, 8, 4, 1)),    # fp32: runs of 4, so 32 a lane is 8 chunks
    (4100, "rms", 2, (3, 6, 4, 0)),      # 17 chunks: one warp of 17 is too many; ceiling
    (64, "rms", 2, (1, 1, 8, 0)),
])
def test_the_wave64_strip_plan(cols, kind, nbytes, want):
    """A chunk is 64 lanes x RUN; 32 elements a lane, the warps chosen to
    split the chunks exactly (tools/hk-bench/strip_plan.py on an MI355X)."""
    assert norm.strip_plan(cols, kind, nbytes, wave=64) == want


def test_the_wave64_capacity_is_the_same_per_lane():
    assert norm.strip_plan(16 * 64 * 64, "rms", 2, wave=64) == (16, 8, 8, 1)
    assert norm.strip_plan(16 * 64 * 64 + 512, "rms", 2, wave=64) is None


def test_a_row_past_a_strips_capacity_goes_to_the_tiles():
    assert norm.strip_plan(16 * 64 * 32, "rms", 2) is not None
    assert norm.strip_plan(16 * 64 * 32 + 256, "rms", 2) is None




class _BF16:
    """Enough of torch.bfloat16 for the plan, without torch."""
    itemsize = 2

    def __str__(self):
        return "torch.bfloat16"


def test_the_occupancy_cap_starts_past_the_infinity_cache(monkeypatch):
    """48 KB of LDS reserved -- two workgroups a WGP -- only when a call moves
    more than the 96 MB the cache holds: in it the cap costs (18.4 -> 22.3 us
    at 2048x4096), past it it pays (177 -> 174 at 8192x4096)."""
    monkeypatch.setattr(norm, "_strips_here", lambda: True)
    monkeypatch.setattr(norm, "_PLAN_CACHE", {})
    dt = _BF16()
    assert norm._resolve_strip("rms", dt, 4096, 2048)[4] == 0              # 33.5 MB
    assert norm._resolve_strip("rms", dt, 4096, 8192)[4] == 48 << 10       # 134 MB
    assert norm._resolve_strip("addrmsw", dt, 4096, 4096)[4] == 48 << 10   # 4 streams
    k = norm._resolve_strip("rms", dt, 4096, 8192)[0]
    assert "kittens::sv<float, 16> (&reserve)[768]" in k.source(
        CHUNKS=2, RUN=8, EXACT=1, RESERVE=48 << 10)


def test_gfx950s_cap_starts_at_its_256_mb_cache(monkeypatch):
    """53 KB reserved -- three workgroups a CU -- from the 256 MiB cache's
    size: 8192x8192 rmsnorm (exactly 256 MiB) 39.2 -> 37.8 us, 8192x4096
    20.1 -> 22.6 had it been capped."""
    monkeypatch.setattr(norm, "_strips_here", lambda: True)
    monkeypatch.setattr(norm, "_PLAN_CACHE", {})
    monkeypatch.setattr(norm, "_RESERVE_HERE", [norm._STRIP_RESERVE_AT["gfx950"]])
    monkeypatch.setattr(norm, "_wave", lambda: 64)
    dt = _BF16()
    assert norm._resolve_strip("rms", dt, 4096, 8192)[4] == 0              # 134 MB
    assert norm._resolve_strip("rms", dt, 8192, 8192)[4] == 53 << 10       # 268 MB
    assert norm._resolve_strip("addrmsw", dt, 4096, 8192)[4] == 53 << 10   # 4 streams


def test_off_rdna3_the_norms_stay_on_the_tiles(monkeypatch):
    monkeypatch.setattr(norm, "_strips_here", lambda: False)
    monkeypatch.setattr(norm, "_PLAN_CACHE", {})
    assert norm._resolve_strip("rms", _BF16(), 4096, 1) is None
