"""What a schedule's build will report, predicted before it is compiled.

LDS is exact: it is the allocations' sizes. Registers are a model: the live
tiles' VGPRs (accumulators, operand fragments, staging), counted the way the
target lays them out, plus a per-family overhead for addresses, loop state
and the compiler's temporaries, fitted to the builds the legality test
samples (tests/hk/codegen/test_schedules_legality.py) and kept above every
one of them. A prediction over the limit is refused as a spill.
"""

from __future__ import annotations

from typing import Any, Dict

#: Per (family, arch): (scale on the counted tile VGPRs, overhead).
#: Fitted so that the prediction is at or above every sampled build.
MODEL: Dict[tuple, tuple] = {
    ("rdna", "gfx1100"): (1.0, 72),
    ("decode", "gfx1100"): (1.0, 48),
    ("decode", "gfx950"): (1.0, 40),
    ("pp", "gfx950"): (1.0, 48),
    ("dp", "gfx950"): (1.0, 48),
}


def _tiles_rdna(s) -> Dict[str, int]:
    cols = s.warps // s.warp_rows
    reg_m, reg_n = s.block_m // s.warp_rows, s.block_n // cols
    acc = (reg_m // 16) * (reg_n // 16) * 8
    frags = 2 if s.stream_a else reg_m // 16
    a = frags * (s.dot_slice // 16) * 8
    b = (reg_n // 16) * (s.dot_slice // 16) * 8
    staging = (s.block_m + s.block_n) * s.k_step * 2 // (32 * s.warps * 4)
    return dict(acc=acc, a=a, b=b, staging=staging)


def _tiles_pp(s) -> Dict[str, int]:
    reg_m = s.block_m // 2 // s.warps_m
    reg_n = s.block_n // 2 // s.warps_n
    acc = 4 * reg_m * reg_n // 64
    a = reg_m * s.k_step // 128
    b = 2 * reg_n * s.k_step // 128
    return dict(acc=acc, a=a, b=b)


def _tiles_dp(s) -> Dict[str, int]:
    reg_m, reg_n = s.block_m // s.warps_m, s.block_n // s.warps_n
    sk = 64 if s.lds_shape == "16x64" else 32
    acc = reg_m * reg_n // 64
    if s.pipelined:
        ops = 2 * (reg_m + reg_n) * s.k_step // 128
    else:
        ops = 2 * (reg_m + reg_n) * sk // 128
    return dict(acc=acc, ops=ops)


def _tiles_decode(s, wave: int) -> Dict[str, int]:
    # 16 x (16 tokens) fp32 accumulator (C^T), W's 16 x kc and X's
    # (16 tokens) x kc operands, and the partials the combine holds.
    kc = 16 * s.slices
    per = 4 * 64 // wave          # VGPRs of a 16x16 fp32 tile on this wave
    acc = s.tokens * per
    w = 16 * kc * 2 // (wave * 4)
    x = 16 * s.tokens * kc * 2 // (wave * 4)
    depth = s.combine_depth or s.warps
    comb = min(depth, max(s.warps - 1, 0)) * s.tokens * per if s.warps > 1 else 0
    return dict(acc=acc, w=w, x=x, combine=comb)


def lds(s, arch: str) -> int:
    f = s.family
    if f == "rdna":
        return s.lds_buffers * (s.block_m + s.block_n) * s.k_step * 2
    if f == "pp":
        return 4 * (s.block_m // 2 + s.block_n // 2) * s.k_step * 2
    if f == "dp":
        return s.stages * (s.block_m + s.block_n) * s.k_step * 2
    if f == "decode":
        if s.warps <= 1:
            return 16 if s.splits > 1 else 0
        from ..target import get_target  # noqa: PLC0415

        tgt = get_target(arch)
        part = 16 * s.tokens * 16 * 4
        slots = s.warps
        if s.resident:
            budget = tgt.lds_pool_bytes // s.resident - (16 if s.splits > 1 else 0)
            slots = max(s.warps, budget // part)
        return slots * part + (16 if s.splits > 1 else 0)
    return 0


def tiles(s, arch: str) -> Dict[str, int]:
    from ..target import get_target  # noqa: PLC0415

    f = s.family
    if f == "rdna":
        return _tiles_rdna(s)
    if f == "pp":
        return _tiles_pp(s)
    if f == "dp":
        return _tiles_dp(s)
    if f == "decode":
        return _tiles_decode(s, get_target(arch).wave)
    return {}


def predict(s, arch: str) -> Dict[str, Any]:
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    t = tiles(s, arch)
    model = MODEL.get((s.family, arch))
    vgpr = None
    occ = None
    if model is not None and t:
        scale, over = model
        vgpr = int(round(scale * sum(t.values()) + over))
        occ = tgt.occupancy(min(vgpr, 512))
    hint = {"rdna": "a smaller block, n_split or k_step, or stream_a",
            "pp": "a smaller block or k_step",
            "dp": "a smaller block or k_step, or pipelined=False",
            "decode": "fewer tokens or slices, or a combine_depth"}.get(s.family, "")
    return dict(vgpr=vgpr, lds=lds(s, arch), occupancy=occ, tiles=t, hint=hint)
