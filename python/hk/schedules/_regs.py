"""What a schedule's build will report, predicted before it is compiled.

LDS is exact: it is the allocations' sizes. Registers are a model. Each
family's live tiles -- accumulators, operand fragments, staging, the
partial sums a combine or a split's reduction holds -- are counted the way
the target lays them out (`features`), and a linear model per (family,
arch) turns them into the registers the build reports: arch VGPRs on
gfx1100, VGPRs plus AGPRs on gfx950 (whose compiler moves what does not fit
in the 256 VGPRs into AGPRs before it spills). The coefficients are an upper
bound on every sampled build that fit, over the limit on every one that
spilled (`MODEL`; tools/hk-bench/schedules_legality.py samples, scores and
refits it). A prediction over the wave's share of the register file is
refused as a spill.
"""

from __future__ import annotations

from typing import Any, Dict

#: Registers a wave may hold before the build spills: arch VGPRs on
#: gfx1100 and gfx1201; VGPRs + AGPRs on gfx950.
LIMIT = {"gfx1100": 256, "gfx1201": 256, "gfx950": 512}

#: VGPRs added to every prediction for what the fit has not seen: another
#: hipcc (CI's is 7.1, the fit's 7.2.3), knobs combined in ways the samples
#: did not.
MARGIN = 0.0

#: (family@arch) -> {"coef": {feature: weight, "1": intercept}, "offset"}:
#: the tightest linear upper envelope of the sampled builds' registers
#: (min sum(pred - actual) subject to pred >= actual, every spill over the
#: limit, every shipped schedule under it; linprog), fitted on ~19,000 builds
#: under hipcc 7.2.3 and 7.2.4 by tools/hk-bench/schedules_legality.py fit.
MODEL: Dict[str, Dict[str, Any]] = {
    "decode@gfx1100": {"offset": MARGIN, "coef": {
        "acc": 2.9251, "batched": 0.4687, "combine": 0.7057, "one": 22.5826,
        "reduce": 0.4648, "reduce1": 0.0469, "split": -31.7445, "w": 2.3901, "x": -0.3901,
        "1": 50.2621}},
    # gfx1201: 1,249 builds under hipcc 7.2.3 (987 fit, 262 spilled). Infeasible as a hard
    # envelope -- the shipped 4-token schedule builds at 255 of 256 beside points with its
    # features that built at 256 and spilled (`resident` is not a feature) -- so three
    # spills are left to the build's gate (schedules_legality.py fit, the soft stage).
    "decode@gfx1201": {"offset": MARGIN, "coef": {
        "acc": 4.042, "batched": 0.0391, "combine": 0.2724, "one": 3.75, "reduce": 0.026,
        "reduce1": 0.0, "split": -0.6667, "w": 4.5238, "x": -1.131, "1": 117.8992}},
    "decode@gfx950": {"offset": MARGIN, "coef": {
        "acc": 3.5988, "batched": 0.2655, "combine": 0.3571, "one": 3.1429,
        "reduce": 0.4377, "reduce1": -0.2143, "split": -10.0075, "w": -0.1667, "x": 0.1667,
        "1": 34.6817}},
    "dp@gfx950": {"offset": MARGIN, "coef": {
        "acc": 1.0946, "ops": 1.9477, "piped": -63.6217, "split": -7.5816, "1": 60.4988}},
    "pp@gfx950": {"offset": MARGIN, "coef": {
        "a": 0.8438, "acc": 1.3281, "asym": -0.1562, "b": 1.1563, "early": -0.0,
        "late_b": 0.0, "lds16": 3.5, "nt": -0.0, "setprio": -4.0, "split": 0.0, "tc": 0.0,
        "1": 22.0}},
    # gfx1201: fitted on 1,941 builds under hipcc 7.2.3 (1,545 fit, 396 spilled); a looser
    # envelope than gfx1100's -- it refuses 603 of the points that built -- so a first
    # model, to be refitted with the gfx1201 GEMM's own tuning.
    "rdna@gfx1201": {"offset": MARGIN, "coef": {
        "a": 0.5686, "acc": 0.8802, "asym": 0.3156, "b": 0.6212, "lds2": -1.5, "sched": 0.5741,
        "setprio": -9.0556, "stagger": 118.9246, "staging": 1.2083, "stream_a": -7.2778,
        "1": 85.7407}},
    "rdna@gfx1100": {"offset": MARGIN, "coef": {
        "a": 0.6972, "acc": 0.9218, "asym": 0.3945, "b": 0.7096, "lds2": 0.6036,
        "sched": 21.0995, "setprio": -12.4063, "stagger": 24.0, "staging": 3.1168,
        "stream_a": -2.1024, "1": 37.5921}},
}


def features(s, arch: str) -> Dict[str, float]:
    """The schedule's live tiles in VGPRs a lane, by kind, and indicator
    features for knobs that change the code around them."""
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    wave = tgt.wave
    f = s.family
    if f == "rdna":
        cols = s.warps // s.warp_rows
        reg_m, reg_n = s.block_m // s.warp_rows, s.block_n // cols
        frags = 2 if s.stream_a else reg_m // 16
        # VGPRs a 16x16 bf16 operand fragment: 8 on gfx1100 (mirrored), 4 on gfx1201
        op = tgt.mma["bf16"].operand_vgprs
        a = frags * (s.dot_slice // 16) * op
        b = (reg_n // 16) * (s.dot_slice // 16) * op
        return dict(
            acc=reg_m * reg_n * 4 / (4 * wave), a=a, b=b, asym=abs(a - b),
            staging=(s.block_m + s.block_n) * s.k_step * 2 / (wave * s.warps * 4),
            sched=float(s.sched > 0), stream_a=float(s.stream_a),
            lds2=float(s.lds_buffers == 2), stagger=float(s.stagger > 0),
            setprio=float(s.setprio))
    if f == "pp":
        reg_m = s.block_m // 2 // s.warps_m
        reg_n = s.block_n // 2 // s.warps_n
        a = reg_m * s.k_step * 2 / (4 * wave)
        b = 2 * reg_n * s.k_step * 2 / (4 * wave)
        return dict(acc=4 * reg_m * reg_n * 4 / (4 * wave), a=a, b=b,
                    asym=abs(a - b), split=float(s.splits > 1), tc=float(s.transposed_c),
                    late_b=float(s.late_b), setprio=float(s.setprio),
                    early=float(s.early_store), lds16=float(s.lds_shape == "16x64"),
                    nt=float(bool(s.nt)), sk=float(getattr(s, "stream_k", False)))
    if f == "dp":
        reg_m, reg_n = s.block_m // s.warps_m, s.block_n // s.warps_n
        sk = 64 if s.lds_shape == "16x64" else 32
        depth = s.k_step if s.pipelined else sk
        return dict(acc=reg_m * reg_n * 4 / (4 * wave),
                    ops=2 * (reg_m + reg_n) * depth * 2 / (4 * wave),
                    split=float(s.splits > 1), piped=float(s.pipelined))
    if f == "decode":
        kc = 16 * s.slices
        part = 16 * 16 * s.tokens * 4 / (4 * wave)       # a 16 x 16t fp32 tile
        # The combine's partials: all in flight (combine_depth 0) the
        # compiler streams them; a batch fenced by sched_barriers it holds.
        comb = (s.warps - 1) * part if s.warps > 1 else 0.0
        return dict(acc=part, w=16 * kc * 2 / (4 * wave),
                    x=16 * s.tokens * kc * 2 / (4 * wave),
                    combine=comb if not s.combine_depth else 0.0,
                    batched=comb if s.combine_depth else 0.0,
                    reduce=s.splits * part if s.splits > 1 else 0.0,
                    reduce1=s.splits * part if s.splits > 1 and s.warps == 1 else 0.0,
                    split=float(s.splits > 1), one=float(s.warps == 1))
    return {}


def waves(s) -> int:
    """Waves a workgroup of the schedule has."""
    return {"rdna": lambda: s.warps, "decode": lambda: s.warps, "pp": lambda: 8,
            "dp": lambda: s.warps_m * s.warps_n}.get(s.family, lambda: 1)()


def limit(s, arch: str) -> int:
    """Registers a wave of the schedule may hold: a workgroup's waves share
    a CU's SIMDs, so 16 waves on an MI355X's four leave each 512 / 4 = 128;
    and at most 256 VGPRs on gfx1100, 512 VGPRs + AGPRs on gfx950."""
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    per_simd = -(-waves(s) // 4)
    return min(LIMIT.get(arch, 256), tgt.vgprs_per_simd // per_simd)


def _place(sizes) -> int:
    """hk-lds-alloc's layout: allocations in declaration order, each aligned
    to its own size when that is a power of two (else 16 B)."""
    off = 0
    for size in sizes:
        align = size if size and not size & (size - 1) else 16
        off = -(-off // align) * align + size
    return off


def lds(s, arch: str) -> int:
    """LDS bytes a workgroup of the schedule allocates, as hk-lds-alloc lays
    them out (and wg_arrive's 16 static bytes)."""
    f = s.family
    if f == "rdna":
        return _place([s.lds_buffers * s.block_m * s.k_step * 2,
                       s.lds_buffers * s.block_n * s.k_step * 2])
    if f == "pp":
        arrive = 16 if getattr(s, "stream_k", False) else 0     # wg_arrive's static LDS
        return _place([4 * (s.block_m // 2) * s.k_step * 2,
                       4 * (s.block_n // 2) * s.k_step * 2]) + arrive
    if f == "dp":
        return _place([s.stages * s.block_m * s.k_step * 2,
                       s.stages * s.block_n * s.k_step * 2])
    if f == "decode":
        arrive = 16 if s.splits > 1 else 0          # wg_arrive's static LDS
        if s.warps <= 1:
            return arrive
        from ..target import get_target  # noqa: PLC0415

        tgt = get_target(arch)
        part = 16 * s.tokens * 16 * 4
        slots = s.warps
        if s.resident:
            budget = tgt.lds_pool_bytes // s.resident - arrive
            slots = max(s.warps, budget // part)
        return _place([slots * part]) + arrive
    return 0


def tile_estimate(s, arch: str) -> int:
    """hk-check-budgets' estimate of the register tiles live at once, which
    refuses a kernel over a SIMD's VGPRs while tracing: the dp kernel
    declares both register sets (the unpipelined A/B pair stays live, never
    used), so its operands count three times."""
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    if s.family != "dp":
        return 0
    reg_m, reg_n = s.block_m // s.warps_m, s.block_n // s.warps_n
    sk = 64 if s.lds_shape == "16x64" else 32
    ta, tb = tgt.tile_vgprs(16, reg_m, sk), tgt.tile_vgprs(16, reg_n, sk)
    est = 2 * ta + 2 * tb + tgt.tile_vgprs(32, reg_m, reg_n)
    if s.pipelined:
        est += 2 * (s.k_step // sk) * (ta + tb)
    return est


def predict(s, arch: str) -> Dict[str, Any]:
    """{"vgpr": predicted registers (an upper bound; VGPRs + AGPRs on
    gfx950), "limit": the register file a wave may use, "lds": bytes,
    "occupancy": waves a SIMD at that many registers, "tiles": the
    features, "hint": what to shrink}."""
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    feats = features(s, arch)
    model = MODEL.get(f"{s.family}@{arch}")
    vgpr = occ = None
    if model is not None and feats:
        c = model["coef"]
        v = c.get("1", 0.0) + sum(w * feats.get(n, 0.0) for n, w in c.items() if n != "1")
        vgpr = int(-(-(v + model["offset"]) // 1))
        occ = tgt.occupancy(max(1, min(vgpr, tgt.vgprs_per_simd)))
    hint = {"rdna": "a smaller block, n_split or k_step, or stream_a",
            "pp": "a smaller block or k_step",
            "dp": "a smaller block or k_step, or pipelined=False",
            "decode": "fewer tokens, slices or splits, or a combine_depth"}.get(s.family, "")
    return dict(vgpr=vgpr, limit=limit(s, arch), lds=lds(s, arch), occupancy=occ,
                tiles=feats, hint=hint, estimate=tile_estimate(s, arch))
