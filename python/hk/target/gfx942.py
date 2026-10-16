"""gfx942 -- CDNA3, Instinct MI300X / MI325X.

COMPILE-ONLY: no gfx942 part here. The register model is the same LLVM clobber
sweep as gfx950 (identical to it: 512 unified, granule 8, 8 waves/SIMD); the
LDS figure is include/cdna3's MAX_SHARED_MEMORY. `verified=False`.

Closer to hk's existing model than gfx950 is: the bf16 MFMA is 16x16x16
(v_mfma_f32_16x16x16_bf16), so a register tile is the 16x16 fragment hk
already prices, and include/cdna3 spells it `rt<T, R, C, layout>` with no shape
parameter -- exactly what the translation writes for gfx1100. What changes is
wave64, the unified register file, and an LDS read of a col-layout 16-bit tile
that is scalar here (no ds_read_b64_tr_b16 before gfx950).
"""

from .base import MmaShape, Target

GFX942 = Target(
    arch="gfx942",
    gpu_target="CDNA3",
    kittens_define="KITTENS_CDNA3",

    wave=64,
    max_waves_per_simd=8,

    vgprs_per_simd=512,
    vgpr_granule=8,
    sgprs_per_wave=102,

    lds_per_workgroup=65536,
    lds_pool_bytes=65536,
    simds_per_lds_pool=4,

    mma={
        # v_mfma_f32_16x16x16_bf16 (the _1k builtin): 4 bf16 a lane per
        # operand, 4 fp32 a lane in the accumulator.
        "bf16": MmaShape(m=16, n=16, k=16, operand_vgprs=2, accum_vgprs=4),
        "fp16": MmaShape(m=16, n=16, k=16, operand_vgprs=2, accum_vgprs=4),
    },

    # buffer_load_dword ... lds: 32 bits a lane, not gfx950's 128.
    has_global_to_lds_dma=True,
    mma_operands_mirrored=False,
    fast_smem_layouts=("row",),

    verified=False,
    unified_agprs=True,
    lgkmcnt_max=15,
    extra_includes=('"hk/cdna.cuh"',),
)
