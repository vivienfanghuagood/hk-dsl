"""gfx950 -- CDNA4, Instinct MI355X / MI350X.

The register numbers were read off LLVM's own Occupancy remark by a clobber
sweep (tools/cdna-probes/granule_sweep.py) and then timed on an MI355X
(tools/cdna-probes/occupancy_timed.py): kernels pinning N VGPRs (and M AGPRs)
run a latency-bound chain of dependent loads, and the time to push 65536
waves through steps at exactly the model's boundaries -- 64/65, 72/73, 80/81,
96/97, 128/129 and 168/169 VGPRs, and 64 VGPRs + 1 AGPR -- by the model's
ratios. So `verified=True`, as gfx1100's timed 24-register granule is. The
LDS figure is HipKittens' include/cdna4 MAX_SHARED_MEMORY.

What is different from gfx1100, and why each one matters to hk:

* wave64. A warp is 64 threads; every `warps * 32` in the frontend and the
  translation is a kernel that launches half its lanes and computes half its
  answer.
* One 512-entry register file per lane, shared by the VGPRs and the AGPRs
  (the MFMA accumulation registers): occupancy = 512 / round8(align4(VGPRs) +
  AGPRs), capped at 8 waves/SIMD. Measured: 64 VGPRs is 8 waves, 65 is 7; 64
  VGPRs plus one AGPR is also 7.
* The MFMA operand fragment is 16x32, not 16x16: v_mfma_f32_16x16x32_bf16
  reads 8 bf16 per lane (4 VGPRs), and include/cdna4 spells that register tile
  `rt<bf16, R, C, row, rt_16x32>`. The accumulator is the 16x16 fragment, 4
  VGPRs, and lives in the col layout -- which hk's GEMMs already use.
* Global -> LDS without VGPRs exists (buffer_load ... lds, up to 128 bits a
  lane on gfx950): the async-load pillar HipKittens is built around applies
  here, where on gfx11 it does not.
* LDS reads of a col-layout 16-bit tile vectorise through ds_read_b64_tr_b16,
  so the slow-layout warning that is right on gfx1100 is wrong here.
"""

from .base import MmaShape, Target

GFX950 = Target(
    arch="gfx950",
    gpu_target="CDNA4",
    kittens_define="KITTENS_CDNA4",

    wave=64,
    max_waves_per_simd=8,

    # Unified VGPR+AGPR file, per lane. The sweep: occupancy steps at 65, 73,
    # 81, 97, 129, 169 VGPRs -- 512 // round8(n) exactly.
    vgprs_per_simd=512,
    vgpr_granule=8,
    # LLVM's addressable SGPRs for gfx9 (gfx10+ is 106).
    sgprs_per_wave=102,

    # include/cdna4/common/util.cuh MAX_SHARED_MEMORY. The CU has 160 KiB;
    # HipKittens caps a request at 160000 B and so does hk.
    lds_per_workgroup=160000,
    # A CU is 4 SIMD64 and owns its LDS (no WGP pairing as on RDNA).
    lds_pool_bytes=163840,
    simds_per_lds_pool=4,

    mma={
        # v_mfma_f32_16x16x32_{bf16,f16}: A and B are 16x32, 8 elements a lane,
        # unmirrored; C/D is 16x16 fp32, 4 a lane. include/cdna4 mma.cuh.
        "bf16": MmaShape(m=16, n=16, k=32, operand_vgprs=4, accum_vgprs=4),
        "fp16": MmaShape(m=16, n=16, k=32, operand_vgprs=4, accum_vgprs=4),
    },

    has_global_to_lds_dma=True,
    mma_operands_mirrored=False,
    fast_smem_layouts=("row", "col"),

    verified=True,
    unified_agprs=True,
    lgkmcnt_max=15,
    tile_shapes=True,
    extra_includes=('"hk/cdna.cuh"',),
    strips=True,
    kernarg_preload=True,
)

#: The two numbers a result is reported against, for an MI355X: AMD's
#: datasheet dense bf16 MFMA rate and HBM3E bandwidth. NOT measured here --
#: replace with a measured ceiling before quoting a percentage.
CEILINGS = {
    "mfma_bf16_tflops": 2500.0,
    "hbm_gbps": 8000.0,
}
