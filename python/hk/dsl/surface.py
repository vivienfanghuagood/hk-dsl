"""The surface registry: every public name of the language, every dialect op and every C++
helper in include/hk/, with its kind (docs/design/dsl-surface.md §4).

This is data, not behaviour: nothing in the compiler reads it. tests/hk/ir/test_surface.py holds
the code to it, so a name, an op or a C++ helper that is not registered here fails the IR tier:

- every name documented in docs/api for the language (`hk`, `hk.arch`, `hk.lds`, `hk.group`,
  `hk.collective`, `hk.comm`, `hk.math`, `hk.placement`) is an ENTRY or in PRIMITIVES;
- every op in HKOps.td is in OPS;
- every function defined in include/hk/*.cuh is in CPP;
- every use of an escape hatch in hk.ops, the templates and the examples is listed, with a
  reason, in scripts/escape_hatches_allow.txt.

A new kernel adds no kernel-specific helper or one-off op (CONTRIBUTING.md). A new general
primitive is added to PRIMITIVES (and OPS / CPP) in the PR that adds it; an escape hatch gets an
ENTRY with its boundary, after review.

Kinds:
  primitive   general; any kernel may use it
  format      one data layout (hk's own, or a library's); a new layout is an argument, not an op
  escape      allowed only inside its boundary; every in-tree use is in the allowlist
  experiment  never on by default; has a sunset
  internal    C++ only, never a DSL name: a library fix, a gap fill, plumbing, or a
              type-selected specialization (bitwise the generic path's)
  merge       folds into a general abstraction (`target`) in migration step `step`
  delete      goes away in migration step `step`
  deprecated  a shim kept until `step` (the release it goes away in)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, NamedTuple, Optional

KINDS = ("primitive", "format", "escape", "experiment", "internal", "merge", "delete",
         "deprecated")


class Entry(NamedTuple):
    """One registered name: its kind, the inventory item it is (dsl-surface.md §2), the boundary
    an escape hatch or experiment may be used within, and for merge / delete the target and the
    migration step."""

    kind: str
    item: str = ""
    boundary: str = ""
    target: str = ""
    step: str = ""


#: When a shim goes (dsl-surface.md §9, decision 2).
_GONE = ("the release after the last call from outside the repository is migrated, at most two "
         "minor releases after its deprecation")

_MX = "gfx950's scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4)"

#: The inventoried names (dsl-surface.md §2) that are not plain primitives, and the keyword
#: forms that are. A `(kw=)` suffix names one keyword of a documented function.
ENTRIES: Dict[str, Entry] = {
    # 2.1 MX block-scaled operands
    "hk.load_mx_scales": Entry("format", "A1", f"E8M0 scales of {_MX}'s operands; a new layout "
                               "is a layout= value, not a new op"),
    "hk.copy_mx_scales": Entry("deprecated", "A2", target="MxScales.copy(into=)",
                               step=_GONE),
    "hk.mma_ABt_mx": Entry("deprecated", "A3", target="hk.mma with Tile.scaled operands",
                           step=_GONE),
    "hk.load_fp4_frags": Entry("format", "A4", "AITER's pre-shuffled MXFP4 MoE weights only"),
    "hk.load_fp4_frags_lds": Entry("deprecated", "A5", target="hk.load_fp4_frags(SharedTile)",
                                   step=_GONE),
    "hk.mx_quant_store": Entry("merge", "A6", target="a G1 composition (hk.mx)", step="S10"),
    "hk.fp4_operand": Entry("deprecated", "A7", target="Tile.as_fragment(ty, fmt='fp4')",
                            step=_GONE),
    "hk.fp8_operand": Entry("deprecated", "A8", target="Tile.as_fragment(ty)", step=_GONE),
    "hk.mx_scales_loose": Entry("deprecated", "A9", target="Tile.as_scales()", step=_GONE),
    # 2.2 rows by index
    "Global.gather(div=)": Entry("deprecated", "B1", target="hk.RowIndex(div=)", step=_GONE),
    "Global.scatter(sort_topk=)": Entry("deprecated", "B2", target="hk.RowIndex(sort_topk=)",
                                        step=_GONE),
    "Global.scatter(staged=)": Entry("deprecated", "B3",
                                     target="an index staged by Group.stage_rows", step=_GONE),
    "Group.gather_prep": Entry("deprecated", "B4", target="Group.stage_rows", step=_GONE),
    "Group.load(staged=)": Entry("deprecated", "B5", target="Group.load(gather=<a staged "
                                 "hk.RowIndex>)", step=_GONE),
    "Global.scatter(add=)": Entry("primitive", "B6", "atomic scatter-add, in no fixed order: not "
                                  "for a route that promises determinism"),
    # 2.3 strips, elements, lanes
    "hk.map_cpp": Entry("escape", "C1", "an intrinsic or instruction hk has no op for, or "
                        "another kernel's exact instruction sequence; not index arithmetic, "
                        "selects, bit ops or float expressions the operators can write; never "
                        "in a template"),
    "hk.iota": Entry("merge", "C2", target="hk.lane_coords(type)[1] (alias kept)", step="S6"),
    "hk.mfma_strip": Entry("merge", "C3", target="hk.mma on Tile.as_fragment views", step="S9"),
    # 2.4 attention, RDNA, the kernel object
    "hk.Granules": Entry("escape", "D2", "gfx11 kernels whose LDS address registers are the "
                         "measured limiter; not on CDNA"),
    "SharedTile.granules": Entry("escape", "D2", "the same as hk.Granules'"),
    "Granules.load_frag": Entry("escape", "D2", "the same as hk.Granules'"),
    "Granules.store_frag": Entry("escape", "D2", "the same as hk.Granules'"),
    "hk.arch.agpr": Entry("experiment", "D3", "behind an option that is off by default; "
                          "deleted after 2026-10-31 12:00Z unless the task evaluating it adopts "
                          "it (dsl-surface.md §9, decision 6)"),
    "Tile.make_causal": Entry("deprecated", "D5", target="Tile.tril_(0, v) / Tile.triu_(0, v)",
                              step="0.10, or sooner once nothing outside the repository calls "
                              "it"),
    "Kernel.with_defines": Entry("escape", "D6", "#define NAME value knobs and A/B variants; "
                                 "never overridden to splice or rewrite C++"),
    "Kernel.build_flags": Entry("escape", "D7", "tests and probes; not hk.ops kernels"),
    # 2.5 kept as primitives / formats
    "hk.load_w4": Entry("format", "E8", "hk's own int4 layout (hk.ops.w4a16's repack)"),
}

#: Every other documented name of the language: general, any kernel may use it.
PRIMITIVES: FrozenSet[str] = frozenset("""
Analysis.summary Bool.to_int Bool.type Build.report Coord.type Coord.unit Digit.of Global.at
Global.batch Global.cols Global.depth Global.dtype Global.gather Global.load_elems
Global.post_elems Global.rows Global.scatter Global.store_elems Global.take_elems Global.tile
Global.type Granules.type Group.commit Group.load Group.stage Group.stage_buffer
Group.stage_rows Group.store Group.threads Group.wait_for Int.type Kernel.arch Kernel.build
Kernel.compile Kernel.placed Kernel.placement_choice Kernel.preloaded Kernel.resident_capacity
Kernel.source Kernel.specialize Kernel.trace MxScales.copy MxScales.type Placement.chunked
Placement.cluster Placement.none Placement.suffix RowIndex.offset RowIndex.staged
SharedTile.atomic_add_elems SharedTile.load SharedTile.load_elems SharedTile.store
SharedTile.store_elems SharedTile.subtile SharedTile.type SharedVec.load SharedVec.store
SharedVec.type Tile.T Tile.abs Tile.as_fragment Tile.as_operand Tile.as_scales Tile.bitcast
Tile.copy Tile.dgelu Tile.dtype Tile.exp Tile.exp2 Tile.exp2_ Tile.exp_ Tile.fill_
Tile.first_lane Tile.gelu Tile.group_max Tile.group_min Tile.group_sum Tile.lane_gather
Tile.lane_strip Tile.left_fill_ Tile.log Tile.log2 Tile.lower_fill_ Tile.max Tile.maximum
Tile.min Tile.minimum Tile.neg_infty_ Tile.prod Tile.relu Tile.right_fill_ Tile.rotate_half
Tile.rsqrt Tile.rsqrt_ Tile.run_xor Tile.scaled Tile.silu Tile.sqrt Tile.step Tile.subtile
Tile.sum Tile.swap_halves Tile.to Tile.tril_ Tile.triu_ Tile.type Tile.upper_fill_ Tile.zero_
TracedKernel.cpp TracedKernel.dump TracedKernel.tensors TracedKernel.threads
TracedKernel.written_tensors Vec.abs Vec.any_above Vec.any_ne Vec.copy Vec.dgelu Vec.exp
Vec.exp2 Vec.exp2_ Vec.exp_ Vec.fill_ Vec.gelu Vec.log Vec.log2 Vec.max_ Vec.maximum Vec.min_
Vec.minimum Vec.neg_infty_ Vec.relu Vec.rotate_half Vec.rsqrt Vec.rsqrt_ Vec.silu Vec.sqrt
Vec.step Vec.swap_halves Vec.type Vec.zero_ View.atomic_add View.atomic_add_scalar
View.atomic_and_scalar View.atomic_exch_scalar View.atomic_max_scalar View.atomic_min_scalar
View.atomic_or_scalar View.atomic_xor_scalar View.load View.load_scalar View.store
View.store_scalar hk.Bool hk.Build hk.CompileError hk.Constexpr hk.Coord hk.DType hk.Digit
hk.Global hk.Int hk.Kernel hk.MxScales hk.Placement hk.RS hk.RT hk.ResourceError hk.RowIndex
hk.ST hk.SV hk.Scaled hk.SharedTile hk.SharedVec hk.TARGETS hk.Tensor hk.Tile hk.TracedKernel
hk.Vec hk.VerifyError hk.View hk.arch.SCHED_CLASSES hk.arch.barrier hk.arch.block_idx
hk.arch.lane_id hk.arch.mma_fence hk.arch.pin hk.arch.pingpong hk.arch.sched_barrier
hk.arch.sched_group_barrier hk.arch.setprio hk.arch.target hk.arch.vm_wait hk.arch.warp_id
hk.arch.wg_arrive hk.arch.xcd_count hk.arch.xcd_id hk.as_vec hk.attention hk.broadcast_col
hk.broadcast_row hk.cdiv hk.col_vec hk.collective.COMBINERS hk.collective.cross_warp
hk.collective.fold_rows hk.comm.REALTIME_HZ hk.comm.arrive hk.comm.barrier hk.comm.claim
hk.comm.post hk.comm.signal hk.comm.take hk.comm.wait hk.comm.xcd_barrier hk.coord
hk.detect_arch hk.elem_coord hk.full hk.func hk.get_target hk.group.Group hk.group.group hk.if_
hk.kernel hk.lane_coords hk.lds.alloc hk.lds.alloc_vec hk.lds.lds_loads hk.lds.reserve
hk.lds.shared_vec hk.lds.wait_for hk.math.abs hk.math.add hk.math.dgelu hk.math.div hk.math.exp
hk.math.exp2 hk.math.gelu hk.math.log hk.math.log2 hk.math.maximum hk.math.minimum hk.math.mul
hk.math.relu hk.math.rsqrt hk.math.silu hk.math.sqrt hk.math.sub hk.max hk.min hk.mma hk.mx
hk.neg_infty hk.placement.Analysis hk.placement.Choice hk.placement.Digit
hk.placement.MIN_ROUNDS_NONUNIFORM hk.placement.MIN_STREAMS hk.placement.Placement
hk.placement.analyze hk.placement.check_emitted hk.placement.check_kernel hk.placement.choose
hk.placement.emitted_decode hk.placement.place_host hk.placement.recording
hk.placement.renumber hk.placement.xcd_count hk.placement.xcd_range hk.rand hk.range
hk.range_constexpr hk.row_vec hk.runtime.SymmetricBuffer hk.scope hk.templates hk.tile_coord
hk.torch_op hk.transpose hk.transpose_sep hk.uniform hk.zeros
""".split())

#: The dialect's ops (HKOps.td mnemonics) and their kinds; the inventory's are not primitive.
_OP_KINDS: Dict[str, str] = {
    "load_mx_scales": "format", "mx_scales_copy": "primitive", "mma_ABt_mx": "primitive",
    "load_fp4_frags": "format", "load_fp4_frags_lds": "format", "mx_quant_store": "merge",
    "fp4_operand": "primitive", "fp8_operand": "primitive", "mx_scales_loose": "primitive",
    "gather_prep": "primitive", "map_cpp": "escape", "iota": "merge",
    "mfma_strip": "merge", "granules": "escape", "load_frag": "escape",
    "store_frag": "escape",
    "load_w4": "format",
}
OPS: Dict[str, str] = {**{op: "primitive" for op in """
block_idx warp_id xcd_id lane_id coord extent any_ne any_above load store load_rows store_rows
store_scalar load_scalar wg_arrive gather_rows scatter_rows atomic_add atomic_scalar signal
arrive wait post take philox_uniform alloc_shared shared_at subtile load_shared store_shared
lds_wait_for alloc_shared_vec store_shared_vec load_shared_vec stage_buffer stage_load
stage_commit vm_wait group_load group_wait_for pingpong_begin pingpong_end group_store barrier
sched_barrier sched_group_barrier pin setprio zero full neg_infty add sub mul div max min exp
exp2 log log2 abs relu copy gelu dgelu sqrt rsqrt silu step first_lane lane_gather post_elems
take_elems lds_load_elems lds_store_elems lds_atomic_add_elems load_elems store_elems
group_reduce swap_halves row_max row_min row_sum row_prod col_max col_min col_sum col_prod
add_row sub_row mul_row div_row add_col sub_col mul_col div_col broadcast_row broadcast_col
retype_vec left_fill right_fill upper_fill lower_fill triu tril cast bitcast transpose
transpose_sep rt_subtile as_operand mma_AB mma_ABt mma_AtB mma_AtBt lane_strip lane_coord run_xor
""".split()}, **_OP_KINDS}

#: Every function defined in include/hk/*.cuh (as test_surface extracts the names: the token
#: before the parameter list of a __device__ definition), by kind. `primitive`: the C++ of a
#: general DSL op; `internal`: plumbing, library fixes, gap fills and type-selected
#: specializations (dsl-surface.md §2.5); the rest as their DSL entries.
CPP: Dict[str, Dict[str, str]] = {
    "include/hk/cdna.cuh": {
        **{n: "primitive" for n in """
arrive as_operand atomic_add atomic_scalar atomic_scalar_scoped bin_map claim col_map copy
lane_coord lane_strip run_xor dgelu gelu group_load_async group_load_async_gather
group_load_async_rows group_reduce_ordered iota lane_gather lds_atomic_add_elems lds_load_elems
lds_load_elems_runs lds_store_elems lds_wait lds_wait_for left_fill load load_bounded
load_elems load_elems_runs load_gather load_period load_scalar load_scalar_invariant
load_scoped lower_fill neg_infty one philox_uniform post post_elems post_elems_runs right_fill
row_map row_sum_times_ordered rsqrt signal silu sqrt step store store_bounded store_elems
store_elems_runs store_scalar store_scatter store_scoped swap_halves take take_elems
take_elems_runs tril triu unary_map upper_fill vm_wait wait wg_arrive wg_arrive_xcd xcd_id zero
""".split()},
        **{n: "internal" for n in """
__attribute__ _col _ordered _row batch col_reduce cols convert depth fast_tanh group_load_sync
group_store_bounded hk_any_live hk_as_elem hk_atomic_op hk_bind_regs hk_bits hk_buf_ld128
hk_buf_ld16 hk_buf_ld32 hk_buf_ld64 hk_buf_ld8 hk_buf_ld_lds hk_buf_load_bytes hk_buf_load_elem
hk_buf_st128 hk_buf_st16 hk_buf_st32 hk_buf_st64 hk_buf_st8 hk_buf_store_bytes
hk_buf_store_elem hk_col_rows_fill hk_copy_srsrc hk_cvt hk_cvt_pk_bf16 hk_dpp hk_elem_off
hk_elems_srsrc hk_elems_srsrc_row hk_frag_coord hk_from_bits hk_gather_index hk_gather_rows_lds
hk_get hk_lane_arg hk_lane_opaque hk_ld_aux hk_load_tile hk_mailbox_slot hk_mfma_mx
hk_mfma_mx_b hk_mx_operand hk_ninf_where_key_above hk_opaque hk_philox_first hk_pinned
hk_rv_load hk_rv_store hk_scatter_ids_lds hk_scatter_index hk_scoped_ld32 hk_scoped_st32 hk_set
hk_srsrc hk_st_aux hk_st_elem hk_static_for hk_store_tile hk_swap_op hk_tile_srsrc hk_lane_each
hk_strip_at hk_uniform_from hk_w4_word hk_whole_srsrc hk_xor_op idx lds_bind mma_ABt name num
op row_stride rows shape strided_gl strip_axis_fill strip_butterfly strip_col_reduce strip_each
strip_read strip_row_dot_ordered strip_row_reduce strip_row_reduce_ordered strip_scalar
strip_srsrc strip_tree strip_write stride warpid_uniform word
""".split()},
        "load_mx_scales": "format", "load_mx_scales_aiter": "format", "load_fp4_frags": "format",
        "load_w4": "format", "mma_ABt_mx": "primitive", "load_fp4_frags_lds": "format",
        "mx_quant_store": "merge", "fp4_operand": "primitive",
        "fp8_operand": "primitive", "mx_scales_loose": "primitive", "gather_prep": "primitive",
        "map_cpp": "escape", "mfma_strip_f32_16x16x4": "merge",
    },
    "include/hk/cdna_agpr.cuh": {n: "experiment" for n in """
col_scale div_col drain mfma mma_AB mma_ABt mma_AtB mma_AtBt mul_col read write zero
""".split()},
}


def kind_of(name: str) -> Optional[str]:
    """The registered kind of a documented name (or a `Name(kw=)` keyword form), else None."""
    if name in ENTRIES:
        return ENTRIES[name].kind
    if name in PRIMITIVES:
        return "primitive"
    return None


__all__ = ["KINDS", "Entry", "ENTRIES", "PRIMITIVES", "OPS", "CPP", "kind_of"]
