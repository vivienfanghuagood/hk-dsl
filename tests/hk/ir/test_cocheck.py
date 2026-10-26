"""hk.runtime.cocheck's reading of a code object, on text: the metadata
note, and the wait scan over an instruction stream. No toolchain."""

from __future__ import annotations

from hk.runtime.cocheck import Inst, parse_notes, wait_hazards

NOTES = """
AMDGPU Metadata
amdhsa.kernels:
  - .agpr_count:     64
    .args:
      - .name:           g
        .offset:         0
        .size:           144
    .name:           _Z1kv
    .private_segment_fixed_size: 72
    .sgpr_count:     61
    .sgpr_spill_count: 0
    .symbol:         _Z1kv.kd
    .vgpr_count:     164
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
"""


def _code(*lines):
    return [Inst(8 * i, t) for i, t in enumerate(lines)]


def test_the_note_gives_the_registers_scratch_and_occupancy():
    (k,) = parse_notes(NOTES).values()
    assert (k.name, k.arch, k.vgpr, k.agpr, k.sgpr, k.scratch) == ("_Z1kv", "gfx950", 164, 64, 61, 72)
    assert k.occupancy == 3          # 512 // round8(164), the AGPRs already in
    assert not k.ok                  # scratch


MFMA = "v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]"


def test_an_mfma_on_registers_an_lds_read_has_not_filled_is_a_hazard():
    assert wait_hazards(_code("ds_read_b128 v[0:3], v8", MFMA))
    assert not wait_hazards(_code("ds_read_b128 v[0:3], v8", "s_waitcnt lgkmcnt(0)", MFMA))


def test_a_counted_wait_retires_the_older_reads_only():
    two = ("ds_read_b128 v[0:3], v8", "ds_read_b128 v[4:7], v8 offset:512", "s_waitcnt lgkmcnt(1)")
    assert wait_hazards(_code(*two, MFMA))                         # v[4:7] still in flight
    assert not wait_hazards(_code(*two, "v_mov_b32 v9, v0"))       # v0 has landed


def test_a_load_at_a_loops_bottom_is_in_flight_at_its_top():
    # 8 bytes an instruction here: the branch at 16 goes to 24 - 4*6 = 0.
    loop = _code(MFMA, "ds_read_b128 v[0:3], v8", "s_cbranch_scc1 -6", "s_endpgm")
    assert wait_hazards(loop)


def test_a_load_does_not_survive_the_end_of_the_program():
    assert not wait_hazards(_code("v_mov_b32 v0, 0", MFMA, "ds_read_b128 v[0:3], v8", "s_endpgm"))


def test_an_address_read_by_the_next_instruction_is_a_warning():
    warn = []
    code = _code("ds_read_b64_tr_b16 v[6:7], v1 offset:1024", "ds_read_b64_tr_b16 v[8:9], v6 offset:1024")
    assert not wait_hazards(code, benign=warn)
    assert warn


def test_gfx9_stores_count_on_vmcnt():
    code = _code("buffer_load_dwordx2 v[16:17], v20, s[4:7], 0 offen",
                 "buffer_store_dword v1, v2, s[4:7], 0 offen",
                 "s_waitcnt vmcnt(1)", "v_cvt_f32_f16_e32 v18, v16")
    assert not wait_hazards(code, arch="gfx950")    # the store is the one left
    assert wait_hazards(code, arch="gfx1100")       # stores have their own counter there


def test_copies_into_lds_count_on_vmcnt():
    """Attention's prologue: Q into registers, then K's and V's blocks into
    LDS. vmcnt(2) leaves the two copies in flight and Q landed."""
    q = "buffer_load_dwordx4 v[30:33], v34, s[40:43], 0 offen"
    dma = "buffer_load_dwordx4 v216, s[20:23], 0 offen lds"
    read = "v_and_b32_e32 v59, 0xffff0000, v31"
    assert not wait_hazards(_code(q, dma, dma, "s_waitcnt vmcnt(2)", read), arch="gfx950")
    assert wait_hazards(_code(q, dma, dma, "s_waitcnt vmcnt(3)", read), arch="gfx950")


WMMA12 = "v_wmma_f32_16x16x16_bf16 v[16:23], v[0:3], v[4:7], v[16:23]"


def test_gfx12_split_waits_are_read_as_counts():
    """gfx12 spells each counter as its own instruction with a hex immediate
    (s_wait_dscnt 0x1), not lgkmcnt(1) -- read as a leading digit, 0x1 was a
    full drain, and the hazard behind it was invisible."""
    two = ("ds_load_b128 v[0:3], v8", "ds_load_b128 v[4:7], v8 offset:512")
    assert wait_hazards(_code(*two, "s_wait_dscnt 0x1", WMMA12), arch="gfx1201")
    assert not wait_hazards(_code(*two, "s_wait_dscnt 0x0", WMMA12), arch="gfx1201")
    assert not wait_hazards(_code(*two, "s_wait_dscnt 0x1", "v_mov_b32 v9, v0"), arch="gfx1201")
    loads = ("global_load_b128 v[0:3], v8, s[0:1]", "global_load_b128 v[4:7], v8, s[0:1] offset:16")
    assert wait_hazards(_code(*loads, "s_wait_loadcnt 0x1", WMMA12), arch="gfx1201")
    assert not wait_hazards(_code(*loads, "s_wait_loadcnt 0x0", WMMA12), arch="gfx1201")


def test_gfx12_combined_wait_packs_loadcnt_over_dscnt():
    """s_wait_loadcnt_dscnt: loadcnt in bits 13:8, dscnt in bits 5:0."""
    code = ("global_load_b128 v[24:27], v8, s[0:1]", "ds_load_b128 v[0:3], v8",
            "ds_load_b128 v[4:7], v8 offset:512")
    # one LDS read left in flight (v[4:7]), no global load: the WMMA reads v[0:7]
    assert wait_hazards(_code(*code, "s_wait_loadcnt_dscnt 0x1", WMMA12), arch="gfx1201")
    assert not wait_hazards(_code(*code, "s_wait_loadcnt_dscnt 0x100", WMMA12), arch="gfx1201")
    # the global load left in flight: reading it is the hazard, the LDS reads are retired
    assert wait_hazards(_code(*code, "s_wait_loadcnt_dscnt 0x100", "v_mov_b32 v9, v24"),
                        arch="gfx1201")
    assert not wait_hazards(_code(*code, "s_wait_loadcnt_dscnt 0x0", "v_mov_b32 v9, v24"),
                            arch="gfx1201")
