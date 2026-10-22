"""hk.profile without a GPU: the ISA analysis, the counters' arithmetic, the
grid model, the matrix-idle attribution and the verdict logic -- on
synthetic code and on measurements recorded on an MI355X
(tests/hk/profile_fixtures, written by `bench/profile/run.py`'s `record=`).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[1] / "profile_fixtures"


# ------------------------------------------------------------------ the ISA


def test_classify_covers_both_arches():
    from hk.profile.isa import classify

    cases = {
        "v_mfma_f32_16x16x32_bf16": "mfma", "v_wmma_f32_16x16x16_bf16": "wmma",
        "v_add_f32_e32": "valu", "v_exp_f32_e32": "valu_trans", "v_accvgpr_read_b32": "valu",
        "s_add_u32": "salu", "s_load_dwordx4": "smem", "s_buffer_load_dword": "smem",
        "buffer_load_dwordx4": "vmem_load", "global_load_b128": "vmem_load",
        "buffer_store_dwordx2": "vmem_store", "global_atomic_add_f32": "vmem_atomic",
        "ds_read_b128": "ds_read", "ds_load_b64": "ds_read", "ds_read_b64_tr_b16": "ds_read",
        "ds_write_b64": "ds_write", "ds_store_b32": "ds_write", "ds_bpermute_b32": "ds_other",
        "s_waitcnt": "waitcnt", "s_waitcnt_vscnt": "waitcnt", "s_barrier": "barrier",
        "s_cbranch_scc1": "branch", "s_branch": "branch", "s_nop": "nop",
        "s_setprio": "other", "s_delay_alu": "other", "s_endpgm": "other",
    }
    for mn, want in cases.items():
        assert classify(mn) == want, mn
    # gfx950's DMA: a buffer load that writes LDS.
    assert classify("buffer_load_dwordx4", "v132, s[24:27], 0 offen lds") == "vmem_lds"


def test_matrix_flops_from_the_mnemonic():
    from hk.profile.isa import mma_flops

    assert mma_flops("v_mfma_f32_16x16x32_bf16") == 2 * 16 * 16 * 32
    assert mma_flops("v_mfma_f32_32x32x16_bf16") == 2 * 32 * 32 * 16
    assert mma_flops("v_wmma_f32_16x16x16_bf16") == 2 * 16 * 16 * 16
    assert mma_flops("v_mfma_f32_32x32x8bf16_1k") == 2 * 32 * 32 * 8
    # The legacy multi-block ones: 2 blocks of 32x32, 16 of 4x4.
    assert mma_flops("v_mfma_f32_32x32x4bf16_1k") == 2 * 32 * 32 * 4 * 2
    assert mma_flops("v_mfma_f32_4x4x4bf16_1k") == 2 * 4 * 4 * 4 * 16
    assert mma_flops("v_add_f32") == 0


def test_waits_and_widths():
    from hk.profile.isa import access_bytes, wait_kind

    assert wait_kind("s_waitcnt", "vmcnt(6)") == "vmem"
    assert wait_kind("s_waitcnt", "lgkmcnt(0)") == "lgkm"
    assert wait_kind("s_waitcnt", "vmcnt(0) lgkmcnt(0)") == "vmem+lgkm"
    assert wait_kind("s_waitcnt_vscnt", "null, 0x0") == "vmem"
    assert access_bytes("buffer_load_dwordx4") == 16
    assert access_bytes("buffer_load_dwordx2") == 8
    assert access_bytes("buffer_store_short") == 2
    assert access_bytes("global_load_b128") == 16
    assert access_bytes("ds_read_b64_tr_b16") == 8
    assert access_bytes("v_add_f32") is None


# A kernel in llvm-objdump's format: a prologue, a loop (0x1014-0x1030) of
# waits, an LDS read, an MFMA and a barrier, a drain MFMA, an epilogue store.
OBJDUMP = """
/tmp/k.co:\tfile format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z6kernel>:
\ts_load_dwordx4 s[0:3], s[4:5], 0x0                         // 000000001000: C00A0000 00000000
\tv_mov_b32_e32 v0, 0                                        // 000000001008: 7E000280
\tbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen            // 00000000100C: E05C1000 80000401
\ts_waitcnt vmcnt(0)                                         // 000000001014: BF8C0F70
\tds_read_b128 v[8:11], v2                                   // 000000001018: D9FE0000 08000002
\ts_waitcnt lgkmcnt(0)                                       // 000000001020: BF8CC07F
\tv_mfma_f32_16x16x32_bf16 v[12:15], v[8:11], v[8:11], v[12:15] // 000000001024: D3B5000C 04321108
\ts_barrier                                                  // 00000000102C: BF8A0000
\ts_cbranch_scc1 65528                                       // 000000001030: BF85FFF8 <_Z6kernel+0x14>
\tv_mfma_f32_16x16x32_bf16 v[12:15], v[8:11], v[8:11], v[12:15] // 000000001034: D3B5000C 04321108
\tv_cvt_pk_bf16_f32 v0, v12, v13                             // 00000000103C: D2680000 00021B0C
\tbuffer_store_dwordx2 v[0:1], v1, s[0:3], 0 offen           // 000000001044: E0741000 80000001
\ts_endpgm                                                   // 00000000104C: BF810000
"""


def test_objdump_parse_loop_and_parts():
    from hk.profile import isa

    (f,) = isa.parse_objdump(OBJDUMP)
    assert f.name == "_Z6kernel" and len(f.insts) == 13
    br = next(i for i in f.insts if i.cls == "branch")
    assert br.target == 0x1014
    (lp,) = isa.loops(f)
    assert (lp.start, lp.end) == (0x1014, 0x1030)
    res = isa.analyze(f)
    ml = res["main_loop"]
    assert ml["instructions"] == 6
    assert ml["mix"]["classes"] == {"mfma": 1, "ds_read": 1, "waitcnt": 2, "barrier": 1,
                                    "branch": 1}
    assert ml["mix"]["waits"] == {"vmem": 1, "lgkm": 1}
    assert ml["mix"]["per_matrix_instruction"] == 5.0
    assert ml["mix"]["mma_flops"] == 16384
    assert res["prologue"]["instructions"] == 3
    assert res["prologue"]["mix"]["bytes_per_lane"] == {"vmem_load": {"16": 1}}
    # After the loop: the drain ends at the last MFMA, the epilogue follows.
    assert res["drain"]["instructions"] == 1 and res["drain"]["mix"]["classes"] == {"mfma": 1}
    assert res["epilogue"]["mix"]["classes"] == {"valu": 1, "vmem_store": 1, "other": 1}
    assert isa.regions(res) == [(0, "prologue"), (0x1014, "main_loop"), (0x1034, "drain"),
                                (0x103C, "epilogue")]


def test_trace_cycles_choose_the_main_loop():
    """With a thread trace the main loop is the one the waves spent their
    cycles in, and each part gets its share."""
    from hk.profile import isa

    (f,) = isa.parse_objdump(OBJDUMP)
    stats = {i.addr: {"hits": 1, "latency": 10, "stall": 0, "idle": 0} for i in f.insts}
    stats[0x102C] = {"hits": 4, "latency": 400, "stall": 380, "idle": 0}   # the barrier
    res = isa.analyze(f, att=stats, waves_traced=2)
    total = 10 * 12 + 400
    assert res["main_loop"]["cycles"] == 5 * 10 + 400
    assert res["main_loop"]["cycle_share"] == round(450 / total, 4)
    by = res["main_loop"]["att"]["by_class"]
    assert by["barrier"]["stall"] == 380 and by["waitcnt:vmem"]["latency"] == 10


def test_thread_trace_rows_rebuild_the_code():
    from hk.profile import isa

    code = [
        ["; _Z6kernel", 0, 0, "kernel(args)", 1, 4096, 0, 0, 0, 0],
        ["s_waitcnt vmcnt(0)", 0, 1, "", 1, 4096, 8, 80, 70, 0],
        ["v_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[4:7], v[0:3]", 0, 2, "", 1, 4100, 8,
         128, 64, 0],
        ["s_cbranch_scc1 65532", 0, 3, "", 1, 4108, 8, 8, 0, 0],
        ["s_endpgm", 0, 4, "", 1, 4112, 1, 4, 0, 0],
    ]
    funcs, stats = isa.from_att(code)
    (f,) = funcs
    assert f.name == "kernel(args)" and len(f.insts) == 4
    assert f.insts[2].target == 4096          # 4108 + 4 + 4 * (-4)
    assert stats[4100] == {"hits": 8, "latency": 128, "stall": 64, "idle": 0}
    res = isa.analyze(f, att=stats, waves_traced=2)
    assert res["main_loop"]["start"] == hex(4096)
    n, fl = isa.matrix_ops_per_wave(f, stats, 2)
    assert n == 4 and fl == 4 * 16384


def test_sibling_hot_loops_merge():
    """A K loop compiled in two copies: the main loop spans both."""
    from hk.profile import isa

    text = OBJDUMP.replace(
        "\ts_endpgm                                                   // 00000000104C: BF810000",
        "\tv_mfma_f32_16x16x32_bf16 v[12:15], v[8:11], v[8:11], v[12:15] // 00000000104C: D3B5000C 04321108\n"
        "\ts_cbranch_scc1 65533                                       // 000000001054: BF85FFFD <_Z6kernel+0x4c>\n"
        "\ts_endpgm                                                   // 000000001058: BF810000")
    (f,) = isa.parse_objdump(text)
    assert len(isa.loops(f)) == 2
    res = isa.analyze(f)
    assert res["main_loop"]["merged_loops"] == 2
    assert res["main_loop"]["start"] == hex(0x1014) and res["main_loop"]["end"] == hex(0x1054)


def _elf_with(sections):
    """A minimal ELF64 holding `sections` ({name: bytes})."""
    names = b"\0" + b"".join(n.encode() + b"\0" for n in [".shstrtab", *sections])
    body = bytearray(64)
    offs = {}
    for n, data in sections.items():
        offs[n] = len(body)
        body += data
    stroff = len(body)
    body += names
    shoff = len(body)
    hdrs = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    pos = 1
    hdrs.append(struct.pack("<IIQQQQIIQQ", pos, 3, 0, 0, stroff, len(names), 0, 0, 1, 0))
    pos += len(".shstrtab") + 1
    for n, data in sections.items():
        hdrs.append(struct.pack("<IIQQQQIIQQ", pos, 1, 2, 0, offs[n], len(data), 0, 0, 1, 0))
        pos += len(n) + 1
    body += b"".join(hdrs)
    body[0:4] = b"\x7fELF"
    struct.pack_into("<Q", body, 0x28, shoff)
    struct.pack_into("<HHH", body, 0x3A, 64, len(hdrs), 1)
    return bytes(body)


def test_code_object_out_of_the_offload_bundle(tmp_path):
    from hk.profile import isa

    dev = b"\x7fELF-device-code-object"
    entries = [("host-x86_64-unknown-linux-gnu-", b""),
               ("hipv4-amdgcn-amd-amdhsa--gfx950", dev)]
    head = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", len(entries))
    meta_len = len(head) + sum(24 + len(t) for t, _ in entries)
    blob = bytearray(head)
    data_off = meta_len
    for t, d in entries:
        blob += struct.pack("<QQQ", data_off, len(d), len(t)) + t.encode()
        data_off += len(d)
    for _, d in entries:
        blob += d
    so = tmp_path / "k.so"
    so.write_bytes(_elf_with({".text": b"\0" * 16, ".hip_fatbin": bytes(blob)}))
    assert isa.code_object(so, "gfx950") == dev
    with pytest.raises(ValueError, match="gfx1100"):
        isa.code_object(so, "gfx1100")
    co = tmp_path / "kernel.co"
    co.write_bytes(_elf_with({".text": b"\0" * 8}))
    assert isa.code_object(co) == co.read_bytes()


# ------------------------------------------------------------------ counters, grid


def _record(name):
    p = FIX / f"gfx950-{name}.json"
    if not p.exists():
        pytest.skip(f"fixture {p.name} missing")
    return json.loads(p.read_text())


def test_counters_derive_on_a_recorded_gemm():
    """Llama gate_up at M=1024 on an MI355X: 448 workgroups of 8 waves on 256
    CUs, so 1.75 rounds -- the CUs busy ~80%, the waves ~1.6 of 2 a SIMD."""
    from hk.profile.counters import derive

    doc = _record("a")
    k = doc["kernels"][0]
    m = derive("gfx950", k["rec"]["counters"], doc["agent"])
    assert m["waves"] == 448 * 8
    assert 0.7 < m["cu_busy"] < 0.9
    assert 1.4 < m["achieved_waves_per_simd"] < 1.8
    assert 0.5 < m["mfma_busy"] < 0.75
    # SQ_INSTS_MFMA is 2MNK / 16384 exactly; the MOPS counters give the FLOPs.
    assert m["instructions"]["mfma"] == 2 * 1024 * 28672 * 4096 // 16384
    assert m["matrix_flops"] == 2 * 1024 * 28672 * 4096
    # C, written once: its bytes past the L2 (within 0.1%).
    assert abs(m["write_bytes"] / (1024 * 28672 * 2) - 1) < 1e-3


def test_streaming_kernel_bytes_are_its_tensors():
    """GELU-and-mul at 4096x14336 reads two inputs and writes one, each
    117 MB: TCC_EA0_*_DRAM_32B x 32 gives exactly that (FETCH_SIZE gave half)."""
    from hk.profile.counters import derive

    doc = _record("c")
    m = derive("gfx950", doc["kernels"][0]["rec"]["counters"], doc["agent"])
    one = 4096 * 14336 * 2
    assert abs(m["fetch_bytes"] - 2 * one) / (2 * one) < 0.01
    assert abs(m["write_bytes"] - one) / one < 0.01


def test_grid_fit_names_the_tail():
    from hk.profile.analyze import grid_fit

    agent = {"simds": 1024, "cus": 256, "wave": 64}
    g = grid_fit("gfx950", agent, {"workgroups": 448, "workgroup": 512, "lds_bytes": 131072,
                                   "vgpr": 234, "agpr": 0})
    assert g["slots"] == 256 and g["workgroups_per_slot"] == 1
    assert g["rounds"] == 1.75 and g["idle_share"] == 0.125
    assert g["last_round"] == {"workgroups": 192, "of": 256}
    assert "lds" in g["bound_by"] and g["theoretical_waves_per_simd"] == 2
    # gfx1100: the slot is a WGP (4 SIMD32 sharing 128 KB): 96 CUs are 48.
    g = grid_fit("gfx1100", {"simds": 192, "cus": 96, "wave": 32},
                 {"workgroups": 96, "workgroup": 256, "lds_bytes": 32768, "vgpr": 96})
    assert g["slot"] == "WGP" and g["slots"] == 48
    assert g["workgroups_per_slot"] == 4 and g["rounds"] == 0.5
    assert g["idle_share"] == 0.5


def test_ceilings_are_hks_own():
    from hk.check import CARDS
    from hk.integration.headroom import PEAKS
    from hk.profile.analyze import CEILINGS

    for arch in ("gfx950", "gfx1100"):
        c = CEILINGS[arch]
        assert c["matrix_flops"]["bf16"] == PEAKS[arch][0]
        assert c["dram_bytes_s"] == PEAKS[arch][1]
        assert c["cache_bytes_s"] == CARDS[arch]["cache_bytes_s"]
        assert c["cache_bytes"] == CARDS[arch]["cache_bytes"]


def test_counter_groups_named_and_bounded():
    from hk.profile.counters import CANNOT, PASSES

    for arch, groups in PASSES.items():
        assert CANNOT[arch]
        names = [n for n, _ in groups]
        assert len(set(names)) == len(names)
        for _, cs in groups:
            assert 1 <= len(cs) <= 12


def test_counters_that_read_zero_are_dropped():
    from hk.profile import _counters_read_zero

    assert _counters_read_zero([{"median_ns": 1000, "counters": {"GRBM_GUI_ACTIVE": 0.0,
                                                                 "SQ_WAVES": 64}}])
    assert not _counters_read_zero([{"median_ns": 1000, "counters": {"GRBM_GUI_ACTIVE": 9.0}}])
    assert not _counters_read_zero([{"median_ns": 1000, "counters": {}}])


# ------------------------------------------------------------------ the matrix-idle model


def test_matrix_idle_attributes_the_gaps():
    """One SIMD, two waves: an MFMA busy 16 cycles from issue, then one wave
    at a barrier and the other at a vmem wait for 32 cycles -- the idle half
    split between them."""
    from hk.profile.timeline import matrix_idle

    labels = {0: ("matrix_core", "main_loop"), 1: ("barrier", "main_loop"),
              2: ("vmem_wait", "epilogue")}
    w0 = {"cu": 1, "simd": 0, "begin": 0, "end": 64,
          "instructions": [[0, 6, 0, 16, 0], [16, 9, 32, 48, 1]]}
    w1 = {"cu": 1, "simd": 0, "begin": 0, "end": 64,
          "instructions": [[16, 6, 0, 16, 0], [32, 9, 32, 32, 2]]}
    r = matrix_idle([w0, w1], arch="gfx950", label=lambda i: labels[i],
                    pipe_cycles=lambda i: 16, is_matrix=lambda i: i == 0)
    assert r["busy"] == 0.5
    assert r["idle"] == {"barrier": 0.25, "vmem_wait": 0.25}
    assert r["idle_by_region"] == {"main_loop": 0.25, "epilogue": 0.25}
    assert r["no_wave"] == 0.0


# ------------------------------------------------------------------ verdicts on recorded runs


def _replay(name):
    from hk.profile import replay

    p = FIX / f"gfx950-{name}.json"
    if not p.exists():
        pytest.skip(f"fixture {p.name} missing")
    return replay(p)


def test_case_a_names_the_tail_and_the_epilogue():
    """(a) gate_up at M=1024: compute-bound with a 1.75-wave tail; the
    template's bias+GELU puts more of the matrix core's idle time after the
    main loop, where nothing overlaps it (one workgroup a CU)."""
    (plain,) = _replay("a")
    (ep,) = _replay("a.vs-bias_gelu")
    v = plain["verdict"]
    assert v["bound"] == "compute_bound"
    assert "tail" in v["losses"]
    tail = next(f for f in plain["findings"] if f["kind"] == "tail")
    assert tail["evidence"]["rounds"] == 1.75 and tail["cost"] == 0.125
    after = [k["stalls"]["matrix_idle"]["idle_by_region"] for k in (plain, ep)]
    tail_idle = [a.get("drain", 0) + a.get("epilogue", 0) for a in after]
    assert tail_idle[1] > 1.8 * tail_idle[0]
    assert ep["instructions"]["valu"] > 1.15 * plain["instructions"]["valu"]
    assert ep["grid"]["workgroups_per_slot"] == 1


def test_case_b_names_the_issue_bound_route():
    """(b) 2048x16384x4096 on dp128: one 4-wave workgroup a CU (its LDS), a
    loop of 2.4 other instructions per MFMA, the matrix core idle ~half the
    time; hipBLASLt's loop has under one."""
    (dp,) = _replay("b")
    (blas,) = _replay("b.vs-hipblaslt")
    assert "issue_bound" in dp["verdict"]["losses"]
    ib = next(f for f in dp["findings"] if f["kind"] == "issue_bound")
    assert ib["evidence"]["waves_per_simd"] == 1 and ib["evidence"]["bound_by"] == "lds"
    assert ib["evidence"]["per_matrix_instruction"] > 2
    assert ib["cost"] > 0.3
    assert dp["matrix_core"]["busy"] < 0.5 < blas["matrix_core"]["busy"]
    assert blas["isa"]["main_loop"]["mix"]["per_matrix_instruction"] < 1


def test_case_c_streaming():
    """(c) GELU-and-mul: the strip kernel is memory-bound (>= 80% of HBM at
    least); the 16x64-tile one is not, waits at its 8-byte loads' issue."""
    (tiles,) = _replay("c")
    (strips,) = _replay("c.vs-strips")
    assert strips["verdict"]["bound"] == "memory_bound"
    assert strips["roofline"]["of_dram_min"] >= 0.8
    assert tiles["verdict"]["bound"] == "latency_bound"
    kinds = [f["kind"] for f in tiles["findings"]]
    assert "narrow_access" in kinds
    na = next(f for f in tiles["findings"] if f["kind"] == "narrow_access")
    assert na["evidence"]["bytes_per_lane"]["vmem_load"] == {"8": 8}
    assert tiles["stalls"]["att"]["by_reason"]["vmem_issue"]["share"] > 0.5
    assert "matrix" not in tiles["verdict"]["line"]


def test_verdict_kinds_are_the_documented_ones():
    from hk.profile.analyze import KINDS

    for name in ("a", "a.vs-bias_gelu", "b", "b.vs-hipblaslt", "c", "c.vs-strips"):
        for k in _replay(name):
            for f in k["findings"]:
                assert f["kind"] in KINDS
                assert set(f) >= {"kind", "cost", "text", "evidence"}
            assert k["verdict"]["bound"] in (None, "memory_bound", "compute_bound",
                                             "latency_bound")


def test_cli_parses():
    from hk.profile.__main__ import build_parser

    a = build_parser().parse_args(["x.py:k", "--inputs", "bf16[16,64]", "--vs", "b=y.py:k",
                                   "--no-att", "--json", "-"])
    assert a.target == "x.py:k" and a.vs == "b=y.py:k" and a.no_att and a.json == "-"


def test_gfx1100_without_counters():
    """W7900D, hk.ops.matmul at 4096^3 against hipBLASLt, the counters
    unavailable there: the verdict comes from the timing, the thread trace
    and the ISA -- compute-bound at ~72% of the WMMA peak, the matrix pipe
    of the traced SIMD busy about as much; hipBLASLt at 4 waves a SIMD with
    more of its loop spent on other instructions."""
    from hk.profile import replay

    p = FIX / "gfx1100-gemm.json"
    if not p.exists():
        pytest.skip("fixture missing")
    (hk_,) = replay(p)
    (blas,) = replay(FIX / "gfx1100-gemm.vs-hipblaslt.json")
    assert "counters" not in hk_
    assert hk_["verdict"]["bound"] == "compute_bound"
    assert hk_["matrix_core"]["source"].startswith("derived")
    assert abs(hk_["stalls"]["matrix_idle"]["busy"] - hk_["matrix_core"]["busy"]) < 0.1
    assert hk_["grid"]["slot"] == "WGP" and hk_["grid"]["slots"] == 48
    assert blas["matrix_core"]["busy"] < hk_["matrix_core"]["busy"]
    assert (blas["isa"]["main_loop"]["mix"]["per_matrix_instruction"]
            > hk_["isa"]["main_loop"]["mix"]["per_matrix_instruction"])
