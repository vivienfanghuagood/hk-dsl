"""The spill gate reads the kernel-resource-usage remarks of every tested compiler.

clang 22 and 23 print them as clang diagnostics (`k.cpp:3:1: remark: VGPRs: 2
[-Rpass-analysis=kernel-resource-usage]`). clang 24 (TheRock, rocm/pytorch 10.1)
prints them bare for a -shared build -- `k.cpp:3:0:     VGPRs: 2`, from
clang-linker-wrapper, which generates the device code at link time -- and the
parser that read only the first shape failed every compile there with "the
build emitted no kernel-resource-usage remarks". The fixtures are those
compilers' output, captured as tests/hk/remark_fixtures/SOURCES.txt says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hk.runtime.resources import ResourceError, parse, remark

FIX = Path(__file__).resolve().parents[1] / "remark_fixtures"

SCALE, STACK4, STACK256 = "_Z5scalePff", "_Z5stackILi4EEvPfPKi", "_Z5stackILi256EEvPfPKi"

#: fixture -> {mangled: (vgpr, sgpr, scratch, occupancy)}, read off the files by eye.
EXPECTED = {
    "clang22-gfx950-shared.txt": {SCALE: (2, 11, 0, 8), STACK4: (7, 14, 0, 8),
                                  STACK256: (36, 33, 1040, 8)},
    "clang23-gfx950-shared.txt": {SCALE: (2, 11, 0, 8), STACK4: (7, 14, 0, 8),
                                  STACK256: (40, 21, 1040, 8)},
    "clang24-gfx950-shared.txt": {SCALE: (2, 11, 0, 8), STACK4: (7, 14, 0, 8),
                                  STACK256: (40, 21, 1040, 8)},
    "clang24-gfx950-genco.txt": {SCALE: (2, 11, 0, 8), STACK4: (7, 14, 0, 8),
                                 STACK256: (40, 21, 1040, 8)},
    "clang24-gfx1100-shared.txt": {SCALE: (2, 4, 0, 16), STACK4: (8, 10, 0, 16),
                                   STACK256: (49, 15, 1040, 16)},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_compiler_s_remarks_parse(name):
    ks = {k.mangled: k for k in parse((FIX / name).read_text())}
    assert set(ks) == set(EXPECTED[name])
    for mangled, (vgpr, sgpr, scratch, occ) in EXPECTED[name].items():
        k = ks[mangled]
        assert (k.vgpr, k.sgpr, k.scratch, k.occupancy) == (vgpr, sgpr, scratch, occ), mangled
        assert (k.agpr, k.lds, k.vgpr_spill, k.sgpr_spill) == (0, 0, 0, 0), mangled
    # the scratch frame is what the gate refuses, in either shape
    assert ks[STACK256].spills and not ks[SCALE].spills


def test_the_two_shapes_say_the_same_thing():
    """clang 24's bare -shared output and its --genco diagnostics: same kernels, same numbers."""
    def rows(name):
        return sorted((k.mangled, k.vgpr, k.agpr, k.sgpr, k.scratch, k.occupancy, k.lds,
                       k.vgpr_spill, k.sgpr_spill) for k in parse((FIX / name).read_text()))

    assert rows("clang24-gfx950-shared.txt") == rows("clang24-gfx950-genco.txt")
    assert "remark:" not in (FIX / "clang24-gfx950-shared.txt").read_text()


def test_kernels_are_grouped_by_their_function_name_line():
    log = "\n".join([
        "/tmp/hk-build-x/kernel.cpp:12:0: Function Name: _Z1av",
        "/tmp/hk-build-x/kernel.cpp:12:0:     VGPRs: 10",
        "/tmp/hk-build-x/kernel.cpp:12:0:     VGPRs Spill: 3",
        "/tmp/hk-build-x/kernel.cpp:40:0: Function Name: _Z1bv",
        "/tmp/hk-build-x/kernel.cpp:40:0:     VGPRs: 20",
        "/tmp/hk-build-x/kernel.cpp:40:0:     SGPRs Spill: 1",
        "/tmp/hk-build-x/kernel.cpp:40:0:     Dynamic Stack: True",
    ])
    ks = {k.mangled: k for k in parse(log)}
    assert (ks["_Z1av"].vgpr, ks["_Z1av"].vgpr_spill, ks["_Z1av"].sgpr_spill) == (10, 3, 0)
    assert (ks["_Z1bv"].vgpr, ks["_Z1bv"].vgpr_spill, ks["_Z1bv"].sgpr_spill) == (20, 0, 1)


@pytest.mark.parametrize("line", [
    # clang's own diagnostics that are not these remarks
    "k.cpp:3:1: warning: VGPRs: 2",
    "k.cpp:3:1: note: VGPRs: 2",
    "k.cpp:3:1: remark: VGPRs: 2 [-Rpass=inline]",
    "k.cpp:3:1: remark: VGPRs: 2",
    "k.cpp:3:1: remark: Function Name: _Z1fv [-Rpass=inline]",
    # a bare line needs a file:line:col location and a known key
    "VGPRs: 2",
    "    VGPRs: 2",
    "Function Name: _Z1fv",
    "k.cpp:3:     VGPRs: 2",
    "k.cpp:3:0:     Registers: 2",
    "k.cpp:3:0:     VGPRs: 2 (rounded)",
    "k.cpp:3:0: error: Function Name: _Z1fv",
    "ld.lld: warning: VGPRs: 2",
    # clang's source snippet under a diagnostic
    "    3 | __global__ void scale(float* x, float a) { x[threadIdx.x] *= a; }",
    "      | ^",
])
def test_other_output_is_not_read_as_a_remark(line):
    assert remark(line) is None


@pytest.mark.parametrize("line,want", [
    ("k.cpp:3:1: remark: Function Name: _Z5scalePff [-Rpass-analysis=kernel-resource-usage]",
     ("name", "_Z5scalePff")),
    ("k.cpp:3:1: remark:     ScratchSize [bytes/lane]: 16 [-Rpass-analysis=kernel-resource-usage]",
     ("scratch", 16)),
    ("k.cpp:3:0: Function Name: _Z5scalePff", ("name", "_Z5scalePff")),
    ("k.cpp:3:0:     Occupancy [waves/SIMD]: 8", ("occupancy", 8)),
    ("/a dir/kernel.cpp:7:0:     LDS Size [bytes/block]: 32768", ("lds", 32768)),
])
def test_one_line_of_either_shape(line, want):
    assert remark(line) == want


def test_no_remarks_names_the_compiler_and_shows_its_output():
    with pytest.raises(ResourceError) as e:
        parse("ld.lld: warning: something else\nk.cpp:3:1: warning: unused\n",
              compiler="AMD clang version 99.0.0git")
    msg = str(e.value)
    assert "AMD clang version 99.0.0git" in msg
    assert "ld.lld: warning: something else" in msg
    # the old message sent people to check a flag hk always passes
    assert "survived into the hipcc command line" not in msg

    with pytest.raises(ResourceError, match="printed no output at all"):
        parse("", compiler="AMD clang version 99.0.0git")
