"""Generate, compile, and read the resource remarks back.

Needs hipcc; does not need a GPU or torch, because nothing here launches. That
is the point of this tier: the thing most likely to break -- generated code that
compiles but spills, or compiles for the wrong arch -- is caught on a laptop.

Cold, each kernel costs a few seconds of hipcc. The content-hash cache makes
every rerun free, so keep the cache directory between runs.
"""

from __future__ import annotations

import os
import struct

import pytest

import hk
from hk import bf16, fp32
from hk.ops.elementwise import KERNELS
from hk.runtime import resources
from hk.runtime.compile import cache_key


@hk.kernel(
    arch="gfx1100",
    grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1),
)
def add2(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16], *,
         ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    o[idx] = a[idx].load(t) + b[idx].load(t)


@pytest.fixture(scope="module")
def build():
    return add2.build()


def test_it_compiles_to_a_shared_object(build):
    assert build.so_path.exists()
    assert build.so_path.suffix == ".so"


def test_the_resource_remarks_are_parsed(build):
    """If this comes back empty the spill gate is vacuous -- it would pass
    every kernel, including the ones that spill."""
    assert build.kernels, "no -Rpass-analysis remarks were parsed"
    (k,) = [k for k in build.kernels if "add2" in k.name]
    assert k.scratch == 0 and not k.spills
    assert k.vgpr > 0 and (k.occ_by_reg or k.occupancy) > 0


def test_the_second_build_is_a_cache_hit(build):
    again = add2.build()
    assert again.key == build.key
    assert again.so_path == build.so_path


def test_the_cache_key_covers_the_compiler_flags():
    """The include-path bug this guards against was invisible for a while: the
    fix changed the flags but not the key, so the broken .so was served from
    cache and the fix looked like it had not worked."""
    t = hk.get_target("gfx1100")
    src = add2.source()
    assert cache_key(src, t) != cache_key(src, t, extra=("-DFOO=1",))


def test_a_different_constexpr_is_a_different_build():
    wide = add2.build(COLS=128)
    assert wide.key != add2.build().key
    assert "rt<bf16, 16, 128," in add2.source(COLS=128)


# -- the gates ----------------------------------------------------------------


def test_the_occupancy_gate_can_fail_a_build():
    """The no-spill rule is unconditional and (correctly) never fires on these
    kernels, so exercise the mechanism through the optional gate instead. A
    gate that has never been seen to fail is not known to work."""
    k = add2.specialize()
    k.min_occupancy = 99
    with pytest.raises(resources.ResourceError, match="min_occupancy"):
        k.build()


def test_the_vgpr_gate_can_fail_a_build():
    k = add2.specialize()
    k.max_vgprs = 4
    with pytest.raises(resources.ResourceError, match="max_vgprs"):
        k.build()


def test_a_failed_build_keeps_its_source():
    """The generated .cpp is the artifact you take to the disassembler; losing
    it on failure is losing the only thing worth looking at."""
    from hk.runtime.compile import CompileError

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def broken(o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        o[i] = hk.zeros(hk.RT(bf16, 16, 16))

    src = broken.source().replace("kittens::store", "kittens::no_such_op")
    with pytest.raises(CompileError) as e:
        hk.runtime.build(src, "gfx1100", name="broken")
    assert "failed-" in str(e.value)


def test_an_unknown_arch_is_refused_rather_than_defaulted():
    """Guessing --offload-arch either fails at load time or, on a near-enough
    arch, computes the wrong answer."""
    with pytest.raises(KeyError, match="unknown arch"):
        hk.get_target("gfx9999")


# -- every shipped kernel -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_shipped_kernel_builds_without_spilling(name):
    b = KERNELS[name].build()
    assert b.kernels
    for k in b.kernels:
        assert not k.spills, f"{name}: {k.line()}"


# -- the key covers the headers -----------------------------------------------


_PROBE_KERNEL = """#include "kittens.cuh"
#include "probe/outer.cuh"
extern "C" __global__ void hk_probe(int* out) { out[0] = PROBE_INNER; out[1] = PROBE_DEVICE; }
"""

_PROBE_MODULE = _PROBE_KERNEL + """#include <pybind11/pybind11.h>
static int probe_host() { return PROBE_HOST; }
PYBIND11_MODULE(HK_MODULE_NAME, m) { m.def("host", &probe_host); }
"""


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A header overlay ahead of hk's include path: probe/outer.cuh includes inner.cuh
    (two deep from the kernel), device.cuh under __HIP_DEVICE_COMPILE__ only, host.cuh
    otherwise; probe/unrelated.cuh is included by nothing."""
    from hk.runtime import compile as C

    real = C.include_dirs()
    d = tmp_path / "overlay" / "probe"
    d.mkdir(parents=True)
    (d / "outer.cuh").write_text('#pragma once\n#include "inner.cuh"\n'
                                 '#if defined(__HIP_DEVICE_COMPILE__)\n#include "device.cuh"\n#define PROBE_HOST 0\n'
                                 '#else\n#include "host.cuh"\n#define PROBE_DEVICE 0\n#endif\n')
    vals = {}

    def put(name, macro, v):
        (d / name).write_text(f"#pragma once\n#define {macro} {v:#x}\n")
        vals[macro] = v
    put("inner.cuh", "PROBE_INNER", 0x5EED0101)
    put("device.cuh", "PROBE_DEVICE", 0x5EED0201)
    put("host.cuh", "PROBE_HOST", 0x5EED0301)
    put("unrelated.cuh", "PROBE_UNRELATED", 0x5EED0401)
    monkeypatch.setenv("HK_INCLUDE_DIR", os.pathsep.join(map(str, [tmp_path / "overlay", *real])))
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(C, "_FILE_HASH", {})

    def build(kind="co"):
        b = hk.runtime.build(_PROBE_KERNEL if kind == "co" else _PROBE_MODULE, "gfx1100",
                             name="probe", kind=kind)
        data = b.so_path.read_bytes()
        return b, {m: struct.pack("<I", v) in data for m, v in vals.items()}
    return put, build


def test_a_header_edit_two_includes_deep_recompiles(probe):
    """On 0.8.0 an edit to include/hk/cdna_agpr.cuh (included by cdna.cuh) or to
    include/pyutils/hk_bind.cuh kept the old key, and the old kernel was served."""
    put, build = probe
    b0, has = build()
    assert not b0.cached and has["PROBE_INNER"] and has["PROBE_DEVICE"]
    assert build()[0].cached
    put("inner.cuh", "PROBE_INNER", 0x5EED0102)
    b1, has = build()
    assert not b1.cached and b1.key != b0.key and has["PROBE_INNER"]


def test_a_header_only_the_device_side_includes_is_keyed(probe):
    """hipcc's -MD lists the host side's headers only (and none for a code object)."""
    put, build = probe
    b0, _ = build()
    put("device.cuh", "PROBE_DEVICE", 0x5EED0202)
    b1, has = build()
    assert not b1.cached and b1.key != b0.key and has["PROBE_DEVICE"]


def test_a_header_only_the_host_side_includes_is_keyed(probe):
    put, build = probe
    b0, has = build("so")
    assert has["PROBE_HOST"]
    put("host.cuh", "PROBE_HOST", 0x5EED0302)
    b1, has = build("so")
    assert not b1.cached and b1.key != b0.key and has["PROBE_HOST"]


def test_a_header_the_kernel_does_not_include_leaves_it_cached(probe):
    put, build = probe
    b0, _ = build()
    put("unrelated.cuh", "PROBE_UNRELATED", 0x5EED0402)
    b1, _ = build()
    assert b1.cached and b1.key == b0.key
