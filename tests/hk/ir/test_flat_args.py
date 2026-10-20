"""On CDNA a noalias kernel takes the globals' fields as scalar arguments
(HKToCpp's flat form): what hk.ops's strip kernels generate for gfx950, the
`globals` they still take on RDNA, the layout table the code-object launcher
packs by, and where the preloaded twin asks hipcc to preload them."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.ops import elementwise, fused, norm, quant  # noqa: E402

CONSTS = dict(CHUNKS=4, RUN=8, EXACT=1)


def _on(k, arch):
    return hk.kernel(k.fn, arch=arch, warps=k.warps, grid=k.grid_fn,
                     name=f"{k.name}_{arch}", defines=k.defines, noalias=k.noalias)


def _addrmsw(arch):
    return _on(norm.STRIP_KERNELS["strip_addrmsw_bf16_w2"], arch)


def test_the_strip_kernels_are_noalias():
    tables = [norm.STRIP_KERNELS, elementwise.STRIP_KERNELS, quant.STRIP_KERNELS,
              fused.STRIP_KERNELS, fused.ROTARY_KERNELS, fused.ROTARY16_KERNELS]
    assert all(k.noalias for t in tables for k in t.values())
    assert not any(k.noalias for k in norm.KERNELS.values())


def test_on_gfx950_a_noalias_kernel_takes_flat_arguments():
    tr = _addrmsw("gfx950").trace(**CONSTS)
    assert tr.flat_args
    src = scaffold.render(tr, "bare")
    name = "strip_addrmsw_bf16_w2_gfx950_kernel"
    ptrs = ", ".join(f"bf16 *__restrict__ hk_{t}" for t in ("x", "r", "w", "o", "r_out"))
    assert f"namespace hk_flat {{\n__global__ __launch_bounds__(128, 1)\nvoid {name}({ptrs}, " \
           "int hk_x_rs, int hk_r_rs, " in src
    assert "const globals g{decltype(globals::x)(hk_x, hk_x_b, hk_x_d, hk_x_r, hk_x_c, " \
           "hk_x_rs), " in src
    # Two warps combine through LDS: the launch sets its size on the flat kernel.
    assert f"hipFuncSetAttribute((void *)&hk_flat::{name}, " in src
    assert re.search(rf"hk_flat::{name}<<<g\.grid\(\), g\.block\(\), \d+, stream>>>"
                     r"\(g\.x\.raw_ptr, g\.r\.raw_ptr, ", src)
    assert scaffold.kernel_symbol(tr) == f"_ZN7hk_flat{len(name)}{name}E"


def test_on_rdna_it_takes_globals_and_the_module_is_unchanged():
    tr = _addrmsw("gfx1100").trace(**CONSTS)
    assert not tr.flat_args and "hk.noalias" not in tr.dump()
    src = scaffold.render(tr, "bare")
    assert "_kernel(const globals g) {" in src and "hk_flat" not in src
    assert scaffold.kernel_symbol(tr).endswith("_kernel7globals")


def test_the_layout_table_is_the_flat_parameters():
    src = scaffold.device(_addrmsw("gfx950").trace(**CONSTS))
    table = src[src.index("hk_layout[]"):]
    assert "(long long)(offsetof(hk_kernarg, hk_r_out_d) + sizeof(int))" in table
    assert table.count("(long long)(0)") == 5
    assert "(long long)(offsetof(hk_kernarg, hk_x_rs))" in table


def test_the_preloaded_twin_asks_for_preload_only_where_it_exists():
    k = _addrmsw("gfx950")
    tr = k.trace(**CONSTS)
    assert k.build_flags(tr) == ()
    assert k.preloaded() is k.preloaded() and k.preloaded().preloaded() is k.preloaded()
    assert k.preloaded().build_flags(tr) == ("-mllvm", "-amdgpu-kernarg-preload-count=16")
    r = _addrmsw("gfx1100").preloaded()
    assert r.build_flags(r.trace(**CONSTS)) == ()
