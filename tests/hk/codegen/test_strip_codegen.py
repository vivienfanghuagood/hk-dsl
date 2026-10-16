"""hk.ops.norm's strip kernels through hipcc: they compile without spilling,
and the ISA is the one the layout exists for -- 16-byte accesses only, the
row reduction a DPP butterfly and not ds_bpermute, every kernel argument
read at entry.

Needs hipcc, not a GPU.
"""

from __future__ import annotations

import re

import pytest

from hk.ops import norm
from hk.target import current_arch, get_target

pytestmark = pytest.mark.skipif(
    get_target(current_arch()).kittens_define != "KITTENS_RDNA3",
    reason="strips are include/rdna3's")

#: (kernel, CHUNKS, RUN, EXACT): each kind at the 4096-column bf16 plan, the
#: bounded form at a ragged width, and an fp32 one.
CASES = [(f"strip_{k}_bf16_w8", 2, 8, 1) for k in ("rms", "layer", "softmax", "rmsw", "addrmsw")]
CASES += [(f"strip_{k}_bf16_w9", 4, 4, 0) for k in ("softmax", "addrmsw")]
CASES += [("strip_layer_fp32_w8", 4, 4, 1), ("strip_rmsw_fp16_w16", 1, 8, 1)]


@pytest.mark.hipcc
@pytest.mark.parametrize("name,chunks,run,exact", CASES)
def test_a_strip_kernel_compiles_without_spilling(name, chunks, run, exact):
    b = norm.STRIP_KERNELS[name].build(CHUNKS=chunks, RUN=run, EXACT=exact)
    assert b.kernels
    for k in b.kernels:
        assert k.scratch == 0 and not k.spills, f"{name}: {k.line()}"
        assert (k.occ_by_reg or k.occupancy) >= 8, f"{name}: {k.line()}"


def _isa(kernel, **consts) -> str:
    import subprocess
    import tempfile
    from pathlib import Path

    from hk.runtime.compile import flags, hipcc
    from hk.target import gfx1100

    with tempfile.TemporaryDirectory() as d:
        cpp, asm = Path(d) / "k.cpp", Path(d) / "k.s"
        cpp.write_text(kernel.source(scaffold="bare", **consts))
        cmd = [hipcc(), *[f for f in flags(gfx1100.GFX1100, "probe")
                          if f not in ("-shared", "-fPIC",
                                       "-Rpass-analysis=kernel-resource-usage")],
               "--offload-device-only", "-S", str(cpp), "-o", str(asm)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return asm.read_text()


@pytest.mark.hipcc
def test_the_weighted_strip_moves_each_run_once_in_16_bytes():
    """4096 bf16 columns over 8 warps: two runs a lane of the row, two of the
    weight, two stored -- where the tile kernel issued 16 loads and 8 stores
    a warp and a weight broadcast by element."""
    text = _isa(norm.STRIP_KERNELS["strip_rmsw_bf16_w8"], CHUNKS=2, RUN=8, EXACT=1)
    loads = re.findall(r"\bglobal_load_(\w+)", text)
    stores = re.findall(r"\bglobal_store_(\w+)", text)
    assert loads == ["b128"] * 4, loads
    assert stores == ["b128"] * 2, stores
    assert "row_xmask" in text and "v_permlanex16_b32" in text
    assert "ds_bpermute" not in text


@pytest.mark.hipcc
def test_the_arguments_are_read_in_one_clause():
    """Three tensors are 144 bytes of arguments, three 64-byte lines; all
    three are loaded before the first wait (kittens::touch_kernargs)."""
    text = _isa(norm.STRIP_KERNELS["strip_rmsw_bf16_w8"], CHUNKS=2, RUN=8, EXACT=1)
    head = text[text.index("_kernel7globals:"):]
    head = head[:head.index("s_waitcnt lgkmcnt(0)")]
    offsets = {int(m, 16) // 64 for m in re.findall(r"s_load_b\d+ s\[?[\d:]+\]?, s\[0:1\], 0x([0-9a-f]+)", head)}
    assert offsets >= {0, 1, 2}, head
