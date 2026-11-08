"""Every kernel docs/language.md defines traces and translates for the tier's arch (the hk.comm one on
gfx950 only, as the page says), so an API change cannot leave the page's examples behind."""

from __future__ import annotations

import sys
from pathlib import Path

import hk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _doc_blocks  # noqa: E402

GFX950_ONLY = {"drain"}   # hk.comm through a device-scoped tensor


def test_the_page_defines_its_kernels(tmp_path):
    mod = _doc_blocks.load(tmp_path)
    names = {n for n, v in vars(mod).items() if isinstance(v, hk.Kernel)}
    assert {"row_sumsq", "scale_pow2", "exp_ldexp", "drain", "gated_silu"} <= names, names
    assert mod.silu_mul(dtype=hk.fp16).name == "silu_mul__dtypefp16_warps4"   # as the page says
    assert mod.gated_silu.variant(T=hk.bf16).name == "gated_silu__Tbf16_warps4"


def test_each_kernel_translates(tmp_path):
    mod = _doc_blocks.load(tmp_path)
    arch = hk.target.current_arch()
    kernels = {n: v for n, v in vars(mod).items() if isinstance(v, hk.Kernel)}
    kernels["silu_mul"] = mod.silu_mul()
    for name, k in kernels.items():
        if name in GFX950_ONLY and arch != "gfx950":
            continue
        src = k.source(T=hk.bf16) if isinstance(k, hk.GenericKernel) else k.source()
        assert "__global__" in src, name
