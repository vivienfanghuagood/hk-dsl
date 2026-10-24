"""SGLang-Diffusion (`sglang.multimodal_gen`): hk's attention behind its `torch_sdpa` backend.

    python3 -m hk.integration sglang-diffusion install     # once, into this Python's site-packages
    HK_SGLD=1 python3 my_generate.py                       # every process of the run, workers too
    HK_SGLD=attn python3 my_generate.py                    # attention only (or linear only)
    python3 -m hk.integration sglang-diffusion status
    python3 -m hk.integration sglang-diffusion uninstall

On RDNA3, `torch_sdpa` is the only attention backend SGLang-Diffusion has
(aiter is CDNA's, FlashAttention 3 Hopper's), and every DiT layer's attention
goes through `SDPAImpl.forward`, which calls
`torch.nn.functional.scaled_dot_product_attention` at call time. So the patch
is `hk.ops.sdpa.patch()`: the module attribute is replaced by hk's drop-in,
which takes what its kernel covers (bf16, head_dim 64/128, no mask, MHA or
GQA) and hands everything else -- masked text-prefix attention, odd head
dims -- to torch untouched. `hk.ops.sdpa.STATS` counts both branches.

The DiT's projections (`UnquantizedLinearMethod.apply` of
`sglang.multimodal_gen.runtime.layers.linear`) go to `hk.ops.matmul` where
they are bf16, without bias and at least `MIN_M` rows -- at Qwen-Image 2.1's
1024^2 shapes hk's gfx1100 GEMM measured 1.27-1.28x hipBLASLt -- and to the
engine's GEMM otherwise. `HK_SGLD=attn` or `HK_SGLD=linear` applies one of the
two.

**Why a .pth and an environment variable.** SGLang-Diffusion runs the model
in worker processes it spawns, and unlike SGLang's LLM engine it loads no
plugins there: a patch made in the process that built `DiffGenerator` never
reaches them. `install` writes two files into site-packages -- a one-line
`hk_sgld.pth` and the module it imports, `hk_sgld_boot` -- so every
interpreter of that environment runs the boot, which costs one environment
lookup unless `HK_SGLD` is set. With it set, the boot registers an import
hook on SGLang-Diffusion's SDPA backend module, and applies the patch the
moment a process (parent or worker) imports it. hk itself is imported then,
not at interpreter start. Nothing in SGLang's tree is modified.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import List

#: The module whose import applies the patch: every DiT attention of the
#: torch_sdpa backend is a call it makes.
TARGET = "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa"

ENV = "HK_SGLD"
_OFF = ("0", "", "off", "no")

_BOOT = '''"""Installed by `python -m hk.integration sglang-diffusion install` (hk-dsl).

With HK_SGLD set, applies hk.integration.sglang_diffusion when a process
imports SGLang-Diffusion's SDPA backend; otherwise does nothing.
"""
import os
import sys

if os.environ.get("HK_SGLD", "0") not in ("0", "", "off", "no"):
    import importlib.abc
    import importlib.util

    _TARGET = "%(target)s"

    class _HkSgld(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            run = spec.loader.exec_module

            def exec_module(module, _run=run):
                _run(module)
                try:
                    from hk.integration import sglang_diffusion
                    note = sglang_diffusion.apply()
                except BaseException as e:  # a failed patch is a slower run, not a broken one
                    note = "not applied (%%s: %%s)" %% (type(e).__name__, e)
                print("[hk] sglang-diffusion: %%s (pid %%d)" %% (note, os.getpid()), flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _HkSgld())
''' % {"target": TARGET}

_PTH = "import hk_sgld_boot\n"


#: The module whose UnquantizedLinearMethod every DiT projection goes through.
LINEAR_HOME = "sglang.multimodal_gen.runtime.layers.linear"

#: Below this many rows a projection keeps the engine's GEMM: hk's gfx1100 GEMM
#: was measured at 1.27-1.28x hipBLASLt on Qwen-Image 2.1's DiT projections at
#: M = 4096 (1024^2 images) and not at small M, where the text prefix and the
#: modulation projections run.
MIN_M = 1024

LINEAR_STATS = {"hk": 0, "engine": 0}


def _parts(value: str):
    v = value.strip().lower()
    if v in _OFF:
        return set()
    if v in ("1", "on", "yes", "all"):
        return {"attn", "linear"}
    return {p.strip() for p in v.split(",") if p.strip()}


def _hk_linear(x, weight, bias):
    """hk's GEMM for y = x @ weight.T, or None where the engine's should run."""
    import torch  # noqa: PLC0415

    from ..ops import gemm  # noqa: PLC0415

    if (bias is not None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_cuda or weight.dim() != 2 or not weight.is_contiguous()):
        return None
    k = x.shape[-1]
    n = weight.shape[0]
    m = x.numel() // k if k else 0
    if m < MIN_M or weight.shape[1] != k:
        return None
    a = x.reshape(m, k)
    if not a.is_contiguous():
        return None
    s = gemm.schedule_for(m, n, k)
    if k % s["k_step"] or n % s["block_n"]:
        return None
    return gemm.matmul(a, weight).view(*x.shape[:-1], n)


def apply_linear() -> str:
    """Route SGLang-Diffusion's unquantized projections (bf16, no bias, M >=
    MIN_M) through hk's GEMM; every other call keeps the engine's. Idempotent."""
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    cls = getattr(importlib.import_module(LINEAR_HOME), "UnquantizedLinearMethod")
    if getattr(cls.apply, "_hk", False):
        return "linear: already patched"
    orig = cls.apply

    @torch.compiler.disable
    def apply(self, layer, x, bias=None, *args, **kwargs):
        if not args and not kwargs:
            y = _hk_linear(x, layer.weight, bias)
            if y is not None:
                LINEAR_STATS["hk"] += 1
                return y
        LINEAR_STATS["engine"] += 1
        return orig(self, layer, x, bias, *args, **kwargs)

    apply._hk = True
    cls.apply = apply
    return f"linear -> hk.ops.matmul at M >= {MIN_M} ({LINEAR_HOME}.UnquantizedLinearMethod.apply)"


def apply() -> str:
    """Apply what HK_SGLD names: "1" both, or a comma list of "attn" and
    "linear". Idempotent."""
    import atexit  # noqa: PLC0415

    parts = _parts(os.environ.get(ENV, "1"))
    notes = []
    if "attn" in parts:
        from ..ops import sdpa  # noqa: PLC0415

        sdpa.patch()
        notes.append("F.scaled_dot_product_attention -> hk.ops.sdpa (torch_sdpa backend)")
    if "linear" in parts:
        notes.append(apply_linear())
    if not getattr(apply, "_done", False):
        # The worker that ran the DiT says, as it exits, which branch it took.
        atexit.register(lambda: print(f"[hk] sglang-diffusion: {stats()} (pid {os.getpid()})",
                                      flush=True))
        apply._done = True
    return "; ".join(notes) or f"nothing ({ENV} names neither attn nor linear)"


def stats() -> str:
    """Which branch each patch took, in this process."""
    from ..ops import sdpa  # noqa: PLC0415

    s = sdpa.STATS
    fb = "; ".join(f"{shape} {why}" for (shape, _), why in list(sdpa.FALLBACKS.items())[:4])
    return (f"attention: hk {s['kernel']} ({s.get('bnhd', 0)} in place), torch {s['fallback']}"
            + (f" ({fb})" if fb else "")
            + f"; linear: hk {LINEAR_STATS['hk']}, engine {LINEAR_STATS['engine']}")


def site_dir() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def install(target: Path = None) -> List[Path]:
    """Write hk_sgld.pth and hk_sgld_boot.py into `target` (default: this
    interpreter's site-packages). Returns the paths written."""
    d = Path(target) if target else site_dir()
    boot, pth = d / "hk_sgld_boot.py", d / "hk_sgld.pth"
    boot.write_text(_BOOT)
    pth.write_text(_PTH)
    return [pth, boot]


def uninstall(target: Path = None) -> List[Path]:
    d = Path(target) if target else site_dir()
    gone = []
    for p in (d / "hk_sgld.pth", d / "hk_sgld_boot.py"):
        if p.exists():
            p.unlink()
            gone.append(p)
    return gone


def status() -> str:
    d = site_dir()
    installed = (d / "hk_sgld.pth").exists() and (d / "hk_sgld_boot.py").exists()
    on = os.environ.get(ENV, "0") not in _OFF
    return (f"sglang-diffusion: {'installed' if installed else 'not installed'} in {d}; "
            f"{ENV}={'on' if on else 'off'}")


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python3 -m hk.integration sglang-diffusion",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("install", "uninstall", "status"))
    ap.add_argument("--target", default=None, help="site-packages directory (default: this Python's)")
    a = ap.parse_args(argv)
    if a.action == "install":
        for p in install(a.target):
            print(f"wrote {p}")
        print(f"set {ENV}=1 in the environment of the run to apply it")
    elif a.action == "uninstall":
        gone = uninstall(a.target)
        print("\n".join(f"removed {p}" for p in gone) or "nothing installed")
    else:
        print(status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
