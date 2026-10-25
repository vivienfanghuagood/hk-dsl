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
import threading
import time
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
#: modulation projections run. Above it each shape is timed once (`_measure`).
MIN_M = 1024

LINEAR_STATS = {"hk": 0, "engine": 0}


def _parts(value: str):
    v = value.strip().lower()
    if v in _OFF:
        return set()
    if v in ("1", "on", "yes", "all"):
        # convrot is opt-in: HK_SGLD=attn,linear,convrot (it acts only on --quantization convrot_int8)
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
    op = _gemm_op(s)
    key = (_bucket(m), n, k)
    if key not in DECISIONS:
        DECISIONS[key] = _measure(op, a, weight)
        if os.environ.get("HK_SGLD_LOG", "0") not in _OFF:
            print(f"[hk] sglang-diffusion linear {m}x{n}x{k}: {DECISIONS[key][1]}", flush=True)
    if not DECISIONS[key][0]:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    op(a, weight, out)
    return out.view(*x.shape[:-1], n)


#: (M bucket, N, K) -> (take hk's GEMM?, why), decided the first time a shape
#: is seen in this process.
DECISIONS = {}

#: hk must beat the engine by this much at a shape to take it: the decision is
#: a few timed calls, and a near-tie is not worth a different kernel.
MARGIN = 1.03


def _bucket(m: int) -> int:
    b = 1
    while b < m:
        b *= 2
    return b


def _measure(op, a, weight):
    """Time hk's GEMM against F.linear on this call's own operands, interleaved,
    best of 5 each after a warm-up, on the current stream. hk's gfx1100 GEMM
    won at Qwen-Image's DiT shapes (1.27x), and with every M >= 1024 projection
    patched MiniMax-H3's decode stage was 35% slower: a shape decides for itself."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415,N812

    out = torch.empty((a.shape[0], weight.shape[0]), device=a.device, dtype=torch.bfloat16)
    arms = {"hk": lambda: op(a, weight, out), "engine": lambda: F.linear(a, weight)}
    for f in arms.values():
        f()
    best = {name: float("inf") for name in arms}
    for _ in range(5):
        for name, f in arms.items():
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            f()
            e1.record()
            e1.synchronize()
            best[name] = min(best[name], e0.elapsed_time(e1))
    ratio = best["engine"] / best["hk"]
    take = ratio >= MARGIN
    return take, (f"{'hk' if take else 'engine'} (engine {best['engine'] * 1e3:.0f} us, "
                  f"hk {best['hk'] * 1e3:.0f} us, {ratio:.2f}x)")


_OPS = {}


def _gemm_op(sched):
    """hk's gfx1100 GEMM for `sched` as a torch op: it launches on torch's
    current stream. hk.ops.matmul's pybind launch goes to the null stream,
    which waits for every other stream -- under SGLang-Diffusion's layer-wise
    offload that serialised the weight prefetch behind each GEMM and made
    MiniMax-H3's denoise loop slower with the GEMM patch than without it."""
    from ..ops import gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    kern = gemm.TUNER._kernel_for(sched)
    op = _OPS.get(kern.name)
    if op is None:
        op = _OPS[kern.name] = torch_op(kern, ns="hk_sgld")
    return op


def apply_linear() -> str:
    """Route SGLang-Diffusion's unquantized projections (bf16, no bias, M >=
    MIN_M) through hk's GEMM; every other call keeps the engine's. Idempotent."""
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    cls = getattr(importlib.import_module(LINEAR_HOME), "UnquantizedLinearMethod")
    if getattr(cls.apply, "_hk", False):
        return "linear: already patched"
    orig = cls.apply

    def apply(self, layer, x, bias=None, *args, **kwargs):
        # Inside a torch.compile region the engine's GEMM stays: a
        # compiler-disabled wrapper would break the graph at every projection.
        if not args and not kwargs and not torch.compiler.is_compiling():
            y = _hk_linear(x, layer.weight, bias)
            if y is not None:
                LINEAR_STATS["hk"] += 1
                return y
        LINEAR_STATS["engine"] += 1
        return orig(self, layer, x, bias, *args, **kwargs)

    apply._hk = True
    cls.apply = apply
    return f"linear -> hk.ops.matmul at M >= {MIN_M} ({LINEAR_HOME}.UnquantizedLinearMethod.apply)"


#: SGLang-Diffusion's ConvRot INT8 config: both of its backends are CUDA-only.
CONVROT_HOME = "sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config"
CONVROT_BACKEND = "hk"
CONVROT_GROUPS = (16, 64, 256)
CONVROT_STATS = {"hk": 0, "fallback": 0}


def apply_convrot() -> str:
    """A third convrot_int8 backend, "hk" (SGLANG_DIFFUSION_CONVROT_INT8_BACKEND=hk,
    and what "auto" picks on ROCm): hk.ops.convrot's W8A8 linear, for online
    (a bf16 checkpoint quantized at load) and serialized ConvRot checkpoints.
    Only takes effect when the run asks for --quantization convrot_int8."""
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    if not torch.version.hip:
        return "convrot: not ROCm, left alone"
    mod = importlib.import_module(CONVROT_HOME)
    if getattr(mod, "_hk", False):
        return "convrot: already patched"
    linear_mod = importlib.import_module(LINEAR_HOME)
    LinearMethodBase = linear_mod.LinearMethodBase
    Unquantized = linear_mod.UnquantizedLinearMethod
    from torch.nn.parameter import Parameter  # noqa: PLC0415

    from ..ops import convrot  # noqa: PLC0415

    class HkConvRotInt8LinearMethod(LinearMethodBase):
        """convrot_int8 on hk: weights rotated and quantized once (or loaded so),
        activations rotated, quantized and multiplied by hk's int8 GEMM."""

        def __init__(self, quant_config, *, group_size, is_checkpoint_serialized):
            self.quant_config = quant_config
            self.group_size = group_size
            self.is_checkpoint_serialized = is_checkpoint_serialized

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            if input_size_per_partition % self.group_size:
                raise ValueError(f"convrot_int8 (hk): input size {input_size_per_partition} "
                                 f"is not a multiple of the group {self.group_size}")
            Unquantized.create_weights(
                self, layer, input_size_per_partition, output_partition_sizes, input_size,
                output_size, torch.int8 if self.is_checkpoint_serialized else params_dtype,
                **extra_weight_attrs)
            if self.is_checkpoint_serialized:
                from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs  # noqa: PLC0415

                scale = Parameter(torch.empty(sum(output_partition_sizes), 1, dtype=torch.float32),
                                  requires_grad=False)
                set_weight_attrs(scale, {"output_dim": 0})
                set_weight_attrs(scale, extra_weight_attrs)
                layer.register_parameter("weight_scale", scale)

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            if self.is_checkpoint_serialized:
                self.quant_config.note_loaded()
                return
            if w.dtype == torch.int8:
                return
            home = w.device
            q, sc = convrot.prepare_weight(w.to("cuda", torch.bfloat16), self.group_size)
            layer.weight = Parameter(q.to(home), requires_grad=False)
            layer.register_parameter("weight_scale", Parameter(sc.view(-1, 1).to(home),
                                                               requires_grad=False))
            self.quant_config.note_quantized(w.numel() * w.element_size())

        def apply(self, layer, x, bias=None):
            dt = x.dtype
            xb = x if dt == torch.bfloat16 else x.to(torch.bfloat16)
            w, ws = layer.weight, layer.weight_scale.view(-1)
            if w.shape[0] % 128 == 0 and w.shape[1] % 64 == 0:
                CONVROT_STATS["hk"] += 1
                y = convrot.linear(xb, w, ws, self.group_size, bias=bias)
            else:
                CONVROT_STATS["fallback"] += 1
                y = convrot.reference(xb, w, ws, self.group_size, bias=bias)
            return y if y.dtype == dt else y.to(dt)

    mod.BACKENDS = tuple(mod.BACKENDS) + (CONVROT_BACKEND,)
    mod.BACKEND_GROUP_SIZES[CONVROT_BACKEND] = CONVROT_GROUPS
    cls = mod.ConvRotInt8Config
    resolve0, select0 = cls.resolve_backend, cls._select

    def resolve_backend(self, group_size=None):
        g = self.group_size if group_size is None else group_size
        if self.backend in ("auto", CONVROT_BACKEND):
            return CONVROT_BACKEND if g in CONVROT_GROUPS else None
        return resolve0(self, group_size)

    def _select(self, *, prefix, backend, group_size, serialized):
        if backend != CONVROT_BACKEND:
            return select0(self, prefix=prefix, backend=backend, group_size=group_size,
                           serialized=serialized)
        self.selected.append(prefix)
        by = self.selected_by_backend
        (by.setdefault(backend, []) if isinstance(by, dict) else by[backend]).append(prefix)
        return HkConvRotInt8LinearMethod(self, group_size=group_size,
                                         is_checkpoint_serialized=serialized)

    cls.resolve_backend, cls._select = resolve_backend, _select
    mod._hk = True
    return f"convrot_int8 -> backend 'hk' (hk.ops.convrot; group {CONVROT_GROUPS})"


def apply() -> str:
    """Apply what HK_SGLD names: "1" all, or a comma list of "attn", "linear"
    and "convrot". Idempotent."""
    import atexit  # noqa: PLC0415

    parts = _parts(os.environ.get(ENV, "1"))
    notes = []
    if "attn" in parts:
        from ..ops import sdpa  # noqa: PLC0415

        sdpa.patch()
        notes.append("F.scaled_dot_product_attention -> hk.ops.sdpa (torch_sdpa backend)")
    if "linear" in parts:
        notes.append(apply_linear())
    if "convrot" in parts:
        try:
            notes.append(apply_convrot())
        except ImportError as e:   # an SGLang without convrot_int8
            notes.append(f"convrot: not available ({e})")
    if not getattr(apply, "_done", False):
        # Each process says which branch it took: as it exits, and -- because the worker that
        # runs the DiT is torn down without running atexit (its counters were never seen) --
        # from a daemon thread whenever they changed, at most every HK_SGLD_REPORT_S seconds.
        atexit.register(_report, True)
        threading.Thread(target=_reporter, name="hk-sgld-report", daemon=True).start()
        apply._done = True
    return "; ".join(notes) or f"nothing ({ENV} names neither attn nor linear)"


_LAST = [None]


def _report(force: bool = False) -> None:
    s = stats()
    if force or s != _LAST[0]:
        _LAST[0] = s
        print(f"[hk] sglang-diffusion: {s} (pid {os.getpid()})", flush=True)


def _reporter() -> None:
    period = float(os.environ.get("HK_SGLD_REPORT_S", "60"))
    while True:
        time.sleep(period)
        try:
            _report()
        except Exception:  # noqa: BLE001 -- a report must never take the worker down
            pass


def stats() -> str:
    """Which branch each patch took, in this process."""
    from ..ops import sdpa  # noqa: PLC0415

    s = sdpa.STATS
    fb = "; ".join(f"{shape} {why}" for (shape, _), why in list(sdpa.FALLBACKS.items())[:4])
    return (f"attention: hk {s['kernel']} ({s.get('bnhd', 0)} in place, {s.get('k3', 0)} on attn_k3), "
            f"torch {s['fallback']}"
            + (f" ({fb})" if fb else "")
            + (f"; attn_k3 shapes {dict(list(sdpa.K3_SHAPES.items())[:4])}" if sdpa.K3_SHAPES else "")
            + f"; linear: hk {LINEAR_STATS['hk']}, engine {LINEAR_STATS['engine']}"
            + f"; convrot: hk {CONVROT_STATS['hk']}, fallback {CONVROT_STATS['fallback']}")


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
