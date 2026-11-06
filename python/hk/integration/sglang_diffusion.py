"""SGLang-Diffusion (`sglang.multimodal_gen`): hk's attention behind its `torch_sdpa` backend.

    python3 -m hk.integration sglang-diffusion install     # once, into this Python's site-packages
    python3 my_generate.py                                 # hk on, in every process, workers too
    HK_DISABLE=1 python3 my_generate.py                    # stock: nothing patched
    HK_SGLD_LINEAR=0 python3 my_generate.py                # attention only (HK_SGLD_ATTN=0: linear only)
    HK_SGLD=attn,linear,convrot python3 my_generate.py     # exactly these (convrot is opt-in)
    python3 -m hk.integration sglang-diffusion status
    python3 -m hk.integration sglang-diffusion uninstall

**On by default** once installed, on the GPU it was validated on (gfx1100): attention,
linear, Wan's VAE conv3d and Qwen-Image 2.1's RoPE (`hk.integration.switches`). ConvRot INT8 is lossy and
stays opt-in. An install made before hk was on by default keeps its old boot, which needs
HK_SGLD=1; run `install` again.

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

Wan's VAE conv3d (`WanCausalConv3d`, sgld_vaeconv) runs as calls of 2 output
frames (`hk.ops.conv3d.frames`), each the engine's own conv: MIOpen (gfx1100)
takes the decoder's full-resolution 6-frame convs to its naive reference
kernel, 100x slower. Qwen-Image 2.1's complex RoPE (`apply_rope`, sgld_rope) runs
on `hk.ops.rotary_pairs`: one launch for the engine's four, bit for bit the same.

**Why a .pth and an environment variable.** SGLang-Diffusion runs the model
in worker processes it spawns, and unlike SGLang's LLM engine it loads no
plugins there: a patch made in the process that built `DiffGenerator` never
reaches them. `install` writes two files into site-packages -- a one-line
`hk_sgld.pth` and the module it imports, `hk_sgld_boot` -- so every
interpreter of that environment runs the boot, which costs two environment
lookups. Unless HK_DISABLE=1 or HK_SGLD=0, the boot registers an import
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
_LINEAR_ORIG = []

_BOOT = '''"""Installed by `python -m hk.integration sglang-diffusion install` (hk-dsl).

Unless HK_DISABLE=1 or HK_SGLD=0, applies hk.integration.sglang_diffusion when a
process imports SGLang-Diffusion's SDPA backend (hk is on by default).
"""
import os
import sys

_OFF = ("0", "off", "no", "false")
_D = os.environ.get("HK_DISABLE", "").strip().lower()
_S = os.environ.get("HK_SGLD", "").strip().lower()
if not (_D and _D not in _OFF) and _S not in _OFF:
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

#: Projections with a bias (Wan's) take hk's GEMM with its bias epilogue too; False leaves
#: every biased projection to the engine.
LINEAR_BIAS = True


def _hk_linear(x, weight, bias):
    """hk's GEMM for y = x @ weight.T (+ bias), or None where the engine's should run. A bias
    is added to the fp32 accumulator in the GEMM's store (hk.templates.gemm's bias epilogue),
    as hipBLASLt's bias epilogue does: one rounding to bf16."""
    import torch  # noqa: PLC0415

    from ..ops import gemm  # noqa: PLC0415

    if (x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_cuda or weight.dim() != 2 or not weight.is_contiguous()):
        return None
    if bias is not None and (not LINEAR_BIAS or bias.dtype != torch.bfloat16 or bias.dim() != 1
                             or bias.shape[0] != weight.shape[0]):
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
    op = _gemm_op(s) if bias is None else _gemm_bias_op(s, m, n)
    key = (_bucket(m), n, k, bias is not None)
    if key not in DECISIONS:
        DECISIONS[key] = _measure(op, a, weight, bias)
        if os.environ.get("HK_SGLD_LOG", "0") not in _OFF:
            print(f"[hk] sglang-diffusion linear {m}x{n}x{k}{' +bias' if bias is not None else ''}: "
                  f"{DECISIONS[key][1]}", flush=True)
    if not DECISIONS[key][0]:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if bias is None:
        op(a, weight, out)
    else:
        op(a, weight, out, bias.contiguous())
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


def _measure(op, a, weight, bias=None):
    """Time hk's GEMM against F.linear on this call's own operands, interleaved,
    best of 5 each after a warm-up, on the current stream. hk's gfx1100 GEMM
    won at Qwen-Image's DiT shapes (1.27x), and with every M >= 1024 projection
    patched MiniMax-H3's decode stage was 35% slower: a shape decides for itself."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415,N812

    out = torch.empty((a.shape[0], weight.shape[0]), device=a.device, dtype=torch.bfloat16)
    if bias is None:
        arms = {"hk": lambda: op(a, weight, out), "engine": lambda: F.linear(a, weight)}
    else:
        b = bias.contiguous()
        arms = {"hk": lambda: op(a, weight, out, b), "engine": lambda: F.linear(a, weight, bias)}
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
    from ..runtime.torch_ext import launcher  # noqa: PLC0415

    kern = gemm.TUNER._kernel_for(sched)
    op = _OPS.get(kern.name)
    if op is None:
        op = _OPS[kern.name] = launcher(kern, ns="hk_sgld")
    return op


def _gemm_bias_op(sched, m, n):
    """_gemm_op's GEMM with hk.templates.gemm's bias epilogue (acc + bias, bias (N,) bf16),
    on torch's current stream: launched as op(a, weight, out, bias)."""
    import torch  # noqa: PLC0415

    from ..runtime.torch_ext import launcher  # noqa: PLC0415
    # The module's names, not the package's `gemm` (the template object of that name).
    from ..templates.gemm import Step, bias, bind, kernel_for  # noqa: PLC0415
    from ..types import bf16  # noqa: PLC0415

    ep = bind(bias, {"bias": torch.empty(n, dtype=torch.bfloat16, device="meta")}, m, n, bf16)
    kern = kernel_for(Step("rdna", dict(sched), (0, n), True), ep)
    op = _OPS.get(kern.name)
    if op is None:
        op = _OPS[kern.name] = launcher(kern, ns="hk_sgld")
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
    _LINEAR_ORIG[:] = [(cls, orig)]
    return f"linear -> hk.ops.matmul at M >= {MIN_M} ({LINEAR_HOME}.UnquantizedLinearMethod.apply)"


#: The module of Wan's VAE, whose WanCausalConv3d every decoder conv3d goes through.
VAE_HOME = "sglang.multimodal_gen.runtime.models.vaes.wanvae"

VAE_STATS = {"chunked": 0, "whole": 0}
_VAE_ORIG = []


def apply_vaeconv() -> str:
    """Run Wan's VAE conv3d (WanCausalConv3d) through hk.ops.conv3d.frames: calls of
    conv3d.FRAMES output frames along time, each the engine's own conv. On gfx1100 MIOpen runs
    the decoder's full-resolution 6-frame convs on its naive reference kernel otherwise.
    Idempotent."""
    import importlib  # noqa: PLC0415

    from ..ops import conv3d  # noqa: PLC0415

    cls = getattr(importlib.import_module(VAE_HOME), "WanCausalConv3d")
    if getattr(cls._conv_forward, "_hk", False):
        return "vaeconv: already patched"
    orig = cls._conv_forward

    def _conv_forward(self, input, weight, bias):  # noqa: A002 -- nn.Conv3d's names
        # WanCausalConv3d pads time itself and sets the conv's padding to 0: only then is a
        # slice of the input the conv of a slice of the output.
        pad = self.padding[0] if isinstance(self.padding, tuple) else -1
        calls = []

        def conv(x, w, b):
            calls.append(1)
            return orig(self, x, w, b)

        y = conv3d.frames(conv, input, weight, bias, time_stride=self.stride[0],
                          time_dilation=self.dilation[0], time_padding=pad)
        VAE_STATS["chunked" if len(calls) > 1 else "whole"] += 1
        return y

    _conv_forward._hk = True
    cls._conv_forward = _conv_forward
    _VAE_ORIG[:] = [(cls, orig)]
    return (f"vaeconv: {VAE_HOME}.WanCausalConv3d in calls of {conv3d.FRAMES} output frames "
            f"(hk.ops.conv3d.frames)")


def revert_vaeconv() -> str:
    if not _VAE_ORIG:
        return "vaeconv: not patched"
    cls, orig = _VAE_ORIG.pop()
    cls._conv_forward = orig
    return "vaeconv -> engine"


#: Qwen-Image 2.1's DiT module, whose `apply_rope` turns every image and text q and k.
QWEN21_HOME = "sglang.multimodal_gen.runtime.models.dits.qwen_image21"

ROPE_STATS = {"hk": 0, "engine": 0}
_ROPE_ORIG = []
#: The last rope's cos and sin, (rope, its version, cos, sin): one rope serves every block's
#: q and k of a forward. The rope is held, so `is` cannot match a new tensor at its address.
_ROPE_TABLES = [None]


def _version(t):
    # an inference-mode tensor (SGLang-Diffusion's forward) has no version counter
    return None if t.is_inference() else t._version


def _rope_tables(rope):
    got = _ROPE_TABLES[0]
    if got is not None and got[0] is rope and got[1] == _version(rope):
        return got[2], got[3]
    cos, sin = rope.real.contiguous(), rope.imag.contiguous()
    _ROPE_TABLES[0] = (rope, _version(rope), cos, sin)
    return cos, sin


def apply_rope() -> str:
    """Qwen-Image 2.1's complex RoPE (`apply_rope` of QWEN21_HOME:
    `view_as_complex(x.float()) * rope`, back to bf16) on hk.ops.rotary_pairs, one launch for
    the engine's four (cast, complex multiply, cast, copy), bit for bit the same. bf16 q / k
    of one sequence (B = 1) with a contiguous complex64 (S, D/2) rope; anything else keeps
    the engine's. Idempotent."""
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from ..ops import fused  # noqa: PLC0415

    mod = importlib.import_module(QWEN21_HOME)
    if getattr(mod.apply_rope, "_hk", False):
        return "rope: already patched"
    orig = mod.apply_rope

    def apply_rope(x, rope):
        if (x.dtype == torch.bfloat16 and x.dim() == 4 and x.shape[0] == 1 and x.is_contiguous()
                and x.shape[-1] == 128 and rope.dtype == torch.complex64 and rope.is_contiguous()
                and tuple(rope.shape) == (x.shape[1], 64) and not torch.compiler.is_compiling()):
            cos, sin = _rope_tables(rope)
            ROPE_STATS["hk"] += 1
            return fused.rotary_pairs(x, cos, sin)
        ROPE_STATS["engine"] += 1
        return orig(x, rope)

    apply_rope._hk = True
    mod.apply_rope = apply_rope
    _ROPE_ORIG[:] = [(mod, orig)]
    return f"rope: {QWEN21_HOME}.apply_rope -> hk.ops.rotary_pairs (bf16, head_dim 128)"


QKNORM_STATS = {"hk": 0, "engine": 0}
_QKNORM_ORIG = []


def _qknorm_fits(x, norm, rope) -> bool:
    """SGLang's RMSNorm as rmsnorm_rotary_pairs computes it: cast before the weight, no
    residual, no variance override, a bf16 (128,) weight, the activation's dtype out."""
    import torch  # noqa: PLC0415

    w = getattr(norm, "weight", None)
    return (getattr(norm, "cast_x_before_out_mul", False)
            and getattr(norm, "variance_size_override", 0) is None
            and getattr(norm, "override_orig_dtype", 0) is None
            and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16
            and tuple(w.shape) == (128,) and w.is_contiguous()
            and x.dtype == torch.bfloat16 and x.dim() == 4 and x.shape[0] == 1
            and x.is_contiguous() and x.shape[-1] == 128 and rope.dtype == torch.complex64
            and rope.is_contiguous() and tuple(rope.shape) == (x.shape[1], 64)
            and not torch.compiler.is_compiling())


def apply_qknorm() -> str:
    """Qwen-Image 2.1's q / k RMSNorm and complex RoPE (`apply_qk_norm_rope` of QWEN21_HOME:
    SGLang's RMSNorm, native on ROCm -- five launches -- then apply_rope) as one
    hk.ops.fused.rmsnorm_rotary_pairs launch. Where the norm or the shapes are not the ones it
    computes, the engine's. Idempotent."""
    import importlib  # noqa: PLC0415

    from ..ops import fused  # noqa: PLC0415

    mod = importlib.import_module(QWEN21_HOME)
    if getattr(mod.apply_qk_norm_rope, "_hk", False):
        return "qknorm: already patched"
    orig = mod.apply_qk_norm_rope

    def apply_qk_norm_rope(x, norm, rope):
        if _qknorm_fits(x, norm, rope):
            cos, sin = _rope_tables(rope)
            QKNORM_STATS["hk"] += 1
            return fused.rmsnorm_rotary_pairs(x, norm.weight, norm.variance_epsilon, cos, sin)
        QKNORM_STATS["engine"] += 1
        return orig(x, norm, rope)

    apply_qk_norm_rope._hk = True
    mod.apply_qk_norm_rope = apply_qk_norm_rope
    _QKNORM_ORIG[:] = [(mod, orig)]
    return (f"qknorm: {QWEN21_HOME}.apply_qk_norm_rope -> hk.ops.fused.rmsnorm_rotary_pairs "
            f"(bf16, head_dim 128)")


def revert_qknorm() -> str:
    if not _QKNORM_ORIG:
        return "qknorm: not patched"
    mod, orig = _QKNORM_ORIG.pop()
    mod.apply_qk_norm_rope = orig
    return "qknorm -> engine"


def revert_rope() -> str:
    if not _ROPE_ORIG:
        return "rope: not patched"
    mod, orig = _ROPE_ORIG.pop()
    mod.apply_rope = orig
    _ROPE_TABLES[0] = None
    return "rope -> engine"


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


def revert_linear() -> str:
    if not _LINEAR_ORIG:
        return "linear: not patched"
    cls, orig = _LINEAR_ORIG.pop()
    cls.apply = orig
    return "linear -> engine"


def apply() -> str:
    """Apply the SGLang-Diffusion routes hk.integration.switches has on: attention and linear
    Wan's VAE conv3d and Qwen-Image 2.1's RoPE by default on gfx1100; convrot only when asked for
    (HK_SGLD=...,convrot or HK_SGLD_CONVROT=1).
    Checks each route's engine contract first; a broken or raising route stays off with its own
    warning line and the others are applied. Idempotent."""
    import atexit  # noqa: PLC0415

    from . import contract, switches  # noqa: PLC0415

    if switches.disabled():
        return f"off ({switches.DISABLE}=1), stock"
    routes = switches.on("sglang_diffusion")
    if not routes:
        return f"no route on ({switches.decide('sgld_attn').why}), stock"
    def attn() -> str:
        from ..ops import sdpa  # noqa: PLC0415

        sdpa.patch()
        return "F.scaled_dot_product_attention -> hk.ops.sdpa (torch_sdpa backend)"

    def convrot() -> str:
        try:
            return apply_convrot()
        except ImportError as e:   # an SGLang without convrot_int8
            return f"convrot: not available ({e})"

    # Each route's contract, then each route that holds, one at a time: a broken or raising
    # route is off with its own warning line, and the others are applied (contract.py).
    notes, applied, off = contract.apply_routes(
        "sglang_diffusion", routes,
        [("sgld_attn", attn), ("sgld_linear", apply_linear), ("sgld_convrot", convrot),
         ("sgld_vaeconv", apply_vaeconv), ("sgld_rope", apply_rope),
         ("sgld_qknorm", apply_qknorm)])
    if not applied:
        return f"no route applied ({', '.join(off)} off, see the warnings): stock"
    from ..runtime.compile import arm_jit_notice  # noqa: PLC0415

    arm_jit_notice("SGLang-Diffusion patch")
    if not getattr(apply, "_done", False):
        # Each process says which branch it took: as it exits, and -- because the worker that
        # runs the DiT is torn down without running atexit (its counters were never seen) --
        # from a daemon thread whenever they changed, at most every HK_SGLD_REPORT_S seconds.
        atexit.register(_report, True)
        threading.Thread(target=_reporter, name="hk-sgld-report", daemon=True).start()
        apply._done = True
    on = ", ".join(f"{n} ({d.why})" for n, d in routes.items() if n in applied)
    left = f"; OFF: {', '.join(off)}" if off else ""
    return f"routes on: {on}{left}; " + "; ".join(notes)


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
            + f"; convrot: hk {CONVROT_STATS['hk']}, fallback {CONVROT_STATS['fallback']}"
            + f"; vaeconv: chunked {VAE_STATS['chunked']}, whole {VAE_STATS['whole']}"
            + f"; rope: hk {ROPE_STATS['hk']}, engine {ROPE_STATS['engine']}"
            + f"; qknorm: hk {QKNORM_STATS['hk']}, engine {QKNORM_STATS['engine']}")


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
    stale = installed and "HK_DISABLE" not in (d / "hk_sgld_boot.py").read_text()
    from . import switches  # noqa: PLC0415

    on = {r.name: switches.decide(r.name) for r in switches.routes("sglang_diffusion")}
    return (f"sglang-diffusion: {'installed' if installed else 'not installed'} in {d}"
            + ("; the installed boot is from before hk was on by default (it needs HK_SGLD=1): "
               "run install again" if stale else "") + "; "
            + ", ".join(f"{n} {'on' if x.on else 'off'} ({x.why})" for n, x in on.items()))


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
        print("hk is on by default in every process of this environment; "
              "HK_DISABLE=1 (or HK_SGLD=0) turns it off for a run")
    elif a.action == "uninstall":
        gone = uninstall(a.target)
        print("\n".join(f"removed {p}" for p in gone) or "nothing installed")
    else:
        print(status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
