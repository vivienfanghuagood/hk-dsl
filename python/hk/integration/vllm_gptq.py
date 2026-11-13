"""vLLM's GPTQ linear layers on hk's 4-bit decode GEMM (`HK_VLLM_GPTQ`, on by default on gfx1100
and gfx1201).

On RDNA3, vLLM runs a 4-bit symmetric GPTQ checkpoint through
`AutoGPTQLinearMethod` with `RDNA3W4A16LinearKernel`, its own HIP W4A16 GEMM
(`ops.gptq_gemm_rdna3`) on GPTQ's shuffled codes. The patch wraps the method's
two entry points:

  * `process_weights_after_loading` writes hk's symmetric layout of the codes
    (`hk.ops.w4a16.repack_gptq(..., symmetric=True)`: codes and scales in
    16-row blocks, no zero-times-scale) *before* vLLM's runs, since vLLM's
    shuffles the codes in place. It is a second copy of the codes, the size of
    the first, kept so that every batch hk does not take still runs vLLM's
    kernel on vLLM's own tensors. The zero point is 8, as
    `RDNA3W4A16LinearKernel` takes it for uint4b8; a layer whose stored zero
    points are not all 8 is refused (counted), not assumed.
  * `apply` makes one call of the opaque custom op `hk_vllm::gptq_w4a16`.
    Inside it the token count is a plain number, not a symbolic size. 16-bit
    batches of `MIN_M` to 64 tokens go to hk's kernel, and the rest to vLLM's.

On gfx1201 vLLM gives the layers `RDNAHybridW4A16LinearKernel` instead: its HIP skinny GEMM up to
5 tokens (2 on a deep K such as down_proj, an LDS limit), its Triton GEMM above, 3.5-4x slower at
the switch. There hk reads vLLM's own tensors in place (`hk.ops.w4a16.exllama_matmul`: hk's
decode forms on vLLM's ExLlama-shuffled codes), so no second copy is held, and takes exactly
every batch vLLM would send to Triton (`hk_vllm::gptq_w4a16_hybrid`): its decode forms up to
`HYBRID_MAX_M` tokens, its large-batch form above (prefill). vLLM's own op runs the batches its HIP
kernel takes and any shape hk's forms do not. That op is called without zero points for these symmetric layers:
this vLLM passes their stored qzeros to it, and its Triton path asserts on their stride at start.

What it refuses, counted in `REFUSED` by reason: act-order checkpoints
(`desc_act`, a column permutation hk's layout does not carry), anything but
4-bit symmetric with groups, and layers vLLM gave another kernel than
RDNA3W4A16LinearKernel or (on HYBRID_ARCHES) RDNAHybridW4A16LinearKernel. Each of those layers
keeps vLLM's path.

Like the other linear patches it changes what the compiled graph calls, so it
gives the engine its own compile-cache root, `hk-gptq` under the current one.

    HK_PATCH_VLLM=1 vllm serve <a GPTQ-Int4 checkpoint> --dtype float16
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from . import _common, engine

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None

#: The fewest tokens hk's kernel takes. Its symmetric form (packed-fp16 decode,
#: scales in 16-row blocks) is 1.04-1.12x vLLM's kernel at one token on every
#: Qwen3.8-27B TP1 shape and 1.07-1.31x at 2-4 (tools/hk-bench/gptq_w4a16.py on a
#: W7900D); the asymmetric form was 0.87-0.99x at one token, hence 4 before.
MIN_M = 1

#: Layers left on vLLM's path, by reason.
REFUSED: Counter = Counter()


def _fake(x, w_q, w_zp, w_s, g_idx, hk_qw, hk_s, hk_zs, group):
    return x.new_empty((x.shape[0], hk_qw.shape[0]))


PATCH = engine.Op(
    "hk_vllm::gptq_w4a16",
    "gptq_w4a16(Tensor x, Tensor w_q, Tensor w_zp, Tensor w_s, Tensor? g_idx, Tensor hk_qw, "
    "Tensor hk_s, Tensor hk_zs, int group) -> Tensor",
    _fake, paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats

_RDNA3, _HYBRID = "RDNA3W4A16LinearKernel", "RDNAHybridW4A16LinearKernel"
#: Where hk takes RDNAHybridW4A16LinearKernel's layers, and the most tokens its decode forms take
#: there: on Qwen3.8-27B-GPTQ's shapes (group 32) hk is 1.8-8.3x vLLM's Triton GEMM on every shape
#: from where vLLM leaves its HIP kernel to 64 tokens. Above, the large-batch form
#: (`hk.ops.w4a16.exllama_large_matmul`) takes every batch: 1.12-2.48x vLLM's Triton GEMM at
#: 65-4096 tokens on those shapes (95-108 TF/s against vLLM's 39-94).
HYBRID_ARCHES = ("gfx1201",)
HYBRID_MAX_M = 64


def _fake_hybrid(x, w_q, w_s, group, cu):
    return x.new_empty((x.shape[0], w_q.shape[0]))


HYBRID = engine.Op(
    "hk_vllm::gptq_w4a16_hybrid",
    "gptq_w4a16_hybrid(Tensor x, Tensor w_q, Tensor w_s, int group, int cu) -> Tensor",
    _fake_hybrid, paths=("hk", "hk_large", "fallback"), graph="safe")

#: vLLM's own switch from its HIP to its Triton GEMM (rdna_hybrid_w4a16), read at patch time.
_SKINNY: dict = {}


def _vllm_skinny(m: int, k: int) -> bool:
    """Whether vLLM's hybrid kernel runs its HIP skinny GEMM for an [m, k] x."""
    return m <= _SKINNY["max_m"] and (k * m <= _SKINNY["limit"] or _SKINNY["gfx115x"])


def gptq_w4a16_hybrid(x, w_q, w_s, group: int, cu: int):
    """y = x @ W.T on vLLM's own hybrid-kernel tensors: hk's in-place small-batch form for the
    batches vLLM would run on Triton up to HYBRID_MAX_M tokens, its large-batch form above, and
    vLLM's op otherwise."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    n = w_q.shape[0]
    fp16 = x.dtype == torch.float16 and w_s.dtype == torch.float16
    if fp16 and m <= HYBRID_MAX_M and not _vllm_skinny(m, k):
        HYBRID.count("hk")
        return hk_w4.exllama_matmul(x, w_q.view(torch.int32), w_s, group)
    if fp16 and m > HYBRID_MAX_M and hk_w4.exllama_large_takes(n, k, group):
        HYBRID.count("hk_large")
        return hk_w4.exllama_large_matmul(x, w_q.view(torch.int32), w_s, group)
    HYBRID.count("fallback")
    return torch.ops.vllm.rdna_hybrid_w4a16_apply(x, w_q, w_s, None, None, cu, group)


#: Where hk takes RDNA3W4A16LinearKernel's layers without a second copy (D-20261113-14): the codes are
#: repacked once at load into vLLM's RDNA skinny layout (the layout gfx1201's hybrid kernel keeps,
#: built with vLLM's own transform), vLLM's tensors are freed, and hk serves every batch on them --
#: its in-place forms, measured against vLLM's gptq_gemm_rdna3 on the same codes on a W7900D
#: (Qwen3.8-27B GPTQ, every merged linear): 1.06-2.24x at 1-32 tokens, 1.02-1.25x at 128-4096.
SKINNY_ARCHES = ("gfx1100",)
#: The most tokens the small-batch form takes on a skinny layer; above, the large-batch form.
SKINNY_MAX_M = 64


def skinny_max_m(n: int, k: int) -> int:
    """Where a skinny layer hands from the small-batch form to the large-batch one, by shape (W7900D,
    Qwen3.8-27B-GPTQ, against vLLM's gptq_gemm_rdna3): the wide layers (N >= 12288: qkv, gate_up) are
    faster on the large form from 48 tokens, the small o / out projections (N x K <= 5120 x 6144)
    stay on the small form to 64, the rest hand over above 48."""
    if n >= 12288:
        return 32
    return SKINNY_MAX_M if n * k <= 5120 * 6144 else 48


def _fake_skinny(x, w_q, w_s, group):
    return x.new_empty((x.shape[0], w_q.shape[0]))


SKINNY = engine.Op(
    "hk_vllm::gptq_w4a16_skinny",
    "gptq_w4a16_skinny(Tensor x, Tensor w_q, Tensor w_s, int group) -> Tensor",
    _fake_skinny, paths=("hk", "hk_large", "cast"), graph="safe")


def gptq_w4a16_skinny(x, w_q, w_s, group: int):
    """y = x @ W.T for a layer whose only copy of W is hk's skinny one (`_skinny_pwal`): the
    small-batch form up to SKINNY_MAX_M tokens, the large-batch form above. vLLM's tensors are
    gone, so every batch is hk's: a non-fp16 x runs in fp16 and is cast back (counted)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    dtype = x.dtype
    if dtype != torch.float16:
        SKINNY.count("cast")
        x = x.to(torch.float16)
    x = x if x.is_contiguous() else x.contiguous()
    if x.shape[0] <= skinny_max_m(w_q.shape[0], w_q.shape[1] * 8):
        if dtype == torch.float16:
            SKINNY.count("hk")
        out = hk_w4.exllama_matmul(x, w_q, w_s, group)
    else:
        if dtype == torch.float16:
            SKINNY.count("hk_large")
        out = hk_w4.exllama_large_matmul(x, w_q, w_s, group)
    return out if dtype == torch.float16 else out.to(dtype)


def gptq_w4a16(x, w_q, w_zp, w_s, g_idx, hk_qw, hk_s, hk_zs, group: int):
    """y = x @ W.T for a 4-bit GPTQ W held both ways: hk's kernel where it
    takes the batch, vLLM's RDNA3 kernel on vLLM's tensors otherwise."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    run = (hk_w4.plan(m, hk_qw.shape[0], k, group, x.dtype, True)
           if x.dtype == hk_s.dtype and MIN_M <= m <= hk_w4.MAX_TOKENS * 16 else None)
    if run is not None:
        PATCH.count("hk")
        out = torch.empty((m, hk_qw.shape[0]), device=x.device, dtype=hk_s.dtype)
        run(x if x.is_contiguous() else x.contiguous(), hk_qw, hk_s, out)
        return out
    PATCH.count("fallback")
    return _vllm_gptq(x, w_q, w_zp, w_s, g_idx)


#: What this stand-in can pass to vLLM's `gptq_gemm_rdna3`, by its parameter names. vLLM builds
#: differ: some take the act-order index `b_g_idx` (an empty tensor for a checkpoint without act
#: order), some do not. The call is bound by name to whatever this build's signature is.
_GEMM_ARGS = {"a": "x", "b_q_weight": "w_q", "b_qzeros": "w_zp", "b_scales": "w_s",
              "b_g_idx": "g_idx", "use_v2_format": False}
_GEMM_PLAN: list = []


def plan_call(fn):
    """([(name, source)], problem) for a gptq_gemm_rdna3 signature: each parameter's source is
    one of the stand-in's tensors by name, or a constant. problem is "" when every parameter
    without a default has a source, else what is missing -- the layers then keep vLLM's apply."""
    import inspect  # noqa: PLC0415

    plan, missing = [], []
    for p in inspect.signature(fn).parameters.values():
        if p.name in _GEMM_ARGS:
            plan.append((p.name, _GEMM_ARGS[p.name]))
        elif p.default is p.empty:
            missing.append(p.name)
    return plan, (f"gptq_gemm_rdna3 takes {missing}, which this route has no value for"
                  if missing else "")


def _gemm_plan():
    """plan_call() for this process's vLLM, made once."""
    if not _GEMM_PLAN:
        from vllm import _custom_ops as ops  # noqa: PLC0415

        _GEMM_PLAN.append(plan_call(ops.gptq_gemm_rdna3))
    return _GEMM_PLAN[0]


def _vllm_gptq(x, w_q, w_zp, w_s, g_idx):
    """RDNA3W4A16LinearKernel.apply_weights' own GEMM, for the batches hk leaves, called with
    this vLLM's own parameters by name."""
    from vllm import _custom_ops as ops  # noqa: PLC0415

    have = {"x": x, "w_q": w_q, "w_zp": w_zp, "w_s": w_s, "g_idx": g_idx}
    plan, _ = _gemm_plan()
    return ops.gptq_gemm_rdna3(**{n: (have[v] if isinstance(v, str) else v) for n, v in plan})


def _weights(kern, layer):
    """(w_q, w_s, w_zp, g_idx) by the names the kernel was given, as its _get_weight_params
    reads them; g_idx None where this vLLM's kernel has none."""
    def get(attr):
        name = getattr(kern, attr, None)
        return getattr(layer, name, None) if name else None

    return get("w_q_name"), get("w_s_name"), get("w_zp_name"), get("w_gidx_name")


def _why_not(self) -> str:
    """'' if hk can hold this layer's weights, else why not."""
    c = self.quant_config
    kern = type(getattr(self, "kernel", None)).__name__
    if kern == _HYBRID:
        from ..target import current_arch  # noqa: PLC0415

        if current_arch() not in HYBRID_ARCHES:
            return f"kernel {kern} on {current_arch()}"
    elif kern != _RDNA3:
        return f"kernel {kern}"
    if getattr(c, "desc_act", False):
        return "act-order (desc_act)"
    if str(getattr(c, "quant_type", "")) != "uint4b8":
        return f"quant type {getattr(c, 'quant_type', '?')}"
    if c.group_size <= 0 or c.group_size % 16:
        return f"group {c.group_size}"
    if not getattr(self.kernel, "w_q_name", None) or not getattr(self.kernel, "w_s_name", None):
        return "kernel names no weight parameters (w_q_name / w_s_name)"
    if kern == _HYBRID:
        if not _SKINNY:
            return "this vLLM's rdna_hybrid_w4a16 has no MAX_SKINNY_BATCH_SIZE / MEDIUM_SKINNY_LIMIT"
        if c.group_size % 32:
            return f"group {c.group_size}"
        return ""
    plan, problem = _gemm_plan()
    if problem:
        return problem
    return ""


def process_weights_after_loading(self, layer) -> None:
    """hk's layout of the codes, then vLLM's (which shuffles them in place)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    why = _why_not(self)
    if not why and type(self.kernel).__name__ == _HYBRID:
        _hybrid_pwal(self, layer)
        return
    if not why and type(self.kernel).__name__ == _RDNA3:
        from ..target import current_arch  # noqa: PLC0415

        if current_arch() in SKINNY_ARCHES:
            _skinny_pwal(self, layer)
            return
    s = getattr(layer, "scales", None)
    if not why and (s is None or s.dtype not in (torch.float16, torch.bfloat16)):
        why = f"scales {getattr(s, 'dtype', None)}"
    if not why:
        qw = layer.qweight.data                     # [K/8, N], codes packed along K
        k8, n = qw.shape
        if n % 16 or (k8 * 8) % 32:
            why = f"shape {n}x{k8 * 8}"
    if not why:
        q = getattr(layer, "qweight", None)
        if q is None or s is None:
            why = "no qweight / scales on the layer"
    if why:
        REFUSED[why] += 1
        _ORIG_PWAL(self, layer)
        return
    group = self.quant_config.group_size
    # The stored zero points (GPTQ v1: one short), checked to be 8 -- vLLM's for uint4b8.
    qz = getattr(layer, "qzeros", None)
    qz = qz.data if qz is not None and qz.dim() == 2 else torch.full(
        (k8 * 8 // group, n // 8), 0x77777777, device=qw.device, dtype=torch.int32)
    try:
        hq, hs, _ = hk_w4.repack_gptq(qw, qz, layer.scales.data, group, zero_offset=1,
                                      dtype=layer.scales.dtype, symmetric=True)
    except ValueError:
        REFUSED["zero points not 8"] += 1
        _ORIG_PWAL(self, layer)
        return
    _ORIG_PWAL(self, layer)
    w_q, w_s, w_zp, g_idx = _weights(self.kernel, layer)
    plan, _ = _gemm_plan()
    if w_q is None or w_s is None or w_zp is None or (
            any(n == "b_g_idx" for n, _ in plan) and g_idx is None):
        REFUSED["vLLM's kernel holds no zero points / g_idx here"] += 1
        return                              # vLLM's layer, untouched: its own apply runs
    layer.hk_qw = torch.nn.Parameter(hq, requires_grad=False)
    layer.hk_s = torch.nn.Parameter(hs, requires_grad=False)
    # the op's schema keeps a zs slot; the symmetric form reads none
    layer.hk_zs = torch.nn.Parameter(hs.new_empty(0), requires_grad=False)


def skinny_why(layer, group: int) -> str:
    """'' if this RDNA3W4A16 layer's checkpoint tensors (before vLLM touches them) can become hk's
    only copy -- symmetric codes with GPTQ's stored zero point 8 (0x77777777, one short), fp16
    scales, a shape both of hk's forms take -- else why not."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    q, s = getattr(layer, "qweight", None), getattr(layer, "scales", None)
    if q is None or s is None or q.dim() != 2 or s.dim() != 2:
        return "no qweight / scales on the layer"
    if q.dtype != torch.int32 or s.dtype != torch.float16:
        return f"qweight {q.dtype} / scales {s.dtype} (fp16 scales only)"
    k, n = q.shape[0] * 8, q.shape[1]
    if s.shape != (k // group, n):
        return f"scales {tuple(s.shape)}"
    if not (hk_w4.exllama_takes(1, n, k, group) and hk_w4.exllama_large_takes(n, k, group)):
        return f"shape {n}x{k}"
    qz = getattr(layer, "qzeros", None)
    if qz is not None and qz.dim() == 2 and not bool((qz.data == 0x77777777).all()):
        return "zero points not 8"
    return ""


def _skinny_pwal(self, layer) -> None:
    """RDNA3W4A16LinearKernel's layer as ONE copy in vLLM's RDNA skinny layout: the checkpoint's
    codes ([K/8, N], packed along K) in the skinny layout (`hk.ops.w4a16.gptq_to_skinny`: int32
    [N, K/8], ExLlama nibble order, as vLLM's `pack_skinny_int4`) and the scales as [N, K/group] --
    what gfx1201's hybrid
    kernel holds and hk's forms read in place. vLLM's own processing never runs on the layer, and
    its tensors are freed. A layer hk's forms cannot take keeps vLLM's path (counted)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    group = self.quant_config.group_size
    why = skinny_why(layer, group)
    if why:
        REFUSED[why] += 1
        _ORIG_PWAL(self, layer)
        return
    q, s = layer.qweight.data, layer.scales.data
    w_q = hk_w4.gptq_to_skinny(q)                                  # [N, K/8], ExLlama nibble order
    w_s = s.t().contiguous()                                                              # [N, K/G]
    layer.hk_skinny_q = torch.nn.Parameter(w_q, requires_grad=False)
    layer.hk_skinny_s = torch.nn.Parameter(w_s, requires_grad=False)
    for name in ("qweight", "scales", "qzeros", "g_idx"):          # free vLLM's copy
        t = getattr(layer, name, None)
        if isinstance(t, torch.Tensor):
            t.data = t.data.new_empty((0,))
    SKINNY_LAYERS[0] += 1


#: Layers held as hk's one skinny copy, in this process.
SKINNY_LAYERS = [0]


def _hybrid_pwal(self, layer) -> None:
    """vLLM's own repack, then hk checks it can read the result in place."""
    _ORIG_PWAL(self, layer)
    why = hybrid_why(self.kernel, layer, self.quant_config.group_size)
    if why:
        REFUSED[why] += 1
        return
    layer.hk_hybrid = True


def hybrid_why(kernel, layer, group: int) -> str:
    """'' if hk's forms can read this layer's RDNAHybridW4A16LinearKernel tensors in place (after
    vLLM's repack), else why not. Shared with vllm_quark, whose layers end on the same kernel."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    w_q, w_s, _, _ = _weights(kernel, layer)
    if kernel.config.zero_points or str(kernel.config.weight_type) != "uint4b8":
        return "zero points (not uint4b8)"
    if w_q is None or w_s is None or w_q.dtype != torch.int8 or w_q.dim() != 2:
        return "vLLM's hybrid tensors are not int8 [N, K/2] codes"
    if not w_q.is_contiguous() or not w_s.is_contiguous() or w_s.dim() != 2:
        return "vLLM's hybrid tensors are padded or strided"
    if not hk_w4.exllama_takes(1, w_q.shape[0], w_q.shape[1] * 2, group):
        return f"shape {w_q.shape[0]}x{w_q.shape[1] * 2}"
    if w_s.shape != (w_q.shape[0], w_q.shape[1] * 2 // group):
        return f"scales {tuple(w_s.shape)}"
    return ""


def read_skinny() -> bool:
    """Read vLLM's own HIP/Triton switch (rdna_hybrid_w4a16) into _SKINNY; False where this vLLM
    has none (its hybrid layers are then refused, counted)."""
    try:
        from vllm.model_executor.kernels.linear.mixed_precision import (  # noqa: PLC0415
            rdna_hybrid_w4a16 as rh)

        _SKINNY.update(max_m=rh.MAX_SKINNY_BATCH_SIZE, limit=rh.MEDIUM_SKINNY_LIMIT_ELEMENTS,
                       gfx115x=bool(getattr(rh, "_on_gfx115x", lambda: False)()))
        return True
    except (ImportError, AttributeError):
        _SKINNY.clear()
        return False


def apply(self, layer, x, bias=None):
    if hasattr(layer, "hk_skinny_q"):
        out = SKINNY(x.reshape(-1, x.shape[-1]), layer.hk_skinny_q, layer.hk_skinny_s,
                     self.quant_config.group_size)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], layer.hk_skinny_q.shape[0])
    if getattr(layer, "hk_hybrid", False):
        from vllm.utils.platform_utils import num_compute_units  # noqa: PLC0415

        w_q, w_s, _, _ = _weights(self.kernel, layer)
        out = HYBRID(x.reshape(-1, x.shape[-1]), w_q, w_s, self.quant_config.group_size,
                     num_compute_units())
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], w_q.shape[0])
    if not hasattr(layer, "hk_qw"):
        return _ORIG_APPLY(self, layer, x, bias)
    w_q, w_s, w_zp, g_idx = _weights(self.kernel, layer)
    out = PATCH(x.reshape(-1, x.shape[-1]), w_q, w_zp, w_s, g_idx, layer.hk_qw, layer.hk_s,
                layer.hk_zs, self.quant_config.group_size)
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], layer.hk_qw.shape[0])


def apply_patch() -> str:
    """Route vLLM's GPTQ linear layers. Returns what happened, in one line."""
    global _ORIG_PWAL, _ORIG_APPLY
    if not _common.importable("vllm"):
        return "vllm is not importable; gptq not patched"
    if _ORIG_APPLY is not None:
        return "gptq: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "gptq: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    PATCH.define(gptq_w4a16)
    SKINNY.define(gptq_w4a16_skinny)
    if read_skinny():                       # else its hybrid layers are refused, counted
        HYBRID.define(gptq_w4a16_hybrid)
    cls = auto_gptq.AutoGPTQLinearMethod
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    return (f"gptq -> hk for 4-bit symmetric GPTQ layers on RDNA3W4A16LinearKernel "
            f"({', '.join(SKINNY_ARCHES)}: repacked once into the RDNA skinny layout, vLLM's copy "
            f"freed, every batch on hk's forms), and on "
            f"RDNAHybridW4A16LinearKernel ({', '.join(HYBRID_ARCHES)}) the fp16 batches vLLM "
            f"runs on Triton (decode forms to {HYBRID_MAX_M} tokens, the large-batch form "
            f"above), in place (compile cache "
            f"{_own_compile_cache('hk-gptq')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is None:
        return "gptq: not patched"
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    auto_gptq.AutoGPTQLinearMethod.process_weights_after_loading = _ORIG_PWAL
    auto_gptq.AutoGPTQLinearMethod.apply = _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "gptq -> vllm"


def status() -> str:
    if _ORIG_APPLY is None:
        return "not patched"
    refused = ", ".join(f"{k}: {v}" for k, v in REFUSED.items())
    hyb = f"; {HYBRID.summary()}" if HYBRID.defined else ""
    hyb += f"; skinny layers {SKINNY_LAYERS[0]}: {SKINNY.summary()}" if SKINNY.defined else ""
    return PATCH.summary() + hyb + (f"; layers left on vLLM's path: {refused}" if refused else "")
