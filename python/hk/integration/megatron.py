"""Megatron-LM (with Transformer Engine) on RDNA4: hk's attention and fc1 backward, without changing
Megatron's or TE's source.

    cd Megatron-LM
    python -m hk.integration.megatron pretrain_gpt.py <Megatron's arguments>
    # or: torchrun ... -m hk.integration.megatron pretrain_gpt.py ...

What it installs, each under its own switch (`python -m hk.integration --status` lists them):

* `megatron_attn` (HK_MEGATRON_ATTN): Megatron's core attention is hk.ops.attn_bwd.attention (flash
  attention forward and backward). It replaces TE's DotProductAttention, which runs unfused on gfx1201
  (no fused attention is built there). It applies to causal or unmasked self-attention without dropout,
  bf16, head_dim 64 or 128. The model is built with that provider, so a layer it does not fit raises
  rather than falling back: set the switch to 0 for such a model. HK_MEGATRON_ATTN=sdpa puts torch's
  SDPA (AOTriton) there instead: the best stock attention, for a baseline.
* `megatron_fc1_bwd` (HK_MEGATRON_FC1_BWD): the MLP's fc1 backward under FP8 (delayed scaling).
  - Megatron's SwiGLU backward and the fp8 cast of its output run as one kernel
    (hk.ops.fp8_cast.swiglu_bwd_cast_transpose). It writes the e5m2 gradient with padded row strides.
  - fc1's dgrad and wgrad GEMMs read that gradient with hk.ops.gemm.scaled_mm, wgrad accumulating into
    Megatron's fp32 main_grad.
  - TE's own GEMM ignores strides, so a padded tensor never reaches it: those two GEMMs go to hk, every
    other GEMM to TE, and anything padded headed for TE raises.
  - Needs --disable-bias-linear and SwiGLU.
* `megatron_lmhead_wgrad` (HK_MEGATRON_LMHEAD_WGRAD): configuration, no hk kernel. Megatron's output
  layer accumulates its weight gradient through apex's fused_weight_gradient_mlp_cuda, which picks a slow
  tile on gfx1201. This routes it through TE's general_gemm (hipBLASLt, tunable with TE's algorithm cache).

docs/recipes/megatron-rdna4.md is the whole recipe: the Transformer Engine build, Megatron's flags,
GEMM tuning, and the measured speedup.
"""

from __future__ import annotations

import math
import os
import runpy
import sys

from . import switches

PAD = 64          # bytes of row padding on fc1's fp8 gradient (hk.ops.gemm.FP8_POW2's note)


def _log(msg: str) -> None:
    if not os.environ.get("HK_QUIET"):
        print(f"[hk] megatron: {msg}", flush=True)


# ---------------------------------------------------------------------------------------- attention

def _sdpa(q, k, v, *, causal, scale):
    """torch's SDPA on BSHD views (AOTriton on gfx1201): the recipe's baseline arm."""
    import torch.nn.functional as F  # noqa: PLC0415

    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=scale,
        enable_gqa=q.shape[2] != k.shape[2]).transpose(1, 2)


def _install_attention(impl: str = "") -> None:
    import torch  # noqa: PLC0415
    from megatron.core.extensions import transformer_engine_spec_provider as tsp  # noqa: PLC0415
    from megatron.core.transformer.enums import AttnMaskType  # noqa: PLC0415
    from megatron.core.transformer.module import MegatronModule  # noqa: PLC0415

    from ..ops import attn_bwd  # noqa: PLC0415

    class HKCoreAttention(MegatronModule):
        """core_attention for Megatron's SelfAttention: SBHD q, k, v in, (S, B, H * D) out."""

        def __init__(self, config, layer_number, attn_mask_type, attention_type,
                     attention_dropout=None, softmax_scale=None, cp_comm_type=None,
                     pg_collection=None, **_):
            super().__init__(config=config)
            if getattr(config, "context_parallel_size", 1) != 1:
                raise ValueError("HK_MEGATRON_ATTN: context parallelism is not supported; set it to 0")
            dropout = config.attention_dropout if attention_dropout is None else attention_dropout
            if dropout:
                raise ValueError("HK_MEGATRON_ATTN: attention dropout is not supported "
                                 "(--attention-dropout 0), or set HK_MEGATRON_ATTN=0")
            self.attn_mask_type = attn_mask_type
            self.scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(config.kv_channels)

        def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                    attention_bias=None, packed_seq_params=None, **_):
            if attention_bias is not None or packed_seq_params is not None:
                raise ValueError("HK_MEGATRON_ATTN: attention bias and packed sequences are not "
                                 "supported; set HK_MEGATRON_ATTN=0")
            mt = attn_mask_type if attn_mask_type is not None else self.attn_mask_type
            if mt not in (AttnMaskType.causal, AttnMaskType.no_mask):
                raise ValueError(f"HK_MEGATRON_ATTN: mask type {mt} is not supported")
            # SBHD -> BSHD views; at B = 1 read in place (hk.ops.attn_bwd._bnhd)
            o = attention(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
                                   causal=mt == AttnMaskType.causal, scale=self.scale)
            s, b = query.shape[0], query.shape[1]
            return o.transpose(0, 1).reshape(s, b, -1)

    torch  # noqa: B018 -- imported for the side effect of failing early without it
    if impl not in ("", "sdpa"):
        raise ValueError(f"HK_MEGATRON_ATTN={impl!r}: 1 (hk), sdpa (torch's, a baseline) or 0")
    attention = _sdpa if impl == "sdpa" else attn_bwd.attention
    tsp.TESpecProvider.core_attention = lambda self: HKCoreAttention
    _log("core attention -> " + ("torch SDPA (HK_MEGATRON_ATTN=sdpa)" if impl == "sdpa"
                                  else "hk.ops.attn_bwd.attention"))


# ---------------------------------------------------------------------------------------- fc1 backward

_CUR_MLP = [None]


def _pad_rows(rows: int, cols: int, device):
    import torch  # noqa: PLC0415

    return torch.empty(rows, cols + PAD, dtype=torch.uint8, device=device)[:, :cols]


def _padded_f8(row, col, scale_inv, quantizer, shape):
    import torch  # noqa: PLC0415
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor  # noqa: PLC0415

    class _PaddedF8(Float8Tensor):
        def contiguous(self, memory_format=torch.contiguous_format):   # keep the padded layouts
            return self

    t = _PaddedF8(shape=shape, dtype=torch.bfloat16, data=row, fp8_scale_inv=scale_inv,
                  fp8_dtype=quantizer.dtype, data_transpose=col, quantizer=quantizer,
                  requires_grad=False)
    t._transpose_invalid = False
    t._hk_padded = True
    return t


def _is_padded(t) -> bool:
    if getattr(t, "_hk_padded", False):
        return True
    d, tr = getattr(t, "_data", None), getattr(t, "_transpose", None)
    return (d is not None and not d.is_contiguous()) or (tr is not None and not tr.is_contiguous())


def _swiglu_function():
    import torch  # noqa: PLC0415

    class HKSwiGLU(torch.autograd.Function):
        """Megatron's SwiGLU forward as it is; the backward writes fc1's fp8 gradient directly."""

        @staticmethod
        def forward(ctx, x):
            from megatron.core.fusions.fused_bias_swiglu import swiglu  # noqa: PLC0415

            ctx.save_for_backward(x)
            ctx.mlp = _CUR_MLP[0]
            return swiglu(x)

        @staticmethod
        def backward(ctx, grad):
            import transformer_engine_torch as tex  # noqa: PLC0415

            from ..ops import fp8_cast  # noqa: PLC0415

            (x,) = ctx.saved_tensors
            q = ctx.mlp.linear_fc1.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            g2 = grad.reshape(-1, grad.shape[-1]).contiguous()
            m, f2 = x2.shape
            row, col = _pad_rows(m, f2, x.device), _pad_rows(f2, m, x.device)
            sinv = torch.empty((), dtype=torch.float32, device=x.device)
            fp8_cast.swiglu_bwd_cast_transpose(g2, x2, q.scale, q.amax, out=(row, col),
                                               scale_inv=sinv)
            # TE's layouts: rowwise the logical (S, B, 2F), columnwise (2F, S, B)
            return _padded_f8(row.view(*shape), col.view(shape[-1], *shape[:-1]), sinv, q, shape)

    return HKSwiGLU


def _route_gemm(te_gemm):
    """TE's general_gemm, with the GEMMs whose B is a padded fp8 tensor on hk.ops.gemm.scaled_mm."""
    import torch  # noqa: PLC0415

    from ..ops import gemm as hg  # noqa: PLC0415

    def transpose_of(t):
        if t._transpose is None or t._transpose_invalid:
            t.update_usage(columnwise_usage=True)
        return t._transpose

    def general_gemm(A, B, *args, **kw):
        if not _is_padded(B):
            if _is_padded(A):
                raise RuntimeError("hk megatron_fc1_bwd: a padded fp8 operand would reach TE's GEMM, "
                                   "which ignores strides")
            return te_gemm(A, B, *args, **kw)
        layout = kw.get("layout", "TN")
        if layout == "NN":      # dgrad: dX (M, K) = dY (M, N) @ W (N, K); scaled_mm takes W^T (K, N)
            wt = transpose_of(A)
            dy = B._data.view(-1, B._data.shape[-1]).view(torch.float8_e5m2)
            out = hg.scaled_mm(dy, wt.view(torch.float8_e4m3fn), B._scale_inv.reshape(()),
                               A._scale_inv.reshape(()))
            if kw.get("out") is not None:
                kw["out"].view(out.shape).copy_(out)
                out = kw["out"]
            return out.view(*B.shape[:-1], wt.shape[0]), None, None, None
        if layout == "NT":      # wgrad: dW (N, K) = dY^T (N, M) @ X^T (K, M)^T, into main_grad
            xt = transpose_of(A)
            dyt = transpose_of(B)
            out = kw.get("out")
            acc = bool(kw.get("accumulate", False)) and out is not None
            res = hg.scaled_mm(dyt.view(dyt.shape[0], -1).view(torch.float8_e5m2),
                               xt.view(xt.shape[0], -1).view(torch.float8_e4m3fn),
                               B._scale_inv.reshape(()), A._scale_inv.reshape(()),
                               out=out, accumulate=acc)
            return res, None, None, None
        raise RuntimeError(f"hk megatron_fc1_bwd: a padded fp8 operand in a {layout} GEMM")

    return general_gemm


def _install_fc1_bwd() -> None:
    import megatron.core.transformer.mlp as mlp_mod  # noqa: PLC0415
    import transformer_engine.pytorch.module.layernorm_linear as lnl  # noqa: PLC0415

    fn = _swiglu_function()

    def swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        if bias is not None:
            raise ValueError("HK_MEGATRON_FC1_BWD expects --disable-bias-linear")
        return fn.apply(input)

    mlp_mod.bias_swiglu_impl = swiglu_impl
    forward = mlp_mod.MLP.forward

    def mlp_forward(self, *a, **k):
        _CUR_MLP[0] = self
        return forward(self, *a, **k)

    mlp_mod.MLP.forward = mlp_forward
    lnl.general_gemm = _route_gemm(lnl.general_gemm)
    _log("fc1 backward -> hk.ops.fp8_cast.swiglu_bwd_cast_transpose + hk.ops.gemm.scaled_mm "
         "(fp8 delayed scaling)")


# ---------------------------------------------------------------------------------------- LM head

def _install_lmhead_wgrad() -> None:
    import megatron.core.tensor_parallel.layers as tpl  # noqa: PLC0415
    from transformer_engine.pytorch.cpp_extensions import general_gemm  # noqa: PLC0415

    class _Wgrad:
        @staticmethod
        def wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
            x = total_input.reshape(-1, total_input.shape[-1])
            g = grad_output.reshape(-1, grad_output.shape[-1])
            general_gemm(x, g, out_dtype=main_grad.dtype, layout="NT", out=main_grad,
                         accumulate=True, grad=True)

        @staticmethod
        def wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
            raise RuntimeError("HK_MEGATRON_LMHEAD_WGRAD expects an fp32 main_grad")

    tpl.fused_weight_gradient_mlp_cuda = _Wgrad
    _log("output layer weight-gradient accumulation -> TE general_gemm")


_INSTALLERS = (("megatron_attn", _install_attention), ("megatron_fc1_bwd", _install_fc1_bwd),
               ("megatron_lmhead_wgrad", _install_lmhead_wgrad))


def install(env=None) -> list:
    """Install every megatron route its switch turns on; the names installed."""
    done = []
    for name, fn in _INSTALLERS:
        if switches.enabled(name, env):
            if name == "megatron_attn":
                fn(switches.value(name, env).lower())
            else:
                fn()
            done.append(name)
    return done


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or not argv[0].endswith(".py"):
        raise SystemExit("usage: python -m hk.integration.megatron pretrain_gpt.py <Megatron arguments>")
    script = os.path.abspath(argv[0])
    sys.path.insert(0, os.path.dirname(script))      # Megatron's own imports, as `python script`
    install()
    sys.argv = [script] + argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
