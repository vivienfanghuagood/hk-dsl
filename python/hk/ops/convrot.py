"""ConvRot W8A8 linear on gfx1100: y = dequant(int8(rot(x)) @ int8(rot(W))^T) (+ bias).

ConvRot (arXiv:2512.03673), as SGLang-Diffusion's `convrot_int8` runs it on CUDA
(sglang/kernels/jit/csrc/gemm/convrot_int8): every `g`-wide group of a row is
rotated by H / sqrt(g), H the Kronecker power of the *regular* 4x4 Hadamard

    H4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]

(every row sums to +1, so a group's mean stays spread out rather than piling up
in one coefficient), then quantized per row, dynamically and symmetrically
(scale = absmax / 127, round to nearest even, clamp to +-127). The weights take
the same transform once, row by row ([N, K]); since H / sqrt(g) is orthogonal,
rot(x) . rot(w) == x . w. The product is an int8 GEMM accumulated in int32 and
dequantized as acc * x_scale[m] * w_scale[n].

On gfx1100 the int8 GEMM is hk's (v_wmma_i32_16x16x16_iu8, hk.ops.gemm with
op_dtype=i8): at the f16 rate, so the win on offer is bytes, not FLOPs. The
rotation here is a GEMM against the block-diagonal H (bf16 in, fp32 out) and the
quantization hk.ops.quant's row kernel: three launches, a first form.

    wq, ws = prepare_weight(w, group)          # once, at load: int8 [N, K], fp32 [N]
    y = linear(x, wq, ws, group, bias=None)    # x bf16 [..., K] -> bf16 [..., N]
"""

from __future__ import annotations

import functools

from ..types import bf16, i8

H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


@functools.lru_cache(maxsize=None)
def _hadamard(group: int, device: str):
    """H / sqrt(group), bf16 (entries +-1/sqrt(group): exact for group a power of 4)."""
    import torch  # noqa: PLC0415

    n, h = group, torch.tensor(H4, dtype=torch.float64)
    m = torch.ones(1, 1, dtype=torch.float64)
    while m.shape[0] < n:
        m = torch.kron(m, h)
    if m.shape[0] != n:
        raise ValueError(f"convrot: group {group} is not a power of 4 (16, 64, 256)")
    return (m / n ** 0.5).to(device=device, dtype=torch.bfloat16)


def rotate(x, group: int):
    """Every `group`-wide slice of x's rows times H / sqrt(group), in fp32."""
    from ..templates.gemm import gemm  # noqa: PLC0415  -- templates import ops

    import torch  # noqa: PLC0415

    k = x.shape[-1]
    if k % group:
        raise ValueError(f"convrot: K={k} is not a multiple of the group {group}")
    rows = x.reshape(-1, group)
    h = _hadamard(group, str(x.device))
    # (rows, g) @ H^T; H is symmetric
    return gemm(rows, h, out_dtype=torch.float32).view(*x.shape)


def quantize_rows(v):
    """Per-row symmetric int8 of an fp32 tensor: (q int8, scale fp32 [rows])."""
    from .quant import quantize  # noqa: PLC0415

    return quantize(v.contiguous())


def prepare_weight(w, group: int):
    """A [N, K] weight rotated and quantized per output channel: (int8 [N, K], fp32 [N])."""
    q, s = quantize_rows(rotate(w.to(device=w.device), group))
    return q, s


def reference(x, wq, ws, group: int, bias=None):
    """The same product in torch, fp32 throughout (int32 GEMM exact): the oracle."""
    import torch  # noqa: PLC0415

    h = _hadamard(group, str(x.device)).float()
    v = (x.float().reshape(-1, group) @ h).view(-1, x.shape[-1])
    amax = v.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    xq = torch.round(torch.clamp(v / scale, -127, 127)).to(torch.int8)
    acc = torch._int_mm(xq, wq.t().contiguous()) if xq.shape[0] > 16 else (xq.int() @ wq.int().t())
    y = acc.float() * scale * ws.view(1, -1)
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype).view(*x.shape[:-1], wq.shape[0])


def _scale(acc, ctx):
    return acc * ctx.x_scale * ctx.w_scale


def _scale_bias(acc, ctx):
    return acc * ctx.x_scale * ctx.w_scale + ctx.bias


@functools.lru_cache(maxsize=None)
def _kernel(with_bias: bool):
    from ..templates import gemm as T  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from . import gemm as G  # noqa: PLC0415

    fn = _scale_bias if with_bias else _scale
    # shapes only classify the operands: xs one value a row of C, ws (and bias) one a column
    ops = {"x_scale": torch.empty(2, 1, device="meta"), "w_scale": torch.empty(3, device="meta")}
    if with_bias:
        ops["bias"] = torch.empty(3, dtype=torch.bfloat16, device="meta")
    ep = T.bind(fn, ops, 2, 3, bf16)
    return G.gemm_kernel(f"convrot_i8_{'bias' if with_bias else 'nobias'}", op_dtype=i8,
                         epilogue=ep, **G.DEFAULT), ep


def linear(x, wq, ws, group: int, bias=None, out=None):
    """bf16 y = dequant(int8(rot(x)) @ wq^T) (+ bias); x [..., K] bf16, wq int8 [N, K], ws fp32 [N]."""
    import torch  # noqa: PLC0415

    from ..templates import gemm as T  # noqa: PLC0415

    k = x.shape[-1]
    rows = x.reshape(-1, k)
    m, n = rows.shape[0], wq.shape[0]
    xq, xs = quantize_rows(rotate(rows, group))
    kern, ep = _kernel(bias is not None)
    if out is None:
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    tensors = {"x_scale": xs.view(m, 1), "w_scale": ws.view(n)}
    if bias is not None:
        tensors["bias"] = bias.view(n).to(torch.bfloat16)
    prep = dict(T._preparer(ep, m))
    args = [prep[name](tensors[name]) for name in ep.tensors]
    kern(xq, wq, out, *args)
    return out.view(*x.shape[:-1], n)
