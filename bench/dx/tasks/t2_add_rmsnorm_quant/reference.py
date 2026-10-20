"""T2 add_rmsnorm_quant: the reference, the inputs and the check. task.md is the problem."""

import torch

from bench.dx.common import Mismatch, assert_close

SHAPES = [
    dict(name="T1-H4096", T=1, H=4096),
    dict(name="T16-H4096", T=16, H=4096),
    dict(name="T37-H5120", T=37, H=5120),
    dict(name="T256-H4096", T=256, H=4096),
    dict(name="T2048-H4096", T=2048, H=4096),
    dict(name="T8192-H4096", T=8192, H=4096),
    dict(name="T4096-H5120", T=4096, H=5120),
    dict(name="T2048-H8192", T=2048, H=8192),
]
QUICK = ("T16-H4096", "T37-H5120", "T8192-H4096")
SMALL = [dict(name="small", T=16, H=1024), dict(name="small-ragged", T=5, H=4096)]
MUTABLE_INPUTS = ("x", "residual")


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t, h = shape["T"], shape["H"]
    x = torch.randn(t, h, generator=g, device=device, dtype=torch.bfloat16)
    residual = torch.randn(t, h, generator=g, device=device, dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(h, generator=g, device=device)).to(torch.bfloat16)
    return {"x": x, "residual": residual, "weight": weight, "eps": 1e-6}


def reference(x, residual, weight, eps):
    r = (x.float() + residual.float()).to(torch.bfloat16)
    rf = r.float()
    y = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    scale = (y.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-10)
    q = torch.round(y / scale).clamp(-128, 127).to(torch.int8)
    return q, scale, r


def torch_impl(x, residual, weight, eps):
    """What a PyTorch user writes (vLLM's forward_native of the two ops)."""
    r = x + residual
    rf = r.float()
    y = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    scale = y.abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(y / scale).clamp(-128, 127).to(torch.int8)
    return q, scale, r


def check(out, ref, inputs):
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise Mismatch(f"expected (q, scale, r), got {type(out).__name__}")
    q, scale, r = out
    q_ref, s_ref, r_ref = ref
    msgs = [assert_close("r", r, r_ref, rtol=2.0 ** -8, atol=1e-6, dtype=torch.bfloat16),
            assert_close("scale", scale, s_ref, rtol=1e-2, atol=0.0, dtype=torch.float32)]
    if q.dtype != torch.int8 or tuple(q.shape) != tuple(q_ref.shape):
        raise Mismatch(f"q: {q.dtype} {tuple(q.shape)}, expected int8 {tuple(q_ref.shape)}")
    d = (q.int() - q_ref.int()).abs()
    if d.max().item() > 1:
        i = int(d.flatten().argmax())
        raise Mismatch(f"q: {int((d > 1).sum())} elements off by more than 1, first at flat "
                       f"index {i}: got {q.flatten()[i].item()}, want {q_ref.flatten()[i].item()}")
    msgs.append(f"q: {int((d == 1).sum())}/{d.numel()} off by one")
    return "; ".join(msgs)


def work(shape):
    t, h = shape["T"], shape["H"]
    return {"bytes": 2 * t * h * 2 + h * 2 + t * h * 3 + 4 * t}
