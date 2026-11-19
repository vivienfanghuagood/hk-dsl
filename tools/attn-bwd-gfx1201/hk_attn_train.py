"""[hk-A] D-20261119-17 drop-in: o = fn(q, k, v, causal, scale) on BSHD bf16, autograd-capable.

hk forward (base-2 LSE kept), hk Delta, hk dQ, hk dK, hk dV -- all reading BSHD in place: a (B, S, H, D) tensor
whose H and D are contiguous is the (B, 1, S, H * D) global the bnhd kernels address, row stride free, so the
trainer's views of one fused QKV tensor (a slice of the last axis) need no copy. Megatron's SBHD (batch stride not
S * row stride) is copied to BSHD first. Outputs are BSHD contiguous.
"""
import torch
import hk.ops.attn as A
import hk.ops.attn_bwd as AB

_K = {}
DELTA_WARPS = 4


def _kernels(D, causal, scale, n):
    key = (D, causal, scale, n >= 1024)
    k = _K.get(key)
    if k is None:
        default = scale is None or abs(scale - D ** -0.5) < 1e-12
        sc = None if default else float(scale)
        sfx = f"d{D}{'_c' if causal else ''}" + ("" if default else f"_s{abs(hash(sc)) % 10**8}")
        til = A._tiling(D, causal, n)
        k = _K[key] = dict(
            fwd=A.attn_kernel(f"tb_fwd_lse_{sfx}", head_dim=D, causal=causal, scale=sc, lse=True, bnhd=True, **til),
            delta=AB.delta_kernel(f"tb_delta_d{D}", head_dim=D, warps=DELTA_WARPS, bnhd=True),
            dq=AB.dq_kernel(f"tb_dq_{sfx}", head_dim=D, causal=causal, scale=sc, bnhd=True),
            dk=AB.dkv_kernel(f"tb_dk_{sfx}", mode="dk", head_dim=D, causal=causal, scale=sc, bnhd=True),
            dv=AB.dkv_kernel(f"tb_dv_{sfx}", mode="dv", head_dim=D, causal=causal, scale=sc, bnhd=True))
    return k


def _bnhd(t):
    """(B, S, H, D) -> the (B, 1, S, H * D) view the kernels read, copying only when the batch is not
    S rows of the row stride apart (SBHD) or H, D are not contiguous."""
    B, S, H, D = t.shape
    if not (t.stride(3) == 1 and t.stride(2) == D and t.stride(0) == S * t.stride(1)):
        t = t.contiguous()
    return t.flatten(2).unsqueeze(1)


def _pad(n):
    r = 16 * DELTA_WARPS
    return -(-n // r) * r


class _Attn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        B, S, H, D = q.shape
        kn = _kernels(D, causal, scale, S)
        qb, kb, vb = _bnhd(q), _bnhd(k), _bnhd(v)
        o = torch.empty(B, S, H, D, device=q.device, dtype=q.dtype)
        lse = torch.empty(B, H, 1, _pad(S), device=q.device, dtype=torch.float32)
        kn["fwd"](qb, kb, vb, _bnhd(o), lse)
        ctx.save_for_backward(qb, kb, vb, o, lse)
        ctx.causal, ctx.scale, ctx.hk = causal, scale, k.shape[2]
        return o

    @staticmethod
    def backward(ctx, do):
        qb, kb, vb, o, lse = ctx.saved_tensors
        B, S, H, D = o.shape
        Hk = ctx.hk
        kn = _kernels(D, ctx.causal, ctx.scale, S)
        dob, ob = _bnhd(do), _bnhd(o)
        delta = torch.empty_like(lse)
        kn["delta"](ob, dob, delta)
        dq = torch.empty(B, S, H, D, device=o.device, dtype=o.dtype)
        dk = torch.empty(B, S, Hk, D, device=o.device, dtype=o.dtype)
        dv = torch.empty_like(dk)
        kn["dq"](qb, kb, vb, dob, lse, delta, _bnhd(dq))
        kn["dk"](qb, kb, vb, dob, lse, delta, _bnhd(dk))
        kn["dv"](qb, kb, vb, dob, lse, delta, _bnhd(dv))
        return dq, dk, dv, None, None


def fn(q, k, v, causal, scale):
    return _Attn.apply(q, k, v, causal, scale)
