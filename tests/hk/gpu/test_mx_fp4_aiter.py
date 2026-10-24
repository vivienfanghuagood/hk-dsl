"""gfx950: FP8 activations x AITER's shuffled MXFP4 MoE weights on the scaled MFMA (b_fmt="fp4").

The weights and scales are laid out the way AITER's gfx950 `shuffle_weight` / `shuffle_scale`
(is_guinterleave=True, gate_up=True) store DeepSeek-V4.1's w13 -- reimplemented here in torch so
the test needs no AITER. The activations are FP8 with ordinary 1x32 E8M0 scales, row-major as
they come: hk's FP8 16x128 fragment (HipKittens rt_16x128, stride 16) already holds, in lane group
g, x[16g, 16g+16) and x[64+16g, 64+16g+16) -- the scaled MFMA's FP8 K order -- while an FP4
operand's lane holds 32 consecutive K. Checked against a torch dequantized product.
"""

from __future__ import annotations

import pytest

import hk
from hk import fp8e4m3, fp32
from hk.target import current_arch

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("the scaled f8f6f4 MFMA is gfx950's", allow_module_level=True)

E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


def aiter_shuffle_w13(w):
    """[E, 2I, K/2] uint8 (gate rows, then up) -> AITER gfx950 is_guinterleave gate_up layout."""
    e, n, kp = w.shape
    x = w.view(e, 2, n // 2 // 16, 16, kp // 64, 4, 16)
    return x.permute(0, 2, 1, 4, 5, 3, 6).contiguous().view(e * n, kp)


def aiter_shuffle_w13_scale(s, e):
    """[E * 2I, K/32] uint8 -> AITER's is_guinterleave gate_up scale layout."""
    rows, kb = s.shape
    n = rows // e
    x = s.view(e, 2, n // 32, 16, kb // 8, 2, 4)
    return x.permute(0, 2, 4, 6, 3, 5, 1).contiguous().view(rows, kb)


def kernel(K, I, GR, name="mx_fp4_aiter_t"):
    def body(x, xs, w, ws, o):
        nb = hk.arch.block_idx.x
        e = hk.arch.block_idx.y
        acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
        for kc in range(K // 128):
            a = x[hk.elem_coord(0, 0, 0, kc * 128)].load(hk.RT(fp8e4m3, 16, 128))
            s_a = hk.load_mx_scales(xs, 16, 4, 0, kc * 4)
            b = hk.load_fp4_frags(hk.RT(fp8e4m3, 32, 128), w, e * GR + nb * 16, kc,
                                  group_rows=GR, gate_up=True)
            s_b = hk.load_mx_scales(ws, 32, 4, e * GR + nb * 16, kc * 4,
                                    layout="aiter_gate_up", group_rows=GR)
            acc = hk.mma_ABt_mx(a, b, acc, s_a, s_b, b_fmt="fp4", into=acc)
        o[hk.elem_coord(0, e, 0, nb * 32)] = acc

    body.__name__ = name
    body.__annotations__ = {"x": hk.Tensor[fp8e4m3], "xs": hk.Tensor[hk.i8], "w": hk.Tensor[hk.i8],
                            "ws": hk.Tensor[hk.i8], "o": hk.Tensor[fp32]}
    return hk.kernel(body, arch="gfx950", warps=1, name=name,
                     grid=lambda p: (I // 16, p.o.depth, 1))


@pytest.mark.parametrize("unit_scales", [True, False])
def test_fp8_x_aiter_fp4_matches_dequant(unit_scales):
    torch.manual_seed(0)
    E, I, K, T = 2, 64, 512, 16
    GR = 2 * I
    w = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    ws = torch.randint(118, 136, (E * 2 * I, K // 32), dtype=torch.uint8)
    x = (torch.randn(T, K) * 2).to(torch.float8_e4m3fn)
    xs = torch.randint(122, 132, (T, K // 32), dtype=torch.uint8)
    if unit_scales:
        ws.fill_(127)
        xs.fill_(127)
    lut = torch.tensor(E2M1)
    b = w.long()
    wd = torch.stack([lut[b & 15], lut[b >> 4]], -1).reshape(E, 2 * I, K)
    wd = wd * torch.pow(2.0, ws.float() - 127).view(E, 2 * I, K // 32).repeat_interleave(32, -1)
    xd = x.float() * torch.pow(2.0, xs.float() - 127).repeat_interleave(32, 1)
    ref = torch.einsum("tk,enk->etn", xd, wd)                     # [E, T, 2I]: gate | up
    # hk's output: per 16-row block nb, its gate then its up 16 columns
    want = torch.cat([torch.cat([ref[:, :, nb * 16:nb * 16 + 16], ref[:, :, I + nb * 16:I + nb * 16 + 16]], -1)
                      for nb in range(I // 16)], -1)

    dev = "cuda"
    o = torch.full((1, E, T, 2 * I), float("nan"), device=dev)
    k = kernel(K, I, GR)
    k(x.to(dev), xs.to(dev).view(torch.int8),
      aiter_shuffle_w13(w).to(dev).view(torch.int8), aiter_shuffle_w13_scale(ws, E).to(dev).view(torch.int8), o)
    got = o[0].cpu()
    err = ((got - want).abs().max() / want.abs().max()).item()
    assert err < 1e-5, err


def kernel_t(K, I, GR, name="mx_fp4_aiter_t_t"):
    """The same product transposed: the FP4 weights as A (a_fmt), the FP8 tokens as B, so the gate
    and up blocks are accumulator rows (subtile-able) -- the form a SiLU-mul epilogue wants."""
    def body(x, xs, w, ws, o):
        nb = hk.arch.block_idx.x
        e = hk.arch.block_idx.y
        acc = hk.zeros(hk.RT(fp32, 32, 16, "col"))
        for kc in range(K // 128):
            a = hk.load_fp4_frags(hk.RT(fp8e4m3, 32, 128), w, e * GR + nb * 16, kc,
                                  group_rows=GR, gate_up=True)
            s_a = hk.load_mx_scales(ws, 32, 4, e * GR + nb * 16, kc * 4,
                                    layout="aiter_gate_up", group_rows=GR)
            b = x[hk.elem_coord(0, 0, 0, kc * 128)].load(hk.RT(fp8e4m3, 16, 128))
            s_b = hk.load_mx_scales(xs, 16, 4, 0, kc * 4)
            acc = hk.mma_ABt_mx(a, b, acc, s_a, s_b, a_fmt="fp4", into=acc)
        o[hk.elem_coord(0, e, nb * 32, 0)] = acc

    body.__name__ = name
    body.__annotations__ = {"x": hk.Tensor[fp8e4m3], "xs": hk.Tensor[hk.i8], "w": hk.Tensor[hk.i8],
                            "ws": hk.Tensor[hk.i8], "o": hk.Tensor[fp32]}
    return hk.kernel(body, arch="gfx950", warps=1, name=name,
                     grid=lambda p: (I // 16, p.o.depth, 1))


def test_aiter_fp4_as_a_operand():
    torch.manual_seed(1)
    E, I, K, T = 2, 64, 512, 16
    GR = 2 * I
    w = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
    ws = torch.randint(118, 136, (E * 2 * I, K // 32), dtype=torch.uint8)
    x = (torch.randn(T, K) * 2).to(torch.float8_e4m3fn)
    xs = torch.randint(122, 132, (T, K // 32), dtype=torch.uint8)
    lut = torch.tensor(E2M1)
    b = w.long()
    wd = torch.stack([lut[b & 15], lut[b >> 4]], -1).reshape(E, 2 * I, K)
    wd = wd * torch.pow(2.0, ws.float() - 127).view(E, 2 * I, K // 32).repeat_interleave(32, -1)
    xd = x.float() * torch.pow(2.0, xs.float() - 127).repeat_interleave(32, 1)
    ref = torch.einsum("tk,enk->ent", xd, wd)                     # [E, 2I, T]
    want = torch.cat([torch.cat([ref[:, nb * 16:nb * 16 + 16], ref[:, I + nb * 16:I + nb * 16 + 16]], 1)
                      for nb in range(I // 16)], 1)              # rows: gate nb, up nb, ...
    o = torch.full((1, E, 2 * I, T), float("nan"), device="cuda")
    kernel_t(K, I, GR)(x.cuda(), xs.cuda().view(torch.int8), aiter_shuffle_w13(w).cuda().view(torch.int8),
                       aiter_shuffle_w13_scale(ws, E).cuda().view(torch.int8), o)
    got = o[0].cpu()
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-5
