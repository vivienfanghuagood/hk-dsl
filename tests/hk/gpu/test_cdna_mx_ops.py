"""gfx950's MX / FP4 operand ops, each in a small kernel against torch (hk.mx's reference layouts
and quantization): scale layouts (bytes, packed dwords, AITER's 32-row and gate|up layouts), scale
offsets, FP4 operands from AITER's shuffled MoE weights, and the MX-quantizing store."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp8e4m3, fp32, i8, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled f8f6f4 MFMA", allow_module_level=True)

pytestmark = pytest.mark.gpu


def _rel(out, ref):
    return ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def _fp8(*shape, g):
    return (torch.randn(*shape, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)


def _scales(*shape, g):
    return torch.randint(124, 131, shape, device="cuda", dtype=torch.uint8, generator=g)


# -- scale layouts ------------------------------------------------------------------------


def _packed_kernel(K):
    def mx_packed(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i32],
                  sb: hk.Tensor[i32], c: hk.Tensor[fp32]):
        ta = a[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 64, K))
        tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 64, K))
        s_a = hk.load_mx_scales(sa, 64, K // 32, 0, 0)
        s_b = hk.load_mx_scales(sb, 64, K // 32, 0, 0)
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma_ABt_mx(ta, tb, hk.zeros(hk.RT(fp32, 64, 64, "col")),
                                                     s_a, s_b)

    return hk.kernel(mx_packed, arch="gfx950", warps=1, name=f"mx_packed_{K}",
                     grid=lambda p: (1, 1, 1))


def test_packed_dword_scales_give_the_scaled_product():
    g = torch.Generator(device="cuda").manual_seed(2)
    K = 256
    a, b = _fp8(64, K, g=g), _fp8(64, K, g=g)
    sa, sb = _scales(64, K // 32, g=g), _scales(64, K // 32, g=g)
    c = torch.zeros(64, 64, device="cuda")
    _packed_kernel(K)(a, b, hk.mx.pack_scales(sa), hk.mx.pack_scales(sb), c)
    ref = hk.mx.dequant(a, sa) @ hk.mx.dequant(b, sb).T
    assert _rel(c, ref) < 1e-4


def _aiter_offsets_kernel():
    """B's scales in AITER's 32-row layout, loaded once for two K steps; two 16-row A tiles
    take them at sa_at offsets -- the two-step dwords of a 32-row block."""
    def mx_aiter(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i8],
                 sb: hk.Tensor[i8], c: hk.Tensor[fp32]):
        s_a = hk.load_mx_scales(sa, 32, 8, 0, 0, layout="aiter", group_rows=32)
        s_b = hk.load_mx_scales(sb, 32, 8, 0, 0, layout="aiter", group_rows=32)
        tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 32, 128))
        tb2 = b[hk.elem_coord(0, 0, 0, 128)].load(hk.RT(fp8e4m3, 32, 128))
        for t in range(2):
            acc = hk.zeros(hk.RT(fp32, 16, 32, "col"))
            ta = a[hk.elem_coord(0, 0, 16 * t, 0)].load(hk.RT(fp8e4m3, 16, 128))
            acc = hk.mma_ABt_mx(ta, tb, acc, s_a, s_b, sa_at=(t, 0), sb_at=(0, 0))
            ta2 = a[hk.elem_coord(0, 0, 16 * t, 128)].load(hk.RT(fp8e4m3, 16, 128))
            acc = hk.mma_ABt_mx(ta2, tb2, acc, s_a, s_b, sa_at=(t, 1), sb_at=(0, 1))
            c[hk.elem_coord(0, 0, 16 * t, 0)] = acc

    return hk.kernel(mx_aiter, arch="gfx950", warps=1, name="mx_aiter_offsets",
                     grid=lambda p: (1, 1, 1))


def test_aiter_scales_at_tile_and_step_offsets_give_the_scaled_product():
    g = torch.Generator(device="cuda").manual_seed(3)
    a, b = _fp8(32, 256, g=g), _fp8(32, 256, g=g)
    sa, sb = _scales(32, 8, g=g), _scales(32, 8, g=g)
    c = torch.zeros(32, 32, device="cuda")
    _aiter_offsets_kernel()(a, b, hk.mx.aiter_scales(sa).view(torch.int8),
                            hk.mx.aiter_scales(sb).view(torch.int8), c)
    ref = hk.mx.dequant(a, sa) @ hk.mx.dequant(b, sb).T
    assert _rel(c, ref) < 1e-4


# -- FP4 operands from AITER's shuffled MoE weights ---------------------------------------


def _fp4_kernel(gate_up, n, K):
    lay = "aiter_gate_up" if gate_up else "aiter"

    def mx_fp4(w: hk.Tensor[i8], ws: hk.Tensor[i8], x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8],
               c: hk.Tensor[fp32]):
        tw = hk.load_fp4_frags(hk.RT(fp8e4m3, n, K), w, 0, 0, group_rows=n, gate_up=gate_up)
        s_w = hk.load_mx_scales(ws, n, K // 32, 0, 0, layout=lay, group_rows=n)
        tx = x[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, K))
        s_x = hk.load_mx_scales(xs, 16, K // 32, 0, 0)
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma_ABt_mx(tw, tx, hk.zeros(hk.RT(fp32, n, 16, "col")),
                                                     s_w, s_x, a_fmt="fp4")

    return hk.kernel(mx_fp4, arch="gfx950", warps=1, name=f"mx_fp4_{int(gate_up)}_{n}_{K}",
                     grid=lambda p: (1, 1, 1))


@pytest.mark.parametrize("gate_up", [False, True], ids=["plain", "gate_up"])
def test_fp4_weights_in_aiters_layout_give_the_scaled_product(gate_up):
    g = torch.Generator(device="cuda").manual_seed(4 + gate_up)
    n, K = 32, 256
    w = torch.randint(0, 256, (1, n, K // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws = _scales(n, K // 32, g=g)
    x, xs = _fp8(16, K, g=g), _scales(16, K // 32, g=g)
    lay_s = hk.mx.aiter_gate_up_scales(ws, 1) if gate_up else hk.mx.aiter_scales(ws)
    c = torch.zeros(n, 16, device="cuda")
    _fp4_kernel(gate_up, n, K)(hk.mx.aiter_fp4_weights(w, gate_up).view(torch.int8),
                               lay_s.view(torch.int8), x, xs.view(torch.int8), c)
    wd = hk.mx.fp4_values(w[0]) * torch.pow(2.0, ws.float() - 127).repeat_interleave(32, -1)
    if gate_up:     # base tiles alternate gate and up of each 16-row block
        h = n // 2
        wd = torch.stack([wd[:h].view(-1, 16, K), wd[h:].view(-1, 16, K)], 1).reshape(n, K)
    ref = wd @ hk.mx.dequant(x, xs).T
    assert _rel(c, ref) < 1e-4


def test_an_expert_past_the_weights_reads_zero_not_another_experts_memory():
    """load_fp4_frags bases one buffer resource on each expert, in 64 bits (an MoE's weights
    pass 4 GB): an expert past the tensor gets an empty range."""
    g = torch.Generator(device="cuda").manual_seed(6)
    n, K = 32, 256
    w = torch.randint(0, 256, (1, n, K // 2), device="cuda", dtype=torch.uint8, generator=g)

    def mx_fp4_oob(w: hk.Tensor[i8], ws: hk.Tensor[i8], x: hk.Tensor[fp8e4m3],
                   xs: hk.Tensor[i8], c: hk.Tensor[fp32]):
        tw = hk.load_fp4_frags(hk.RT(fp8e4m3, n, K), w, n, 0, group_rows=n)   # expert 1 of 1
        s_w = hk.load_mx_scales(ws, n, K // 32, 0, 0, layout="aiter", group_rows=n)
        tx = x[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, K))
        s_x = hk.load_mx_scales(xs, 16, K // 32, 0, 0)
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma_ABt_mx(tw, tx, hk.zeros(hk.RT(fp32, n, 16, "col")),
                                                     s_w, s_x, a_fmt="fp4")

    k = hk.kernel(mx_fp4_oob, arch="gfx950", warps=1, name="mx_fp4_oob", grid=lambda p: (1, 1, 1))
    c = torch.full((n, 16), float("nan"), device="cuda")
    k(hk.mx.aiter_fp4_weights(w).view(torch.int8),
      hk.mx.aiter_scales(_scales(n, K // 32, g=g)).view(torch.int8), _fp8(16, K, g=g),
      _scales(16, K // 32, g=g).view(torch.int8), c)
    assert torch.equal(c, torch.zeros_like(c))


# -- the MX-quantizing store --------------------------------------------------------------


def _mxq_kernel(R, C):
    def mxq(x: hk.Tensor[bf16], q: hk.Tensor[fp8e4m3], s: hk.Tensor[i8]):
        g = hk.group(4)
        st = hk.lds.alloc(hk.ST(bf16, R, C), name="st")
        r0, c0 = hk.arch.block_idx.x * R, hk.arch.block_idx.y * C
        g.load(st, x.at[r0, c0])
        hk.arch.barrier()
        hk.mx_quant_store(st, q, s, r0, c0)

    return hk.kernel(mxq, arch="gfx950", warps=4, name=f"mxq_{R}_{C}",
                     grid=lambda p: (p.x.rows // R, p.x.cols // C, 1))


@pytest.mark.parametrize("cols", [192, 1152])
def test_the_mx_quantizing_store_is_bit_for_bit_the_reference(cols):
    g = torch.Generator(device="cuda").manual_seed(cols)
    rows = 128
    x = (torch.randn(rows, cols, device="cuda", generator=g)
         * torch.rand(rows, 1, device="cuda", generator=g) * 50).to(torch.bfloat16)
    x[3, :32] = 0                                                    # an all-zero group
    q = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
    s = torch.empty(rows, -(-cols // 256) * 8, device="cuda", dtype=torch.uint8)
    _mxq_kernel(64, 192)(x, q, s.view(torch.int8))      # a workgroup a 64 x 192 block
    q_ref, s_ref = hk.mx.quant(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    real = hk.mx.aiter_scales(torch.ones_like(s_ref)) == 1           # pad bytes are not written
    assert torch.equal(s[real], hk.mx.aiter_scales(s_ref)[real])


def test_fp4_weights_from_an_lds_tile_match_the_global_read():
    """load_fp4_frags_lds: the weights' raw bytes copied into LDS (the [E * N/16, K/2 * 16] view,
    a row a 16-row block, 1 KB a 128-wide K step), fragment (i, s) from row row0 + i."""
    g = torch.Generator(device="cuda").manual_seed(9)
    n, K = 256, 256

    def mx_fp4_lds(wv: hk.Tensor[fp8e4m3], ws: hk.Tensor[i8], x: hk.Tensor[fp8e4m3],
                   xs: hk.Tensor[i8], c: hk.Tensor[fp32]):
        grp = hk.group(1)
        sw = hk.lds.alloc(hk.ST(fp8e4m3, 16, 1024 * (K // 128)), name="sw")
        grp.load(sw, wv.at[0, 0])
        hk.arch.barrier()
        tw = hk.load_fp4_frags_lds(hk.RT(fp8e4m3, 32, K), sw, 2, 0)        # rows 32..63
        s_w = hk.load_mx_scales(ws, 32, K // 32, 32, 0, layout="aiter", group_rows=n)
        tx = x[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, K))
        s_x = hk.load_mx_scales(xs, 16, K // 32, 0, 0)
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma_ABt_mx(tw, tx, hk.zeros(hk.RT(fp32, 32, 16, "col")),
                                                     s_w, s_x, a_fmt="fp4")

    k = hk.kernel(mx_fp4_lds, arch="gfx950", warps=1, name="mx_fp4_lds", grid=lambda p: (1, 1, 1))
    w = torch.randint(0, 256, (1, n, K // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws, x, xs = _scales(n, K // 32, g=g), _fp8(16, K, g=g), _scales(16, K // 32, g=g)
    wv = hk.mx.aiter_fp4_weights(w).reshape(n // 16, -1).view(torch.float8_e4m3fn)
    c = torch.zeros(32, 16, device="cuda")
    k(wv, hk.mx.aiter_scales(ws).view(torch.int8), x, xs.view(torch.int8), c)
    wd = hk.mx.fp4_values(w[0, 32:64]) * torch.pow(2.0, ws[32:64].float() - 127).repeat_interleave(32, -1)
    assert _rel(c, wd @ hk.mx.dequant(x, xs).T) < 1e-4
