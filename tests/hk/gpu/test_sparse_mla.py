"""hk.ops.sparse_mla on gfx950: DeepSeek-V4.1's sparse MLA prefill against a float64 reference, the V = 1
probe, empty and one-row queries, determinism, and the MXFP8 rows against vLLM's inverse-RoPE quant
(bit for bit where vLLM is installed, the gfx950-vllm job; else a torch reference of its arithmetic)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's DMA gather and MFMA", allow_module_level=True)

from hk.ops import sparse_mla  # noqa: E402

pytestmark = pytest.mark.gpu

H, D, SCALE = 32, 512, 512 ** -0.5


def _case(lens, rows=3000, seed=0, sink=True, v1=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = len(lens)
    q = (torch.randn(T, H, D, generator=g) * 0.5).bfloat16().cuda()
    kv = torch.ones(rows, D) if v1 else torch.randn(rows, D, generator=g) * 0.5
    kv = kv.bfloat16().cuda()
    ptr = torch.zeros(T + 1, dtype=torch.int32)
    ptr[1:] = torch.tensor(lens).cumsum(0)
    # over-allocated past indptr[T], as vLLM's are
    ind = torch.randint(0, rows, (int(ptr[-1]) + 64,), generator=g, dtype=torch.int32)
    s = torch.randn(H, generator=g).float() if sink else torch.full((H,), float("-inf"))
    return q, kv, ind.cuda(), ptr.cuda(), s.cuda()


def _ref(q, kv, ind, ptr, sink):
    out = torch.zeros(q.shape, dtype=torch.float64)
    for t in range(q.shape[0]):
        a, b = int(ptr[t]), int(ptr[t + 1])
        if b == a:
            continue
        k = kv[ind[a:b].long()].double()
        s = (q[t].double() @ k.T) * SCALE
        mx = torch.maximum(s.max(dim=1).values, sink.double())
        p = torch.exp(s - mx[:, None])
        out[t] = (p @ k) / (p.sum(dim=1) + torch.exp(sink.double() - mx))[:, None]
    return out


LENS = [0, 1, 2, 31, 32, 33, 64, 65, 127, 128, 129, 300, 512, 600, 640] * 3


def test_matches_float64_and_is_deterministic():
    q, kv, ind, ptr, sink = _case(LENS)
    o1 = sparse_mla.prefill(q, kv, ind, ptr, sink, SCALE, torch.empty_like(q))
    o2 = sparse_mla.prefill(q, kv, ind, ptr, sink, SCALE, torch.full_like(q, float("nan")))
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)
    ref = _ref(q.cpu(), kv.cpu(), ind.cpu(), ptr.cpu(), sink.cpu())
    err = (o1.cpu().double() - ref).abs()
    assert torch.isfinite(o1).all()
    # each head's row within 4 bf16 ulps of its largest value, the whole within 1e-2
    rowmax = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    assert (err / rowmax).max() <= 4 * 2.0 ** -8, (err / rowmax).max()
    assert err.max() / ref.abs().max() < 1e-2
    assert (o1[torch.tensor(LENS) == 0] == 0).all()           # an empty query is 0


def test_v_one_is_exactly_one():
    q, kv, ind, ptr, sink = _case([1, 7, 64, 333, 640] * 4, v1=True, sink=False)
    o = sparse_mla.prefill(q, kv, ind, ptr, sink, SCALE, torch.empty_like(q))
    torch.cuda.synchronize()
    assert (o == 1).all()


def _vllm_quant():
    try:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_mxfp8_rows
    except Exception:  # noqa: BLE001
        return None
    return rocm_inverse_rope_mxfp8_rows


def _torch_quant(o, pos, cs):
    """vLLM's _inverse_rope_mxfp8_quant_kernel in torch: GPT-J inverse RoPE on lanes 448..511, then
    one E8M0 scale a 32 lanes."""
    x = o.float().clone()
    c, s = cs[pos][:, None, :32], cs[pos][:, None, 32:]
    ev, od = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2] = ev * c + od * s
    x[..., 449::2] = od * c - ev * s
    blocks = x.reshape(x.shape[0], -1, 32)
    amax = blocks.abs().amax(-1).clamp_min(1.1754943508222875e-38)
    bits = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
    qv = (blocks * torch.exp2(127.0 - bits)[..., None]).reshape(x.shape[0], -1)
    return qv.to(torch.float8_e4m3fn), bits.to(torch.uint8)


def test_mxfp8_rows_are_vllms_quant_of_prefills_bf16_rows():
    lens = [640, 128, 600, 1, 0, 300] * 6
    q, kv, ind, ptr, sink = _case(lens, seed=3)
    T = q.shape[0]
    pos = (torch.arange(T, device="cuda") * 37 + 5).to(torch.int64)
    ang = torch.arange(int(pos.max()) + 1, device="cuda", dtype=torch.float32)[:, None] * \
        (10000.0 ** (-torch.arange(32, device="cuda", dtype=torch.float32) / 32))[None]
    cs = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()
    md = torch.empty(T, H * D, dtype=torch.float8_e4m3fn, device="cuda")
    ms = torch.empty(T, H * D // 32, dtype=torch.uint8, device="cuda")
    sparse_mla.prefill_mxfp8(q, kv, ind, ptr, sink, SCALE, pos, cs, md, ms)
    # the bf16 rows prefill writes: the MXFP8 rows are made from the same O, in registers
    out = sparse_mla.prefill(q, kv, ind, ptr, sink, SCALE, torch.empty_like(q))
    torch.cuda.synchronize()
    vq = _vllm_quant()
    if vq is not None:
        rd, rs = torch.empty_like(md), torch.empty_like(ms)
        vq(out, pos, cs, 64, rd, rs)
        torch.cuda.synchronize()
        assert torch.equal(rd.view(torch.uint8), md.view(torch.uint8))
        assert torch.equal(rs, ms)
    else:
        rd, rs = _torch_quant(out, pos, cs)
        # torch's log2 and division need not round as Triton's do at a power of two
        assert (rs != ms).float().mean() < 1e-4
        assert (rd.view(torch.uint8) != md.view(torch.uint8)).float().mean() < 1e-3
