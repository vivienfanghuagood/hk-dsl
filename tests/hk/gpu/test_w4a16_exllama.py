"""hk.ops.w4a16.exllama_matmul: the small-batch form reading vLLM's RDNA skinny W4A16 layout in place
(ExLlama-shuffled codes, int32 [N, K/8] row-major; scales [N, K/group]), what vllm_gptq runs on
gfx1201's RDNAHybridW4A16LinearKernel layers.

Both forms that read it -- the small-batch form (`hk.gemv_w4(exllama=True)`, up to EXL_GEMV_MAX
tokens) and the WMMA form (`hk.load_w4(exllama=True)`, to 64 tokens): against a float64 reference
at 1-64 tokens, exact probes, 200 launches into NaN-filled outputs, every allocation end-fenced,
and no scratch or spill in any instantiation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.ops import w4a16  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    hk.target.current_arch() not in ("gfx1100", "gfx1201"),
    reason="hk.gemv_w4 is RDNA's (wave32); the in-place form is routed on gfx1201")]

LAUNCHES = 200
EXL = (0, 2, 4, 6, 1, 3, 5, 7)          # nibble i of a word holds column EXL[i]


def _skinny(q):
    """Codes [N, K] -> vLLM's skinny layout: int32 [N, K/8], nibble i of word w column 8w + EXL[i]."""
    n, k = q.shape
    g = q.to(torch.int64).reshape(n, k // 8, 8)
    words = sum(g[:, :, c] << (4 * i) for i, c in enumerate(EXL))
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).contiguous()


def _layer(n, k, group, g, q=None, s=None):
    """A symmetric W (zero point 8) in vLLM's layout, and the float64 W [N, K] it means."""
    if q is None:
        q = torch.randint(0, 16, (n, k), device="cuda", generator=g, dtype=torch.int32)
    if s is None:
        s = (torch.rand(n, k // group, device="cuda", generator=g) * 0.02 + 0.005).half()
    w = (q - 8).double() * s.double().repeat_interleave(group, 1)
    return _skinny(q), s.contiguous(), w


def _within_two_ulps_a_row(got, want):
    top = want.abs().amax(dim=1, keepdim=True)
    ulp = torch.pow(2.0, torch.floor(torch.log2(top.clamp_min(2.0 ** -14)))) * 2.0 ** -10
    return ((got.double() - want).abs() / ulp).max().item()


def test_the_layout_is_vllms():
    """_skinny is vLLM's pack_skinny_int4, where this vLLM has it."""
    rh = pytest.importorskip("vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16")
    q = torch.randint(0, 16, (32, 256), device="cuda", dtype=torch.int32)
    assert torch.equal(rh.pack_skinny_int4(q).view(torch.int32), _skinny(q))


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("m", [1, 3, 4, 5, 8, 12, 16, 17, 31, 32, 33, 48, 64])
def test_within_two_ulps_a_row_of_float64(m, group):
    g = torch.Generator(device="cuda").manual_seed(500 + 10 * m + group)
    n, k = 272, 2048
    qw, s, w = _layer(n, k, group, g)
    assert w4a16.exllama_takes(m, n, k, group)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    got = w4a16.exllama_matmul(x, qw, s, group)
    assert _within_two_ulps_a_row(got, x.double() @ w.t()) <= 2


def test_exact_probes():
    """Codes all 9 with scales 1 (W = 1) and x all 1: every output is exactly K. Codes all 8: W = 0,
    exactly 0."""
    n, k, group = 64, 1024, 32
    for code, want in ((9, float(k)), (8, 0.0)):
        q = torch.full((n, k), code, device="cuda", dtype=torch.int32)
        qw, s, _ = _layer(n, k, group, None, q=q,
                          s=torch.ones(n, k // group, device="cuda", dtype=torch.float16))
        for m in (1, 4, 7, 9, 32, 64):
            x = torch.ones(m, k, device="cuda", dtype=torch.float16)
            got = w4a16.exllama_matmul(x, qw, s, group)
            assert torch.equal(got.cpu(), torch.full((m, n), want, dtype=torch.float16))


def test_the_same_bits_on_every_launch_into_nan_filled_outputs():
    g = torch.Generator(device="cuda").manual_seed(13)
    n, k, group = 5120, 6144, 32
    qw, s, _ = _layer(n, k, group, g)
    for m in (3, 8, 13, 32, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
        first, differ, nan = None, 0, 0
        for _ in range(LAUNCHES):
            out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
            w4a16.exllama_matmul(x, qw, s, group, out=out)
            torch.cuda.synchronize()
            nan += bool(out.isnan().any())
            if first is None:
                first = out.clone()
            else:
                differ += not torch.equal(out, first)
        assert (nan, differ) == (0, 0), (m, nan, differ)


def test_every_allocation_end_fenced():
    """The last row of x and of W end flush against an unmapped page."""
    from _fence import fenced

    fenced('''
    from hk.ops import w4a16
    g = torch.Generator(device="cuda").manual_seed(3)
    for n, k, group in ((16, 1024, 32), (48, 2048, 128)):
        qw = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), device="cuda", generator=g,
                           dtype=torch.int64).to(torch.int32)
        s = torch.rand(n, k // group, device="cuda", generator=g).half()
        for m in (1, 8, 9, 64):
            x = torch.randn(m, k, device="cuda", generator=g).half()
            y = step(f"exllama N{n} K{k} M{m}", lambda: w4a16.exllama_matmul(x, qw, s, group))
            assert not y.isnan().any()
    ''')


def test_the_wmma_form_runs_from_past_exl_gemv_max_to_64():
    """Past EXL_GEMV_MAX tokens exllama_matmul is one WMMA launch at the decode schedules."""
    n, k, group = 512, 2048, 32
    assert all(w4a16.takes(m, n, k, group) for m in range(w4a16.EXL_GEMV_MAX + 1, 65))
    qw, s, w = _layer(n, k, group, torch.Generator(device="cuda").manual_seed(9))
    for m in (w4a16.EXL_GEMV_MAX + 1, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w4a16.exllama_matmul(x, qw, s, group)
    assert any(key[0] == "exl-wmma" for key in w4a16._OPS)


def test_no_instantiation_spills_or_uses_scratch():
    for t, sch in w4a16.SCHEDULES.items():
        for group in (32, 64, 128):
            slices = min(sch["slices"], group // 16)
            b = w4a16.kernel_for(t, slices, sch["warps"], group, sch["blocks"], "fp16", True,
                                 True).build()
            for r in b.kernels:
                assert (r.scratch, r.vgpr_spill) == (0, 0), ("wmma", t, group, r)
    for tokens in range(1, w4a16.EXL_CHUNK + 1):
        for group in (32, 64, 128):
            b = w4a16.w4a16_gemv_kernel(f"exl_t{tokens}_g{group}", tokens=tokens,
                                        warps=w4a16.GEMV_WARPS, group=group, symmetric=True,
                                        exllama=True).build()
            for r in b.kernels:
                assert (r.scratch, r.vgpr_spill) == (0, 0), (tokens, group, r)
