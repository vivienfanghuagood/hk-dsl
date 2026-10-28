"""HipKittens' CDNA4 GEMM and GQA forward on an MI355X: right against fp32,
inside the shapes they take, and the engine ops around them.

Which hipcc built them decides whether there is anything to test. On the
MI355X the GEMM is right built by ROCm 7.2 and wrong built by 7.1, and the
attention kernels build only under 7.1. So a kernel this toolchain cannot
have is a skip that says so, and the check that keeps a miscompiled kernel
out is tested directly. Run the tier under both, HK_HIPCC choosing:

    HK_ARCH=gfx950 python3 -m pytest tests/hk/gpu/test_hipkittens.py
    HK_ARCH=gfx950 HK_HIPCC=/opt/rocm-7.1.0/bin/hipcc python3 -m pytest tests/hk/gpu/test_hipkittens.py
"""

from __future__ import annotations

import math
import warnings

import pytest

torch = pytest.importorskip("torch")

from hk.integration import engine, vllm_gemm  # noqa: E402
from hk.ops import hipkittens as hkk  # noqa: E402

import _attn_ref  # noqa: E402


def _arch():
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001
        return ""


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or _arch() != "gfx950",
    reason="needs an MI350X/MI355X (gfx950)")

TOL = 2e-2


def rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


def _gemm_or_skip():
    a = torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16)
    try:
        hkk.gemm_bf16(a, a)
    except hkk.Unavailable as e:
        pytest.skip(f"under this hipcc: {e}")


def _gqa_or_skip(h, h_kv, n, d, causal):
    q = torch.zeros(1, n, h, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(1, n, h_kv, d, device="cuda", dtype=torch.bfloat16)
    try:
        hkk.gqa_fwd(q, kv, kv, causal=causal)
    except hkk.Unavailable as e:
        pytest.skip(f"under this hipcc: {e}")


def _wx(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return x, w


@pytest.fixture(scope="module", autouse=True)
def ops():
    hkk.register_ops()
    vllm_gemm.register_op()


# ---------------------------------------------------------------- GEMM


@pytest.mark.parametrize("m,n,k", [(256, 256, 128), (512, 1024, 256), (2048, 4096, 4096),
                                   (1024, 6144, 4096), (768, 4096, 14336), (256, 28672, 4096)])
def test_the_gemm_is_x_wt(m, n, k):
    _gemm_or_skip()
    x, w = _wx(m, n, k)
    y = hkk.gemm_bf16(x, w)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    assert rel(y, x.float() @ w.float().t()) < TOL
    assert torch.equal(y, hkk.gemm_bf16(x, w))


def test_the_gemm_writes_c_and_nothing_past_it():
    _gemm_or_skip()
    x, w = _wx(512, 512, 256)
    buf = torch.full((512 * 512 + 4096,), 7.0, device="cuda", dtype=torch.bfloat16)
    out = buf[:512 * 512].view(512, 512)
    hkk.gemm_bf16(x, w, out=out)
    assert rel(out, x.float() @ w.float().t()) < TOL
    assert bool((buf[512 * 512:] == 7.0).all())


@pytest.mark.parametrize("m,n,k", [(1000, 4096, 4096), (4096, 4096, 4160), (256, 151936, 4096)])
def test_a_shape_it_cannot_take_raises_and_the_op_falls_back(m, n, k):
    x, w = _wx(m, n, k)
    with pytest.raises(hkk.Unsupported):
        hkk.gemm_bf16(x, w)
    before = dict(hkk.GEMM_OP.stats)
    y = torch.ops.hk_cdna.gemm_bf16(x, w)
    assert hkk.GEMM_OP.stats["fallback"] == before["fallback"] + 1
    assert rel(y, x.float() @ w.float().t()) < TOL


def test_a_strided_or_misaligned_tensor_is_refused():
    _gemm_or_skip()
    x, w = _wx(512, 512, 512)
    with pytest.raises(hkk.Unsupported, match="contiguous"):
        hkk.gemm_bf16(x.t(), w)
    flat = torch.empty(512 * 512 + 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(hkk.Unsupported, match="aligned"):
        hkk.gemm_bf16(flat[1:].view(512, 512), w)


def test_the_op_takes_the_kernel_and_counts_it():
    _gemm_or_skip()
    x, w = _wx(1024, 4096, 4096)
    before = dict(hkk.GEMM_OP.stats)
    y = torch.ops.hk_cdna.gemm_bf16(x, w)
    assert hkk.GEMM_OP.stats["hipkittens"] == before["hipkittens"] + 1
    assert rel(y, x.float() @ w.float().t()) < TOL


def test_the_op_compiles_fullgraph():
    _gemm_or_skip()
    x, w = _wx(512, 1024, 512)
    f = torch.compile(lambda a, b: torch.ops.hk_cdna.gemm_bf16(a, b) * 2, fullgraph=True)
    assert rel(f(x, w), 2 * (x.float() @ w.float().t())) < TOL


def test_the_op_replays_from_a_cuda_graph():
    _gemm_or_skip()
    x, w = _wx(1024, 4096, 4096)
    torch.ops.hk_cdna.gemm_bf16(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = torch.ops.hk_cdna.gemm_bf16(x, w)
    x.copy_(_wx(1024, 4096, 4096, seed=1)[0])
    graph.replay()
    torch.cuda.synchronize()
    assert rel(y, x.float() @ w.float().t()) < TOL


def test_a_kernel_that_fails_its_check_is_refused_and_the_op_falls_back(monkeypatch):
    # The check is what keeps the ROCm 7.1 GEMM out; force a failure on a
    # build that exists, through the real path.
    _gemm_or_skip()
    monkeypatch.setattr(hkk, "_gemm_check", lambda mod, dev: "forced by the test")
    monkeypatch.setattr(hkk, "_CHECKED", {})
    x, w = _wx(512, 512, 512)
    with pytest.raises(hkk.Miscompiled, match="forced by the test"):
        hkk.gemm_bf16(x, w)
    before = dict(hkk.GEMM_OP.stats)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        y = torch.ops.hk_cdna.gemm_bf16(x, w)
        torch.ops.hk_cdna.gemm_bf16(x, w)
    assert hkk.GEMM_OP.stats["fallback"] == before["fallback"] + 2
    assert sum("forced by the test" in str(m.message) for m in ws) == 1
    assert rel(y, x.float() @ w.float().t()) < TOL


def test_the_first_launch_under_capture_is_refused_not_frozen(monkeypatch):
    _gemm_or_skip()
    monkeypatch.setattr(hkk, "_CHECKED", {})
    x, w = _wx(512, 512, 512)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(engine.GraphCaptureError, match="run this shape once"):
        with torch.cuda.graph(graph):
            hkk.gemm_bf16(x, w)


@pytest.mark.parametrize("m", [2048, 2047, 3840, 4096])
def test_vllms_linear_on_the_gfx950_table(m):
    _gemm_or_skip()
    n, k = 4096, 4096
    vllm_gemm.set_routes(engine.shipped_table("vllm_linear", "gfx950", "HK_TEST_NO_SUCH_ENV"))
    try:
        path = vllm_gemm.route_for(m, n, k)
        x, w = _wx(m, n, k)
        before = dict(vllm_gemm.STATS)
        y = torch.ops.hk_vllm.linear(x.view(1, m, k), w, None)
        took = {p: vllm_gemm.STATS[p] - before[p] for p in before}
        assert took == {p: int(p == path) for p in took}
        assert y.shape == (1, m, n)
        assert rel(y[0], x.float() @ w.float().t()) < TOL
    finally:
        vllm_gemm.set_routes(None)
    assert path == ("hipkittens" if m in (2048, 3840) else "fallback")


# ---------------------------------------------------------------- GQA


def _qkv(b, n, h, h_kv, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(b, n, h, d, device="cuda", dtype=torch.bfloat16, generator=g),
            torch.randn(b, n, h_kv, d, device="cuda", dtype=torch.bfloat16, generator=g),
            torch.randn(b, n, h_kv, d, device="cuda", dtype=torch.bfloat16, generator=g))


def _gqa_ref(q, k, v, causal):
    """(out, lse) in float64, out as float32: (B, N, H, D) operands."""
    b, n, h, d = q.shape
    g = h // k.shape[2]
    qf = q.double().transpose(1, 2)
    kf = k.double().transpose(1, 2).repeat_interleave(g, dim=1)
    vf = v.double().transpose(1, 2).repeat_interleave(g, dim=1)
    s = qf @ kf.transpose(-1, -2) / math.sqrt(d)
    if causal:
        s = s.masked_fill(torch.ones(n, n, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    return (torch.softmax(s, -1) @ vf).transpose(1, 2).float(), torch.logsumexp(s, -1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,n,h,h_kv,d", [(2, 256, 8, 2, 128), (2, 1024, 16, 2, 128),
                                          (3, 512, 8, 8, 128), (2, 512, 16, 4, 64)])
def test_gqa_is_softmax_qk_v(b, n, h, h_kv, d, causal):
    _gqa_or_skip(h, h_kv, n, d, causal)
    q, k, v = _qkv(b, n, h, h_kv, d)
    o, lse = hkk.gqa_fwd(q, k, v, causal=causal, return_lse=True)
    want_o, want_lse = _gqa_ref(q, k, v, causal)
    assert o.shape == q.shape and lse.shape == (b, h, n)
    _attn_ref.check(o, want_o)
    # the log-sum-exp is fp32 from bf16 scores: at most 3.4e-3 off on gfx950
    assert (lse.double() - want_lse).abs().max().item() < 1e-2
    assert torch.equal(o, hkk.gqa_fwd(q, k, v, causal=causal))


def test_gqa_one_build_serves_every_batch():
    _gqa_or_skip(8, 2, 256, 128, False)
    n_builds = len(hkk._MODULES)
    for b in (1, 5):
        q, k, v = _qkv(b, 256, 8, 2, 128, seed=b)
        _attn_ref.check(hkk.gqa_fwd(q, k, v), _gqa_ref(q, k, v, False)[0])
    assert len(hkk._MODULES) == n_builds


def test_gqa_op_falls_back_where_the_kernel_cannot_go():
    q, k, v = _qkv(1, 320, 8, 2, 128)                # N not a multiple of 256
    before = dict(hkk.GQA_OP.stats)
    o = torch.ops.hk_cdna.gqa_fwd(q, k, v, True)
    assert hkk.GQA_OP.stats["fallback"] == before["fallback"] + 1
    _attn_ref.check(o, _gqa_ref(q, k, v, True)[0])


def test_gqa_op_takes_the_kernel_and_replays_from_a_cuda_graph():
    _gqa_or_skip(16, 2, 512, 128, True)
    q, k, v = _qkv(2, 512, 16, 2, 128)
    before = dict(hkk.GQA_OP.stats)
    torch.ops.hk_cdna.gqa_fwd(q, k, v, True)
    assert hkk.GQA_OP.stats["hipkittens"] == before["hipkittens"] + 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = torch.ops.hk_cdna.gqa_fwd(q, k, v, True)
    q.copy_(_qkv(2, 512, 16, 2, 128, seed=3)[0])
    graph.replay()
    torch.cuda.synchronize()
    _attn_ref.check(o, _gqa_ref(q, k, v, True)[0])
