"""The vllm_sparse_mla route against vLLM's own functions, in the vLLM image (gfx950-vllm): the patched
rocm_sparse_attn_prefill against vLLM's (opus) on the same dense indices, the MXFP8 rows written in the
kernel against vLLM's quant of the same bf16 rows, the rows left to vLLM's quant when a chunk falls back,
and revert. No model: the functions are called as _forward_prefill calls them."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950", allow_module_level=True)
ops = pytest.importorskip("vllm.v1.attention.ops.rocm_aiter_mla_sparse")
pytest.importorskip("vllm.models.deepseek_v41.amd.rocm")

from hk.integration import vllm_sparse_mla as R  # noqa: E402

pytestmark = pytest.mark.gpu
H, D, SCALE = 32, 512, 512 ** -0.5


@pytest.fixture
def patched():
    msg = R.apply_patch()
    assert msg.startswith("sparse_mla -> hk"), msg
    yield
    R.revert()


def _dense(T, rows=4096, width=640, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(T, H, D, generator=g) * 0.5).bfloat16().cuda()
    kv = (torch.randn(rows, 1, D, generator=g) * 0.5).bfloat16().cuda()
    lens = torch.randint(1, width + 1, (T,), generator=g)
    ind = torch.randint(0, rows, (T, width), generator=g, dtype=torch.int32)
    ind[torch.arange(width)[None, :] >= lens[:, None]] = -1
    sink = torch.randn(H, generator=g).float().cuda()
    return q, kv, ind.cuda(), lens.int().cuda(), sink


def _call(q, kv, ind, lens, sink, out):
    ops_mod = __import__("vllm.models.deepseek_v41.amd.rocm", fromlist=["x"])
    ops_mod.rocm_sparse_attn_prefill(q=q, kv=kv, indices=ind, topk_length=lens, scale=SCALE, head_dim=D,
                                     nope_head_dim=448, rope_head_dim=64, attn_sink=sink, output=out)


def test_the_route_takes_a_large_call_and_matches_opus(patched):
    q, kv, ind, lens, sink = _dense(1024)
    ours = torch.empty_like(q)
    R.PATCH.reset()
    _call(q, kv, ind, lens, sink, ours)
    assert R.STATS["hk"] == 1
    R.ENABLED = False
    try:
        ref = torch.empty_like(q)
        _call(q, kv, ind, lens, sink, ref)
    finally:
        R.ENABLED = True
    torch.cuda.synchronize()
    err = (ours.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 1e-2, err


def test_a_small_call_stays_vllms(patched):
    q, kv, ind, lens, sink = _dense(R.MIN_T // 4)
    R.PATCH.reset()
    _call(q, kv, ind, lens, sink, torch.empty_like(q))
    assert R.STATS["fallback"] == 1 and R.STATS["hk"] == 0


def test_mxfp8_rows_in_the_kernel_and_the_rest_by_vllm(patched):
    """Two chunk calls of one layer: the first hk's (MXFP8 rows written in the kernel, no bf16 rows),
    the second too small (vLLM's, so its rows go to vLLM's pass); every row must equal vLLM's quant
    of the bf16 rows hk's and vLLM's kernels make."""
    T1, T2 = 1024, 64
    q1, kv, ind1, lens1, sink = _dense(T1, seed=1)
    q2, _, ind2, lens2, _ = _dense(T2, seed=2)
    q = torch.cat([q1, q2])
    pos = torch.arange(T1 + T2, device="cuda", dtype=torch.int64) * 3 + 11
    ang = torch.arange(int(pos.max()) + 1, device="cuda", dtype=torch.float32)[:, None] * \
        (10000.0 ** (-torch.arange(32, device="cuda", dtype=torch.float32) / 32))[None]
    cs = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()
    out = torch.empty_like(q)
    md = torch.zeros(T1 + T2, H * D, dtype=torch.float8_e4m3fn, device="cuda")
    ms = torch.zeros(T1 + T2, H * D // 32, dtype=torch.uint8, device="cuda")
    R._CTX.layer = R._Layer(q, pos, cs, md, ms)
    try:
        _call(q[:T1], kv, ind1, lens1, sink, out[:T1])
        _call(q[T1:], kv, ind2, lens2, sink, out[T1:])
        rocm = __import__("vllm.models.deepseek_v41.amd.rocm", fromlist=["x"])
        rocm.rocm_inverse_rope_mxfp8_rows(out, pos, cs, 64, md, ms)
    finally:
        R._CTX.layer = None
    torch.cuda.synchronize()
    # the hk chunk left its bf16 rows unwritten (vLLM's are scratch): hk's own, for the reference
    ref = out.clone()
    R._CTX.layer = None
    _call(q[:T1], kv, ind1, lens1, sink, ref[:T1])
    torch.cuda.synchronize()
    rd, rs = torch.empty_like(md), torch.empty_like(ms)
    R._ORIG["quant"](ref, pos, cs, 64, rd, rs)
    torch.cuda.synchronize()
    assert torch.equal(rd.view(torch.uint8), md.view(torch.uint8))
    assert torch.equal(rs, ms)


def test_revert_restores_vllms_functions():
    msg = R.apply_patch()
    assert msg.startswith("sparse_mla -> hk"), msg
    rocm = __import__("vllm.models.deepseek_v41.amd.rocm", fromlist=["x"])
    assert rocm.rocm_sparse_attn_prefill is R.rocm_sparse_attn_prefill
    R.revert()
    assert rocm.rocm_sparse_attn_prefill is ops.rocm_sparse_attn_prefill
    assert rocm.DeepseekV41ROCMAiterMLAAttention._forward_prefill is not R._forward_prefill


def test_the_contract_holds_and_catches_a_resigned_entry_point(monkeypatch):
    from hk.integration import contract

    assert contract.check(["vllm_sparse_mla"]) == []

    def rocm_sparse_attn_prefill(q, kv, indices, topk_length, scale, head_dim, attn_sink, output):
        raise AssertionError("not called")

    monkeypatch.setattr(ops, "rocm_sparse_attn_prefill", rocm_sparse_attn_prefill)
    got = contract.check(["vllm_sparse_mla"])
    assert len(got) == 1 and got[0].startswith("vllm_sparse_mla: vLLM's prefill entry point takes")
    assert R.apply_patch().startswith("sparse_mla: not patched")


def test_the_plugin_applies_it_by_default_and_not_at_0(monkeypatch):
    """On by default on gfx950 (the owner's decision, 2026-10-29); HK_VLLM_SPARSE_MLA=0 is the off
    switch."""
    from hk.integration import _vllm_plugin, switches

    for e in ["HK_DISABLE", "HK_PATCH_VLLM", "HK_VLLM_USER", *[r.env for r in switches.routes("vllm")]]:
        monkeypatch.delenv(e, raising=False)
    rocm = __import__("vllm.models.deepseek_v41.amd.rocm", fromlist=["x"])
    try:
        for r in switches.routes("vllm"):          # every other route off: nothing else to revert
            if r.name != "vllm_sparse_mla":
                monkeypatch.setenv(r.env, "0")
        monkeypatch.setenv("HK_VLLM_SPARSE_MLA", "0")
        _vllm_plugin.register()
        assert rocm.rocm_sparse_attn_prefill is ops.rocm_sparse_attn_prefill      # off at 0
        monkeypatch.delenv("HK_VLLM_SPARSE_MLA")
        _vllm_plugin.register()
        assert rocm.rocm_sparse_attn_prefill is R.rocm_sparse_attn_prefill        # on by default
    finally:
        R.revert()


def test_an_mxfp8_destination_of_another_layout_is_left_to_vllm(patched):
    """A destination prefill_mxfp8 does not write (a strided scale view here) is refused, so
    _forward_prefill notes no layer: the chunks run hk's bf16 kernel and vLLM's pass quantizes them."""
    import torch as T_

    n = 8
    md = T_.zeros(n, H * D, dtype=T_.float8_e4m3fn, device="cuda")
    ms = T_.zeros(n, H * D // 16, dtype=T_.uint8, device="cuda")[:, ::2]
    lay = R._Layer(T_.zeros(n, H, D, dtype=T_.bfloat16, device="cuda"),
                   T_.arange(n, device="cuda"), T_.zeros(4, 64, device="cuda"), md, ms)
    assert not R._mx_layout_ok(lay)
    lay.mxs = ms.contiguous()
    assert R._mx_layout_ok(lay)
