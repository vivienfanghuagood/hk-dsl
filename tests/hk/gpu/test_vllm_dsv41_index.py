"""hk.ops.dsv41_index, the launch hk.integration.vllm_dsv41_index routes DeepSeek-V4.1's index layers
to, against vLLM's own chain (the kernels mono_decode._select runs) on synthetic layers: the
compressor ring, the compressed-cache records, the indexer K cache, q / q_scale and the head
weights, then the top-k (and the candidate source's candidates). Needs vLLM's ROCm build and
aiter: the vLLM image."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("aiter")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled MFMA", allow_module_level=True)

from hk.ops import dsv41_index as X  # noqa: E402

DEV = "cuda"
NPAGE = 64                      # records a page of the compressed and indexer K caches
SCALE = 0.08838834764831845     # DeepSeek-V4.1-Flash's indexer softmax scale


def _layer(kind: str, g, width=None):
    """A synthetic index layer: "A" a dense index source (compress ratio 2, the ring), "B" the
    candidate source (ratio 1), "C" a consumer (no compressor: the K cache is another layer's).
    `width`: the indexer's max_model_len (default: max_model_len 8192's)."""
    r = lambda *s, sd=1.0: (torch.randn(*s, device=DEV, generator=g) * sd)  # noqa: E731
    ratio = 2 if kind == "A" else 1
    w = dict(kind=kind, ratio=ratio, haskv=kind != "C", width=width or (4096 if kind == "A" else 8192),
             cand=kind == "B",
             wp=r(X.NH, X.H, sd=0.02).bfloat16(),
             wqb=r(X.NQ, X.QR).to(torch.float8_e4m3fn),
             wqb_s=torch.randint(118, 124, (X.NQ // 32, X.QR // 32), device=DEV, dtype=torch.uint8,
                                 generator=g),
             cs_a=(torch.rand(16640, 64, device=DEV, generator=g) * 2 - 1),
             cs_i=(torch.rand(16640, 64, device=DEV, generator=g) * 2 - 1))
    if w["haskv"]:
        w.update(wkv=r(512 * ratio, X.H, sd=0.02).bfloat16(), norm=(1 + r(512, sd=0.1)).bfloat16(),
                 wk=r(X.HD, 512, sd=0.05).bfloat16(), k_norm=(1 + r(X.HD, sd=0.1)).bfloat16())
    return w


def _step(w, positions, g):
    """A decode step's inputs, slots and caches (row i on its own pages), as vLLM lays them out."""
    m = len(positions)
    ratio, width = w["ratio"], w["width"]
    pos = torch.tensor(positions, dtype=torch.int64, device=DEV)
    per_row = -(-width // NPAGE)
    kpages = m * per_row + 1                       # page 0 unused, as vLLM's
    kidx = torch.randint(0, 256, (kpages, NPAGE * 68), dtype=torch.uint8, device=DEV, generator=g)
    kidx[:, NPAGE * 64:] = torch.randint(118, 126, (kpages, NPAGE * 4), dtype=torch.uint8, device=DEV,
                                         generator=g)  # e8m0 scales: no NaN or overflow
    bt = (1 + torch.arange(m, device=DEV)[:, None] * per_row
          + torch.arange(per_row, device=DEV)[None]).to(torch.int32).contiguous()
    lens = torch.tensor([(p + 1) // ratio for p in positions], dtype=torch.int32, device=DEV)
    live = [(p + 1) % ratio == 0 for p in positions]
    kslot = torch.tensor([int(bt[i, (p // ratio) // NPAGE]) * NPAGE + (p // ratio) % NPAGE if live[i] else -1
                          for i, p in enumerate(positions)], dtype=torch.int64, device=DEV)
    s = dict(m=m, pos=pos, kidx=kidx, bt=bt, lens=lens, kslot=kslot,
             bends=((lens + 7) // 8).to(torch.int32),
             normed=torch.randn(m, X.H, device=DEV, generator=g).bfloat16(),
             qr=torch.randn(m, X.QR, device=DEV, generator=g).bfloat16())
    if w["haskv"]:
        s["comp"] = torch.randint(0, 256, (kpages, NPAGE * 584), dtype=torch.uint8, device=DEV, generator=g)
        s["cslot"] = kslot.clone()                 # the compressed cache shares the K cache's pages
    if ratio == 2:
        s["ring"] = torch.randn(m, 8 * 1024, device=DEV, generator=g)
        s["rslot"] = torch.tensor([i * 8 + p % 8 for i, p in enumerate(positions)], dtype=torch.int64,
                                  device=DEV)
    return s


def _clone(s):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}


def _vllm(w, s):
    """vLLM's chain on the step (mono_decode._select's kernels, as the decode path runs them)."""
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.models.deepseek_v41.amd.rocm import rocm_mxfp4_indexer_k_store, rocm_mxfp4_indexer_q_quant
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm, rope_quant_insert)
    from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as RI

    m, ratio, pos = s["m"], w["ratio"], s["pos"]
    if w["haskv"]:
        kv_score = torch.mm(s["normed"], w["wkv"].T, out_dtype=torch.float32)
        latent = torch.empty(m, 512, dtype=torch.bfloat16, device=DEV)
        if ratio == 2:
            ar = torch.arange(m + 1, dtype=torch.int32, device=DEV)
            fused_save_compress_norm(kv_score, pos, s["ring"].view(m, 8, 1024), s["rslot"], ar, ar[:m],
                                     w["norm"], 1e-20, ratio, latent)
        else:
            fused_save_compress_norm(kv_score, pos, None, s["cslot"], None, None, w["norm"], 1e-20, ratio,
                                     latent)
        rope_quant_insert(latent, pos, w["cs_a"], s["comp"].view(-1, NPAGE, 584), s["cslot"], ratio)
        k_pre = torch.nn.functional.linear(latent, w["wk"])
        rocm_mxfp4_indexer_k_store(k_pre, pos, w["cs_i"], w["k_norm"], 1e-20, s["kidx"].view(-1, NPAGE, 68),
                                   s["kslot"], ratio, True, num_heads=X.NH)
    wproj = torch.nn.functional.linear(s["normed"], w["wp"])
    xq, xs = mxfp8_e4m3_quantize(s["qr"])
    q = rocm_mxfp8_block32_gemm(xq, xs, w["wqb"], w["wqb_s"], torch.bfloat16).view(m, X.NH, X.HD)
    (q4, q4s), wts = rocm_mxfp4_indexer_q_quant(pos, q, w["cs_i"], wproj, SCALE, X.NH ** -0.5)
    kv = RI._kv_view(s["kidx"].view(-1, NPAGE, 68), X.HD)
    logits = torch.empty(m, w["width"], dtype=torch.float32, device=DEV)
    q_scale = q4s.view(torch.uint8).view(m, X.NH, -1)
    RI._aiter().paged_mxfp4_mqa_logits(q4.unsqueeze(1), q_scale.unsqueeze(1), kv, wts, s["lens"], s["bt"],
                                       w["width"], out_logits=logits, clean_logits=False)
    topk = torch.full((m, X.TOPK), -1, dtype=torch.int32, device=DEV)
    RI._topk(logits, s["lens"], topk, X.TOPK)
    return dict(q4=q4.reshape(-1).view(torch.uint8), q4s=q4s.reshape(-1).view(torch.uint8),
                wts=wts.reshape(-1), topk=topk)


def _hk(w, s):
    """hk's launch on the step, the logits and top-k inside it."""
    from hk.mx import expand_block_scales
    from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as RI

    m = s["m"]
    z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, dtype=dt, device=DEV)  # noqa: E731
    none = torch.full((m,), -1, dtype=torch.int64, device=DEV)
    i32 = lambda t: t.contiguous().view(torch.int32)  # noqa: E731
    qs8, qss = X.qr_scratch(DEV)
    mb = X.mailbox(DEV)
    ep = torch.tensor([7], dtype=torch.int32, device=DEV)
    q4, q4s = z(m * X.NH * 64, dt=torch.uint8), z(m * X.NH * 4, dt=torch.uint8)
    wts = z(m * X.NH, dt=torch.float32)
    topk = torch.full((m, X.TOPK), -7, dtype=torch.int32, device=DEV)
    cand = torch.full((m, 2048), -7, dtype=torch.int32, device=DEV)
    comp = s.get("comp", z(1, NPAGE * 584, dt=torch.uint8))
    ring = s.get("ring", z(1, 8192, dt=torch.float32))
    lay = RI.rocm_paged_mxfp4_cache_layout(X.NH, X.HD, NPAGE)
    k = X.index_kernel(SCALE * X.NH ** -0.5)
    k(s["normed"], w.get("wkv", z(512, X.H)), w["wp"], s["qr"], qs8, qss, w["wqb"],
      expand_block_scales(w["wqb_s"], 32).contiguous().view(torch.int8), mb, ep,
      i32(s["pos"]), i32(s.get("rslot", none)), i32(s.get("cslot", none)), i32(s["kslot"]), ring,
      comp.view(torch.float8_e4m3fn), comp.view(torch.bfloat16), comp.view(torch.int8), s["kidx"].view(torch.int8),
      w["cs_a"].view(-1), w["cs_i"].view(-1), w.get("norm", z(512)).view(torch.int32), w.get("wk", z(X.HD, 512)),
      w.get("k_norm", z(X.HD)), q4.view(torch.int8), q4s.view(torch.int8), wts, q4.view(torch.int32),
      s["kidx"].view(torch.int32), s["bt"], s["lens"], topk, cand, s["bends"],
      SEL=1, CAND=int(w["cand"]), NCAND=2048, WIDTH=X.width(w["width"]), BTW=s["bt"].shape[1], RATIO=w["ratio"], MC=m,
      HASKV=int(w["haskv"]), CBLK=NPAGE, KBLK=NPAGE, NPT=lay.n_per_tile, DPT=lay.d_per_tile, SL=lay.scale_lanes)
    return dict(q4=q4, q4s=q4s, wts=wts, topk=topk, cand=cand)


#: positions a step: long rows (the in-launch logits and radix select) and a short one (<= TOPK
#: entries: the identity), at 1 and 8 rows
STEPS = {1: [2901], 8: [2900, 1301, 3333, 777, 4001, 2048, 300, 3090]}


@pytest.mark.parametrize("kind,width", [("A", None), ("B", None), ("C", None),
                                        # max_model_len 16640's dense layers: 8320 a row, WIDTH 12288
                                        ("A", 8320)])
@pytest.mark.parametrize("m", [1, 8])
def test_the_launch_is_vllms_chain(kind, width, m):
    g = torch.Generator(device=DEV).manual_seed(m * 7 + ord(kind))
    w = _layer(kind, g, width)
    steps = STEPS[m] if width is None else [p * 4 + 1 for p in STEPS[m]]   # rows to ~8000 entries
    pre = _step(w, steps, g)
    ref_s, hk_s = _clone(pre), _clone(pre)
    ref = _vllm(w, ref_s)
    got = _hk(w, hk_s)
    torch.cuda.synchronize()
    # the caches: byte for byte, but for the compressor's summation order. Its kv_score row is the
    # launch's K split summed in order, not torch.mm's: the ring's fp32 rows agree within ~1e-7,
    # and a latent element can round to the neighbouring bf16 -- one fp8 code or one bf16 ulp of the
    # RoPE part in a record, now and then (1 byte of 8 records at max_model_len 16640's widths; none
    # on the real captures)
    assert torch.equal(hk_s["kidx"], ref_s["kidx"])
    if w["haskv"]:
        off = (hk_s["comp"] != ref_s["comp"]).nonzero()
        assert off.shape[0] <= 2, off.shape
        step = (hk_s["comp"][off[:, 0], off[:, 1]].int() - ref_s["comp"][off[:, 0], off[:, 1]].int()).abs()
        assert bool((step <= 1).all()) and bool((off[:, 1] < NPAGE * 576).all())   # a value, not a scale
        assert not torch.equal(hk_s["kidx"], pre["kidx"])          # it wrote something
    if w["ratio"] == 2:
        a, b = ref_s["ring"], hk_s["ring"]
        assert float((a - b).abs().max() / a.abs().max()) < 1e-6
    # the logits' inputs
    assert torch.equal(got["q4"], ref["q4"])
    assert torch.equal(got["q4s"], ref["q4s"])
    assert float((got["wts"] - ref["wts"]).abs().max() / ref["wts"].abs().max()) < 1e-2
    # the top-k: the same sets (logits within rounding of aiter's; a tie at the 512th could differ)
    for i in range(m):
        n = int(pre["lens"][i])
        a, b = set(ref["topk"][i].tolist()), set(got["topk"][i].tolist())
        assert len(a & b) >= 0.99 * min(n, X.TOPK), (i, n, len(a & b))
        if n <= X.TOPK:
            assert got["topk"][i].tolist() == list(range(n)) + [-1] * (X.TOPK - n)
    if w["cand"]:                       # every block of the row while they fit the list
        e = pre["bends"].tolist()
        for i in range(m):
            assert got["cand"][i].tolist() == list(range(e[i])) + [-1] * (2048 - e[i])
