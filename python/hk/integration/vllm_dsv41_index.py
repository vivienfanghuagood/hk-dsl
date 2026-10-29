"""DeepSeek-V4.1's index layers on hk's index launch, under vLLM's `--monokernel` (`HK_VLLM_DSV41_INDEX=1`,
gfx950).

With `--monokernel`, a decode step's backbone layer runs as dsv41_mono's two persistent launches
(K1, K2). Between them an index layer (a compressor and the sparse-attention indexer) runs
vLLM's chain, `mono_decode._select`: 9-12 kernels -- the compressor's projection, ring, RMSNorm
and compressed-cache insert, the indexer's K store, qr's MXFP8 quant and wq_b GEMM,
weights_proj, the Q RoPE / MXFP4 quant, then the logits and the top-k. The route replaces that
function: a decode step of at most `hk.ops.dsv41_index.ROWS` rows runs the whole chain as one
resident launch (`hk.ops.dsv41_index`), the logits and top-k included while the model length
fits the launch (`WIDTH_MAX`) -- the candidate source's block top-k and the consumers' candidate
mask too; past that, vLLM's indexer op runs on the launch's q, q_scale and weights. Anything else -- more rows,
a DSpark draft layer, a layer whose shapes or metadata the launch does not know -- runs vLLM's
`_select` unchanged. The launch's mailbox tags come from mono's own epoch (the runner's), which
moves with every K1, so steps and layers never read each other's values; CUDA-graph replays
included.

Measured on DeepSeek-V4.1-Flash, TP2, MI355X (vLLM e68efc685): caches, q and q_scale byte-identical
to vLLM's chain in vLLM, on all 8 index layers; top-512 sets equal at M=1, overlap >= 0.994 at
M=8 (head weights within one bf16 ulp: vLLM's weights_proj rounds a few heads differently). In the
serving scorecard, with hk's other default routes, TPOT is 4.5 / 4.4 / 3.1% below mono's at batch
1 / 4 / 8 and GSM8K-1319 within stock's noise (docs/integration.md, "The DeepSeek-V4.1 index launch").

    vllm serve <DeepSeek-V4.1-Flash> --monokernel ...          # on by default on gfx950
    HK_VLLM_DSV41_INDEX=0 vllm serve ...                       # mono's own chain
    HK_DSV41_INDEX_TOPK=0 vllm serve ...                       # the launch without the logits / top-k

`HK_DSV41_INDEX_CHECK=N` checks the launch in place: for the first N eager calls of each index
layer it runs vLLM's chain after hk's on the same rows and prints the bytes that differ
(caches, q, q_scale) and the top-k agreement, then keeps vLLM's results.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Callable, Optional

from . import engine

#: Decode rows the launch takes, at most (hk.ops.dsv41_index.ROWS; not imported here: the plugin
#: loads this module in every process, the kernel's module is for the ones that run it).
ROWS = 8
#: The module the route patches, and the function.
MODULE = "vllm.models.deepseek_v41.amd.mono_decode"
TARGET = "_select"
_ORIG: Optional[Callable] = None
_MOD = None
_FINDER = None
#: Route hk-able calls to hk (True) or leave every call to vLLM (False): an in-process A/B.
ENABLED = True
#: The logits and top-k inside the launch (layers whose model length and candidates it takes).
TOPK = os.environ.get("HK_DSV41_INDEX_TOPK", "1") not in ("0", "", "off", "no")
#: In-place check: the first N eager calls of each index layer (module docstring).
CHECK = int(os.environ.get("HK_DSV41_INDEX_CHECK", "0") or 0)

PATCH = engine.Patch("hk_vllm::dsv41_index", paths=("hk", "stock"))
STATS = PATCH.stats
#: Why a layer stays on vLLM's chain, by layer id (set once, at its first call).
STOCK_LAYERS: dict = {}
_LAYERS: dict = {}
_BUFS: dict = {}
_NONE: dict = {}
_KERN: dict = {}
_CALLS: Counter = Counter()


def _bufs(dev):
    """The device's mailbox, qr scratch, outputs and the stand-ins of absent caches (once)."""
    b = _BUFS.get(dev)
    if b is None:
        import torch  # noqa: PLC0415

        from ..ops import dsv41_index as X  # noqa: PLC0415

        z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        qs8, qss = X.qr_scratch(dev)
        b = _BUFS[dev] = dict(
            mb=X.mailbox(dev), qs8=qs8, qss=qss,
            q4=z(X.ROWS * X.NH * 64, dt=torch.uint8), q4s=z(X.ROWS * X.NH * 4, dt=torch.uint8),
            wts=z(X.ROWS * X.NH, dt=torch.float32),
            ring0=z(1, 8192, dt=torch.float32), comp0=z(1, 64 * 584, dt=torch.uint8),
            i32_0=z(1, 64, dt=torch.int32))
    return b


def _none(dev, m: int):
    """A step's "no slot" mapping, made once (a torch.full a call is a fill kernel in every graph)."""
    t = _NONE.get((dev, m))
    if t is None:
        import torch  # noqa: PLC0415

        t = _NONE[(dev, m)] = torch.full((m,), -1, dtype=torch.int64, device=dev)
    return t


def _kernel(wscale: float):
    k = _KERN.get(wscale)
    if k is None:
        from ..ops import dsv41_index as X  # noqa: PLC0415

        k = _KERN[wscale] = X.index_kernel(wscale)
    return k


def _plan(layer) -> Optional[dict]:
    """The layer's weights and constants as the launch takes them, or None (and why, in
    STOCK_LAYERS) when the launch does not take the layer. Once a layer."""
    attn = layer.attn
    lid = attn.layer_id
    if lid in _LAYERS:
        return _LAYERS[lid]
    try:
        if getattr(getattr(layer, "mono", None), "draft", False):
            p, STOCK_LAYERS[lid] = None, "a DSpark draft layer (its own runner and epoch)"
        else:
            p = _make_plan(layer)
    except Exception as e:  # noqa: BLE001 -- a layer not as measured stays on vLLM's chain
        p, STOCK_LAYERS[lid] = None, f"{type(e).__name__}: {e}"
    if p is None and lid not in STOCK_LAYERS:
        STOCK_LAYERS[lid] = "not taken"
    if p is None:
        print(f"[hk] dsv41_index: layer {lid} stays on vLLM's chain ({STOCK_LAYERS[lid]})", flush=True)
    _LAYERS[lid] = p
    return p


def _make_plan(layer) -> Optional[dict]:
    import torch  # noqa: PLC0415
    from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as RI  # noqa: PLC0415

    from ..mx import expand_block_scales  # noqa: PLC0415
    from ..ops import dsv41_index as X  # noqa: PLC0415

    attn = layer.attn
    comp, ind = attn.compressor, attn.indexer
    dev = ind.wq_b.weight.device
    z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
    haskv = comp is not None
    ratio = comp.compress_ratio if haskv else 1
    p = dict(haskv=int(haskv), ratio=ratio,
             wkv=comp.fused_wkv_wgate.weight if haskv else z(512, X.H),
             wp=ind.weights_proj.weight, wqb=ind.wq_b.weight,
             wqb_s=expand_block_scales(ind.wq_b.weight_scale, 32).contiguous().view(torch.int8),
             normw=(comp.norm.weight if haskv else z(512)).contiguous().view(torch.int32),
             wk=ind.wk.weight.contiguous() if haskv else z(X.HD, 512),
             knw=ind.k_norm.weight if haskv else z(X.HD),
             cs_a=attn.rotary_emb.cos_sin_cache.view(-1), cs_i=attn.indexer_rotary_emb.cos_sin_cache.view(-1),
             wscale=ind.softmax_scale * ind.n_head ** -0.5)
    shapes = {"n_head": (ind.n_head, X.NH), "head_dim": (ind.head_dim, X.HD),
              "wq_b": (tuple(p["wqb"].shape), (X.NQ, X.QR)), "weights_proj": (tuple(p["wp"].shape), (X.NH, X.H)),
              "wkv": (tuple(p["wkv"].shape), (512 * ratio, X.H)), "wk": (tuple(p["wk"].shape), (X.HD, 512)),
              "ratio": (ratio in (1, 2), True),
              "dtypes": ((p["wkv"].dtype, p["wp"].dtype, p["wqb"].dtype),
                         (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn))}
    bad = [f"{k} {got} (the launch: {want})" for k, (got, want) in shapes.items() if got != want]
    if bad or not all(p[k].is_contiguous() for k in ("wkv", "wp", "wqb")):
        STOCK_LAYERS[attn.layer_id] = "; ".join(bad) or "non-contiguous weights"
        return None
    kc = ind.k_cache.kv_cache
    p["kblk"] = kc.view(kc.shape[0], -1).shape[1] // 68
    op = ind.indexer_op
    p["width"] = int(ind.max_model_len)
    cb = getattr(op, "candidate_blocks", None)
    cbs = int(getattr(op, "candidate_block_size", 0) or 0)
    # the candidate source / consumers: while a row's blocks fit the candidate list, the source's
    # block top-k keeps every block and the consumers' pool is the whole row -- both reduce to the
    # dense selection (the source also writes its candidates: every block, in order); past it the
    # launch does the source's block top-k and the consumers' mask itself
    p["cand"] = int(bool(getattr(op, "candidate_write", False)))
    p["cons"] = int(cb is not None and not p["cand"])
    p["sel"] = int(TOPK and (cb is None or cbs == X.CBLOCK) and p["width"] <= X.WIDTH_MAX
                   and ind.topk_tokens == X.TOPK)
    p["cb"] = cb
    lay = RI.rocm_paged_mxfp4_cache_layout(ind.n_head, ind.head_dim, p["kblk"])
    p.update(npt=lay.n_per_tile, dpt=lay.d_per_tile, sl=lay.scale_lanes)
    return p


def _launch(module, layer, positions, normed, qr, p):
    """The launch on this step's rows: (q, q_scale, weights, the cache rows it wrote, whether it
    wrote the top-k)."""
    import torch  # noqa: PLC0415
    from vllm.forward_context import get_forward_context  # noqa: PLC0415

    from ..ops import dsv41_index as X  # noqa: PLC0415

    attn = layer.attn
    comp, ind = attn.compressor, attn.indexer
    m = normed.shape[0]
    dev = normed.device
    b = _bufs(dev)
    md = get_forward_context().attn_metadata
    i32 = lambda t: t.contiguous().view(torch.int32)  # noqa: E731
    if p["haskv"]:
        cache = attn._compressed_kv_cache()
        comp_v = cache.view(cache.shape[0], -1)
        cblk = cache.shape[1]
        cslot = md[comp.k_cache_prefix].slot_mapping[:m]
        if comp.state_cache is not None:
            rc = comp.state_cache.kv_cache
            ring = rc.view(rc.shape[0], -1)
            rslot = md[comp.state_cache.prefix].slot_mapping[:m]
        else:
            ring, rslot = b["ring0"], _none(dev, m)
    else:
        comp_v, cblk, cslot, ring, rslot = b["comp0"], 64, _none(dev, m), b["ring0"], _none(dev, m)
    kc = ind.k_cache.kv_cache
    kidx = kc.view(kc.shape[0], -1)
    mk = md[ind.k_cache.prefix]
    kslot = mk.slot_mapping[:m]
    runner = module._mono_runner(dev, None)
    sel = p["sel"] and mk.decode_native is None and not getattr(mk, "decode_use_gather", False)
    if sel:
        bt = mk.decode.block_table
        lens = mk.decode_row_lens
        tk = ind.topk_indices_buffer
        sel = (bt.dtype == torch.int32 and bt.stride(1) == 1 and lens.dtype == torch.int32
               and lens.shape[0] == m and tk.dtype == torch.int32 and tk.is_contiguous()
               and tk.shape[1] == X.TOPK)
    cand, bends = b["i32_0"], b["i32_0"]
    if sel and (p["cand"] or p["cons"]):
        cand, bends = p["cb"], mk.decode_block_ends
        sel = (cand.dtype == torch.int32 and cand.is_contiguous() and bends is not None
               and bends.dtype == torch.int32)
    if not sel:
        bt, lens, tk, cand, bends = (b["i32_0"],) * 5
    _kernel(p["wscale"])(
        normed, p["wkv"], p["wp"], qr, b["qs8"], b["qss"], p["wqb"], p["wqb_s"], b["mb"], runner.epoch[:1],
        i32(positions[:m]), i32(rslot), i32(cslot), i32(kslot), ring,
        comp_v.view(torch.float8_e4m3fn), comp_v.view(torch.bfloat16), comp_v.view(torch.int8), kidx.view(torch.int8),
        p["cs_a"], p["cs_i"], p["normw"], p["wk"], p["knw"], b["q4"].view(torch.int8), b["q4s"].view(torch.int8),
        b["wts"], b["q4"].view(torch.int32), kidx.view(torch.int32), bt, lens, tk, cand, bends,
        SEL=int(sel), CAND=(p["cand"] if sel else 0), CONS=(p["cons"] if sel else 0),
        NCAND=(cand.shape[1] if sel and (p["cand"] or p["cons"]) else 2048),
        WIDTH=(X.width(p["width"]) if sel else 4096), BTW=(bt.shape[1] if sel else 64),
        RATIO=p["ratio"], MC=m, HASKV=p["haskv"], CBLK=cblk, KBLK=p["kblk"], NPT=p["npt"], DPT=p["dpt"],
        SL=p["sl"])
    q = b["q4"][:m * X.NH * 64].view(m, X.NH, 64)
    q_scale = b["q4s"][:m * X.NH * 4].view(torch.int32).view(m, X.NH)
    w = b["wts"][:m * X.NH].view(m, X.NH)
    written = dict(comp=(comp_v, cslot, cblk), ring=(ring, rslot, 8), kidx=(kidx, kslot, p["kblk"]))
    return q, q_scale, w, written, bool(sel)


def _select(layer, positions, normed, qr):
    """mono_decode._select: an index source's compressor and top-k selection, from K1's normed
    input and bf16 qr -- on hk's launch where it takes the step, else vLLM's."""
    p = _plan(layer) if ENABLED and normed.shape[0] <= ROWS else None
    if p is None:
        PATCH.count("stock")
        return _ORIG(layer, positions, normed, qr)
    PATCH.count("hk")
    q, q_scale, w, written, sel = _launch(_MOD, layer, positions, normed, qr, p)
    lid = layer.attn.layer_id
    n = _CALLS[lid]
    _CALLS[lid] += 1
    if n < CHECK and not engine.capturing():
        return _check(layer, positions, normed, qr, q, q_scale, w, written, sel, p, n)
    if not sel:                     # the launch stopped at the logits' inputs
        layer.attn.indexer.indexer_op(normed, (q, q_scale), None, w)
    return None


def _rows(t, slot, per_block):
    """The cache rows (blocks) the slots name, as a CPU copy."""
    import torch  # noqa: PLC0415

    s = slot[slot >= 0]
    if s.numel() == 0:
        return None
    return t.index_select(0, torch.unique(s // per_block)).cpu().clone()


def _check(layer, positions, normed, qr, q, q_scale, w, written, sel, p, n):
    """HK_DSV41_INDEX_CHECK: vLLM's chain after hk's launch on the same rows, compared; vLLM's
    results stay."""
    import torch  # noqa: PLC0415
    from vllm.forward_context import get_forward_context  # noqa: PLC0415

    attn = layer.attn
    ind = attn.indexer
    m = normed.shape[0]
    snap = {k: _rows(*v) for k, v in written.items()}
    hk_q = (q.cpu().clone(), q_scale.cpu().clone(), w.cpu().clone())
    hk_topk = ind.topk_indices_buffer[:m].cpu().clone() if sel else None
    hk_cand = p["cb"][:m].cpu().clone() if sel and p["cand"] else None
    _ORIG(layer, positions, normed, qr)                      # vLLM's chain rewrites the same rows
    torch.cuda.synchronize()
    res = {}
    for k, v in written.items():
        a = _rows(*v)
        if a is not None and snap[k] is not None:
            if k == "ring":                                  # fp32 rows: the accumulation order
                res["ring max rel"] = float((a - snap[k]).abs().max() / a.abs().max().clamp_min(1e-30))
            else:
                res[f"{k} bytes off"] = int((a.view(torch.uint8) != snap[k].view(torch.uint8)).sum())
    q2, s2, w2 = ind.forward_q(qr, None, ind.weights_proj(normed)[0], positions, attn.indexer_rotary_emb)
    res["q bytes off"] = int((q2.cpu().view(torch.uint8).reshape(-1) != hk_q[0].view(torch.uint8).reshape(-1)).sum())
    res["q_scale off"] = int((s2.cpu().reshape(-1) != hk_q[1].reshape(-1)).sum())
    res["weights max rel"] = float((w2.cpu() - hk_q[2]).abs().max() / w2.cpu().abs().max())
    if sel:
        st = ind.topk_indices_buffer[:m].cpu()
        res["row lens"] = get_forward_context().attn_metadata[ind.k_cache.prefix].decode_row_lens[:m].cpu().tolist()
        res["top-k sets = vLLM's"] = sum(set(st[i].tolist()) == set(hk_topk[i].tolist()) for i in range(m))
        res["top-k overlap min"] = round(min(len(set(st[i].tolist()) & set(hk_topk[i].tolist()))
                                             / max(1, len(set(st[i].tolist()) - {-1})) for i in range(m)), 4)
    if hk_cand is not None:
        sc = p["cb"][:m].cpu()
        res["candidate rows = vLLM's"] = sum(bool((sc[i] == hk_cand[i]).all()) for i in range(m))
    print(f"[hk] dsv41_index check L{attn.layer_id} call {n} M={m}: {res}", flush=True)
    return None


def _patch_module(module) -> str:
    global _ORIG, _MOD
    from . import contract  # noqa: PLC0415

    orig = getattr(module, TARGET)
    problems = contract.same_parameters(orig, _select, f"{MODULE}.{TARGET}")
    if problems or not hasattr(module, "_mono_runner"):
        print(f"[hk] WARNING: dsv41_index: {MODULE} is not the one measured, not patched: "
              f"{'; '.join(problems) or 'no _mono_runner'}", flush=True)
        return "dsv41_index: not patched (contract)"
    _ORIG, _MOD = orig, module
    setattr(module, TARGET, PATCH.guard(_select))      # (no host sync under capture)
    return "dsv41_index -> hk"


class _Finder:
    """Patches MODULE when vLLM imports it (the plugin runs before the model is imported, and
    importing the model package here would cost every process seconds)."""

    def find_spec(self, name, path, target=None):
        if name != MODULE:
            return None
        import importlib.machinery  # noqa: PLC0415

        _unhook()
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch_module(module)
        spec.loader.exec_module = exec_module
        return spec


def _unhook() -> None:
    global _FINDER
    if _FINDER is not None:
        if _FINDER in sys.meta_path:
            sys.meta_path.remove(_FINDER)
        _FINDER = None


def apply_patch() -> str:
    """Route mono_decode._select. Returns what happened, in one line."""
    global _FINDER
    from . import _common  # noqa: PLC0415

    if not _common.importable("vllm"):
        return "vllm is not importable; dsv41_index not patched"
    if _ORIG is not None or _FINDER is not None:
        return "dsv41_index: already patched"
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() != "gfx950":
        return f"dsv41_index: not patched -- gfx950's scaled MFMA, not {current_arch()}"
    if source_path() is None:
        return "dsv41_index: not patched -- this vLLM has no DeepSeek-V4.1 mono decode path"
    what = (f"DeepSeek-V4.1 index layers under --monokernel -> hk at decode steps of <= 8 rows"
            f"{'' if TOPK else ' (logits / top-k: vLLM)'}")
    if MODULE in sys.modules:
        r = _patch_module(sys.modules[MODULE])
        return what if r.endswith("hk") else r
    _FINDER = _Finder()
    sys.meta_path.insert(0, _FINDER)
    return what + " (when vLLM imports it)"


def revert() -> str:
    global _ORIG, _MOD
    _unhook()
    if _ORIG is None:
        return "dsv41_index: not patched"
    setattr(_MOD, TARGET, _ORIG)
    _ORIG = _MOD = None
    return "dsv41_index -> vllm"


def status() -> str:
    if _ORIG is None:
        return "not patched" + (" yet (waiting for vLLM's import)" if _FINDER is not None else "")
    stock = f"; layers on vLLM's chain: {STOCK_LAYERS}" if STOCK_LAYERS else ""
    return PATCH.summary() + stock


def source_path() -> Optional[str]:
    """MODULE's source file, found without importing it (or its package), or None."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        f = os.path.join(root, *MODULE.split(".")[1:]) + ".py"
        if os.path.isfile(f):
            return f
    return None
