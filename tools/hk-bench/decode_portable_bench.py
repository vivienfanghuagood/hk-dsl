#!/usr/bin/env python3
"""One decode-attention source on RDNA and CDNA: hk.ops.paged_decode_portable, timed against
vLLM's TRITON_ATTN unified_attention (when vLLM is importable) on the same cache, and on RDNA also
against hk.ops.paged_decode (the RDNA-tuned walk).

    python3 tools/hk-bench/decode_portable_bench.py --model minicpm5 --batch 1,4,16 --ctx 16384

Same process, every arm's burst over `layers` caches captured as one CUDA graph, replays timed,
arms alternating, best of --rounds; the layer caches cycle through >= 1 GiB (cold KV: past the
W7900D's 96 MB Infinity Cache and the MI355X's 256 MB MALL). Each arm is checked against float32
first. Bandwidth = every request's K and V bytes over the call's time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

MODELS = {
    "minicpm5": dict(hq=16, hkv=2, d=128, page=16, order="planes"),     # MiniCPM5-2B
    "qwen3.8": dict(hq=24, hkv=4, d=256, page=784, order="heads"),      # Qwen3.8-27B, TP1
    "qwen3-8b": dict(hq=32, hkv=8, d=128, page=16, order="planes"),     # Qwen3-8B
}
COLD_BYTES = 1 << 30


def cache_views(nb, page, hkv, d, order, device):
    if order == "planes":
        st = torch.empty(nb, 2, page, hkv, d, device=device, dtype=torch.bfloat16)
        return st, st[:, 0], st[:, 1], st.view(nb, 2, page, hkv * d)
    if order == "tokens":
        st = torch.empty(nb, page, hkv, 2 * d, device=device, dtype=torch.bfloat16)
        return st, st[..., :d], st[..., d:], st.view(nb, 1, page, hkv * 2 * d)
    st = torch.empty(nb, hkv, page, 2 * d, device=device, dtype=torch.bfloat16)
    return st, st[..., :d].transpose(1, 2), st[..., d:].transpose(1, 2), st


class Case:
    def __init__(self, m, batch, ctx, device="cuda", seed=0):
        self.m, self.batch, self.ctx = m, batch, ctx
        hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
        g = torch.Generator(device=device).manual_seed(seed)
        pages = -(-ctx // page)
        nb = batch * pages + 1
        self.layers = max(2, min(64, math.ceil(COLD_BYTES / (nb * page * hkv * d * 4))))
        self.kv_bytes = batch * ctx * hkv * d * 4
        self.caches = []
        for _ in range(self.layers):
            st, k, v, kv = cache_views(nb, page, hkv, d, m["order"], device)
            st.normal_(generator=g)
            self.caches.append((k, v, kv))
        perm = torch.randperm(nb - 1, generator=torch.Generator().manual_seed(seed + 1)) + 1
        self.block_table = perm[: batch * pages].view(batch, pages).to(torch.int32).to(device)
        self.lens_host = torch.full((batch,), ctx, dtype=torch.int32)
        self.seq_lens = self.lens_host.to(device)
        self.q_start = torch.arange(batch + 1, dtype=torch.int32, device=device)
        self.q = torch.randn(batch, hq, d, device=device, dtype=torch.bfloat16, generator=g)
        self.scale = d ** -0.5

    def reference(self, layer=0):
        m = self.m
        hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
        k, v, _ = self.caches[layer]
        out = torch.empty(self.batch, hq, d, device=self.q.device, dtype=torch.float32)
        for r in range(self.batch):
            n = int(self.lens_host[r])
            idx = self.block_table[r, :-(-n // page)].long()
            kk = k[idx].reshape(-1, hkv, d)[:n].float()
            vv = v[idx].reshape(-1, hkv, d)[:n].float()
            qq = self.q[r].float().view(hkv, hq // hkv, d)
            p = torch.softmax(torch.einsum("hgd,nhd->hgn", qq, kk) * self.scale, dim=-1)
            out[r] = torch.einsum("hgn,nhd->hgd", p, vv).reshape(hq, d)
        return out


def triton_arm(case):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: PLC0415

    m = case.m
    hq, hkv, d = m["hq"], m["hkv"], m["d"]
    thr, segs = 128 // hkv, 16
    dp = 1 << (d - 1).bit_length()
    dev = case.q.device
    so = torch.empty(thr, hq, segs, dp, dtype=torch.float32, device=dev)
    sm = torch.empty(thr, hq, segs, dtype=torch.float32, device=dev)
    se = torch.empty(thr, hq, segs, dtype=torch.float32, device=dev)
    out = torch.empty_like(case.q)
    desc = torch.ones(1, device=dev, dtype=torch.float32).expand(case.batch, hkv)
    max_k = int(case.lens_host.max())

    def call(layer):
        k, v, _ = case.caches[layer]
        unified_attention(q=case.q, k=k, v=v, out=out, cu_seqlens_q=case.q_start, max_seqlen_q=1,
                          seqused_k=case.seq_lens, max_seqlen_k=max_k, softmax_scale=case.scale,
                          causal=True, window_size=(-1, -1), block_table=case.block_table,
                          softcap=0, q_descale=None, k_descale=desc, v_descale=desc,
                          seq_threshold_3D=thr, num_par_softmax_segments=segs,
                          softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    return call, out


def hk_arm(case, portable: bool, splits: int):
    from hk.ops import paged_decode as pd  # noqa: PLC0415
    from hk.ops import paged_decode_portable as pdp  # noqa: PLC0415
    from hk.runtime.torch_ext import launcher  # noqa: PLC0415

    m = case.m
    hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
    grp = hq // hkv
    s = splits or (pdp if portable else pd).plan_splits(case.batch, hkv)
    mod = pdp if portable else pd
    walk = launcher(mod.walk_for(d, grp, page, m["order"], s))
    merge = launcher(pd.merge_for(d, grp, s)) if s > 1 else None
    out = torch.empty_like(case.q)
    o_part, ml = pd.scratch(case.batch * s, hkv, grp, d, case.q.device)
    qh = case.q.unsqueeze(0)
    o4 = out.view(case.batch, hkv, grp, d)

    def call(layer):
        walk(qh, case.caches[layer][2], case.block_table, case.seq_lens, case.q_start, o4,
             o_part, ml)
        if merge is not None:
            merge(case.seq_lens, case.q_start, o4, o_part, ml)
    call.splits = s
    return call, out


def ceiling_arm(case):
    """Not attention: a sum over one layer's whole cache storage, the card's read rate on the
    same bytes (reported as 'GB/s' of the KV the decode reads, so comparable)."""
    out = torch.empty((), device=case.q.device, dtype=torch.float32)

    def call(layer):
        kv = case.caches[layer][2]
        torch.sum(kv.reshape(-1), 0, dtype=torch.float32, out=out)
    return call, None


def graphed(call, layers):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(layers):
            call(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(layers):
            call(i)
    return g.replay


def time_arms(replays, layers, rounds):
    best = {k: float("inf") for k in replays}
    for fn in replays.values():
        fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for name, fn in replays.items():
            for _ in range(3):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                best[name] = min(best[name], a.elapsed_time(b) * 1000 / layers)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="minicpm5", choices=sorted(MODELS))
    ap.add_argument("--batch", default="1,4,16")
    ap.add_argument("--ctx", default="16384")
    ap.add_argument("--splits", default="0", help="portable arm's split counts, e.g. 0,8,32 (0: plan)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--page", type=int, default=0,
                    help="override the model's page (gfx950 takes whole 32-key blocks: 32, 544)")
    ap.add_argument("--json")
    args = ap.parse_args()
    from hk.target import current_arch  # noqa: PLC0415

    arch = current_arch()
    try:
        import vllm  # noqa: F401,PLC0415
        have_vllm = True
    except Exception:  # noqa: BLE001
        have_vllm = False
    rows = []
    m = dict(MODELS[args.model])
    if args.page:
        m["page"] = args.page
    for ctx in [int(x) for x in args.ctx.split(",")]:
        for batch in [int(x) for x in args.batch.split(",")]:
            case = Case(m, batch, ctx)
            arms = {}
            if have_vllm:
                arms["triton"] = triton_arm(case)
            for s in [int(x) for x in args.splits.split(",")]:
                arms[f"portable_s{s or 'plan'}"] = hk_arm(case, True, s)
            if arch.startswith("gfx1") and m["page"] % 16 == 0:
                arms["rdna_tuned"] = hk_arm(case, False, 0)
            arms["read_ceiling"] = ceiling_arm(case)
            ref = case.reference(0)
            err = {}
            for name, (call, out) in arms.items():
                if out is None:
                    err[name] = 0.0
                    continue
                out.fill_(float("nan"))
                call(0)
                torch.cuda.synchronize()
                e = (out.float() - ref).abs().max().item()
                err[name] = round(e, 5) if math.isfinite(e) else "nan"
            ok = {n: c for n, (c, _) in arms.items() if err[n] != "nan" and err[n] < 2e-2}
            t = time_arms({n: graphed(c, case.layers) for n, c in ok.items()}, case.layers,
                          args.rounds)
            row = dict(arch=arch, model=args.model, page=m["page"], batch=batch, ctx=ctx, layers=case.layers,
                       splits={n: getattr(arms[n][0], "splits", None) for n in arms},
                       err=err, us={n: round(v, 2) for n, v in t.items()},
                       gbs={n: round(case.kv_bytes / v / 1e3, 1) for n, v in t.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            del case, arms
            torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
