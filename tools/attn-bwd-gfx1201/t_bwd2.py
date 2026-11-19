"""Gate 1: hk fwd LSE and hk dQ vs an fp32 reference (torch autograd in fp32)."""
import math, sys, torch
import hk.ops.attn as A, hk.ops.attn_bwd as AB
torch.manual_seed(0); dev = "cuda"
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
fails = 0
for (B, H, Hk, S, D, causal) in [(2, 16, 16, 2048, 128, True), (2, 16, 16, 2048, 128, False), (1, 8, 2, 1000, 128, True),
                                 (1, 16, 16, 777, 64, False), (2, 8, 8, 2049, 64, True)]:
    sc = D ** -0.5
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16); k = torch.randn(B, Hk, S, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k); do = torch.randn_like(q)
    tag = f"B{B} H{H}/{Hk} S{S} D{D} {'causal' if causal else 'full'}"
    try:
        fk = A.attn_kernel(f"t_fwd_lse_d{D}{'_c' if causal else ''}", head_dim=D, causal=causal, lse=True)
        o = torch.empty_like(q); lse = torch.empty(B, H, 1, S, device=dev, dtype=torch.float32)
        fk(q, k, v, o, lse)
        # reference in fp32
        qf, kf, vf, dof = (t.float().requires_grad_(True) for t in (q, k, v, do))
        g = H // Hk
        kx, vx = kf.repeat_interleave(g, 1), vf.repeat_interleave(g, 1)
        s = (qf @ kx.transpose(-1, -2)) * sc
        if causal:
            s = s.masked_fill(torch.ones(S, S, device=dev, dtype=torch.bool).triu(1), float("-inf"))
        lse_ref = torch.logsumexp(s, -1) / math.log(2)
        p = torch.softmax(s, -1); oref = p @ vx
        dq_r, dk_r, dv_r = torch.autograd.grad(oref, (qf, kf, vf), dof)
        e_o, e_l = rel(o, oref), rel(lse[:, :, 0], lse_ref)
        delta = (o.float() * do.float()).sum(-1)[:, :, None, :].contiguous()
        qk = AB.dq_kernel(f"t_dq_d{D}{'_c' if causal else ''}", head_dim=D, causal=causal)
        dq = torch.empty_like(q)
        qk(q, k, v, do, lse, delta, dq)
        e_dq = rel(dq, dq_r)
        kdk = AB.dkv_kernel(f"t_dk_d{D}{'_c' if causal else ''}", mode="dk", head_dim=D, causal=causal)
        kdv = AB.dkv_kernel(f"t_dv_d{D}{'_c' if causal else ''}", mode="dv", head_dim=D, causal=causal)
        dk = torch.empty_like(k); dv = torch.empty_like(v)
        kdk(q, k, v, do, lse, delta, dk); kdv(q, k, v, do, lse, delta, dv)
        e_dk, e_dv = rel(dk, dk_r), rel(dv, dv_r)
        ok = e_o < 2e-2 and e_l < 1e-3 and e_dq < 2e-2 and e_dk < 2e-2 and e_dv < 2e-2 and dq_r.norm() > 0
        fails += not ok
        print(f"{tag}: o {e_o:.2e} lse {e_l:.2e} dq {e_dq:.2e} dk {e_dk:.2e} dv {e_dv:.2e} {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        fails += 1; print(f"{tag}: ERR {type(e).__name__}: {str(e)[:600]}", flush=True)
print("FAILS", fails); sys.exit(1 if fails else 0)
