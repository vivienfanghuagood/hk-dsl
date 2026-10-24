"""hk vt_kernel vs torch V^T for odd/even N, BNHD B=1: which columns differ."""
import torch
from hk.ops import attn_k3 as K3

for n in (1002, 1001, 641, 33):
    h, d = 2, 128
    v = torch.randn((1, n, h, d), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    n_pad = -(-n // 128) * 128
    ref = K3._vt_torch(v, n_pad)
    vf = v.transpose(1, 2).reshape(1, 1, n, h * d)
    rs, off = vf.stride(), vf.storage_offset()
    ve = vf.as_strided((1, 1, (n + 1) // 2, h * d), (rs[0], rs[0], 2 * rs[2], 1), off)
    vo = vf.as_strided((1, 1, n // 2, h * d), (rs[0], rs[0], 2 * rs[2], 1), off + rs[2])
    vt = torch.full((1, h, d, n_pad), float("nan"), device="cuda", dtype=torch.float16)
    eye = torch.eye(16, device="cuda", dtype=torch.bfloat16).view(1, 1, 16, 16)
    K3._launch(K3._kernels()["vt"], ve, vo, eye, vt)
    torch.cuda.synchronize()
    nf = n // 128 * 128
    print(f"  pre-fix kernel, real columns (< n) only: bad "
          f"{int((~((vt == ref) | (torch.isnan(vt) & torch.isnan(ref))))[..., :nf].sum())} in the first {nf}")
    bad = ~((vt == ref) | (torch.isnan(vt) & torch.isnan(ref)))
    cols = bad.any(dim=2)[0]          # [h, n_pad]
    idx = cols.nonzero()
    print(f"n {n}: bad elems {int(bad.sum())}, bad cols {idx[:, 1].unique().tolist()[:20]} "
          f"(of {n_pad}), nan in vt {int(torch.isnan(vt).sum())}", flush=True)
