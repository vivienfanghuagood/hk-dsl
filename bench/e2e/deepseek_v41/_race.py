import torch, sys
from hk.ops import mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
torch.manual_seed(0)
S = {'256a1': dict(scale_ahead=1), '256x128a1': dict(block_n=128, split_n=1, scale_ahead=1)}
M, N, K = 2048, 16384, 1280
x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
xs = torch.randint(120, 135, (M, K // 32), device="cuda", dtype=torch.uint8)
w = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
wsb = torch.randint(120, 135, (N // 32, K // 32), device="cuda", dtype=torch.uint8)
ref = rocm_mxfp8_block32_gemm(x, xs, w, wsb, torch.bfloat16)
xp = mxfp8.pack_scales(xs); wp = mxfp8.pack_scales(mxfp8.expand_block_scales(wsb))
for k, sc in list(S.items()) + list(S.items()):
    res = []
    for _ in range(6):
        out = torch.full((M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        mxfp8.gemm_4w(x, w, xp, wp, out=out, **sc)
        res.append(int((out != ref).sum()))
    print(k, res, flush=True)
