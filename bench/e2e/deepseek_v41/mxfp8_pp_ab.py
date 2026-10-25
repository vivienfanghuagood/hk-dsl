import statistics, torch
from hk.ops import mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
SH = [(1792, 5120), (2304, 5120), (4096, 1280), (5120, 1152), (5120, 4096), (16384, 1280)]
for M in (2048, 8192):
    for N, K in SH:
        copies = max(1, min(8, (3 * 256 * 2**20) // (N * K)))
        x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        xs = torch.randint(124, 131, (M, K // 32), device="cuda", dtype=torch.uint8)
        ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
        wsb = [torch.randint(124, 131, (N // 32, K // 32), device="cuda", dtype=torch.uint8) for _ in ws]
        wse = [mxfp8.expand_block_scales(s) for s in wsb]
        arms = {"vllm": lambda i: rocm_mxfp8_block32_gemm(x, xs, ws[i], wsb[i], torch.bfloat16),
                "pp256x128": lambda i: mxfp8.gemm_pp(x, ws[i], xs, wse[i], block_n=128),
                "pp128x256": lambda i: mxfp8.gemm_pp(x, ws[i], xs, wse[i], block_n=256, block_m=128)}
        ref = arms["vllm"](0).float()
        errs = {a: ((f(0).float() - ref).abs().max() / ref.abs().max()).item() for a, f in arms.items()}
        res = {a: [] for a in arms}
        for r in range(9):
            order = list(arms.items()); order = order[r % 3:] + order[:r % 3]
            for a, f in order:
                torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for i in range(copies): f(i)
                e1.record(); torch.cuda.synchronize()
                res[a].append(e0.elapsed_time(e1) / copies)
        t = {a: statistics.median(v) * 1e3 for a, v in res.items()}
        print(f"M={M} N={N} K={K}: " + "  ".join(f"{a} {t[a]:.1f}us ({t['vllm']/t[a]:.2f}x, diff {errs[a]:.2g})" for a in arms), flush=True)
