"""hk.ops.mxfp8.gemm vs vLLM's rocm_mxfp8_block32_gemm (Triton tl.dot_scaled, tuned per DeepSeek-V4.1 shape),
interleaved, CUDA events, weights rotated past the LLC."""
import statistics, torch
from hk.ops import mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
SH = [(1792, 5120), (2304, 5120), (4096, 1280), (5120, 1152), (5120, 4096), (16384, 1280)]
for M in (2048, 8192):
    for N, K in SH:
        copies = max(1, (3 * 256 * 2**20) // (N * K))
        x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        xs = torch.randint(124, 131, (M, K // 32), device="cuda", dtype=torch.uint8)
        ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(min(copies, 8))]
        wsb = [torch.randint(124, 131, (N // 32, K // 32), device="cuda", dtype=torch.uint8) for _ in ws]
        wse = [mxfp8.expand_block_scales(s) for s in wsb]
        # correctness of both against each other
        o1 = mxfp8.gemm(x, ws[0], xs, wse[0]); o2 = rocm_mxfp8_block32_gemm(x, xs, ws[0], wsb[0], torch.bfloat16)
        err = ((o1.float() - o2.float()).abs().max() / o2.float().abs().max()).item()
        arms = {"vllm": lambda i: rocm_mxfp8_block32_gemm(x, xs, ws[i], wsb[i], torch.bfloat16),
                "hk": lambda i: mxfp8.gemm(x, ws[i], xs, wse[i])}
        res = {a: [] for a in arms}
        for r in range(9):
            for a, f in (list(arms.items()) if r % 2 else list(arms.items())[::-1]):
                torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for i in range(len(ws)): f(i)
                e1.record(); torch.cuda.synchronize()
                res[a].append(e0.elapsed_time(e1) / len(ws))
        t = {a: statistics.median(v) * 1e3 for a, v in res.items()}
        f = 2 * M * N * K
        print(f"M={M} N={N} K={K}: vllm {t['vllm']:.1f} us ({f/t['vllm']/1e6:.0f} TF/s) hk {t['hk']:.1f} us ({f/t['hk']/1e6:.0f} TF/s) hk/vllm speed {t['vllm']/t['hk']:.2f}x  diff {err:.3g}", flush=True)
