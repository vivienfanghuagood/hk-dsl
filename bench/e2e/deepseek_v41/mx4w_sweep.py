"""mx4w schedules against vLLM's MXFP8 GEMM at DeepSeek-V4.1-Flash prefill shapes (one process, interleaved)."""
import statistics, sys, torch
from hk.ops import mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm

SCHEDS = eval(sys.argv[1])
MS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2048, 5120, 8192]
SH = [(1792, 5120), (2304, 5120), (5120, 1152), (5120, 4096), (16384, 1280), (12800, 6144)]
for M in MS:
    for N, K in SH:
        copies = max(1, min(4, (3 * 256 * 2**20) // (N * K)))
        x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        xs = torch.randint(120, 135, (M, K // 32), device="cuda", dtype=torch.uint8)
        xp = mxfp8.pack_scales(xs)
        ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
        wsb = [torch.randint(120, 135, (N // 32, K // 32), device="cuda", dtype=torch.uint8) for _ in ws]
        wp = [mxfp8.pack_scales(mxfp8.expand_block_scales(s)) for s in wsb]
        arms = {"vllm": lambda i: rocm_mxfp8_block32_gemm(x, xs, ws[i], wsb[i], torch.bfloat16)}
        ref = arms["vllm"](0)
        for k, sc in SCHEDS.items():
            if N % sc.get("block_n", 256):
                continue
            f = (lambda sc: lambda i: mxfp8.gemm_4w(x, ws[i], xp, wp[i], **sc))(sc)
            try:
                o = f(0)
            except Exception as e:  # noqa: BLE001
                print(f"  {k}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
            if not torch.equal(o, ref):
                print(f"  {k}: NOT bit-identical", flush=True); continue
            arms[k] = f
        res = {a: [] for a in arms}
        for r in range(7):
            order = list(arms.items()); order = order[r % len(order):] + order[:r % len(order)]
            for a, f in order:
                torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for i in range(copies): f(i)
                e1.record(); torch.cuda.synchronize()
                res[a].append(e0.elapsed_time(e1) / copies)
        t = {a: statistics.median(v) * 1e3 for a, v in res.items()}
        fl = 2 * M * N * K
        print(f"M={M} N={N} K={K}: vllm {t['vllm']:.1f}us {fl/t['vllm']/1e6:.0f}TF/s | " + "  ".join(
            f"{a} {fl/t[a]/1e6:.0f} ({t['vllm']/t[a]:.2f}x)" for a in arms if a != "vllm"), flush=True)
