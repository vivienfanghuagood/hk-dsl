"""mx4w (hk.ops.mxfp8.gemm_4w) against vLLM's MXFP8 GEMM and FlyDSL: correctness, then time."""
import statistics, sys, torch
from hk.ops import mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm

sched = dict(eval(sys.argv[1])) if len(sys.argv) > 1 else {}
shapes = [(8192, 8192, 8192), (5120, 2304, 5120), (5120, 5120, 4096), (5120, 12800, 6144), (8192, 16384, 1280)]
for M, N, K in shapes:
    copies = max(1, min(4, (3 * 256 * 2**20) // (N * K)))
    x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.randint(120, 135, (M, K // 32), device="cuda", dtype=torch.uint8)
    xp = mxfp8.pack_scales(xs)
    ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
    wsb = [torch.randint(120, 135, (N // 32, K // 32), device="cuda", dtype=torch.uint8) for _ in ws]
    wp = [mxfp8.pack_scales(mxfp8.expand_block_scales(s)) for s in wsb]
    arms = {"vllm": lambda i: rocm_mxfp8_block32_gemm(x, xs, ws[i], wsb[i], torch.bfloat16),
            "hk4w": lambda i: mxfp8.gemm_4w(x, ws[i], xp, wp[i], **sched)}
    ref = arms["vllm"](0).float()
    o = arms["hk4w"](0).float()
    err = ((o - ref).abs().max() / ref.abs().max()).item()
    same = torch.equal(o, ref)
    res = {a: [] for a in arms}
    for r in range(7):
        for a, f in (list(arms.items()) if r % 2 else list(arms.items())[::-1]):
            torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for i in range(copies): f(i)
            e1.record(); torch.cuda.synchronize()
            res[a].append(e0.elapsed_time(e1) / copies)
    t = {a: statistics.median(v) * 1e3 for a, v in res.items()}
    fl = 2 * M * N * K
    print(f"M={M} N={N} K={K}: vllm {t['vllm']:.1f}us {fl/t['vllm']/1e6:.0f}TF/s  hk4w {t['hk4w']:.1f}us "
          f"{fl/t['hk4w']/1e6:.0f}TF/s ({t['vllm']/t['hk4w']:.2f}x)  err {err:.2g} bitexact {same}", flush=True)
