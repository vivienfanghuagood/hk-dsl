"""Opportunity table, MI355X: DeepSeek-V4.1-Flash's MXFP8 32x32-block GEMMs at prefill M against
every gfx950 baseline in the vLLM image -- vLLM's rocm_mxfp8_block32_gemm (Triton dot_scaled, what the
engine runs), FlyDSL's gfx950 MX bmm (aiter), hipBLASLt MX through torch._scaled_mm -- and hk's, as
TF/s and % of the 5.03 PF/s dense fp8 peak. Interleaved, CUDA events, weights rotated past the LLC.

    hkx950v <checkout> GPU python3 bench/e2e/deepseek_v41/mxfp8_opp.py [M ...]
"""
import statistics, sys, torch

PEAK = 5.03e15
# (N, K) of the TP2 projections the prefill trace showed (runs/pre-shapes), and the indexer's
SH = [(1792, 5120), (2304, 5120), (5120, 1152), (5120, 4096), (16384, 1280), (12800, 6144)]


def arms_for(x, xs, ws, wsb):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
    arms = {"vllm": lambda i: rocm_mxfp8_block32_gemm(x, xs, ws[i], wsb[i], torch.bfloat16)}
    M, K = x.shape
    N = ws[0].shape[0]
    try:
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import run_bmm_a8w8_mxfp8_gfx950
        from aiter.ops.shuffle import shuffle_weight
        wsh = [shuffle_weight(w.view(1, N, K), layout=(16, 16)) for w in ws]
        e8 = torch.float8_e8m0fnu
        x3, xs3 = x.view(M, 1, K), xs.view(e8).view(M, 1, K // 32)
        wsb3 = [s.view(e8).view(1, N // 32, K // 32) for s in wsb]
        out = torch.empty(M, 1, N, device="cuda", dtype=torch.bfloat16)
        arms["flydsl"] = lambda i: run_bmm_a8w8_mxfp8_gfx950(x3, wsh[i], xs3, wsb3[i], out).view(M, N)
    except Exception as e:  # noqa: BLE001
        print(f"  flydsl unavailable: {type(e).__name__}: {str(e)[:160]}", flush=True)
    try:
        e8 = torch.float8_e8m0fnu
        sa = xs.view(e8)
        sbs = [s.repeat_interleave(32, 0).contiguous().view(e8) for s in wsb]
        arms["hipblaslt"] = lambda i: torch._scaled_mm(x, ws[i].t(), sa, sbs[i], out_dtype=torch.bfloat16)
        arms["hipblaslt"](0)
    except Exception as e:  # noqa: BLE001
        arms.pop("hipblaslt", None)
        print(f"  hipblaslt MX unavailable: {type(e).__name__}: {str(e)[:160]}", flush=True)
    try:
        from hk.ops import mxfp8
        wse = [mxfp8.expand_block_scales(s) for s in wsb]
        arms["hk"] = lambda i: mxfp8.gemm_pp(x, ws[i], xs, wse[i], block_n=128)
    except Exception as e:  # noqa: BLE001
        print(f"  hk unavailable: {type(e).__name__}: {str(e)[:160]}", flush=True)
    return arms


def main():
    torch.manual_seed(0)
    ms = [int(a) for a in sys.argv[1:]] or [2048, 5120, 8192]
    for M in ms:
        for N, K in SH:
            copies = max(1, min(8, (3 * 256 * 2**20) // (N * K)))
            x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
            xs = torch.randint(124, 131, (M, K // 32), device="cuda", dtype=torch.uint8)
            ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
            wsb = [torch.randint(124, 131, (N // 32, K // 32), device="cuda", dtype=torch.uint8) for _ in ws]
            arms = arms_for(x, xs, ws, wsb)
            ref = arms["vllm"](0).float()
            ok = {}
            for a, f in arms.items():
                try:
                    err = ((f(0).float() - ref).abs().max() / ref.abs().max()).item()
                except Exception as e:  # noqa: BLE001
                    print(f"  M={M} N={N} K={K} {a} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
                    continue
                if err < 1e-2:
                    ok[a] = f
                else:
                    print(f"  M={M} N={N} K={K} {a} wrong: {err:.3g}", flush=True)
            res = {a: [] for a in ok}
            for r in range(9):
                order = list(ok.items()); order = order[r % len(order):] + order[:r % len(order)]
                for a, f in order:
                    torch.cuda.synchronize(); torch.cuda._sleep(2_000_000)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for i in range(copies): f(i)
                    e1.record(); torch.cuda.synchronize()
                    res[a].append(e0.elapsed_time(e1) / copies)
            t = {a: statistics.median(v) * 1e3 for a, v in res.items()}
            fl = 2 * M * N * K
            print(f"MXFP8 M={M} N={N} K={K}: " + "  ".join(
                f"{a} {t[a]:.1f}us {fl / t[a] / 1e6:.0f}TF/s ({fl / t[a] / 1e6 / PEAK * 1e12 * 100:.0f}%)" for a in ok),
                flush=True)


if __name__ == "__main__":
    main()
