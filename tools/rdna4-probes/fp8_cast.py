"""fp8 training casts and GEMMs on gfx1201 (hk.ops.fp8_cast, hk.ops.gemm.scaled_mm), against what Megatron-LM with
Transformer Engine runs, one process.

    HK_ARCH=gfx1201 python tools/rdna4-probes/fp8_cast.py      # needs ROCm transformer_engine for the baseline

1. swiglu_bwd_cast_transpose against Megatron's swiglu_back (torch.compile, as its jit_fuser) then TE's quantize:
   fp8 codes compared, then timed, columnwise copy dense and padded, and the rowwise-only ablation.
2. The wgrad GEMM, scaled_mm(dY^T, X^T, accumulate=True) into fp32, dense and padded operands, against
   torch._scaled_mm (hipBLASLt, fp32 out, no accumulate: a lower bound for it), stock and TunableOp-tuned.
"""
import statistics

import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from hk.ops import fp8_cast as H
from hk.ops.gemm import scaled_mm


@torch.compile
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat((g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1)


def t(f, reps=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    r = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            f()
        e1.record()
        torch.cuda.synchronize()
        r.append(e0.elapsed_time(e1) / reps)
    return statistics.median(r)


def act():
    for m, f in ((2048, 8192), (4096, 8192)):
        g = torch.randn(m, f, device="cuda", dtype=torch.bfloat16) * 0.01
        y = torch.randn(m, 2 * f, device="cuda", dtype=torch.bfloat16) * 2
        scale = torch.full((1,), 2000.0, device="cuda")
        am_te, am_hk = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
        q = Float8Quantizer(scale, am_te, tex.DType.kFloat8E5M2, rowwise=True, columnwise=True)
        ref = q(swiglu_back(g, y))
        rw, cw = H.swiglu_bwd_cast_transpose(g, y, scale, am_hk)
        tr, tc = ref._data.view(torch.uint8), ref._transpose.view(torch.uint8)
        hr, hc = rw.contiguous().view(torch.uint8), cw.contiguous().view(torch.uint8)
        print(f"M{m} F{f}: fp8 codes differing rowwise {(hr != tr).sum().item()}/{tr.numel()} "
              f"(max {(hr.int() - tr.int()).abs().max().item()}), columnwise {(hc != tc).sum().item()}; "
              f"amax TE {am_te.item():.6g} hk {am_hk.item():.6g}", flush=True)
        base = t(lambda: q(swiglu_back(g, y)))
        r = {k: t(lambda kw=kw: H.swiglu_bwd_cast_transpose(g, y, scale, am_hk, **kw))
             for k, kw in (("dense", {}), ("padded", dict(pad_col=64)), ("rowwise only", dict(colwise=False)))}
        print(f"   Megatron {base * 1000:.0f} us; hk " + ", ".join(f"{k} {v * 1000:.0f} us ({base / v:.2f}x)"
                                                         for k, v in r.items()), flush=True)


def wgrad():
    tun = torch.cuda.tunable
    m = 4096
    for name, n, k in (("qkv", 6144, 2048), ("o", 2048, 2048), ("fc1", 16384, 2048), ("fc2", 2048, 8192)):
        fl = 2 * m * n * k

        def mk(rows, dt, pad):
            return (torch.randn(rows, m + pad, device="cuda") * .5).to(dt)[:, :m]
        sa, sb = torch.tensor(0.5, device="cuda"), torch.tensor(2.0, device="cuda")
        res = {}
        for pa, pb in ((0, 0), (64, 0), (64, 64)):
            a, b = mk(n, torch.float8_e5m2, pa), mk(k, torch.float8_e4m3fn, pb)
            c = torch.randn(n, k, device="cuda")
            want = c + (a.float() @ b.float().T)
            got = scaled_mm(a, b, sa, sb, out=c.clone(), accumulate=True)
            assert ((got - want).abs().max() / want.abs().max()).item() < 1e-3, (name, pa, pb)
            cc = c.clone()
            res[f"hk pad {pa}/{pb}"] = t(lambda: scaled_mm(a, b, sa, sb, out=cc, accumulate=True), 10)
        a, b = mk(n, torch.float8_e5m2, 0), mk(k, torch.float8_e4m3fn, 0)

        def blas():
            return torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.float32)
        tun.enable(True)
        tun.tuning_enable(True)
        blas()
        torch.cuda.synchronize()
        tun.tuning_enable(False)
        res["hipBLASLt tuned"] = t(blas, 10)
        tun.enable(False)
        res["hipBLASLt stock"] = t(blas, 10)
        print(f"wgrad {name} N{n} K{k} M{m}: " + " | ".join(f"{x} {v * 1000:.0f} us {fl / v / 1e9:.0f} TF"
                                                         for x, v in res.items()), flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__)
    act()
    wgrad()
