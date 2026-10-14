#!/usr/bin/env python3
"""Which FlyDSL kernels run on this GPU, and whether FlyDSL and hk share a process.

FlyDSL documents gfx942/gfx950/gfx1250/gfx1201; gfx1100 support arrived with a
GFX11 WMMA atom (ROCm/FlyDSL#544) and kernels/gemm/rdna3_f16_gemm.py (#567).
Before a scoreboard can put FlyDSL next to hk, each of its kernels has to be
shown to build and be right here. And the two each embed an MLIR (hk._mlir,
flydsl's _mlir), so loading both into one process -- which the measurement
protocol needs -- is checked, not assumed: an hk kernel is traced before
FlyDSL is imported and launched after every FlyDSL kernel ran.

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/flydsl_probe.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))

import torch                                       # noqa: E402
import torch.nn.functional as F                    # noqa: E402

BF16 = torch.bfloat16


def rel(got, want):
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max()).item()


def probe(name, fn):
    try:
        err = fn()
        ok = err < 3e-2
        print(f"  {name:<24} {'ok' if ok else 'WRONG'}  rel err {err:.2e}",
              flush=True)
        return ok
    except Exception as e:                         # noqa: BLE001
        line = str(e).strip().splitlines()[0] if str(e).strip() else ""
        print(f"  {name:<24} FAILED  {type(e).__name__}: {line[:140]}",
              flush=True)
        if os.environ.get("PROBE_TRACE"):
            traceback.print_exc()
        return False


def gemm():
    from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module
    m = n = k = 4096
    a = torch.randn(m, k, device="cuda", dtype=BF16)
    bt = torch.randn(n, k, device="cuda", dtype=BF16)
    c = torch.empty(m, n, device="cuda", dtype=BF16)
    launch, *_ = create_wmma_gemm_module(m, n, k, "bf16", "bf16")
    launch(c, a, bt, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return rel(c, a.float() @ bt.float().T)


def rmsnorm():
    import flydsl.compiler as flyc
    from kernels.norm.rmsnorm_kernel import build_rmsnorm_module
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16)
    y = torch.empty_like(x)
    s = torch.cuda.current_stream()
    f = flyc.compile(build_rmsnorm_module(n, "bf16"), x, w, y, m, s)
    f(x, w, y, m, s)
    torch.cuda.synchronize()
    xf = x.float()
    return rel(y, xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
               * w.float())


def fused_add_rmsnorm():
    import flydsl.compiler as flyc
    from kernels.norm.rmsnorm_kernel import build_fused_add_rmsnorm_module
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    r = torch.randn_like(x)
    w = torch.rand(n, device="cuda", dtype=BF16)
    y, r_out = torch.empty_like(x), torch.empty_like(x)
    s = torch.cuda.current_stream()
    f = flyc.compile(build_fused_add_rmsnorm_module(n, "bf16"), x, r, w, y,
                     r_out, m, s)
    f(x, r, w, y, r_out, m, s)
    torch.cuda.synchronize()
    h = x.float() + r.float()
    return rel(y, h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
               * w.float())


def softmax():
    from kernels.norm.softmax_kernel import build_softmax_module
    m, n = 8192, 4096
    a = torch.randn(m, n, device="cuda", dtype=BF16)
    c = torch.empty_like(a)
    build_softmax_module(m, n, "bf16")(a, c, m, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return rel(c, torch.softmax(a.float(), dim=-1))


def flash_attn():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func
    b, s, h, hkv, d = 1, 4096, 32, 8, 128
    q = torch.randn(b, s, h, d, device="cuda", dtype=BF16)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    out = flydsl_flash_attn_func(q, k, v, causal=True, num_kv_heads=hkv)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True, enable_gqa=True).transpose(1, 2)
    return rel(out, ref)


def main() -> int:
    print(f"{torch.cuda.get_device_name()} "
          f"{torch.cuda.get_device_properties(0).gcnArchName}")
    from hk.ops import norm as hk_norm
    hk_k = next(iter(hk_norm.KERNELS.values()))
    hk_k.trace()                                   # hk's MLIR first

    import flydsl
    print(f"flydsl {getattr(flydsl, '__version__', '?')} from "
          f"{Path(flydsl.__file__).parent}")
    results = {name: probe(name, fn) for name, fn in (
        ("gemm bf16 4096^3", gemm), ("rmsnorm 8192x4096", rmsnorm),
        ("fused add+rmsnorm", fused_add_rmsnorm),
        ("softmax 8192x4096", softmax), ("flash attn causal gqa", flash_attn))}

    def hk_after():
        import hk
        x = torch.randn(8192, 4096, device="cuda", dtype=BF16)
        y = hk.ops.rmsnorm(x)                      # unweighted
        xf = x.float()
        return rel(y, xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))
    results["hk after flydsl"] = probe("hk rmsnorm after flydsl", hk_after)
    return 0 if results["hk after flydsl"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
