"""FlyDSL on gfx1201, outside hk: its RDNA4 WMMA GEMM and its dense attention
entry point, each against an fp32 reference.

    FLYDSL_ROOT=/path/to/FlyDSL python3 fly_probe.py

FlyDSL 0.4.0 on a Radeon AI PRO R9700: the GEMM is right (rel err 2.8e-3, as
torch's); the attention never returns -- flash_attn_generic's
raw_ptr_buffer_load_lds is a global->LDS load gfx12 does not have, and LLVM
aborts the process ("LLVM ERROR: Do not know how to expand this operator's
operand"). So the attention runs last.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))
import flydsl  # noqa: E402

BF = torch.bfloat16
print("flydsl", flydsl.__version__, torch.cuda.get_device_properties(0).gcnArchName)

from kernels.gemm.rdna_f16_gemm import create_wmma_gemm_module  # noqa: E402

for m, n, k in [(4096, 4096, 4096), (2048, 6144, 4096)]:
    a = torch.randn(m, k, device="cuda", dtype=BF)
    b = torch.randn(n, k, device="cuda", dtype=BF) / k ** 0.5
    c = torch.empty(m, n, device="cuda", dtype=BF)
    launch, *_ = create_wmma_gemm_module(m, n, k, "bf16", "bf16")
    launch(c, a, b, torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    print(f"gemm {m}x{n}x{k}: rel err {((c.float() - ref).abs().max() / ref.abs().max()).item():.2e}")

from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # noqa: E402

q = torch.randn(1, 4096, 32, 128, device="cuda", dtype=BF)       # BSHD, GQA 4
kv = [torch.randn(1, 4096, 8, 128, device="cuda", dtype=BF) for _ in range(2)]
o = flydsl_flash_attn_func(q, *kv, causal=True)
o = o[0] if isinstance(o, tuple) else o
r = F.scaled_dot_product_attention(*(x.float().transpose(1, 2) for x in (q, *kv)),
                                   is_causal=True, enable_gqa=True).transpose(1, 2)
print(f"attention causal: max abs err {(o.float() - r).abs().max().item():.4f}")
