"""The cross-architecture questions of docs/dx/api-consistency.md, asked of the
GPU this runs on: one JSON record per probe (max abs error against an fp32
torch reference, or the error raised). Run it on each GPU and compare:

    HK_ARCH=gfx950 python3 tools/api_consistency.py > gfx950.json
"""
import json
import torch
import torch.nn.functional as F
import hk
from hk.target import current_arch

dev = "cuda"
bf = torch.bfloat16
out = {}


def probe(name):
    def deco(fn):
        try:
            r = fn()
            out[name] = {"ok": True, "result": r}
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
        torch.cuda.synchronize()
        return fn
    return deco


def err(got, want):
    return float((got.float() - want.float()).abs().max())


@probe("matmul_tie_rounding")
def _():
    a = torch.zeros(128, 64, dtype=bf, device=dev); b = torch.zeros(128, 64, dtype=bf, device=dev)
    a[:, 0] = 1; a[:, 1] = 2 ** -8; b[:, 0] = 1; b[:, 1] = 1
    return hk.ops.matmul(a, b)[0, 0].item()


@probe("matmul_k0")
def _():
    o = hk.ops.matmul(torch.empty(32, 0, dtype=bf, device=dev), torch.empty(128, 0, dtype=bf, device=dev))
    return [list(o.shape), float(o.float().abs().max()) if o.numel() else None]


@probe("matmul_noncontig_b_decode")
def _():
    x = torch.randn(16, 4096, dtype=bf, device=dev); w = torch.randn(4096, 4096, dtype=bf, device=dev).t()
    return err(hk.ops.matmul(x, w), x.float() @ w.float().t())


@probe("matmul_noncontig_b_prefill")
def _():
    x = torch.randn(1024, 4096, dtype=bf, device=dev); w = torch.randn(4096, 4096, dtype=bf, device=dev).t()
    return err(hk.ops.matmul(x, w), x.float() @ w.float().t())


@probe("matmul_k_not_64")
def _():
    x = torch.randn(256, 96, dtype=bf, device=dev); w = torch.randn(256, 96, dtype=bf, device=dev)
    return err(hk.ops.matmul(x, w), x.float() @ w.float().t())


@probe("attention_n64")
def _():
    q = torch.randn(1, 4, 64, 128, dtype=bf, device=dev)
    return err(hk.attention(q, q, q), F.scaled_dot_product_attention(q.float(), q.float(), q.float()))


@probe("attention_n256_d64")
def _():
    q = torch.randn(1, 4, 256, 64, dtype=bf, device=dev)
    return err(hk.attention(q, q, q), F.scaled_dot_product_attention(q.float(), q.float(), q.float()))


@probe("silu_mul_strided_halves")
def _():
    gu = torch.randn(64, 2 * 4096, dtype=bf, device=dev)
    g, u = gu[:, :4096], gu[:, 4096:]
    return err(hk.ops.silu_mul(g, u), F.silu(g.float()) * u.float())


@probe("silu_mul_small_8x100")
def _():
    a = torch.randn(8, 100, dtype=bf, device=dev); b = torch.randn_like(a)
    return err(hk.ops.silu_mul(a, b), F.silu(a.float()) * b.float())


@probe("add_100x1000")
def _():
    a = torch.randn(100, 1000, dtype=bf, device=dev); b = torch.randn_like(a)
    return err(hk.ops.add(a, b), a.float() + b.float())


@probe("gelu_bf16")
def _():
    a = torch.randn(64, 1024, dtype=bf, device=dev)
    return err(hk.ops.gelu(a), F.gelu(a.float(), approximate="tanh"))


@probe("gelu_fp32_is_tanh_form")
def _():
    a = torch.randn(64, 1024, dtype=torch.float32, device=dev) * 3
    g = hk.ops.gelu(a)
    return {"vs_tanh": err(g, F.gelu(a, approximate="tanh")), "vs_erf": err(g, F.gelu(a))}


@probe("quantize_8x100")
def _():
    x = torch.randn(8, 100, dtype=bf, device=dev)
    q, s = hk.ops.quantize(x)
    return err(q.float() * s[..., None], x.float())


@probe("rotary_neox")
def _():
    T, H, D = 16, 8, 128
    x = torch.randn(T, H, D, dtype=bf, device=dev)
    cs = torch.randn(T, D, dtype=torch.float32, device=dev)
    o = hk.ops.rotary(x, cs)
    return list(o.shape)


@probe("rms_norm_rows3_cols4096")
def _():
    x = torch.randn(3, 4096, dtype=bf, device=dev); w = torch.randn(4096, dtype=bf, device=dev)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    return err(hk.ops.rms_norm(x, w), ref)


@probe("rmsnorm_rows3_cols4096")
def _():
    x = torch.randn(3, 4096, dtype=bf, device=dev)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return err(hk.ops.rmsnorm(x), ref)


@probe("softmax_rows3_cols100")
def _():
    x = torch.randn(3, 100, dtype=bf, device=dev)
    return err(hk.ops.softmax(x), torch.softmax(x.float(), -1))


@probe("layernorm_rows16_cols65536")
def _():
    x = torch.randn(16, 65536, dtype=bf, device=dev)
    return err(hk.ops.layernorm(x), F.layer_norm(x.float(), (65536,)))


@probe("paged_kernels_get_vs_getitem")
def _():
    from hk.ops import paged
    return [getattr(paged.KERNELS.get(128), "name", None), paged.KERNELS[128].name]


@probe("paged_merge_kernel_12")
def _():
    from hk.ops import paged
    return paged.merge_kernel(128, 12).name


@probe("import_hk_resolves_arch")
def _():
    import hk.target as t
    return list(t._CURRENT)


print(json.dumps({"arch": current_arch(), "device": torch.cuda.get_device_name(0),
                  "torch": torch.__version__, "probes": out}, indent=1))
