"""Every hk.ops route touches nothing past the end of any tensor, at ragged shapes.

A read past a tensor usually lands in the caching allocator's next block and goes unnoticed --
the results stay bit-exact -- until the tensor happens to end its segment and the page after it
is unmapped. In vLLM that is a memory fault some runs and not others: mxfp8.quant's whole-tile
copy of x did it in B1's real-weight GSM8K runs (PR #47). Here every allocation of a child
process ends flush against an unmapped VMM granule (_fence.py, fence_alloc.cpp), so any such
access faults at once, on every run, on gfx950 and gfx1100 alike.

One child per op family; each case prints its name before it launches and synchronizes after,
so a failure names the call (HK_FENCE_EACH=1: a child per case, every failing case reported).
The shapes are the ones a tile gets wrong: M (or the sequence) at 1, a tile - 1, a tile + 1,
odd values, and the DeepSeek-V4.1 shapes; N tails where the op takes them. Inputs are fresh
tensors (a view's fence is its base's end), sized to whole 16 bytes where it matters, so the
allocation ends exactly where the tensor does. Results are checked against torch, on the host.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from _fence import fenced, fenced_cases  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = pytest.mark.gpu

ARCH = current_arch()
HERE = os.path.dirname(os.path.abspath(__file__))
gfx950 = pytest.mark.skipif(ARCH != "gfx950", reason="gfx950's routes")
gfx1100 = pytest.mark.skipif(ARCH != "gfx1100", reason="gfx1100's routes")

# Imports and the relative-error check every family uses.
COMMON = f"ARCH = {ARCH!r}\nimport sys\nsys.path.insert(0, {HERE!r})\n" + """
import hk
import torch.nn.functional as F
import _attn_ref

def rel(what, got, want, tol):
    # Both maxima in the message: an all-zero reference makes the error max|got| / 1e-30,
    # which reads like a garbage output but is not one.
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    top = want.abs().max().item()
    err = ((got - want).abs().max() / max(top, 1e-30)).item()
    assert err < tol, (f"{what}: relative error {err:.3g} >= {tol} (max |got| "
                       f"{got.abs().max().item():.4g}, max |want| {top:.4g}"
                       + (", an all-zero reference" if top == 0 else "") + ")")

def attn(what, got, want):
    # attention's bounds, the GPU tests' own (_attn_ref.check)
    try:
        _attn_ref.check(got.detach().float().cpu(), want.detach().float().cpu())
    except AssertionError as e:
        raise AssertionError(f"{what}: {e}") from None

def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)

REFUSED = object()

def built(label, fn, also=()):
    # step(), or REFUSED for a kernel this compiler's build refuses (the spill gate, or `also`:
    # HipKittens' kernels this hipcc does not build): not ours to test here, and the op's own
    # tests say so
    from hk.runtime.resources import ResourceError
    try:
        return step(label, fn)
    except (ResourceError, *also) as e:
        print(f"{label}: refused by the build ({str(e).splitlines()[:2]})", flush=True)
        return REFUSED
"""


def test_fence_catches_one_element_past_the_end():
    """The harness itself: a load one element past a tensor must fail the run."""
    with pytest.raises(pytest.fail.Exception, match="past the end"):
        fenced("""
        import triton
        import triton.language as tl

        @triton.jit
        def _peek(x_ptr, out_ptr, n):
            i = tl.arange(0, 1)
            tl.store(out_ptr + i, tl.load(x_ptr + n + i))

        x = torch.zeros(1000, device="cuda")
        out = torch.empty(1, device="cuda")
        step("in bounds", lambda: _peek[(1,)](x, out, 999))
        step("one past the end", lambda: _peek[(1,)](x, out, 1000))
        """)


# --------------------------------------------------------------------------- both GPUs


def test_the_group_copies_stop_at_the_last_row():
    """The DSL's own copies at a ragged last tile: g.load (synchronous) and g.store through
    LDS. On gfx950 include/cdna4's group load spanned the
    tile's whole rows and its store wrote them all (mxfp8.quant's fault, #47); hk/cdna.cuh's
    stop at the tensor's last row."""
    fenced_cases(COMMON, {
        f"lds copy {m}x{k} {dt}": f"""
        from hk.target import current_arch
        dt = hk.{dt}
        st = hk.ST(dt, 64, 128)

        def lds_copy(x: hk.Tensor[dt], y: hk.Tensor[dt]):
            g = hk.group(4)
            Xs = hk.lds.alloc(st, name="Xs")
            r0 = hk.arch.block_idx.y * 64
            c0 = hk.arch.block_idx.x * 128
            g.load(Xs, x.at[r0, c0])
            hk.arch.barrier()
            g.store(y.at[r0, c0], Xs)

        k1 = hk.kernel(lds_copy, arch=current_arch(), warps=4, name="fence_lds_copy_{dt}",
                       grid=lambda p: (p.x.cols // 128, hk.cdiv(p.x.rows, 64), 1))
        x = rnd({m}, {k}, dtype=torch.{"bfloat16" if dt == "bf16" else "float16"})
        y = torch.zeros_like(x)
        step("g.load / g.store", lambda: k1(x, y))
        same("lds copy", y, x)
        """
        for m, k in ((1, 128), (63, 256), (65, 128), (130, 384), (1000, 128))
        for dt in ("bf16", "fp16")
    })


def test_elementwise():
    """hk.ops' pointwise ops on strips: any size, the tail bounded."""
    fenced_cases(COMMON, {
        f"{op} {shape} {dt}": f"""
        a = rnd(*{shape}, dtype=torch.{dt}); b = rnd(*{shape}, dtype=torch.{dt})
        got = step("{op} {shape} {dt}", lambda: hk.ops.{op}(a, b))
        rel("{op}", got, torch.{ref}(a.float(), b.float()), 1e-2)
        """
        for op, ref in (("add", "add"), ("mul", "mul"), ("maximum", "maximum"))
        for shape in ((1,), (17,), (3, 1001), (2, 3, 4097), (1, 4104), (64, 1032))
        for dt in ("bfloat16", "float32")
    } | {
        f"{op} {shape}": f"""
        a = rnd(*{shape}).clamp(-8, 8)
        got = step("{op} {shape}", lambda: hk.ops.{op}(a))
        rel("{op}", got, torch.{op}(a.float()), 1e-2)
        """
        for op in ("exp", "relu", "neg", "abs")
        for shape in ((17,), (3, 1001), (5, 4104))
    } | {
        f"gelu {shape}": f"""
        a = rnd(*{shape}, dtype=torch.float32)
        rel("gelu", step("gelu {shape}", lambda: hk.ops.gelu(a)), F.gelu(a, approximate="tanh"), 1e-3)
        """
        for shape in ((17,), (3, 1001))
    })


def test_norms_and_quantize():
    """The row ops: one workgroup a row, any rows and columns."""
    shapes = ((1, 64), (3, 65), (2, 129), (17, 1001), (7, 1032), (5, 3000), (3, 4100),
              (4101, 64))
    fenced_cases(COMMON, {
        f"norms {m}x{n}": f"""
        x = rnd({m}, {n}); r = rnd({m}, {n}); w = torch.rand({n}, device="cuda") + 0.5
        xf = x.float()
        rms = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        rel("rmsnorm", step("rmsnorm", lambda: hk.ops.rmsnorm(x)), rms, 2e-2)
        rel("rms_norm", step("rms_norm", lambda: hk.ops.rms_norm(x, w.bfloat16())),
            rms * w.bfloat16().float(), 2e-2)
        rel("layernorm", step("layernorm", lambda: hk.ops.layernorm(x)),
            F.layer_norm(xf, ({n},), eps=1e-5), 2e-2)
        rel("softmax", step("softmax", lambda: hk.ops.softmax(x)), torch.softmax(xf, -1), 2e-2)
        xi, ri = x.clone(), r.clone()
        step("fused_add_rms_norm", lambda: hk.ops.fused_add_rms_norm(xi, ri, w.bfloat16()))
        same("residual", ri, (x.float() + r.float()).bfloat16())
        if ARCH == "gfx950" or {n} % 2 == 0 or ({m} >= 16 and {n} >= 64):   # else refused
            q, s = step("quantize", lambda: hk.ops.quantize(x * 3))
            amax = (x * 3).float().abs().amax(-1).clamp_min(1e-12)
            rel("quantize scales", s, amax / 127, 1e-5)
        if {m} >= 16 and {n} >= 64:
            d = step("dequantize", lambda: hk.ops.dequantize(q, s))
            rel("dequantize", d, q.float() * s.float()[:, None], 1e-2)
        """
        for m, n in shapes
    } | {
        f"norms (tile layout) {m}x{n}": f"""
        from hk.ops import norm
        norm.LAYOUT = "tile"
        x = rnd({m}, {n}); xf = x.float()
        rel("rmsnorm", step("rmsnorm", lambda: hk.ops.rmsnorm(x)),
            xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6), 2e-2)
        rel("layernorm", step("layernorm", lambda: hk.ops.layernorm(x)),
            F.layer_norm(xf, ({n},), eps=1e-5), 2e-2)
        rel("softmax", step("softmax", lambda: hk.ops.softmax(x)), torch.softmax(xf, -1), 2e-2)
        norm.LAYOUT = "strip"
        """
        for m, n in ((16, 64), (17, 65), (33, 1001), (100, 200))
    })


def test_silu_mul_and_rotary():
    """silu_mul on fresh halves and on a gate/up output in place; rotary with positions that
    reach the table's last row."""
    cases = {
        f"silu_mul {m}x{i}": f"""
        g, u = rnd({m}, {i}), rnd({m}, {i})
        rel("silu_mul", step("silu_mul", lambda: hk.ops.silu_mul(g, u)),
            F.silu(g.float()) * u.float(), 1e-2)
        x = rnd({m}, 2 * {i})
        rel("silu_mul halves", step("halves", lambda: hk.ops.silu_mul(x[:, :{i}], x[:, {i}:])),
            F.silu(x[:, :{i}].float()) * x[:, {i}:].float(), 1e-2)
        """
        for m, i in ((1, 14336), (3, 72), (5, 1000), (17, 1032), (33, 2304))
    }
    cases |= {
        f"rotary {t}x{h}x{d}": f"""
        x = rnd({t}, {h}, {d})
        ang = torch.rand({t} + 3, {d} // 2, device="cuda") * 6.28
        cs = torch.cat((ang.cos(), ang.sin()), -1).contiguous()
        p = torch.arange({t}, device="cuda", dtype=torch.int32) + 3      # the last is P - 1
        rows = cs[p.long()]
        c, s = rows[:, None, :{d} // 2].float(), rows[:, None, {d} // 2:].float()
        x1, x2 = x[..., :{d} // 2].float(), x[..., {d} // 2:].float()
        want = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1)
        rel("rotary", step("rotary", lambda: hk.ops.rotary(x, cs, positions=p)), want, 1e-2)
        """
        for t, h, d in ((1, 32, 128), (5, 3, 64), (7, 8, 256), (33, 40, 128), (129, 8, 128))
    }
    cases |= {
        f"rope {bh}x{seq}x{d}": f"""
        x = rnd({bh}, {seq}, {d})
        ang = torch.rand({seq}, {d} // 2, device="cuda") * 6.28
        cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
        x1, x2 = x[..., :{d} // 2].float(), x[..., {d} // 2:].float()
        want = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
        rel("rope", step("rope", lambda: hk.ops.rope(x, cos, sin)), want, 1e-2)
        """
        for bh, seq, d in ((6, 32, 128), (3, 16, 256))
    }
    fenced_cases(COMMON, cases)


def test_gemm_at_ragged_m():
    """hk.ops.matmul (gfx950: gemm_cdna's routes; gfx1100: the WMMA GEMM) and the templates'
    GEMM with a tile operand and a row vector, at M around every tile edge."""
    ms = (1, 15, 16, 17, 31, 33, 63, 65, 127, 129, 255, 257, 513)
    nks = ((256, 192), (1408, 1152)) if ARCH == "gfx950" else ((256, 192), (384, 1024))
    cases = {
        f"matmul M{m} N{n} K{k}": f"""
        a = rnd({m}, {k}); b = rnd({n}, {k}, scale={k} ** -0.5)
        got = step("matmul", lambda: hk.ops.matmul(a, b))
        rel("matmul", got, a.float() @ b.float().T, 2e-2)
        """
        for n, k in nks for m in ms
    }
    cases |= {
        f"templates.gemm M{m}": f"""
        T = hk.templates.gemm
        a = rnd({m}, 640); b = rnd(1408, 640, scale=640 ** -0.5)
        res = rnd({m}, 1408); bias = rnd(1408)
        got = step("residual", lambda: T(a, b, epilogue=T.bias_residual, bias=bias, residual=res))
        rel("bias_residual", got, a.float() @ b.float().T + bias.float() + res.float(), 2e-2)
        """
        for m in (33, 129, 257, 300)
    }
    cases |= {
        f"vllm_gemm.linear M{m}": f"""
        from hk.integration import vllm_gemm
        vllm_gemm.OPTIONAL = {{}}          # hk's paths and the fallback, not FlyDSL's
        x = rnd({m}, 4096); w = rnd(6144, 4096, scale=4096 ** -0.5)
        got = step(f"linear {{vllm_gemm.route_for({m}, 6144, 4096)}}",
                   lambda: vllm_gemm.linear(x, w))
        rel("linear", got, x.float() @ w.float().T, 2e-2)
        """
        for m in (1, 17, 65, 129, 1025)
    }
    fenced_cases(COMMON, cases)


def test_decode_gemm_and_w4a16():
    """The skinny GEMMs: decode_gemm at M 1..64 and the 4-bit-weight decode."""
    cases = {
        f"decode_gemm M{m} N{n} K{k}": f"""
        from hk.ops import decode_gemm
        x = rnd({m}, {k}); w = rnd({n}, {k}, scale={k} ** -0.5)
        rel("decode_gemm", step("decode_gemm", lambda: decode_gemm.matmul(x, w)),
            x.float() @ w.float().T, 2e-2)
        """
        for n, k in ((16, 256), (4096, 4096)) for m in (1, 15, 16, 17, 31, 32, 33, 63, 64)
    }
    cases |= {
        f"w4a16 M{m}": f"""
        from hk.ops import w4a16
        n, k, group = 512, 2048, 128
        q = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
        z = torch.randint(0, 16, (k // group, n), device="cuda", dtype=torch.int32)
        s = (torch.rand(k // group, n, device="cuda") * 0.02 + 0.005).half()
        def pack(v):
            r, c = v.shape
            v = v.reshape(r, c // 8, 8).to(torch.int64)
            shift = torch.tensor([4 * o for o in w4a16.AWQ_ORDER], device=v.device)
            words = (v << shift).sum(-1)
            return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
        qw, sc, zs = w4a16.repack_awq(pack(q), pack(z), s, group, dtype=torch.bfloat16)
        qw, sc, zs = qw.clone(), sc.clone(), zs.clone()
        x = rnd({m}, k)
        got = step("w4a16", lambda: w4a16.matmul(x, qw, sc, zs, group))
        rel("w4a16", got, x.float() @ w4a16.dequant(qw, sc, zs, group).float().t(), 2e-2)
        """
        for m in (1, 7, 16, 17, 33, 64)
    }
    fenced_cases(COMMON, cases)


def test_moe_gemm():
    """vLLM's fused-MoE GEMM (hk.ops.moe), routed and plain, the routing weights a fresh
    (tokens, top_k) tensor."""
    fenced_cases(COMMON, {
        f"moe_gemm tokens{t} bm{bm} {'routed' if routed else 'plain'}": f"""
        from hk.ops import moe
        experts, top_k, n, k = 8, 2, 256, 1024
        a = rnd({t}, k); w = rnd(experts, n, k, scale=k ** -0.5)
        scores = torch.rand({t}, experts, device="cuda")
        tw, ti = scores.topk(top_k, dim=1)
        tw, ti = tw.softmax(dim=1).float().contiguous(), ti.to(torch.int32).contiguous()
        sid, eid, ntp = moe.moe_align(ti, {bm}, experts)
        c = torch.zeros({t}, top_k, n, device="cuda", dtype=torch.bfloat16)
        step("moe_gemm", lambda: moe.moe_gemm(a, w, c, sid, eid, ntp, top_k=top_k, block_m={bm},
                                               topk_weights=tw if {routed} else None))
        flat = ti.flatten().long().cpu()
        want = torch.stack([a[p // top_k].float().cpu() @ w[flat[p]].float().cpu().t()
                            for p in range({t} * top_k)])
        if {routed}:
            want *= tw.flatten().cpu()[:, None]
        rel("moe_gemm", c.view(-1, n), want, 2e-2)
        """
        for t in (1, 5, 33) for bm in (16, 32) for routed in (False, True)
    })


def test_attention():
    """hk.attention and the SDPA route at ragged sequence lengths, n_kv != n_q."""
    if ARCH == "gfx950":
        shapes = [(1, 2, 1, 1, False), (1, 2, 63, 65, False), (2, 2, 65, 257, False),
                  (1, 4, 255, 255, True), (1, 2, 257, 257, True), (1, 2, 1001, 333, False)]
    else:
        shapes = [(1, 2, 192, 192, False), (1, 2, 193, 33, False), (2, 2, 383, 383, True),
                  (1, 2, 1001, 1001, True), (1, 2, 4097, 97, False)]
    cases = {}
    for b, h, n, nk, causal in shapes:
        for d, hkv in ((128, h), (64, 1)):
            cases[f"attention B{b} H{h}/{hkv} N{n} Nkv{nk} D{d} causal={causal}"] = f"""
            q = rnd({b}, {h}, {n}, {d}); k = rnd({b}, {hkv}, {nk}, {d}); v = rnd({b}, {hkv}, {nk}, {d})
            got = step("attention", lambda: hk.attention(q, k, v, causal={causal}))
            kk = k.float().repeat_interleave({h} // {hkv}, 1); vv = v.float().repeat_interleave({h} // {hkv}, 1)
            want = F.scaled_dot_product_attention(q.float().cpu(), kk.cpu(), vv.cpu(), is_causal={causal})
            attn("attention", got, want)
            got = step("sdpa", lambda: hk.ops.sdpa.attention(q, k, v, causal={causal}))
            attn("sdpa", got, want)
            """
    fenced_cases(COMMON, cases)


PAGED = COMMON + """
from hk.ops import paged, paged_prefill

PAGE = paged.KV_BLOCK

def cache(lens, d, h_kv):
    # No spare blocks, and the last request's last page is the cache's last block.
    pages = [max(1, -(-n // PAGE)) for n in lens]
    mp = max(pages)
    nb = len(lens) * mp
    kc = rnd(nb, h_kv, PAGE, d); vc = rnd(nb, h_kv, PAGE, d)
    table = torch.arange(nb, device="cuda", dtype=torch.int32).view(len(lens), mp).contiguous()
    return kc, vc, table, torch.tensor(lens, device="cuda", dtype=torch.int32)

def host(*ts):
    # Each tensor the check reads, read back whole, once a case, twice, compared bitwise. The
    # reference is then built on the host: no per-head gather, cast and readback through fresh
    # fenced allocations (dozens a case before), and a readback that is not stable fails as
    # the harness's, not as the op's. The one paged failure seen in CI (gfx950, "relative
    # error 9.73e+29") matches max|got| 0.97265625 / 1e-30, an all-zero reference.
    out = []
    for t in ts:
        a, b = t.cpu(), t.cpu()
        assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), \\
            f"two readbacks of one {tuple(t.shape)} {t.dtype} tensor differ: the harness's fault"
        out.append(a)
    return out

def gathered(kc, vc, table, i, n, h):
    # host tensors (host()): the request's first n rows of K and V for kv head h
    pages = table[i, : -(-n // PAGE)].tolist()
    return (torch.cat([kc[p, h] for p in pages])[:n].float(),
            torch.cat([vc[p, h] for p in pages])[:n].float())
"""


def test_paged_decode_and_prefill():
    """Paged attention: the decode (plain, fused + merge at several splits) at lengths that
    end inside a page, a padded (zero-length) request, and the chunked prefill."""
    cases = {}
    for d in (64, 128):
        lens = [1, 31, 32, 33, 200, 1000]
        cases[f"plain decode D{d}"] = f"""
        h_kv, group = 2, 4
        kc, vc, table, lens = cache({lens}, {d}, h_kv)
        q = torch.zeros(len({lens}), h_kv, paged.Q_TILE, {d}, device="cuda", dtype=torch.bfloat16)
        q[:, :, :group] = rnd(len({lens}), h_kv, group, {d})
        o = torch.empty_like(q)
        step("plain", lambda: paged.KERNELS[{d}](q, kc, vc, table, lens, o))
        kc, vc, table, q, o = host(kc, vc, table, q, o)
        for i, n in enumerate({lens}):
            for h in range(h_kv):
                kk, vv = gathered(kc, vc, table, i, n, h)
                want = F.scaled_dot_product_attention(q[i, h, :group].float()[None, None],
                                                      kk[None, None], vv[None, None])[0, 0]
                attn(f"plain {{i}} {{h}}", o[i, h, :group], want)
        """
        for splits in (1, 2, 8, 32):
            for lens2 in ([1, 31, 32, 33, 200, 1000], [0, 100, 0, 37]):
                cases[f"fused decode D{d} splits{splits} {lens2}"] = f"""
                h_kv, group, splits = 2, 4, {splits}
                kc, vc, table, lens = cache({lens2}, {d}, h_kv)
                r = len({lens2})
                q = rnd(r, h_kv, group, {d})
                out = torch.empty_like(q)
                scratch = paged.fused_scratch(r * splits, h_kv, {d}, q.device, group=group)
                step("fused", lambda: paged.decode(paged.fused_kernel({d}, splits),
                                                   paged.merge_kernel({d}, splits), q, kc, vc,
                                                   table, lens, out, scratch, splits))
                kc, vc, table, q, out = host(kc, vc, table, q, out)
                for i, n in enumerate({lens2}):
                    for h in range(h_kv):
                        if n == 0:
                            # a padded request answers zero through the merge (splits > 1);
                            # alone, gfx1100's walk leaves its rows unspecified
                            if splits > 1 or ARCH == "gfx950":
                                same("padded", out[i, h], torch.zeros_like(out[i, h]))
                            continue
                        kk, vv = gathered(kc, vc, table, i, n, h)
                        want = F.scaled_dot_product_attention(q[i, h].float()[None, None],
                                                              kk[None, None], vv[None, None])[0, 0]
                        attn(f"fused {{i}} {{h}}", out[i, h], want)
                """
    for reqs in ([(0, 300)], [(1000, 77)], [(64, 100), (300, 50), (0, 200)],
                 [(0, 192), (500, 5), (40, 600), (33, 1)], [(0, 1)], [(31, 257)]):
        cases[f"prefill {reqs}"] = f"""
        reqs = {reqs}
        h_kv, group, d = 2, 4, 128
        kc, vc, table, lens = cache([c + n for c, n in reqs], d, h_kv)
        n_tok = sum(n for _, n in reqs)
        q = rnd(n_tok, h_kv * group, d)
        starts = [0]
        for _, n in reqs:
            starts.append(starts[-1] + n)
        q_start = torch.tensor(starts, device="cuda", dtype=torch.int32)
        out = torch.full_like(q, 7.0)
        tiles = paged_prefill.work_list([(i, reqs[i][1]) for i in range(len(reqs))], q.device)
        step("prefill", lambda: paged_prefill.kernel_for(d, group)(
            q.view(n_tok, -1), kc, vc, table, lens, q_start, tiles, out.view(n_tok, -1)))
        kc, vc, table, q, out = host(kc, vc, table, q, out)
        for i, (ctx, n) in enumerate(reqs):
            s = ctx + n
            keep = torch.arange(s)[None, :] <= (ctx + torch.arange(n))[:, None]
            for hh in range(h_kv * group):
                kk, vv = gathered(kc, vc, table, i, s, hh // group)
                sc = (q[starts[i]:starts[i + 1], hh].float() @ kk.T) * d ** -0.5
                want = torch.softmax(sc.masked_fill(~keep, float("-inf")), -1) @ vv
                attn(f"prefill {{i}} {{hh}}", out[starts[i]:starts[i + 1], hh], want)
        """
    fenced_cases(PAGED, cases)


def test_templates_rowwise_and_elementwise():
    """User kernels from hk.templates: row strips, a multi-pass stream, folded rows, the
    elementwise form on gate/up halves."""
    setup = COMMON + """
T = hk.templates

@T.rowwise
def rms_norm(x, w, *, EPS=1e-6):
    return x * (T.row_mean(x * x) + EPS).rsqrt() * w

@T.rowwise(outputs=2)
def add_rms_norm(x, r, w, *, EPS=1e-6):
    h = x + r
    return h * (T.row_mean(h * h) + EPS).rsqrt() * w, h

@T.rowwise
def softmax(x):
    e = (x - T.row_max(x)).exp()
    return e / T.row_sum(e)

@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32)))
def quantize(x):
    amax = T.row_max(abs(x)).maximum(1e-12)
    return x * (127.0 / amax), amax / 127.0

silu_mul = T.elementwise(lambda g, u: hk.math.silu(g) * u)
add = T.elementwise(lambda a, b: a + b)
"""
    cases = {
        f"rowwise {m}x{n}": f"""
        x = rnd({m}, {n}); r = rnd({m}, {n}); w = rnd({n})
        xf = x.float()
        rel("rms_norm", step("rms_norm", lambda: rms_norm(x, w)),
            xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float(), 2e-2)
        y, h = step("add_rms_norm", lambda: add_rms_norm(x, r, w))
        rel("add_rms_norm h", h, x.float() + r.float(), 1e-2)
        rel("softmax", step("softmax", lambda: softmax(x)), torch.softmax(xf, -1), 2e-2)
        if ARCH == "gfx950" or {n} % 2 == 0:   # gfx1100: no one-byte runs (refused)
            q, s = step("quantize", lambda: quantize(x))
            rel("quantize scale", s.view(-1), xf.abs().amax(-1).clamp_min(1e-12) / 127, 1e-5)
        """
        for m, n in ((1, 64), (3, 1000), (7, 4100), (1023, 64), (1025, 1032), (2, 70001),
                     (3, 151936))
    }
    cases |= {
        f"elementwise {m}x{i}": f"""
        x = rnd({m}, 2 * {i})
        rel("silu_mul", step("silu_mul", lambda: silu_mul(x[:, :{i}], x[:, {i}:])),
            F.silu(x[:, :{i}].float()) * x[:, {i}:].float(), 1e-2)
        a, b = rnd({m}, {i}), rnd({m}, {i})
        rel("add", step("add", lambda: add(a, b)), a.float() + b.float(), 1e-2)
        """
        for m, i in ((1, 14336), (3, 1001), (37, 9216), (2, 64))
    }
    fenced_cases(setup, cases)


# --------------------------------------------------------------------------- gfx950


@gfx950
def test_quant_stops_at_the_last_row():
    """mxfp8.quant at M % 64 != 0, both scale layouts, bit for bit hk.mx.quant + the layout."""
    fenced("""
    import hk
    from hk.ops import mxfp8

    for m, k in [(1, 128), (63, 1280), (65, 5120), (2049, 5120), (2111, 4096)]:
        x = torch.randn(m, k, device="cuda").bfloat16()
        q, s = step(f"quant packed M{m} K{k}", lambda: mxfp8.quant(x))
        rq, rs = hk.mx.quant(x)
        assert torch.equal(q.view(torch.uint8).cpu(), rq.view(torch.uint8).cpu()), (m, k)
        assert torch.equal(s.cpu(), hk.mx.pack_scales(rs).cpu()), (m, k)
    for m, k in [(32, 1024), (2080, 5120)]:
        x = torch.randn(m, k, device="cuda").bfloat16()
        q, s = step(f"quant aiter M{m} K{k}", lambda: mxfp8.quant(x, packed=False))
        rq, rs = hk.mx.quant(x)
        assert torch.equal(q.view(torch.uint8).cpu(), rq.view(torch.uint8).cpu()), (m, k)
        assert torch.equal(s.cpu(), hk.mx.aiter_scales(rs).cpu()), (m, k)
    """)


@gfx950
def test_mxfp8_route_at_ragged_m():
    """vllm_mxfp8's hk path -- quant, then gemm_4w or gemm_8w_tail -- at ragged M."""
    fenced("""
    from hk.integration import vllm_mxfp8 as P
    from hk.ops import mxfp8

    for (n, k), ms in [((1792, 5120), (2049, 2111)), ((2304, 5120), (2049,)),
                       ((5120, 4096), (5121, 8193)), ((16384, 1280), (2049, 16447)),
                       # their own smallest M (MIN_M_NK): the indexer's wq_b, engram's wkv
                       ((4096, 1280), (2825, 3073)), ((12800, 6144), (1025, 2047)),
                       # inside the measured ranges below MIN_M (RANGES), ragged against 128
                       ((12800, 6144), (161, 300, 700)), ((5120, 4096), (417, 1200, 1600)),
                       ((5120, 1152), (513, 1700)), ((16384, 1280), (577, 1700)),
                       ((1792, 5120), (1217,)), ((2304, 5120), (961, 1857))]:
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.full((n // 32, k // 32), 127, device="cuda", dtype=torch.uint8)
        rows = mxfp8.expand_block_scales(ws)[:n]
        wa, wp = mxfp8.aiter_scales(rows), mxfp8.pack_scales(rows)
        for m in ms:
            x = torch.randn(m, k, device="cuda").bfloat16()
            step(f"mxfp8_linear M{m} N{n} K{k} {P.schedule(m, n, k)}",
                 lambda: P.mxfp8_linear(x, w, ws, wa, wp, torch.bfloat16))
    """)


@gfx950
def test_mxfp8_gemms_at_ragged_m():
    """Every MXFP8 GEMM form at ragged M with tight scales: the pingpong on byte scales, the
    8-wave (packed A, AITER B), the 4-wave (packed both, its scale prefetch), the tails."""
    setup = COMMON + """
from hk.ops import mxfp8

def mx(m, n, k):
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(124, 131, (n, k // 32), device="cuda", dtype=torch.uint8)
    af = a.float().cpu() * torch.pow(2.0, sa.float().cpu() - 127).repeat_interleave(32, 1)
    bf = b.float().cpu() * torch.pow(2.0, sb.float().cpu() - 127).repeat_interleave(32, 1)
    return a, b, sa, sb, af @ bf.T
"""
    cases = {}
    for m, n, k in ((1, 512, 384), (65, 512, 1152), (129, 768, 512), (300, 512, 1152),
                    (2049, 1792, 1280), (257, 2304, 640)):
        cases[f"mxfp8 M{m} N{n} K{k}"] = f"""
        a, b, sa, sb, want = mx({m}, {n}, {k})
        sap, sbp, sba = mxfp8.pack_scales(sa).clone(), mxfp8.pack_scales(sb).clone(), \\
            mxfp8.aiter_scales(sb).clone()
        pad = -{m} % 32
        sa32 = mxfp8.aiter_scales(torch.cat([sa, sa.new_full((pad, {k} // 32), 127)]) if pad
                                  else sa).clone()
        for name, fn in (
                ("gemm_pp", lambda: mxfp8.gemm_pp(a, b, sa, sb, block_n=256, block_m=128)),
                ("gemm_4w", lambda: mxfp8.gemm_4w(a, b, sap, sbp)),
                ("gemm_4w rolled", lambda: mxfp8.gemm_4w(a, b, sap, sbp, rolled=True)),
                ("gemm_8w", lambda: mxfp8.gemm_8w(a, b, sap, sba)),
                ("gemm_8w direct32", lambda: mxfp8.gemm_8w(a, b, sa32, sba, block_m=128)),
                ("gemm_4w_tail", lambda: mxfp8.gemm_4w_tail(a, b, sap, sbp)),
                ("gemm_8w_tail", lambda: mxfp8.gemm_8w_tail(a, b, sap, sba, sbp))):
            got = built(name, fn)
            if got is not REFUSED:
                rel(name, got, want, 5e-3)
        """
    fenced_cases(setup, cases)


@gfx950
def test_gemm_cdna_candidates_at_ragged_m():
    """Every gfx950 GEMM candidate the router may pick, forced, at ragged M -- including the
    128-column tail launches, split-K and the odd K-tile counts."""
    from hk.ops import gemm_cdna

    shapes = ((1, 256, 128), (33, 1152, 1280), (130, 1792, 768), (257, 640, 1152),
              (300, 1408, 1024))
    cases = {}
    for m, n, k in shapes:
        names = [c for c in gemm_cdna.CANDIDATES if gemm_cdna.fits(c, m, n, k)]
        cases[f"gemm_cdna M{m} N{n} K{k}"] = f"""
        from hk.ops import gemm_cdna
        a = rnd({m}, {k}); b = rnd({n}, {k}, scale={k} ** -0.5)
        want = a.float().cpu() @ b.float().cpu().T
        for name in {names}:
            out = torch.empty({m}, {n}, device="cuda", dtype=torch.bfloat16)
            if built(f"gemm_cdna {{name}}", lambda: gemm_cdna.run(name, a, b, out)) is not REFUSED:
                rel(name, out, want, 2e-2)
        """
    fenced_cases(COMMON, cases)


@gfx950
def test_gemm_route_around_a_refused_candidate():
    """hk.ops.gemm.matmul where the table routes to dp128x224, at ragged M: under the vLLM
    image's hipcc the spill gate refuses it and the call takes the next candidate (in
    rocm/pytorch it runs); then with dp128x224 refused in either."""
    fenced_cases(COMMON, {"gemm route around a refused candidate": """
    from hk.ops import gemm, gemm_cdna

    def check(m, n, k):
        a = rnd(m, k); b = rnd(n, k, scale=k ** -0.5)
        want = a.float().cpu() @ b.float().cpu().T
        got = step(f"matmul M{m} N{n} K{k} {gemm_cdna.route(m, n, k)}",
                   lambda: gemm.matmul(a, b))
        rel(f"matmul M{m} N{n} K{k}", got, want, 2e-2)

    for m, n in ((257, 28672), (511, 14336)):
        check(m, n, 4096)
    print(f"refused by this build: {gemm_cdna._REFUSED}", flush=True)
    gemm_cdna._REFUSED.setdefault("dp128x224", "refused by the test")
    check(193, 28672, 4096)
    """})


@gfx950
def test_moe_route_at_ragged_m():
    """vllm_moe_a8w4's hk path (AITER's sort and quant, hk's stages) at ragged M, uniform and
    skewed routing."""
    pytest.importorskip("aiter")
    fenced("""
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight
    from hk.integration import vllm_moe_a8w4 as PM

    e, k, inter, topk = 8, 5120, 1152, 6
    f4 = torch.float4_e2m1fn_x2
    w1 = torch.randint(0, 256, (e, 2 * inter, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, inter // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.full((e * 2 * inter, k // 32), 120, device="cuda", dtype=torch.uint8)
    s2 = torch.full((e * k, inter // 32), 120, device="cuda", dtype=torch.uint8)
    w1s = shuffle_weight(w1.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=True)
    w2s = shuffle_weight(w2.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=False)
    sc1 = shuffle_scale(s1, experts_cnt=e, is_guinterleave=True, gate_up=True)
    sc2 = shuffle_scale(s2, experts_cnt=e, is_guinterleave=True, gate_up=False)
    for m in (2561, 5437, 8193, 8255):     # from MIN_M (2560); 5437: a serving chunk's M
        x = torch.randn(m, k, device="cuda").bfloat16()
        wt = torch.full((m, topk), 1.0 / topk, device="cuda")
        ids = torch.rand(m, e, device="cuda").topk(topk, 1).indices.to(torch.int32)
        step(f"moe M{m}", lambda: PM.moe(x, w1s, w2s, wt, ids, sc1, sc2, 10.0))
        ids = (torch.arange(m * topk, device="cuda", dtype=torch.int32) % 7).view(m, topk)
        step(f"moe M{m} skewed", lambda: PM.moe(x, w1s, w2s, wt, ids, sc1, sc2, 10.0))
    """)


@gfx950
def test_moe_a8w4_stages_on_tight_buffers():
    """moe_a8w4's stages called directly: the sort metadata without slack rows past the last
    block, the sorted routing weights exactly one per sorted row, ragged token counts."""
    setup = COMMON + """
from hk.ops import moe_a8w4 as M
from hk.ops import mxfp8

def sort(ids, n_experts, bm):
    m, topk = ids.shape
    flat = ids.reshape(-1).long()
    rows, eids = [], []
    for e in range(n_experts):
        p = torch.nonzero(flat == e).view(-1)
        if p.numel() == 0:
            continue
        v = ((p // topk) | ((p % topk) << 24)).to(torch.int32)
        v = torch.cat([v, torch.full((-v.numel() % bm,), m, dtype=torch.int32, device=ids.device)])
        rows.append(v)
        eids += [e] * (v.numel() // bm)
    sid = torch.cat(rows).contiguous()
    se = torch.tensor(eids, dtype=torch.int32, device=ids.device)
    nv = torch.tensor([sid.numel(), m], dtype=torch.int32, device=ids.device)
    return sid, se, nv
"""
    cases = {}
    for t, mt, pp1, pp2, depth, pers, inter in ((77, 4, 3, 4, 2, 0, 768), (1, 2, 3, 2, 2, 0, 768),
                                                (33, 4, 3, 4, 2, 8, 768),
                                                (129, 4, 3, 4, 2, 0, 1152)):
        cases[f"sorted stages T{t} mt{mt} pers{pers} I{inter}"] = f"""
        E, K, D, T, topk, I = 6, 512, 512, {t}, 2, {inter}
        w1 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8)
        w1s = torch.randint(122, 132, (E * 2 * I, K // 32), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8)
        w2s = torch.randint(122, 132, (E * D, I // 32), dtype=torch.uint8)
        hidden = rnd(T, K)
        ids = torch.stack([torch.randperm(E - 1)[:topk] for _ in range(T)]).cuda()
        wt = torch.rand(T, topk, device="cuda")
        sid, se, nv = sort(ids, E, 16 * {mt})
        xq, xs = M.mx_quant(hidden)
        tok = (sid & 0xFFFFFF).long()
        xs_sorted = torch.full((sid.numel(), K // 32), 127, dtype=torch.uint8, device="cuda")
        live = tok < T
        xs_sorted[live] = xs[tok[live]]
        xsa = mxfp8.aiter_scales(xs_sorted).clone()
        a13, s13 = M.aiter_layout_w13(w1).cuda(), M.aiter_layout_w13_scale(w1s, E).cuda()
        h = step("stage1_sorted", lambda: M.stage1_sorted(xq, xsa, a13, s13, sid, se, nv,
                                                          inter=I, mt={mt}, pp={pp1}, limit=10.0,
                                                          depth={depth}))
        hq_e, hs_e = step("stage1_sorted mxq", lambda: M.stage1_sorted(
            xq, xsa, a13, s13, sid, se, nv, inter=I, mt={mt}, pp={pp1}, limit=10.0,
            depth={depth}, mxq=True))
        hq, hs = M.mx_quant(h)
        sw = torch.zeros(sid.numel(), device="cuda")
        nvalid = int(nv[0])
        lv = live[:nvalid]
        sw[:nvalid][lv] = wt[tok[:nvalid][lv], (sid[:nvalid][lv] >> 24).long()]
        hsa = mxfp8.aiter_scales(hs).clone()
        a2, s2 = M.aiter_layout_w2(w2).cuda(), M.aiter_layout_w2_scale(w2s).cuda()
        y2 = step("stage2_sorted", lambda: M.stage2_sorted(hq, hsa, a2, s2, sid, se, nv, sw, T,
                                                           topk, E, mt={mt}, pp={pp2},
                                                           depth={depth}, persistent={pers}))
        step("reduce", lambda: M.reduce(y2, None, wt, T, inv=M.identity(T, topk, "cuda"),
                                        wrow=sw.view(-1, 1), weighted=False))
        """
    fenced_cases(setup, cases)


@gfx950
def test_hipkittens_kernels():
    """The vendored HipKittens GEMM and GQA forward on whole tiles, the tensors ending at the
    last tile's last byte."""
    pytest.importorskip("hk.ops.hipkittens")
    fenced_cases(COMMON, {
        "hipkittens gemm": """
        from hk.ops import hipkittens as H
        a = rnd(256, 128); b = rnd(256, 128, scale=128 ** -0.5)
        got = built("gemm_bf16", lambda: H.gemm_bf16(a, b), also=(H.Unavailable,))
        if got is not REFUSED:
            rel("gemm_bf16", got, a.float() @ b.float().T, 2e-2)
        """,
        "hipkittens gqa": """
        from hk.ops import hipkittens as H
        q = rnd(1, 256, 8, 128); k = rnd(1, 256, 2, 128); v = rnd(1, 256, 2, 128)
        got = built("gqa_fwd", lambda: H.gqa_fwd(q, k, v), also=(H.Unavailable,))
        if got is not REFUSED:
            t = lambda x: x.float().transpose(1, 2).cpu()  # noqa: E731
            want = F.scaled_dot_product_attention(t(q), t(k).repeat_interleave(4, 1),
                                                  t(v).repeat_interleave(4, 1))
            rel("gqa_fwd", got.transpose(1, 2), want, 5e-2)
        """,
    })


# --------------------------------------------------------------------------- gfx1100


@gfx1100
def test_rdna_int8_gemm_and_convrot():
    """The int8 WMMA GEMM and the ConvRot W8A8 linear at ragged M."""
    cases = {
        f"convrot M{m} group{g}": f"""
        from hk.ops import convrot
        n, k = 768, 1024
        x = rnd({m}, k); w = rnd(n, k, scale=k ** -0.5); bias = rnd(n)
        wq, ws = convrot.prepare_weight(w, {g})
        wq, ws = wq.clone(), ws.clone()
        y = step("convrot", lambda: convrot.linear(x, wq, ws, {g}, bias=bias))
        on = "cpu" if {m} <= 16 else "cuda"    # torch has no int32 matmul on the GPU
        rel("convrot", y, convrot.reference(x.to(on), wq.to(on), ws.to(on), {g},
                                            bias=bias.to(on)), 2e-2)
        """
        for m in (1, 17, 127, 129, 1023) for g in (16, 256)
    }
    cases |= {
        f"int8 gemm M{m}": f"""
        from hk.ops import gemm
        n, k = 256, 512
        a = torch.randint(-128, 128, ({m}, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
        c = torch.zeros({m}, n, device="cuda", dtype=torch.int32)
        step("int8 gemm", lambda: gemm.gemm_kernel("fence_gemm_i8", op_dtype=hk.i8)(a, b, c))
        same("int8 gemm", c, a.cpu().int() @ b.cpu().int().T)
        """
        for m in (1, 129, 300)
    }
    fenced_cases(COMMON, cases)


@gfx1100
def test_attn_k3():
    """K3's attention (gfx1100, D 128, MHA) at ragged sequence and KV lengths, both layouts."""
    cases = {}
    for b, h, n, nk in ((1, 2, 128, 33), (1, 2, 129, 127), (1, 4, 640, 1001), (2, 3, 1485, 1485)):
        cases[f"attn_k3 {b}x{h}x{n}x{nk}"] = f"""
        from hk.ops import attn_k3
        q = rnd({b}, {h}, {n}, 128); k = rnd({b}, {h}, {nk}, 128); v = rnd({b}, {h}, {nk}, 128)
        want = F.scaled_dot_product_attention(q.float().cpu(), k.float().cpu(), v.float().cpu())
        rel("attn_k3", step("attn_k3", lambda: attn_k3.attention(q, k, v)), want, 1e-2)
        qb, kb, vb = (rnd({b}, n_, {h}, 128) for n_ in ({n}, {nk}, {nk}))
        want = F.scaled_dot_product_attention(qb.float().transpose(1, 2).cpu(),
                                              kb.float().transpose(1, 2).cpu(),
                                              vb.float().transpose(1, 2).cpu())
        got = step("attn_k3 bnhd", lambda: attn_k3.attention(qb.transpose(1, 2), kb.transpose(1, 2),
                                                             vb.transpose(1, 2)))
        rel("attn_k3 bnhd", got, want, 1e-2)
        """
    fenced_cases(COMMON, cases)


@gfx950
def test_gathered_copy_reads_no_index_past_its_rows():
    """g.load(gather=(ids, at), rows=...): an index list ending flush against the fence, inside a
    tile, read by tiles that run past it -- no id past the row limit is read."""
    fenced(COMMON + '''
from hk.types import bf16, i32
st = hk.ST(bf16, 32, 64); rt = hk.RT(bf16, 32, 64, "row")

def body(x: hk.Tensor[bf16], ids: hk.Tensor[i32], out: hk.Tensor[bf16]):
    g = hk.group(4)
    s = hk.lds.alloc(st, name="s")
    at = hk.arch.block_idx.x * 32
    g.load(s, x.at[0, 64], wait=False, gather=(ids, at), rows=hk.max(ids.cols - at, 0))
    g.wait_for(s)
    hk.arch.barrier()
    t = s.load(rt)
    if hk.arch.warp_id() == 0:
        out[hk.elem_coord(0, 0, at, 0)].store(t)

body.__name__ = "gather_fenced_t"
k = hk.kernel(body, arch="gfx950", warps=4, name="gather_fenced_t", grid=lambda p: (4, 1, 1))
x = rnd(100, 128)
ids = torch.randint(0, 100, (1, 76), dtype=torch.int32).cuda()     # 304 bytes: ends at the fence
out = torch.zeros(128, 64, device="cuda", dtype=torch.bfloat16)
step("gather rows=", lambda: k(x, ids, out))
got, xs, ii = out.cpu(), x.cpu(), ids.cpu()[0].long()
want = torch.zeros(128, 64, dtype=torch.bfloat16); want[:76] = xs[ii, 64:]
same("gathered", got, want)
''')


SMLA = COMMON + """
from hk.ops import sparse_mla as SM

def smla_case(lens, rows, seed):
    # the CSR as vLLM's build_ragged_indices_from_dense packs it, but exactly nnz long: the
    # allocation ends at the last query's last index (vLLM's ends there when every row is full)
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = len(lens)
    q = (torch.randn(T, SM.H, SM.D, generator=g) * 0.5).bfloat16().cuda()
    kv = (torch.randn(rows, SM.D, generator=g) * 0.5).bfloat16().cuda()
    ind = torch.cat([torch.randint(0, rows, (n,), generator=g, dtype=torch.int32) for n in lens]).cuda()
    ptr = torch.zeros(T + 1, dtype=torch.int32)
    ptr[1:] = torch.tensor(lens).cumsum(0)
    sink = torch.randn(SM.H, generator=g).float().cuda()
    pos = (torch.arange(T) * 7 + 3).to(torch.int64).cuda()
    cs = torch.randn(int(pos.max()) + 1, 64, generator=g).cuda()
    return q, kv, ind, ptr.cuda(), sink, pos, cs

def smla_run(lens, rows=4096, seed=0):
    q, kv, ind, ptr, sink, pos, cs = smla_case(lens, rows, seed)
    T = q.shape[0]
    out = step("prefill", lambda: SM.prefill(q, kv, ind, ptr, sink, 512 ** -0.5, torch.empty_like(q)))
    md = torch.empty(T, SM.H * SM.D, dtype=torch.float8_e4m3fn, device="cuda")
    ms = torch.empty(T, SM.H * SM.D // 32, dtype=torch.uint8, device="cuda")
    step("prefill_mxfp8", lambda: SM.prefill_mxfp8(q, kv, ind, ptr, sink, 512 ** -0.5, pos, cs, md, ms))
    o = out.cpu().float()
    assert torch.isfinite(o).all(), "non-finite output"
"""


@gfx950
def test_sparse_mla():
    """hk.ops.sparse_mla where the index list ends exactly at the last query's last row: every row
    full (vLLM's packing leaves no slack then), a last query whose rows end mid-block, empty and
    one-row queries, and a call whose last query is empty."""
    cases = {
        "every query 640 rows, T 1024": "smla_run([640] * 1024)",
        "ragged, last query 600 rows": "smla_run([640, 1, 0, 128, 333] * 120 + [600], seed=1)",
        "last query empty": "smla_run([640] * 600 + [0], seed=2)",
        "one query of 1 row": "smla_run([1], rows=64, seed=3)",
        "every query 128 rows (sliding window), T 2048": "smla_run([128] * 2048, seed=4)",
    }
    fenced_cases(SMLA, cases)


def _has_vllm_rocm_sparse() -> bool:
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec("vllm.v1.attention.ops.rocm_aiter_mla_sparse") is not None
    except ModuleNotFoundError:
        return False


@gfx950
@pytest.mark.skipif(not _has_vllm_rocm_sparse(), reason="vLLM's ROCm sparse MLA (the vLLM image)")
def test_vllm_sparse_mla_route_on_a_mixed_step():
    """The vllm_sparse_mla route as a real-weight server's mixed step calls it (decode rows run
    elsewhere): one chunk call holding a full prefill's rows, a partial first chunk and a
    continuation chunk whose rows are all full, and a continuation-only call where every row is
    full -- vLLM's build_ragged_indices_from_dense then packs the list flush against its
    allocation's end. A call below MIN_T stays vLLM's. The server faulted here before the
    gathered copy took a row limit."""
    fenced(SMLA + '''
from hk.integration import vllm_sparse_mla as R
import vllm.models.deepseek_v41.amd.rocm as rocm
print(R.apply_patch(), flush=True)
W, ROWS = 640, 6000

def dense(lens, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ind = torch.full((len(lens), W), -1, dtype=torch.int32)
    for i, n in enumerate(lens):
        ind[i, :n] = torch.randint(0, ROWS, (n,), generator=g, dtype=torch.int32)
    return ind.cuda(), torch.tensor(lens, dtype=torch.int32).cuda()

def call(lens, seed):
    T = len(lens)
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(T, SM.H, SM.D, generator=g) * 0.5).bfloat16().cuda()
    kv = (torch.randn(ROWS, 1, SM.D, generator=g) * 0.5).bfloat16().cuda()
    sink = torch.randn(SM.H, generator=g).float().cuda()
    ind, lens_t = dense(lens, seed)
    out = torch.empty_like(q)
    step(f"route T {T}", lambda: rocm.rocm_sparse_attn_prefill(
        q=q, kv=kv, indices=ind, topk_length=lens_t, scale=512 ** -0.5, head_dim=512, nope_head_dim=448,
        rope_head_dim=64, attn_sink=sink, output=out))
    assert torch.isfinite(out.cpu().float()).all(), "non-finite output"

full = [min(W, 128 + p // 2) for p in range(1200)]          # a prefill from position 0
part = [min(W, 128 + p // 2) for p in range(896)]           # a partial first chunk
cont = [W] * 487                                             # a continuation: every row full
call(full + part + cont, 1)
call([W] * 600, 2)                                           # continuation-only: no slack at all
call(cont, 3)                                                # below MIN_T: vLLM's
# the faulting step's prefill rows (its 2 decode rows run in the decode path): prefills of 7562 and
# 7437 tokens, an 896-token first chunk of a 6687-token prompt, a 487-token continuation chunk
step_lens = ([min(W, 128 + p // 2) for p in range(7562)] + [min(W, 128 + p // 2) for p in range(7437)]
             + part + [W] * 487)
call(step_lens, 4)
print(R.STATS, flush=True)
assert R.STATS["hk"] == 3 and R.STATS["fallback"] == 1, R.STATS
''')
