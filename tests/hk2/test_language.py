"""Language coverage: every kernel runs compiled and interpreted, both checked against torch."""
import pytest
import torch

import hk2 as hk

dev = "cuda"


def both(kernel, grid, make, ref_fn, atol=1e-5, rtol=1e-5, **kw):
    """Run compiled and interpreted on fresh copies of the arguments; compare outputs with ref_fn(args)."""
    for interpret in (False, True):
        args = make()
        kernel[grid](*args, interpret=interpret, **kw)
        for got, want in ref_fn(args):
            torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol,
                                       msg=lambda m: f"{'interpreted' if interpret else 'compiled'}: {m}")


# ---------------------------------------------------------------- memory
@hk.jit
def copy2d(x, out, BM: hk.constexpr = 32, BN: hk.constexpr = 32):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    hk.store(out, (m0, n0), hk.load(x, (m0, n0), (BM, BN)))


@pytest.mark.parametrize("view", ["contig", "transposed", "sliced", "offset"])
def test_strided_views(view):
    base = torch.randn(70, 90, device=dev)
    x = {"contig": base, "transposed": base.T, "sliced": base[:, 3:80:2], "offset": base[1:, 3:]}[view]
    both(copy2d, (hk.cdiv(x.shape[0], 32), hk.cdiv(x.shape[1], 32)),
         lambda: (x, torch.zeros(x.shape, device=dev)), lambda a: [(a[1], x)])


@hk.jit
def shifted(x, out, s, BLOCK: hk.constexpr = 64):
    i = hk.program_id(0) * BLOCK
    hk.store(out, i, hk.load(x, i + s, BLOCK, other=-1.0))


def test_unaligned_offsets():
    x = torch.randn(1000, device=dev, dtype=torch.bfloat16)
    ref = torch.cat([x[3:], torch.full((3,), -1.0, device=dev, dtype=x.dtype)])
    both(shifted, (16,), lambda: (x, torch.empty_like(x), 3), lambda a: [(a[1], ref)])


@hk.jit
def gather3d(x, out, BN: hk.constexpr = 64, D: hk.constexpr = 32):
    b, n0 = hk.program_id(1), hk.program_id(0) * BN
    t = hk.load(x, (b, n0, 0), (BN, D))
    hk.store(out, (b, n0, 0), t * 2)


def test_three_dims():
    x = torch.randn(3, 100, 32, device=dev)
    both(gather3d, (2, 3), lambda: (x, torch.zeros_like(x)), lambda a: [(a[1], x * 2)])


# ---------------------------------------------------------------- integers and casts
@hk.jit
def int_ops(x, out, BLOCK: hk.constexpr = 256):
    i = hk.program_id(0) * BLOCK
    t = hk.load(x, i, BLOCK)
    r = (t * 3 + 7) // 4 + (t % 5) - (t >> 1) + (t & 6) + (t << 2)
    hk.store(out, i, r)


def test_integer_math():
    x = torch.randint(0, 1000, (1000,), device=dev, dtype=torch.int32)
    ref = (x * 3 + 7) // 4 + (x % 5) - (x >> 1) + (x & 6) + (x << 2)
    both(int_ops, (4,), lambda: (x, torch.empty_like(x)), lambda a: [(a[1], ref)])


@hk.jit
def casts(x, out_i, out_h, BLOCK: hk.constexpr = 256):
    i = hk.program_id(0) * BLOCK
    t = hk.load(x, i, BLOCK)
    hk.store(out_i, i, (t * 10).to(hk.i32))
    hk.store(out_h, i, t.to(hk.f16))


def test_casts():
    x = torch.randn(600, device=dev)
    both(casts, (3,), lambda: (x, torch.empty(600, device=dev, dtype=torch.int32),
                               torch.empty(600, device=dev, dtype=torch.float16)),
         lambda a: [(a[1], (x * 10).to(torch.int32)), (a[2], x.half())], atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------- control flow
@hk.jit
def nested(x, out, n_outer, BLOCK: hk.constexpr = 128):
    i = hk.program_id(0) * BLOCK
    acc = hk.zeros((BLOCK,))
    for a in range(n_outer):
        t = hk.load(x, i, BLOCK)
        for b in range(a + 1):
            acc = acc + t * (b + 1)
        if a % 2 == 1:
            acc = acc - 1.0
    hk.store(out, i, acc)


def test_nested_loops_and_if():
    x = torch.randn(512, device=dev)
    n = 5
    ref = torch.zeros_like(x)
    for a in range(n):
        for b in range(a + 1):
            ref = ref + x * (b + 1)
        if a % 2 == 1:
            ref = ref - 1
    both(nested, (4,), lambda: (x, torch.empty_like(x), n), lambda a: [(a[1], ref)], atol=1e-4, rtol=1e-4)


@hk.jit
def countdown(out, n, BLOCK: hk.constexpr = 64):
    acc = hk.zeros((BLOCK,), hk.i32)
    for k in range(n, 0, -1):
        acc = acc * 2 + k
    hk.store(out, 0, acc)


def test_negative_step():
    r = 0
    for k in range(6, 0, -1):
        r = r * 2 + k
    both(countdown, (1,), lambda: (torch.empty(64, device=dev, dtype=torch.int32), 6),
         lambda a: [(a[0], torch.full((64,), r, dtype=torch.int32, device=dev))])


# ---------------------------------------------------------------- reductions and broadcasts
@hk.jit
def center_cols(x, out, BM: hk.constexpr = 128, BN: hk.constexpr = 32):
    n0 = hk.program_id(0) * BN
    t = hk.load(x, (0, n0), (BM, BN))
    mean = t.sum(0) / x.shape[0]
    hk.store(out, (0, n0), t - mean[None, :])


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_column_reduction(warps):
    x = torch.randn(128, 96, device=dev)
    both(center_cols, (3,), lambda: (x, torch.empty_like(x)), lambda a: [(a[1], x - x.mean(0, keepdim=True))],
         atol=1e-5, rtol=1e-4, num_warps=warps)


@hk.jit
def int_max(x, out, BM: hk.constexpr = 16, BN: hk.constexpr = 64):
    m0 = hk.program_id(0) * BM
    t = hk.load(x, (m0, 0), (BM, BN), other=-1000)
    hk.store(out, m0, hk.maximum(t.max(1), 0))


def test_int_reduction():
    x = torch.randint(-50, 50, (40, 60), device=dev, dtype=torch.int32)
    both(int_max, (3,), lambda: (x, torch.empty(40, device=dev, dtype=torch.int32)),
         lambda a: [(a[1], x.amax(1).clamp(min=0))])


@hk.jit
def tiny_tile(x, out):
    t = hk.load(x, (0, 0), (16, 16))
    hk.store(out, (0, 0), t + t.T)


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_tile_smaller_than_workgroup(warps):
    x = torch.randn(16, 16, device=dev)
    both(tiny_tile, (1,), lambda: (x, torch.empty_like(x)), lambda a: [(a[1], x + x.T)], num_warps=warps)


# ---------------------------------------------------------------- matrix cores
@hk.jit
def chain(a, b, c, out, M: hk.constexpr = 64, K: hk.constexpr = 32, N: hk.constexpr = 64, P: hk.constexpr = 32):
    x = hk.load(a, (0, 0), (M, K))
    y = hk.load(b, (0, 0), (K, N))
    z = hk.load(c, (0, 0), (N, P))
    t = hk.dot(x, y).to(hk.bf16)
    hk.store(out, (0, 0), hk.dot(t, z))


@pytest.mark.parametrize("warps", [1, 4])
def test_dot_chain(warps):
    a = torch.randn(64, 32, device=dev, dtype=torch.bfloat16)
    b = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)
    c = torch.randn(64, 32, device=dev, dtype=torch.bfloat16)
    ref = (a.float() @ b.float()).bfloat16().float() @ c.float()
    both(chain, (1,), lambda: (a, b, c, torch.empty(64, 32, device=dev)), lambda x: [(x[3], ref)],
         atol=0.5, rtol=2e-2, num_warps=warps)


@hk.jit
def gemm_bias_silu(a, w, bias, out, BM: hk.constexpr = 64, BN: hk.constexpr = 64, BK: hk.constexpr = 32):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN))
    for k in range(0, a.shape[1], BK):
        acc = hk.dot(hk.load(a, (m0, k), (BM, BK)), hk.load(w, (n0, k), (BN, BK)).T, acc)
    y = acc + hk.load(bias, n0, BN).to(hk.f32)[None, :]
    hk.store(out, (m0, n0), y / (1.0 + hk.exp(-y)))


def test_gemm_epilogue():
    a = torch.randn(130, 96, device=dev, dtype=torch.bfloat16)
    w = torch.randn(70, 96, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(70, device=dev, dtype=torch.bfloat16)
    ref = torch.nn.functional.silu(a.float() @ w.float().T + bias.float())
    both(gemm_bias_silu, (3, 2), lambda: (a, w, bias, torch.empty(130, 70, device=dev, dtype=torch.bfloat16)),
         lambda x: [(x[3], ref)], atol=0.1, rtol=2e-2)


# ---------------------------------------------------------------- errors point at the source line
def test_errors_name_the_line():
    @hk.jit
    def bad(x):
        t = hk.load(x, 0, 100)  # not a power of two
        hk.store(x, 0, t)

    with pytest.raises(hk.HKError, match=r"test_language.py:\d+: tile shapes must be powers of two"):
        bad[(1,)](torch.zeros(128, device=dev))

    @hk.jit
    def bad2(a, b):
        hk.store(a, (0, 0), hk.dot(hk.load(a, (0, 0), (32, 16)), hk.load(b, (0, 0), (32, 32))))

    with pytest.raises(hk.HKError, match="inner dimensions differ"):
        bad2[(1,)](torch.zeros(32, 32, device=dev, dtype=torch.bfloat16),
                   torch.zeros(32, 32, device=dev, dtype=torch.bfloat16))
