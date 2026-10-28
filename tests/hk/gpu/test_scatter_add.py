"""Scattered stores and atomic adds by row index on gfx950, against torch: AITER's MoE sort ids
(token | choice << 24, padding past the tokens) and `scatter(..., add=True)` from a register tile
-- the GPU test that keeps `scatter(add=True)` a primitive (docs/design/dsl-surface.md §9,
decision 7)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp32, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("scatter(add) is written for gfx950's atomics", allow_module_level=True)

pytestmark = pytest.mark.gpu

T32 = hk.RT(fp32, 16, 64)
M, TOPK, C = 40, 3, 128


def _sort_ids(g):
    """Every (token, choice) once, shuffled, in 16-row blocks, with padding rows (token M)."""
    pairs = torch.arange(M * TOPK, device="cuda", dtype=torch.int32)
    pairs = pairs[torch.randperm(pairs.numel(), device="cuda", generator=g)]
    ids = (pairs // TOPK) | ((pairs % TOPK) << 24)
    pad = torch.full((16 - ids.numel() % 16,), M, device="cuda", dtype=torch.int32)
    return torch.cat([ids, pad])


def _rows_kernel(add, sort_topk, out_dtype):
    def scat(x, ids, o):
        at = hk.arch.block_idx.y * 16
        col = hk.arch.block_idx.x * 64
        t = x[hk.elem_coord(0, 0, at, col)].load(T32)
        o.scatter(t, ids, at, col=col, sort_topk=sort_topk, add=add)

    scat.__annotations__ = {"x": hk.Tensor[fp32], "ids": hk.Tensor[i32], "o": hk.Tensor[out_dtype]}

    return hk.kernel(scat, arch="gfx950", warps=1,
                     name=f"scat_{int(add)}_{sort_topk}_{out_dtype.name}",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))


def _want(x, ids, add, dtype):
    tok, ch = (ids & 0xFFFFFF).long(), (ids >> 24).long()
    live = tok < M
    if add:
        out = torch.zeros(M, C, device="cuda")
        out.index_add_(0, tok[live], x[live])
    else:
        out = torch.zeros(TOPK * M, C, device="cuda")
        out[ch[live] * M + tok[live]] = x[live]
    return out.to(dtype)


def test_sort_ids_scatter_each_pair_to_its_choice_major_row():
    g = torch.Generator(device="cuda").manual_seed(1)
    ids = _sort_ids(g)
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    out = torch.zeros(TOPK * M, C, device="cuda", dtype=torch.bfloat16)
    _rows_kernel(False, TOPK, bf16)(x, ids.view(1, -1), out)
    assert torch.equal(out, _want(x, ids, False, torch.bfloat16))


@pytest.mark.parametrize("dtype", [fp32, bf16], ids=["fp32", "bf16"])
def test_an_atomic_scatter_sums_each_tokens_choices(dtype):
    g = torch.Generator(device="cuda").manual_seed(2)
    ids = _sort_ids(g)
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    td = torch.float32 if dtype is fp32 else torch.bfloat16
    out = torch.zeros(M, C, device="cuda", dtype=td)
    _rows_kernel(True, TOPK, dtype)(x, ids.view(1, -1), out)
    want = _want(x, ids, True, torch.float32)
    tol = 1e-5 if dtype is fp32 else 2e-2      # bf16 adds round each partial sum
    assert ((out.float() - want).abs().max() / want.abs().max()).item() < tol


def _sink_probe():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools" / "cdna-probes" / "mfma_scale_sink.py"
    spec = importlib.util.spec_from_file_location("hk_mfma_scale_sink", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sink_case(probe, steps, pinned, seed=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K, n, rows, m, topk = 128 * steps, 32, probe.ROWS, 40, probe.TOPK
    w = torch.randint(0, 256, (1, n, K // 2), device="cuda", dtype=torch.uint8, generator=g)
    ws = torch.randint(124, 131, (n, K // 32), device="cuda", dtype=torch.uint8, generator=g)
    x = (torch.randn(rows, K, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    xs = torch.randint(124, 131, (rows, K // 32), device="cuda", dtype=torch.uint8, generator=g)
    # 64 rows: 36 (token, choice) pairs and 28 padding rows (token m), shuffled together
    pairs = torch.randperm(m * topk, device="cuda", generator=g)[:36].int()
    ids = torch.cat([(pairs // topk) | ((pairs % topk) << 24),
                     torch.full((rows - 36,), m, device="cuda", dtype=torch.int32)])
    ids = ids[torch.randperm(rows, device="cuda", generator=g)]
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    probe.kernel(steps, pinned)(hk.mx.aiter_fp4_weights(w, False).view(torch.int8),
                                hk.mx.aiter_scales(ws).view(torch.int8), x, xs.view(torch.int8),
                                ids.view(1, -1), out)
    wd = hk.mx.fp4_values(w[0]) * torch.pow(2.0, ws.float() - 127).repeat_interleave(32, -1)
    c = hk.mx.dequant(x, xs) @ wd.T
    tok = (ids & 0xFFFFFF).long()
    live = tok < m
    want = torch.zeros(m, n, device="cuda")
    want.index_add_(0, tok[live], c[live])
    return ((out.float() - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("steps", [2, 6])
def test_a_scaled_mfma_tile_adds_its_valid_rows(steps):
    """K1's stage-2 shape: FP4 weights times 16-row fp8 tiles, each scatter-added. roc-7.2.3
    sinks the unpinned MFMAs into the scatter's per-row `if`. This kernel survives that, because
    its operands are complete before the `if`. K1's did not: its operand copies went in too, and
    stage 2 was 0.28 off. tests/hk/codegen/test_mfma_scale_sink.py catches the sink itself;
    this test checks the values."""
    assert _sink_case(_sink_probe(), steps, pinned=True) < 2e-2


def _staged_kernel(staged):
    """Four 16-row tiles a workgroup (a 64-row block); `staged`: the block's ids read from LDS."""
    def scat_staged(x: hk.Tensor[fp32], ids: hk.Tensor[i32], o: hk.Tensor[bf16]):
        g = hk.group(1)
        r0 = hk.arch.block_idx.y * 64
        col = hk.arch.block_idx.x * 64
        if staged:
            g.gather_prep(ids, r0, 64, raw=True)
        for i in range(4):
            t = x[hk.elem_coord(0, 0, r0 + 16 * i, col)].load(T32)
            o.scatter(t, ids, r0 + 16 * i, col=col, sort_topk=TOPK, staged=64 if staged else 0)

    return hk.kernel(scat_staged, arch="gfx950", warps=1, name=f"scat_staged_{int(staged)}",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 64, 1))


def test_a_scatter_reads_its_ids_from_lds_as_from_global():
    """scatter(staged=64) after g.gather_prep(raw=True): the same rows as the global-index scatter,
    bit for bit (the ids it reads from LDS keep the choice bits gather_prep's default masks off)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    ids = _sort_ids(g)
    ids = torch.cat([ids, torch.full((-ids.numel() % 64,), M, device="cuda", dtype=torch.int32)])
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    outs = []
    for staged in (False, True):
        out = torch.zeros(TOPK * M, C, device="cuda", dtype=torch.bfloat16)
        _staged_kernel(staged)(x, ids.view(1, -1), out)
        outs.append(out)
    assert torch.equal(outs[1], outs[0])
    assert torch.equal(outs[1], _want(x, ids, False, torch.bfloat16))
