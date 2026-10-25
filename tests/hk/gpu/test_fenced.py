"""The DeepSeek-V4.1 routes' kernels touch nothing past the end of any tensor, at ragged M.

A read past a tensor usually lands in the caching allocator's next block and goes unnoticed --
the results stay bit-exact -- until the tensor happens to end its segment and the page after it
is unmapped. In vLLM that is a memory fault some runs and not others: mxfp8.quant's whole-tile
copy of x did it in B1's real-weight GSM8K runs. Here every allocation of a child process comes
from fence_alloc.cpp, which ends each one flush against an unmapped granule, so any such access
faults at once, on every run. Each case prints its name before it launches and synchronizes
after, so a failure names the call. Results are compared on the host: under a pluggable
allocator, torch.equal and full reductions on the device crash the process (PyTorch's, not the
fence's).
"""

from __future__ import annotations

import functools
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's routes", allow_module_level=True)

pytestmark = pytest.mark.gpu

HERE = pathlib.Path(__file__).parent

_PRELUDE = """
import torch
torch.cuda.memory.change_current_allocator(
    torch.cuda.memory.CUDAPluggableAllocator({lib!r}, "fence_malloc", "fence_free"))

def step(label, fn):
    print(label, flush=True)
    out = fn()
    torch.cuda.synchronize()
    return out
"""


@functools.lru_cache(maxsize=None)
def _library() -> str:
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    if not os.path.exists(hipcc):
        pytest.skip("no hipcc to build the fence allocator")
    out = os.path.join(tempfile.mkdtemp(prefix="hk-fence-"), "fence_alloc.so")
    subprocess.run([hipcc, "-O2", "-shared", "-fPIC", "-o", out, str(HERE / "fence_alloc.cpp")],
                   check=True, capture_output=True)
    return out


def _no_cores():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def fenced(code: str, timeout: int = 900) -> None:
    """Run `code` in a child whose every allocation is end-fenced; fail with the last step."""
    src = _PRELUDE.format(lib=_library()) + textwrap.dedent(code) + '\nprint("FENCE CLEAN")\n'
    # 16-byte alignment: allocations end exactly where the tensor does for every case below
    env = dict(os.environ, FENCE_ALIGN="16", HSA_DISABLE_COREDUMP_ON_EXCEPTION="1")
    with tempfile.TemporaryDirectory(prefix="hk-fence-") as d:
        script = os.path.join(d, "fenced_case.py")      # a file: Triton's @jit reads its source
        with open(script, "w") as f:
            f.write(src)
        p = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True,
                           timeout=timeout, preexec_fn=_no_cores)
    if p.returncode or "FENCE CLEAN" not in p.stdout:
        steps = [ln for ln in p.stdout.splitlines() if ln.strip()]
        pytest.fail(f"fenced run failed (rc {p.returncode}) in step {steps[-1:]}\n"
                    f"{p.stderr[-3000:]}")


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


def test_mxfp8_route_at_ragged_m():
    """vllm_mxfp8's hk path -- quant, then gemm_4w or gemm_8w_tail -- at ragged M."""
    fenced("""
    from hk.integration import vllm_mxfp8 as P
    from hk.ops import mxfp8

    for (n, k), ms in {(1792, 5120): (2049, 2111), (2304, 5120): (2049,),
                       (5120, 4096): (5121, 8193), (16384, 1280): (2049, 16447)}.items():
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.full((n // 32, k // 32), 127, device="cuda", dtype=torch.uint8)
        rows = mxfp8.expand_block_scales(ws)[:n]
        wa, wp = mxfp8.aiter_scales(rows), mxfp8.pack_scales(rows)
        for m in ms:
            x = torch.randn(m, k, device="cuda").bfloat16()
            step(f"mxfp8_linear M{m} N{n} K{k} {P.schedule(m, n, k)}",
                 lambda: P.mxfp8_linear(x, w, ws, wa, wp, torch.bfloat16))
    """)


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
    for m in (8193, 8255):
        x = torch.randn(m, k, device="cuda").bfloat16()
        wt = torch.full((m, topk), 1.0 / topk, device="cuda")
        ids = torch.rand(m, e, device="cuda").topk(topk, 1).indices.to(torch.int32)
        step(f"moe M{m}", lambda: PM.moe(x, w1s, w2s, wt, ids, sc1, sc2, 10.0))
        ids = (torch.arange(m * topk, device="cuda", dtype=torch.int32) % 7).view(m, topk)
        step(f"moe M{m} skewed", lambda: PM.moe(x, w1s, w2s, wt, ids, sc1, sc2, 10.0))
    """)
