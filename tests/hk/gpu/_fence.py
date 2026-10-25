"""Run GPU code with every allocation end-fenced: the harness of test_fenced.py.

A read or write past the end of a tensor usually lands in the caching allocator's next block and
goes unnoticed -- results stay bit-exact -- until the tensor happens to end its segment and the
page after it is unmapped. In vLLM that is a memory fault some runs and not others (mxfp8.quant's
whole-tile copy of x did it in B1's real-weight GSM8K runs, PR #47). Under `fenced`, every
allocation of a child process comes from fence_alloc.cpp, which ends it flush against an unmapped
VMM granule, so any such access faults at once, on every run.

    from _fence import fenced
    fenced('''
    from hk.ops import mxfp8
    x = torch.randn(65, 128, device="cuda").bfloat16()
    q, s = step("quant M65", lambda: mxfp8.quant(x))
    same("q", q, reference_q)
    ''')

The code runs in a child (from a file: Triton's @jit reads its source) after a prelude that
installs the allocator and defines:

  step(label, fn)          print the label, call fn, synchronize: a fault names the call
  same(what, got, want)    bitwise equality, on the host
  close(what, got, want, atol=0, rtol=0)   torch.testing on the host

Compare on the host: torch 2.12 (the vLLM image) segfaults reading a scalar of VMM memory back
-- .item(), torch.equal -- and the prelude routes Python's scalar reads through a host copy, but
C++'s (torch.equal) it cannot. Batch the cases of an op family into one child: each child pays
for torch's import and a cold kernel cache in its own process.

FENCE_ALIGN=16: an allocation ends exactly where its tensor does for every tensor whose byte size
is a multiple of 16 (the allocator's default, 256, would leave up to 255 bytes of slack that
hides a short over-read). The children never write a core: RLIMIT_CORE 0 and
HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 (ROCr's GPU core dump).
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

HERE = pathlib.Path(__file__).parent
SOURCE = HERE / "fence_alloc.cpp"

_PRELUDE = """
import torch
torch.cuda.memory.change_current_allocator(
    torch.cuda.memory.CUDAPluggableAllocator({lib!r}, "fence_malloc", "fence_free"))


# torch 2.12 (the vLLM image) segfaults reading one element of VMM memory back -- in
# _local_scalar_dense_cuda, under .item(), float(t), int(t), bool(t), torch.equal; torch 2.10
# does not. Through a host copy it is fine: Python's own reads take one (the library's too).
def _via_host(name):
    orig = getattr(torch.Tensor, name)
    setattr(torch.Tensor, name, lambda self, *a: orig(self.cpu() if self.is_cuda else self, *a))


for _name in ("item", "__float__", "__int__", "__bool__", "__index__"):
    _via_host(_name)


def step(label, fn):
    print(label, flush=True)
    out = fn()
    torch.cuda.synchronize()
    return out


def _host(t):
    return t.detach().cpu()


def same(what, got, want):
    got, want = _host(got), _host(want)
    assert got.shape == want.shape and got.dtype == want.dtype, (what, got.shape, want.shape,
                                                                 got.dtype, want.dtype)
    if got.dtype.is_floating_point and got.element_size() in (1, 2, 4):
        bits = {{1: torch.uint8, 2: torch.int16, 4: torch.int32}}[got.element_size()]
        got, want = got.view(bits), want.view(bits)
    assert torch.equal(got, want), f"{{what}}: not bitwise equal"


def close(what, got, want, atol=0.0, rtol=0.0):
    torch.testing.assert_close(_host(got).float(), _host(want).float(), atol=atol, rtol=rtol,
                               msg=lambda m: f"{{what}}: {{m}}")
"""


def _cache_dir() -> pathlib.Path:
    base = os.environ.get("HK_CACHE_DIR") or os.path.join(tempfile.gettempdir(), f"hk-{os.getuid()}")
    return pathlib.Path(base) / "fence"


@functools.lru_cache(maxsize=None)
def library() -> str:
    """fence_alloc.cpp built once per source (cached next to hk's kernel cache)."""
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    if not os.path.exists(hipcc):
        pytest.skip("no hipcc to build the fence allocator")
    key = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"fence_alloc-{key}.so"
    if not out.exists():
        tmp = tempfile.mkdtemp(prefix="fence-build-", dir=d)
        so = os.path.join(tmp, "fence_alloc.so")
        subprocess.run([hipcc, "-O2", "-shared", "-fPIC", "-o", so, str(SOURCE)],
                       check=True, capture_output=True)
        os.replace(so, out)         # atomic: concurrent pytest processes may race to build it
        shutil.rmtree(tmp, ignore_errors=True)
    return str(out)


def _no_cores():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run(code: str, timeout: int = 1200, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `code` (dedented) after the prelude in a fenced child; return the finished process."""
    src = _PRELUDE.format(lib=library()) + textwrap.dedent(code) + '\nprint("FENCE CLEAN")\n'
    e = dict(os.environ, FENCE_ALIGN="16", HSA_DISABLE_COREDUMP_ON_EXCEPTION="1")
    e.update(env or {})
    with tempfile.TemporaryDirectory(prefix="hk-fence-") as d:
        script = os.path.join(d, "fenced_case.py")      # a file: Triton's @jit reads its source
        with open(script, "w") as f:
            f.write(src)
        return subprocess.run([sys.executable, script], env=e, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=_no_cores)


def _failure(p: subprocess.CompletedProcess) -> str | None:
    if p.returncode == 0 and "FENCE CLEAN" in p.stdout:
        return None
    steps = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return f"fenced run failed (rc {p.returncode}) in step {steps[-1:]}\n{p.stderr[-3000:]}"


def fenced(code: str, timeout: int = 1200, env: dict | None = None) -> str:
    """Run `code` fenced; fail naming the last step it printed. Returns the child's stdout."""
    p = run(code, timeout, env)
    why = _failure(p)
    if why:
        pytest.fail(why, pytrace=False)
    return p.stdout


def fenced_cases(setup: str, cases: dict, timeout: int = 1800) -> None:
    """An op family: `setup` (imports, helpers), then each of `cases` ({name: code}) in one
    child. A fault kills the child, so the cases after it do not run; HK_FENCE_EACH=1 runs
    each case in a child of its own instead and reports every one that fails (an audit)."""
    setup = textwrap.dedent(setup)
    if os.environ.get("HK_FENCE_EACH") == "1":
        bad = []
        for name, code in cases.items():
            why = _failure(run(setup + f"\nprint({name!r}, flush=True)\n" + textwrap.dedent(code),
                               timeout))
            if why:
                bad.append(f"[{name}] {why}")
        if bad:
            pytest.fail("\n\n".join(bad), pytrace=False)
        return
    body = "".join(f"\nprint({name!r}, flush=True)\n" + textwrap.dedent(code)
                   for name, code in cases.items())
    fenced(setup + body, timeout)
