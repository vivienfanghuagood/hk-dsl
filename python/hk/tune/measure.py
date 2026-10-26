"""Timing GEMM candidates on the GPU, the way the project times them.

`GemmBench(m, n, k)` holds one problem: activations, weight copies rotated
past the arch's Infinity Cache (a model's weights are not resident when its
next layer runs), an output, and an fp32 reference. Then:

  * `check(fn)`: the candidate's output against fp32 (relative max error),
    before it is ever timed -- a fast wrong kernel is not a candidate;
  * `time(arms, rounds)`: interleaved rounds in this one process, each arm's
    order rotated every round; a round of an arm is one pass over every
    weight copy, queued behind a GPU spin longer than the host takes to
    launch the pass (docs/history/32: launches that reach an idle GPU 5-16 us
    late run ~10 us slow, so an unqueued burst times the launcher), between
    two events; its time is per call;
  * `refresh()`: new activations, for the next set of rounds.

It also keeps the GPU-seconds it spent (`gpu_seconds`): every second between
the first and the last GPU operation of each check and timing call.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

#: Bytes of weights one pass streams: three times the arch's Infinity Cache.
ROTATE = {"gfx950": 3 * 256 * 2**20, "gfx1100": 3 * 96 * 2**20, "gfx1201": 3 * 64 * 2**20}
MAX_COPIES = 48
#: Relative max error a candidate may have against fp32 (the route tools').
TOL = 2e-2

_SPIN_HZ: List[float] = []


def spin_hz(torch) -> float:
    """The clock torch.cuda._sleep counts in, cycles a second, measured once:
    after a warm-up call (the first one loads its module, ~15 ms, which a
    calibration that times it reads as a clock 100x too slow -- and a spin
    100x too short, which queues nothing), the best of three 2M-cycle spins."""
    if not _SPIN_HZ:
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            torch.cuda._sleep(2_000_000)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        _SPIN_HZ.append(2_000_000 / (best * 1e-3))
    return _SPIN_HZ[0]


def _spin(torch, seconds: float) -> None:
    """Hold the stream for `seconds` with torch's spin kernel."""
    torch.cuda._sleep(int(seconds * spin_hz(torch)))


class GemmBench:
    """One M x N x K problem on the current GPU."""

    def __init__(self, m: int, n: int, k: int, arch: str, *, seed: int = 0,
                 rotate_bytes: Optional[int] = None, device=None):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.m, self.n, self.k, self.arch = m, n, k, arch
        self.dev = torch.device("cuda") if device is None else torch.device(device)
        rot = ROTATE.get(arch, ROTATE["gfx950"]) if rotate_bytes is None else rotate_bytes
        self.copies = max(2, min(MAX_COPIES, -(-rot // (n * k * 2))))
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.ws = [torch.randn(n, k, device=self.dev, dtype=torch.bfloat16, generator=self.gen)
                   * k ** -0.5 for _ in range(self.copies)]
        self.out = torch.empty(m, n, device=self.dev, dtype=torch.bfloat16)
        self.gpu_seconds = 0.0
        self.load_seconds = 0.0
        self.refresh()
        self._ref = None

    def refresh(self) -> None:
        """New activations: the next set's rounds run on inputs of their own."""
        t = self.torch
        self.x = t.randn(self.m, self.k, device=self.dev, dtype=t.bfloat16, generator=self.gen)
        self._ref = None

    def _reference(self):
        if self._ref is None:
            w32 = self.ws[0].float().t()
            step = max(1, (256 << 20) // (4 * self.n))
            self._ref = [self.x[i:i + step].float() @ w32 for i in range(0, self.m, step)]
            self._scale = max(r.abs().max().item() for r in self._ref)
        return self._ref

    def load(self, fn: Callable) -> float:
        """The candidate's first call, synchronized: its modules loaded (and
        anything the compiler did not build, compiled). CPU time, kept apart
        from gpu_seconds; returns seconds."""
        t0 = time.perf_counter()
        fn(self.x, self.ws[0], self.out)
        self.torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.load_seconds += dt
        return dt

    def check(self, fn: Callable) -> float:
        """Relative max error of fn(x, w, out) against fp32 (inf if it raised
        or wrote non-finite values)."""
        t = self.torch
        t0 = time.perf_counter()
        try:
            ref = self._reference()
            self.out.fill_(float("nan"))
            y = fn(self.x, self.ws[0], self.out)
            y = self.out if y is None else y
            t.cuda.synchronize()
            step = max(1, (256 << 20) // (4 * self.n))
            err = 0.0
            for i, r in zip(range(0, self.m, step), ref):
                d = (y[i:i + step].float() - r).abs().max().item()
                if d != d:
                    return float("inf")
                err = max(err, d)
            return err / max(self._scale, 1e-30)
        finally:
            self.gpu_seconds += time.perf_counter() - t0

    def time(self, arms: Dict[str, Callable], rounds: int) -> Dict[str, List[float]]:
        """{arm: [ms per call, per round]}, interleaved, queued."""
        t = self.torch
        t0 = time.perf_counter()
        try:
            names = list(arms)
            per: Dict[str, List[float]] = {a: [] for a in names}
            host: Dict[str, float] = {}
            for a in names:
                fn = arms[a]
                fn(self.x, self.ws[-1], self.out)
                t.cuda.synchronize()
                h0 = time.perf_counter()
                for w in self.ws:
                    fn(self.x, w, self.out)
                host[a] = time.perf_counter() - h0
                t.cuda.synchronize()
            for r in range(rounds):
                j = r % len(names)
                order = names[j:] + names[:j]
                if r % 2:
                    order = order[::-1]
                for a in order:
                    fn = arms[a]
                    fn(self.x, self.ws[-1], self.out)
                    t.cuda.synchronize()
                    _spin(t, 2 * host[a] + 5e-4)
                    s, e = t.cuda.Event(True), t.cuda.Event(True)
                    s.record()
                    for w in self.ws:
                        fn(self.x, w, self.out)
                    e.record()
                    t.cuda.synchronize()
                    per[a].append(s.elapsed_time(e) / self.copies)
            return per
        finally:
            self.gpu_seconds += time.perf_counter() - t0

    def close(self) -> None:
        del self.ws, self.x, self.out
        self._ref = None
        self.torch.cuda.empty_cache()


class Compiler:
    """Builds candidates' kernels on the CPU, `workers` hipcc processes at a
    time, while the GPU times something else.

    Tracing stays on the calling thread (hk's IR builder is a module-global
    stack, hk.runtime.warm); `submit` traces and emits C++ at once and hands
    hipcc to a thread pool; `result` waits for one candidate's builds."""

    def __init__(self, workers: int = 32):
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: Dict[object, list] = {}
        self.errors: Dict[object, str] = {}
        self.cpu_seconds = 0.0

    def submit(self, key, jobs) -> None:
        """Trace `jobs` ([(label, kernel, constexprs)]) here and hand their
        hipcc builds to the pool: the artifact the launch path will ask for,
        a device-only code object under HK_LAUNCH=codeobj (what hk.tune
        launches its candidates through, codeobj.load's build exactly), else
        the pybind module."""
        import os  # noqa: PLC0415

        from ..runtime.compile import build  # noqa: PLC0415

        if key in self.futures or key in self.errors:
            return
        co = os.environ.get("HK_LAUNCH", "pybind") == "codeobj"
        futs = []
        for label, kernel, consts in jobs:
            try:
                if co:
                    from ..codegen import scaffold as sc  # noqa: PLC0415

                    ir = kernel.trace(**consts)
                    src = sc.device(ir)
                    wd = getattr(kernel, "with_defines", None)
                    if wd:
                        src = wd(src)
                    bf = getattr(kernel, "build_flags", None)
                    args = dict(name=ir.name, kind="co", extra_flags=bf(ir) if bf else (),
                                max_vgprs=getattr(kernel, "max_vgprs", None),
                                min_occupancy=getattr(kernel, "min_occupancy", None))
                    arch = ir.arch
                else:
                    src = kernel.source("pybind", **consts)
                    args = dict(name=kernel.name, max_vgprs=kernel.max_vgprs,
                                min_occupancy=kernel.min_occupancy)
                    arch = kernel.arch
            except BaseException as e:  # noqa: BLE001 -- a refusal is an answer
                self.errors[key] = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}"
                return

            def one(src=src, arch=arch, args=args):
                t0 = time.perf_counter()
                try:
                    return build(src, arch, **args)
                finally:
                    self.cpu_seconds += time.perf_counter() - t0
            futs.append(self.pool.submit(one))
        self.futures[key] = futs

    def result(self, key) -> str:
        """"" if every kernel of the candidate built, else why not."""
        from ..errors import VerifyError  # noqa: PLC0415
        from ..runtime.resources import ResourceError  # noqa: PLC0415

        if key in self.errors:
            return self.errors[key]
        for f in self.futures.get(key, []):
            try:
                f.result()
            except (ResourceError, VerifyError) as e:
                self.errors[key] = "rejected: " + " ".join(str(e).split())[:240]
                return self.errors[key]
            except BaseException as e:  # noqa: BLE001
                self.errors[key] = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}"
                return self.errors[key]
        return ""

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
