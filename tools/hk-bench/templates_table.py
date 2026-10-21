"""hk.templates.elementwise / rowwise against the tuned hk.ops kernel for the
same job and torch.compile, interleaved in one process: the tables in
docs/templates/elementwise-rowwise.md.

    python3 tools/hk-bench/templates_table.py silu,gelu 1,32,2048,8192 14336,4096
    python3 tools/hk-bench/templates_table.py rms,add_rms,softmax,quant 1,32,2048,8192 4096,5120,8192 out.json

Every arm in one process, rounds rotating the arm order, each round queued
behind a GPU spin, the median round of 11 x 30 calls; eager (host overhead
included) and, where the arm is captured, replayed from a CUDA graph (device
time). On RDNA hk's launcher takes no stream, so its arms are not captured:
read the eager column there. torch.compile: default mode, dynamic=False, one
function a shape, Dynamo's recompile limit raised.
"""
import json
import statistics
import sys

import torch
import torch.nn.functional as F

import hk
from hk.codegen.scaffold import takes_stream

T = hk.templates
M = T.meta

_spin = None


def _spin_gpu(us=200):
    # Keep the GPU busy so each round's calls queue behind work, not an idle
    # GPU (docs/history/32: a late launch onto an idle GPU costs ~10 us).
    global _spin
    if _spin is None:
        _spin = torch.empty(1 << 22, device="cuda")
    for _ in range(max(1, us // 20)):
        _spin.mul_(1.0000001)


def _graph(fn, calls):
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                fn()
        torch.cuda.synchronize()
        return g.replay
    except Exception as e:  # noqa: BLE001
        torch.cuda.synchronize()
        return None


def timeit(arms, rounds=11, calls=30, warm=3, graph=False):
    names = list(arms)
    for f in arms.values():
        for _ in range(warm):
            f()
    torch.cuda.synchronize()
    batch = {}
    for n, f in arms.items():
        g = _graph(f, calls) if graph else None
        if g is not None:
            batch[n] = (g, 1)
        else:
            batch[n] = (f, calls)
    timeit.graphed = {n for n in names if batch[n][1] == 1}
    res = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            f, reps = batch[n]
            _spin_gpu()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(reps):
                f()
            e.record()
            torch.cuda.synchronize()
            res[n].append(s.elapsed_time(e) * 1e3 / calls)
    return {n: statistics.median(v) for n, v in res.items()}


timeit.graphed = set()


def versions():
    import hk
    try:
        import triton
        tv = triton.__version__
    except Exception:
        tv = "-"
    return (f"torch {torch.__version__}, triton {tv}, {torch.cuda.get_device_name()} "
            f"({hk.target.current_arch()}), hk {hk.__version__}")

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


@T.rowwise(outputs=T.per_row(hk.fp32))
def cross_entropy(logits, label):
    loss = T.row_logsumexp(logits) - T.pick(logits, label)
    return loss * (label != -100).to_int()


@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32), hk.bf16))
def add_rms_quant(x, r, w, *, EPS=1e-6):
    h = x + r
    y = h * (T.row_mean(h * h) + EPS).rsqrt() * w
    amax = T.row_max(abs(y)).maximum(1e-10)
    return y * (127.0 / amax), amax / 127.0, h


TEMPLATES = {
    "rms": (rms_norm, lambda r, c: (M((r, c), B), M((c,), B))),
    "add_rms": (add_rms_norm, lambda r, c: (M((r, c), B), M((r, c), B), M((c,), B))),
    "softmax": (softmax, lambda r, c: (M((r, c), B),)),
    "quant": (quantize, lambda r, c: (M((r, c), B),)),
    "ce": (cross_entropy, lambda r, c: (M((r, c), B), M((r,), "torch.int64"))),
    "t2": (add_rms_quant, lambda r, c: (M((r, c), B), M((r, c), B), M((c,), B))),
}

B = torch.bfloat16
# One compiled function a shape: past Dynamo's default recompile limit (8) a
# torch.compile'd function falls back to eager without a word.
torch._dynamo.config.cache_size_limit = 4096
torch._dynamo.config.accumulated_cache_size_limit = 4096
print(versions(), flush=True)
silu_mul = T.elementwise(lambda g, u: hk.math.silu(g) * u)
gelu_mul = T.elementwise(lambda g, u: hk.math.gelu(g) * u)


def c_rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def c_add_rms(x, r, w):
    h = x + r
    hf = h.float()
    return (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype), h


def c_softmax(x):
    return torch.softmax(x.float(), -1).to(x.dtype)


def c_quant(x):
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return (xf * (127.0 / amax)).round().clamp(-128, 127).to(torch.int8), (amax / 127).squeeze(-1)


def case(op, M, N):
    """(arms, check) for one op and shape; N is the width (for the gate/up
    ops, the intermediate size I: the input is (M, 2I))."""
    if op in ("silu", "gelu"):
        x = torch.randn(M, 2 * N, device="cuda", dtype=B)
        g, u = x[:, :N], x[:, N:]
        out = torch.empty(M, N, device="cuda", dtype=B)
        t = silu_mul if op == "silu" else gelu_mul
        act = F.silu if op == "silu" else (lambda v: F.gelu(v, approximate="tanh"))
        cf = torch.compile(lambda x: act(x[:, :N]) * x[:, N:], dynamic=False)
        arms = {"tmpl": lambda: t(g, u, out=out), "compile": lambda: cf(x)}
        try:
            hk.ops.silu_mul(g, u, out)
            arms["ops" if op == "silu" else "ops_silu_proxy"] = lambda: hk.ops.silu_mul(g, u, out)
        except Exception as e:  # noqa: BLE001
            print("   hk.ops.silu_mul refuses:", str(e)[:90])
        err = (t(g, u).float() - act(g.float()) * u.float()).abs().max().item()
        return arms, err, t.plan(g, u)
    x = torch.randn(M, N, device="cuda", dtype=B)
    w = torch.randn(N, device="cuda", dtype=B)
    if op == "rms":
        out = torch.empty_like(x)
        cf = torch.compile(c_rms, dynamic=False)
        err = (rms_norm(x, w).float() - c_rms(x, w).float()).abs().max().item()
        return ({"tmpl": lambda: rms_norm(x, w, out=out),
                 "ops": lambda: hk.ops.rms_norm(x, w, out=out),
                 "compile": lambda: cf(x, w)}, err, rms_norm.plan(x, w))
    if op == "add_rms":
        r = torch.randn_like(x)
        xs, rs = x.clone(), r.clone()
        cf = torch.compile(c_add_rms, dynamic=False)
        y, h = add_rms_norm(x, r, w)
        ry, rh = c_add_rms(x, r, w)
        err = max((y.float() - ry.float()).abs().max().item(), (h.float() - rh.float()).abs().max().item())
        return ({"tmpl": lambda: add_rms_norm(xs, rs, w, out=(xs, rs)),
                 "ops": lambda: hk.ops.fused_add_rms_norm(xs, rs, w),
                 "compile": lambda: cf(x, r, w)}, err, add_rms_norm.plan(x, r, w))
    if op == "softmax":
        out = torch.empty_like(x)
        cf = torch.compile(c_softmax, dynamic=False)
        err = (softmax(x).float() - c_softmax(x).float()).abs().max().item()
        return ({"tmpl": lambda: softmax(x, out=out),
                 "ops": lambda: hk.ops.softmax(x, out=out),
                 "compile": lambda: cf(x)}, err, softmax.plan(x))
    if op == "quant":
        q = torch.empty(M, N, device="cuda", dtype=torch.int8)
        s = torch.empty(M, device="cuda", dtype=torch.float32)
        cf = torch.compile(c_quant, dynamic=False)
        gq, gs = quantize(x)
        rq, rs_ = c_quant(x)
        err = max((gq.float() - rq.float()).abs().max().item(), ((gs - rs_).abs() / rs_).max().item())
        return ({"tmpl": lambda: quantize(x, out=(q, s)),
                 "ops": lambda: hk.ops.quantize(x, out=q, scales=s),
                 "compile": lambda: cf(x)}, err, quantize.plan(x))
    raise ValueError(op)


rows = []
for op in sys.argv[1].split(","):
    for N in (int(v) for v in sys.argv[3].split(",")):
        for M in (int(v) for v in sys.argv[2].split(",")):
            arms, err, plan = case(op, M, N)
            for n in list(arms):
                try:
                    arms[n]()
                except Exception as e:  # noqa: BLE001
                    print(f"   {n} refuses: {str(e)[:90]}")
                    del arms[n]
            eager = timeit(arms)
            # RDNA's launcher takes no stream: a graph would capture nothing of hk.
            dev = timeit(arms, graph=takes_stream(hk.target.current_arch()))
            graphed = set(timeit.graphed)
            row = dict(op=op, M=M, N=N, err=err, plan=str(plan), eager=eager, device=dev,
                       graphed=sorted(graphed))
            rows.append(row)
            ref = "ops" if "ops" in arms else None

            def r(d, a, b):
                return (d[b] / d[a]) if (a in d and b in d) else float("nan")
            print(f"{op} {M}x{N} err {err:.3g} | eager " + " ".join(f"{k} {v:.2f}" for k, v in eager.items())
                  + " | device " + " ".join(f"{k} {v:.2f}{'' if k in graphed else '*'}" for k, v in dev.items())
                  + f" | dev tmpl/ops {r(dev, 'tmpl', 'ops'):.3f} tmpl/compile {r(dev, 'tmpl', 'compile'):.3f}"
                  f" | eager tmpl/ops {r(eager, 'tmpl', 'ops'):.3f} tmpl/compile {r(eager, 'tmpl', 'compile'):.3f}"
                  f" | {plan}", flush=True)
if len(sys.argv) > 4:
    json.dump({"versions": versions(), "rows": rows}, open(sys.argv[4], "w"), indent=1)
