"""vLLM's unquantized linear layers on the generated GEMMs, where they are faster.

GEMM is 57.6% of Qwen3-8B's device time at batch 16 / context 4096
(tools/hk-bench/vllm_profile.py), and on gfx1100 every one of those GEMMs ends
in `torch.nn.functional.linear` -- hipBLASLt. The installed vLLM's skinny-GEMM
path is gated on gfx9, and correctly: its `wvSplitK` is compiled for gfx9 only
and fails a device-side assert on gfx1100.

Where each call goes is a measured table, not a rule:
`hk/tuned/routes_vllm_linear_<arch>.json`, written by
tools/hk-bench/route_tune.py, which times the engine's path, the prefill GEMM
and the decode GEMV on each (N, K) of the model at every M up to 64, every 16
rows to 1024 and coarser above -- each gated against fp32 first -- and routes
an M to a kernel only where it won by 3%. Shapes this machine measured
(`hk.autotune.routes`) are laid over it, and the patch measures the loaded
model's shapes the table lacks when vLLM loads it (`vllm_autotune`), within
`HK_AUTOTUNE_BUDGET`; `python -m hk.integration tune --model X` does the
same before the server starts. `$HK_VLLM_ROUTES` names another table, which
is then the whole table. A shape nobody measured takes vLLM's path, as does
every shape on an arch with no table. See `engine.RouteTable` for how an M
between two measured ones is answered.

The table replaces four hand-written lines because the GEMM's time is a comb
in M, not a curve: the last 128-row block is computed whole however little of
it is kept, so between multiples of 128 the kernel does extra work, and at
small grids two workgroups share a WGP. Thresholds either route the teeth or
give up the gaps.

Decode is a different problem: M is the batch, the math is free, and the
whole cost is streaming the weights once. `ops.decode_gemm` is the kernel for
that, at 1.93-2.03x hipBLASLt per step at M <= 32.

On gfx950 the candidate is a third path, "hipkittens": HipKittens' own CDNA4
GEMM (`ops.hipkittens`), which takes M and N in multiples of 256. Its table,
routes_vllm_linear_gfx950.json, is measured the same way, against F.linear.
gfx1201's, routes_vllm_linear_gfx1201.json, is gfx1100's two paths measured
in the Radeon AI PRO R9700 vLLM image against its rocm_unquantized_gemm.

A table may also name kernels hk does not ship: FlyDSL's GEMM
(`flydsl_gemm`), where FlyDSL is installed and measured faster. Those rows
are an overlay on the table's own (see `engine.RouteTable`), so without
FlyDSL the routes are exactly the table's own ones.

How it gets in. `UnquantizedLinearMethod.apply` calls
`dispatch_unquantized_gemm()` on every forward, as does the LM head, so
rebinding that one name -- by identity, in every vLLM module that imported it
-- routes every unquantized linear without touching a model file.

Why a custom op rather than a Python function. Under vLLM's torch.compile the
token count is a symbolic shape, so branching on it in traced code is a guard
and a recompile. An opaque op with a fake implementation keeps the branch out
of the graph; it is what vLLM does for its own `rocm_unquantized_gemm`, and it
launches on torch's stream, so CUDA-graph capture sees it. Its route depends
on shapes alone, which a graph captured per batch size fixes, so it declares
itself graph-safe -- and `engine` holds it to that under capture.

    HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Optional

from ..runtime import hostpath as _hostpath
from . import _common, engine, flydsl_gemm

PREFIXES = ("vllm",)

#: Paths a table may prefer that come from libraries hk does not depend on:
#: name -> a module with `variant(m, n, k)` (None where it does not take the
#: shape or is not installed) and `run(x, weight, out)`.
OPTIONAL = {flydsl_gemm.NAME: flydsl_gemm}

#: Where a routing table override comes from.
ROUTES_ENV = "HK_VLLM_ROUTES"

_ORIG_DISPATCH: Optional[Callable] = None
#: Layers left on vLLM's own GEMM when they were first called, by reason: the table routes their
#: (dtype, N, K) at no M, so no wrapper stands in their path (see `_takes_layer`).
UNROUTED: "Counter[str]" = Counter()
_GEMMS: Dict[Any, Callable] = {}
_FALLBACK: Optional[Callable] = None
_OPS: Dict[str, Callable] = {}
_OPS_LEGACY: Dict[str, Callable] = {}
_ARCH_OK: Dict[int, bool] = {}
_TABLE: list = []


def _linear_fake(x, weight, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


#: What a table's runs may name: hk's kernels, HipKittens' and the engine's.
#: Optional paths come only through a table's overlay, where their variant
#: is checked.
RUN_PATHS = ("hk", "hk_decode", "hipkittens", "fallback")

PATCH = engine.Op("hk_vllm::linear",
                  "linear(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor",
                  _linear_fake, paths=RUN_PATHS + tuple(OPTIONAL), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def engine_name() -> Optional[str]:
    """The name of the path vLLM's linear runs here, as the tuner records it,
    or None outside a process that has imported vLLM."""
    import sys  # noqa: PLC0415

    if "vllm" not in sys.modules:
        return None
    from ..autotune import routes as _routes  # noqa: PLC0415

    try:
        return _routes.engine_path("vllm")[1]
    except Exception:  # noqa: BLE001 -- a name is for filtering, not for failing
        return None


def routes() -> Optional[engine.RouteTable]:
    """The routing table for this process's arch, or None: the shipped one
    with this machine's measured shapes over it (`engine.shipped_table`).
    Loaded once; `set_routes(None)` reloads it."""
    if not _TABLE:
        from ..target import current_arch  # noqa: PLC0415

        _TABLE.append(engine.shipped_table("vllm_linear", current_arch(), ROUTES_ENV,
                                           engine=engine_name()))
    return _TABLE[0]


def set_routes(table: Optional[engine.RouteTable]) -> None:
    """Route by `table` from now on; None for the shipped one."""
    if any(PATCH.captured.values()):
        import warnings  # noqa: PLC0415

        warnings.warn("hk_vllm::linear: routes changed after a CUDA graph was "
                      "captured; graphs captured before keep replaying the old "
                      "routes", stacklevel=2)
    _TABLE.clear()
    _PLANS.clear()
    if table is not None:
        _TABLE.append(table)


def route_for(m: int, n: int, k: int, *, bf16: bool = True, bias: bool = False,
              contiguous_weight: bool = True) -> str:
    """"hk", "hk_decode", "hipkittens", "flydsl" or "fallback" for y = x @ W.T,
    x (m, k), W (n, k).

    The table says which kernel measured fastest; this also asks whether that
    kernel takes the call at all. An optional path the table prefers here is
    taken only if its library would run the variant that was measured;
    otherwise the table's own answer stands. A bias falls back rather than
    paying a second pass over the output; no Qwen3 linear has one.
    """
    from ..ops import decode_gemm  # noqa: PLC0415

    if m < 1 or not bf16 or bias or not contiguous_weight:
        return "fallback"
    table = routes()
    if table is None:
        return "fallback"
    for name, var in table.preferred(m, n, k):
        opt = OPTIONAL.get(name)
        if opt is not None and opt.variant(m, n, k) == var:
            return name
    path = table.lookup(m, n, k)
    if path == "hk" and not _gemm_takes(n, k):
        return "fallback"
    if path == "hk_decode" and not decode_gemm.fits(m, n, k):
        return "fallback"
    if path == "hipkittens":
        from ..ops import hipkittens  # noqa: PLC0415

        if not hipkittens.gemm_fits(m, n, k):
            return "fallback"
    return path if path in RUN_PATHS else "fallback"


def _gemm_takes(n: int, k: int) -> bool:
    """Whether the "hk" path's GEMM takes an N x K weight on this arch: on
    gfx950 hk's own GEMMs (hk.ops.gemm_cdna, by its measured table), else
    the generated RDNA GEMM."""
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() == "gfx950":
        from ..ops import gemm_cdna  # noqa: PLC0415

        return not (n % gemm_cdna.BLOCK_N_MIN or k % gemm_cdna.K_STEP)
    from ..ops import gemm as hk_gemm  # noqa: PLC0415

    return not (n % hk_gemm.BLOCK_N or k % hk_gemm.K_STEP)


def route(m: int, n: int, k: int, **how) -> bool:
    """Whether the call goes to the prefill GEMM."""
    return route_for(m, n, k, **how) == "hk"


def route_decode(m: int, n: int, k: int, **how) -> bool:
    """Whether the call goes to `ops.decode_gemm`."""
    return route_for(m, n, k, **how) == "hk_decode"


def _arch_ok(device) -> bool:
    """Whether `device` is the arch the kernels are compiled for."""
    idx = device.index if device.index is not None else 0
    ok = _ARCH_OK.get(idx)
    if ok is None:
        import torch  # noqa: PLC0415

        from ..target import current_arch  # noqa: PLC0415

        name = torch.cuda.get_device_properties(idx).gcnArchName or ""
        ok = _ARCH_OK[idx] = name.split(":")[0] == current_arch()
    return ok


def _cdna(device) -> bool:
    """Whether the "hk" path is hk's gfx950 GEMMs here (`_arch_ok` has
    already said the device is the arch the kernels are built for)."""
    from ..target import current_arch  # noqa: PLC0415

    return current_arch() == "gfx950"


def _op_for(m: int, n: int, k: int) -> Callable:
    """The launch of the schedule this shape gets: its torch.ops registration's
    implementation, called directly (`codeobj.direct`) -- `linear` is the body
    of an opaque op, eager code torch.compile does not trace."""
    from ..ops import gemm as hk_gemm  # noqa: PLC0415

    s = hk_gemm.schedule_for(m, n, k)
    key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = _direct(hk_gemm.TUNER._kernel_for(s))
    return op


def _decode_op(s: Dict[str, Any]) -> Callable:
    from ..ops import decode_gemm  # noqa: PLC0415

    key = "decode_" + "_".join(f"{k}{v}" for k, v in sorted(s.items()))
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = _direct(decode_gemm.kernel_for(**s))
    return op


def _op_for_legacy(m: int, n: int, k: int) -> Callable:
    """`_op_for` before the host fast paths: the torch op (HK_HOST_FASTPATH=0)."""
    from ..ops import gemm as hk_gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    s = hk_gemm.schedule_for(m, n, k)
    key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
    op = _OPS_LEGACY.get(key)
    if op is None:
        op = _OPS_LEGACY[key] = torch_op(hk_gemm.TUNER._kernel_for(s))
    return op


def _decode_op_legacy(s: Dict[str, Any]) -> Callable:
    from ..ops import decode_gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    key = "decode_" + "_".join(f"{k}{v}" for k, v in sorted(s.items()))
    op = _OPS_LEGACY.get(key)
    if op is None:
        op = _OPS_LEGACY[key] = torch_op(decode_gemm.kernel_for(**s))
    return op


def _direct(kernel) -> Callable:
    import os  # noqa: PLC0415

    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    if os.environ.get("HK_TORCH_OP", "codeobj") == "library":
        return torch_op(kernel)
    from ..runtime import codeobj  # noqa: PLC0415

    return codeobj.direct(kernel)


def _fallback(x, weight, bias):
    """What vLLM would have run: its own op where it has one, else F.linear."""
    global _FALLBACK
    if _FALLBACK is None:
        import torch  # noqa: PLC0415

        try:
            _FALLBACK = torch.ops.vllm.rocm_unquantized_gemm
        except (AttributeError, RuntimeError):
            _FALLBACK = torch.nn.functional.linear
    return _FALLBACK(x, weight, bias)


#: (M, N, K, dtypes, bias, weight contiguous, device) -> (path, run): where a
#: call of that shape goes and the launch that takes it there, decided once.
#: The decision reads only those, the table and OPTIONAL, so the memo is kept
#: for one table and one OPTIONAL (`_PLANS_FOR`) and dropped when either
#: changes. Deciding took 20 of the 32 us a routed decode call spent in
#: Python on the W7900D.
_PLANS: Dict[tuple, tuple] = {}
_PLANS_FOR: list = [None, None]
#: Shapes memoized; past it a call decides afresh (a serving engine's prefill
#: has a new M per batch).
_PLAN_CAP = 4096


def _plan(x, weight, bias, m: int, n: int, k: int) -> tuple:
    import torch  # noqa: PLC0415

    path = route_for(m, n, k,
                     bf16=x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16,
                     bias=bias is not None, contiguous_weight=weight.is_contiguous())
    if path == "fallback" or not _arch_ok(x.device):
        return "fallback", None
    if path == "hipkittens":
        return path, None
    if path == "hk" and _cdna(x.device):
        from ..ops import gemm_cdna  # noqa: PLC0415

        return path, gemm_cdna.matmul
    if path in OPTIONAL:
        return path, OPTIONAL[path].run
    from ..runtime.resources import ResourceError  # noqa: PLC0415

    # The kernels build here, on the first call of a shape: one this compiler
    # refuses (a spill; hk does not run spilling kernels) sends the shape to
    # vLLM's path, said once, instead of raising inside the model's forward.
    try:
        if path == "hk":
            return path, _op_for(m, n, k)
        from ..ops import decode_gemm  # noqa: PLC0415

        s = decode_gemm.schedule_for(m, k)
        op = _decode_op(s)
    except ResourceError as e:
        _refused(path, e)
        return "fallback", None
    if s["splits"] > 1:
        scratch = decode_gemm.scratch_for(x.device, s["splits"], s["tokens"], n)
        return path, lambda x2, w, out: op(x2, w, out, *scratch)
    return path, op


#: Paths whose kernel this compiler refused, said once each.
_REFUSED: set = set()


def _refused(path: str, err: Exception) -> None:
    if path in _REFUSED:
        return
    _REFUSED.add(path)
    import sys  # noqa: PLC0415

    print(f"[hk] linear: the {path} kernel does not build with this compiler, so the shapes "
          f"routed to it run vLLM's path ({str(err).splitlines()[0]})", file=sys.stderr, flush=True)


def linear(x, weight, bias=None):
    """y = x @ weight.T (+ bias), on the kernel the table names."""
    if not _hostpath.ENABLED:
        return _linear_legacy(x, weight, bias)
    import torch  # noqa: PLC0415

    k = x.shape[-1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    table = routes()
    if _PLANS_FOR[0] is not table or _PLANS_FOR[1] is not OPTIONAL:
        _PLANS.clear()
        _PLANS_FOR[:] = [table, OPTIONAL]
    key = (m, n, k, x.dtype, weight.dtype, bias is None, weight.is_contiguous(),
           x.get_device())
    plan = _PLANS.get(key)
    if plan is None:
        plan = _plan(x, weight, bias, m, n, k)
        if len(_PLANS) < _PLAN_CAP:
            _PLANS[key] = plan
    path, run = plan
    if path == "fallback":
        PATCH.count("fallback")
        return _fallback(x, weight, bias)
    x2 = x.reshape(m, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if path == "hipkittens":
        return _hipkittens(x, x2, weight, bias, m, n)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    run(x2, weight, out)
    PATCH.count(path)
    return out.view(*x.shape[:-1], n)


def _linear_legacy(x, weight, bias=None):
    """`linear` before the host fast paths, line for line (HK_HOST_FASTPATH=0)."""
    import torch  # noqa: PLC0415

    k = x.shape[-1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    path = route_for(m, n, k,
                     bf16=x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16,
                     bias=bias is not None, contiguous_weight=weight.is_contiguous())
    if path == "fallback" or not _arch_ok(x.device):
        PATCH.count("fallback")
        return _fallback(x, weight, bias)
    x2 = x.reshape(m, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if path == "hipkittens":
        return _hipkittens(x, x2, weight, bias, m, n)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if path == "hk" and _cdna(x.device):
        from ..ops import gemm_cdna  # noqa: PLC0415

        gemm_cdna.matmul(x2, weight, out)
    elif path == "hk":
        _op_for_legacy(m, n, k)(x2, weight, out)
    elif path in OPTIONAL:
        OPTIONAL[path].run(x2, weight, out)
    else:
        from ..ops import decode_gemm  # noqa: PLC0415

        s = decode_gemm.schedule_for(m, k)
        if s["splits"] > 1:
            _decode_op_legacy(s)(x2, weight, out,
                                 *decode_gemm.scratch_for(x.device, s["splits"], s["tokens"], n))
        else:
            _decode_op_legacy(s)(x2, weight, out)
    PATCH.count(path)
    return out.view(*x.shape[:-1], n)


def _hipkittens(x, x2, weight, bias, m, n):
    """The "hipkittens" path, or the fallback if this toolchain cannot have the
    kernel (it did not build, or failed its on-device check) or the tensors
    are not ones it takes."""
    from ..ops import hipkittens  # noqa: PLC0415

    if hipkittens.gemm_takes(x2, weight):
        try:
            out = hipkittens.gemm_bf16(x2, weight)
        except hipkittens.Unavailable as e:
            hipkittens.warn_unavailable(PATCH.name, e)
        else:
            PATCH.count("hipkittens")
            return out.view(*x.shape[:-1], n)
    PATCH.count("fallback")
    return _fallback(x, weight, bias)


def register_op():
    """Define `torch.ops.hk_vllm.linear`. Idempotent."""
    PATCH.define(linear)


def _unrouted(weight, bias) -> str:
    """Why the table can take no call of a layer with this weight and bias, at any M -- "" if it
    can take some."""
    import torch  # noqa: PLC0415

    if weight.dtype != torch.bfloat16:
        return f"weight {str(weight.dtype).replace('torch.', '')}"
    if bias is not None:
        return "bias"
    if weight.dim() != 2 or not weight.is_contiguous():
        return "weight not a contiguous matrix"
    table = routes()
    n, k = weight.shape
    if table is None or not table.routes_any(n, k):
        return "no route at any M"
    return ""


def _takes_layer(layer, weight, bias) -> bool:
    """Whether this layer's calls go through hk's op at all. Decided on the layer's first call --
    after the weights are loaded and the missing shapes measured (vllm_autotune), before vLLM's
    profile run or any graph capture -- and kept on the layer for the table it was decided
    against. A layer the table routes at no M keeps vLLM's own GEMM, unwrapped: in a captured
    decode graph the wrapper is an op of its own on every call, measured at ~0.8% of TPOT for a
    model whose dense layers are all fp16."""
    table = routes()
    memo = getattr(layer, "_hk_gemm", None)
    if memo is not None and memo[0] is table:
        return memo[1]
    why = _unrouted(weight, bias)
    if why and (memo is None or memo[1]):
        UNROUTED[why] += 1
    try:
        layer._hk_gemm = (table, not why)
    except AttributeError:          # a layer that takes no attributes: decide every call
        pass
    return not why


def _gemm(layer, x, weight, bias=None):
    """What `dispatch_unquantized_gemm()` hands back once patched, for a layer it routes."""
    import torch  # noqa: PLC0415

    if not x.is_cuda:
        return torch.nn.functional.linear(x, weight, bias)
    if not _hostpath.ENABLED:
        return torch.ops.hk_vllm.linear(x, weight, bias)
    return PATCH(x, weight, bias)


def dispatch_unquantized_gemm(*args, **kwargs):
    """Stands in for vLLM's. The arguments -- a linear-backend name in newer
    releases -- select among CUDA backends and mean nothing on ROCm; they are
    passed on to vLLM's own dispatch, whose GEMM a layer the table never
    routes keeps (`_takes_layer`)."""
    try:
        key = (args, tuple(sorted(kwargs.items())))
        hit = _GEMMS.get(key)
    except TypeError:
        key, hit = None, None
    if hit is not None:
        return hit
    orig = _ORIG_DISPATCH(*args, **kwargs)

    def gemm(layer, x, weight, bias=None):
        if not _takes_layer(layer, weight, bias):
            return orig(layer, x, weight, bias)
        return _gemm(layer, x, weight, bias)

    if key is not None:
        _GEMMS[key] = gemm
    return gemm


def _own_compile_cache(name: str = "hk-linear") -> str:
    """Give the patched engine a vLLM cache root of its own, `name` under the
    current one, and return it.

    vLLM keys the graphs it compiles -- torch_compile_cache, and since 0.2x
    the AOT cache that skips Dynamo altogether -- by its config and its
    source, not by what is patched into it. Sharing a root with an unpatched
    run, the engine loads back a graph whose linear layers call vLLM's GEMM,
    and the routes are never reached. Read before the engine caches its
    environment, so this runs before the engine starts.

    Several routes in one engine (hk is on by default) share one directory named for all of
    them, `hk-int4+mxfp8`, whatever the order they were applied in. A worker inherits its
    parent's root and adds nothing new, so it gets the same one.
    """
    import os  # noqa: PLC0415

    root = os.path.normpath(os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "vllm"))
    names = {name[3:] if name.startswith("hk-") else name}
    while os.path.basename(root).startswith("hk-"):
        names |= set(os.path.basename(root)[3:].split("+"))
        root = os.path.dirname(root)
    root = os.path.join(root, "hk-" + "+".join(sorted(names)))
    os.environ["VLLM_CACHE_ROOT"] = root
    return root


def apply() -> str:
    """Route vLLM's linear layers. Returns what happened, in one line."""
    global _ORIG_DISPATCH
    if not _common.importable("vllm"):
        return "vllm is not importable; linear not patched"
    if _ORIG_DISPATCH is not None:
        return "linear: already patched"
    from vllm.model_executor.layers import utils as vu  # noqa: PLC0415
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "linear: not patched -- not a ROCm platform"
    table = routes()
    if table is None:
        from ..target import current_arch  # noqa: PLC0415

        return (f"linear: not patched -- no routing table for {current_arch()}; "
                f"tools/hk-bench/route_tune.py writes one")
    register_op()
    _GEMMS.clear()
    _ORIG_DISPATCH = vu.dispatch_unquantized_gemm
    names = _common.rebind(PREFIXES, _ORIG_DISPATCH, dispatch_unquantized_gemm)
    cache = _own_compile_cache()
    from . import vllm_autotune  # noqa: PLC0415

    return (f"linear -> hk where {table_status(table)} measured it faster "
            f"({len(names)} binding{'' if len(names) == 1 else 's'} rebound; "
            f"compile cache {cache})" + optional_status(table)
            + f"; {vllm_autotune.install()}")


def table_status(table: engine.RouteTable) -> str:
    """Where a table's shapes were measured, for a patch's report line."""
    by: Dict[str, int] = {}
    for src in table.origin.values():
        by[src] = by.get(src, 0) + 1
    return ", ".join(f"{src or 'records'} ({n} shape{'s' * (n != 1)})"
                     for src, n in by.items()) or "no table"


def optional_status(table: engine.RouteTable) -> str:
    """What became of the table's optional rows, for a patch's report line."""
    return "".join(f"; {name} rows: {OPTIONAL[name].status()}"
                   for name in sorted(table.optional_paths()) if name in OPTIONAL)


def revert() -> str:
    global _ORIG_DISPATCH
    if _ORIG_DISPATCH is None:
        return "linear: not patched"
    names = _common.rebind(PREFIXES, dispatch_unquantized_gemm, _ORIG_DISPATCH)
    _ORIG_DISPATCH = None
    _GEMMS.clear()
    return f"linear -> vllm ({len(names)} rebound)"


def decide_layers(model) -> str:
    """Decide every unquantized linear layer of a loaded vLLM model, and its LM head, now --
    after the weights and the shapes measured at load, before the profile run or any capture --
    rather than on each one's first call. Returns the report: how many go through hk's op, and
    how many were left on vLLM's own GEMM, by reason."""
    import torch  # noqa: PLC0415

    head = getattr(model, "lm_head", None)
    routed = 0
    for _, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        qm = type(getattr(mod, "quant_method", None)).__name__
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            continue
        if qm == "UnquantizedEmbeddingMethod":
            if mod is not head and type(mod).__name__ != "ParallelLMHead":
                continue                    # an embedding lookup, not a GEMM
            bias = None
        elif qm == "UnquantizedLinearMethod":
            bias = None if getattr(mod, "skip_bias_add", False) else getattr(mod, "bias", None)
        else:
            continue
        routed += _takes_layer(mod, w, bias)
    return f"{routed} layer{'s' * (routed != 1)} through hk's op; {unrouted_status()}"


def unrouted_status() -> str:
    """The layers left on vLLM's GEMM at load, by reason, for a report line."""
    if not UNROUTED:
        return "unrouted at load: none"
    return "unrouted at load: " + ", ".join(f"{n} ({why})" for why, n in UNROUTED.most_common())


def status() -> str:
    if _ORIG_DISPATCH is None:
        return "not patched"
    return f"patched; {unrouted_status()}"
