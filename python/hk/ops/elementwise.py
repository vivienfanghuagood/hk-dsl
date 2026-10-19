"""Elementwise kernels, written in the DSL.

These ship with the package, so they double as the worked examples: each one is
a kernel somebody could have written, not a special case the compiler knows
about.

The kernels are 2D. An elementwise op does not care about the shape, only about
the element count, so the wrappers flatten to (rows, cols) and let the pybind
layer left-pad that to the 4D global layout. What the kernels do *not* do yet is
handle a partial tile: the tail block would read past the end of the tensor,
which reads garbage rather than failing, so `_as_2d` refuses that shape instead
of masking it. Bounds-checked tails come with the reductions in Phase 2.
"""


from functools import lru_cache as _lru_cache

from typing import Any, Tuple

from .. import dsl as hk
from ..dsl.values import _FP32_ONLY
from ..runtime.resources import ResourceError

#: Tile the elementwise kernels work in. 16x64 is 4 base tiles: 32 VGPRs for a
#: bf16 tile on gfx1100, so three live tiles sit nowhere near the granule
#: boundary, and 64 columns is wide enough that the row-layout global load
#: issues 16-byte accesses rather than falling back to the elementwise path.
ROWS, COLS = 16, 64


def _grid(p):
    return (hk.cdiv(p.o.cols, p.COLS), hk.cdiv(p.o.rows, p.ROWS), 1)


def _tile(dtype, rows, cols):
    """The register tile a block moves. Where tiles take a fragment shape
    (gfx950) a 16-bit tile is laid out in 16x32 fragments: a lane's run is then 8 elements, one
    16-byte buffer_load_dwordx4, where the 16x16 fragment's is an 8-byte
    dwordx2. Measured on an MI355X with the 16x256 tile the wrappers pick
    there, bf16 add at 8192x8192 went from 0.58x torch to 0.92x."""
    if hk.arch.target().tile_shapes and dtype.bits == 16:
        return hk.RT(dtype, rows, cols, frag=(16, 32))
    return hk.RT(dtype, rows, cols)


#: op -> what it does to two tiles, or to one.
_BINARY_FN = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
              "mul": lambda x, y: x * y, "max": hk.max, "min": hk.min}
_UNARY_FN = {"exp": lambda x: x.exp(), "exp2": lambda x: x.exp2(),
             "relu": lambda x: x.relu(), "gelu": lambda x: x.gelu(),
             "abs": lambda x: x.abs(), "neg": lambda x: -x}


def _binary_kernel(name: str, op: str, dtype):
    """Build a two-input elementwise kernel for one dtype.

    A factory rather than fifteen copied function bodies: the body is four lines
    and the only thing that varies is which map it calls, so copies would only
    be a place for them to drift.
    """
    fn = _BINARY_FN[op]

    def body(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
             ROWS=ROWS, COLS=COLS):
        t = _tile(dtype, ROWS, COLS)
        idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
        x = a[idx].load(t)
        y = b[idx].load(t)
        o[idx] = fn(x, y)

    body.__name__ = name
    return hk.kernel(body, warps=1, grid=_grid, name=name)


def _unary_kernel(name: str, op: str, dtype):
    fn = _UNARY_FN[op]

    def body(a: hk.Tensor[dtype], o: hk.Tensor[dtype], *, ROWS=ROWS, COLS=COLS):
        t = _tile(dtype, ROWS, COLS)
        idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
        o[idx] = fn(a[idx].load(t))

    body.__name__ = name
    return hk.kernel(body, warps=1, grid=_grid, name=name)


def _strip_kernel(name: str, op: str, dtype, warps: int):
    """`op` on strips over a flat (1, n) view: warp w of workgroup g owns
    elements [(g*WARPS + w)*CW, +CW), CW a constexpr of whole wave x RUN
    chunks, in fp32 between a load that widens and a store that rounds to
    nearest even -- torch's arithmetic for a 16-bit tensor. Without EXACT the
    accesses are bounded: a run at or past n reads zero and is not written."""
    unary = op in _UNARY_FN
    fn = _UNARY_FN[op] if unary else _BINARY_FN[op]

    def strip(a, CW, RUN):
        return hk.RS(hk.fp32, 1, CW, RUN)

    def at(CW):
        blk = hk.arch.block_idx.x * warps + hk.arch.warp_id() if warps > 1 else \
            hk.arch.block_idx.x
        return blk * CW

    if unary:
        def body(a: hk.Tensor[dtype], o: hk.Tensor[dtype], *, CW=512, RUN=8, EXACT=1):
            c0 = at(CW)
            x = a.at[0, c0].load(strip(a, CW, RUN), bounded=not EXACT)
            o.at[0, c0].store(fn(x), bounded=not EXACT)
    else:
        def body(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
                 CW=512, RUN=8, EXACT=1):
            c0 = at(CW)
            S = strip(a, CW, RUN)
            x = a.at[0, c0].load(S, bounded=not EXACT)
            y = b.at[0, c0].load(S, bounded=not EXACT)
            o.at[0, c0].store(fn(x, y), bounded=not EXACT)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (hk.cdiv(p.o.cols, warps * p.CW), 1, 1))


_DTYPES = {"bf16": hk.bf16, "fp16": hk.fp16, "fp32": hk.fp32}
_BINARY = ("add", "sub", "mul", "max", "min")
_UNARY = ("exp", "exp2", "relu", "gelu", "abs", "neg")


def _supports(op: str, suffix: str) -> bool:
    """Whether include/rdna3 implements `op` for this element type.

    gelu is specialised for float only, and instantiating it for bf16 fails at
    link time; the DSL refuses it at trace time for the same reason. `neg` is
    a multiply by -1, so it inherits mul's support.
    """
    return suffix == "fp32" or op not in _FP32_ONLY


#: name -> Kernel, e.g. KERNELS["add_bf16"]. Nothing is compiled until called.
#: Not every (op, dtype) pair is present; use `_kernel_for` to get a diagnosis
#: rather than a KeyError.
KERNELS = {}
for _suffix, _dt in _DTYPES.items():
    for _op in _BINARY:
        if not _supports(_op, _suffix):
            continue
        _n = f"{_op}_{_suffix}"
        KERNELS[_n] = _binary_kernel(_n, _op, _dt)
    for _op in _UNARY:
        if not _supports(_op, _suffix):
            continue
        _n = f"{_op}_{_suffix}"
        KERNELS[_n] = _unary_kernel(_n, _op, _dt)
del _suffix, _dt, _op, _n

#: Warp counts a strip workgroup may have.
STRIP_WARPS = (1, 2, 4, 8)

#: (op, dtype suffix, warps) -> the strip kernel; CW, RUN and EXACT are
#: constexprs, so one entry covers every size.
STRIP_KERNELS = {f"strip_{_op}_{_s}_w{_w}": _strip_kernel(f"strip_{_op}_{_s}_w{_w}", _op, _dt, _w)
                 for _s, _dt in _DTYPES.items() for _op in _BINARY + _UNARY
                 if _supports(_op, _s) for _w in STRIP_WARPS}

#: Where the elementwise ops run on strips: gfx950, measured there
#: (scoreboard.py --only llm-pointwise). gfx1100 keeps the tile kernels it
#: was tuned on.
STRIP_ARCHS = ("gfx950",)

#: The strip plan, (warps, chunks a warp) -- a chunk is wave x RUN elements,
#: one 16-byte access a lane: eight warps of one chunk, 4096 bf16 a
#: workgroup, except for a two-input op moving 24-48 MB, which takes eight
#: chunks a warp. Measured on an MI355X (tools/hk-bench/strip_pointwise_ab.py,
#: bf16, one process, each plan's time over the fastest of tuned Triton,
#: Inductor and torch; two runs, the worse of the two):
#:
#:                      moved    8x1     8x8     4x2     1x2
#:     add  2048x4096    48 MB   0.88    1.18    0.93    0.84
#:     add  2048x5120    60 MB   0.97    0.93    1.00    0.97
#:     add  8192x8192   384 MB   0.97    0.88    0.95    0.98
#:     exp  2048x4096    32 MB   1.01    0.79    0.98    1.01
#:     exp  8192x8192   256 MB   1.00    0.93    0.97    0.98
#:     12 shapes x 2 runs, mean  0.985   0.959   0.980   0.984
#:
#: At 1-32 rows every plan is at the ~3.9 us floor. The cache-resident sizes
#: move with where the buffers sit: in a run with three buffer sets the 8x8
#: add at 2048x4096 took 7.96-8.34 us to tuned Triton's 7.5, where these two
#: runs, and the plan sweep on another GPU, had it 18-20% ahead.
STRIP_PLANS = (((24 << 20, 48 << 20), (8, 8)), (None, (8, 1)))

#: Past 2 GiB a strip's bounded offsets would reach the out-of-range marker.
_STRIP_MAX_BYTES = 1 << 31


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(
            f"hk elementwise handles {sorted(set(_TORCH_DTYPES.values()))}, not {t.dtype}"
        )
    return name


@_lru_cache(maxsize=None)
def _col_choices(suffix: str) -> Tuple[int, ...]:
    """The tile widths to try, widest first. On a wave64 (CDNA) target a
    wider tile is more bytes in flight a lane (with _tile's 16x32 fragment,
    16-byte runs): measured on an MI355X, 16x256 was the fastest 16-bit
    shape and 16x128 the fastest fp32 one, within 1% of each other across
    1-4 warps. Elsewhere, the 16x64 the kernels were tuned for."""
    from ..target import current_arch, get_target  # noqa: PLC0415

    if get_target(current_arch()).wave != 64:
        return (COLS,)
    return (128, COLS) if suffix == "fp32" else (256, 128, COLS)


@_lru_cache(maxsize=None)
def _tile_shape(shape: Tuple[int, ...], numel: int, cols: int = COLS) -> Tuple[int, int]:
    """Pick the (rows, cols) 2D view to launch over.

    Memoized: `_apply` asks it once per operand plus once for the output, and
    the answer depends on nothing but the two arguments. lru_cache does not
    cache the ValueError below, so a shape that does not tile keeps raising.

    Preferring the tensor's own last axis keeps rows contiguous exactly as the
    caller laid them out; the flattened fallback is for the case where the last
    axis is not a multiple of the tile width, which for a hidden dimension it
    essentially never is.
    """
    last = shape[-1] if shape else 0
    if last and last % cols == 0 and (numel // last) % ROWS == 0:
        return numel // last, last
    if numel % (ROWS * cols) == 0:
        return numel // cols, cols
    raise ValueError(
        f"shape {tuple(shape)} ({numel} elements) does not tile: hk's elementwise "
        f"kernels need either a last dim that is a multiple of {COLS} with a "
        f"multiple of {ROWS} rows, or an element count divisible by "
        f"{ROWS * COLS}. A partial tile would read past the end of the tensor, "
        f"which returns garbage instead of failing, so it is refused."
    )


def _as_2d(t, cols: int = COLS):
    if not t.is_contiguous():
        raise ValueError(
            "hk kernels take contiguous tensors; call .contiguous() first "
            "(the global layout is a base pointer plus strides, and a "
            "non-contiguous view does not have the strides it claims)."
        )
    r, c = _tile_shape(tuple(t.shape), t.numel(), cols)
    return t if t.dim() == 2 and t.shape[0] == r and t.shape[1] == c \
        else t.view(r, c)


@_lru_cache(maxsize=None)
def _kernel_for(op: str, suffix: str):
    k = KERNELS.get(f"{op}_{suffix}")
    if k is None:
        have = sorted(s for s in _DTYPES if f"{op}_{s}" in KERNELS)
        raise TypeError(
            f"hk has no {suffix} {op} kernel -- include/rdna3 implements it for "
            f"{', '.join(have)} only. Cast the tensor first."
        )
    return k


def _apply(op: str, *tensors: Any, out=None):
    a = tensors[0]
    for t in tensors[1:]:
        if t.shape != a.shape or t.dtype != a.dtype:
            raise ValueError(
                f"elementwise {op}: operands differ -- {tuple(a.shape)}/{a.dtype} vs "
                f"{tuple(t.shape)}/{t.dtype}. These kernels do not broadcast."
            )
    if out is None:
        import torch  # noqa: PLC0415  -- see norm._run
        out = torch.empty_like(a)
    elif out.shape != a.shape or out.dtype != a.dtype:
        raise ValueError(
            f"elementwise {op}: out is {tuple(out.shape)}/{out.dtype}, "
            f"expected {tuple(a.shape)}/{a.dtype}"
        )

    suffix = _suffix_of(a)
    k = _kernel_for(op, suffix)
    shape, n = tuple(a.shape), a.numel()
    plan = _strip_plan(op, suffix, n, a.element_size())
    if plan is not None:
        for t in tensors:
            if not t.is_contiguous():
                raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
        if not out.is_contiguous():
            raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
        if n:
            sk, cw, run, exact = plan
            sk(*(t.view(1, n) for t in tensors), out.view(1, n), CW=cw, RUN=run, EXACT=exact)
        return out
    choices = _col_choices(suffix)
    for cols in choices:
        if cols != COLS and ((op, suffix, cols) in _REFUSED or not _fits(shape, n, cols)):
            continue
        views = [_as_2d(t, cols) for t in tensors] + [_as_2d(out, cols)]
        if cols == COLS:
            k(*views)
            return out
        try:
            k(*views, COLS=cols)
            return out
        except ResourceError:
            # The gate picks the width: a wider tile that spills on this
            # compiler (bf16 relu at 256 spilled SGPRs under ROCm 7.2.3 and
            # not 7.1) steps down, once, rather than being chosen by hand.
            _REFUSED.add((op, suffix, cols))
    return out


#: (op, dtype, width) builds the resource gate refused, in this process.
_REFUSED: set = set()

_STRIP_PLANS: dict = {}


def _strip_plan(op: str, suffix: str, n: int, nbytes: int):
    """(kernel, CW, RUN, EXACT) for `op` over n elements on strips, or None
    where the elementwise ops do not run on strips here."""
    key = (op, suffix, n)
    got = _STRIP_PLANS.get(key, False)
    if got is False:
        from ..target import current_arch, get_target  # noqa: PLC0415
        from .norm import strip_run  # noqa: PLC0415

        got = None
        arch = current_arch()
        if arch in STRIP_ARCHS and n * nbytes < _STRIP_MAX_BYTES:
            moved = n * nbytes * (2 if op in _UNARY_FN else 3)
            warps, chunks = next(p for window, p in STRIP_PLANS
                                 if window is None or (window[0] <= moved <= window[1]
                                                       and op not in _UNARY_FN))
            run = strip_run(n, nbytes) if n else 16 // nbytes
            cw = chunks * get_target(arch).wave * run
            got = (STRIP_KERNELS[f"strip_{op}_{suffix}_w{warps}"], cw, run,
                   int(n % (warps * cw) == 0))
        _STRIP_PLANS[key] = got
    return got


def _fits(shape, numel, cols) -> bool:
    try:
        _tile_shape(shape, numel, cols)
        return True
    except ValueError:
        return False


def add(a, b, out=None):
    """Elementwise a + b."""
    return _apply("add", a, b, out=out)


def sub(a, b, out=None):
    """Elementwise a - b."""
    return _apply("sub", a, b, out=out)


def mul(a, b, out=None):
    """Elementwise a * b."""
    return _apply("mul", a, b, out=out)


def maximum(a, b, out=None):
    """Elementwise max(a, b)."""
    return _apply("max", a, b, out=out)


def minimum(a, b, out=None):
    """Elementwise min(a, b)."""
    return _apply("min", a, b, out=out)


def exp(a, out=None):
    """Elementwise exp(a)."""
    return _apply("exp", a, out=out)


def relu(a, out=None):
    """Elementwise max(a, 0)."""
    return _apply("relu", a, out=out)


def gelu(a, out=None):
    """Elementwise GELU, the tanh approximation -- base_ops::gelu is
    `x * (0.5 + 0.5 * fast_tanh(...))`, i.e. torch's approximate='tanh', not
    the erf form. fp32 only."""
    return _apply("gelu", a, out=out)


def neg(a, out=None):
    """Elementwise -a."""
    return _apply("neg", a, out=out)


def abs(a, out=None):  # noqa: A001 -- matches torch.abs, not the builtin
    """Elementwise |a|."""
    return _apply("abs", a, out=out)


__all__ = [
    "KERNELS", "ROWS", "COLS",
    "add", "sub", "mul", "maximum", "minimum",
    "exp", "relu", "gelu", "neg", "abs",
]
