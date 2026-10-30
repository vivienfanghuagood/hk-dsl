"""``@hk2.jit``: specialize, trace, compile, cache and launch a kernel."""
from __future__ import annotations

import ast
import inspect
import os
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import arch as A
from . import ir
from . import language as hl
from . import passes
from . import runtime as rt
from . import types as T
from .compiler import Compiler
from .ir import PtrType, ScalarType

__all__ = ["jit", "func", "Kernel"]


# ================================================================ the AST rewrite
def _stores(stmts) -> List[str]:
    """Names a statement list assigns, in order, outside nested scopes."""
    out: List[str] = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("__hk_"):
                out.append(node.name)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.append(node.id)
        for c in ast.iter_child_nodes(node):
            visit(c)
    for s in stmts:
        visit(s)
    seen, res = set(), []
    for n in out:
        if n not in seen and not n.startswith("__hk_"):
            seen.add(n)
            res.append(n)
    return res


def _jumps(stmts, loop_level=True) -> bool:
    """A return anywhere, or a break / continue belonging to this loop."""
    def visit(node, in_loop):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return False
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, (ast.Break, ast.Continue)) and not in_loop:
            return True
        nested = in_loop or isinstance(node, (ast.For, ast.While))
        return any(visit(c, nested) for c in ast.iter_child_nodes(node))
    return any(visit(s, False) for s in stmts)


class _Rewriter(ast.NodeTransformer):
    def __init__(self):
        self.n = 0

    def _name(self, s):
        self.n += 1
        return f"__hk_{s}_{self.n}"

    @staticmethod
    def _carry_assign(names, value):
        return ast.Assign(targets=[ast.Tuple([ast.Name(n, ast.Store()) for n in names], ast.Store())], value=value)

    @staticmethod
    def _tuple_load(names):
        return ast.Tuple([ast.Name(n, ast.Load()) for n in names], ast.Load())

    @staticmethod
    def _gets(names):
        locs = ast.Call(ast.Name("locals", ast.Load()), [], [])
        return ast.Tuple([ast.Call(ast.Attribute(ast.Name("__hk_rt", ast.Load()), "get", ast.Load()),
                                   [locs, ast.Constant(n)], []) for n in names], ast.Load())

    def _fn(self, name, args, names, body):
        stmts = []
        if names:
            stmts.append(self._carry_assign(names, ast.Name("__hk_c", ast.Load())))
        stmts += body
        stmts.append(ast.Return(self._tuple_load(names)))
        fd = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[ast.arg(a) for a in args],
                                                           kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=stmts, decorator_list=[], returns=None)
        if sys.version_info >= (3, 12):
            fd.type_params = []
        return fd

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if node.orelse or _jumps(node.body):
            return node
        names = [n for n in _stores(node.body) if n not in _stores([node.target])]
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            it = ast.Call(ast.Attribute(ast.Name("__hk_rt", ast.Load()), "range_", ast.Load()), it.args, it.keywords)
        fname = self._name("for")
        if isinstance(node.target, ast.Name):
            args, pre = [node.target.id, "__hk_c"], []
        else:
            args = ["__hk_t", "__hk_c"]
            pre = [ast.Assign(targets=[node.target], value=ast.Name("__hk_t", ast.Load()))]
        fdef = self._fn(fname, args, names, pre + node.body)
        call = ast.Call(ast.Attribute(ast.Name("__hk_rt", ast.Load()), "for_loop", ast.Load()),
                        [it, ast.Name(fname, ast.Load()), self._gets(names),
                         ast.Tuple([ast.Constant(n) for n in names], ast.Load())], [])
        out = [fdef, self._carry_assign(names, call) if names else ast.Expr(call)]
        return [ast.copy_location(s, node) for s in out]

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if _jumps(node.body) or _jumps(node.orelse):
            return node
        names = _stores(node.body + node.orelse)
        tn, en = self._name("then"), self._name("else")
        out = [self._fn(tn, ["__hk_c"], names, node.body or [ast.Pass()]),
               self._fn(en, ["__hk_c"], names, node.orelse or [ast.Pass()])]
        call = ast.Call(ast.Attribute(ast.Name("__hk_rt", ast.Load()), "if_", ast.Load()),
                        [node.test, ast.Name(tn, ast.Load()), ast.Name(en, ast.Load()), self._gets(names),
                         ast.Tuple([ast.Constant(n) for n in names], ast.Load())], [])
        out.append(self._carry_assign(names, call) if names else ast.Expr(call))
        return [ast.copy_location(s, node) for s in out]


def rewrite(fn: Callable) -> Callable:
    """The kernel function with device control flow: ``for ... in range`` and ``if`` on runtime values."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fdef = tree.body[0]
    fdef.decorator_list = []
    _Rewriter().visit(tree)
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, fn.__code__.co_firstlineno - 1)
    code = compile(tree, filename=fn.__code__.co_filename, mode="exec")
    g = dict(fn.__globals__)
    g["__hk_rt"] = hl
    if fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            g[name] = cell.cell_contents
    exec(code, g)
    return g[fdef.name]


class func:
    """``@hk2.func``: a helper a kernel calls, with the same device control flow as a kernel."""

    def __init__(self, fn):
        self.fn = fn
        self._rw = None

    def __call__(self, *args, **kwargs):
        if self._rw is None:
            self._rw = rewrite(self.fn)
        hl._State.files.add(self.fn.__code__.co_filename)
        return self._rw(*args, **kwargs)


# ================================================================ kernels
@dataclass
class Compiled:
    code: rt.CodeObject
    ir: str
    asm: str
    stats: Dict[str, int]
    lds: int
    threads: int
    params: List[Tuple[str, str]]   # (kind, source) for packing: kind p/i/f, source = (arg index, field)


def _is_constexpr(p: inspect.Parameter) -> bool:
    return p.annotation is hl.constexpr or p.annotation == "hk.constexpr" or p.kind == p.KEYWORD_ONLY or (
        isinstance(p.annotation, str) and p.annotation.endswith("constexpr"))


def _tensor_key(t) -> Tuple:
    from .types import from_torch
    dt = from_torch(t.dtype)
    if not 1 <= t.dim() <= 4:
        raise ValueError(f"hk2 kernels take tensors of 1 to 4 dimensions, got {t.dim()}-D")
    contig = t.stride(-1) == 1
    vec = max(1, 16 // dt.bytes)
    aligned = contig and t.data_ptr() % 16 == 0 and t.shape[-1] % vec == 0 and all(
        (s * dt.bytes) % 16 == 0 for s in t.stride()[:-1])
    if t.numel() and t.storage_offset() * dt.bytes + t.untyped_storage().nbytes() >= 2 ** 31:
        pass  # buffer offsets are 32-bit: checked per launch below
    return ("T", dt.name, t.dim(), contig, aligned)


_STAT_RE = {k: re.compile(p) for k, p in {
    "vgprs": r";\s*NumVgprs:\s*(\d+)", "agprs": r";\s*NumAgprs:\s*(\d+)", "sgprs": r";\s*NumSgprs:\s*(\d+)",
    "scratch": r";\s*ScratchSize:\s*(\d+)", "occupancy": r";\s*Occupancy:\s*(\d+)",
    "lds": r";\s*LDSByteSize:\s*(\d+)"}.items()}


def asm_stats(asm: str) -> Dict[str, int]:
    out = {}
    for k, r in _STAT_RE.items():
        m = r.search(asm)
        if m:
            out[k] = int(m.group(1))
    return out


class Kernel:
    def __init__(self, fn: Callable, num_warps: Optional[int] = None):
        self.fn = fn
        self.name = fn.__name__
        self.sig = inspect.signature(fn)
        self.num_warps = num_warps
        self.cache: Dict[Tuple, Compiled] = {}
        self._rw: Optional[Callable] = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.run(grid, *args, **kwargs)
        return launch

    def __call__(self, *args, **kwargs):
        raise TypeError(f"launch {self.name} with a grid: {self.name}[grid](...)")

    def _bind(self, args, kwargs):
        opts = {k: kwargs.pop(k) for k in ("num_warps", "arch", "interpret") if k in kwargs}
        ba = self.sig.bind(*args, **kwargs)
        ba.apply_defaults()
        return ba, opts

    def specialize(self, ba, arch: str, num_warps: int) -> Tuple:
        import torch
        key = [arch, num_warps]
        for name, p in self.sig.parameters.items():
            v = ba.arguments[name]
            if _is_constexpr(p):
                key.append(("C", v if not isinstance(v, T.DType) else v.name))
            elif isinstance(v, torch.Tensor):
                key.append(_tensor_key(v))
            elif isinstance(v, bool):
                key.append(("i", 1))
            elif isinstance(v, int):
                key.append(("i", 16 if v % 16 == 0 else 1))
            elif isinstance(v, float):
                key.append(("f",))
            else:
                raise TypeError(f"{self.name}: argument {name} is a {type(v).__name__}; runtime arguments are "
                                f"tensors, ints and floats (mark compile-time values hk2.constexpr)")
        return tuple(key)

    def compile(self, key, ba, arch_name: str, num_warps: int) -> Compiled:
        arch = A.get_arch(arch_name)
        top, params, packing, param_div = self.trace(key, ba)
        if os.environ.get("HK2_PIPELINE", "1") != "0":
            passes.pipeline_loads(top)
        c = Compiler(arch, num_warps, self.name, rt.llvm_major())
        text = c.compile(top, params, param_div)
        if os.environ.get("HK2_DUMP"):
            print(ir.dump(top))
            print(text)
        co = rt.compile_ir(text, arch.name)
        stats = asm_stats(co.asm)
        stats["barriers"] = c.n_barriers
        if stats.get("scratch", 0) and not os.environ.get("HK2_ALLOW_SPILL"):
            warnings.warn(f"hk2: {self.name} spills {stats['scratch']} bytes per lane to scratch on {arch.name} "
                          f"(vgprs {stats.get('vgprs')}); smaller tiles or fewer warps' work per thread avoid it",
                          stacklevel=4)
        return Compiled(rt.CodeObject(co.hsaco, self.name), text, co.asm, stats, c.lds_size,
                        num_warps * arch.wave, packing)

    def trace(self, key, ba):
        """The kernel as tile IR for one specialization: (top block, params, packing, divisibility)."""
        if self._rw is None:
            self._rw = rewrite(self.fn)
        b = ir.Builder()
        hl._State.builder = b
        hl._State.files = {self.fn.__code__.co_filename}
        params: List[Tuple[ir.Value, str, str]] = []
        packing: List[Tuple[str, Tuple]] = []
        param_div: Dict[int, int] = {}
        call_args = {}
        try:
            for i, (name, p) in enumerate(self.sig.parameters.items()):
                v = ba.arguments[name]
                k = key[2 + i]
                if k[0] == "C":
                    call_args[name] = v
                    continue
                if k[0] == "T":
                    _, dtn, nd, contig, aligned = k
                    dt = getattr(T, dtn)
                    ptr = ir.Value(PtrType(dt))
                    params.append((ptr, "ptr addrspace(1)", f"{name}"))
                    packing.append(("p", (name, "ptr")))
                    shape = []
                    for d in range(nd):
                        sv = ir.Value(ScalarType(T.i32))
                        params.append((sv, "i32", f"{name}.s{d}"))
                        packing.append(("i", (name, "shape", d)))
                        shape.append(hl.Scalar(sv))
                    strides = []
                    for d in range(nd):
                        if d == nd - 1 and contig:
                            strides.append(1)
                            continue
                        sv = ir.Value(ScalarType(T.i32))
                        params.append((sv, "i32", f"{name}.st{d}"))
                        packing.append(("i", (name, "stride", d)))
                        if aligned:
                            param_div[sv.id] = 16 // dt.bytes
                        strides.append(hl.Scalar(sv))
                    call_args[name] = hl.TensorArg(name, ptr, dt, shape, strides,
                                                   (16 // dt.bytes) if aligned else 1)
                    if aligned:
                        param_div[shape[-1].v.id] = 16 // dt.bytes
                elif k[0] == "i":
                    sv = ir.Value(ScalarType(T.i32))
                    params.append((sv, "i32", name))
                    packing.append(("i", (name,)))
                    param_div[sv.id] = k[1]
                    call_args[name] = hl.Scalar(sv)
                else:
                    sv = ir.Value(ScalarType(T.f32))
                    params.append((sv, "float", name))
                    packing.append(("f", (name,)))
                    call_args[name] = hl.Scalar(sv)
            b.top.params = [p[0] for p in params]
            ret = self._rw(**call_args)
            if ret is not None:
                raise hl.HKError(f"{self.name}: kernels return nothing; write results with hk2.store")
        finally:
            hl._State.builder = None
        return b.top, params, packing, param_div

    def interpret(self, grid, ba):
        """Run on the CPU through hk2.interpret (no GPU compiler involved); results land in the tensors."""
        from . import interpret as it
        key = self.specialize(ba, "interpret", 1)
        top, params, packing, _ = self.trace(key, ba)
        if callable(grid):
            grid = grid(dict(ba.arguments))
        grid = tuple(int(g) for g in (grid if isinstance(grid, (tuple, list)) else (grid,)))
        it.run(top, params, packing, dict(ba.arguments), grid + (1,) * (3 - len(grid)))

    def warmup(self, *args, **kwargs) -> Compiled:
        ba, opts = self._bind(list(args), dict(kwargs))
        arch = opts.get("arch") or rt.current_arch()
        nw = opts.get("num_warps") or self.num_warps or 4
        key = self.specialize(ba, arch, nw)
        if key not in self.cache:
            self.cache[key] = self.compile(key, ba, arch, nw)
        return self.cache[key]

    def run(self, grid, *args, **kwargs):
        import torch
        ba, opts = self._bind(list(args), dict(kwargs))
        if opts.get("interpret", os.environ.get("HK2_INTERPRET", "0") not in ("", "0")):
            return self.interpret(grid, ba)
        arch = opts.get("arch") or rt.current_arch()
        nw = opts.get("num_warps") or self.num_warps or 4
        key = self.specialize(ba, arch, nw)
        k = self.cache.get(key)
        if k is None:
            k = self.cache[key] = self.compile(key, ba, arch, nw)
        if callable(grid):
            grid = grid(dict(ba.arguments))
        grid = tuple(int(g) for g in (grid if isinstance(grid, (tuple, list)) else (grid,)))
        grid = grid + (1,) * (3 - len(grid))
        if min(grid) <= 0:
            return k
        vals = []
        dev = None
        for kind, src in k.params:
            v = ba.arguments[src[0]]
            if kind == "p":
                if not v.is_cuda:
                    raise ValueError(f"{self.name}: {src[0]} is not on the GPU")
                dev = v.device.index
                vals.append(("p", v.data_ptr()))
            elif len(src) == 3:
                vals.append(("i", v.shape[src[2]] if src[1] == "shape" else v.stride(src[2])))
            else:
                vals.append((kind, int(v) if kind == "i" else float(v)))
        dev = torch.cuda.current_device() if dev is None else dev
        stream = torch.cuda.current_stream(dev).cuda_stream
        k.code.launch(grid, k.threads, rt.pack_args(vals), 0, stream, dev)
        return k


def jit(fn=None, *, num_warps: Optional[int] = None):
    """``@hk2.jit``: a GPU kernel. Launch it as ``kernel[grid](args...)``."""
    if fn is None:
        return lambda f: Kernel(f, num_warps)
    return Kernel(fn, num_warps)
