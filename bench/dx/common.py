"""What the DX suite's tasks, solutions, runner and test share.

A *task* is a package `bench/dx/tasks/<id>/` with `task.md` (the problem as a
user or an agent receives it) and `reference.py` (the golden reference, the
input generator, the shapes and the correctness check). A *solution* is a file
`bench/dx/solutions/<id>/<name>.py` with `solve(**inputs)`; bench/dx/README.md
is the contract. This module finds both, loads them, checks outputs and counts
a solution's lines.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import io
import os
import subprocess
import sys
import tokenize
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent            # bench/dx
REPO = ROOT.parents[1]
TASKS_DIR = ROOT / "tasks"
SOLUTIONS_DIR = ROOT / "solutions"

# Kinds a solution declares (KIND = ...). Baselines are what a solution is
# compared against; "proxy" is timed for scale but computes different math
# (it is never checked, never a "best baseline", and the tables mark it).
BASELINE_KINDS = ("baseline", "hk_ops")
KINDS = ("baseline", "hk_ops", "proxy", "docs_only", "template", "agent", "expert")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_PY = REPO / "python"
if _PY.is_dir() and str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))


# --------------------------------------------------------------------------- tasks

def task_ids() -> List[str]:
    return sorted(p.name for p in TASKS_DIR.iterdir()
                  if (p / "reference.py").is_file())


def resolve_tasks(spec: Optional[str]) -> List[str]:
    """'t1,t3' or 't1_gelu_tanh_and_mul' or None (all) -> task ids."""
    ids = task_ids()
    if not spec:
        return ids
    out = []
    for want in spec.split(","):
        want = want.strip()
        hit = [t for t in ids if t == want or t.split("_", 1)[0] == want]
        if not hit:
            raise SystemExit(f"no task {want!r}; tasks are {', '.join(ids)}")
        out += [t for t in hit if t not in out]
    return out


def load_task(task_id: str) -> ModuleType:
    return importlib.import_module(f"bench.dx.tasks.{task_id}.reference")


# --------------------------------------------------------------------------- solutions

class Solution:
    """One contender: a file in solutions/<task>/, or a built-in torch baseline."""

    def __init__(self, task_id: str, name: str, module: Optional[ModuleType] = None,
                 fn: Optional[Callable] = None, kind: str = "baseline",
                 label: str = "", lines: Optional[int] = None,
                 setup: Optional[Callable] = None, supports: Optional[Callable] = None,
                 proxy: str = "", path: Optional[Path] = None):
        self.task_id, self.name, self.module = task_id, name, module
        self.kind, self.label, self.proxy, self.path = kind, label, proxy, path
        self._solve, self._setup, self._supports = fn, setup, supports
        self.lines = lines

    @classmethod
    def from_file(cls, task_id: str, path: Path) -> "Solution":
        modname = f"bench.dx.solutions.{task_id}.{path.stem}"
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        kind = getattr(mod, "KIND", "agent")
        if kind not in KINDS:
            raise ValueError(f"{path}: KIND={kind!r}, expected one of {KINDS}")
        if not hasattr(mod, "solve") and not hasattr(mod, "setup"):
            raise ValueError(f"{path}: defines neither solve() nor setup()")
        return cls(task_id, path.stem, mod, getattr(mod, "solve", None), kind,
                   getattr(mod, "LABEL", ""), count_lines(path),
                   getattr(mod, "setup", None), getattr(mod, "supports", None),
                   getattr(mod, "PROXY", ""), path)

    def supports(self, inputs: Dict[str, Any]):
        """True, or a string saying why this solution does not take these inputs."""
        if self._supports is None:
            return True
        return self._supports(**inputs)

    def prepare(self, inputs: Dict[str, Any]) -> Callable[[], Any]:
        """The zero-argument callable that is checked and timed. setup() runs
        here, outside the timing; with no setup() it is solve(**inputs)."""
        if self._setup is not None:
            return self._setup(**inputs)
        solve = self._solve
        return lambda: solve(**inputs)


def builtin_solutions(task_id: str, task: ModuleType) -> List[Solution]:
    """torch eager (the task's `torch_impl`) and torch.compile of it -- or the
    task's `compile_setup(**inputs) -> callable`, when torch.compile's natural
    code differs from eager's (flex_attention for a modified attention)."""
    import torch

    eager = task.torch_impl
    sols = [Solution(task_id, "torch_eager", fn=eager, kind="baseline",
                     label="torch eager, the natural PyTorch code",
                     lines=count_fn_lines(eager))]
    custom = getattr(task, "compile_setup", None)
    if custom is not None:
        sols.append(Solution(task_id, "torch_compile", fn=None, kind="baseline",
                             label=(custom.__doc__ or "torch.compile").strip().splitlines()[0],
                             lines=count_fn_lines(custom), setup=custom))
        return sols

    def compile_setup(**inputs):
        # One wrapper per shape; dynamic=False, default mode, as the goal says.
        f = torch.compile(eager, dynamic=False)
        return lambda: f(**inputs)

    sols.append(Solution(task_id, "torch_compile", fn=None, kind="baseline",
                         label="torch.compile(torch_eager, default mode, dynamic=False)",
                         lines=count_fn_lines(eager), setup=compile_setup))
    return sols


def solutions_for(task_id: str, task: ModuleType, names: Optional[List[str]] = None,
                  kinds: Optional[List[str]] = None) -> List[Solution]:
    sols = builtin_solutions(task_id, task)
    d = SOLUTIONS_DIR / task_id
    if d.is_dir():
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            sols.append(Solution.from_file(task_id, p))
    if names:
        sols = [s for s in sols if s.name in names]
    if kinds:
        sols = [s for s in sols if s.kind in kinds]
    return sols


def configure_torch_compile() -> None:
    """dynamic=False recompiles per shape; the default limit (8) would make a
    long shape list fall back to eager silently."""
    import torch._dynamo as dynamo

    for knob in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                 "accumulated_cache_size_limit"):
        if hasattr(dynamo.config, knob):
            setattr(dynamo.config, knob, 1024)


# --------------------------------------------------------------------------- checking

class Mismatch(AssertionError):
    pass


def as_tuple(out) -> tuple:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def assert_close(name: str, got, want, *, rtol: float, atol: float, dtype=None,
                 shape=None) -> str:
    """|got - want| <= atol + rtol * |want| elementwise (torch.isclose), with
    the output's dtype and shape checked. Returns a one-line summary."""
    import torch

    if not isinstance(got, torch.Tensor):
        raise Mismatch(f"{name}: expected a tensor, got {type(got).__name__}")
    if dtype is not None and got.dtype != dtype:
        raise Mismatch(f"{name}: dtype {got.dtype}, expected {dtype}")
    want_shape = tuple(shape) if shape is not None else tuple(want.shape)
    if tuple(got.shape) != want_shape:
        raise Mismatch(f"{name}: shape {tuple(got.shape)}, expected {want_shape}")
    g, w = got.double(), want.double()
    if not torch.isfinite(g).all():
        raise Mismatch(f"{name}: {int((~torch.isfinite(g)).sum())} non-finite values")
    err = (g - w).abs()
    bad = err > atol + rtol * w.abs()
    scale = max(w.abs().max().item(), 1e-30)
    summary = f"{name}: max abs err {err.max().item():.2e}, rel to max|ref| {err.max().item() / scale:.2e}"
    if bad.any():
        i = int(bad.flatten().nonzero()[0])
        raise Mismatch(f"{summary}; {int(bad.sum())}/{bad.numel()} elements outside "
                       f"atol={atol:g} rtol={rtol:g}, first at flat index {i}: got "
                       f"{g.flatten()[i].item():.6g}, want {w.flatten()[i].item():.6g}")
    return summary


# --------------------------------------------------------------------------- lines

def _code_lines(src: str) -> int:
    """Lines that hold code: not blank, not only a comment, not a docstring."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return len([ln for ln in src.splitlines() if ln.strip()])
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant) and isinstance(
                    body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    code = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in doc_lines:
                code.add(ln)
    return len(code)


def count_lines(path: Path) -> int:
    return _code_lines(Path(path).read_text())


def count_fn_lines(fn) -> Optional[int]:
    try:
        import textwrap
        return _code_lines(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError):
        return None


# --------------------------------------------------------------------------- environment

def versions() -> Dict[str, Any]:
    """Everything a result file must say about where it was measured."""
    import platform

    import torch

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "host": platform.node(),
    }
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        info["triton"] = f"not importable: {type(e).__name__}"
    try:
        import hk
        info["hk"] = hk.__version__
        from hk.target import current_arch
        info["hk_arch"] = current_arch()
    except Exception as e:  # noqa: BLE001
        info["hk"] = f"not importable: {type(e).__name__}: {e}"
    commit = os.environ.get("HK_DX_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "python", "bench"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            if dirty:
                commit += "+dirty"
        except Exception:  # noqa: BLE001
            commit = None
    info["hk_commit"] = commit or "unknown"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_arch"] = getattr(p, "gcnArchName", "?")
        # ROCm names both "AMD Radeon Graphics"; these are the parts the suite runs on.
        info["gpu_product"] = {"gfx950": "Instinct MI355X", "gfx1100": "Radeon PRO W7900D"}.get(
            info["gpu_arch"].split(":")[0], p.name)
        info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
        info["cu"] = p.multi_processor_count
    info["visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "ROCR_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def arch_short() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "gpu").split(":")[0]
