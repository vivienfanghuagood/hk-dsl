"""Compile LLVM IR to a code object with the ROCm installation's own LLVM, load it, launch it.

hk2 has no native build step. The toolchain is whatever ROCm the process runs on: ``clang`` turns the
IR hk2 emits into an object, ``ld.lld`` links the code object, and the HIP runtime torch already loaded
runs it on torch's current stream.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["rocm_llvm", "compile_ir", "CodeObject", "current_arch", "cache_dir"]


class CompileError(RuntimeError):
    pass


@functools.lru_cache(None)
def rocm_llvm() -> Path:
    """The bin directory of ROCm's LLVM: $HK2_LLVM, else $ROCM_PATH/llvm/bin, else /opt/rocm/llvm/bin."""
    for cand in (os.environ.get("HK2_LLVM"),
                 os.path.join(os.environ.get("ROCM_PATH", ""), "llvm", "bin") if os.environ.get("ROCM_PATH") else None,
                 "/opt/rocm/llvm/bin"):
        if cand and (Path(cand) / "clang").exists():
            return Path(cand)
    clang = shutil.which("amdclang") or shutil.which("clang")
    if clang:
        return Path(clang).resolve().parent
    raise CompileError("hk2 needs ROCm's LLVM (clang, ld.lld): set ROCM_PATH or HK2_LLVM")


@functools.lru_cache(None)
def llvm_version() -> str:
    out = subprocess.run([str(rocm_llvm() / "clang"), "--version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


@functools.lru_cache(None)
def llvm_major() -> int:
    import re
    m = re.search(r"version (\d+)", llvm_version())
    return int(m.group(1)) if m else 0


def cache_dir() -> Path:
    d = Path(os.environ.get("HK2_CACHE_DIR") or Path.home() / ".cache" / "hk2")
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_arch() -> str:
    import torch
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


@dataclass
class Compiled:
    hsaco: bytes
    asm: str
    path: Path


def compile_ir(ir: str, arch: str, *, opt: str = "-O3", extra: Sequence[str] = ()) -> Compiled:
    """IR text -> code object (cached by IR, arch, flags and LLVM version)."""
    key = hashlib.sha256("\0".join([ir, arch, opt, *extra, llvm_version()]).encode()).hexdigest()[:24]
    d = cache_dir() / key
    if (d / "k.hsaco").exists() and (d / "k.s").exists():
        return Compiled((d / "k.hsaco").read_bytes(), (d / "k.s").read_text(), d)
    tmp = Path(tempfile.mkdtemp(prefix="hk2-", dir=cache_dir()))
    (tmp / "k.ll").write_text(ir)
    bin_ = rocm_llvm()
    common = ["-x", "ir", "--target=amdgcn-amd-amdhsa", f"-mcpu={arch}", opt, *extra]
    for cmd in ([str(bin_ / "clang"), *common, "-c", "k.ll", "-o", "k.o"],
                [str(bin_ / "clang"), *common, "-S", "k.ll", "-o", "k.s"],
                [str(bin_ / "ld.lld"), "-shared", "k.o", "-o", "k.hsaco"]):
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if r.returncode:
            raise CompileError(f"{Path(cmd[0]).name} failed ({tmp / 'k.ll'}):\n{r.stderr[-4000:]}")
    try:
        tmp.rename(d)
    except OSError:  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return Compiled((d / "k.hsaco").read_bytes(), (d / "k.s").read_text(), d)


# ---------------------------------------------------------------- HIP, through the runtime torch loaded
@functools.lru_cache(None)
def hip():
    import torch  # noqa: F401  (loads libamdhip64 into the process)
    lib = ctypes.CDLL("libamdhip64.so")
    vp = ctypes.c_void_p
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = [vp] + [ctypes.c_uint] * 7 + [vp, ctypes.POINTER(vp), ctypes.POINTER(vp)]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    return lib


def _check(err: int, what: str) -> None:
    if err:
        raise RuntimeError(f"{what}: {hip().hipGetErrorString(err).decode()} ({err})")


_LAUNCH_BUF, _LAUNCH_SIZE, _LAUNCH_END = 1, 2, 3  # HIP_LAUNCH_PARAM_* (hip_runtime_api.h)


class CodeObject:
    """A loaded kernel, one module per device."""

    def __init__(self, hsaco: bytes, symbol: str):
        self.hsaco, self.symbol = hsaco, symbol
        self._fn: Dict[int, ctypes.c_void_p] = {}
        self._keep: List[object] = []

    def function(self, device: int) -> ctypes.c_void_p:
        fn = self._fn.get(device)
        if fn is None:
            mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
            _check(hip().hipModuleLoadData(ctypes.byref(mod), self.hsaco), "hipModuleLoadData")
            _check(hip().hipModuleGetFunction(ctypes.byref(fn), mod, self.symbol.encode()), "hipModuleGetFunction")
            self._fn[device] = fn
        return fn

    def launch(self, grid: Tuple[int, int, int], block: int, args: bytes, lds: int, stream: int, device: int):
        buf = ctypes.create_string_buffer(args, max(len(args), 8))
        size = ctypes.c_size_t(len(args))
        extra = (ctypes.c_void_p * 5)(_LAUNCH_BUF, ctypes.cast(buf, ctypes.c_void_p).value,
                                      _LAUNCH_SIZE, ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value, _LAUNCH_END)
        _check(hip().hipModuleLaunchKernel(self.function(device), grid[0], grid[1], grid[2], block, 1, 1,
                                           lds, ctypes.c_void_p(stream), None, extra), self.symbol)


def pack_args(values: Sequence[Tuple[str, object]]) -> bytes:
    """Kernarg bytes: ('p', ptr) 8-aligned, ('i', i32) / ('f', f32) 4-aligned, in order."""
    out = bytearray()
    for kind, v in values:
        size, fmt = {"p": (8, "<Q"), "i": (4, "<i"), "f": (4, "<f"), "l": (8, "<q")}[kind]
        out += b"\0" * (-len(out) % size)
        out += struct.pack(fmt, v)
    return bytes(out)
