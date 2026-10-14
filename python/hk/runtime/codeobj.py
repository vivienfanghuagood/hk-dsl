"""A kernel as a device-only code object, launched from Python.

The pybind module and the torch registration both compile a host half --
pybind's headers, or libtorch's -- and that is most of a build: 4.4 s for the
pybind module against 0.6-0.9 for the code object on the workstation, 17 s for
the torch op on the pod. A code object (`codegen.scaffold.device`,
`compile.build(kind="co")`) has none of it. What the host half did, this does
in Python:

* **The kernel argument.** `globals` is passed by value; the code object
  carries a table of its layout (`scaffold.LAYOUT_SYMBOL`), read here straight
  out of the ELF -- no GPU needed -- and every launch packs the tensors' base
  pointers, extents and row strides at those offsets.
* **The grid.** The traced grid expressions (`TracedKernel.grid`) evaluated
  over the tensors' extents, as the C++ `grid()` evaluates them.
* **The launch.** `hipModuleLaunchKernel`, through the HIP runtime torch has
  already loaded, on the stream the caller names -- torch's current one for a
  torch op, so it is ordered against torch's work and captured into a CUDA
  graph like any other launch.

    op = hk.runtime.codeobj.torch_op(kernel)     # torch.ops.hk.<name>
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from . import compile as _compile
from .compile import CompileError

# ---------------------------------------------------------------- the ELF


def elf_symbol(data: bytes, name: str) -> bytes:
    """The initialised bytes of symbol `name` in an ELF64 little-endian file.
    Enough of ELF for one lookup, so reading a layout needs no dependency."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise CompileError("not an ELF64 little-endian code object")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
                for i in range(shnum)]
    for sh in sections:
        if sh[1] != 2:                          # SHT_SYMTAB
            continue
        strtab = sections[sh[6]]
        _, _, _, _, symoff, symsize, _, _, _, entsize = sh
        for j in range(symsize // entsize):
            st_name, _, _, shndx, value, size = struct.unpack_from(
                "<IBBHQQ", data, symoff + j * entsize)
            end = data.index(b"\0", strtab[4] + st_name)
            if data[strtab[4] + st_name:end].decode() != name:
                continue
            sec = sections[shndx]
            at = sec[4] + (value - sec[3])      # sh_offset + (value - sh_addr)
            return data[at:at + size]
    raise CompileError(f"no symbol {name!r} in the code object")


# ---------------------------------------------------------------- HIP

_HIP: List[Any] = []


def hip():
    """The HIP runtime this process already runs on.

    Found through /proc/self/maps rather than by name: a torch wheel ships its
    own libamdhip64, and a second copy loaded beside it would be a second
    runtime, whose streams and memory are not torch's."""
    if _HIP:
        return _HIP[0]
    path = None
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            if "libamdhip64.so" in line:
                path = line.split()[-1]
                break
    except OSError:
        pass
    lib = ctypes.CDLL(path or "libamdhip64.so")
    vp = ctypes.c_void_p
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = (
        [vp] + [ctypes.c_uint] * 7 + [vp, ctypes.POINTER(vp), ctypes.POINTER(vp)])
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _HIP.append(lib)
    return lib


def _check(err: int, what: str) -> None:
    if err:
        raise RuntimeError(f"{what}: {hip().hipGetErrorString(err).decode()} "
                           f"({err})")


#: hipModuleLaunchKernel's `extra` keys (hip_runtime_api.h).
_PARAM_BUFFER, _PARAM_SIZE, _PARAM_END = 1, 2, 3

# ---------------------------------------------------------------- a kernel


class CodeObjectKernel:
    """One traced kernel, built as a code object, ready to launch.

    Loaded per device on first use: a module belongs to the device that was
    current when it was loaded."""

    def __init__(self, ir, build):
        from ..codegen import scaffold as sc  # noqa: PLC0415

        self.ir = ir
        self.build = build
        self.image = Path(build.so_path).read_bytes()
        self.symbol = sc.kernel_symbol(ir)
        self.tensors = [p.name for p in ir.tensors()]
        self.dtypes = [p.type.dtype.name for p in ir.tensors()]
        self.block = ir.threads
        self.shmem = ir.lds_bytes
        self.grid = ir.grid
        raw = elf_symbol(self.image, sc.LAYOUT_SYMBOL)
        n = len(raw) // 8
        table = struct.unpack(f"<{n}q", raw)
        per = 1 + len(sc.GL_FIELDS)
        if n != 1 + per * len(self.tensors):
            raise CompileError(f"{ir.name}: layout table has {n} entries for "
                               f"{len(self.tensors)} tensors")
        self.size = table[0]
        #: Per tensor: (pointer offset, (batch, depth, rows, cols) offsets,
        #: row stride offset), all from the start of the argument.
        self.offsets = []
        for i in range(len(self.tensors)):
            base, ptr, b, d, r, c, rs = table[1 + per * i:1 + per * (i + 1)]
            self.offsets.append((base + ptr, (base + b, base + d, base + r,
                                              base + c), base + rs))
        self._fn: Dict[int, ctypes.c_void_p] = {}
        self._modules: Dict[int, ctypes.c_void_p] = {}

    def function(self, device: int) -> ctypes.c_void_p:
        fn = self._fn.get(device)
        if fn is None:
            mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
            _check(hip().hipModuleLoadData(ctypes.byref(mod), self.image),
                   f"{self.ir.name}: hipModuleLoadData")
            _check(hip().hipModuleGetFunction(ctypes.byref(fn), mod,
                                              self.symbol.encode()),
                   f"{self.ir.name}: hipModuleGetFunction({self.symbol})")
            self._modules[device], self._fn[device] = mod, fn
        return fn

    def pack(self, shapes: Sequence[Tuple[int, int, int, int]],
             ptrs: Sequence[int], row_strides: Sequence[int]) -> bytes:
        buf = bytearray(self.size)
        for (po, dims, ro), s, p, rs in zip(self.offsets, shapes, ptrs,
                                            row_strides):
            struct.pack_into("<Q", buf, po, p)
            for o, v in zip(dims, s):
                struct.pack_into("<i", buf, o, v)
            struct.pack_into("<i", buf, ro, rs)
        return bytes(buf)

    def grid_for(self, shapes) -> Tuple[int, int, int]:
        ext = dict(zip(self.tensors, shapes))
        return tuple(g if isinstance(g, int) else g.value(ext)
                     for g in self.grid)

    def launch(self, device: int, stream: int, shapes, ptrs, row_strides):
        """Launch on `stream` (a hipStream_t as an int; 0 is the null stream)."""
        gx, gy, gz = self.grid_for(shapes)
        if gx <= 0 or gy <= 0 or gz <= 0:
            return                      # an empty grid launches nothing
        arg = ctypes.create_string_buffer(self.pack(shapes, ptrs, row_strides),
                                          self.size)
        size = ctypes.c_size_t(self.size)
        extra = (ctypes.c_void_p * 5)(
            _PARAM_BUFFER, ctypes.addressof(arg), _PARAM_SIZE,
            ctypes.addressof(size), _PARAM_END)
        _check(hip().hipModuleLaunchKernel(
            self.function(device), gx, gy, gz, self.block, 1, 1, self.shmem,
            ctypes.c_void_p(stream), None, extra),
            f"{self.ir.name}: hipModuleLaunchKernel")


def load(kernel, **const_overrides) -> CodeObjectKernel:
    """Trace, build as a code object (cached, gated), and wrap."""
    from ..codegen import scaffold as sc  # noqa: PLC0415

    ir = kernel.trace(**const_overrides)
    src = sc.device(ir)
    with_defines = getattr(kernel, "with_defines", None)
    if with_defines:
        src = with_defines(src)
    b = _compile.build(src, ir.arch, name=ir.name, kind="co",
                       max_vgprs=getattr(kernel, "max_vgprs", None),
                       min_occupancy=getattr(kernel, "min_occupancy", None))
    return CodeObjectKernel(ir, b)


# ---------------------------------------------------------------- torch

_TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
                 "i32": "int32", "i8": "int8", "fp8e4m3": "float8_e4m3fn",
                 "fp8e5m2": "float8_e5m2"}


def tensor_args(k: CodeObjectKernel, tensors, op: str):
    """(shapes, pointers, row strides) for `tensors`, checked the way the C++
    torch op checks them (scaffold.torch_library's hk_check and hk_gl): on
    the GPU, of the kernel's dtype, 1-4D, contiguous or a slice of the last
    axis, left-padded to (batch, depth, rows, cols). Refused with
    RuntimeError, which is what TORCH_CHECK raises there, so a caller cannot
    tell which registration it got."""
    import torch  # noqa: PLC0415

    shapes, ptrs, rows = [], [], []
    for name, want, t in zip(k.tensors, k.dtypes, tensors):
        if not t.is_cuda:
            raise RuntimeError(f"{op}: {name} must be on the GPU")
        if t.dtype != getattr(torch, _TORCH_DTYPES[want]):
            raise RuntimeError(f"{op}: {name} must be {want}, got {t.dtype}")
        nd = t.dim()
        if not 1 <= nd <= 4:
            raise RuntimeError(f"{op}: {name} must have 1 to 4 dimensions, "
                               f"got {nd}")
        size = list(t.shape)
        if t.is_contiguous():
            row = size[-1]
        else:
            st = t.stride()
            row = st[-2] if nd >= 2 else size[-1]
            ok = (size[-1] == 1 or st[-1] == 1) and row >= size[-1]
            if nd >= 3:
                ok = ok and (size[-3] == 1 or st[-3] == size[-2] * row)
            if nd >= 4:
                ok = ok and (size[0] == 1 or st[0] == size[1] * st[1])
            if not ok:
                raise RuntimeError(f"{op}: {name} is taken non-contiguous only "
                                   f"as a slice of its last axis; call "
                                   f".contiguous()")
        shapes.append(tuple([1] * (4 - nd) + size))
        ptrs.append(t.data_ptr())
        rows.append(row)
    return shapes, ptrs, rows


#: (namespace, op) -> the registered op, and the libraries that keep them.
_LOADED: Dict[Tuple[str, str], Any] = {}
_LIBS: List[Any] = []


def torch_op(kernel, *, ns: str = "hk", name: str = "", **const_overrides):
    """`torch.ops.<ns>.<name>` over the code object: the same schema, the
    same out-parameter form and the same checks as the C++ registration
    (runtime.torch_ext), without compiling libtorch's headers."""
    import torch  # noqa: PLC0415

    from ..codegen import scaffold as sc  # noqa: PLC0415

    ir = kernel.trace(**const_overrides)
    op = name or ir.name
    if (hit := _LOADED.get((ns, op))) is not None:
        return hit
    written = sc.written_tensors(ir)
    if not written:
        raise ValueError(f"{ir.name} stores into no global; a torch op over it "
                         f"would mutate nothing and return nothing")
    k = load(kernel, **const_overrides)
    qual = f"{ns}::{op}"

    def impl(*tensors):
        shapes, ptrs, rows = tensor_args(k, tensors, qual)
        dev = tensors[0].device
        if any(t.device != dev for t in tensors[1:]):
            raise RuntimeError(f"{qual}: every tensor must be on one device")
        with torch.cuda.device(dev):
            k.launch(dev.index, torch.cuda.current_stream(dev).cuda_stream,
                     shapes, ptrs, rows)

    def meta(*tensors):
        return None

    lib = torch.library.Library(ns, "FRAGMENT")
    lib.define(sc.torch_schema(ir, op))
    lib.impl(op, impl, "CUDA")
    lib.impl(op, meta, "Meta")
    _LIBS.append(lib)
    got = _LOADED[(ns, op)] = getattr(getattr(torch.ops, ns), op)
    return got


def forget() -> None:
    """Drop the memo; for tests. It cannot unregister an op."""
    _LOADED.clear()


__all__ = ["CodeObjectKernel", "elf_symbol", "hip", "load", "torch_op",
           "tensor_args", "forget"]
