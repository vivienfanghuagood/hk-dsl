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
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import compile as _compile
from . import hostpath
from .compile import CompileError

# ---------------------------------------------------------------- the ELF


def _elf_symbols(data: bytes):
    """(name, section headers, symbol's section index, value, size) for every
    symbol of an ELF64 little-endian file."""
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
            yield data[strtab[4] + st_name:end].decode(), sections, shndx, value, size


def elf_symbol(data: bytes, name: str) -> bytes:
    """The initialised bytes of symbol `name` in an ELF64 little-endian file.
    Enough of ELF for one lookup, so reading a layout needs no dependency."""
    for sym, sections, shndx, value, size in _elf_symbols(data):
        if sym != name:
            continue
        sec = sections[shndx]
        at = sec[4] + (value - sec[3])          # sh_offset + (value - sh_addr)
        return data[at:at + size]
    raise CompileError(f"no symbol {name!r} in the code object")


def elf_kernel(data: bytes, prefix: str) -> str:
    """The one kernel whose symbol starts with `prefix`: its descriptor is
    `<symbol>.kd`."""
    got = {s[:-3] for s, *_ in _elf_symbols(data)
           if s.startswith(prefix) and s.endswith(".kd")}
    if len(got) != 1:
        raise CompileError(f"{len(got)} kernels named {prefix}* in the code object")
    return got.pop()


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


def _layout(offsets, size: int, extra=()) -> Tuple[struct.Struct, Optional[List[int]]]:
    """One struct for the whole argument, so a launch packs it in one call
    instead of one per field: every field at its offset, in offset order,
    with padding between. Also the permutation from the natural order (per
    tensor: pointer, batch, depth, rows, cols, row stride; then the strided
    tensors' three 64-bit strides, `extra`) to offset order, or None where the
    two agree."""
    fields = []
    for po, dims, ro in offsets:
        fields.append((po, "Q", len(fields)))
        for o in dims:
            fields.append((o, "i", len(fields)))
        fields.append((ro, "i", len(fields)))
    for o in extra:
        fields.append((o, "q", len(fields)))
    fields.sort()
    fmt, at = ["<"], 0
    for off, code, _ in fields:
        if off < at:
            raise CompileError(f"argument fields overlap at offset {off}")
        if off > at:
            fmt.append(f"{off - at}x")
        fmt.append(code)
        at = off + struct.calcsize("<" + code)
    if at > size:
        raise CompileError(f"argument fields end at {at}, past its {size} bytes")
    if size > at:
        fmt.append(f"{size - at}x")
    perm = [i for _, _, i in fields]
    return struct.Struct("".join(fmt)), (None if perm == list(range(len(perm))) else perm)

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
        if getattr(ir, "flat_args", False):
            self.symbol = elf_kernel(self.image, self.symbol)
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
        #: The strided tensors (kittens::gl_strided): their indices, and the
        #: offsets of their depth, batch and granule strides.
        self.strided: List[Tuple[int, Tuple[int, int, int]]] = []
        try:
            raw = elf_symbol(self.image, sc.STRIDED_SYMBOL)
        except CompileError:
            raw = b""
        if raw:
            t = struct.unpack(f"<{len(raw) // 8}q", raw)
            self.strided = [(t[1 + 4 * j], tuple(t[2 + 4 * j:5 + 4 * j])) for j in range(t[0])]
        self._struct, self._perm = _layout(
            self.offsets, self.size, [o for _, offs in self.strided for o in offs])
        self._fn: Dict[int, ctypes.c_void_p] = {}
        self._modules: Dict[int, ctypes.c_void_p] = {}
        self._local = threading.local()
        self.torch_dtypes = None

    def function(self, device: int) -> ctypes.c_void_p:
        fn = self._fn.get(device)
        if fn is None:
            from .arch import check_device  # noqa: PLC0415

            check_device(self.ir.arch, f"kernel {self.ir.name!r}", device)
            mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
            _check(hip().hipModuleLoadData(ctypes.byref(mod), self.image),
                   f"{self.ir.name}: hipModuleLoadData")
            _check(hip().hipModuleGetFunction(ctypes.byref(fn), mod,
                                              self.symbol.encode()),
                   f"{self.ir.name}: hipModuleGetFunction({self.symbol})")
            self._modules[device], self._fn[device] = mod, fn
        return fn

    def _values(self, shapes, ptrs, row_strides) -> List[int]:
        """`row_strides`: per tensor its row stride, or for a strided tensor
        (row, depth, batch, granule) strides (tensor_args)."""
        flat: List[int] = []
        for s, p, rs in zip(shapes, ptrs, row_strides):
            flat.append(p)
            flat.extend(s)
            flat.append(rs if isinstance(rs, int) else rs[0])
        for i, _ in self.strided:
            flat.extend(row_strides[i][1:])
        return flat if self._perm is None else [flat[i] for i in self._perm]

    def pack(self, shapes: Sequence[Tuple[int, int, int, int]],
             ptrs: Sequence[int], row_strides: Sequence[int]) -> bytes:
        buf = bytearray(self.size)
        self._struct.pack_into(buf, 0, *self._values(shapes, ptrs, row_strides))
        return bytes(buf)

    def _argument(self):
        """This thread's argument buffer and the `extra` array naming it. The
        launch copies the argument out before it returns, so one buffer
        serves every launch; one per thread, since ctypes drops the GIL for
        the call."""
        held = getattr(self._local, "arg", None)
        if held is None:
            buf = ctypes.create_string_buffer(self.size)
            size = ctypes.c_size_t(self.size)
            extra = (ctypes.c_void_p * 5)(
                _PARAM_BUFFER, ctypes.addressof(buf), _PARAM_SIZE,
                ctypes.addressof(size), _PARAM_END)
            held = self._local.arg = (buf, size, extra)
        return held

    def grid_for(self, shapes) -> Tuple[int, int, int]:
        ext = dict(zip(self.tensors, shapes))
        return tuple(g if isinstance(g, int) else g.value(ext)
                     for g in self.grid)

    def launch(self, device: int, stream: int, shapes, ptrs, row_strides):
        """Launch on `stream` (a hipStream_t as an int; 0 is the null stream)."""
        gx, gy, gz = self.grid_for(shapes)
        if gx <= 0 or gy <= 0 or gz <= 0:
            return                      # an empty grid launches nothing
        buf, _, extra = self._argument()
        self._struct.pack_into(buf, 0, *self._values(shapes, ptrs, row_strides))
        err = hip().hipModuleLaunchKernel(
            self.function(device), gx, gy, gz, self.block, 1, 1, self.shmem,
            stream, None, extra)
        if err:
            _check(err, f"{self.ir.name}: hipModuleLaunchKernel")

    def launcher(self, op: str, *, native: bool = True):
        """`launch(*tensors)`: tensor_args, then launch on torch's current
        stream for the tensors' device -- what the torch op does -- with the
        per-call work cut to what depends on the call.

        The checks and the argument depend on the tensors' dtypes, device,
        shapes and strides, and only the pointers change between calls of
        one shape. So a call whose dtypes are the kernel's, whose tensors are
        on one GPU (`get_device`, -1 off it, the same for all) and whose
        shapes -- and strides, if any tensor is not contiguous -- were seen
        before reads its grid and argument from a memo and packs only the
        pointers. Anything else -- a new shape, a wrong dtype or device --
        takes `tensor_args`, which checks everything and raises what the C++
        op would; a shape it accepts is then memoized. Every launch is the
        same as the unmemoized path's, field for field."""
        import torch  # noqa: PLC0415

        if self.torch_dtypes is None:
            self.torch_dtypes = [getattr(torch, _TORCH_DTYPES[d]) for d in self.dtypes]
        n = len(self.tensors)
        get_device = torch._C._cuda_getDevice
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None) or (
            lambda i: torch.cuda.current_stream(i).cuda_stream)
        plans: Dict[Tuple, Any] = {}
        slots = self._pointer_slots()
        pack = self._struct.pack_into
        launch_kernel = hip().hipModuleLaunchKernel
        block, shmem = self.block, self.shmem
        fns = self._fn

        def slow(tensors):
            shapes, ptrs, rows = tensor_args(self, tensors, op)
            idx = tensors[0].get_device()
            for t in tensors[1:]:
                if t.get_device() != idx:
                    raise RuntimeError(f"{op}: every tensor must be on one device")
            if len(plans) < _PLAN_CAP:
                key = tuple(t.shape for t in tensors)
                if not all(t.is_contiguous() for t in tensors):
                    key += tuple(t.stride() for t in tensors)
                plans[key] = (self.grid_for(shapes), self._values(shapes, [0] * n, rows))
            if get_device() == idx:
                self.launch(idx, raw(idx), shapes, ptrs, rows)
            else:
                with torch.cuda.device(idx):
                    self.launch(idx, raw(idx), shapes, ptrs, rows)

        # One function per tensor count, written out: the generic loop over
        # the tensors is most of what this saves.
        ts = [f"t{i}" for i in range(n)]
        legacy = _legacy_impl(self, op)
        src = [
            f"def launch({', '.join(ts)}):",
            "    if not hostpath.ENABLED:",
            f"        return legacy({', '.join(ts)})",
            f"    if {' or '.join(f'{t}.dtype is not D[{i}]' for i, t in enumerate(ts))}:",
            f"        return slow(({', '.join(ts)},))",
            "    idx = t0.get_device()",
            "    if idx < 0" + "".join(f" or {t}.get_device() != idx" for t in ts[1:]) + ":",
            f"        return slow(({', '.join(ts)},))",
            f"    if {' and '.join(f'{t}.is_contiguous()' for t in ts)}:",
            f"        plan = plans.get(({', '.join(f'{t}.shape' for t in ts)},))",
            "    else:",
            f"        plan = plans.get(({', '.join(f'{t}.shape' for t in ts)}, "
            f"{', '.join(f'{t}.stride()' for t in ts)}))",
            "    if plan is None:",
            f"        return slow(({', '.join(ts)},))",
            "    (gx, gy, gz), v = plan",
            "    if gx <= 0 or gy <= 0 or gz <= 0:",
            "        return",
            "    v = v.copy()",
            *[f"    v[{s}] = {ts[i]}.data_ptr()" for i, s in enumerate(slots)],
            "    if get_device() != idx:",
            "        with torch.cuda.device(idx):",
            "            return self.launch_values(idx, raw(idx), (gx, gy, gz), v)",
            "    buf, _, extra = self._argument()",
            "    pack(buf, 0, *v)",
            "    fn = fns.get(idx)",
            "    if fn is None:",
            "        fn = self.function(idx)",
            "    err = launch_kernel(fn, gx, gy, gz, block, 1, 1, shmem, raw(idx), None, extra)",
            "    if err:",
            "        _check(err, f'{name}: hipModuleLaunchKernel')",
        ]
        env = dict(D=tuple(self.torch_dtypes), slow=slow, legacy=legacy, hostpath=hostpath,
                   plans=plans, get_device=get_device,
                   raw=raw, pack=pack, launch_kernel=launch_kernel, block=block, shmem=shmem,
                   fns=fns, self=self, torch=torch, _check=_check, name=self.ir.name)
        exec("\n".join(src), env)  # noqa: S102 -- source built above from ints and names
        launch = env["launch"]
        mod = _native() if native else None
        progs = tuple(_grid_program(g, self.tensors) for g in self.grid)
        if mod is None or None in progs or n > 32 or self.size > 4096:
            return launch
        # The same call in C (hk._mlir's _hkLaunch): what this function does when its memo
        # hits, without the interpreter; everything else -- a new shape included, since C
        # computes the grid itself -- it hands to this function.
        def function(idx):
            return self.function(idx).value

        def on_error(err):
            _check(err, f"{self.ir.name}: hipModuleLaunchKernel")

        return mod.Launcher(
            tuple(self.torch_dtypes), self.size,
            tuple((p, *dims, rs) for p, dims, rs in self.offsets), progs,
            self.block, self.shmem,
            ctypes.cast(launch_kernel, ctypes.c_void_p).value,
            raw, get_device, function, on_error, launch, hostpath)

    def _pointer_slots(self) -> List[int]:
        """Where each tensor's pointer lands in `_values`' list, by tensor."""
        per = 2 + 4                     # pointer, four extents, row stride
        natural = [i * per for i in range(len(self.tensors))]
        if self._perm is None:
            return natural
        where = {src: dst for dst, src in enumerate(self._perm)}
        return [where[i] for i in natural]

    def launch_values(self, device: int, stream: int, grid, values) -> None:
        """Launch with `_values`' list already made (`launcher`)."""
        gx, gy, gz = grid
        buf, _, extra = self._argument()
        self._struct.pack_into(buf, 0, *values)
        err = hip().hipModuleLaunchKernel(
            self.function(device), gx, gy, gz, self.block, 1, 1, self.shmem,
            stream, None, extra)
        if err:
            _check(err, f"{self.ir.name}: hipModuleLaunchKernel")


_NATIVE: List[Any] = []


def _native():
    """hk._mlir's C launcher module, or None where hk._mlir was built without it."""
    if not _NATIVE:
        try:
            from .._mlir._mlir_libs import _hkLaunch  # noqa: PLC0415
        except ImportError:
            _hkLaunch = None
        _NATIVE.append(_hkLaunch)
    return _NATIVE[0]


#: HostBin's operators, as _hkLaunch's postfix opcodes.
_OPCODES = {"+": 2, "-": 3, "*": 4, "/": 5, "%": 6}


def _grid_program(g, tensors) -> Optional[Tuple[int, ...]]:
    """A grid dimension as _hkLaunch's postfix program over the tensors'
    (batch, depth, rows, cols): (0, v) a constant, (1, tensor, dim) an extent,
    then an opcode per operator. None for anything it does not spell."""
    from ..dsl import host as H  # noqa: PLC0415

    if isinstance(g, int):
        return (0, g)
    if isinstance(g, H.HostConst):
        return (0, g.const)
    if isinstance(g, H.HostDim):
        return (1, tensors.index(g.tensor), H.HostDim.DIMS.index(g.dim))
    if isinstance(g, H.HostBin) and g.op in _OPCODES:
        a, b = _grid_program(g.lhs, tensors), _grid_program(g.rhs, tensors)
        return None if a is None or b is None else a + b + (_OPCODES[g.op],)
    return None


def _legacy_impl(k: "CodeObjectKernel", qual: str):
    """The torch op's implementation as it was before `launcher`: what runs
    under HK_HOST_FASTPATH=0 (hk.runtime.hostpath)."""
    import torch  # noqa: PLC0415

    raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None) or (
        lambda i: torch.cuda.current_stream(i).cuda_stream)

    def impl(*tensors):
        shapes, ptrs, rows = tensor_args(k, tensors, qual)
        idx = tensors[0].get_device()
        for t in tensors[1:]:
            if t.get_device() != idx:
                raise RuntimeError(f"{qual}: every tensor must be on one device")
        stream = raw_stream(idx)
        if torch.cuda.current_device() == idx:
            k.launch(idx, stream, shapes, ptrs, rows)
        else:
            with torch.cuda.device(idx):
                k.launch(idx, stream, shapes, ptrs, rows)

    return impl


#: Shapes a launcher memoizes, per kernel: past it, new shapes take the
#: checked path every call (still correct, just not faster). A serving
#: engine's prefill sees a new M per batch, so the memo must not grow without
#: bound; 4096 shapes are under 2 MB.
_PLAN_CAP = 4096


def load(kernel, **const_overrides) -> CodeObjectKernel:
    """Trace, build as a code object (cached, gated), and wrap."""
    from ..codegen import scaffold as sc  # noqa: PLC0415

    ir = kernel.trace(**const_overrides)
    src = sc.device(ir)
    with_defines = getattr(kernel, "with_defines", None)
    if with_defines:
        src = with_defines(src)
    flags = getattr(kernel, "build_flags", None)
    b = _compile.build(src, ir.arch, name=ir.name, kind="co",
                       extra_flags=flags(ir) if flags else (),
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
    dtypes = k.torch_dtypes
    if dtypes is None:
        import torch  # noqa: PLC0415

        dtypes = k.torch_dtypes = [getattr(torch, _TORCH_DTYPES[d]) for d in k.dtypes]
    shapes, ptrs, rows = [], [], []
    strided = {i for i, _ in getattr(k, "strided", ())}
    for i, (name, want, dt, t) in enumerate(zip(k.tensors, k.dtypes, dtypes, tensors)):
        if not t.is_cuda:
            raise RuntimeError(f"{op}: {name} must be on the GPU")
        if t.dtype != dt:
            raise RuntimeError(f"{op}: {name} must be {want}, got {t.dtype}")
        if i in strided:
            shape, st = strided_args(t, name, op)
            shapes.append(shape)
            ptrs.append(t.data_ptr())
            rows.append(st)
            continue
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


def strided_args(t, name: str, op: str):
    """A strided tensor's (batch, depth, rows, cols) and its (row, depth, batch,
    granule) strides in elements (kittens::gl_strided). Taken as it is: a 1-4-D
    view whose last axis is contiguous (columns in granules of G = 16 bytes,
    `G` elements apart), or a 5-D view (b, d, r, granules, G) whose last axis is
    exactly one contiguous 16-byte granule -- the granule stride is then the
    fourth axis's (vLLM's x-split K cache, permuted to (blocks, heads, block,
    head_dim / G, G))."""
    g = 16 // t.element_size()
    nd, size, st = t.dim(), list(t.shape), list(t.stride())
    if nd == 5:
        if size[4] != g or st[4] != 1:
            raise RuntimeError(f"{op}: {name} is 5-D, so its last axis must be one contiguous "
                               f"16-byte granule ({g} elements, stride 1); got size {size[4]}, "
                               f"stride {st[4]}")
        return (size[0], size[1], size[2], size[3] * g), (st[2], st[1], st[0], st[3])
    if not 1 <= nd <= 4:
        raise RuntimeError(f"{op}: {name} (strided) must have 1 to 5 dimensions, got {nd}")
    if size[-1] != 1 and st[-1] != 1:
        raise RuntimeError(f"{op}: {name} (strided) must be contiguous along its last axis")
    size = [1] * (4 - nd) + size
    st = [0] * (4 - nd) + st
    return tuple(size), (st[2], st[1], st[0], g)


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

    impl = k.launcher(qual)

    def meta(*tensors):
        return None

    lib = torch.library.Library(ns, "FRAGMENT")
    lib.define(sc.torch_schema(ir, op))
    lib.impl(op, impl, "CUDA")
    lib.impl(op, meta, "Meta")
    _LIBS.append(lib)
    _DIRECT[(ns, op)] = impl
    got = _LOADED[(ns, op)] = getattr(getattr(torch.ops, ns), op)
    return got


#: (namespace, op) -> the registered op's implementation, for `direct`.
_DIRECT: Dict[Tuple[str, str], Any] = {}


def direct(kernel, *, ns: str = "hk", name: str = "", **const_overrides):
    """What `torch_op(kernel)` runs under the dispatcher, without it: the same
    checks, the same launch on torch's current stream, 1-1.5 us and a boxing
    of every argument less. For a caller that is itself eager code -- the
    body of an engine's opaque op, which torch.compile never traces into.
    Code torch.compile may trace calls the op (`torch_ext.launcher`)."""
    ir = kernel.trace(**const_overrides)
    op = name or ir.name
    torch_op(kernel, ns=ns, name=name, **const_overrides)
    return _DIRECT[(ns, op)]


def forget() -> None:
    """Drop the memo; for tests. It cannot unregister an op."""
    _LOADED.clear()


__all__ = ["CodeObjectKernel", "direct", "elf_symbol", "hip", "load", "torch_op",
           "tensor_args", "strided_args", "forget"]
