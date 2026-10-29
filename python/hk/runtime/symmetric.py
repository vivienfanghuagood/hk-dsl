"""Symmetric memory for hk.comm kernels: one buffer a rank of a torch.distributed
group, every rank holding every peer's buffer mapped into its own address space
(hipIpc), as torch tensors. A kernel reaches a peer through a tensor of
`peer(r, ...)` bound to a scoped parameter (`hk.Tensor[dtype, "system"]`).

The binding is checked: while any SymmetricBuffer is open, a launch that passes
memory of one (its own or a peer's) to an *unscoped* parameter raises -- the
stores through it would stay in this XCD's L2 and the loads would re-read stale
lines (measured with tools/cdna-probes/litmus_comm.hip).
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence, Tuple

#: open buffers' address ranges [lo, hi): every rank's mapping in this process
_RANGES: List[Tuple[int, int]] = []

_IPC_HANDLE_BYTES = 64
_hip = None


class _IpcHandle(ctypes.Structure):
    """hipIpcMemHandle_t: 64 opaque bytes, passed BY VALUE to hipIpcOpenMemHandle
    (a ctypes byte array would pass a pointer: hip error 1)."""
    _fields_ = [("reserved", ctypes.c_char * _IPC_HANDLE_BYTES)]


def _lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
    return _hip


def _ck(rc: int, what: str) -> None:
    if rc != 0:
        f = _lib().hipGetErrorString
        f.restype = ctypes.c_char_p
        msg = f(rc)
        raise RuntimeError(f"{what}: hip error {rc} ({msg.decode() if msg else '?'})")


class _DeviceBytes:
    """__cuda_array_interface__ of `nbytes` at a raw device address: a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1",
                                         "data": (ptr, False), "version": 3}


def in_symmetric(ptr: int) -> bool:
    """Whether a device address is inside an open SymmetricBuffer (any rank's)."""
    return any(lo <= ptr < hi for lo, hi in _RANGES)


def check_bindings(kernel_name: str, params: Sequence, tensors: Sequence) -> None:
    """Refuse symmetric memory bound to an unscoped parameter (see the module
    docstring). Called on a launch only while some buffer is open."""
    for (name, ty), t in zip(params, tensors):
        if getattr(ty, "scope", "") in ("device", "system"):
            continue
        ptr = getattr(t, "data_ptr", None)
        if ptr is not None and in_symmetric(ptr()):
            raise TypeError(
                f"{kernel_name}(): {name!r} is symmetric memory (hk.runtime.SymmetricBuffer) but "
                f"its parameter is unscoped: annotate it hk.Tensor[..., \"system\"]. An unscoped "
                f"store stays in this XCD's L2; an unscoped load re-reads a stale line.")


class SymmetricBuffer:
    """`nbytes` on every rank of `group` (collective: every rank constructs it,
    in the same order), mapped into every rank. Zeroed.

    `uncached`: hipDeviceMallocUncached memory (every access past the caches,
    as aiter's mailboxes); default coarse-grained hipMalloc, which is correct
    for scoped accesses and fast for bulk data."""

    def __init__(self, nbytes: int, group=None, *, uncached: bool = False, device=None):
        import torch
        import torch.distributed as dist

        self.nbytes = int(nbytes)
        self.group = group
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None \
            else torch.device(device)
        hip = _lib()
        _ck(hip.hipSetDevice(ctypes.c_int(self.device.index)), "hipSetDevice")
        p = ctypes.c_void_p()
        if uncached:
            _ck(hip.hipExtMallocWithFlags(ctypes.byref(p), ctypes.c_size_t(self.nbytes),
                                          ctypes.c_uint(0x3)), "hipExtMallocWithFlags(uncached)")
        else:
            _ck(hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(self.nbytes)), "hipMalloc")
        _ck(hip.hipMemset(p, ctypes.c_int(0), ctypes.c_size_t(self.nbytes)), "hipMemset")
        _ck(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        self._local = p.value
        self._opened: List[int] = []
        self.addresses = [self._local]
        if self.world > 1:
            h = _IpcHandle()
            _ck(hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(self._local)), "hipIpcGetMemHandle")
            handles: List[Optional[bytes]] = [None] * self.world
            dist.all_gather_object(handles, bytes(h), group=group)
            self.addresses = []
            for r, hb in enumerate(handles):
                if r == self.rank:
                    self.addresses.append(self._local)
                    continue
                q = ctypes.c_void_p()
                hh = _IpcHandle.from_buffer_copy(hb)
                _ck(hip.hipIpcOpenMemHandle(ctypes.byref(q), hh, ctypes.c_uint(1)),
                    f"hipIpcOpenMemHandle(rank {r})")
                self._opened.append(q.value)
                self.addresses.append(q.value)
            dist.barrier(group=group)
        for a in self.addresses:
            _RANGES.append((a, a + self.nbytes))

    def tensor(self, rank: int, dtype, shape, offset: int = 0):
        """Rank `rank`'s buffer (this rank's own for rank == self.rank) as a
        torch tensor of `dtype` and `shape` at byte `offset`: no copy, no owner."""
        import math

        import torch

        n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if offset < 0 or offset + n > self.nbytes:
            raise ValueError(f"[{offset}, {offset + n}) is outside the {self.nbytes} B buffer")
        raw = torch.as_tensor(_DeviceBytes(self.addresses[rank] + offset, n), device=self.device)
        return raw.view(dtype).view(*shape)

    def local(self, dtype, shape, offset: int = 0):
        return self.tensor(self.rank, dtype, shape, offset)

    def peer(self, rank: int, dtype, shape, offset: int = 0):
        return self.tensor(rank, dtype, shape, offset)

    def close(self) -> None:
        """Unmap the peers and free this rank's buffer; collective, idempotent.
        Only once no kernel of any rank can still touch the buffer."""
        hip = _lib()
        for a in self.addresses:
            try:
                _RANGES.remove((a, a + self.nbytes))
            except ValueError:
                pass
        for q in self._opened:
            hip.hipIpcCloseMemHandle(ctypes.c_void_p(q))
        self._opened = []
        if self._local:
            hip.hipFree(ctypes.c_void_p(self._local))
            self._local = 0
        self.addresses = []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 -- interpreter shutdown
            pass


def symmetric_open() -> bool:
    return bool(_RANGES)


__all__ = ["SymmetricBuffer", "check_bindings", "in_symmetric", "symmetric_open"]
