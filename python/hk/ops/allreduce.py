"""An exact bf16 all-reduce across two GPUs (TP2), written with hk.comm (gfx950).

    ar = hk.ops.allreduce.AllReduce(group, max_numel=8192 * 5120)
    y = ar(x)            # x: bf16, contiguous, on this rank's GPU; y: the sum, valid until the
                         # call after next (ar(x, out=o) copies it out instead)

*Exact*: every element is the fp32 sum of the two ranks' values rounded once to bf16 -- bitwise the
same on both ranks, and as accurate as a bf16 result can be (RCCL rounds in bf16 along its ring).
*Push-only*: every byte over the link is a remote store (remote loads lose to stores on the link).
*Few workgroups* (`workgroups`, default up to 32). Running it on a second stream beside a GEMM
hides little (measured <= 32%: the GEMM, sized for every CU, loses a wave, and a resident kernel whose
workgroups cannot all start stalls both ranks); overlap needs the comm inside the compute kernel.

Two-shot. Rank r owns half r of the tiles (16x32 bf16 = 1 KiB of the flat buffer). Workgroup w, warp q
take tiles in groups of G:
  1. my partial of the peer's half -> the peer's `rs`; signal its flag[w] = e
  2. wait my flag[w] >= e; s = mine + rs in fp32 -> y (bf16) here and in the peer's y; signal
  3. wait for the peer's half of y
The bytes that cross the link move as fp32 16x32 tiles of a bit view: bf16 tiles store 8 B row pieces,
which run the link at 34 GB/s where fp32 tiles run it at 56 (KernelWiki
hk-bf16-tile-stores-half-rate-over-xgmi). `e` is a device epoch the last workgroup to finish moves on,
so a call is one launch and a CUDA graph can capture it; y alternates between two buffers by e's parity,
so a rank one call ahead never writes into a y the other is still reading.
"""

from __future__ import annotations

import functools
from typing import Optional

import hk
from hk import i32

WARPS = 4
TILE_ELEMS = 16 * 32          # a bf16 16x32 tile of the flat [N, 32] view
FLAG_STRIDE = 16              # a flag a 64 B line


@functools.lru_cache(maxsize=None)
def _kernel(nwg: int, G: int):
    TB32 = hk.RT(hk.fp32, 16, 32)     # bf16 values, held in fp32 for the sum
    TF = hk.RT(hk.fp32, 16, 32)       # 2 KiB of bits
    H = G // 2

    @hk.kernel(arch="gfx950", warps=WARPS, name=f"allreduce2_nwg{nwg}_g{G}",
               grid=lambda p: (nwg, 1, 1), resident=True)
    def allreduce2(x_b: hk.Tensor[hk.bf16], x_f: hk.Tensor[hk.fp32],
                   rs_mine_b: hk.Tensor[hk.bf16, "system"], rs_peer_f: hk.Tensor[hk.fp32, "system"],
                   y_mine_b: hk.Tensor[hk.bf16, "system"], y_mine_f: hk.Tensor[hk.fp32, "system"],
                   y_peer_f: hk.Tensor[hk.fp32, "system"],
                   fl_mine: hk.Tensor[i32, "system"], fl_peer: hk.Tensor[i32, "system"],
                   ctl: hk.Tensor[i32, "device"], err: hk.Tensor[i32],
                   *, RANK: int = 0, NT: int = 2, NTMAX: int = 2):
        w = hk.arch.block_idx.x
        group = w * WARPS + hk.arch.warp_id()
        e = ctl.at[0].load_scalar(invariant=True) + 1
        mine0 = RANK * NT
        peer0 = (1 - RANK) * NT
        yb = (e % 2) * 2 * NTMAX           # this call's y buffer, in bf16 tiles
        for tf in hk.range(group * H, NT // 2, nwg * WARPS * H):
            ts = [x_f[hk.tile_coord(0, 0, peer0 // 2 + tf + g, 0)].load(TF) for g in hk.range_constexpr(H)]
            for g in hk.range_constexpr(H):
                rs_peer_f[hk.tile_coord(0, 0, tf + g, 0)] = ts[g]
        hk.comm.signal(fl_peer.at[FLAG_STRIDE * w], e)
        hk.comm.wait(fl_mine.at[FLAG_STRIDE * w], e, timeout_s=2.0, error=err.at[0])
        for t in hk.range(group * G, NT, nwg * WARPS * G):
            a = [x_b[hk.tile_coord(0, 0, mine0 + t + g, 0)].load(TB32) for g in hk.range_constexpr(G)]
            b = [rs_mine_b[hk.tile_coord(0, 0, t + g, 0)].load(TB32) for g in hk.range_constexpr(G)]
            for g in hk.range_constexpr(G):
                y_mine_b[hk.tile_coord(0, 0, yb + mine0 + t + g, 0)] = a[g] + b[g]   # rounded once
            hk.arch.vm_wait(0)
            ys = [y_mine_f[hk.tile_coord(0, 0, (yb + mine0 + t) // 2 + g, 0)].load(TF)
                  for g in hk.range_constexpr(H)]
            for g in hk.range_constexpr(H):
                y_peer_f[hk.tile_coord(0, 0, (yb + mine0 + t) // 2 + g, 0)] = ys[g]
        hk.comm.signal(fl_peer.at[FLAG_STRIDE * (nwg + w)], e)
        hk.comm.wait(fl_mine.at[FLAG_STRIDE * (nwg + w)], e, timeout_s=2.0, error=err.at[0])
        # the last workgroup to get here moves the epoch on (every workgroup read e at its start) and
        # resets the count for the next call, which may have another grid
        arrived = hk.comm.arrive(ctl.at[16], 1)
        if arrived == nwg - 1:
            if hk.arch.warp_id() == 0:          # a scalar atomic is one a warp
                ctl.at[16].atomic_add_scalar(-nwg)
                ctl.at[0].atomic_max_scalar(e)

    return allreduce2


@functools.lru_cache(maxsize=None)
def _kernel_int8(nwg: int, G: int):
    """The INT8 codec (`codec="int8"`): what crosses the link is int8 per 32-element group (a row of
    the flat view) with an fp32 scale (absmax / 127), as quickreduce's INT8 mode carries. The partials
    are quantized for the reduce-scatter and the sum again for the all-gather; both ranks take
    dequant(quant(sum)) -- the owner rank too -- so the replicas stay bitwise equal."""
    assert G % 4 == 0
    T32 = hk.RT(hk.fp32, 16, 32)
    TF = hk.RT(hk.fp32, 16, 32)       # 64 int8 rows = 4 bf16 tiles' codes, as bits
    vt = hk.col_vec(T32)
    Q = G // 4

    @hk.kernel(arch="gfx950", warps=WARPS, name=f"allreduce2_int8_nwg{nwg}_g{G}",
               grid=lambda p: (nwg, 1, 1), resident=True)
    def allreduce2_int8(x_b: hk.Tensor[hk.bf16], y_b: hk.Tensor[hk.bf16],
                        tx_q: hk.Tensor[hk.i8], tx_f: hk.Tensor[hk.fp32],
                        rs_q: hk.Tensor[hk.i8, "system"], rs_s: hk.Tensor[hk.fp32, "system"],
                        rs_peer_f: hk.Tensor[hk.fp32, "system"], rs_s_peer: hk.Tensor[hk.fp32, "system"],
                        ag_q: hk.Tensor[hk.i8, "system"], ag_s: hk.Tensor[hk.fp32, "system"],
                        ag_peer_f: hk.Tensor[hk.fp32, "system"], ag_s_peer: hk.Tensor[hk.fp32, "system"],
                        fl_mine: hk.Tensor[i32, "system"], fl_peer: hk.Tensor[i32, "system"],
                        ctl: hk.Tensor[i32, "device"], err: hk.Tensor[i32],
                        *, RANK: int = 0, NT: int = 4):
        w = hk.arch.block_idx.x
        group = w * WARPS + hk.arch.warp_id()
        step = nwg * WARPS * G
        e = ctl.at[0].load_scalar(invariant=True) + 1
        mine0 = RANK * NT
        peer0 = (1 - RANK) * NT

        def quantize(v, t, s_peer):
            # v: fp32 tile of 16 groups of 32 -> int8 codes in the local staging, scales to the peer;
            # returns the scales
            amax = hk.zeros(vt)
            amax.max_(v.abs().max(axis=1))
            amax.maximum(1e-12, into=amax)
            scale = amax * (1.0 / 127.0)
            s_peer[hk.elem_coord(0, 0, 0, 16 * t)] = scale
            v *= hk.full(vt, 127.0) / amax
            tx_q[hk.tile_coord(0, 0, t, 0)] = v          # round + clamp in the store
            return scale

        def ship(t, peer_f):
            hk.arch.vm_wait(0)                           # this warp's codes are in the staging
            ts = [tx_f[hk.tile_coord(0, 0, t // 4 + g, 0)].load(TF) for g in hk.range_constexpr(Q)]
            for g in hk.range_constexpr(Q):
                peer_f[hk.tile_coord(0, 0, t // 4 + g, 0)] = ts[g]

        def dequant(q, s, t):
            v = q[hk.tile_coord(0, 0, t, 0)].load(T32)
            v *= s[hk.elem_coord(0, 0, 0, 16 * t)].load(vt)
            return v

        # 1. my partial of the peer's half, quantized -> the peer's rs
        for t in hk.range(group * G, NT, step):
            for g in hk.range_constexpr(G):
                quantize(x_b[hk.tile_coord(0, 0, peer0 + t + g, 0)].load(T32), t + g, rs_s_peer)
            ship(t, rs_peer_f)
        hk.comm.signal(fl_peer.at[FLAG_STRIDE * w], e)
        # 2. mine + the peer's (dequantized) -> quantized once more; my y is its dequantization
        hk.comm.wait(fl_mine.at[FLAG_STRIDE * w], e, timeout_s=2.0, error=err.at[0])
        for t in hk.range(group * G, NT, step):
            scales = []
            for g in hk.range_constexpr(G):
                s = x_b[hk.tile_coord(0, 0, mine0 + t + g, 0)].load(T32) + dequant(rs_q, rs_s, t + g)
                scales.append(quantize(s, t + g, ag_s_peer))
            ship(t, ag_peer_f)                           # (its vm_wait also covers the codes read back)
            for g in hk.range_constexpr(G):
                v = tx_q[hk.tile_coord(0, 0, t + g, 0)].load(T32)
                v *= scales[g]
                y_b[hk.tile_coord(0, 0, mine0 + t + g, 0)] = v   # what the peer computes from them
        hk.comm.signal(fl_peer.at[FLAG_STRIDE * (nwg + w)], e)
        hk.comm.wait(fl_mine.at[FLAG_STRIDE * (nwg + w)], e, timeout_s=2.0, error=err.at[0])
        # 3. the peer's half: dequantize its codes
        for t in hk.range(group * G, NT, step):
            for g in hk.range_constexpr(G):
                y_b[hk.tile_coord(0, 0, peer0 + t + g, 0)] = dequant(ag_q, ag_s, t + g)
        arrived = hk.comm.arrive(ctl.at[16], 1)
        if arrived == nwg - 1:
            if hk.arch.warp_id() == 0:
                ctl.at[16].atomic_add_scalar(-nwg)
                ctl.at[0].atomic_max_scalar(e)

    return allreduce2_int8


class AllReduce:
    """The two-rank exact bf16 all-reduce (module docstring). Collective to construct: every rank of
    `group` (a torch.distributed group of size 2; a gloo group is enough, it only exchanges IPC
    handles) constructs it, in the same order. `max_numel`: the largest tensor it will reduce."""

    def __init__(self, group, max_numel: int, *, workgroups: int = 32, codec: str = "exact",
                 device=None):
        import torch
        import torch.distributed as dist

        from ..runtime import SymmetricBuffer

        if dist.get_world_size(group) != 2:
            raise ValueError("hk.ops.allreduce.AllReduce reduces across exactly two ranks (TP2)")
        if codec not in ("exact", "int8"):
            raise ValueError(f"AllReduce(codec={codec!r}): \"exact\" or \"int8\"")
        self.codec = codec
        if codec == "int8":
            self._init_int8(group, max_numel, workgroups, device)
            return
        self.rank = dist.get_rank(group)
        self.max_tiles = -(-max_numel // (2 * TILE_ELEMS)) * 2        # even: two halves
        self.workgroups = workgroups
        ntmax = self.max_tiles // 2
        self._rs_bytes = ntmax * TILE_ELEMS * 2
        self._y_bytes = 2 * self.max_tiles * TILE_ELEMS * 2            # two buffers (by parity), each the whole tensor
        self._fl_bytes = 2 * workgroups * FLAG_STRIDE * 4
        self.buf = SymmetricBuffer(self._rs_bytes + self._y_bytes + self._fl_bytes, group, device=device)
        dev = self.buf.device
        self.ctl = torch.zeros(64, dtype=torch.int32, device=dev)      # [0] epoch, [16] arrivals
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        peer = 1 - self.rank
        f32 = lambda t: t.view(torch.float32).view(-1, 32)            # noqa: E731
        rs_m = self.buf.local(torch.bfloat16, (ntmax * 16, 32))
        rs_p = self.buf.peer(peer, torch.bfloat16, (ntmax * 16, 32))
        y_m = self.buf.local(torch.bfloat16, (2 * self.max_tiles * 16, 32), offset=self._rs_bytes)
        y_p = self.buf.peer(peer, torch.bfloat16, (2 * self.max_tiles * 16, 32), offset=self._rs_bytes)
        off = self._rs_bytes + self._y_bytes
        n_fl = self._fl_bytes // 4
        self._args = (rs_m, f32(rs_p), y_m, f32(y_m), f32(y_p),
                      self.buf.local(torch.int32, (n_fl,), offset=off),
                      self.buf.peer(peer, torch.int32, (n_fl,), offset=off), self.ctl, self.err)
        self._y = y_m
        self._calls = 0

    def _init_int8(self, group, max_numel, workgroups, device):
        import torch
        import torch.distributed as dist

        from ..runtime import SymmetricBuffer

        self.rank = dist.get_rank(group)
        self.max_tiles = -(-max_numel // (8 * TILE_ELEMS)) * 8      # all tiles; a multiple of 4 a half
        self.workgroups = workgroups
        rows = self.max_tiles // 2 * 16                           # rows of 32 in a half
        q, sc = rows * 32, rows * 4
        fl = 2 * workgroups * FLAG_STRIDE * 4
        self.buf = SymmetricBuffer(2 * (q + sc) + fl, group, device=device)
        dev = self.buf.device
        self.ctl = torch.zeros(64, dtype=torch.int32, device=dev)
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        peer = 1 - self.rank
        f32 = lambda t: t.view(torch.float32).view(-1, 32)          # noqa: E731
        L, P = self.buf.local, (lambda *a, **k: self.buf.peer(peer, *a, **k))
        rs_q, rs_s = L(torch.int8, (rows, 32)), L(torch.float32, (rows,), offset=q)
        ag_q, ag_s = L(torch.int8, (rows, 32), offset=q + sc), L(torch.float32, (rows,), offset=2 * q + sc)
        self._tx = torch.zeros(rows, 32, dtype=torch.int8, device=dev)
        self._args8 = (self._tx, f32(self._tx),
                       rs_q, rs_s, f32(P(torch.int8, (rows, 32))), P(torch.float32, (rows,), offset=q),
                       ag_q, ag_s, f32(P(torch.int8, (rows, 32), offset=q + sc)),
                       P(torch.float32, (rows,), offset=2 * q + sc),
                       L(torch.int32, (fl // 4,), offset=2 * (q + sc)),
                       P(torch.int32, (fl // 4,), offset=2 * (q + sc)), self.ctl, self.err)
        self._calls = 0

    def _call_int8(self, x, out):
        import torch

        n = x.numel()
        if n % (8 * TILE_ELEMS):
            raise ValueError(f"AllReduce(int8): {n} elements; a multiple of {8 * TILE_ELEMS}")
        nt = n // (2 * TILE_ELEMS)
        if 2 * nt > self.max_tiles:
            raise ValueError(f"AllReduce: {n} elements, more than max_numel")
        plan = self._plan(nt, Gs=(8, 4))
        if plan is None:
            raise ValueError(f"AllReduce(int8): {n} elements do not split into workgroups")
        nwg, G = plan
        y = out if out is not None else torch.empty_like(x)
        _kernel_int8(nwg, G)(x.view(-1, 32), y.view(-1, 32), *self._args8, RANK=self.rank, NT=nt)
        self._calls += 1
        return y

    def _plan(self, nt: int, Gs=(8, 4, 2)):
        for G in Gs:
            if nt % G == 0 and (nt // 2) % (G // 2) == 0:
                for nwg in (self.workgroups, 32, 16, 8, 4, 2, 1):
                    if nwg <= self.workgroups and nt % (nwg * WARPS * G) == 0:
                        return nwg, G
        return None

    def __call__(self, x, out=None):
        import torch

        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise TypeError("AllReduce takes a contiguous bf16 tensor")
        if self.codec == "int8":
            return self._call_int8(x, out)
        n = x.numel()
        if n % (2 * TILE_ELEMS):
            raise ValueError(f"AllReduce: {n} elements; a multiple of {2 * TILE_ELEMS} (two halves of "
                             f"16x32 tiles)")
        nt = n // (2 * TILE_ELEMS)
        if 2 * nt > self.max_tiles:
            raise ValueError(f"AllReduce: {n} elements, more than max_numel")
        plan = self._plan(nt)
        if plan is None:
            raise ValueError(f"AllReduce: {n} elements do not split into workgroups of 16x32 tiles")
        nwg, G = plan
        xb = x.view(-1, 32)
        xf = x.view(-1).view(torch.float32).view(-1, 32)      # the same bytes, 2 KiB fp32 tiles
        _kernel(nwg, G)(xb, xf, *self._args, RANK=self.rank, NT=nt,
                        NTMAX=self.max_tiles // 2)
        self._calls += 1
        parity = self._calls % 2          # the kernel's e = calls made so far (its epoch + 1)
        y = self._y[parity * self.max_tiles * 16:][: 2 * nt * 16].view(x.shape)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def error(self) -> int:
        """1 if a wait of any call so far timed out (the peer never arrived): its result is wrong."""
        return int(self.err[0])

    def close(self) -> None:
        self.buf.close()


__all__ = ["AllReduce"]
