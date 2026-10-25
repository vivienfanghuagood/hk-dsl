"""hk.comm: hand-offs between the XCDs of a GPU and between GPUs (gfx950).

A *scoped* tensor -- `hk.Tensor[dtype, "system"]` for memory other GPUs touch
while the kernel runs (a peer's buffer, a mailbox), `"device"` for memory other
XCDs of this GPU touch -- carries its scope into every access through it. The
scope is the tensor's, not the access's, because on gfx950 *both* sides of a
hand-off must carry it (measured, KernelWiki pitfalls/gfx950-cross-gpu-memory-
model.md): a plain store stays in the writer's L2 and is not seen however long
the reader waits for the flag after it; a plain load after a correct flag
re-reads the line the reader had before -- on uncached memory too. A tensor
whose every access is scoped cannot get one of them wrong.

The protocol, the one the litmus suite found correct in every placement:

    writer                                  reader
    peer[idx] = tile         (scoped store) v = hk.comm.wait(flag.at[0], n)
    hk.comm.signal(flag.at[0], n)           t = peer[idx].load(T) (scoped load)

`signal` waits for this workgroup's stores to be acknowledged (vmcnt(0)),
passes a barrier and stores the flag from one thread; `wait` polls with the
scope and s_sleep backoff, and turns a hang into an error after `timeout_s`.

A message that is one int32 needs neither: `post` stores (value, tag) as one
8-byte store and `take` polls for the tag and returns the value -- one link
crossing instead of data, drain, flag and a dependent read (8 B stores never
tore over xGMI in the litmus suite).
"""

from __future__ import annotations

from typing import Optional

from .._mlir.dialects import hk as hkd
from ..types import ScalarType, i32
from .values import Int, View, _i32, _i64, _t, as_int, scope_aux

#: s_memrealtime runs at 100 MHz on gfx9 (not the shader clock: clock64() on
#: gfx1100 is 20 bits and wraps, KernelWiki gfx1100-device-flags-unreliable).
REALTIME_HZ = 100_000_000


def _error(error, what: str):
    if error is None:
        return None, None
    if not isinstance(error, View) or error.idx.unit != "element" or error.tensor.dtype is not i32:
        raise TypeError(f"hk.comm.{what}(error=...) takes an int32 element view: err.at[0]")
    return error.tensor.v, error.idx.v


def _ticks(timeout_s: float, what: str) -> int:
    if timeout_s < 0:
        raise TypeError(f"hk.comm.{what}(timeout_s={timeout_s!r}): seconds >= 0")
    return int(round(timeout_s * REALTIME_HZ))


def _flag(view, what: str) -> int:
    if not isinstance(view, View):
        raise TypeError(f"hk.comm.{what} takes an element view of a flag: flag.at[i]")
    if view.idx.unit != "element":
        raise TypeError(f"hk.comm.{what} takes an element coordinate: flag.at[i]")
    if view.tensor.dtype is not i32:
        raise TypeError(f"hk.comm.{what}: a flag is an int32 element, not "
                        f"{view.tensor.dtype.name}")
    if not getattr(view.tensor.ty, "scope", ""):
        raise TypeError(
            f"hk.comm.{what} on {view.tensor.name!r}, an unscoped tensor: annotate it "
            f"hk.Tensor[hk.i32, \"system\"] (another GPU) or \"device\" (another XCD). An "
            f"unscoped flag store stays in this XCD's L2 and an unscoped poll never sees "
            f"the peer's store -- both measured on gfx950.")
    return scope_aux(view.tensor, f"hk.comm.{what}")


def signal(view, value) -> None:
    """Publish `value` into the int32 element `view` (a scoped tensor's): every
    store this workgroup issued before it is acknowledged first, then one
    thread stores the flag with the tensor's scope. What it announces must have
    gone to scoped tensors (their stores carry the scope; a plain store would
    sit in this XCD's L2). Contains a barrier: every warp must reach it."""
    aux = _flag(view, "signal")
    hkd.SignalOp(view.tensor.v, view.idx.v, as_int(value).v, _i64(aux), loc=_t().loc())


def arrive(view, value=1) -> Int:
    """As `signal`, but one thread *adds* `value` to the int32 element `view`
    (a scoped tensor's) with an atomic of the tensor's scope; returns the
    element's value before the add (wave-uniform). Every store this workgroup
    issued before it is acknowledged first. Contains a barrier."""
    aux = _flag(view, "arrive")
    op = hkd.ArriveOp(_i32(), view.tensor.v, view.idx.v, as_int(value).v, _i64(aux),
                      loc=_t().loc("arrived"))
    return Int(op.result, ScalarType(i32))


def claim(view, value=1) -> Int:
    """A work queue's take: one thread adds `value` to the int32 element `view` (a scoped
    tensor's) with an atomic of its scope and the workgroup gets the value before the add -- a
    distinct one for every claim. No drain: what this workgroup stored before is not ordered by it
    (use `signal` / `arrive` for that). In a persistent kernel:

        t = hk.comm.claim(q.at[0])
        while t < n:
            ...task t...
            t = hk.comm.claim(q.at[0])

    Contains barriers: every warp must reach it."""
    aux = _flag(view, "claim")
    op = hkd.ArriveOp(_i32(), view.tensor.v, view.idx.v, as_int(value).v, _i64(aux), nodrain=True,
                      loc=_t().loc("claimed"))
    return Int(op.result, ScalarType(i32))


def barrier(view, target, *, timeout_s: float = 2.0, error: Optional[View] = None) -> Int:
    """Every workgroup that reaches it adds 1 to the int32 element `view` and
    waits until it is >= `target` -- with an epoch, target = epoch x the number
    of arrivals, so the counter never needs resetting. With a "device" counter
    it is a grid barrier (the kernel must be `resident=True`: every workgroup on
    the GPU at once), with a "system" one a barrier across GPUs. The stores
    before it are visible after it to accesses of the same scope."""
    arrive(view, 1)
    return wait(view, target, timeout_s=timeout_s, error=error)


def xcd_barrier(cnt, epoch, workgroups: int, *, timeout_s: float = 2.0,
                error: Optional[View] = None) -> Int:
    """A grid barrier aggregated per XCD: each workgroup arrives at its XCD's counter
    (`cnt.at[16 * (1 + hk.arch.xcd_id())]`), the last of each XCD at the top one (`cnt.at[0]`), and
    everyone waits for the eight -- 8 global atomics a round instead of `workgroups`. Measured on an
    MI355X, 256 workgroups: 2.53 us a barrier against 3.71 flat. `cnt`: a "device"-scoped int32 tensor
    of at least 144 elements, zero at the first round; `epoch`: 1, 2, ... a round. The grid must be
    `resident=True` with `workgroups` a multiple of 8, spread evenly over the XCDs (as the hardware
    deals a resident grid: round robin) -- if not, the barrier times out into `error`."""
    if workgroups % 8:
        raise TypeError(f"hk.comm.xcd_barrier: {workgroups} workgroups, not a multiple of the 8 XCDs")
    from . import arch  # noqa: PLC0415

    x = arch.xcd_id()
    per = workgroups // 8
    old = arrive(cnt.at[16 * (1 + x)], 1)
    _arrive_if(old == epoch * per - 1, cnt.at[0])       # the XCD's last arrival goes up one level
    return wait(cnt.at[0], epoch * 8, timeout_s=timeout_s, error=error)


def _arrive_if(cond, view):
    from .control import if_  # noqa: PLC0415

    with if_(cond):
        arrive(view, 1)


def wait(view, value, *, cmp: str = ">=", timeout_s: float = 2.0,
         error: Optional[View] = None) -> Int:
    """Poll the int32 element `view` (a scoped tensor's) until it is `>=` (signed)
    or `==` `value`; return the value seen (wave-uniform). Every warp polls,
    with s_sleep between polls. After `timeout_s` seconds (0: never) it gives
    up: with `error` (an int32 element, e.g. err.at[0]) lane 0 ORs 1 into it
    and the value seen is returned -- check it on the host; without, the
    kernel traps, so a lost signal is an error rather than a hang."""
    aux = _flag(view, "wait")
    if cmp not in (">=", "=="):
        raise TypeError(f"hk.comm.wait(cmp={cmp!r}): \">=\" or \"==\"")
    ticks = _ticks(timeout_s, "wait")
    err, err_idx = _error(error, "wait")
    op = hkd.WaitOp(_i32(), view.tensor.v, view.idx.v, as_int(value).v,
                    "ge" if cmp == ">=" else "eq", _i64(ticks), _i64(aux),
                    err=err, err_idx=err_idx, loc=_t().loc("seen"))
    return Int(op.result, ScalarType(i32))


def post(view, value, tag) -> None:
    """Post a tagged message: one thread stores `value` into the int32 element `view` (a scoped
    tensor's) and `tag` into the next element, as ONE 8-byte store of the tensor's scope. The two
    elements are the message's slot: start it at an even element of an 8-byte aligned tensor (the
    kernel traps on a misaligned slot -- the store could tear). No drain and no barrier: the value
    travels with its tag, so `take` sees both or neither; it orders nothing else, so a message
    about *other* data is still `signal`'s job. A slot holds one message: the writer reuses it after
    the reader has taken it (an ack), or posts round r into slot r % S with tag r.

        hk.comm.post(box.at[0], count, r)        # writer
        n = hk.comm.take(box.at[0], r)           # reader: the count of round r
    """
    aux = _flag(view, "post")
    hkd.PostOp(view.tensor.v, view.idx.v, as_int(value).v, as_int(tag).v, _i64(aux), loc=_t().loc())


def take(view, tag, *, timeout_s: float = 2.0, error: Optional[View] = None) -> Int:
    """Take a tagged message: poll the slot `post` writes at the int32 element `view` (a scoped
    tensor's) -- one 8-byte load of the tensor's scope a poll, s_sleep between -- until its tag
    is `tag`, and return its value (wave-uniform). Every warp polls. Timeout and `error` as
    `wait`."""
    aux = _flag(view, "take")
    err, err_idx = _error(error, "take")
    op = hkd.TakeOp(_i32(), view.tensor.v, view.idx.v, as_int(tag).v, _i64(_ticks(timeout_s, "take")),
                    _i64(aux), err=err, err_idx=err_idx, loc=_t().loc("taken"))
    return Int(op.result, ScalarType(i32))


__all__ = ["signal", "arrive", "claim", "wait", "barrier", "xcd_barrier", "post", "take", "REALTIME_HZ"]
