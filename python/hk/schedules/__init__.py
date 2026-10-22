"""hk.schedules: kernel schedules as data.

A schedule is every knob of one kernel family, as a frozen dataclass: it
says whether it can run a shape (check, with reasons, before compiling),
enumerates its neighbours and its family's legal space, round-trips through
the tuned records and route tables, and builds and launches its kernels.

    import hk.schedules as S

    s = S.lookup(4096, 4096, 4096, arch="gfx950")   # what hk.ops.matmul runs
    s.check(2048, 16384, 4096, arch="gfx950")        # [] or the reasons
    for t in S.space(2048, 16384, 4096, arch="gfx950", around=s):
        ...                                          # t.kernel().build(), t.launch(a, b)
    rec = s.to_record()                              # {"family": "pp", ...}
    S.from_record(rec) == s
    hk.templates.gemm(x, w, epilogue=hk.templates.gemm.bias, bias=b, schedule=s)

GEMM families (hk.schedules.gemm): Rdna (gfx1100's WMMA GEMM), Decode (the
decode GEMV, both GPUs), PingPong and Dp (gfx950's two), and Columns (two
schedules on column ranges of C). docs/schedules.md is the guide;
docs/design/schedules.md the design.
"""

from __future__ import annotations

from ._base import Axis, Illegal, Schedule, families, family, from_record, knob
from .gemm import (CUS, Columns, Decode, Dp, GemmSchedule, PingPong, Rdna, Step,
                   families_for, from_candidate, lookup, space)

__all__ = ["Schedule", "GemmSchedule", "Axis", "Illegal", "Rdna", "Decode", "PingPong",
           "Dp", "Columns", "Step", "knob", "family", "families", "families_for",
           "from_record", "from_candidate", "lookup", "space"]
