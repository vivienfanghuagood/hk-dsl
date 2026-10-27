"""The switch for the host fast paths: `HK_HOST_FASTPATH=0` turns them off.

An engine route spends its host time deciding what only the call's shapes
decide -- which kernel, which schedule, the launch's grid and argument -- and
doing it again on every call. The fast paths keep those decisions per shape
and launch code-object kernels without the torch dispatcher in between
(`codeobj.CodeObjectKernel.launcher`, `codeobj.direct`, `torch_ext.launcher`,
the routes' plan memos). They launch the same kernels with the same
arguments, so outputs do not change; only the host time does.

Off, every call takes the path it took before them, line for line. That is
for two things: a user who suspects them can turn them off, and an A/B can
flip `ENABLED` between rounds of one process (`set_enabled`), which is the
only comparison of host costs this repository accepts -- two processes differ
by more than what is being measured.

Read at every call, so flipping it takes effect on the next call. A memo made
while on is kept while off and used again when on.
"""

from __future__ import annotations

import os

#: Whether the fast paths run. `HK_HOST_FASTPATH` at import; `set_enabled` after.
ENABLED = os.environ.get("HK_HOST_FASTPATH", "1").strip().lower() not in ("0", "off", "no", "false")


def set_enabled(on: bool) -> bool:
    """Turn the fast paths on or off for every call from now on; returns the previous value."""
    global ENABLED
    prev, ENABLED = ENABLED, bool(on)
    return prev


__all__ = ["ENABLED", "set_enabled"]
