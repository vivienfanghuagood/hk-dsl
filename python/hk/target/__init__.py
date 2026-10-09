"""Architecture facts, as data.

Every number here was measured on the part, not read off a datasheet. The
provenance of each one is in the module that defines it. One definition, shared
by the verifier, the LDS allocator, the register-budget model, the autotuner and
the C++ emitter -- these facts used to be scattered across RDNA.md, kernel
comments and a notebook, and they drifted.
"""

from .base import MmaShape, Target
from .gfx1100 import GFX1100
from .gfx1201 import GFX1201

TARGETS = {t.arch: t for t in (GFX1100, GFX1201)}

# Accept the marketing names too; the Makefiles use these.
_ALIASES = {
    "RDNA3": "gfx1100",
    "RDNA4": "gfx1201",
    "gfx11": "gfx1100",
    "gfx12": "gfx1201",
}


def get_target(arch: str):
    """Look up a target by gfx name or by the GPU_TARGET name common.mk uses."""
    key = _ALIASES.get(arch, arch)
    if key not in TARGETS:
        raise KeyError(
            f"unknown arch {arch!r}; known: {sorted(TARGETS)} "
            f"(aliases: {sorted(_ALIASES)}). Adding one means running the "
            f"probes in tools/rdna-probes, not editing a table."
        )
    return TARGETS[key]


__all__ = ["GFX1100", "GFX1201", "TARGETS", "get_target", "Target", "MmaShape"]
