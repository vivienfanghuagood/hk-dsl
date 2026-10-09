"""The part of a framework patch that is the same for every framework.

Patching `torch.nn.functional.scaled_dot_product_attention` covers any caller
that reaches it through the module -- `F.scaled_dot_product_attention(...)` --
and misses every caller that did `from torch.nn.functional import
scaled_dot_product_attention` at import time, because that one holds its own
reference to the original function object. Frameworks do both.

So the patch does both: it rebinds the attribute on `F`, and then walks the
modules the framework has already imported and rebinds every name that is still
bound to the original object. Walking by identity rather than by name means it
does not need to know which version of the framework this is, and it cannot
rebind something that merely has the same name.

The consequence to document: `apply()` has to run *after* the framework's
modules are imported and *before* a model runs. A name imported later still
gets the patched function, because by then `F`'s attribute is already ours.
"""

from __future__ import annotations

import sys
from typing import List, Tuple


def rebind(prefixes: Tuple[str, ...], old, new) -> List[str]:
    """Rebind every module attribute under `prefixes` that *is* `old`."""
    hits = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(prefixes):
            continue
        try:
            members = list(vars(mod).items())
        except TypeError:  # pragma: no cover -- an exotic module object
            continue
        for attr, val in members:
            if val is old:
                try:
                    setattr(mod, attr, new)
                except Exception:  # noqa: BLE001,S110 -- a frozen module
                    continue
                hits.append(f"{name}.{attr}")
    return hits


def importable(name: str) -> bool:
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def patch_sdpa(prefixes: Tuple[str, ...]) -> Tuple[str, List[str]]:
    """Point the framework's attention at the generated kernel.

    Returns (report, rebound names). The generated kernel only takes what it
    takes -- bf16, head_dim 64 or 128, no mask, no dropout -- and
    `hk.ops.sdpa.scaled_dot_product_attention` forwards everything else to
    torch, so a framework that calls it with something else keeps working.
    """
    import torch.nn.functional as F  # noqa: PLC0415,N812

    from ..ops import sdpa  # noqa: PLC0415

    original = F.scaled_dot_product_attention
    sdpa.patch()
    if original is sdpa.scaled_dot_product_attention:
        return "already patched", []
    names = rebind(prefixes, original, sdpa.scaled_dot_product_attention)
    return (
        f"attention -> hk ({len(names)} direct import"
        f"{'' if len(names) == 1 else 's'} rebound)",
        names,
    )


def unpatch_sdpa(prefixes: Tuple[str, ...]) -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not sdpa.patched():
        return "not patched"
    ours = sdpa.scaled_dot_product_attention
    sdpa.unpatch()
    import torch.nn.functional as F  # noqa: PLC0415,N812

    names = rebind(prefixes, ours, F.scaled_dot_product_attention)
    return f"attention -> torch ({len(names)} rebound)"
