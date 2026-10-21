"""The hook a GEMM kernel factory takes for its store: `epilogue=`.

hk.templates.gemm builds the object; the kernels here only know its shape,
so hk.ops does not import hk.templates. An epilogue has

    params      [(name, GlobalType)]: tensors the kernel takes after C
    consts      {name: value}: constexprs it adds (keyword-only)
    out_dtype   C's element type
    tag         a suffix for the kernel's name
    store(c, ops, tile, r, col, bounded=True)
                traced where the kernel would store `tile` at element (r, col)
                of C: applies the user's function and stores what it returns.
                `ops` is the kernel's keyword arguments past its own -- the
                params and consts above, by name.

`sign` gives a body written `def body(a, b, c, *, WGM=8, **ops)` the
signature hk.kernel reads: the epilogue's tensors after the body's own, C
retyped, its constexprs last. With no epilogue it only hides `**ops`.
"""

from __future__ import annotations

import inspect

from ..types import GlobalType


def sign(body, epilogue, out: str = "c"):
    sig = inspect.signature(body)
    pos, kw = [], []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if p.name == out and epilogue is not None:
            p = p.replace(annotation=GlobalType(epilogue.out_dtype))
        (kw if p.kind is inspect.Parameter.KEYWORD_ONLY else pos).append(p)
    if epilogue is not None:
        pos += [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation=ty) for n, ty in epilogue.params]
        kw += [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=v)
               for n, v in epilogue.consts.items()]
    body.__signature__ = sig.replace(parameters=pos + kw)
    return body


def name(base: str, epilogue) -> str:
    return base if epilogue is None else f"{base}_{epilogue.tag}"


__all__ = ["sign", "name"]
