"""hk.math: elementwise functions of a tile or a vector."""

from __future__ import annotations

from .values import binary, unary

_FP32 = (" fp32 tiles and vectors only: load bf16/fp16 data into an fp32 tile "
         "(the load converts) and `.to(hk.bf16)` the result.")
_ANY = " bf16, fp16 and fp32."


def _unary(name, what):
    def f(x, *, into=None):
        return unary(name, x, into=into)

    f.__name__ = name
    f.__doc__ = (f"Elementwise {what}; kittens::{name}. Also `x.{name}()`. "
                 f"`into=` writes an existing value instead of making one.")
    return f


exp = _unary("exp", "e**x" + "." + _ANY)
exp2 = _unary("exp2", "2**x (the hardware's v_exp_f32; exp(x) = exp2(x * log2(e)))."
              + _ANY)
log = _unary("log", "natural logarithm." + _ANY)
log2 = _unary("log2", "base-2 logarithm." + _ANY)
abs = _unary("abs", "|x|." + _ANY)  # noqa: A001
relu = _unary("relu", "max(x, 0)." + _ANY)
gelu = _unary("gelu", "GELU, the tanh approximation: "
              "x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x**3))), which is "
              "`F.gelu(x, approximate=\"tanh\")`, not the erf form torch defaults "
              "to (the two differ by up to 4.7e-4)." + _FP32)
dgelu = _unary("dgelu", "derivative of `gelu` (the tanh form), for a backward pass."
               + _FP32)
sqrt = _unary("sqrt", "square root (one v_sqrt_f32)." + _FP32)
rsqrt = _unary("rsqrt", "1/sqrt(x) (one v_rsq_f32, ~1 ulp)." + _FP32)
silu = _unary("silu", "SiLU, x * sigmoid(x) = x / (1 + exp(-x)), through the "
              "hardware exp2 and reciprocal: `F.silu`." + _FP32)


def _binary(name, op, sym):
    def f(a, b, *, into=None):
        return binary(op, a, b, into=into)

    f.__name__ = name
    bcast = (" or, for a tile `a`, a vector off a tile of the same shape "
             "(a col_vec applies along rows, a row_vec along columns)"
             if op in ("add", "sub", "mul", "div") else "")
    f.__doc__ = (f"Elementwise {sym}: what the operator does, with `into=` "
                 f"naming the value it writes (which may be an operand). `b` is "
                 f"a value of `a`'s type, a Python number, a device `Int`{bcast}.")
    return f


add = _binary("add", "add", "a + b")
sub = _binary("sub", "sub", "a - b")
mul = _binary("mul", "mul", "a * b")
div = _binary("div", "div", "a / b")
maximum = _binary("maximum", "max", "max(a, b) (`hk.max(a, b)`, `a.maximum(b)`)")
minimum = _binary("minimum", "min", "min(a, b) (`hk.min(a, b)`, `a.minimum(b)`)")


__all__ = ["exp", "exp2", "log", "log2", "abs", "relu", "gelu", "dgelu",
           "sqrt", "rsqrt", "silu", "add", "sub", "mul", "div", "maximum",
           "minimum"]
