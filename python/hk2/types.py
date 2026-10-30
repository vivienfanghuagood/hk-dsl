"""Element types."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DType", "float32", "float16", "bfloat16", "int32", "int16", "int8", "uint8", "int1",
           "f32", "f16", "bf16", "i32", "i16", "i8", "u8", "i1", "from_torch"]


@dataclass(frozen=True)
class DType:
    name: str
    bits: int
    llvm: str
    kind: str  # "f" float, "i" signed int, "u" unsigned int, "b" bool

    @property
    def is_float(self) -> bool:
        return self.kind == "f"

    @property
    def is_int(self) -> bool:
        return self.kind in "iub"

    @property
    def bytes(self) -> int:
        return max(1, self.bits // 8)

    def __repr__(self) -> str:
        return f"hk2.{self.name}"


float32 = f32 = DType("float32", 32, "float", "f")
float16 = f16 = DType("float16", 16, "half", "f")
bfloat16 = bf16 = DType("bfloat16", 16, "bfloat", "f")
int32 = i32 = DType("int32", 32, "i32", "i")
int16 = i16 = DType("int16", 16, "i16", "i")
int8 = i8 = DType("int8", 8, "i8", "i")
uint8 = u8 = DType("uint8", 8, "i8", "u")
int1 = i1 = DType("int1", 1, "i1", "b")

_TORCH = {"torch.float32": f32, "torch.float16": f16, "torch.bfloat16": bf16, "torch.int32": i32,
          "torch.int16": i16, "torch.int8": i8, "torch.uint8": u8, "torch.bool": i1}


def from_torch(dt) -> DType:
    try:
        return _TORCH[str(dt)]
    except KeyError:
        raise TypeError(f"hk2: tensors of {dt} are not supported") from None


def promote(a: DType, b: DType) -> DType:
    """The result type of a binary op: floats win over ints, then the wider one."""
    if a == b:
        return a
    if a.is_float != b.is_float:
        return a if a.is_float else b
    if a.bits != b.bits:
        return a if a.bits > b.bits else b
    # bf16 with f16: compute in f32
    return f32 if a.is_float else a
