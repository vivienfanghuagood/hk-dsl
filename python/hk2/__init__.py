"""hk2: a tile-level kernel language for AMD GPUs, CDNA and RDNA from one source.

    import hk2 as hk

    @hk.jit
    def add(x, y, out, BLOCK: hk.constexpr = 1024):
        i = hk.program_id(0) * BLOCK
        hk.store(out, i, hk.load(x, i, BLOCK) + hk.load(y, i, BLOCK))

    add[(hk.cdiv(n, 1024),)](x, y, out)

Pure Python: kernels compile with the ROCm installation's own LLVM, no native build.
"""
__version__ = "0.0.1.dev0"

from .types import (DType, float32, float16, bfloat16, int32, int16, int8, uint8, int1,  # noqa: F401
                    f32, f16, bf16, i32, i16, i8, u8, i1)
from .language import (program_id, zeros, full, arange, load, store, dot, sum, max, min, exp, exp2,  # noqa: F401
                       log, log2, sqrt, rsqrt, abs, maximum, minimum, where, cdiv, static_range, constexpr, print,
                       Tile, Scalar, TensorArg, HKError)
from .jit import jit, func, Kernel  # noqa: F401
from .arch import ARCHES, get_arch  # noqa: F401
