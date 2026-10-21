"""hk's decode GEMV (hk.ops.decode_gemm.matmul) where it takes the shape, then
the residual add in torch: the hk.ops route an expert picks at small M on
gfx1100, where hk.ops.matmul does not route there by itself."""

from hk.ops import decode_gemm

KIND = "hk_ops"
LABEL = "hk.ops.decode_gemm.matmul + torch residual add (small M only)"


def supports(x, weight, residual):
    m, k = x.shape
    n = weight.shape[0]
    return True if decode_gemm.fits(m, n, k) else f"decode GEMM does not take M={m}, N={n}, K={k}"


def solve(x, weight, residual):
    return decode_gemm.matmul(x, weight).add_(residual)
