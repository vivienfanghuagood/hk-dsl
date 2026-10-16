"""HipKittens' own CDNA4 kernels as hk ops: the bf16 GEMM and GQA attention.

hk's native GEMM and attention do not build for gfx950 yet
(docs/design/cdna.md, 3.1). HipKittens ships hand-tuned MI355X kernels for
both. This builds them, unmodified, through hk's compile cache and spill gate,
launches them on torch's stream, and puts them behind `engine.Op`:

* `gemm_bf16(a, b)` is C = a @ b.T, with a (M, K) and b (N, K) both
  K-contiguous, which is F.linear's layout. The kernel is
  kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp: a 256x256 output
  tile per workgroup of 8 waves, 64-wide K steps taken two a trip, fp32
  accumulation, a bf16 result.
* `gqa_fwd(q, k, v, causal=)` is softmax(q k^T / sqrt(D)) v, with H / H_KV
  query heads per KV head. q is (B, N, H, D) and k, v are (B, N, H_KV, D),
  the flash-attn layout. The kernels are kernels/cdna4/attn/gqa/kernel.cpp
  and gqa_causal/kernel.cpp, plus their _d64 twins.

**Shapes.** Each function takes exactly the shapes its kernel computes
correctly. Anything else raises `Unsupported`, a ValueError that names the
constraint. Nothing is padded. The constraints come from reading the kernels:

* GEMM: M and N must be multiples of 256. The grid is (M/256)(N/256) whole
  tiles, and the C store is a plain pointer write, so a partial tile would
  write past the end of C. K must be a multiple of 128 and at least 128: the
  main loop takes two 64-wide steps a trip and the epilogue takes the last
  two. M*K and N*K must stay under 2**30, because the descriptors are sized
  as a 32-bit int count of bytes.
* GQA: D must be 64 or 128. N must be a multiple of 256: a workgroup is 8
  waves of 32 query rows, with no predicate. The query and key lengths must be
  equal (no KV cache); the causal mask is top-left, which is torch's
  `is_causal` at equal lengths. H must be a multiple of H_KV. The softmax
  scale is 1/sqrt(D), compiled in. B*N*H*D must stay under 2**31: the q and o
  descriptors are a 32-bit byte count, computed as an int.
* bf16, contiguous, 16-byte-aligned tensors on a gfx950 device.

**One GQA build per (H, H_KV, N, D, causal).** The kernel takes
ATTN_H/H_KV/N/D as compile-time macros. B is not one: upstream's ATTN_B is
only the grid's z, and the launch here computes that grid from the batch it
is handed, so every batch size shares one build. The first call at a new
configuration runs hipcc, about 5 s. hk's content-addressed cache makes that a
load in every later process, and the loaded modules are kept for the life of
this one. `gqa_build(...)` pays the compile ahead of time.

**Launch.** Upstream's `dispatch_micro` launches on the null stream and sets
the dynamic-LDS attribute every call. A CUDA graph cannot capture the null
stream, and vLLM's streams are not ordered against it. So the entry point here
is hk's own: it is appended after the verbatim upstream source, takes pointers
and torch's current stream, and sets the attribute once per device. The
tensor checks happen in Python, before the call. The C++ glue lives in this
file rather than in include/, so the cache key covers it through the source
text, and editing it does not change the include-tree fingerprint that every
other kernel is keyed on.

**gfx950 only.** These kernels use the 16x16x32 MFMA, 160 KB of LDS and
buffer-load-to-LDS. Asking for another arch raises `WrongArch`; so does a
launch on a device that is not gfx950. A mismatched code object segfaults
instead of failing (cdna.md 3.2, risk 6).

**Checked on the device before first use.** Whether these kernels compute the
right answer depends on which hipcc built them, and the spill gate cannot see
that. Measured on an MI355X, with HipKittens 49fa4a38 untouched:

* The GEMM is correct when built by ROCm 7.2.3 (clang 22). Built by ROCm 7.1
  (clang 20), it builds clean, passes the gate, and is wrong: rows 128-191 of
  every 256-row tile are wrong, at every shape. Those rows are `C_accum[1][*]`
  of the four waves that start one barrier phase ahead. Upstream's own
  `dispatch_micro`, built the same way, does the same.
* The attention kernels do not compile with clang 22: it finds `zero`,
  `copy`, `mul` and others ambiguous between include/cdna4's rt, art and rv
  overloads. They compile and are correct with ROCm 7.1.

So the first launch of each build on each device runs it once on seeded
inputs and compares against an fp32 torch reference: a 512x512x512 GEMM, or
the compiled attention shape at B=1 checked on three query blocks. A kernel
that fails the check raises `Miscompiled` and is never used. `HK_HIPCC` picks
the toolchain. That first launch synchronizes, so it must not happen under
CUDA-graph capture; it raises `GraphCaptureError` if it does.

**Engine ops.** `hk_cdna::gemm_bf16` and `hk_cdna::gqa_fwd` are torch custom
ops with fake impls. Each takes the HipKittens kernel where the shape fits it
and torch otherwise (F.linear, scaled_dot_product_attention), and counts the
path taken. A kernel that cannot be had at all -- it does not compile under
this hipcc, or it failed its check -- also goes to torch, with one warning
that says why. Both ops are graph="safe": every launch is on torch's stream,
and the route depends on shapes alone. Run each new shape once before
capturing it, so that the build, the load and the check happen outside
capture; vLLM's warm-up runs already do this. Which linear layers *should*
take the GEMM is a separate question, answered by measurement:
hk/tuned/routes_vllm_linear_gfx950.json, the "hipkittens" path of
`hk.integration.vllm_gemm`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from ..integration import engine

ARCH = "gfx950"

#: Overrides where the HipKittens kernel sources (kernels/cdna4) are read from.
#: A checkout has them in the submodule; a wheel carries only the headers.
KERNELS_ENV = "HK_HIPKITTENS_KERNELS"

GEMM_SRC = "gemm/bf16fp32/256_256_64_32_with16x32.cpp"
GQA_SRC = {
    (False, 128): "attn/gqa/kernel.cpp",
    (False, 64): "attn/gqa/kernel_d64.cpp",
    (True, 128): "attn/gqa_causal/kernel.cpp",
    (True, 64): "attn/gqa_causal/kernel_d64.cpp",
}

#: The GEMM's output tile, and its K granularity (two 64-wide steps a trip).
GEMM_BLOCK = 256
GEMM_K_MULTIPLE = 128
#: A GQA workgroup's query rows: 8 waves x 32.
GQA_BLOCK = 256
HEAD_DIMS = (64, 128)
#: The GEMM sizes its operand descriptors as an int count of bytes (M * K * 2).
GEMM_MAX_ELEMS = 2 ** 30
#: The attention kernels' q/o descriptors are a uint32 count of bytes, from an
#: int product of the extents.
GQA_MAX_ELEMS = 2 ** 31

#: kernels/cdna4/attn/gqa*/Makefile; the GEMM's Makefile adds nothing to common.mk.
GQA_FLAGS = ("-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math")


class Unsupported(ValueError):
    """A shape or tensor the HipKittens kernel would compute wrongly. Refused."""


class WrongArch(RuntimeError):
    """A HipKittens CDNA4 kernel asked for, or launched on, an arch it is not for."""


class Unavailable(RuntimeError):
    """A kernel that cannot be had under this toolchain: it did not build, or it
    built and computed the wrong answer. Remembered, so it is not retried."""


class Miscompiled(Unavailable):
    """A kernel that built and failed its on-device check against fp32 torch."""


def check_arch(arch: str) -> None:
    if arch != ARCH:
        raise WrongArch(
            f"hk.ops.hipkittens wraps HipKittens' CDNA4 kernels, which are written "
            f"for gfx950 (MI350X/MI355X) only -- the 16x16x32 MFMA, 160 KB LDS and "
            f"buffer-load-to-LDS they are built from do not exist on {arch}. "
            f"There is no fallback build for {arch}; use torch, or hk's own kernels.")


# ---------------------------------------------------------------- shapes


def gemm_unsupported(m: int, n: int, k: int) -> Optional[str]:
    """Why the GEMM cannot compute C (m, n) = A (m, k) @ B (n, k).T, or None."""
    if m < 1 or n < 1 or k < 1:
        return f"M={m}, N={n}, K={k}: an empty GEMM"
    if m % GEMM_BLOCK or n % GEMM_BLOCK:
        return (f"M={m}, N={n}: both must be multiples of {GEMM_BLOCK}; the kernel "
                f"computes whole 256x256 tiles and its C store is unpredicated, so a "
                f"partial tile would write past the end of C")
    if k % GEMM_K_MULTIPLE:
        return (f"K={k}: must be a multiple of {GEMM_K_MULTIPLE}; the main loop "
                f"takes two 64-wide K steps a trip and the epilogue the last two")
    if m * k >= GEMM_MAX_ELEMS or n * k >= GEMM_MAX_ELEMS:
        return (f"M*K={m * k}, N*K={n * k}: each must be under 2**30; the buffer "
                f"descriptors are sized as an int count of bytes")
    return None


def gemm_fits(m: int, n: int, k: int) -> bool:
    return gemm_unsupported(m, n, k) is None


def gqa_unsupported(b: int, n: int, h: int, h_kv: int, d: int,
                    n_kv: Optional[int] = None) -> Optional[str]:
    """Why GQA forward cannot take q (b, n, h, d), k/v (b, n_kv, h_kv, d), or None."""
    if min(b, n, h, h_kv, d) < 1:
        return f"B={b}, N={n}, H={h}, H_KV={h_kv}, D={d}: an empty attention"
    if d not in HEAD_DIMS:
        return f"D={d}: HipKittens ships GQA forward for head dims {HEAD_DIMS} only"
    if n_kv is not None and n_kv != n:
        return (f"N_q={n}, N_kv={n_kv}: the kernel takes one sequence length for "
                f"queries and keys (no KV cache)")
    if n % GQA_BLOCK:
        return (f"N={n}: must be a multiple of {GQA_BLOCK}; a workgroup is 8 waves of "
                f"32 query rows with no predicate on the last one")
    if h % h_kv:
        return f"H={h}, H_KV={h_kv}: query heads must be a multiple of KV heads"
    if b * n * h * d >= GQA_MAX_ELEMS:
        return (f"B*N*H*D={b * n * h * d}: must be under 2**31; the q and o "
                f"descriptors are a 32-bit count of bytes")
    return None


def gqa_fits(b: int, n: int, h: int, h_kv: int, d: int,
             n_kv: Optional[int] = None) -> bool:
    return gqa_unsupported(b, n, h, h_kv, d, n_kv) is None


# ---------------------------------------------------------------- sources

_COMMON_GLUE = r"""
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

// The dynamic-LDS attribute, once per device, where upstream's dispatch_micro
// sets it before every launch. Internal linkage: two hk modules' flags must
// never resolve to one symbol, or the second kernel launches without it.
template <typename F>
void hk_allow_lds(F *fn, int bytes) {
    static bool done[64] = {};
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) dev = -1;
    if (dev >= 0 && dev < 64 && done[dev]) return;
    const hipError_t e = hipFuncSetAttribute(
        reinterpret_cast<const void *>(fn),
        hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e != hipSuccess)
        throw std::runtime_error(std::string("hk.ops.hipkittens: hipFuncSetAttribute: ")
                                 + hipGetErrorString(e));
    if (dev >= 0 && dev < 64) done[dev] = true;
}

void hk_check_launch(const char *what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess)
        throw std::runtime_error(std::string("hk.ops.hipkittens: ") + what
                                 + " failed to launch: " + hipGetErrorString(e));
}

}  // namespace
"""

_GEMM_GLUE = r"""
namespace {

void hk_gemm_launch(std::uintptr_t a, std::uintptr_t b, std::uintptr_t c,
                    int m, int n, int k, std::uintptr_t stream) {
    micro_globals g{
        make_gl<_gl_A>(a, 1, 1, m, k),
        make_gl<_gl_B>(b, 1, 1, n, k),
        make_gl<_gl_C>(c, 1, 1, m, n),
        reinterpret_cast<hipStream_t>(stream)};
    const int lds = static_cast<int>(g.dynamic_shared_memory());
    hk_allow_lds(micro_tk, lds);
    micro_tk<<<g.grid(), g.block(), lds, g.stream>>>(g, g.M, g.N, g.K);
    hk_check_launch("gemm_bf16");
}

}  // namespace

PYBIND11_MODULE(HK_MODULE_NAME, mod) {
    mod.doc() = "HipKittens CDNA4 bf16 GEMM, C = A @ B^T, on the caller's stream";
    mod.def("launch", &hk_gemm_launch);
}
"""

_GQA_GLUE = r"""
namespace {

// ATTN_B is only upstream grid()'s z, so the batch is taken at run time and
// every batch size shares this build.
void hk_gqa_launch(std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
                   std::uintptr_t o, std::uintptr_t lse, int batch,
                   std::uintptr_t stream) {
    attn_globals<ATTN_D> g{
        make_gl<_gl_QKVO>(q, batch, ATTN_N, ATTN_H, ATTN_D),
        make_gl<_gl_QKVO>(k, batch, ATTN_N, ATTN_H_KV, ATTN_D),
        make_gl<_gl_QKVO>(v, batch, ATTN_N, ATTN_H_KV, ATTN_D),
        make_gl<_gl_QKVO>(o, batch, ATTN_N, ATTN_H, ATTN_D),
        make_gl<gl<float, -1, -1, -1, -1>>(lse, batch, ATTN_H, 1, ATTN_N),
        reinterpret_cast<hipStream_t>(stream)};
    const int lds = static_cast<int>(g.dynamic_shared_memory());
    hk_allow_lds(attend_ker<ATTN_D>, lds);
    const dim3 grid(ATTN_H, (ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS, batch);
    attend_ker<ATTN_D><<<grid, g.block(), lds, g.stream>>>(g);
    hk_check_launch("gqa_fwd");
}

}  // namespace

PYBIND11_MODULE(HK_MODULE_NAME, mod) {
    mod.doc() = "HipKittens CDNA4 GQA attention forward, on the caller's stream";
    mod.def("launch", &hk_gqa_launch);
}
"""


def kernels_dir() -> Path:
    """HipKittens' kernels/cdna4: `$HK_HIPKITTENS_KERNELS`, else the submodule."""
    env = os.environ.get(KERNELS_ENV)
    d = Path(env) if env else (
        Path(__file__).resolve().parents[3] / "third_party" / "HipKittens" / "kernels" / "cdna4")
    if not (d / GEMM_SRC).is_file():
        raise FileNotFoundError(
            f"HipKittens' CDNA4 kernel sources are not at {d}. A checkout has them "
            f"in third_party/HipKittens (git submodule update --init); a wheel "
            f"ships only the headers -- point ${KERNELS_ENV} at a HipKittens "
            f"checkout's kernels/cdna4.")
    return d


def _assemble(rel: str, defines: Sequence[Tuple[str, object]], glue: str) -> str:
    text = (kernels_dir() / rel).read_text()
    head = [
        f"// hk.ops.hipkittens: HipKittens kernels/cdna4/{rel}, verbatim between the",
        "// markers. Its own PYBIND11_MODULE is renamed out of the way; hk's entry",
        "// point, which launches on the caller's stream, follows the source.",
        "#define tk_kernel hk_hipkittens_upstream_module",
        *(f"#define {k} {v}" for k, v in defines),
        f"// ---- begin kernels/cdna4/{rel} ----",
    ]
    tail = [f"// ---- end kernels/cdna4/{rel} ----", "#undef tk_kernel"]
    return "\n".join(head) + "\n" + text + "\n" + "\n".join(tail) + _COMMON_GLUE + glue


def gemm_source() -> str:
    return _assemble(GEMM_SRC, (), _GEMM_GLUE)


def gqa_source(h: int, h_kv: int, n: int, d: int = 128, causal: bool = False) -> str:
    why = gqa_unsupported(1, n, h, h_kv, d)
    if why:
        raise Unsupported(why)
    defines = (("ATTN_H", h), ("ATTN_H_KV", h_kv), ("ATTN_N", n), ("ATTN_D", d))
    return _assemble(GQA_SRC[(bool(causal), d)], defines, _GQA_GLUE)


# ---------------------------------------------------------------- builds


def gemm_build(arch: str = ARCH, **kw):
    """hk.runtime.compile.build of the GEMM: cached, and spill-gated."""
    check_arch(arch)
    from ..runtime.compile import build  # noqa: PLC0415

    return build(gemm_source(), arch, name="hipkittens_gemm_bf16", **kw)


def gqa_build(h: int, h_kv: int, n: int, d: int = 128, causal: bool = False,
              arch: str = ARCH, **kw):
    """The GQA forward build for one (H, H_KV, N, D, causal); any batch size."""
    check_arch(arch)
    from ..runtime.compile import build  # noqa: PLC0415

    flags = (*GQA_FLAGS, *kw.pop("extra_flags", ()))
    name = f"hipkittens_gqa{'_causal' if causal else ''}_h{h}_kv{h_kv}_n{n}_d{d}"
    return build(gqa_source(h, h_kv, n, d, causal), arch, name=name,
                 extra_flags=flags, **kw)


#: Tolerance of the on-device check, max|got - want| / max|want| against fp32
#: (the LSE: max absolute error). The correct builds measure 5e-3; the
#: miscompiled GEMM 0.1 to 0.9.
CHECK_TOL = 2e-2

_MODULES: Dict[tuple, object] = {}
_BROKEN: Dict[tuple, Unavailable] = {}
_CHECKED: Dict[tuple, Optional[Miscompiled]] = {}
_HIPCC_VERSION: list = []


def hipcc_version() -> str:
    """The first line of `hipcc --version` for the hipcc builds use (`HK_HIPCC`)."""
    if not _HIPCC_VERSION:
        import subprocess  # noqa: PLC0415

        from ..runtime.compile import hipcc  # noqa: PLC0415

        try:
            out = subprocess.run([hipcc(), "--version"], capture_output=True,
                                 text=True, timeout=120).stdout
        except Exception as e:  # noqa: BLE001
            out = f"(hipcc --version failed: {e})"
        _HIPCC_VERSION.append(next((ln for ln in out.splitlines() if ln.strip()), "?").strip())
    return _HIPCC_VERSION[0]


def _ready(key: tuple, what: str, make, check, device):
    """The module for `key`, built, loaded, and checked on `device` once.

    Raises `Unavailable` -- remembered, so a failing hipcc runs once -- when it
    does not build or fails its check."""
    if key in _BROKEN:
        raise _BROKEN[key]
    mod = _MODULES.get(key)
    if mod is None:
        from ..runtime.compile import CompileError  # noqa: PLC0415
        from ..runtime.module import load  # noqa: PLC0415
        from ..runtime.resources import ResourceError  # noqa: PLC0415

        try:
            b = make()
        except (CompileError, ResourceError) as e:
            first = next((ln for ln in str(e).splitlines() if "error" in ln), str(e).splitlines()[0])
            _BROKEN[key] = Unavailable(
                f"{what} does not build under {hipcc_version()}: {first.strip()}")
            raise _BROKEN[key] from e
        mod = _MODULES[key] = load(b.so_path, b.module_name)
    ck = (key, _index(device))
    if ck not in _CHECKED:
        if engine.capturing():
            raise engine.GraphCaptureError(
                f"{what}: the first launch on a device checks the kernel against fp32 "
                f"torch, which synchronizes; run this shape once before capturing it")
        why = check(mod, device)
        _CHECKED[ck] = None if why is None else Miscompiled(
            f"{what}, built by {hipcc_version()}, computes the wrong answer on this "
            f"device ({why}) and will not be used. Which hipcc builds which kernel "
            f"right is in hk.ops.hipkittens' docstring; HK_HIPCC picks one.")
    if _CHECKED[ck] is not None:
        raise _CHECKED[ck]
    return mod


def _rel(got, want) -> float:
    return ((got.float() - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


def _gemm_check(mod, device) -> Optional[str]:
    """512^3 on seeded inputs: the loop, the epilogue, both wave rows, 4 tiles."""
    import torch  # noqa: PLC0415

    m = n = k = 512
    g = torch.Generator(device=device).manual_seed(20261016)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g) * k ** -0.5
    c = torch.zeros(m, n, device=device, dtype=torch.bfloat16)
    _launch(mod, device, a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k)
    err = _rel(c, a.float() @ b.float().t())
    return None if err < CHECK_TOL else f"rel err {err:.3g} at {m}x{n}x{k}"


def _gqa_check(mod, device, h, h_kv, n, d, causal) -> Optional[str]:
    """The compiled shape at B=1, compared on the first, a middle and the last
    32-row query block, over every head."""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415

    g = torch.Generator(device=device).manual_seed(20261016)
    q = torch.randn(1, n, h, d, device=device, dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, n, h_kv, d, device=device, dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, n, h_kv, d, device=device, dtype=torch.bfloat16, generator=g)
    o = torch.empty_like(q)
    lse = torch.empty(1, h, 1, n, device=device, dtype=torch.float32)
    _launch(mod, device, q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
            lse.data_ptr(), 1)
    rows = torch.cat([torch.arange(s, s + 32, device=device)
                      for s in (0, (n // 2) // 32 * 32, n - 32)])
    kf = k[0].float().repeat_interleave(h // h_kv, dim=1)
    vf = v[0].float().repeat_interleave(h // h_kv, dim=1)
    s = torch.einsum("rhd,nhd->hrn", q[0, rows].float(), kf) / math.sqrt(d)
    if causal:
        s = s.masked_fill(torch.arange(n, device=device) > rows[:, None], float("-inf"))
    want_o = torch.einsum("hrn,nhd->rhd", torch.softmax(s, -1), vf)
    err = _rel(o[0, rows], want_o)
    lerr = (lse[0, :, 0, rows] - torch.logsumexp(s, -1)).abs().max().item()
    if err < CHECK_TOL and lerr < CHECK_TOL:
        return None
    return f"o rel err {err:.3g}, lse abs err {lerr:.3g} at H={h} H_KV={h_kv} N={n} D={d}"


# ---------------------------------------------------------------- launch

_DEVICE_ARCH: Dict[int, str] = {}
_RAW_STREAM: list = []


def _index(device) -> int:
    import torch  # noqa: PLC0415

    return device.index if device.index is not None else torch.cuda.current_device()


def device_arch(device) -> str:
    """The gfx name of a torch device, read once per index."""
    idx = _index(device)
    a = _DEVICE_ARCH.get(idx)
    if a is None:
        import torch  # noqa: PLC0415

        name = torch.cuda.get_device_properties(idx).gcnArchName or ""
        a = _DEVICE_ARCH[idx] = name.split(":")[0]
    return a


def _stream(idx: int) -> int:
    if not _RAW_STREAM:
        import torch  # noqa: PLC0415

        _RAW_STREAM.append(getattr(torch._C, "_cuda_getCurrentRawStream", None)
                           or (lambda i: torch.cuda.current_stream(i).cuda_stream))
    return _RAW_STREAM[0](idx)


def _tensor_unsupported(dtype, *ts) -> Optional[str]:
    dev = ts[0].device
    for t in ts:
        if not t.is_cuda:
            return "tensors must be on the GPU"
        if t.device != dev:
            return f"tensors on {dev} and {t.device}"
        if t.dtype != dtype:
            return f"dtype {t.dtype}; the kernel takes {dtype}"
        if not t.is_contiguous():
            return "a tensor is not contiguous; call .contiguous()"
        if t.data_ptr() % 16:
            return "a tensor's data is not 16-byte aligned (the loads are 128-bit)"
    return None


def _check_device(device) -> None:
    a = device_arch(device)
    if a != ARCH:
        raise WrongArch(f"hk.ops.hipkittens: {device} is {a}; these kernels run on {ARCH} only")


def _launch(mod, device, *args) -> None:
    import torch  # noqa: PLC0415

    idx = _index(device)
    if idx == torch.cuda.current_device():
        mod.launch(*args, _stream(idx))
    else:
        with torch.cuda.device(idx):
            mod.launch(*args, _stream(idx))


def gemm_takes(a, b) -> bool:
    """Whether `gemm_bf16(a, b)` would take these tensors rather than refuse
    them. Shapes and tensors only: not the arch, the build or the check."""
    import torch  # noqa: PLC0415

    return (a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[1]
            and gemm_fits(a.shape[0], b.shape[0], a.shape[1])
            and _tensor_unsupported(torch.bfloat16, a, b) is None)


def gemm_bf16(a, b, out=None):
    """out (M, N) = a (M, K) @ b (N, K).T, bf16, on HipKittens' CDNA4 GEMM.

    Raises `Unsupported` for a shape or tensor the kernel does not take (see
    the module docstring), `WrongArch` off gfx950, and `Unavailable` when the
    kernel did not build or failed its check under this hipcc. `out`, if
    given, must be a contiguous (M, N) bf16 tensor that aliases neither input.
    """
    import torch  # noqa: PLC0415

    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise Unsupported(f"a {tuple(a.shape)}, b {tuple(b.shape)}: want a (M, K), b (N, K)")
    m, k = a.shape
    n = b.shape[0]
    why = gemm_unsupported(m, n, k)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != (m, n):
        why = why or f"out {tuple(out.shape)}; want ({m}, {n})"
    why = why or _tensor_unsupported(torch.bfloat16, a, b, out)
    if why:
        raise Unsupported(why)
    _check_device(a.device)
    mod = _ready(("gemm",), "HipKittens' CDNA4 GEMM", gemm_build, _gemm_check, a.device)
    _launch(mod, a.device, a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k)
    return out


def gqa_takes(q, k, v) -> bool:
    """Whether `gqa_fwd(q, k, v)` would take these tensors rather than refuse
    them. Shapes and tensors only: not the arch, the build or the check."""
    import torch  # noqa: PLC0415

    if q.dim() != 4 or k.dim() != 4 or tuple(k.shape) != tuple(v.shape):
        return False
    b, n, h, d = q.shape
    return (k.shape[0] == b and k.shape[3] == d
            and gqa_fits(b, n, h, k.shape[2], d, k.shape[1])
            and _tensor_unsupported(torch.bfloat16, q, k, v) is None)


def gqa_fwd(q, k, v, *, causal: bool = False, return_lse: bool = False):
    """GQA attention forward on HipKittens' CDNA4 kernel.

    q (B, N, H, D), k and v (B, N, H_KV, D), bf16, contiguous; the scale is
    1/sqrt(D). Returns o (B, N, H, D) bf16 and, with `return_lse`, the
    natural-log log-sum-exp of each query row, (B, H, N) fp32. The first
    call at a new (H, H_KV, N, D, causal) compiles, and the first on each
    device checks the build (see the module docstring). Raises like
    `gemm_bf16`.
    """
    import torch  # noqa: PLC0415

    if q.dim() != 4 or k.dim() != 4 or tuple(k.shape) != tuple(v.shape):
        raise Unsupported(f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: "
                          f"want q (B, N, H, D) and k, v (B, N, H_KV, D)")
    b, n, h, d = q.shape
    h_kv = k.shape[2]
    why = None
    if k.shape[0] != b or k.shape[3] != d:
        why = f"q {tuple(q.shape)}, k {tuple(k.shape)}: batch and head dim must agree"
    why = why or gqa_unsupported(b, n, h, h_kv, d, k.shape[1])
    why = why or _tensor_unsupported(torch.bfloat16, q, k, v)
    if why:
        raise Unsupported(why)
    _check_device(q.device)
    causal = bool(causal)
    mod = _ready(("gqa", h, h_kv, n, d, causal),
                 f"HipKittens' CDNA4 GQA forward ({'causal, ' if causal else ''}H={h} "
                 f"H_KV={h_kv} N={n} D={d})",
                 lambda: gqa_build(h, h_kv, n, d, causal),
                 lambda mod, dev: _gqa_check(mod, dev, h, h_kv, n, d, causal), q.device)
    o = torch.empty_like(q)
    lse = torch.empty((b, h, 1, n), device=q.device, dtype=torch.float32)
    _launch(mod, q.device, q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
            lse.data_ptr(), b)
    return (o, lse.view(b, h, n)) if return_lse else o


# ---------------------------------------------------------------- engine ops


def _gemm_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[0]))


def _gqa_fake(q, k, v, causal=False):
    return q.new_empty(q.shape)


GEMM_OP = engine.Op("hk_cdna::gemm_bf16", "gemm_bf16(Tensor a, Tensor b) -> Tensor",
                    _gemm_fake, paths=("hipkittens", "fallback"), graph="safe")
GQA_OP = engine.Op("hk_cdna::gqa_fwd",
                   "gqa_fwd(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor",
                   _gqa_fake, paths=("hipkittens", "fallback"), graph="safe")

_WARNED: set = set()


def warn_unavailable(where: str, err: Unavailable) -> None:
    """One warning per distinct reason a kernel was passed over for torch."""
    key = (where, str(err))
    if key not in _WARNED:
        import warnings  # noqa: PLC0415

        _WARNED.add(key)
        warnings.warn(f"{where}: falling back to torch -- {err}", RuntimeWarning, stacklevel=3)


def _gemm_impl(a, b):
    if gemm_takes(a, b) and device_arch(a.device) == ARCH:
        try:
            out = gemm_bf16(a, b)
        except Unavailable as e:
            warn_unavailable(GEMM_OP.name, e)
        else:
            GEMM_OP.count("hipkittens")
            return out
    import torch  # noqa: PLC0415

    GEMM_OP.count("fallback")
    return torch.nn.functional.linear(a, b)


def sdpa_bnhd(q, k, v, causal: bool = False):
    """torch's scaled_dot_product_attention on the flash-attn layout: what
    `hk_cdna::gqa_fwd` falls back to."""
    import torch  # noqa: PLC0415

    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal,
        enable_gqa=q.shape[2] != k.shape[2])
    return o.transpose(1, 2).contiguous()


def _gqa_impl(q, k, v, causal=False):
    if gqa_takes(q, k, v) and device_arch(q.device) == ARCH:
        try:
            out = gqa_fwd(q, k, v, causal=causal)
        except Unavailable as e:
            warn_unavailable(GQA_OP.name, e)
        else:
            GQA_OP.count("hipkittens")
            return out
    GQA_OP.count("fallback")
    return sdpa_bnhd(q, k, v, causal)


def register_ops() -> None:
    """Define torch.ops.hk_cdna.gemm_bf16 and .gqa_fwd. Idempotent."""
    GEMM_OP.define(_gemm_impl)
    GQA_OP.define(_gqa_impl)


__all__ = [
    "ARCH", "Unsupported", "WrongArch", "Unavailable", "Miscompiled", "GEMM_OP", "GQA_OP",
    "gemm_bf16", "gemm_build", "gemm_fits", "gemm_source", "gemm_takes", "gemm_unsupported",
    "gqa_fwd", "gqa_build", "gqa_fits", "gqa_source", "gqa_takes", "gqa_unsupported",
    "hipcc_version", "kernels_dir", "register_ops", "sdpa_bnhd", "warn_unavailable",
]
