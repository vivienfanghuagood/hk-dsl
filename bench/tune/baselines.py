"""The baselines a tuned hk GEMM is judged against, per GPU, each a function
fn(a, b, out) -> C for C = a @ b.T (b as (N, K), F.linear's layout).

    torch          F.linear: hipBLASLt's default heuristic (torch on ROCm)
    tunableop      F.linear under PyTorch TunableOp: hipBLASLt (and rocBLAS)
                   solutions searched for the shape at its first call --
                   "hipBLASLt tuned", as far as a user can reach it
    aiter          aiter's tuned GEMM dispatcher (tgemm.mm): its tuned CSVs
                   (asm, hipBLASLt solutions, FlyDSL, Triton, skinny) where it
                   has the shape, else its default; gfx950, vLLM ROCm image
    aiter_triton   aiter's Triton gemm_a16w16 with aiter's per-arch configs
    flydsl         FlyDSL's GEMM: on gfx950 gemm_a16w16_gfx950 as aiter ships
                   it, its default 256x256 half-tile-interleaved schedule; on
                   gfx1100 its RDNA3 GEMM at its dispatcher's tile
                   (hk.integration.flydsl_gemm, FLYDSL_ROOT)
    hipkittens     HipKittens' CDNA4 GEMM (hk.ops.hipkittens), where it takes
                   the shape
    triton         a standard autotuned Triton matmul (bench/tune/triton_gemm)

`available(arch)` says which import here and why the others do not; every
arm is checked against fp32 before it is timed (bench/tune/run.py), and an
arm that is wrong is dropped from "best available", with the reason kept.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F  # noqa: N812

ALL = ("torch", "tunableop", "aiter", "aiter_triton", "flydsl", "hipkittens", "triton")
_TUNABLE = {"on": False}


def _torch(a, b, out):
    if _TUNABLE["on"]:
        torch.cuda.tunable.enable(False)
        _TUNABLE["on"] = False
    return F.linear(a, b)


def _tunableop(a, b, out):
    if not _TUNABLE["on"]:
        torch.cuda.tunable.enable(True)
        _TUNABLE["on"] = True
    return F.linear(a, b)


def reset() -> None:
    """TunableOp off again. It is a process-wide switch: left on after a
    scoring pass, every torch GEMM after it -- the next shape's fp32
    reference, the tuner's F.linear -- would be tuned at its first call, on
    the GPU budget of whatever runs next."""
    if _TUNABLE["on"]:
        torch.cuda.tunable.enable(False)
        _TUNABLE["on"] = False


def _setup_tunableop(path: str) -> None:
    t = torch.cuda.tunable
    t.set_filename(path)
    t.tuning_enable(True)
    t.set_max_tuning_duration(300)       # ms of timing per solution set
    t.set_max_tuning_iterations(30)
    t.enable(False)


def available(arch: str, want=ALL) -> Tuple[Dict[str, Callable], Dict[str, str]]:
    """({name: fn}, {name: why it is not here})."""
    fns: Dict[str, Callable] = {}
    why: Dict[str, str] = {}
    for name in want:
        try:
            if name == "torch":
                fns[name] = _torch
            elif name == "tunableop":
                _setup_tunableop(os.environ.get("HK_TUNE_TUNABLEOP_FILE",
                                                f"/tmp/tunableop_{arch}_%d.csv"))
                fns[name] = _tunableop
            elif name == "aiter":
                if arch != "gfx950":
                    raise RuntimeError("aiter's GEMMs are CDNA's")
                from aiter.tuned_gemm import tgemm  # noqa: PLC0415

                fns[name] = lambda a, b, out: tgemm.mm(a, b, None, None)
            elif name == "aiter_triton":
                if arch != "gfx950":
                    raise RuntimeError("aiter is CDNA's")
                from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16  # noqa: PLC0415

                fns[name] = lambda a, b, out: gemm_a16w16(a, b, dtype=torch.bfloat16, y=out)
            elif name == "flydsl" and arch == "gfx950":
                from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16 as fly  # noqa: PLC0415

                fns[name] = lambda a, b, out: fly(a, b.t(), out=out, layout="nt")
            elif name == "flydsl":
                # FlyDSL's RDNA3 GEMM at the tile its dispatcher picks, as hk's
                # route overlay calls it (hk.integration.flydsl_gemm).
                if not os.environ.get("FLYDSL_ROOT") and os.path.isdir("/workspace/FlyDSL"):
                    os.environ["FLYDSL_ROOT"] = "/workspace/FlyDSL"
                from hk.integration import flydsl_gemm  # noqa: PLC0415

                if not flydsl_gemm.available():
                    raise RuntimeError("hk.integration.flydsl_gemm is not available here")

                def fly_run(a, b, out, fg=flydsl_gemm):
                    if fg.variant(a.shape[0], b.shape[0], a.shape[1]) is None:
                        raise ValueError("no FlyDSL tile divides this shape")
                    return fg.run(a, b, out)
                fns[name] = fly_run
            elif name == "hipkittens":
                if arch != "gfx950":
                    raise RuntimeError("HipKittens' GEMM here is CDNA4's")
                from hk.ops import hipkittens as hkk  # noqa: PLC0415

                def hkk_run(a, b, out, hkk=hkk):
                    if not hkk.gemm_fits(a.shape[0], b.shape[0], a.shape[1]):
                        raise ValueError(hkk.gemm_unsupported(a.shape[0], b.shape[0], a.shape[1]))
                    return hkk.gemm_bf16(a, b, out=out)
                fns[name] = hkk_run
            elif name == "triton":
                from . import triton_gemm  # noqa: PLC0415

                fns[name] = lambda a, b, out: triton_gemm.matmul_abt(a, b, out)
            else:
                raise ValueError(f"no baseline {name}")
        except Exception as e:  # noqa: BLE001 -- a missing library is a fact to record
            why[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
    return fns, why


def versions() -> Dict[str, str]:
    out = {"torch": torch.__version__, "hip": str(torch.version.hip)}
    for mod in ("triton", "aiter", "flydsl", "vllm"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    try:
        import importlib.metadata as md  # noqa: PLC0415

        for dist in ("amd-aiter", "flydsl"):
            try:
                out[dist] = md.version(dist)
            except md.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return out
