"""Post-import patches for E1's SGLang-Diffusion runs, applied in every process (workers too).

E1_NARROW_H3_GATE=1 (default): sglang-radeon's H3 platform shim scoped to the H3 denoising
module instead of the global platform (see _narrow_h3_gate); this is the real cause of the
CUDA-only kernels being picked on gfx1100.

On PYTHONPATH. E1_ROCM_FIX=1: Qwen-Image 2.1's DiT asks `current_platform.is_cuda()`, which is
true on ROCm, and then calls fused Triton/CUDA kernels whose validators refuse HIP
(`torch.version.hip is None`) -- every request fails with "invalid input for
qknorm_complex_rope". On HIP the module's `_is_cuda` is set False: the torch paths run.
hk's own patch comes from `python -m hk.integration sglang-diffusion install` + HK_SGLD=1.
"""
import importlib.abc
import importlib.util
import os
import sys


def _qwen21(mod):
    import torch
    if torch.version.hip and os.environ.get("E1_ROCM_FIX", "1") != "0":
        mod._is_cuda = False


if os.environ.get("E1_SDPA_MATH", "0") == "1":
    # A reference arm: torch's math SDPA (fp32 softmax), no fused kernel.
    try:
        import torch
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

PATCHES = {"sglang.multimodal_gen.runtime.models.dits.qwen_image21": _qwen21}


def _narrow_h3_gate():
    """sglang-radeon 0.1.5's h3_denoise_platform_gate, scoped to the module
    that needs it. The original makes the *global* platform's is_cuda() return
    True on HIP, so every `is_cuda()`-gated CUDA kernel in SGLang is chosen on
    gfx1100 -- among them Triton kernels with inline PTX (mul.rn.f32), whose
    compile kills the worker ("couldn't allocate output register for
    constraint 'f'") and leaves the parent waiting forever. Here only the
    MiniMax-H3 denoising module sees a platform that says CUDA."""
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (  # noqa: E501
            denoising,
        )
    except Exception:
        return "skipped (MiniMax-H3 stages not available)"
    platform = denoising.current_platform
    if getattr(platform, "_e1_proxy", False):
        return "already applied"
    if not platform.is_hip():
        return "not needed (not a HIP platform)"

    class _CudaGateProxy:
        _e1_proxy = True

        def __getattr__(self, name):
            return getattr(platform, name)

        @staticmethod
        def is_cuda():
            return True

    denoising.current_platform = _CudaGateProxy()
    return "applied (narrow: the MiniMax-H3 denoising module only)"


def _radeon(mod):
    if os.environ.get("E1_NARROW_H3_GATE", "1") == "0":
        return
    mod._SHIMS = tuple((n, _narrow_h3_gate if n == "h3_denoise_platform_gate" else f, d)
                       for n, f, d in mod._SHIMS)


PATCHES["sglang_radeon_rdna3.compat"] = _radeon


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _name=name):
            _orig(module)
            PATCHES[_name](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())

# sglang-radeon's .pth bootstrap may already have imported its compat module (before this
# file runs); its shims are applied only at the first `import sglang`, so patching the
# registry now is still in time.
if "sglang_radeon_rdna3.compat" in sys.modules:
    _radeon(sys.modules["sglang_radeon_rdna3.compat"])
