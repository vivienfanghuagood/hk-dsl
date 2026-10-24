"""Post-import patches for E1's SGLang-Diffusion runs, applied in every process (workers too).

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
