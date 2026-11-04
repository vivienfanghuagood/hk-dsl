"""moe_tune.py -- vLLM's own Triton fused-MoE tuner (benchmarks/kernels/benchmark_moe.py, vLLM 0.31 source)
run in the row's image, with the search space cut to what applies on RDNA3 (WMMA, wave32):
matrix_instr_nonkdim / kpack are CDNA MFMA knobs (fixed 16 / 1), waves_per_eu fixed 0, BLOCK 16-128,
num_warps 2/4/8, GROUP_M 1/8/16. Everything else is the script's: same kernel, same timing, same config
file format and name (E=..,N=..,device_name=..), so vLLM picks the file up through VLLM_TUNED_CONFIG_FOLDER.

    HK_BENCHMARK_MOE=<vllm source>/benchmarks/kernels/benchmark_moe.py \
        python3 moe_tune.py --model M --model-prefix text_config --tp-size 2 \
        --tune --save-dir DIR --batch-size 1 2 4 ...
"""
import importlib.util, os, sys, types

# Ray does not see ROCm GPUs here (available_resources() has no "GPU"), and one GPU is all we use: a
# synchronous in-process stand-in for the four Ray calls the script makes (remote, init,
# available_resources, get), so BenchmarkWorker runs in this process on the visible GPU.
_ray = types.ModuleType("ray")


class _Proxy:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        fn = getattr(self._obj, name)
        return types.SimpleNamespace(remote=lambda *a, **k: fn(*a, **k))


def _remote(*args, **kwargs):
    def wrap(cls):
        return types.SimpleNamespace(remote=lambda *a, **k: _Proxy(cls(*a, **k)), cls=cls)
    if args and isinstance(args[0], type):
        return wrap(args[0])
    return wrap


_ray.remote = _remote
_ray.init = lambda *a, **k: None
_ray.available_resources = lambda: {"GPU": 1}
_ray.get = lambda x: x
_ray.get_gpu_ids = lambda: [0]
_ray.__path__ = []
_exp = types.ModuleType("ray.experimental"); _exp.__path__ = []
_tq = types.ModuleType("ray.experimental.tqdm_ray")
from tqdm import tqdm as _tqdm  # noqa: E402
_tq.tqdm = _tqdm
sys.modules["ray"] = _ray
sys.modules["ray.experimental"] = _exp
sys.modules["ray.experimental.tqdm_ray"] = _tq

# vLLM's benchmarks/kernels/benchmark_moe.py from the source of the vLLM in the image (the wheel does not
# ship benchmarks/): HK_BENCHMARK_MOE names it.
SRC = os.environ["HK_BENCHMARK_MOE"]
spec = importlib.util.spec_from_file_location("benchmark_moe", SRC)
bm = importlib.util.module_from_spec(spec)
sys.modules["benchmark_moe"] = bm
spec.loader.exec_module(bm)


def rdna_space(use_fp16):
    return {"BLOCK_SIZE_M": [16, 32, 64, 128], "BLOCK_SIZE_N": [16, 32, 64, 128], "BLOCK_SIZE_K": [32, 64, 128],
            "GROUP_SIZE_M": [1, 8, 16], "num_warps": [2, 4, 8], "num_stages": [2], "waves_per_eu": [0],
            "matrix_instr_nonkdim": [16], "kpack": [1]}


bm.get_rocm_tuning_space = rdna_space
if __name__ == "__main__":
    parser = bm.FlexibleArgumentParser() if hasattr(bm, "FlexibleArgumentParser") else None
    sys.argv[0] = SRC
    # re-run the script's own __main__ block with our patched module
    src = open(SRC).read()
    main_block = src[src.index('if __name__ == "__main__":'):]
    exec(compile(main_block, SRC, "exec"), bm.__dict__ | {"__name__": "__main__"})
