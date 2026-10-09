"""Compiling and loading: hipcc, the cache, the spill gate."""

from . import arch, compile, module, resources
from .compile import Build, CompileError, build, cache_dir, include_dir
from .resources import KernelResources, ResourceError

__all__ = [
    "arch", "compile", "module", "resources", "warm",
    "build", "Build", "CompileError", "ResourceError", "KernelResources",
    "cache_dir", "include_dir", "warm_cache", "WarmResult",
]


def __getattr__(name):
    """Load `warm` on demand.

    Importing it eagerly would pull `concurrent.futures` and `argparse` into
    every `import hk`, and -- the reason this is a function and not a line --
    would make `python3 -m hk.runtime.warm` print a RuntimeWarning: the package
    __init__ runs first, puts the module in sys.modules, and runpy then notices
    it is about to execute a module that is already there.
    """
    if name in ("warm", "warm_cache", "WarmResult"):
        # Not `from . import warm`: that goes through _handle_fromlist, which
        # getattr's the package, which lands back here.
        import importlib
        mod = importlib.import_module(".warm", __name__)
        return mod if name == "warm" else getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
