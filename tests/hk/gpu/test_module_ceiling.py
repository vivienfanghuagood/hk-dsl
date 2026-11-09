"""A process loads as many hk kernels as it needs (issue #350).

Every kernel is an extension module of its own. They were pybind11 modules, and pybind11's module
setup took pthread thread-specific-storage keys -- four a module with torch 2.10's pybind11, of
1024 a process -- so a process that loaded ~250 kernels aborted in the next one's import ("Fatal
Python error: Aborted", or "could not initialize the TSS key"), on gfx950 in the GPU tier's long
groups. The modules are plain CPython extensions now and take none.

Cheap, so it can run every time: one kernel built once, its .so loaded from 320 paths -- a dlopen
and a module init each, which is where the keys went -- and every copy launched. Its own group in
scripts/gpu_suite.sh: 320 modules in a shared process would be a burden on the tests beside it.
"""

from __future__ import annotations

import ctypes
import shutil

import pytest

import hk
from hk.codegen.scaffold import takes_stream
from hk.runtime import module as rtm

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

COPIES = 320


def _free_keys() -> int:
    """How many more pthread keys this process can create (each made and deleted again)."""
    libc = ctypes.CDLL(None)
    libc.pthread_key_create.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    libc.pthread_key_delete.argtypes = [ctypes.c_uint]
    made, k = [], ctypes.c_uint()
    while len(made) < 2048 and libc.pthread_key_create(ctypes.byref(k), None) == 0:
        made.append(k.value)
    for x in made:
        libc.pthread_key_delete(x)
    return len(made)


@hk.kernel(grid=lambda p: (1, 1, 1))
def fill(o: hk.Tensor[hk.fp32], *, V=0):
    o[hk.tile_coord(0, 0, 0, 0)] = hk.full(hk.RT(hk.fp32, 16, 64), float(V))


def test_a_kernel_module_takes_no_thread_keys(tmp_path):
    b = fill.build(V=7)
    o = torch.zeros(16, 64, device="cuda")
    args = (o, torch.cuda.current_stream().cuda_stream) if takes_stream(fill.arch) else (o,)
    before = _free_keys()
    for i in range(COPIES):
        so = tmp_path / f"copy{i}" / b.so_path.name
        so.parent.mkdir()
        shutil.copy(b.so_path, so)          # another file: another dlopen, another module init
        launch = getattr(rtm.load(so, b.module_name), fill.name)
        o.zero_()
        launch(*args)
        torch.cuda.synchronize()
        assert float(o[0, 0]) == 7.0 and float(o[15, 63]) == 7.0, i
    after = _free_keys()
    assert before - after < 8, f"{COPIES} kernel modules took {before - after} pthread keys"
