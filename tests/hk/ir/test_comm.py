"""hk.comm and scoped tensors, without a GPU: what they emit and what they refuse."""

from __future__ import annotations

import pytest

import hk
from hk import i32

T = hk.RT(hk.bf16, 16, 64)


def _kernel(body, name, arch="gfx950"):
    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=1, name=name, grid=lambda p: (1, 1, 1))


def _push():
    def body(x: hk.Tensor[hk.bf16], peer: hk.Tensor[hk.bf16, "system"],
             flag: hk.Tensor[i32, "system"], err: hk.Tensor[i32]):
        idx = hk.tile_coord(0, 0, 0, 0)
        peer[idx] = x[idx].load(T)
        hk.comm.signal(flag.at[0], 1)
        hk.comm.wait(flag.at[64], 1, timeout_s=0.5, error=err.at[0])
        x[idx] = peer[idx].load(T)
    return _kernel(body, "comm_push")


def test_a_scoped_tensor_scopes_every_access():
    src = _push().source()
    assert "kittens::store_scoped<17>(g.peer, " in src
    assert "kittens::load_scoped<17>(" in src and ", g.peer, " in src
    # the unscoped tensor keeps its plain accesses
    assert "kittens::load(ld, g.x, " in src


def test_signal_and_wait_emit_the_measured_protocol():
    src = _push().source()
    assert "kittens::signal<17>(g.flag, " in src
    # 0.5 s of the 100 MHz s_memrealtime clock; >= ; the error element
    assert "kittens::wait<17, 0>(g.flag, " in src and "50000000LL" in src
    assert "&g.err[" in src


def test_wait_without_error_traps_and_eq_compares():
    def body(flag: hk.Tensor[i32, "device"]):
        hk.comm.wait(flag.at[0], 3, cmp="==", timeout_s=0)
    src = _kernel(body, "comm_wait_eq").source()
    assert "kittens::wait<16, 1>(g.flag, " in src and ", 0LL, nullptr)" in src


def test_the_signal_and_its_error_element_are_written_tensors():
    k = _push().trace()
    assert set(k.written_tensors()) >= {"peer", "flag", "err"}


def test_a_flag_must_be_scoped():
    def body(flag: hk.Tensor[i32]):
        hk.comm.signal(flag.at[0], 1)
    with pytest.raises(TypeError, match="unscoped"):
        _kernel(body, "comm_plain_flag").trace()


def test_a_flag_is_int32_and_an_element():
    def body(flag: hk.Tensor[hk.fp32, "system"]):
        hk.comm.signal(flag.at[0], 1)
    with pytest.raises(TypeError, match="int32"):
        _kernel(body, "comm_f32_flag").trace()

    def body2(flag: hk.Tensor[i32, "system"]):
        hk.comm.signal(flag[hk.tile_coord(0, 0, 0, 0)], 1)
    with pytest.raises(TypeError, match="element"):
        _kernel(body2, "comm_tile_flag").trace()


def test_scoped_accesses_are_gfx950s():
    def body(peer: hk.Tensor[hk.bf16, "system"], x: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        peer[idx] = x[idx].load(hk.RT(hk.bf16, 16, 64))
    with pytest.raises(TypeError, match="gfx950"):
        _kernel(body, "comm_rdna", arch="gfx1100").trace()


def test_a_scoped_tensor_moves_register_tiles():
    def body(peer: hk.Tensor[hk.fp32, "system"], x: hk.Tensor[hk.fp32]):
        v = x[hk.tile_coord(0, 0, 0, 0)].load(hk.col_vec(hk.RT(hk.fp32, 16, 64)))
        peer[hk.tile_coord(0, 0, 0, 0)] = v
    with pytest.raises(TypeError, match="register tiles"):
        _kernel(body, "comm_vec").trace()


def test_scoped_atomics_carry_the_scope():
    def body(cnt: hk.Tensor[i32, "system"]):
        cnt.at[0].atomic_add_scalar(1)
    assert "kittens::atomic_scalar_scoped<17>(g.cnt, " in _kernel(body, "comm_atomic").source()


def test_the_scope_is_device_or_system():
    with pytest.raises(TypeError, match="scope"):
        hk.Tensor[hk.bf16, "peer"]
    assert hk.Tensor[hk.bf16, "device"].scope == "device"
    assert hk.Tensor[hk.bf16].scope == ""


def test_symmetric_memory_must_bind_to_scoped_parameters(monkeypatch):
    from hk.runtime import symmetric

    class Fake:
        def __init__(self, p): self.p = p
        def data_ptr(self): return self.p

    monkeypatch.setattr(symmetric, "_RANGES", [(0x1000, 0x2000)])
    params = [("x", hk.Tensor[hk.bf16]), ("peer", hk.Tensor[hk.bf16, "system"])]
    symmetric.check_bindings("k", params, [Fake(0x5000), Fake(0x1800)])   # fine
    with pytest.raises(TypeError, match="unscoped"):
        symmetric.check_bindings("k", params, [Fake(0x1800), Fake(0x1800)])


def test_arrive_and_barrier_emit_an_atomic_arrival_then_a_wait():
    def body(cnt: hk.Tensor[i32, "device"], out: hk.Tensor[i32]):
        old = hk.comm.arrive(cnt.at[0], 2)
        out.at[0].atomic_max_scalar(old)
        hk.comm.barrier(cnt.at[16], 4, timeout_s=0.1)
    k = _kernel(body, "comm_arrive")
    src = k.source()
    assert "kittens::arrive<16>(g.cnt, " in src
    assert src.count("kittens::arrive<16>") == 2 and "kittens::wait<16, 0>(g.cnt, " in src
    # each arrival's __shared__ slot is in the static LDS count
    assert "hk.lds_static = 32 : i64" in k.trace().dump()


def test_an_unscoped_counter_is_refused():
    def body(cnt: hk.Tensor[i32]):
        hk.comm.barrier(cnt.at[0], 1)
    with pytest.raises(TypeError, match="unscoped"):
        _kernel(body, "comm_plain_barrier").trace()


def test_resident_is_a_kernel_option_that_survives_specialization():
    def body(cnt: hk.Tensor[i32, "device"], *, N: int = 1):
        hk.comm.barrier(cnt.at[0], N)
    body.__name__ = "comm_resident"
    k = hk.kernel(body, arch="gfx950", warps=1, grid=lambda p: (4, 1, 1), resident=True)
    assert k.resident and k.specialize(N=4).resident and k.preloaded().resident
