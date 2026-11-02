"""One P.V source for RDNA and CDNA (#212, #213): on RDNA, `S.as_operand(bf16)` and a col-layout
16-bit LDS read of V give the same answer, bit for bit, as the RDNA-native form (`S.to(bf16)`, V
read row-major and transposed in registers), and as a float32 reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk import dsl as hk  # noqa: E402
from hk.types import bf16, fp32  # noqa: E402

pytestmark = pytest.mark.gpu


def _wave32() -> bool:
    from hk.target import current_arch, get_target

    return get_target(current_arch()).wave == 32


def _kernel(form: str, kv: int, d: int):
    def body(k: hk.Tensor[bf16], q: hk.Tensor[bf16], v: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        v_s = hk.lds.alloc(hk.ST(bf16, kv, d), name="v_s")
        hk.group(1).load(v_s, v[hk.elem_coord(0, 0, 0, 0)])
        hk.arch.barrier(drain=True)
        kt = k[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(bf16, kv, 32, "row"))
        qt = q[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(bf16, 16, 32, "row"))
        s = hk.zeros(hk.RT(fp32, kv, 16, "col"))           # S^T = K Q^T
        s += kt @ qt.T
        acc = hk.zeros(hk.RT(fp32, d, 16, "col"))           # O^T = V^T P^T
        if form == "portable":
            acc += v_s.load(hk.RT(bf16, kv, d, "col")).T @ s.as_operand(bf16)
        else:
            vr = v_s.load(hk.RT(bf16, kv, d, "row"), wait=False)
            hk.lds.wait_for(vr)
            acc += hk.transpose_sep(vr) @ s.to(bf16)
        o[hk.elem_coord(0, 0, 0, 0)] = acc

    name = f"t_rdna_operand_{form}_{kv}x{d}"
    body.__name__ = name
    return hk.kernel(body, warps=1, grid=lambda p: (1, 1, 1), name=name)


@pytest.mark.skipif(not _wave32(), reason="RDNA's lowering of the CDNA forms")
@pytest.mark.parametrize("kv,d", [(32, 16), (32, 32), (64, 16)])
def test_the_cdna_form_of_pv_is_the_rdna_form_bitwise(kv, d):
    from hk.runtime.torch_ext import launcher

    g = torch.Generator(device="cuda").manual_seed(kv * 100 + d)
    k = torch.randn(kv, 32, device="cuda", dtype=torch.bfloat16, generator=g)
    q = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(kv, d, device="cuda", dtype=torch.bfloat16, generator=g)
    outs = {}
    for form in ("portable", "native"):
        o = torch.full((d, 16), float("nan"), device="cuda", dtype=torch.float32)
        launcher(_kernel(form, kv, d))(k, q, v, o)
        torch.cuda.synchronize()
        outs[form] = o
    assert torch.equal(outs["portable"], outs["native"])
    s = (k.float() @ q.float().T).to(torch.bfloat16).float()
    ref = v.float().T @ s
    assert (outs["portable"] - ref).abs().max().item() < 1e-3 * ref.abs().max().item()
