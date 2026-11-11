"""sgld_rope's engine contract over both forms SGLang has written Qwen-Image 2.1's apply_rope in:
the complex multiply inline, and (since its fused-rope refactor) in _eager_rope."""

from __future__ import annotations

import pytest

from hk.integration import sglang_diffusion as sgld


ROPE_INLINE = '''
def apply_rope(x, rope):
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
'''

# SGLang after its fused-rope refactor (eb73a50b): the same multiply in _eager_rope
ROPE_EAGER = '''
_is_cuda = False
def _eager_rope(x, rope):
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
def apply_rope(x, rope):
    if _is_cuda and x.is_cuda:
        return fused_complex_rope(x, rope)
    return _eager_rope(x, rope)
'''


@pytest.mark.parametrize("src,ok", [
    (ROPE_INLINE, True),
    (ROPE_EAGER, True),
    (ROPE_EAGER.replace("return _eager_rope(x, rope)", "return _other(x, rope)"), False),
    (ROPE_EAGER.replace("rope[None, :, None]", "rope[:, None]"), False),
], ids=["inline", "eager", "eager-not-called", "eager-changed"])
def test_the_rope_contract_reads_apply_rope_and_its_eager_form(src, ok, tmp_path, monkeypatch):
    from hk.integration import contract  # noqa: PLC0415

    name = f"hkfake_qwen21_{abs(hash(src))}"
    (tmp_path / f"{name}.py").write_text(src)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sgld, "QWEN21_HOME", name)
    assert (contract.CHECKS["sgld_rope"]() == []) is ok
