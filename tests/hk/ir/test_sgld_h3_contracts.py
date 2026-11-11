"""The engine contracts of the MiniMax H3 image-edit routes (sgld_h3vae, sgld_patchembed) and the
helpers they share: an absent engine module is nothing to route, a broken one is an error."""

from __future__ import annotations

import pytest


H3VAE = {"conv3d": True, "forward": ["self", "x"], "sets": True, "one_frame_causal_zeros": True,
         "spatial_pad": True, "conv": True}


@pytest.mark.parametrize("change,ok", [
    ({}, True),
    ({"forward": ["self", "x", "pre_padded"]}, True),        # SGLang's newer BaseConv3d
    ({"forward": ["self", "x", "cache"]}, False),
    ({"one_frame_causal_zeros": False}, False),             # one frame no longer behind zeros
    ({"spatial_pad": False}, False),
    ({"conv": False}, False),
])
def test_the_h3vae_contract(change, ok):
    from hk.integration import contract  # noqa: PLC0415

    assert (contract.judge("sgld_h3vae", dict(H3VAE, **change)) == []) is ok


def test_the_patchembed_contract_wants_c_t_p_p_patches():
    from hk.integration import contract  # noqa: PLC0415

    f = {"forward": ["self", "hidden_states"], "sets": True, "kernel_is_stride": True,
         "patch_order": True}
    assert contract.judge("sgld_patchembed", f) == []
    assert contract.judge("sgld_patchembed", dict(f, patch_order=False))
    assert contract.judge("sgld_patchembed", dict(f, kernel_is_stride=False))


def test_an_absent_engine_module_is_nothing_to_route_but_a_broken_one_is_an_error(tmp_path,
                                                                                  monkeypatch):
    from hk.integration import contract  # noqa: PLC0415

    pkg = tmp_path / "hkfake_engine"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("import hkfake_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert contract._module_or_none("hkfake_engine.minimax_h3") is None
    assert contract._module_or_none("hkfake_absent_engine.x.y") is None
    with pytest.raises(ModuleNotFoundError):
        contract._module_or_none("hkfake_engine.broken")


def test_flat_source_ignores_wrapping_and_trailing_commas():
    from hk.integration import contract  # noqa: PLC0415

    assert contract._flat("F.pad(\n    x,\n    (0, 0, 0, 0, self.kernel_size[0] - 1, 0,),\n)") \
        == "F.pad(x,(0,0,0,0,self.kernel_size[0]-1,0))"
