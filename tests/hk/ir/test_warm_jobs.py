"""The cache warmer builds only what a plan can ask for: no norm job above its kind's TPW cap
(norm._tpw_cap). It used to build the fused-add norm at MAX_TPW tiles a warp, unfolded and inexact --
a configuration norm caps at three because four spill -- and on ROCm 7.2.3's compiler the resource
gate rejected that build, failing `python -m hk.runtime.warm`. No GPU."""

from __future__ import annotations

from hk.ops import norm
from hk.runtime import warm


def test_no_norm_warm_job_exceeds_its_kinds_tpw_cap():
    over = []
    for label, _kernel, consts in warm._all_shipped_jobs():
        if not label.startswith("norm.") or "TPW" not in consts:
            continue
        kind = label.split(" ", 1)[0].rsplit(".", 1)[-1].split("_", 1)[0]
        cap = norm._tpw_cap(kind, consts.get("FOLD", 1), bool(consts.get("EXACT", 0)))
        if consts["TPW"] > cap:
            over.append((label, consts["TPW"], cap))
    assert not over, over


def test_the_top_of_the_tpw_range_is_still_warmed():
    tops = {label for label, _k, c in warm._all_shipped_jobs() if c.get("TPW") == norm.MAX_TPW}
    assert any(".rmsw_" in t for t in tops)                  # MAX_TPW is still exercised
    assert not any(".addrmsw_" in t for t in tops)           # but not where no plan asks for it
