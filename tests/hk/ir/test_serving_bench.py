"""tools/hk-bench/serving: the scorecard's serving rows -- their definitions, the command each arm runs, and
the rule that calls a difference (no GPU, no engine)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

D = Path(__file__).resolve().parents[3] / "tools" / "hk-bench" / "serving"
ROWS = json.loads((D / "rows.json").read_text())
ARMS = json.loads((D / "arms.json").read_text())


def _dry(row, arm, mode="measure"):
    r = subprocess.run([sys.executable, str(D / "serve_bench.py"), "--row", row, "--arm", arm, "--mode", mode,
                        "--gpus", "0", "--model-path", "/models/m", "--work", "/w", "--out", "/o", "--dry-run"],
                       capture_output=True, text=True, check=True,
                       env={k: v for k, v in os.environ.items() if not k.startswith(("HK_", "PYTORCH_TUNABLEOP"))})
    cmd, *env = r.stdout.splitlines()
    return cmd, dict(x.strip().split("=", 1) for x in env)


def test_every_arm_belongs_to_a_row_and_names_only_its_switches():
    assert set(ARMS) <= set(ROWS)
    for row, arms in ARMS.items():
        assert {"ship", "fair", "hk_fair"} <= set(arms), row
        for arm, a in arms.items():
            assert set(a) <= {"label", "to", "args", "env", "hk"}, (row, arm)


def test_a_row_names_its_weights_revision_and_no_machine_path():
    for row, m in ROWS.items():
        repo, _, rev = m["weights"].partition("@")
        assert "/" in repo and len(rev) == 40, row
        text = json.dumps(m)
        assert "/workspace" not in text and "/home/" not in text, row


@pytest.mark.parametrize("row,arm", [(r, a) for r, arms in ARMS.items() for a in arms])
def test_each_arm_runs_vllm_with_hk_on_exactly_when_it_says(row, arm):
    cmd, env = _dry(row, arm)
    assert cmd.startswith("vllm serve /models/m ") and " ".join(ROWS[row]["args"][:2]) in cmd
    assert (env.get("HK_DISABLE") == "1") != bool(ARMS[row][arm].get("hk"))
    assert (env.get("PYTORCH_TUNABLEOP_ENABLED") == "1") == bool(ARMS[row][arm].get("to"))
    assert "{work}" not in cmd + json.dumps(env)


def test_tuning_modes_record_then_tune_online_into_the_rows_file():
    _, rec = _dry("mcpm5", "fair", "to_record")
    assert rec["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] == "1" and rec["PYTORCH_TUNABLEOP_TUNING"] == "0"
    _, onl = _dry("mcpm5", "fair", "to_online")
    assert onl["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert onl["PYTORCH_TUNABLEOP_FILENAME"] == "/w/to/mcpm5%d.csv"


def _start(root, arm, s, tpot, failed=0):
    d = root / f"{arm}-s{s}"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"arm": arm, "mode": "measure", "startup_s": 1, "lock_wait_s": 0,
        "benches": {"x": {"spec": "512/256/8/32", "median_tpot_ms": tpot, "failed": failed, "rc": 0, "foreign": {}}}}))


def test_a_difference_is_called_only_when_the_ranges_do_not_overlap(tmp_path):
    sys.path.insert(0, str(D))
    try:
        import table
    finally:
        sys.path.pop(0)
    for s, (a, b, c) in enumerate([(10.0, 9.0, 9.9), (10.2, 9.1, 10.1), (10.1, 9.05, 10.3)], 1):
        _start(tmp_path, "fair", s, a)
        _start(tmp_path, "hk_fair", s, b)
        _start(tmp_path, "tie", s, c)
    _start(tmp_path, "hk_fair", 4, 5.0, failed=3)          # failed requests: discarded
    rows = table.table([str(tmp_path)], "fair", ["fair", "hk_fair", "tie"], [("TPOT", "512/256/8/32", "median_tpot_ms")])
    assert "**1.116x**" in rows[3] and "9.05 (9.00-9.10)" in rows[3]   # 10.1 / 9.05
    assert "(=)" in rows[4]
    assert any("3 failed requests" in x for x in rows)
