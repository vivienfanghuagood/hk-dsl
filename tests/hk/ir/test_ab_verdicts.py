"""`python -m hk.integration ab --json` without a GPU: the verdict of a set of
rounds, the rows' fixed schema, and judging an earlier run's raw output."""

import json

import pytest

from hk.integration import ab


@pytest.mark.parametrize("ratios, within, verdict", [
    ([1.30, 1.31, 1.29], 0.004, "win"),
    ([0.90, 0.91, 0.89], 0.004, "loss"),
    ([1.005, 0.998, 1.01], 0.004, "tie"),      # inside the 2% floor
    ([1.04, 1.00, 0.97], 0.0, "tie"),          # inside its own 4% spread
    ([1.07, 1.00, 0.94], 0.0, "noise"),        # inside a band too wide to call a tie
    ([1.30, 1.31, 1.29], 0.20, "noise"),       # a band past the cap says nothing
    ([1.30], 0.0, "noise"),                    # one round measures no spread
])
def test_a_verdict_is_only_as_good_as_its_band(ratios, within, verdict):
    ratio, band, got, why, _ = ab.judge(ratios, within)
    assert got == verdict, why
    assert band >= ab.BAND and got in ab.VERDICTS


def test_the_band_is_the_larger_of_the_floor_and_the_noise():
    _, band, _, _, across = ab.judge([1.10, 1.20, 1.15], 0.01)
    assert across == pytest.approx(0.05 / 1.15) and band == pytest.approx(across)
    assert ab.judge([1.10, 1.10], 0.03)[1] == 0.03


def _round(n, total, short, prefill, reps=None):
    v = {"min": total, "short": short, "prefill": prefill}
    if reps:
        v["reps"] = reps
    return {"round": n, "1024x16": v}


def test_rounds_pair_by_number_and_rows_keep_the_schema():
    rounds = {
        "stock": [_round(0, 7.40, 5.40, 3.83), _round(1, 7.42, 5.41, 3.84),
                  _round(2, 7.38, 5.39, 3.82)],
        # round 1 of the arm failed: rounds 0 and 2 pair with stock's 0 and 2
        "gemm": [_round(0, 5.12, 4.40, 2.98), _round(2, 5.11, 4.39, 2.97)],
    }
    rows = ab.verdicts(rounds, tokens=72, short=32,
                       toks={"stock": {"1024x16": [1, 2, 3]}, "gemm": {"1024x16": [1, 2, 4]}})
    assert [tuple(r) for r in rows] == [ab.ROW_KEYS] * 3
    by = {r["metric"]: r for r in rows}
    assert by["total"]["verdict"] == "win" and by["total"]["rounds"] == 2
    assert by["total"]["ratio"] == pytest.approx((7.40 / 5.12 + 7.38 / 5.11) / 2, 1e-4)
    assert by["decode"]["ms"] == pytest.approx(((5.12 - 4.40) + (5.11 - 4.39)) / 2 / 40 * 1e3)
    assert by["total"]["tokens_same"] == "2/3"


def test_a_rounds_reps_widen_its_band():
    reps = {"min": [7.0, 7.7, 7.35], "short": [5.0, 5.0, 5.0], "prefill": [3.0, 3.0, 3.0]}
    rounds = {"stock": [_round(0, 7.0, 5.0, 3.0, reps), _round(1, 7.0, 5.0, 3.0, reps)],
              "user": [_round(0, 7.0, 5.0, 3.0, reps), _round(1, 7.0, 5.0, 3.0, reps)]}
    total = next(r for r in ab.verdicts(rounds, 72, 32) if r["metric"] == "total")
    assert total["noise"]["within"] == pytest.approx(2 * 0.35 / 7.35, 1e-3)
    assert total["verdict"] == "noise"


def test_an_earlier_out_file_is_judged_again(tmp_path):
    out = tmp_path / "raw.json"
    out.write_text(json.dumps({
        "_meta": {"model": "m", "tokens": 72, "short": 32},
        "_rounds": {"stock": [_round(0, 7.4, 5.4, 3.8), _round(1, 7.4, 5.4, 3.8)],
                    "gemm": [_round(0, 5.1, 4.4, 3.0), _round(1, 5.1, 4.4, 3.0)]}}))
    js = tmp_path / "v.json"
    assert ab.main(["--from", str(out), "--json", str(js)]) == 0
    doc = json.loads(js.read_text())
    assert doc["schema"] == ab.SCHEMA and doc["meta"]["from"] == str(out)
    assert {r["verdict"] for r in doc["rows"]} == {"win"}
