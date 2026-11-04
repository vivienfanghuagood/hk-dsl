#!/usr/bin/env python3
"""SGLang serving with hk's routes against SGLang without them, in ONE server process.

    python3 tools/hk-bench/serving/sglang_toggle_ab.py --model Qwen/Qwen3.8-27B --out runs/q38 \\
        --specs 1024/128/1/4 --rounds 3 -- --tp 2 --attention-backend triton --disable-cuda-graph

One `sglang.launch_server` starts with hk installed, its plugin on, and `sglang_toggle/` on the
path. The arms differ only in which methods SGLang's layers call: hk's routes (`sglang_linear`,
`sglang_norm`) or SGLang's own, switched by a signal to every process of the server
(`sglang_toggle/sitecustomize.py`). The weights, the KV cache, the allocator and the
GPU-to-GPU links stay the same between arms. On a Radeon PRO W7900 two separate server starts can
differ by 13-16% from where each start lands, more than most kernel wins, so arms run in separate
processes are not compared here.

Each spec is `IN/OUT/CONCURRENCY/PROMPTS` (random token ids; `sglang.bench_serving`). Every arm is
warmed on every spec first, unrecorded. Then the arms alternate round by round (stock, hk; hk,
stock; ...). After every switch each rank's state file must say the arm, and the hk arm's call
counters must have moved while the stock arm's must not, or the round is refused. The report gives
each round's medians, the ratio of the arms' medians, and whether the arms' per-round ranges are
disjoint. A ratio whose ranges overlap is not a result.

What it does not do: SGLang's CUDA graphs. A captured graph replays the kernels it captured, so a
switch would not reach it. The driver refuses to start without `--disable-cuda-graph` unless
`--allow-cuda-graph` says the graphs hold none of hk's layers. Only `sglang_linear` and
`sglang_norm` switch; any other route stays as the plugin set it in both arms.

Environment for the server, such as `PYTORCH_TUNABLEOP_*` for a tuned baseline or `HK_SGLANG_LINEAR=1`
for an opt-in route, is passed through from the driver's environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOGGLE = HERE / "sglang_toggle"
SIG = {"stock": signal.SIGRTMIN + 3, "hk": signal.SIGRTMIN + 4, "dump": signal.SIGRTMIN + 5}
METRICS = ("median_tpot_ms", "median_ttft_ms", "output_throughput", "median_itl_ms")


def _states(d: Path) -> dict:
    out = {}
    for f in d.iterdir():
        if f.name.isdigit():
            try:
                out[int(f.name)] = f.read_text().splitlines()
            except OSError:
                pass
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def switch(d: Path, arm: str) -> dict:
    """Signal every live server process; return {pid: [state, counters]} of the ranks (hk loaded)."""
    pids = [p for p in _states(d) if _alive(p)]
    for p in pids:
        os.kill(p, SIG[arm])
    time.sleep(2)
    st = {p: s for p, s in _states(d).items() if p in pids and s and s[0] != "none" and s[0] != "started"}
    bad = {p: s[0] for p, s in st.items() if not s[0].startswith(arm if arm != "dump" else "dump")}
    if not st:
        raise SystemExit(f"no server process with hk loaded answered the {arm} switch "
                         f"(is hk's SGLang plugin on, and the toggle on the server's PYTHONPATH?)")
    if bad:
        raise SystemExit(f"switch to {arm} failed in {bad}")
    return st


def bench(a, spec: str, port: int) -> dict:
    inp, out, conc, n = spec.split("/")
    env = {k: v for k, v in os.environ.items() if k != "HK_SGLANG_TOGGLE_DIR"}
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
        res = fh.name
    try:
        cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang",
               "--host", "127.0.0.1", "--port", str(port), "--dataset-name", "random-ids",
               "--random-input-len", inp, "--random-output-len", out, "--random-range-ratio", "1",
               "--max-concurrency", conc, "--num-prompts", n, "--tokenizer", a.tokenizer or a.model,
               "--output-file", res, "--disable-tqdm"]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       timeout=a.bench_timeout)
        with open(res) as fh:
            row = json.loads(fh.read().strip().splitlines()[-1])
    finally:
        os.unlink(res)
    return {k: row.get(k) for k in METRICS + ("completed",)}


def wait_up(port: int, proc, timeout: int) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"the server exited ({proc.returncode}) before it answered")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return
        except Exception:  # noqa: BLE001 -- not up yet
            time.sleep(3)
    raise SystemExit(f"the server did not answer in {timeout} s")


def summarize(rows: list, specs: list) -> list:
    out = []
    for spec in specs:
        for metric in METRICS:
            per = {arm: [r[metric] for r in rows if r["spec"] == spec and r["arm"] == arm and r[metric]]
                   for arm in ("stock", "hk")}
            if not per["stock"] or not per["hk"]:
                continue
            ms, mh = statistics.median(per["stock"]), statistics.median(per["hk"])
            lower_better = metric != "output_throughput"
            ratio = (ms / mh) if lower_better else (mh / ms)
            disjoint = (max(per["hk"]) < min(per["stock"])) if lower_better else (min(per["hk"]) > max(per["stock"]))
            disjoint_other = (max(per["stock"]) < min(per["hk"])) if lower_better else (min(per["stock"]) > max(per["hk"]))
            out.append({"spec": spec, "metric": metric, "stock": per["stock"], "hk": per["hk"],
                        "ratio_hk_over_stock": round(ratio, 4),
                        "verdict": "hk faster, disjoint" if disjoint else
                                   "hk slower, disjoint" if disjoint_other else "ranges overlap: no result"})
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    server_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, server_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF name or local path, as sglang.launch_server takes it")
    ap.add_argument("--tokenizer", default="", help="for the benchmark client (default: --model)")
    ap.add_argument("--specs", default="1024/128/1/4", help="space-separated IN/OUT/CONCURRENCY/PROMPTS")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=30000 + os.getpid() % 2000)
    ap.add_argument("--out", required=True, help="directory for server.log and result.json")
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--bench-timeout", type=int, default=3600)
    ap.add_argument("--allow-cuda-graph", action="store_true",
                    help="run without --disable-cuda-graph (only if the graphs hold none of hk's layers)")
    a = ap.parse_args(argv)
    if "--disable-cuda-graph" not in server_args and not a.allow_cuda_graph:
        raise SystemExit("pass --disable-cuda-graph to the server (after --): a captured graph "
                         "would not see the switch")
    specs = a.specs.split()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="hk-sglang-toggle-"))
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(state),
               PYTHONPATH=os.pathsep.join([str(TOGGLE)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", a.model, "--host", "127.0.0.1",
           "--port", str(a.port)] + server_args
    meta = {"cmd": cmd, "specs": specs, "rounds": a.rounds, "started": time.strftime("%FT%TZ", time.gmtime()),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("HK_", "PYTORCH_TUNABLEOP", "SGLANG_"))}}
    with open(out / "server.log", "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        wait_up(a.port, proc, a.startup_timeout)
        for arm in ("stock", "hk"):                         # warm both arms on every spec, unrecorded
            switch(state, arm)
            for spec in specs:
                bench(a, spec, a.port)
        for r in range(a.rounds):
            for arm in (("stock", "hk") if r % 2 == 0 else ("hk", "stock")):
                before = switch(state, arm)
                for spec in specs:
                    row = bench(a, spec, a.port)
                    after = switch(state, "dump")
                    moved = any(after[p][1:] != before[p][1:] for p in before if p in after)
                    if moved != (arm == "hk"):
                        raise SystemExit(f"round {r} {arm} {spec}: hk's counters "
                                         f"{'moved' if moved else 'did not move'}; the switch did not hold")
                    before = after
                    rows.append(dict(row, round=r, arm=arm, spec=spec))
                    print(f"round {r} {arm:5} {spec}: " + " ".join(f"{k}={row[k]}" for k in METRICS), flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        shutil.rmtree(state, ignore_errors=True)
    summary = summarize(rows, specs)
    (out / "result.json").write_text(json.dumps({"meta": meta, "rows": rows, "summary": summary}, indent=1))
    for s in summary:
        print(f"{s['spec']} {s['metric']}: stock {s['stock']} hk {s['hk']} -> {s['ratio_hk_over_stock']}x "
              f"({s['verdict']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
