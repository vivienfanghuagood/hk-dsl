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
#: SGLang-Diffusion's metric: wall seconds per output, request to file.
DIFF_METRICS = ("seconds",)
_SGLANG_CLI = "import sys; from sglang.cli.main import main; sys.argv[0] = 'sglang'; sys.exit(main())"


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


def _post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def generate(a, port: int, path: Path) -> dict:
    """One SGLang-Diffusion request (the fixed prompt and seed); the output written to `path`.
    Seconds are wall time from the request to the file in hand."""
    import base64  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    base = f"http://127.0.0.1:{port}"
    t = time.perf_counter()
    if a.diffusion == "image":
        r = _post(base + "/v1/images/generations",
                  dict(prompt=a.prompt, size=a.size, num_inference_steps=a.steps, seed=a.seed,
                       n=1, output_format="png", response_format="b64_json"), a.bench_timeout)
        data = base64.b64decode(r["data"][0]["b64_json"])
    else:
        w, h = (int(v) for v in a.size.split("x"))
        job = _post(base + "/v1/videos", dict(prompt=a.prompt, width=w, height=h,
                                              num_frames=a.frames, num_inference_steps=a.steps,
                                              seed=a.seed, n=1), a.bench_timeout)
        t_end = time.time() + a.bench_timeout
        while True:
            st = json.loads(urllib.request.urlopen(f"{base}/v1/videos/{job['id']}", timeout=60).read())
            if st.get("status") in ("completed", "failed", "cancelled"):
                break
            if time.time() > t_end:
                raise SystemExit(f"video {job['id']} not done in {a.bench_timeout} s")
            time.sleep(0.5)
        if st.get("status") != "completed":
            raise SystemExit(f"video request failed: {str(st)[:300]}")
        data = urllib.request.urlopen(f"{base}/v1/videos/{job['id']}/content", timeout=600).read()
    seconds = time.perf_counter() - t
    path.write_bytes(data)
    return {"seconds": round(seconds, 3), "sha256": hashlib.sha256(data).hexdigest()[:16],
            "file": path.name}


def _frames(path: Path):
    """An output as a float array (frames, H, W, 3): a png or an mp4."""
    import numpy as np  # noqa: PLC0415

    if path.suffix == ".png":
        from PIL import Image  # noqa: PLC0415

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)[None]
    import av  # noqa: PLC0415

    return np.stack([f.to_ndarray(format="rgb24") for f in av.open(str(path)).decode(video=0)]
                    ).astype(np.float64)


def psnr(x: Path, y: Path) -> float:
    a, b = _frames(x), _frames(y)
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else round(10 * __import__("math").log10(255 ** 2 / mse), 2)


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


def summarize(rows: list, specs: list, metrics=METRICS) -> list:
    out = []
    for spec in specs:
        for metric in metrics:
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
    d = ap.add_argument_group("SGLang-Diffusion (`sglang serve`; the arms switch hk's sgld_* routes)")
    d.add_argument("--diffusion", choices=("image", "video"), default="",
                   help="an image (/v1/images/generations) or video (/v1/videos) model")
    d.add_argument("--prompt", default="A cat walking on the grass, realistic, natural light")
    d.add_argument("--size", default="1024x1024", help="WxH")
    d.add_argument("--frames", type=int, default=81)
    d.add_argument("--steps", type=int, default=30)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--warm-tol", type=float, default=0.02,
                   help="warm each arm until two requests in a row are within this (MIOpen's find)")
    d.add_argument("--warm-max", type=int, default=6, help="give up warming after this many requests")
    a = ap.parse_args(argv)
    if a.diffusion:
        return diffusion_main(a, server_args)
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
                        diff = {p: (before[p], after[p]) for p in before
                                if p in after and after[p][1:] != before[p][1:]}
                        raise SystemExit(f"round {r} {arm} {spec}: hk's counters "
                                         f"{'moved' if moved else 'did not move'}; the switch did not hold"
                                         f" (rank state before, after: {diff or before})")
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


def _stop(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except ProcessLookupError:              # it had exited (its log says why)
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def _check_moved(r, arm, before, after) -> None:
    moved = any(after[p][1:] != before[p][1:] for p in before if p in after)
    if moved != (arm == "hk"):
        diff = {p: (before[p], after[p]) for p in before if p in after and after[p][1:] != before[p][1:]}
        raise SystemExit(f"round {r} {arm}: hk's counters {'moved' if moved else 'did not move'}; "
                         f"the switch did not hold (rank state before, after: {diff or before})")


def diffusion_main(a, server_args) -> int:
    """`sglang serve` with hk's SGLang-Diffusion routes (hk's boot installed in this Python), the
    arms switched by signal. Each arm is warmed until two requests in a row are within --warm-tol;
    then the arms alternate round by round. Every output is kept; the PSNR of each round's hk
    output against the stock output of the same round (same prompt and seed) is reported."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="hk-sglang-toggle-"))
    env = dict(os.environ, HK_SGLANG_TOGGLE_DIR=str(state),
               PYTHONPATH=os.pathsep.join([str(TOGGLE)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    cmd = [sys.executable, "-c", _SGLANG_CLI, "serve", "--model-path", a.model, "--host", "127.0.0.1",
           "--port", str(a.port)] + server_args
    ext = ".png" if a.diffusion == "image" else ".mp4"
    spec = f"{a.diffusion} {a.size}" + (f"x{a.frames}" if a.diffusion == "video" else "") + f" {a.steps} steps"
    meta = {"cmd": cmd, "spec": spec, "prompt": a.prompt, "seed": a.seed, "rounds": a.rounds,
            "started": time.strftime("%FT%TZ", time.gmtime()),
            "env": {k: v for k, v in os.environ.items()
                    if k.startswith(("HK_", "PYTORCH_TUNABLEOP", "SGLANG_", "MIOPEN_"))}}
    if os.environ.get("HK_TOGGLE_STOCK_NO_CUDNN", "") not in ("", "0"):
        meta["stock"] = "tuned: PyTorch's own convolutions (cudnn / MIOpen off) in the stock arm only"
    with open(out / "server.log", "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows, warm = [], {}
    try:
        wait_up(a.port, proc, a.startup_timeout)
        for arm in ("stock", "hk"):
            switch(state, arm)
            times = []
            for i in range(a.warm_max):
                times.append(generate(a, a.port, out / f"warm-{arm}-{i}{ext}")["seconds"])
                print(f"warm {arm} {i}: {times[-1]} s", flush=True)
                if len(times) >= 2 and abs(times[-1] - times[-2]) <= a.warm_tol * times[-2]:
                    break
            else:
                raise SystemExit(f"{arm} did not settle within {a.warm_tol:.0%} in {a.warm_max} requests: {times}")
            warm[arm] = times
        for r in range(a.rounds):
            for arm in (("stock", "hk") if r % 2 == 0 else ("hk", "stock")):
                before = switch(state, arm)
                row = generate(a, a.port, out / f"r{r}-{arm}{ext}")
                _check_moved(r, arm, before, switch(state, "dump"))
                rows.append(dict(row, round=r, arm=arm, spec=spec))
                print(f"round {r} {arm:5}: {row['seconds']} s {row['sha256']}", flush=True)
    finally:
        _stop(proc)
        shutil.rmtree(state, ignore_errors=True)
    summary = summarize(rows, [spec], DIFF_METRICS)
    quality = []
    for r in range(a.rounds):
        q = {"round": r}
        try:
            q["psnr_hk_vs_stock"] = psnr(out / f"r{r}-hk{ext}", out / f"r{r}-stock{ext}")
            if r:
                q["psnr_stock_vs_stock_r0"] = psnr(out / f"r{r}-stock{ext}", out / f"r0-stock{ext}")
        except ImportError as e:            # numpy / PIL / av not in this Python
            q["psnr"] = f"not computed: {e}"
        quality.append(q)
    (out / "result.json").write_text(json.dumps({"meta": meta, "warm": warm, "rows": rows,
                                                 "summary": summary, "quality": quality}, indent=1))
    for s in summary:
        print(f"{s['spec']} {s['metric']}: stock {s['stock']} hk {s['hk']} -> {s['ratio_hk_over_stock']}x "
              f"({s['verdict']})")
    for q in quality:
        print(f"round {q['round']} quality: {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
