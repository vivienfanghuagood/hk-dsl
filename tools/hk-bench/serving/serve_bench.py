"""serve_bench.py -- one server start of a serving scorecard row (vLLM, CUDA graphs).

    python3 serve_bench.py --row mcpm5 --arm hk_fair --gpus 0 --model-path DIR --work WORK --out OUT
    python3 serve_bench.py ... --mode to_record      # record the GEMM shapes for TunableOp (to_tune.py)
    python3 serve_bench.py ... --mode to_online      # tune online what offline tuning skips (M = 1)
    python3 serve_bench.py ... --dry-run             # print the command and environment, start nothing

One `vllm serve` with the row's serving configuration (rows.json) plus the arm's environment and flags
(arms.json). After the server is up: unmeasured warm-up waves, then every spec
"in/out/concurrency/requests[/mm]" through `vllm bench serve` (random tokens, ignore_eos, a distinct seed
per spec so no bench hits another's prefix cache), then the quality set (quality.py) against the same
server. OUT/result.json has every bench's numbers, the arm, the command, the environment, which
processes held the GPUs before, during and after each bench (KFD's sysfs: a bench is discarded if another
process held more than 64 MiB on a card of the run), and the routes hk said it applied.

A row is several starts of every arm, alternated (docs/performance/scorecard.md, "Method"); table.py
turns the starts into the row. Run it in the row's image, with hk-dsl installed in the image's Python for
the hk arms (`pip install hk-dsl`), or a checkout's python/ via --hk-src. --lock-dir takes an flock per
GPU index (all or none) so two runs never share a card.
"""
import argparse, fcntl, json, os, re, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def hold_all(lock_dir, gpus, what, wait):
    """flock LOCK_DIR/gpu<N>.lock for every GPU of the run, all or none (never one card while waiting for
    another). Returns the open fds and the seconds waited."""
    paths = [os.path.join(lock_dir, f"gpu{g}.lock") for g in gpus]
    t0 = time.time()
    while True:
        fds = []
        for p in paths:
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fds.append(fd)
            except BlockingIOError:
                os.close(fd)
                for f in fds:
                    os.close(f)
                fds = None
                break
        if fds is not None:
            for p in paths:
                open(p[:-5] + ".owner", "w").write(f"pid={os.getpid()} {what}\n")
            return fds, time.time() - t0
        if time.time() - t0 > wait:
            raise SystemExit("GPU locks not free in time")
        time.sleep(20)


def _gpu_ids(gpus):
    """KFD gpu_id of each of the run's GPUs (HIP index), by the PCI address rocm-smi gives the index."""
    ids = {}
    for n in os.listdir("/sys/class/kfd/kfd/topology/nodes"):
        d = f"/sys/class/kfd/kfd/topology/nodes/{n}"
        props = dict(l.split() for l in open(f"{d}/properties") if len(l.split()) == 2)
        g = open(f"{d}/gpu_id").read().strip()
        if g != "0":
            ids[int(props["location_id"]) >> 8] = g
    out = []
    for idx in gpus:
        s = subprocess.run(["rocm-smi", "-d", str(idx), "--showbus"], capture_output=True, text=True).stdout
        bus = re.search(r"PCI Bus: [0-9A-Fa-f]+:([0-9A-Fa-f]+):", s).group(1)
        out.append(ids[int(bus, 16)])
    return out


def kfd_holders(gids):
    """{gpu_id: {host pid: MiB}}: every process with VRAM on the run's GPUs (KFD's sysfs, any user's)."""
    out = {}
    for g in gids:
        out[g] = {}
        for p in os.listdir("/sys/class/kfd/kfd/proc"):
            try:
                v = int(open(f"/sys/class/kfd/kfd/proc/{p}/vram_{g}").read()) >> 20
            except (OSError, ValueError):
                continue
            if v > 0:
                out[g][p] = v
    return out


def _sub(v, work):
    if isinstance(v, str):
        return v.replace("{work}", work)
    if isinstance(v, list):
        return [_sub(x, work) for x in v]
    if isinstance(v, dict):
        return {k: _sub(x, work) for k, x in v.items()}
    return v


def plan(a):
    """(row, arm, command, environment) of a start."""
    M = _sub(json.load(open(os.path.join(HERE, "rows.json")))[a.row], a.work)
    A = _sub(json.load(open(os.path.join(HERE, "arms.json")))[a.row][a.arm], a.work)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=a.gpus, HSA_DISABLE_COREDUMP_ON_EXCEPTION="1")
    env.update(M.get("env", {}))
    env.update(A.get("env", {}))
    if A.get("to") or a.mode != "measure":
        env.update(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="0")
        if a.mode == "measure":
            env["PYTORCH_TUNABLEOP_FILENAME"] = M["to_file"]
        elif a.mode == "to_online":
            # TunableOp's offline tuner skips M = 1 GEMMs ("Offline tuning is not supported for this
            # GEMM"): this start runs the specs with tuning on and the row's results file, so whatever is
            # still untuned is tuned online, in place, with the same rotating buffer
            env.update(PYTORCH_TUNABLEOP_TUNING="1", PYTORCH_TUNABLEOP_FILENAME=M["to_file"],
                       PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=env.get("PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE", "256"))
        else:
            env["PYTORCH_TUNABLEOP_FILENAME"] = os.path.join(a.out, "tunableop%d.csv")
            env["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "1"
            env["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = os.path.join(a.out, "tunableop_untuned%d.csv")
    if A.get("hk"):
        env.pop("HK_DISABLE", None)
        if a.hk_src:
            env["PYTHONPATH"] = os.pathsep.join([os.path.join(a.hk_src, "python")] +
                                                ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    else:
        env["HK_DISABLE"] = "1"
    cmd = (["vllm", "serve", a.model_path, "--port", str(a.port), "--host", "127.0.0.1",
            "--served-model-name", M["served"]] + M["args"] + A.get("args", []))
    return M, A, cmd, env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--row", required=True, help="a key of rows.json")
    ap.add_argument("--arm", required=True, help="a key of arms.json[row]")
    ap.add_argument("--gpus", required=True, help="HIP indices of the run's GPUs, e.g. 0 or 0,1")
    ap.add_argument("--model-path", required=True, help="the row's weights (rows.json names the revision)")
    ap.add_argument("--work", required=True, help="the row's work directory ({work} in rows/arms.json)")
    ap.add_argument("--out", required=True, help="this start's result directory")
    ap.add_argument("--mode", default="measure", choices=["measure", "to_record", "to_online"])
    ap.add_argument("--hk-src", help="a checkout whose python/ the hk arms import instead of the installed hk")
    ap.add_argument("--lock-dir", help="take an flock per GPU in this directory for the whole start")
    ap.add_argument("--lock-wait", type=float, default=4 * 3600)
    ap.add_argument("--port", type=int, default=8270)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    M, A, cmd, env = plan(a)
    shown = {k: v for k, v in env.items() if re.match(r"(HK_|PYTORCH_|PYTHONPATH|HIP_|HSA_|VLLM_|NCCL_|TORCH_|OMP_)", k)}
    if a.dry_run:
        print(" ".join(cmd))
        for k in sorted(shown):
            print(f"  {k}={shown[k]}")
        return
    os.makedirs(a.out, exist_ok=True)
    gpus = a.gpus.split(",")
    fds, waited = (hold_all(a.lock_dir, gpus, f"serve_bench {a.row} {a.arm}", a.lock_wait) if a.lock_dir
                   else ([], 0.0))
    gids = _gpu_ids(gpus)
    res = {"row": a.row, "arm": a.arm, "arm_def": A, "row_def": {k: v for k, v in M.items() if k != "args"},
           "mode": a.mode, "lock_wait_s": waited, "t_start": time.time(), "holders_before": kfd_holders(gids),
           "cmd": cmd, "env_vars": shown, "benches": {}}
    log = open(os.path.join(a.out, "server.log"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                           preexec_fn=lambda: __import__("resource").setrlimit(
                               __import__("resource").RLIMIT_CORE, (0, 0)))
    base = f"http://127.0.0.1:{a.port}"
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit(f"server exited {srv.returncode}")
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=5)
                break
            except Exception:  # noqa: BLE001
                # a start that is not up in time is killed and reported (hang.json, exit 5): some vLLM
                # builds can deadlock loading a warm torch.compile cache; start that arm again
                if time.time() - t0 > M.get("startup_timeout", 1800):
                    json.dump(dict(res, hang=f"not up after {time.time() - t0:.0f} s"),
                              open(os.path.join(a.out, "hang.json"), "w"), indent=1, default=str)
                    print("STARTUP-HANG", flush=True)
                    try:
                        os.killpg(srv.pid, signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(15)
                    os._exit(5)
                time.sleep(5)
        res["startup_s"] = time.time() - t0
        own = kfd_holders(gids)
        res["own_holders"] = own

        def bench(spec, tag, seed):
            parts = spec.split("/")
            IN, OUT, C, N = parts[:4]
            if len(parts) > 4 and parts[4] == "mm":
                ds = ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions", "--dataset-name", "random-mm",
                      "--random-mm-base-items-per-request", "1", "--random-mm-num-mm-items-range-ratio", "0",
                      "--random-mm-limit-mm-per-prompt", '{"image":1,"video":0}',
                      "--random-mm-bucket-config", '{(448, 448, 1): 0.5, (896, 896, 1): 0.5}']
            else:
                ds = ["--backend", "vllm", "--endpoint", "/v1/completions", "--dataset-name", "random"]
            f = f"bench-{tag}"
            c = (["vllm", "bench", "serve"] + ds + ["--base-url", base, "--model", M["served"],
                 "--tokenizer", a.model_path, "--random-input-len", IN, "--random-output-len", OUT, "--ignore-eos",
                 "--max-concurrency", C, "--num-prompts", N, "--percentile-metrics", "ttft,tpot,itl,e2el",
                 "--metric-percentiles", "50,90,99", "--seed", str(seed), "--save-result", "--result-dir", a.out,
                 "--result-filename", f + ".json", "--trust-remote-code"])
            h0 = kfd_holders(gids)
            t = time.time()
            r = subprocess.run(c, stdout=open(os.path.join(a.out, f + ".log"), "w"), stderr=subprocess.STDOUT, env=env)
            h1 = kfd_holders(gids)
            foreign = {g: {p: v for p, v in {**h0[g], **h1[g]}.items() if p not in own[g] and v > 64} for g in h0}
            foreign = {g: v for g, v in foreign.items() if v}
            try:
                j = json.load(open(os.path.join(a.out, f + ".json")))
            except (OSError, ValueError):
                j = {}
            keep = {k: j.get(k) for k in ("completed", "failed", "duration", "total_input_tokens", "total_output_tokens",
                                          "request_throughput", "output_throughput", "mean_ttft_ms", "median_ttft_ms",
                                          "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms",
                                          "p99_tpot_ms", "median_itl_ms", "median_e2el_ms")}
            res["benches"][tag] = dict(keep, spec=spec, seed=seed, rc=r.returncode, t0=t, t1=time.time(), foreign=foreign)
            print(f"{tag} {spec}: tpot {keep.get('median_tpot_ms')} ttft {keep.get('median_ttft_ms')} "
                  f"out {keep.get('output_throughput')} failed {keep.get('failed')} foreign {foreign}", flush=True)

        bench(M["warm"], "warm", 1001)
        for i, w in enumerate(M.get("warm_extra", [])):
            bench(w, f"warm{i + 2}", 1002 + i)
        for i, spec in enumerate(M["specs"]):
            bench(spec, re.sub(r"[^0-9a-z]+", "-", spec), 11 + i)
        if a.mode == "measure" and not a.no_quality and M.get("quality"):
            q = M["quality"]
            qc = [sys.executable, os.path.join(HERE, "quality.py"), str(a.port), M["served"],
                  os.path.join(a.out, "quality.json"), "--gsm", str(q.get("gsm", 0)), "--chartqa", str(q.get("chartqa", 0)),
                  "--conc", "16", "--data", q["gsm_data"]] + (["--chartqa-dir", q["chartqa_dir"]] if q.get("chartqa") else [])
            t = time.time()
            r = subprocess.run(qc, stdout=open(os.path.join(a.out, "quality.log"), "w"), stderr=subprocess.STDOUT)
            res["quality_s"] = time.time() - t
            res["quality_rc"] = r.returncode
        # what hk said it applied, per process: its "routes on" line, or a refused contract
        hk_lines = {}
        for line in open(os.path.join(a.out, "server.log"), errors="replace"):
            if "[hk] vLLM routes on:" in line or "engine contract FAILED" in line or "[hk] vLLM: no route on" in line:
                hk_lines.setdefault(line.split("[hk]")[0].strip() or "main", line.strip()[:600])
        res["hk_routes"] = hk_lines
        res["holders_after"] = kfd_holders(gids)
        res["t_end"] = time.time()
        json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=str)
        print("RESULT-DONE", flush=True)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(90)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(10)   # let the GPU memory go before the next start takes the cards
        for f in fds:
            os.close(f)


if __name__ == "__main__":
    main()
