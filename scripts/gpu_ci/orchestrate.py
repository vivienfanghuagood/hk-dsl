#!/usr/bin/env python3
"""GPU CI: the GPU tier on main, on a W7900D in the cluster and on an MI355X
on this workstation, reported on each commit as a GitHub status.

    orchestrate.py install              copy these scripts to ~/hk-ci, unpack MLIR, add the cron entries
    orchestrate.py daemon               from an SSH login: tick every 10 minutes, with docker access
    orchestrate.py tick [--nightly]     start what is due, collect what finished (cron runs this)
    orchestrate.py run REF [--tier T]   test REF now, on one tier or both
    orchestrate.py status               what each tier last did

GitHub Actions cannot run this tier. The cluster reaches github.com and
api.github.com only through the gh-test and gh-api-test.anruicloud.com
proxies, and not GitHub's Actions service, so no self-hosted runner can live
next to the GPU. This workstation reaches GitHub, so it orchestrates:

- gfx1100: the dev pod runs the commit (pod_run.sh) and fetches it itself,
  through the proxy. This script only installs pod_run.sh, starts it and reads
  its result, over `ssh 0004` and `kubectl exec`; no code from the repository
  runs on the jump host.
- gfx950: the commit runs here, in rocm/pytorch, on one MI355X that is idle
  when the run starts (local_run.sh). The container sees CI's directories,
  not $HOME. Only the daemon can start it: this workstation grants the
  docker group to login sessions only, so cron's tick leaves the commit
  pending for the daemon's.

Each finished run posts a status on its commit (gpu/gfx1100, gpu/gfx950)
linking to its report, an asset of the gpu-ci-logs prerelease. Only main's
head is tested, so commits pushed together are tested together; a commit that
changes only documentation carries its parent's passing result.
"""

import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = "vivienfanghuagood/hk-dsl"
URL = f"https://github.com/{REPO}.git"
POD = "hfang-hk-rdna3-b"
JUMP = "0004"
PROXY = "gh-test.anruicloud.com"
RELEASE = "gpu-ci-logs"
HOME = Path(os.environ.get("HK_CI_HOME", Path.home() / "hk-ci"))
MLIR_TARBALL = Path.home() / "hk-local/ci/mlir-23.1.3-install-cp310-ubuntu22.04-x86_64.tar.zst"
SCRIPTS = ("orchestrate.py", "pod_run.sh", "local_run.sh", "container_run.sh", "summarize.py")
STALE_HOURS = 6
KEEP_ASSETS = 300
KEEP_TESTED = 200
DOCS_ONLY = re.compile(r"^docs/|\.md$|^LICENSE$|^NOTICE$")
CRON_MARK = "# hk-dsl gpu ci"


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def utc_now() -> str:
    return time.strftime("%H:%M UTC", time.gmtime())


def sh(args, *, input="", timeout=120, check=True):
    p = subprocess.run(args, input=input, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"{' '.join(args[:3])} exited {p.returncode}: {p.stderr.strip()[-400:]}")
    return p


def jump(cmd: str, **kw):
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", JUMP, cmd], **kw)


def pod(cmd: str, **kw):
    return jump(f"kubectl exec -i {POD} -- bash -c {shlex.quote(cmd)}", **kw)


def gh(*args, **kw):
    return sh(["gh", *args], **kw)


def release(*args, **kw):
    # ~/hk-ci is no checkout, so gh has to be told the repository.
    return gh("release", *args, "-R", REPO, **kw)


def post_status(sha: str, state: str, context: str, description: str, url=None) -> None:
    args = ["api", "-X", "POST", f"repos/{REPO}/statuses/{sha}", "-f", f"state={state}",
            "-f", f"context={context}", "-f", f"description={description[:140]}"]
    if url:
        args += ["-f", f"target_url={url}"]
    gh(*args)


def docker_ok() -> bool:
    """This workstation grants the docker, render and video groups through
    pam_group, to SSH and console logins only: a cron job has none of them,
    and cannot start the gfx950 container. `orchestrate.py daemon`, started
    from an SSH login, keeps them."""
    return os.access("/var/run/docker.sock", os.R_OK | os.W_OK)


def main_head() -> str:
    return sh(["git", "ls-remote", URL, "refs/heads/main"]).stdout.split()[0]


def mirror() -> Path:
    m = HOME / "mirror.git"
    if not m.exists():
        sh(["git", "clone", "-q", "--bare", URL, str(m)], timeout=900)
    sh(["git", "-C", str(m), "fetch", "-q", "--force", "origin", "+refs/heads/*:refs/heads/*"],
       timeout=900)
    return m


def resolve(ref: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    p = sh(["git", "-C", str(mirror()), "rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    if p.returncode:
        raise SystemExit(f"cannot resolve {ref!r} on {REPO}")
    return p.stdout.strip()


def docs_only(old: str, new: str) -> bool:
    p = sh(["git", "-C", str(mirror()), "diff", "--name-only", old, new], check=False)
    files = p.stdout.split()
    return p.returncode == 0 and bool(files) and all(DOCS_ONLY.search(f) for f in files)


def ensure_release() -> None:
    if release("view", RELEASE, check=False).returncode == 0:
        return
    release("create", RELEASE, "--prerelease", "--target", "main", "--title", "GPU CI logs",
       "--notes", "One report per GPU CI run, linked from its commit's gpu/* status. "
                  "Written by scripts/gpu_ci/orchestrate.py; the oldest are pruned.")


def upload(report: Path, name: str) -> str:
    ensure_release()
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / name
        shutil.copy(report, dst)
        release("upload", RELEASE, str(dst), "--clobber", timeout=300)
    p = release("view", RELEASE, "--json", "assets", check=False)
    if p.returncode == 0:
        assets = sorted(json.loads(p.stdout)["assets"], key=lambda a: a["createdAt"])
        for a in assets[:-KEEP_ASSETS]:
            release("delete-asset", RELEASE, a["name"], "-y", check=False)
    return f"https://github.com/{REPO}/releases/download/{RELEASE}/{name}"


class Tier:
    name = context = ""

    def __init__(self):
        self.path = HOME / "state" / f"{self.name}.json"
        fresh = {"tested": {}, "last": None, "running": None, "nightly": None, "waiting": None}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else fresh

    def save(self) -> None:
        tested = self.state["tested"]
        for sha in list(tested)[:-KEEP_TESTED]:
            del tested[sha]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1) + "\n")
        tmp.replace(self.path)

    # What a tier provides: start a run (None: not now), its result (None: not
    # yet), whether it is still alive, and its report.
    def start(self, sha: str):
        raise NotImplementedError

    def result(self, run: dict):
        raise NotImplementedError

    def alive(self, run: dict) -> bool:
        raise NotImplementedError

    def report(self, run: dict):
        raise NotImplementedError

    def begin(self, sha: str, nightly: bool = False) -> bool:
        run = self.start(sha)
        if run is None:
            return False
        run.update(sha=sha, started=time.time())
        self.state.update(running=run, waiting=None)
        if nightly:
            self.state["nightly"] = datetime.date.today().isoformat()
        self.save()
        post_status(sha, "pending", self.context, run["description"])
        log(f"{self.name}: started {sha[:10]} ({run['description']})")
        return True

    def finish(self, run: dict, res: dict) -> None:
        sha = run["sha"]
        state = res.get("state") if res.get("state") in ("success", "failure", "error") else "error"
        description = res.get("description") or state
        report = self.report(run)
        url = upload(report, f"{time.strftime('%Y%m%d-%H%M')}-{sha[:10]}-{self.name}.txt") if report else None
        post_status(sha, state, self.context, description, url)
        self.state["tested"][sha] = {"state": state, "description": description, "url": url,
                                     "finished": time.strftime("%FT%T")}
        self.state.update(last=sha, running=None)
        self.save()
        log(f"{self.name}: {sha[:10]} {description}")

    def tick(self, head: str, nightly: bool) -> None:
        run = self.state["running"]
        if run:
            res = self.result(run)
            if res is None:
                hours = (time.time() - run["started"]) / 3600
                if hours < STALE_HOURS and self.alive(run):
                    return
                res = {"state": "error", "description": f"error: the run died or ran past {STALE_HOURS} h"}
            self.finish(run, res)
        today = datetime.date.today().isoformat()
        if nightly:
            self.state["nightly_due"] = today
        force = self.state.get("nightly_due") == today and self.state.get("nightly") != today
        if head in self.state["tested"] and not force:
            return
        last = self.state.get("last")
        prev = self.state["tested"].get(last or "", {})
        if not force and last and prev.get("state") == "success" and docs_only(last, head):
            post_status(head, "success", self.context,
                        f"documentation only: carried from {last[:7]}", prev.get("url"))
            self.state["tested"][head] = dict(prev, carried=last)
            self.state["last"] = head
            self.save()
            log(f"{self.name}: {head[:10]} documentation only, carried from {last[:10]}")
            return
        self.begin(head, nightly=force)


class Gfx1100(Tier):
    name, context = "gfx1100", "gpu/gfx1100"

    def start(self, sha):
        for f in ("pod_run.sh", "summarize.py"):
            pod(f"mkdir -p /workspace/ci && cat > /workspace/ci/{f}", input=(HOME / f).read_text())
        ip = jump(f"getent ahostsv4 {PROXY} | head -1 | cut -d' ' -f1").stdout.strip()
        if not re.fullmatch(r"[0-9.]+", ip):
            raise RuntimeError(f"the jump host cannot resolve {PROXY}")
        # setsid -f, its output redirected: nothing left in the pod may hold
        # kubectl exec's stdout open, or this call waits for the whole run.
        pod(f"rm -rf /workspace/ci/runs/{sha}; setsid -f bash /workspace/ci/pod_run.sh "
            f"{sha} {ip} >/dev/null 2>&1 </dev/null")
        return {"description": f"running on the W7900D, GPU 0 (since {utc_now()})"}

    def result(self, run):
        p = pod(f"cat /workspace/ci/runs/{run['sha']}/result.json 2>/dev/null", check=False)
        return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None

    def alive(self, run):
        p = pod(f"kill -0 $(cat /workspace/ci/runs/{run['sha']}/pid) 2>/dev/null && echo alive",
                check=False)
        return "alive" in p.stdout

    def report(self, run):
        p = pod(f"cat /workspace/ci/runs/{run['sha']}/report.txt", check=False)
        if p.returncode or not p.stdout:
            return None
        path = HOME / "runs" / self.name / run["sha"] / "report.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p.stdout)
        return path


class Gfx950(Tier):
    name, context = "gfx950", "gpu/gfx950"

    @staticmethod
    def idle_gpu():
        """The highest-numbered MI355X under 10% use and 3% VRAM, as the CDNA
        runs chose theirs: the machine is shared, and a run takes one GPU."""
        for g in range(7, -1, -1):
            out = sh(["rocm-smi", "-d", str(g), "--showuse", "--showmemuse"], check=False).stdout
            use = re.search(r"GPU use \(%\): (\d+)", out)
            mem = re.search(r"VRAM%\): (\d+)", out)
            if use and mem and int(use.group(1)) <= 10 and int(mem.group(1)) <= 3:
                return g
        return None

    def wait(self, sha, why):
        if self.state.get("waiting") != [sha, why]:
            post_status(sha, "pending", self.context, why)
            self.state["waiting"] = [sha, why]
            self.save()

    def start(self, sha):
        if not docker_ok():
            self.wait(sha, "waiting for the gfx950 launcher (orchestrate.py daemon, started from an SSH login)")
            return None
        g = self.idle_gpu()
        if g is None:
            self.wait(sha, "waiting for an idle MI355X")
            return None
        shutil.rmtree(HOME / "runs" / self.name / sha, ignore_errors=True)
        p = subprocess.Popen(["bash", str(HOME / "local_run.sh"), sha, str(g)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             env={**os.environ, "HK_CI_HOME": str(HOME)})
        return {"gpu": g, "pid": p.pid,
                "description": f"running on the MI355X, GPU {g} (since {utc_now()})"}

    def result(self, run):
        f = HOME / "runs" / self.name / run["sha"] / "result.json"
        return json.loads(f.read_text()) if f.exists() else None

    def alive(self, run):
        try:
            os.kill(run["pid"], 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def report(self, run):
        f = HOME / "runs" / self.name / run["sha"] / "report.txt"
        return f if f.exists() else None


TIERS = {"gfx1100": Gfx1100, "gfx950": Gfx950}


def cmd_install(args) -> None:
    src = Path(__file__).resolve().parent
    for d in ("state", "logs", "runs", "x950"):
        (HOME / d).mkdir(parents=True, exist_ok=True)
    for f in SCRIPTS:
        shutil.copy(src / f, HOME / f)
    if not (HOME / "mlir" / "install-cp310").exists():
        (HOME / "mlir").mkdir(exist_ok=True)
        sh(["tar", "--zstd", "-xf", str(MLIR_TARBALL), "-C", str(HOME / "mlir")], timeout=1800)
    ensure_release()
    if args.no_cron:
        print(f"installed in {HOME}; no cron entries")
        return
    entries = [
        f"*/10 * * * * cd {HOME} && flock -n tick.lock python3 orchestrate.py tick >> logs/tick.log 2>&1 {CRON_MARK}",
        f"30 3 * * * cd {HOME} && flock -w 900 tick.lock python3 orchestrate.py tick --nightly >> logs/tick.log 2>&1 {CRON_MARK}",
    ]
    current = sh(["crontab", "-l"], check=False).stdout.splitlines()
    kept = [l for l in current if CRON_MARK not in l]
    sh(["crontab", "-"], input="\n".join(kept + entries) + "\n")
    print(f"installed in {HOME}; cron: every 10 minutes, and nightly at 03:30")


def cmd_daemon(args) -> None:
    if not docker_ok():
        raise SystemExit("the daemon needs the docker group, which this workstation grants "
                         "only to login sessions: start it from an SSH login")
    if not args.foreground:
        out = open(HOME / "logs" / "daemon.log", "a")
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "daemon", "--foreground"],
                             stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                             start_new_session=True, cwd=HOME)
        print(f"daemon started, pid {p.pid}; log {out.name}")
        return
    single = open(HOME / "daemon.lock", "w")
    try:
        fcntl.flock(single, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("a daemon is already running")
    log(f"daemon {os.getpid()} up")
    while True:
        with open(HOME / "tick.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                cmd_tick(argparse.Namespace(nightly=False))
            except Exception as e:
                log(f"tick: {type(e).__name__}: {e}")
        (HOME / "daemon.json").write_text(
            json.dumps({"pid": os.getpid(), "tick": time.time()}) + "\n")
        time.sleep(600)


def cmd_tick(args) -> None:
    if not docker_ok():
        beat = HOME / "daemon.json"
        age = time.time() - json.loads(beat.read_text())["tick"] if beat.exists() else None
        if age is None or age > 1800:
            log("the gfx950 launcher is down: start `orchestrate.py daemon` from an SSH login")
    head = main_head()
    for cls in TIERS.values():
        tier = cls()
        try:
            tier.tick(head, args.nightly)
        except Exception as e:  # one tier's failure must not stop the other's
            log(f"{tier.name}: {type(e).__name__}: {e}")


def cmd_run(args) -> None:
    sha = resolve(args.ref)
    for name in (TIERS if args.tier == "all" else [args.tier]):
        tier = TIERS[name]()
        if tier.state["running"]:
            print(f"{name}: busy with {tier.state['running']['sha'][:10]}")
        elif not tier.begin(sha):
            print(f"{name}: cannot start now (no idle GPU)")


def cmd_status(_args) -> None:
    for name, cls in TIERS.items():
        st = cls().state
        run = st["running"]
        last = st["tested"].get(st["last"] or "", {})
        print(f"{name}: last {str(st['last'])[:10]} {last.get('description', '-')}")
        if run:
            print(f"  running {run['sha'][:10]}: {run['description']}")
        if last.get("url"):
            print(f"  report {last['url']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--no-cron", action="store_true", help="copy and unpack, but add no cron entries")
    i.set_defaults(fn=cmd_install)
    t = sub.add_parser("tick")
    t.add_argument("--nightly", action="store_true",
                   help="rerun main's head today, at the first tick that can start it")
    t.set_defaults(fn=cmd_tick)
    r = sub.add_parser("run")
    r.add_argument("ref")
    r.add_argument("--tier", choices=["all", *TIERS], default="all")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    d = sub.add_parser("daemon", help="tick every 10 minutes, with the docker access cron lacks")
    d.add_argument("--foreground", action="store_true")
    d.set_defaults(fn=cmd_daemon)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
