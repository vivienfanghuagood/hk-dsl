"""The merge train's rules (scripts/ci/merge_train.py; docs/testing.md, "The merge train"):
which comments make a PR READY, the window a build belongs to, and that the train stacks each PR
exactly as GitHub's squash merge applies it (on a scratch repository; no network)."""

import datetime as dt
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("merge_train", ROOT / "scripts" / "ci" / "merge_train.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

HEAD = "0d7875d1a2b3c4d5e6f708192a3b4c5d6e7f8091"


@pytest.mark.parametrize("body", [
    "READY 0d7875d (contains main eaaae34). CI green.",
    "READY: head 0d7875d on main 8520e49. CI 40596755002 success",
    "READY at head 0d7875d (based on main eaaae34): CI is green",
    "READY #107 0d7875d (contains main eaaae34)",
    "**READY** at head **0d7875d**, on main 216f52f (#60, the latest).",
    "READY. Head 0d7875d1, which is on the latest main",
])
def test_ready_formats_in_use(body):
    assert mt.classify(body) == "ready"
    assert any(HEAD.startswith(t) for t in mt.HEX.findall(mt.first_line(body)))


@pytest.mark.parametrize("body,kind", [
    ("Rebased onto main 6b6ec6f (#103). New head a831f1d; READY follows when green.", None),
    ("READY follows when CI is green", None),
    ("Superseded: the READY above was for ad82efd", None),
    ("NOT READY: a regression in the decode at 4k", "revoke"),
    ("merge-train: kicked from train `train/20261028-1300` (head 0d7875d): it conflicts", "revoke"),
    ("merge-train: in train `train/20261028-1300` at position 2 of 3", None),
])
def test_other_comments(body, kind):
    assert mt.classify(body) == kind


def _c(t, body, who="OWNER"):
    return {"created_at": f"2026-10-28T{t}Z", "body": body, "author_association": who,
            "html_url": "u"}


def test_ready_rules():
    rc = mt.ready_comment
    # the earliest READY for the current head is its place in the queue
    r, _, _ = rc(HEAD, [_c("05:00:00", "READY 0d7875d"), _c("06:00:00", "READY 0d7875d again")])
    assert r["created_at"] == "2026-10-28T05:00:00Z"
    # a READY for another head (before a push) does not count
    r, other, _ = rc(HEAD, [_c("05:00:00", "READY 1111111 (contains main 2222222)")])
    assert r is None and other[0] == "1111111"
    # withdrawn, then READY again: queued at the new time
    cs = [_c("05:00:00", "READY 0d7875d"), _c("05:30:00", "NOT READY: found a bug")]
    assert rc(HEAD, cs)[0] is None
    assert rc(HEAD, cs + [_c("07:00:00", "READY 0d7875d")])[0]["created_at"] == "2026-10-28T07:00:00Z"
    # a kick by the train withdraws it too
    cs = [_c("05:00:00", "READY 0d7875d"), _c("13:05:00", "merge-train: kicked from train `t` (head 0d7875d): red")]
    assert rc(HEAD, cs)[0] is None
    # anyone can comment on a public repository: only the owner and collaborators count
    assert rc(HEAD, [_c("05:00:00", "READY 0d7875d", who="NONE")])[0] is None
    assert rc(HEAD, [_c("05:00:00", "READY 0d7875d", who="COLLABORATOR")])[0] is not None


def _t(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


@pytest.mark.parametrize("when,name", [
    ("2026-10-28T13:00:00", "train/20261028-1300"),
    ("2026-10-28T15:59:00", "train/20261028-1300"),
    ("2026-10-28T01:30:00", "train/20261028-0100"),
    ("2026-10-29T00:30:00", None),
    ("2026-10-28T04:00:00", None),
])
def test_windows(when, name):
    if name:
        assert mt.window_name(_t(when), False)[0] == name
    else:
        with pytest.raises(mt.Fail):
            mt.window_name(_t(when), False)
        assert mt.window_name(_t(when), True)[0] == "train/" + _t(when).strftime("%Y%m%d-%H%M")


def _git_version():
    out = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout
    return tuple(int(x) for x in re.findall(r"\d+", out)[:2])


@pytest.mark.skipif(_git_version() < (2, 40), reason="git merge-tree --merge-base needs git 2.40")
def test_stack_is_githubs_squash(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "BASE_DIR", str(tmp_path))
    repo = tmp_path / "repo.git"
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull}

    def g(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                              env=env).stdout.strip()

    def commit(parent, files, msg):
        g("read-tree", "--empty") if parent is None else g("read-tree", parent)
        for path, text in files.items():
            blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=text,
                                  check=True, capture_output=True, text=True, env=env).stdout.strip()
            g("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        t = g("write-tree")
        return g("commit-tree", t, *(["-p", parent] if parent else []), "-m", msg)

    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    lines = "".join(f"line {i}\n" for i in range(20))
    old = commit(None, {"a.txt": lines, "b.txt": lines}, "old main")
    main = commit(old, {"c.txt": "c\n"}, "main")
    p1 = commit(main, {"a.txt": lines.replace("line 3\n", "line 3 by p1\n")}, "p1")
    p2 = commit(old, {"b.txt": lines.replace("line 9\n", "line 9 by p2\n")}, "p2, behind main")
    p3 = commit(p1, {"a.txt": lines.replace("line 3\n", "line 3 by p1\n").replace("line 15\n", "line 15 by p3\n")},
                "p3, stacked on p1")
    p4 = commit(main, {"a.txt": lines.replace("line 3\n", "line 3 by p4\n")}, "p4, conflicts with p1")

    st = {"name": "train/test", "created": "2026-10-28T13:00:00Z"}
    members = [{"number": n, "head": h, "title": f"p{n}"} for n, h in ((1, p1), (2, p2), (3, p3), (4, p4))]
    entries, conflicts = mt.stack(st, members, main)
    assert [e["number"] for e in entries] == [1, 2, 3]
    assert [(c["number"], c["files"], c["against"]) for c in conflicts] == [(4, ["a.txt"], "#1, #3")]

    # GitHub: squash-merge each in turn onto the main the one before left
    cur = main
    for e in entries:
        base = g("merge-base", cur, e["head"])
        t = g("merge-tree", "--write-tree", f"--merge-base={base}", cur, e["head"]).splitlines()[0]
        cur = g("commit-tree", t, "-p", cur, "-m", f"squash {e['number']}")
        assert g("rev-parse", f"{cur}^{{tree}}") == e["tree"] == g("rev-parse", f"{e['commit']}^{{tree}}")
    # deterministic: the same stack is the same commits
    again, _ = mt.stack(st, members, main)
    assert [e["commit"] for e in again] == [e["commit"] for e in entries]
