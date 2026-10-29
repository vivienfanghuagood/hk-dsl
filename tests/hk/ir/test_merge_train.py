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


# ---------------------------------------------------------------- append-only files: union

CL = """# Changelog

## Unreleased

What is on main after 0.8.0.

**CI and tests**
- An old CI entry from before.

## 0.8.0

**Fixes**
- A fix of 0.8.0.
"""


def _entry(cl, para=None, ci=None):
    if para:
        cl = cl.replace("What is on main after 0.8.0.\n", f"What is on main after 0.8.0.\n\n{para}\n")
    if ci:
        cl = cl.replace("- An old CI entry from before.\n", f"- An old CI entry from before.\n{ci}\n")
    return cl


def _union(o, b, t, tmp_path):
    paths = []
    for nm, x in (("o", o), ("b", b), ("t", t)):
        (tmp_path / nm).write_text(x)
        paths.append(str(tmp_path / nm))
    marked = subprocess.run(["git", "merge-file", "-p", "--diff3", f"--marker-size={mt.MARKS}", *paths],
                            capture_output=True, text=True).stdout
    return mt.union_hunks(marked)


def test_union_two_entries_at_the_same_place(tmp_path):
    o = _entry(CL, "**Ours: a feature.** Its text.", "- Ours: a CI change, long enough.")
    t = _entry(CL, "**Theirs: a fix.** Its text.", "- Theirs: another CI change, long enough.")
    r = _union(o, CL, t, tmp_path)
    assert mt.validate_union(o, CL, t, r) == []
    # both paragraphs, as two paragraphs; both list items, in one list; in Unreleased
    assert "**Ours: a feature.** Its text.\n\n**Theirs: a fix.** Its text.\n" in r
    assert "- Ours: a CI change, long enough.\n- Theirs: another CI change, long enough.\n" in r
    assert r.index("Theirs: another CI") < r.index("## 0.8.0")


def test_union_validation_refuses():
    v = mt.validate_union
    # markers left in
    o, t = _entry(CL, "**A.** text text"), _entry(CL, "**B.** text text")
    assert any("marker" in p for p in v(o, CL, t, o.replace("**A.**", "<<<<<<< ours\n**A.**")))
    # a subsection both sides opened, now twice in one version section
    o = CL.replace("## 0.8.0", "**Fixes**\n- Ours: a fix of the unreleased code.\n\n## 0.8.0")
    t = CL.replace("## 0.8.0", "**Fixes**\n- Theirs: another fix, unreleased.\n\n## 0.8.0")
    r = CL.replace("## 0.8.0", "**Fixes**\n- Ours: a fix of the unreleased code.\n\n"
                               "**Fixes**\n- Theirs: another fix, unreleased.\n\n## 0.8.0")
    assert any("twice" in p for p in v(o, CL, t, r))
    # an entry both sides added, twice
    o = _entry(CL, ci="- The same entry, added by both.")
    t = _entry(CL, ci="- The same entry, added by both.")
    r = _entry(CL, ci="- The same entry, added by both.\n- The same entry, added by both.")
    assert any("entry twice" in p for p in v(o, CL, t, r))
    # a duplicated version section
    assert any("section" in p for p in v(o, CL, o, o.replace("## 0.8.0", "## 0.8.0\n\n## 0.8.0")))


def test_union_entry_must_stay_in_its_version_section(tmp_path):
    # main released meanwhile: Unreleased became 0.9.0 and a new Unreleased opened above it
    o = CL.replace("## Unreleased\n\nWhat is on main after 0.8.0.\n",
                   "## Unreleased\n\nWhat is on main after 0.9.0.\n\n## 0.9.0\n\nReleased.\n")
    t = _entry(CL, ci="- Theirs: written for the unreleased section.")
    r = _union(o, CL, t, tmp_path)
    probs = mt.validate_union(o, CL, t, r)
    assert probs and any("left section" in p for p in probs), (probs, r)


@pytest.mark.skipif(_git_version() < (2, 40), reason="git merge-tree --merge-base needs git 2.40")
def test_stack_unions_changelog_and_the_branch_squashes_to_it(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "BASE_DIR", str(tmp_path))
    repo = tmp_path / "repo.git"
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull}

    def g(*a, inp=None):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                              env=env, input=inp).stdout.strip()

    def commit(parent, files, msg):
        g("read-tree", parent)
        for path, text in files.items():
            g("update-index", "--add", "--cacheinfo", f"100644,{g('hash-object', '-w', '--stdin', inp=text)},{path}")
        return g("commit-tree", g("write-tree"), "-p", parent, "-m", msg)

    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    g("read-tree", "--empty")
    blob = g("hash-object", "-w", "--stdin", inp=CL)
    g("update-index", "--add", "--cacheinfo", f"100644,{blob},CHANGELOG.md")
    main = g("commit-tree", g("write-tree"), "-m", "main")
    p1 = commit(main, {"CHANGELOG.md": _entry(CL, "**One.** text one", "- One: a CI entry here."), "a": "1"}, "p1")
    p2 = commit(main, {"CHANGELOG.md": _entry(CL, "**Two.** text two", "- Two: a CI entry here."), "b": "2"}, "p2")
    p3 = commit(main, {"CHANGELOG.md": _entry(CL, "**Three.** text three"), "a": "3"}, "p3: a conflicts with p1")

    st = {"name": "train/test", "created": "2026-10-28T13:00:00Z"}
    members = [{"number": n, "head": h, "title": f"p{n}"} for n, h in ((1, p1), (2, p2), (3, p3))]
    entries, conflicts = mt.stack(st, members, main)
    assert [(e["number"], e["union"]) for e in entries] == [(1, []), (2, ["CHANGELOG.md"])]
    assert [(c["number"], c["files"]) for c in conflicts] == [(3, ["CHANGELOG.md", "a"])]  # not only CHANGELOG
    cl = g("cat-file", "blob", f"{entries[1]['tree']}:CHANGELOG.md")
    assert "**One.** text one\n\n**Two.** text two" in cl and "<<<" not in cl

    # the merge: #1 squashes as is; #2's branch is brought to the train's tree (a merge of main into
    # it), and GitHub's squash of that head onto main is the train's tree
    def squash(cur, head):
        base = g("merge-base", cur, head)
        out = subprocess.run(["git", "-C", str(repo), "merge-tree", "--write-tree", f"--merge-base={base}", cur, head],
                             capture_output=True, text=True, env=env)
        return out.returncode, out.stdout.splitlines()[0]
    rc, t1 = squash(main, p1)
    assert rc == 0 and t1 == entries[0]["tree"]
    main1 = g("commit-tree", t1, "-p", main, "-m", "squash 1")
    assert squash(main1, p2)[0] == 1          # GitHub would refuse #2 as it is
    head2 = g("commit-tree", entries[1]["tree"], "-p", p2, "-p", main1, "-m", "bring #2 to the train")
    rc, t2 = squash(main1, head2)
    assert rc == 0 and t2 == entries[1]["tree"]


def test_readmission_keeps_the_original_ready_time():
    cs = [_c("07:28:44", "READY 0d7875d (contains main af17846)"),
          _c("13:01:02", "merge-train: kicked from train `train/20261028-1300` (head 0d7875d): it conflicts"),
          _c("13:40:00", "READY 0d7875d (readmitted by merge-train: kicked from `train/20261028-1300` only for a "
                         "conflict in CHANGELOG.md, which the train now resolves by union; original READY "
                         "2026-10-28T07:28:44Z keeps its place in the queue)")]
    assert mt.ready_comment(HEAD, cs[:2])[0] is None
    assert mt.ready_comment(HEAD, cs)[0]["created_at"] == "2026-10-28T07:28:44Z"
