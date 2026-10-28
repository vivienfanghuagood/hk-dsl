#!/usr/bin/env python3
"""merge_train.py: the merge train (docs/testing.md, "The merge train").

Twice a day (01:00Z and 13:00Z) every READY pull request is stacked, in READY order, on one
branch `train/<YYYYMMDD>-<HHMM>`: main, then each PR exactly as GitHub's squash merge will apply
it. CI (ci.yml) and the GPU tier (gpu.yml) run once on that branch. If both are green the PRs are
squash-merged in the same order without a rerun each, and main's tree is checked against the
train's after every merge. A PR that conflicts while stacking, or that turns the train red, is
kicked out and the train rebuilt without it.

    merge_train.py plan    [--prs N,M] [--include-drafts]       READY PRs, in order (read-only)
    merge_train.py build   [--name train/...] [--prs N,M] [--off-window] [--rebuild] [--workflows ci,gpu]
    merge_train.py status  [--wait]                             runs, jobs; on red the suspects
    merge_train.py kick N  [--reason TEXT]                      drop #N, rebuild, rerun
    merge_train.py bisect                                       one step of a bisection over prefixes
    merge_train.py retry                                        rerun the failed jobs (a flake)
    merge_train.py merge   --i-am-the-merger [--admin-bypass]   squash-merge in order, check trees
    merge_train.py cleanup [--abandon] [--prune-yes]            delete the train's refs, prune

Every command takes --dry-run (no push, dispatch, comment, merge or delete; local git only) and
--name (default: the current train). Every step is idempotent: rerunning a command after a crash
resumes it. State and a log of every command are kept in $MERGE_TRAIN_DIR/<train>/.

READY: a comment whose first line starts with `READY` and names the PR's current head (a 7-40
character prefix), e.g. `READY <head> (contains main <sha>)`, by the owner or a collaborator.
Its time is the PR's place in the queue. A later comment starting `NOT READY`, or a kick by the
train, withdraws it; a new READY after that queues the PR again (at the new time). A PR is also
not READY if it is a draft, if its base is not main, or if its head lacks green runs of main's
required checks (it could not be merged without an admin bypass).

Environment: MERGE_TRAIN_REPO (owner/name), MERGE_TRAIN_DIR (state; default
~/.local/state/hk-merge-train), MERGE_TRAIN_PRUNE (a command cleanup runs; MERGE_TRAIN_PRUNE_YES
with --prune-yes),
MERGE_TRAIN_GIT_NAME / MERGE_TRAIN_GIT_EMAIL (author of the train's commits). Needs git >= 2.40
(merge-tree --write-tree --merge-base) and an authenticated `gh`.
"""
import argparse
import datetime as dt
import fcntl
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import zipfile

REPO = os.environ.get('MERGE_TRAIN_REPO', 'vivienfanghuagood/hk-dsl')
BASE_DIR = os.path.expanduser(os.environ.get('MERGE_TRAIN_DIR', '~/.local/state/hk-merge-train'))
PRUNE_CMD = os.environ.get('MERGE_TRAIN_PRUNE', '')          # cleanup runs it (e.g. a dry-run listing)
PRUNE_YES_CMD = os.environ.get('MERGE_TRAIN_PRUNE_YES', '')  # cleanup --prune-yes runs this instead
GIT_NAME = os.environ.get('MERGE_TRAIN_GIT_NAME', 'merge-train')
GIT_EMAIL = os.environ.get('MERGE_TRAIN_GIT_EMAIL', 'merge-train@localhost')

MAIN = 'main'
WINDOWS = ((1, 0), (13, 0))          # UTC
WINDOW_GRACE = dt.timedelta(hours=3)  # a build this long after a window still belongs to it
WORKFLOWS = {'ci': 'ci.yml', 'gpu': 'gpu.yml'}
OBSERVED = ('docs.yml',)              # push-triggered on the train branch when docs change: counts if present
GPU_JOBS = ('gfx950', 'gfx950-vllm', 'gfx1100', 'gfx1201')
ACTIONS_APP = 15368                   # GitHub Actions: the app main's required checks are pinned to
TRUSTED = {'OWNER', 'MEMBER', 'COLLABORATOR'}
MARK = 'merge-train:'
INERT = re.compile(r'^(bench/|.*\.md$)')  # cannot change what a test exercises (not docs/api/)
TEST_ID = re.compile(r'(?:FAILED|ERROR)\s+(tests/[\w./-]+\.py)(?:::[^\s]*)?')
HEX = re.compile(r'(?<![0-9a-zA-Z])[0-9a-f]{7,40}(?![0-9a-zA-Z])')


class Fail(Exception):
    pass


# --------------------------------------------------------------------------------------------
# logging, shell, GitHub

class Log:
    def __init__(self):
        self.path = None
        self.dry = False

    def to(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write(self, msg):
        if self.path:
            with open(self.path, 'a') as f:
                f.write(f"{dt.datetime.now(dt.timezone.utc):%FT%TZ} {'DRY ' if self.dry else ''}{msg}\n")

    def say(self, msg=''):
        print(msg, flush=True)
        self.write(msg)


LOG = Log()


def sh(cmd, *, input=None, check=True, binary=False, env=None, quiet=False):
    r = subprocess.run(cmd, input=input, capture_output=True, text=not binary,
                       env={**os.environ, **(env or {})})
    err = r.stderr if not binary else r.stderr.decode(errors='replace')
    if not quiet:
        LOG.write(f"$ {shlex.join(cmd)} -> {r.returncode}" + (f" | {err.strip()[-400:]}" if r.returncode else ''))
    if check and r.returncode:
        raise Fail(f"{shlex.join(cmd[:4])}... failed ({r.returncode}): {err.strip()[-600:]}")
    return r


def api(path, method='GET', body=None, raw=False, check=True):
    cmd = ['gh', 'api', '-X', method, path]
    inp = None
    if body is not None:
        cmd += ['--input', '-']
        inp = json.dumps(body).encode() if raw else json.dumps(body)
    r = sh(cmd, input=inp, check=False, binary=raw, quiet=(method == 'GET'))
    if r.returncode:
        out = r.stdout.decode(errors='replace') if raw else r.stdout
        err = r.stderr.decode(errors='replace') if raw else r.stderr
        if method != 'GET':
            LOG.write(f"api {method} {path} failed: {out[-300:]} {err[-300:]}")
        if check:
            raise Fail(f"gh api {method} {path}: {out[-300:]} {err[-300:]}")
        return None
    if raw:
        return r.stdout
    return json.loads(r.stdout) if r.stdout.strip() else {}


def api_list(path, key=None):
    out, page = [], 1
    sep = '&' if '?' in path else '?'
    while True:
        got = api(f"{path}{sep}per_page=100&page={page}")
        items = got[key] if key else got
        out += items
        if len(items) < 100:
            return out
        page += 1


def write_api(desc, path, method='POST', body=None, check=True):
    """Every write to GitHub goes through here, so --dry-run covers them all."""
    if LOG.dry:
        LOG.say(f"  [dry-run] would {desc}")
        return None
    LOG.write(f"{desc}: {method} {path}")
    return api(path, method, body, check=check)


def comment(n, body):
    write_api(f"comment on #{n}: {body.splitlines()[0][:100]}",
              f"repos/{REPO}/issues/{n}/comments", body={'body': body})


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


# --------------------------------------------------------------------------------------------
# git: a bare clone of its own, so no worktree is touched

def repo_dir():
    return os.path.join(BASE_DIR, 'repo.git')


def git(*args, check=True, env=None):
    return sh(['git', '-C', repo_dir(), *args], check=check, env=env).stdout.strip()


def fetch():
    d = repo_dir()
    if not os.path.isdir(d):
        os.makedirs(BASE_DIR, exist_ok=True)
        sh(['git', 'init', '-q', '--bare', d])
        git('remote', 'add', 'origin', f'https://github.com/{REPO}.git')
    git('fetch', '-q', '--prune', 'origin', f'+refs/heads/{MAIN}:refs/remotes/origin/{MAIN}',
        '+refs/pull/*/head:refs/mt/pull/*', '+refs/heads/train/*:refs/remotes/origin/train/*')


def rev(ref):
    return git('rev-parse', '--verify', '-q', ref, check=False) or None


def tree(ref):
    return git('rev-parse', f'{ref}^{{tree}}')


def changed_files(base, head):
    return set(filter(None, git('diff', '--name-only', base, head).splitlines()))


def squash_tree(onto, head, main_sha):
    """The tree GitHub's squash merge of `head` gives when main's tree is `onto`'s: a three-way
    merge on merge-base(main, head). Squash commits are never ancestors of a PR head, so that base
    does not change as earlier PRs merge (checked against the repo's merged PRs: all reproduce).
    Returns (tree, None) or (None, [conflicting paths])."""
    base = git('merge-base', main_sha, head)
    r = sh(['git', '-C', repo_dir(), 'merge-tree', '--write-tree', '--name-only', '--no-messages',
            f'--merge-base={base}', onto, head], check=False)
    lines = r.stdout.strip().splitlines()
    if r.returncode == 0:
        return lines[0], None
    if r.returncode == 1:
        return None, lines[1:] or ['?']
    raise Fail(f"git merge-tree failed: {r.stderr.strip()}")


# --------------------------------------------------------------------------------------------
# READY

def first_line(body):
    for line in (body or '').splitlines():
        if line.strip():
            return re.sub(r'^[\s>*_`]+', '', line).strip()
    return ''


def classify(body):
    line = first_line(body)
    if re.match(r'(NOT[ -]READY|UNREADY)\b', line, re.I):
        return 'revoke'
    if line.startswith(MARK) and re.search(r'\bkicked\b', line):
        return 'revoke'
    if re.match(r'READY\b', line) and not re.match(r'READY\s+(follows|when)\b', line):
        return 'ready'
    return None


def required_checks():
    got = api(f"repos/{REPO}/branches/{MAIN}/protection/required_status_checks", check=False)
    if got and got.get('checks') is not None:
        return {c['context']: c.get('app_id') for c in got['checks']}, bool(got.get('strict'))
    return {'gfx950': ACTIONS_APP, 'gfx1100': ACTIONS_APP}, False


def head_checks(sha):
    runs = api_list(f"repos/{REPO}/commits/{sha}/check-runs", 'check_runs')
    by = {}
    for r in runs:
        by.setdefault(r['name'], []).append(r)
    return by


def check_ok(by, name, app=None):
    return any(r['conclusion'] in ('success',) and (app is None or r['app']['id'] == app)
               for r in by.get(name, []))


def ready_comment(head, comments):
    """-> (the READY comment that stands for `head`, (sha, time) of a READY for another head, the
    last withdrawal). Only trusted authors count; the earliest READY since the last withdrawal wins."""
    ready, other, revoked = None, None, None
    for c in sorted(comments, key=lambda c: c['created_at']):
        if c.get('author_association') not in TRUSTED:
            continue
        k = classify(c['body'])
        if k == 'revoke':
            ready, revoked = None, c
        elif k == 'ready':
            toks = HEX.findall(first_line(c['body']))
            if any(head.startswith(t) for t in toks):
                ready = ready or c
            else:
                other = (toks[0] if toks else '?', c['created_at'])
    return ready, other, revoked


def evaluate(pr, required, include_drafts=False, ignore_head_checks=False):
    """-> dict(number, title, head, ready_at, ready_url, ok, why, warn)"""
    n, head = pr['number'], pr['head']['sha']
    e = {'number': n, 'title': pr['title'], 'head': head, 'ready_at': None, 'ready_url': None,
         'draft': pr['draft'], 'base': pr['base']['ref'], 'ok': False, 'why': '', 'warn': []}
    ready, other, revoked = ready_comment(head, api_list(f"repos/{REPO}/issues/{n}/comments"))
    if ready:
        e['ready_at'], e['ready_url'] = ready['created_at'], ready['html_url']
    if pr['draft'] and not include_drafts:
        e['why'] = 'draft'
    elif e['base'] != MAIN:
        e['why'] = f"base is {e['base']}, not {MAIN} (stacked: retarget after its base merges)"
    elif not ready:
        if revoked and (not other or other[1] < revoked['created_at']):
            e['why'] = f"READY withdrawn at {revoked['created_at']} ({first_line(revoked['body'])[:50]})"
        elif other:
            e['why'] = f"READY was for {other[0]}, head is now {head[:7]}"
        else:
            e['why'] = 'no READY comment'
    else:
        by = head_checks(head)
        missing = [c for c, app in required.items() if not check_ok(by, c, app)]
        if missing and not ignore_head_checks:
            e['why'] = f"head lacks green required check(s) {', '.join(missing)}"
        else:
            e['ok'] = True
            if missing:
                e['warn'].append(f"head lacks green {', '.join(missing)} (ignored: a test of the train; "
                                 f"merge would need --admin-bypass)")
        gpu = sum(check_ok(by, j) for j in GPU_JOBS)
        if e['ok'] and gpu < len(GPU_JOBS):
            e['warn'].append(f"head has {gpu}/{len(GPU_JOBS)} GPU jobs green")
        # a push after the READY comment (same sha pushed again, or the READY named a commit
        # before it was pushed) is odd enough to show
        pushes = [ev['created_at'] for ev in api_list(f"repos/{REPO}/issues/{n}/timeline")
                  if ev.get('event') == 'head_ref_force_pushed']
        if pushes and max(pushes) > ready['created_at']:
            e['warn'].append(f"READY ({ready['created_at']}) predates the last force-push ({max(pushes)})")
    return e


def plan_prs(only=None, include_drafts=False, ignore_head_checks=False):
    required, strict = required_checks()
    prs = api_list(f"repos/{REPO}/pulls?state=open")
    if only:
        prs = [p for p in prs if p['number'] in only]
        missing = set(only) - {p['number'] for p in prs}
        if missing:
            raise Fail(f"not open: {sorted(missing)}")
    ev = [evaluate(p, required, include_drafts, ignore_head_checks) for p in prs]
    ready = sorted([e for e in ev if e['ok']], key=lambda e: (e['ready_at'], e['number']))
    rest = sorted([e for e in ev if not e['ok']], key=lambda e: e['number'])
    return ready, rest, required, strict


def print_plan(ready, rest, required, strict):
    LOG.say(f"main's required checks: {', '.join(required)} (strict={strict}); "
            f"train jobs: ci.yml + gpu.yml ({', '.join(GPU_JOBS)})")
    LOG.say(f"READY, in order ({len(ready)}):")
    for k, e in enumerate(ready, 1):
        LOG.say(f"  {k}. #{e['number']:<4} READY {e['ready_at']}  head {e['head'][:7]}  {e['title'][:70]}")
        for w in e['warn']:
            LOG.say(f"       warning: {w}")
    if rest:
        LOG.say(f"not READY ({len(rest)}):")
        for e in rest:
            LOG.say(f"     #{e['number']:<4} head {e['head'][:7]}  {e['why']}  -- {e['title'][:50]}")


# --------------------------------------------------------------------------------------------
# train state

def train_id(name):
    return name.split('/', 1)[1]


def state_path(name):
    return os.path.join(BASE_DIR, train_id(name), 'state.json')


def load(name):
    p = state_path(name)
    if not os.path.exists(p):
        raise Fail(f"no train {name} ({p})")
    with open(p) as f:
        return json.load(f)


def save(st):
    if LOG.dry:
        return
    p = state_path(st['name'])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.tmp', 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(p + '.tmp', p)
    with open(os.path.join(BASE_DIR, 'current'), 'w') as f:
        f.write(st['name'] + '\n')


def current_name(args):
    if getattr(args, 'name', None):
        return args.name
    p = os.path.join(BASE_DIR, 'current')
    if os.path.exists(p):
        return open(p).read().strip()
    raise Fail("no current train: give --name")


def window_name(t, off_window):
    for day in (t.date(), t.date() - dt.timedelta(days=1)):
        for h, m in WINDOWS:
            w = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=dt.timezone.utc)
            if w <= t < w + WINDOW_GRACE:
                return f"train/{w:%Y%m%d-%H%M}", iso(w)
    if not off_window:
        raise Fail(f"{iso(t)} is outside the windows ({', '.join(f'{h:02d}:{m:02d}Z' for h, m in WINDOWS)}, "
                   f"+{WINDOW_GRACE}); outside them only urgent fixes merge. --off-window to build anyway.")
    return f"train/{t:%Y%m%d-%H%M}", iso(t)


def rnd(st):
    return st['rounds'][-1] if st['rounds'] else None


# --------------------------------------------------------------------------------------------
# build

def stack(st, members, main_sha):
    """Stack `members` (dicts with number/head/title) on main_sha. Returns (entries, conflicts)."""
    env = {'GIT_AUTHOR_NAME': GIT_NAME, 'GIT_AUTHOR_EMAIL': GIT_EMAIL, 'GIT_COMMITTER_NAME': GIT_NAME,
           'GIT_COMMITTER_EMAIL': GIT_EMAIL, 'GIT_AUTHOR_DATE': st['created'], 'GIT_COMMITTER_DATE': st['created']}
    cur, entries, conflicts = main_sha, [], []
    for m in members:
        if sh(['git', '-C', repo_dir(), 'merge-base', '--is-ancestor', m['head'], main_sha], check=False).returncode == 0:
            conflicts.append({**m, 'files': [], 'against': 'already in main'})
            continue
        t, files = squash_tree(cur, m['head'], main_sha)
        if t is None:
            mine = set(files)
            against = [f"#{e['number']}" for e in entries
                       if mine & changed_files(git('merge-base', main_sha, e['head']), e['head'])]
            conflicts.append({**m, 'files': files, 'against': ', '.join(against) or f'{MAIN} {main_sha[:7]}'})
            continue
        k = len(entries) + 1
        msg = (f"{st['name']} {k}: {m['title']} (#{m['number']})\n\n"
               f"The squash of #{m['number']} at {m['head']}, as GitHub will apply it\n"
               f"(three-way merge on merge-base({MAIN} {main_sha[:12]}, head)).\n")
        c = sh(['git', '-C', repo_dir(), 'commit-tree', t, '-p', cur, '-p', m['head'], '-m', msg],
               env=env).stdout.strip()
        entries.append({'number': m['number'], 'title': m['title'], 'head': m['head'],
                        'ready_at': m.get('ready_at'), 'position': k, 'commit': c, 'tree': t})
        cur = c
    return entries, conflicts


def cancel_runs(r):
    for wf, d in (r.get('runs') or {}).items():
        if d and d.get('id'):
            run = api(f"repos/{REPO}/actions/runs/{d['id']}", check=False)
            if run and run['status'] != 'completed':
                write_api(f"cancel {wf} run {d['id']}", f"repos/{REPO}/actions/runs/{d['id']}/cancel", check=False)


def find_run(wf, ref, sha, since):
    q = urllib.parse.urlencode({'branch': ref, 'event': 'workflow_dispatch', 'per_page': 30})
    runs = api(f"repos/{REPO}/actions/workflows/{wf}/runs?{q}")['workflow_runs']
    runs = [x for x in runs if x['head_sha'] == sha and x['created_at'] >= since]
    return max(runs, key=lambda x: x['created_at']) if runs else None


def dispatch(st, r, ref, sha, wfs, key='runs'):
    """Dispatch each workflow on ref once per (ref, sha); adopt a run that already exists."""
    r.setdefault(key, {})
    for wf in wfs:
        d = r[key].get(wf)
        if d and d.get('id'):
            continue
        since = (d or {}).get('dispatched') or r['at']
        run = find_run(wf, ref, sha, since)
        if not run:
            if d is None or not d.get('dispatched'):
                t0 = iso(now() - dt.timedelta(seconds=5))
                write_api(f"dispatch {wf} on {ref} ({sha[:7]})",
                          f"repos/{REPO}/actions/workflows/{wf}/dispatches", body={'ref': ref})
                if LOG.dry:
                    continue
                r[key][wf] = {'dispatched': t0}
                save(st)
                since = t0
            for _ in range(24):
                run = find_run(wf, ref, sha, since)
                if run:
                    break
                time.sleep(5)
        if run:
            r[key][wf] = {**(r[key].get(wf) or {}), 'id': run['id'], 'url': run['html_url']}
            LOG.say(f"  {wf}: {run['html_url']}")
            save(st)
        elif not LOG.dry:
            LOG.say(f"  {wf}: dispatched, run not visible yet (rerun the command to pick it up)")


def push_ref(ref, sha, expect):
    """Create or move ref (never main, never a PR branch) with a lease on its old value."""
    assert ref.startswith('train/'), ref
    remote = git('ls-remote', 'origin', f'refs/heads/{ref}').split('\t')[0] or None
    if remote == sha:
        return
    if remote and remote != expect:
        raise Fail(f"{ref} is at {remote[:7]} on GitHub, expected {(expect or 'nothing')[:7]}: someone else moved it")
    if LOG.dry:
        LOG.say(f"  [dry-run] would push {sha[:7]} to {ref}")
        return
    git('push', '-q', f'--force-with-lease=refs/heads/{ref}:{remote or ""}', 'origin', f'{sha}:refs/heads/{ref}')
    LOG.say(f"  pushed {ref} = {sha[:7]}")


def new_round(st, members, why):
    fetch()
    main_sha = rev(f'origin/{MAIN}')
    entries, conflicts = stack(st, members, main_sha)
    prev = rnd(st)
    r = {'round': len(st['rounds']) + 1, 'at': iso(now()), 'why': why, 'base': main_sha,
         'stack': entries, 'conflicts': conflicts, 'head': entries[-1]['commit'] if entries else None,
         'runs': {}, 'commented': []}
    LOG.say(f"round {r['round']} of {st['name']} on {MAIN} {main_sha[:7]} ({why}):")
    for e in entries:
        LOG.say(f"  {e['position']}. #{e['number']} head {e['head'][:7]} -> {e['commit'][:7]}  {e['title'][:60]}")
    for c in conflicts:
        LOG.say(f"  KICK #{c['number']}: conflicts with {c['against']} in {', '.join(c['files'][:6])}")
    for c in conflicts:
        if str(c['number']) not in st['kicked']:
            do_kick(st, c['number'], c['head'],
                    (f"it is already in {MAIN}" if c['against'] == 'already in main' else
                     f"it conflicts with {c['against']} in {', '.join(c['files'][:10])}"),
                    rebuild=False, round_no=r['round'])
    if prev and prev['head'] == r['head'] and prev['base'] == main_sha:
        LOG.say("  same stack as the round before: nothing to rebuild")
        return prev
    if prev:
        cancel_runs(prev)
    if LOG.dry:
        st['rounds'].append(r)
        return r
    st['rounds'].append(r)
    save(st)
    return r


def launch(st, r, wfs):
    """Push the round's head and dispatch its workflows; comment on its PRs. Resumable."""
    if not r['head']:
        LOG.say("the train is empty: nothing to run")
        return
    prev = [x['head'] for x in st['rounds'][:-1] if x['head']]
    remote = git('ls-remote', 'origin', f"refs/heads/{st['name']}").split('\t')[0] or None
    push_ref(st['name'], r['head'], remote if remote in prev + [r['head']] else (prev[-1] if prev else None))
    dispatch(st, r, st['name'], r['head'], wfs)
    n = len(r['stack'])
    for e in r['stack']:
        if e['number'] in r['commented']:
            continue
        links = ' '.join(f"[{wf}]({d['url']})" for wf, d in r['runs'].items() if d.get('url'))
        comment(e['number'], f"{MARK} in train `{st['name']}` at position {e['position']} of {n} "
                             f"(round {r['round']}; train head {r['head'][:7]} on {MAIN} {r['base'][:7]}). "
                             f"{links}\n\nThe train is tested once; if it is green the PRs are squash-merged "
                             f"in this order. Please do not push to this branch until it merges or is kicked.")
        if not LOG.dry:
            r['commented'].append(e['number'])
            save(st)


def cmd_plan(a):
    LOG.to(os.path.join(BASE_DIR, 'log'))
    ready, rest, required, strict = plan_prs(a.prs, a.include_drafts, a.ignore_head_checks)
    print_plan(ready, rest, required, strict)
    return 0


def cmd_build(a):
    t = now()
    if a.name:
        name, window = a.name, iso(t)
        if not re.fullmatch(r'train/[A-Za-z0-9._-]+', name):
            raise Fail("--name must be train/<something>")
    else:
        name, window = window_name(t, a.off_window)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    LOG.say(f"build {name} (window {window})")
    wfs = [WORKFLOWS[w] for w in a.workflows.split(',')]
    exists = os.path.exists(state_path(name))
    if exists:
        st = load(name)
        if st.get('merged') or st.get('cleaned'):
            raise Fail(f"{name} has already merged or been cleaned up")
        r = rnd(st)
        if a.rebuild:
            members = refreshed_members(st)
            r = new_round(st, members, 'rebuild')
        else:
            LOG.say(f"{name} exists (round {r['round']}): resuming its push, dispatches and comments; "
                    f"--rebuild to restack it on the current {MAIN}")
        launch(st, r, st.get('workflows', wfs))
        return 0
    fetch()
    ready, rest, required, strict = plan_prs(a.prs, a.include_drafts, a.ignore_head_checks)
    print_plan(ready, rest, required, strict)
    if not ready:
        LOG.say("nothing is READY: no train")
        return 0
    st = {'name': name, 'window': window, 'created': iso(t), 'repo': REPO, 'workflows': wfs,
          'filter': a.prs, 'include_drafts': a.include_drafts, 'ignore_head_checks': a.ignore_head_checks,
          'members': [{k: e[k] for k in ('number', 'title', 'head', 'ready_at', 'ready_url')} for e in ready],
          'kicked': {}, 'rounds': [], 'merged': [], 'bisect': None}
    r = new_round(st, st['members'], 'window')
    launch(st, r, wfs)
    return 0


def refreshed_members(st):
    """The train's members minus kicked ones; one whose head moved stays only if the new head is
    READY again (it then rides at its new head), else it is kicked."""
    required, _ = required_checks()
    out = []
    for m in st['members']:
        if str(m['number']) in st['kicked']:
            continue
        pr = api(f"repos/{REPO}/pulls/{m['number']}")
        if pr['state'] != 'open':
            LOG.say(f"  #{m['number']} is {pr['state']}: dropped")
            st['kicked'][str(m['number'])] = {'reason': f"{pr['state']}", 'at': iso(now()), 'head': m['head']}
            continue
        if pr['head']['sha'] != m['head']:
            e = evaluate(pr, required, st.get('include_drafts'), st.get('ignore_head_checks'))
            if e['ok']:
                LOG.say(f"  #{m['number']} moved {m['head'][:7]} -> {e['head'][:7]} and is READY again: new head")
                m.update(head=e['head'], ready_at=e['ready_at'], ready_url=e['ready_url'])   # kept in st
            else:
                do_kick(st, m['number'], pr['head']['sha'],
                        f"its head moved from {m['head'][:7]} to {pr['head']['sha'][:7]} after the train was built",
                        rebuild=False)
                continue
        out.append(m)
    return out


# --------------------------------------------------------------------------------------------
# status

def run_jobs(run_id):
    run = api(f"repos/{REPO}/actions/runs/{run_id}")
    jobs = api_list(f"repos/{REPO}/actions/runs/{run_id}/jobs", 'jobs')
    return run, jobs


def verdict(runs):
    """runs: {wf: (run, jobs)} -> 'green' | 'red' | 'running'"""
    if not runs:
        return 'running'
    bad = any(j['conclusion'] in ('failure', 'cancelled', 'timed_out', 'action_required', 'startup_failure')
              for _, jobs in runs.values() for j in jobs)
    bad |= any(run['conclusion'] not in (None, 'success') for run, _ in runs.values())
    done = all(run['status'] == 'completed' for run, _ in runs.values())
    if bad and done:
        return 'red'
    if not done:
        return 'red-running' if bad else 'running'
    if 'gpu.yml' in runs:
        names = {j['name'] for j in runs['gpu.yml'][1] if j['conclusion'] == 'success'}
        if not set(GPU_JOBS) <= names:
            return 'red'
    return 'green'


def collect(st, r, ref=None, sha=None, key='runs'):
    out = {}
    for wf, d in (r.get(key) or {}).items():
        if d.get('id'):
            out[wf] = run_jobs(d['id'])
    sha = sha or r['head']
    for wf in OBSERVED:   # e.g. docs.yml, when the push of the train touched docs
        q = urllib.parse.urlencode({'branch': ref or st['name'], 'per_page': 10})
        got = api(f"repos/{REPO}/actions/workflows/{wf}/runs?{q}", check=False) or {'workflow_runs': []}
        for run in got['workflow_runs']:
            if run['head_sha'] == sha:
                out[wf] = run_jobs(run['id'])
                break
    return out


def failing_test_files(run, job):
    text = api(f"repos/{REPO}/actions/jobs/{job['id']}/logs", raw=True, check=False) or b''
    found = set(TEST_ID.findall(text.decode(errors='replace')))
    arts = api(f"repos/{REPO}/actions/runs/{run['id']}/artifacts", check=False) or {'artifacts': []}
    for art in arts['artifacts']:
        if art['name'] == f"{job['name']}-report" and not art['expired']:
            blob = api(f"repos/{REPO}/actions/artifacts/{art['id']}/zip", raw=True, check=False)
            if blob:
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    for nm in z.namelist():
                        if nm.endswith(('.txt', '.log', '.json')):
                            found |= set(TEST_ID.findall(z.read(nm).decode(errors='replace')))
    return found


def main_result(wf, sha, job_name):
    """The same job on main's own run for the train's base, if there is one."""
    q = urllib.parse.urlencode({'branch': MAIN, 'per_page': 20})
    got = api(f"repos/{REPO}/actions/workflows/{wf}/runs?{q}", check=False) or {'workflow_runs': []}
    for run in got['workflow_runs']:
        if run['head_sha'] == sha:
            for j in api_list(f"repos/{REPO}/actions/runs/{run['id']}/jobs", 'jobs'):
                if j['name'] == job_name:
                    return j['conclusion'] or j['status']
    return None


def suspects(st, r, runs):
    failed = [(wf, run, j) for wf, (run, jobs) in runs.items() for j in jobs
              if j['conclusion'] in ('failure', 'timed_out')]
    files, on_main = set(), []
    for wf, run, j in failed:
        fs = failing_test_files(run, j)
        files |= fs
        m = main_result(wf, r['base'], j['name'])
        LOG.say(f"  failed: {wf} / {j['name']}: {', '.join(sorted(fs)) or 'no test ids in the log or report'}"
                f"; same job on {MAIN} {r['base'][:7]}: {m or 'no run'}")
        if m == 'failure':
            on_main.append(j['name'])
    touched = {e['number']: changed_files(git('merge-base', r['base'], e['head']), e['head']) for e in r['stack']}
    strong = [n for n, fs in touched.items() if fs & files]
    code = [n for n, fs in touched.items() if any(not INERT.match(f) or f.startswith('docs/api/') for f in fs)]
    if on_main:
        LOG.say(f"  {', '.join(on_main)} also failed on {MAIN} itself: likely not a train member's fault. "
                f"`retry` for a flake, or fix {MAIN} (an urgent fix) and `build --rebuild`.")
    if len(strong) == 1:
        LOG.say(f"  suspect: #{strong[0]} (changes {', '.join(sorted(touched[strong[0]] & files))}). "
                f"`kick {strong[0]}` to drop it and rerun.")
    elif len(code) == 1:
        LOG.say(f"  suspect: #{code[0]}, the only member that changes code (the rest is bench/ or prose). "
                f"`kick {code[0]}`.")
    else:
        LOG.say(f"  unclear ({'several' if strong else 'no'} member(s) touch the failing tests"
                f"{': #' + ', #'.join(map(str, strong)) if strong else ''}). "
                f"`bisect` to find the first red prefix, or `retry` if it looks like a flake.")


def show_runs(runs):
    for wf, (run, jobs) in runs.items():
        LOG.say(f"  {wf}: {run['status']}/{run['conclusion']} {run['html_url']}")
        for j in sorted(jobs, key=lambda j: j['name']):
            if wf != 'ci.yml' or j['conclusion'] not in ('success', 'skipped'):
                LOG.say(f"      {j['name']:<34} {j['conclusion'] or j['status']}")
        if wf == 'ci.yml':
            ok = sum(j['conclusion'] in ('success', 'skipped') for j in jobs)
            LOG.say(f"      ({ok}/{len(jobs)} CI jobs green)")


def member_drift(st, r):
    """Members whose PR moved, closed, or lost its READY since the round was built."""
    out = []
    for e in r['stack']:
        pr = api(f"repos/{REPO}/pulls/{e['number']}")
        if pr.get('merged_at'):
            continue
        if pr['state'] != 'open':
            out.append((e['number'], f"is {pr['state']}"))
        elif pr['head']['sha'] != e['head']:
            out.append((e['number'], f"head moved {e['head'][:7]} -> {pr['head']['sha'][:7]}"))
        elif pr['draft'] and not st.get('include_drafts'):
            out.append((e['number'], 'turned into a draft'))
    return out


def cmd_status(a):
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    st = load(name)
    while True:
        r = rnd(st)
        LOG.say(f"{name}, round {r['round']} ({r['why']}, {r['at']}): head {(r['head'] or '-')[:7]} on "
                f"{MAIN} {r['base'][:7]}; members " + ', '.join(f"#{e['number']}" for e in r['stack'])
                + (f"; kicked: {', '.join('#' + k for k in st['kicked'])}" if st['kicked'] else ''))
        if not r['head']:
            return 0
        fetch()
        if rev(f'origin/{MAIN}') != r['base'] and not st['merged']:
            LOG.say(f"  note: {MAIN} moved to {rev('origin/' + MAIN)[:7]} since the build: merge will refuse; "
                    f"`build --rebuild` to restack")
        for n, why in member_drift(st, r):
            LOG.say(f"  note: #{n} {why}: merge will refuse; `kick {n}` or `build --rebuild`")
        missing = [wf for wf in st['workflows'] if not (r['runs'].get(wf) or {}).get('id')]
        if missing:
            LOG.say(f"  not dispatched or not found yet: {', '.join(missing)} (`build` resumes it)")
        runs = collect(st, r)
        show_runs(runs)
        v = verdict(runs) if not missing else 'running'
        LOG.say(f"  => {v.upper()}")
        if v.startswith('red'):
            suspects(st, r, runs)
        if st['merged']:
            LOG.say(f"  merged so far: {', '.join('#%d' % m['number'] for m in st['merged'])}")
        if not a.wait or v in ('green', 'red'):
            return {'green': 0, 'red': 1}.get(v, 2)
        time.sleep(a.wait)


# --------------------------------------------------------------------------------------------
# kick, retry, bisect

def do_kick(st, n, head, reason, rebuild=True, round_no=None):
    st['kicked'][str(n)] = {'reason': reason, 'at': iso(now()), 'head': head,
                            'round': round_no or (rnd(st) or {}).get('round')}
    comment(n, f"{MARK} kicked from train `{st['name']}` (head {head[:7]}): {reason}.\n\n"
               f"This withdraws the READY for this head. Fix it (rebase, or wait for the PRs ahead of it "
               f"to merge), test it, and post a new READY; it rides the next window's train.")
    save(st)
    if rebuild:
        members = refreshed_members(st)
        r = new_round(st, members, f"kicked #{n}")
        launch(st, r, st['workflows'])


def cmd_kick(a):
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    st = load(name)
    if st['merged']:
        raise Fail("merging has started: kick is for before the merge")
    n = a.pr
    m = next((x for x in st['members'] if x['number'] == n), None)
    if not m:
        raise Fail(f"#{n} is not in {name}")
    if str(n) in st['kicked']:
        LOG.say(f"#{n} is already kicked ({st['kicked'][str(n)]['reason']}); resuming the rebuild")
        r = rnd(st)
        members = refreshed_members(st)
        if [e['number'] for e in r['stack']] != [x['number'] for x in members]:
            r = new_round(st, members, f"kicked #{n}")
        launch(st, r, st['workflows'])
        return 0
    head = next((e['head'] for e in rnd(st)['stack'] if e['number'] == n), m['head'])
    do_kick(st, n, head, a.reason or "the train went red with it (see the train's runs); the train reruns without it")
    return 0


def cmd_retry(a):
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    st = load(name)
    r = rnd(st)
    for wf, d in r['runs'].items():
        if not d.get('id'):
            continue
        run = api(f"repos/{REPO}/actions/runs/{d['id']}")
        if run['status'] == 'completed' and run['conclusion'] != 'success':
            write_api(f"rerun the failed jobs of {wf} run {d['id']}",
                      f"repos/{REPO}/actions/runs/{d['id']}/rerun-failed-jobs")
            LOG.say(f"  {wf}: rerunning failed jobs of {run['html_url']}")
    return 0


def cmd_bisect(a):
    """Binary search over the round's prefixes (main + the first k PRs: the train's own commits),
    one probe at a time, for the first k that is red. main (k=0) is taken as green and the full
    train (k=n) as red. Each call advances it; only the workflows that failed are rerun."""
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    st = load(name)
    r = rnd(st)
    n = len(r['stack'])
    b = st.get('bisect')
    if not b or b['round'] != r['round']:
        runs = collect(st, r)
        if verdict(runs) != 'red':
            raise Fail(f"round {r['round']} is not red ({verdict(runs)}): nothing to bisect")
        wfs = sorted({wf for wf, (run, jobs) in runs.items() if wf in st['workflows'] and
                      any(j['conclusion'] in ('failure', 'timed_out') for j in jobs)})
        if not wfs:
            raise Fail("only a workflow the train does not dispatch failed (docs.yml?): read its log")
        b = {'round': r['round'], 'lo': 0, 'hi': n, 'workflows': wfs, 'probes': {}}
        st['bisect'] = b
        save(st)
        LOG.say(f"bisect round {r['round']}: prefixes 0 (main, green) .. {n} (red); rerunning {', '.join(wfs)}")
    while True:
        if b['hi'] - b['lo'] <= 1:
            e = r['stack'][b['hi'] - 1]
            LOG.say(f"first red prefix: {b['hi']} -> culprit #{e['number']} ({e['title'][:60]}). "
                    f"`kick {e['number']}`.")
            return 0
        pending = [k for k, p in b['probes'].items() if p.get('result') is None]
        if not pending:
            k = (b['lo'] + b['hi']) // 2
            ref = f"{name}-r{r['round']}-p{k}"
            e = r['stack'][k - 1]
            b['probes'][str(k)] = {'ref': ref, 'sha': e['commit'], 'at': iso(now()), 'runs': {}, 'result': None}
            save(st)
            LOG.say(f"probe prefix {k} (through #{e['number']}) on {ref}")
            push_ref(ref, e['commit'], None)
            dispatch(st, b['probes'][str(k)], ref, e['commit'], b['workflows'])
            save(st)
            return 2
        k = pending[0]
        p = b['probes'][k]
        dispatch(st, p, p['ref'], p['sha'], b['workflows'])
        runs = collect(st, p, ref=p['ref'], sha=p['sha'])
        show_runs(runs)
        v = verdict(runs)
        if v not in ('green', 'red'):
            LOG.say(f"probe prefix {k} still running: call bisect again when it finishes")
            return 2
        p['result'] = v
        if v == 'green':
            b['lo'] = int(k)
        else:
            b['hi'] = int(k)
        LOG.say(f"prefix {k}: {v}; now lo={b['lo']} hi={b['hi']}")
        save(st)


# --------------------------------------------------------------------------------------------
# merge

def cmd_merge(a):
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    if not a.i_am_the_merger and not LOG.dry:
        raise Fail("merge squash-merges into main and is the merger's action only: "
                   "pass --i-am-the-merger (or --dry-run to see the checks)")
    st = load(name)
    r = rnd(st)
    if not r['head']:
        raise Fail("the train is empty")
    LOG.say(f"merge {name} round {r['round']}: " + ', '.join(f"#{e['number']}" for e in r['stack']))
    fetch()
    if any(not m['tree_ok'] for m in st['merged']):
        raise Fail("an ALERT stopped this train (main's tree differed from the train's): not resuming; "
                   "see the log and the state's `merged`")
    # resume: adopt merges that happened (a crash after the merge call) in train order
    for e in r['stack']:
        if any(m['number'] == e['number'] for m in st['merged']):
            continue
        pr = api(f"repos/{REPO}/pulls/{e['number']}")
        if not pr.get('merged_at'):
            break
        sha = pr['merge_commit_sha']
        parent = st['merged'][-1]['sha'] if st['merged'] else r['base']
        ok = git('rev-parse', f'{sha}^', check=False) == parent and tree(sha) == e['tree']
        st['merged'].append({'number': e['number'], 'sha': sha, 'at': pr['merged_at'], 'tree_ok': ok, 'adopted': True})
        save(st)
        LOG.say(f"  #{e['number']} was already merged as {sha[:7]} (tree {'==' if ok else '!='} train commit)")
        if not ok:
            raise Fail(f"ALERT: #{e['number']} merged as {sha[:7]} whose parent/tree is not the train's")
    done = {m['number']: m for m in st['merged']}
    if len(done) == len(r['stack']):
        LOG.say("every member is merged")
    problems = []

    # 1. the train is green on its current head, nobody moved the train branch
    if rev(f"origin/{name}") != r['head']:
        problems.append(f"{name} on GitHub is {(rev('origin/' + name) or 'gone')[:7]}, the round's head is {r['head'][:7]}")
    missing = [wf for wf in st['workflows'] if not (r['runs'].get(wf) or {}).get('id')]
    runs = collect(st, r)
    v = verdict(runs) if not missing else 'not run'
    if v != 'green':
        problems.append(f"the train is {v} ({', '.join(missing) or 'see status'})")
    for wf, (run, _) in runs.items():
        if run['head_sha'] != r['head']:
            problems.append(f"{wf} run {run['id']} ran {run['head_sha'][:7]}, not the train head")
    # 2. main is where the train left it (or where our own merges took it)
    expect = done[r['stack'][len(done) - 1]['number']]['sha'] if done else r['base']
    if rev(f'origin/{MAIN}') != expect:
        problems.append(f"{MAIN} is {rev('origin/' + MAIN)[:7]}, expected {expect[:7]} "
                        f"({'the last train merge' if done else 'the train base'}): rebuild")
    # 3. each PR still is what was tested, and can merge without a bypass
    required, _ = required_checks()
    prs = {}
    for e in r['stack']:
        if e['number'] in done:
            continue
        pr = prs[e['number']] = api(f"repos/{REPO}/pulls/{e['number']}")
        if pr['state'] != 'open':
            problems.append(f"#{e['number']} is {pr['state']}")
            continue
        if pr['head']['sha'] != e['head']:
            problems.append(f"#{e['number']} head moved {e['head'][:7]} -> {pr['head']['sha'][:7]} since the build")
        if pr['draft']:
            problems.append(f"#{e['number']} is a draft")
        if pr['base']['ref'] != MAIN:
            problems.append(f"#{e['number']} base is {pr['base']['ref']}")
        by = head_checks(e['head'])
        lack = [c for c, app in required.items() if not check_ok(by, c, app)]
        if lack and not a.admin_bypass:
            problems.append(f"#{e['number']} head lacks green {', '.join(lack)}: GitHub needs an admin bypass "
                            f"(--admin-bypass, the merger's decision)")
        ev = evaluate(pr, required, st.get('include_drafts'), True)
        if not ev['ready_at'] and pr['head']['sha'] == e['head']:
            problems.append(f"#{e['number']} READY withdrawn: {ev['why']}")
    if problems:
        LOG.say("REFUSED:")
        for p in problems:
            LOG.say(f"  - {p}")
        return 1
    LOG.say(f"checks passed: train green on {r['head'][:7]}, {MAIN} at {expect[:7]}, every head as built")

    for e in r['stack']:
        n = e['number']
        if n in done:
            continue
        cur = rev(f'origin/{MAIN}')
        pr = prs[n]
        title = f"{pr['title']} (#{n})"
        if LOG.dry:
            LOG.say(f"  [dry-run] would squash-merge #{n} at {e['head'][:7]} as \"{title[:80]}\", "
                    f"then check {MAIN}'s tree == {e['tree'][:10]} (train commit {e['commit'][:7]})")
            continue
        for attempt in range(8):
            # right after a merge GitHub recomputes the next PR's mergeability and may answer 405
            # "Base branch was modified": wait and retry. A moved head (409) is not retried.
            try:
                got = write_api(f"squash-merge #{n}", f"repos/{REPO}/pulls/{n}/merge", 'PUT',
                                {'merge_method': 'squash', 'sha': e['head'], 'commit_title': title})
                break
            except Fail as err:
                if 'Base branch was modified' not in str(err) or attempt == 7:
                    raise
                LOG.say(f"  #{n}: base branch was modified (GitHub still settling); retry in {5 * (attempt + 1)} s")
                time.sleep(5 * (attempt + 1))
                if api(f"repos/{REPO}/pulls/{n}").get('merged_at'):
                    raise Fail(f"#{n} merged meanwhile: rerun merge to adopt it")
        sha = got['sha']
        for _ in range(10):   # the ref update can lag the API's answer by a moment
            fetch()
            if rev(f'origin/{MAIN}') == sha:
                break
            time.sleep(3)
        ok = rev(f'origin/{MAIN}') == sha and git('rev-parse', f'{sha}^', check=False) == cur and tree(sha) == e['tree']
        st['merged'].append({'number': n, 'sha': sha, 'at': iso(now()), 'tree_ok': ok})
        save(st)
        if not ok:
            LOG.say(f"ALERT: after #{n} {MAIN} is {rev('origin/' + MAIN)[:7]} with tree {tree('origin/' + MAIN)[:10]}; "
                    f"the train commit {e['commit'][:7]} has tree {e['tree'][:10]} (parent expected {cur[:7]}). "
                    f"STOPPED: the rest is not merged. Compare: git diff {e['commit']} {sha}")
            comment(n, f"{MARK} ALERT: merged as {sha[:7]}, but {MAIN}'s tree differs from the tested train "
                       f"commit {e['commit'][:7]}. The train is stopped; the merger is looking into it.")
            return 3
        LOG.say(f"  merged #{n} as {sha[:7]}; tree == train commit {e['commit'][:7]}")
        comment(n, f"{MARK} squash-merged as {sha[:7]} from train `{name}` (position {e['position']}); "
                   f"{MAIN}'s tree equals the tested train commit {e['commit'][:7]}.")
    if LOG.dry:
        return 0
    fetch()
    if tree(f'origin/{MAIN}') != tree(r['head']):
        LOG.say(f"ALERT: {MAIN}'s tree != the train's tree after all merges")
        return 3
    LOG.say(f"done: {MAIN} = {rev('origin/' + MAIN)[:7]}, tree == {name} ({r['head'][:7]}). Next: cleanup")
    return 0


# --------------------------------------------------------------------------------------------
# cleanup

def cmd_cleanup(a):
    name = current_name(a)
    LOG.to(os.path.join(BASE_DIR, train_id(name), 'log'))
    st = load(name)
    r = rnd(st)
    complete = r and r['head'] and len(st['merged']) == len(r['stack'])
    if r and r['head'] and not complete and not a.abandon:
        raise Fail(f"{name} is not fully merged ({len(st['merged'])}/{len(r['stack'])}): --abandon to drop it")
    for x in st['rounds']:
        cancel_runs(x)
    for p in ((st.get('bisect') or {}).get('probes') or {}).values():
        cancel_runs(p)
    refs = [l.split('\t')[1][len('refs/heads/'):] for l in
            git('ls-remote', 'origin', f'refs/heads/{name}', f'refs/heads/{name}-r*').splitlines() if l]
    for ref in refs:
        assert ref.startswith('train/')
        write_api(f"delete {ref}", f"repos/{REPO}/git/refs/heads/{urllib.parse.quote(ref)}", 'DELETE', check=False)
    if not refs:
        LOG.say("no train refs left on GitHub")
    if a.abandon and r:
        merged = {m['number'] for m in st['merged']}
        for e in r['stack']:
            if e['number'] not in merged and str(e['number']) not in st['kicked']:
                comment(e['number'], f"{MARK} train `{name}` was abandoned; this PR's READY stands and it rides "
                                     f"the next window's train.")
    prune = PRUNE_YES_CMD if a.prune_yes and PRUNE_YES_CMD else PRUNE_CMD
    if prune:
        if LOG.dry:
            LOG.say(f"  [dry-run] would run: {prune}")
        else:
            LOG.say(f"$ {prune}")
            out = sh([os.path.expanduser(x) for x in shlex.split(prune)], check=False)
            LOG.say(out.stdout.strip()[-4000:])
    st['cleaned'] = iso(now())
    save(st)
    if not LOG.dry and os.path.exists(os.path.join(BASE_DIR, 'current')) and \
            open(os.path.join(BASE_DIR, 'current')).read().strip() == name:
        os.remove(os.path.join(BASE_DIR, 'current'))
    LOG.say(f"{name} {'would be ' if LOG.dry else ''}cleaned up")
    return 0


# --------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dry-run', action='store_true', help='no push, dispatch, comment, merge or delete')
    common.add_argument('--name', help='the train (train/<YYYYMMDD>-<HHMM>); default: the current one')
    sub = p.add_subparsers(dest='cmd', required=True)
    prs = lambda s: sorted({int(x.lstrip('#')) for x in s.split(',') if x})

    s = sub.add_parser('plan', parents=[common])
    s.add_argument('--prs', type=prs, help='only these PRs (comma-separated)')
    s.add_argument('--include-drafts', action='store_true', help='count drafts (tests of the train)')
    s.add_argument('--ignore-head-checks', action='store_true',
                   help="count heads without main's required checks (tests of the train)")
    s = sub.add_parser('build', parents=[common])
    s.add_argument('--prs', type=prs)
    s.add_argument('--include-drafts', action='store_true')
    s.add_argument('--ignore-head-checks', action='store_true')
    s.add_argument('--off-window', action='store_true', help='build outside the 01:00Z/13:00Z windows')
    s.add_argument('--rebuild', action='store_true', help='restack an existing train on the current main')
    s.add_argument('--workflows', default='ci,gpu', help='ci,gpu (default) -- fewer only for tests of the train')
    s = sub.add_parser('status', parents=[common])
    s.add_argument('--wait', type=int, nargs='?', const=300, default=0,
                   help='poll every N s (300) until green or red; exit 0 green, 1 red')
    s = sub.add_parser('kick', parents=[common])
    s.add_argument('pr', type=lambda x: int(x.lstrip('#')))
    s.add_argument('--reason')
    sub.add_parser('bisect', parents=[common])
    sub.add_parser('retry', parents=[common])
    s = sub.add_parser('merge', parents=[common])
    s.add_argument('--i-am-the-merger', action='store_true')
    s.add_argument('--admin-bypass', action='store_true',
                   help='allow PRs whose head lacks main\'s required checks (GitHub then needs an admin)')
    s = sub.add_parser('cleanup', parents=[common])
    s.add_argument('--abandon', action='store_true', help='drop a train that did not (fully) merge')
    s.add_argument('--prune-yes', action='store_true', help='run MERGE_TRAIN_PRUNE_YES instead of MERGE_TRAIN_PRUNE')
    a = p.parse_args()
    LOG.dry = a.dry_run
    os.makedirs(BASE_DIR, exist_ok=True)
    lock = open(os.path.join(BASE_DIR, '.lock'), 'w')
    if a.cmd not in ('plan', 'status'):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another merge-train command is running")
    try:
        LOG.write(f"=== {shlex.join(sys.argv)}")
        rc = globals()[f"cmd_{a.cmd}"](a)
    except Fail as e:
        LOG.say(f"error: {e}")
        rc = 3
    sys.exit(rc)


if __name__ == '__main__':
    main()
