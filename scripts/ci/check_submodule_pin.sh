#!/bin/bash
# check_submodule_pin.sh BASE_SHA HEAD_SHA -- a pull request may move a submodule's pin only on
# purpose and only forward: when HEAD pins a submodule at another commit than BASE does, the PR's
# title or body (env PR_TITLE / PR_BODY) must say "<Name> bump" (e.g. "HipKittens bump"), and the
# new pin must descend from the old one (git merge-base --is-ancestor in the submodule's repo). A
# pin moved by accident -- a stale worktree's checkout committed with the change -- fails here
# instead of undoing an earlier PR's library fix on main.
set -euo pipefail
BASE=$1 HEAD=$2
fail=0
while read -r path; do
  old=$(git ls-tree "$BASE" "$path" | awk '{print $3}')
  new=$(git ls-tree "$HEAD" "$path" | awk '{print $3}')
  [ -n "$old" ] && [ -n "$new" ] && [ "$old" != "$new" ] || continue
  name=$(basename "$path")
  echo "$path: $old -> $new"
  if ! printf '%s\n%s\n' "${PR_TITLE:-}" "${PR_BODY:-}" | grep -qi "$name bump"; then
    echo "::error::$path moves ($old -> $new) but the PR's title or body does not say \"$name bump\"" \
         "-- a stale pin committed by accident? Restore it: git checkout $BASE -- $path"
    fail=1
    continue
  fi
  url=$(git config -f .gitmodules --get "submodule.$path.url")
  repo=$(mktemp -d)
  git -C "$repo" init -q
  git -C "$repo" fetch -q --filter=blob:none "$url" "$old" "$new"
  if git -C "$repo" merge-base --is-ancestor "$old" "$new"; then
    echo "$path: $new descends from $old: ok"
  else
    echo "::error::$path: $new does not descend from main's pin $old -- it would drop commits" \
         "main has; rebase the submodule change onto $old"
    fail=1
  fi
  rm -rf "$repo"
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
exit $fail
