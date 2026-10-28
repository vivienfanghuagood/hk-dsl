"""A failing canary: tests that the merge train finds and kicks the PR that turns it red (throwaway)."""


def test_merge_train_canary():
    assert False, "merge-train canary: this PR must be kicked"
