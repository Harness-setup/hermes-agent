"""Regression tests for _reconcile_diverged_checkout's unrelated-histories detection.

Live incident (2026-09-23): the real upstream this repo tracks force-rewrote its `main` branch
history entirely (a squash / repo re-init), leaving the local `local-fixes` checkout with ZERO
common ancestor with `origin/main` (confirmed live via `git merge-base HEAD origin/main`, which
returned no output and a non-zero exit code). Every `hermes update` attempt that night failed
identically -- a plain `git merge --no-edit origin/<branch>` always hard-fails with "refusing to
merge unrelated histories" in this situation, not a normal resolvable content conflict, and no
number of retries changes that. The OLD code here couldn't tell the two apart: both produced the
same generic "Merge conflict between local commits and upstream ... Resolve manually: git merge
origin/<branch>" message -- advice that fails the exact same way for the user too, since a plain
`git merge` can never succeed here. This adds an up-front `git merge-base` check so the two cases
get different, honest handling instead of one misleading message for both.

These tests run against REAL git repositories (init, commit, orphan-branch) -- not mocked
subprocess.run -- matching test_update_parked_branch_guard.py's own convention for this exact
function's neighborhood.
"""

import subprocess

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture()
def unrelated_histories_repo(tmp_path):
    """A real local checkout on a custom branch ('local-fixes'), with its 'origin/main' remote-
    tracking ref pointing at a history that shares NO common ancestor -- reproducing a force-
    rewritten upstream, not just new commits to reconcile."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    # A real local commit on a custom branch, atop the original shared history.
    _git(clone, "checkout", "-qb", "local-fixes")
    (clone / "local.txt").write_text("local work\n")
    _git(clone, "add", "local.txt")
    _git(clone, "commit", "-qm", "local fix")

    # Rewrite origin's history entirely: a fresh orphan root, unrelated to c1. This is what a
    # squash / repo re-init on the real upstream looks like from the clone's side.
    _git(origin, "checkout", "-q", "--orphan", "rewritten")
    _git(origin, "rm", "-rq", "--cached", ".")
    (origin / "a.txt").write_text("rewritten from scratch\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "rewritten c1")
    _git(origin, "branch", "-qM", "rewritten", "main")

    _git(clone, "fetch", "-q", "origin", "main")
    return clone


@pytest.fixture()
def diverged_but_related_repo(tmp_path):
    """Same shape, but origin/main is genuinely just AHEAD (normal new commits) -- a real shared
    ancestor exists. Used to confirm the new up-front check doesn't fire on the ordinary case."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "checkout", "-qb", "local-fixes")
    (clone / "local.txt").write_text("local work\n")
    _git(clone, "add", "local.txt")
    _git(clone, "commit", "-qm", "local fix")

    # origin advances normally, sharing c1 as a real common ancestor -- and touches a DIFFERENT
    # file than the local commit did, so the eventual merge has no real content conflict either.
    (origin / "b.txt").write_text("two\n")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "c2")

    _git(clone, "fetch", "-q", "origin", "main")
    return clone


@pytest.fixture(autouse=True)
def _no_config(monkeypatch):
    """Isolate from the machine's real config.yaml, matching test_update_parked_branch_guard.py."""
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config", lambda: {})


def test_unrelated_histories_exits_with_an_accurate_message_not_a_doomed_merge(
    unrelated_histories_repo, monkeypatch, capsys
):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", unrelated_histories_repo)

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._reconcile_diverged_checkout(GIT, "main", None)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    # The new, honest message names the real cause...
    assert "shares no common history" in out
    assert "rewritten" in out or "squash" in out or "re-init" in out
    # ...and must NOT repeat the old advice that fails the exact same way for the user.
    assert "git merge origin/main" not in out
    assert "Merge conflict between local commits and upstream" not in out


def test_unrelated_histories_never_attempts_or_leaves_a_merge_in_progress(
    unrelated_histories_repo, monkeypatch
):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", unrelated_histories_repo)
    head_before = _git(unrelated_histories_repo, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(SystemExit):
        update_cmd._reconcile_diverged_checkout(GIT, "main", None)

    # No merge was ever started (and therefore none left half-finished): the working tree is
    # exactly what it was before the call, HEAD never moved, and there's no dangling merge state
    # for the user to have to clean up.
    status = _git(unrelated_histories_repo, "status", "--porcelain")
    assert status.stdout == ""
    assert _git(unrelated_histories_repo, "rev-parse", "HEAD").stdout.strip() == head_before
    assert not (unrelated_histories_repo / ".git" / "MERGE_HEAD").exists()
    assert (unrelated_histories_repo / "local.txt").exists()  # local commit still intact
    # Origin's REWRITTEN content specifically never landed -- distinct from a.txt, which the
    # clone already carried from before the rewrite and proves nothing about a merge attempt.
    assert "rewritten from scratch" not in (unrelated_histories_repo / "a.txt").read_text()


def test_genuinely_related_divergence_still_merges_normally(diverged_but_related_repo, monkeypatch):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", diverged_but_related_repo)

    # Must NOT raise/exit -- a real shared ancestor means the up-front check passes and the
    # ordinary merge path runs (and succeeds here, since the two sides touch different files).
    update_cmd._reconcile_diverged_checkout(GIT, "main", None)

    assert (diverged_but_related_repo / "local.txt").exists()  # local commit preserved
    assert (diverged_but_related_repo / "b.txt").exists()  # origin's new commit merged in
    status = _git(diverged_but_related_repo, "status", "--porcelain")
    assert status.stdout == ""  # clean merge, nothing left staged/conflicted
