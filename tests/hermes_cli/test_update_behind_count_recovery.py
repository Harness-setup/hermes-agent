"""Behind-count recovery via the GitHub compare API (source_check.py).

The class of bug: any code path that knows two tip SHAs but has no local
history to count across (shallow installer clones, ls-remote-only probes)
used to fabricate a count of ``1`` — the UI then rendered "+1" / "1 commit
behind" forever while the real distance grew (#84591: 61 commits behind,
indicator said 1). The fix has two halves:

1. Honesty: never fabricate a number. Uncountable = UPDATE_AVAILABLE_NO_COUNT
   sentinel (CLI) / null (desktop), rendered as a generic "update available".
2. Accuracy: recover the exact count via GitHub's compare API, which knows
   the full graph regardless of local clone depth.
"""

import json
from unittest.mock import patch

import pytest

import hermes_cli.source_check as source_check

SHA_A = "a" * 40
SHA_B = "b" * 40


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, limit=None):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(payload):
    return patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps(payload).encode()),
    )


# ---------------------------------------------------------------------------
# _github_compare_behind
# ---------------------------------------------------------------------------


def test_compare_behind_returns_ahead_by():
    with _patch_urlopen({"ahead_by": 61, "status": "ahead"}):
        assert source_check._github_compare_behind(SHA_A, SHA_B) == 61


def test_compare_behind_zero_means_local_ahead():
    with _patch_urlopen({"ahead_by": 0, "status": "behind"}):
        assert source_check._github_compare_behind(SHA_A, SHA_B) == 0


def test_compare_behind_rejects_short_shas_without_network():
    with patch("urllib.request.urlopen") as mock_open:
        assert source_check._github_compare_behind("abc123", SHA_B) is None
        assert source_check._github_compare_behind(SHA_A, "") is None
        assert source_check._github_compare_behind(None, SHA_B) is None
    mock_open.assert_not_called()


def test_compare_behind_network_failure_returns_none():
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert source_check._github_compare_behind(SHA_A, SHA_B) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "diverged"},  # no ahead_by
        {"ahead_by": -3},  # negative
        {"ahead_by": "12"},  # wrong type
        {"ahead_by": True},  # bool masquerading as int
        [],  # wrong shape
    ],
)
def test_compare_behind_rejects_malformed_payloads(payload):
    with _patch_urlopen(payload):
        assert source_check._github_compare_behind(SHA_A, SHA_B) is None


def test_local_only_head_counts_locally_when_compare_api_cannot_see_it():
    """A parked-branch HEAD (local-fixes) 404s on the compare API; the target is in local history."""
    co = type("Co", (), {"head": SHA_A, "embedded": None, "repository": "owner/repo",
                         "root": None, "git": "git"})()
    with patch.object(source_check, "_github_compare", return_value=None), \
            patch.object(source_check, "_git_ok", side_effect=[False, True]), \
            patch.object(source_check, "_git_count", return_value=7) as count:
        assert source_check._behind_count(co, SHA_B) == (7, [])
    count.assert_called_once_with(["rev-list", "--count", f"{SHA_A}..{SHA_B}"], cwd=None)


def test_local_only_head_compares_from_the_shared_merge_base_for_count_and_notes():
    """HEAD is unknown to GitHub (404), but the merge-base with origin/main is not: compare from it."""
    BASE = "c" * 40
    co = type("Co", (), {"head": SHA_A, "embedded": None, "repository": "owner/repo",
                         "root": None, "git": "git"})()
    payload = {"ahead_by": 311, "commits": [
        {"sha": SHA_B, "commit": {"message": "fix(gateway): stop dropping replies\n\nbody",
                                  "author": {"name": "dev"}, "committer": {"date": "2026-10-02T00:00:00Z"}}}]}
    seen = []

    def fake_compare(current, target, repository="owner/repo"):
        seen.append(current)
        return None if current == SHA_A else payload

    with patch.object(source_check, "_git_ok", return_value=False), \
            patch.object(source_check, "_git_stdout", return_value=BASE), \
            patch.object(source_check, "_github_compare", side_effect=fake_compare):
        behind, commits = source_check._behind_count(co, SHA_B)

    assert seen == [SHA_A, BASE]
    assert behind == 311
    assert commits and commits[0]["summary"] == "fix(gateway): stop dropping replies"
