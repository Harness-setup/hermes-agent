import json
import sqlite3
from unittest.mock import patch

from agent.agent_init import (
    _uncensored_mode_persist_disabled,
    _reconcile_uncensored_session_tracking,
    _delete_session_with_retries,
)


def _make_state_db(db_path, sessions):
    """sessions: list of (id, model, archived) tuples."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, archived INTEGER)")
    conn.executemany("INSERT INTO sessions (id, model, archived) VALUES (?, ?, ?)", sessions)
    conn.commit()
    conn.close()


def test_returns_false_when_state_file_missing(tmp_path):
    assert _uncensored_mode_persist_disabled(
        "default", state_path=tmp_path / "mode-state.json",
    ) is False


def test_returns_true_when_profile_is_a_key_in_the_backup_and_kind_is_uncensored(tmp_path):
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "uncensored",
        "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
        "backup": {
            "default": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
            "researcher": {"model": "claude-sonnet-5", "provider": "anthropic"},
        },
    }))
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is True
    assert _uncensored_mode_persist_disabled("researcher", state_path=state_path) is True


def test_returns_false_for_a_profile_not_in_the_backup(tmp_path):
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "uncensored", "model": "x",
        "backup": {"default": {"model": "x", "provider": "y"}},
    }))
    assert _uncensored_mode_persist_disabled("reviewer", state_path=state_path) is False


def test_returns_false_when_kind_is_local_even_if_profile_is_in_backup(tmp_path):
    # Regression: /mode local must NOT suppress persistence -- only
    # uncensored carries the "no trace" guarantee. Before this fix, the
    # old file (uncensored-mode-state.json) could ONLY ever contain
    # uncensored's own backup, so this ambiguity didn't exist; now that
    # local and uncensored share one file, kind must be checked, not just
    # profile membership in backup.
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "local", "model": "qwen/qwen3.5-9b",
        "backup": {"default": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"}},
    }))
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is False


def test_fails_safe_to_true_on_malformed_state_file(tmp_path):
    # A malformed/partially-written state file must never crash session
    # init. Unlike a missing file (a definite "not uncensored" answer),
    # an EXISTING-but-unreadable file means /mode switched something and
    # we can't tell what it was -- privacy-first: fail toward suppressing
    # persistence, not toward silently persisting a session that should
    # have been private.
    state_path = tmp_path / "mode-state.json"
    state_path.write_text("not valid json{{{")
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is True


def test_returns_false_when_state_file_genuinely_missing_not_just_unreadable(tmp_path):
    # A missing file is unambiguous ("nothing has switched"), unlike a
    # present-but-corrupt one -- must stay False, not get swept up in the
    # corrupt-file fail-closed behavior above.
    assert _uncensored_mode_persist_disabled(
        "default", state_path=tmp_path / "does-not-exist.json",
    ) is False


# --- _reconcile_uncensored_session_tracking / _delete_session_with_retries ---
#
# Tony, 2026-08-24: "I want to see all the conversation we had [during the
# uncensored session]. but after I go to a different session or make a new
# one or change mode, it will erase everything including the one in
# desktop app session history." Uncensored mode now persists normally
# (agent_init.py no longer sets _persist_disabled for it) so Desktop's
# live chat view actually works; this is the "erase on abandonment" half.
#
# The retry/pending_cleanup logic below was added after an automated
# security review of the first version flagged it as fail-open: a failed
# delete_session() call used to silently advance uncensored_session_id to
# the new session anyway, permanently losing track of the undeleted
# "private" one with no record it had ever failed.

def _write_state(state_path, **overrides):
    payload = {"kind": "uncensored", "model": "x", "backup": {"default": {"model": "y", "provider": "z"}}}
    payload.update(overrides)
    state_path.write_text(json.dumps(payload))


def test_first_uncensored_session_just_records_itself_no_delete(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path)  # no uncensored_session_id yet
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s_new"


def test_new_session_id_deletes_the_previously_tracked_one(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_old")
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path)
    m_db.return_value.delete_session.assert_called_once_with("s_old")
    data = json.loads(state_path.read_text())
    assert data["uncensored_session_id"] == "s_new"
    assert data["uncensored_pending_cleanup"] == []


def test_same_session_id_continuing_does_not_delete_itself(tmp_path):
    # A compression rebuild re-runs session init with the SAME session_id --
    # must not treat that as abandonment and delete the live session.
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_current")
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_current", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s_current"


def test_not_uncensored_mode_does_nothing(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, kind="local", uncensored_session_id="s_old")
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    # State file must be left untouched -- this function has no business
    # editing local-mode's state.
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s_old"


def test_missing_state_file_is_a_silent_no_op(tmp_path):
    _reconcile_uncensored_session_tracking("s_new", state_path=tmp_path / "does-not-exist.json")
    # No exception is the assertion here.


def test_delete_session_with_retries_succeeds_after_transient_failures():
    # Transient DB-lock-style errors are the realistic failure mode --
    # must retry rather than giving up on the first exception.
    with patch("hermes_state.SessionDB") as m_db:
        m_db.return_value.delete_session.side_effect = [RuntimeError("db locked"), RuntimeError("db locked"), None]
        assert _delete_session_with_retries("s_x", attempts=3, delay=0) is True
    assert m_db.return_value.delete_session.call_count == 3


def test_delete_session_with_retries_returns_false_after_exhausting_attempts():
    with patch("hermes_state.SessionDB") as m_db:
        m_db.return_value.delete_session.side_effect = RuntimeError("db locked")
        assert _delete_session_with_retries("s_x", attempts=3, delay=0) is False
    assert m_db.return_value.delete_session.call_count == 3


def test_permanently_failed_delete_is_queued_for_retry_not_silently_dropped(tmp_path):
    # The fail-open bug this whole retry/pending_cleanup mechanism fixes:
    # a session that could NOT be deleted must stay tracked (not silently
    # forgotten just because a different, newer session is now live).
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_old")
    with patch("hermes_state.SessionDB") as m_db:
        m_db.return_value.delete_session.side_effect = RuntimeError("db locked")
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path)
    data = json.loads(state_path.read_text())
    assert data["uncensored_session_id"] == "s_new"
    assert data["uncensored_pending_cleanup"] == ["s_old"]


def test_a_previously_queued_failure_is_retried_on_the_next_reconcile_call(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_new", uncensored_pending_cleanup=["s_old"])
    with patch("hermes_state.SessionDB") as m_db:
        # s_old finally succeeds this time; s_new is the same session
        # continuing (a later turn), not itself being abandoned.
        m_db.return_value.delete_session.return_value = None
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path)
    m_db.return_value.delete_session.assert_called_once_with("s_old")
    data = json.loads(state_path.read_text())
    assert data["uncensored_pending_cleanup"] == []
    assert data["uncensored_session_id"] == "s_new"


# --- defensive sweep for sessions orphaned outside `tracked`/`pending_cleanup` ---
#
# Live incident, 2026-09-15: a session persisted under the uncensored model was
# found neither as uncensored_session_id nor in uncensored_pending_cleanup --
# outside this function's own bookkeeping entirely, so it would never get
# deleted no matter how many later sessions started. The sweep below catches
# any stray uncensored-model session by construction, every reconcile call,
# regardless of how it became orphaned.

def test_sweep_catches_a_session_orphaned_outside_tracked_and_pending(tmp_path):
    state_path = tmp_path / "mode-state.json"
    db_path = tmp_path / "state.db"
    _write_state(state_path, uncensored_session_id="s_current")
    _make_state_db(db_path, [
        ("s_orphan", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
        ("s_current", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
    ])
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_current", state_path=state_path, db_path=db_path)
    m_db.return_value.delete_session.assert_called_once_with("s_orphan")
    data = json.loads(state_path.read_text())
    assert data["uncensored_pending_cleanup"] == []
    assert data["uncensored_session_id"] == "s_current"


def test_sweep_ignores_non_uncensored_sessions(tmp_path):
    state_path = tmp_path / "mode-state.json"
    db_path = tmp_path / "state.db"
    _write_state(state_path, uncensored_session_id="s_current")
    _make_state_db(db_path, [
        ("s_cloud", "claude-sonnet-5", 0),
        ("s_local", "qwen/qwen3.5-9b", 0),
        ("s_current", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
    ])
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_current", state_path=state_path, db_path=db_path)
    m_db.return_value.delete_session.assert_not_called()


def test_sweep_ignores_already_archived_sessions(tmp_path):
    """An archived session is already effectively gone from view -- don't
    churn a delete attempt on it every single reconcile call."""
    state_path = tmp_path / "mode-state.json"
    db_path = tmp_path / "state.db"
    _write_state(state_path, uncensored_session_id="s_current")
    _make_state_db(db_path, [
        ("s_archived", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 1),
        ("s_current", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
    ])
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_current", state_path=state_path, db_path=db_path)
    m_db.return_value.delete_session.assert_not_called()


def test_sweep_does_not_duplicate_the_already_tracked_session(tmp_path):
    """The sweep must not re-add the session already handled by the normal
    tracked-pointer path as a duplicate pending entry."""
    state_path = tmp_path / "mode-state.json"
    db_path = tmp_path / "state.db"
    _write_state(state_path, uncensored_session_id="s_old")
    _make_state_db(db_path, [
        ("s_old", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
        ("s_new", "qwen3.5-9b-uncensored-hauhaucs-aggressive", 0),
    ])
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking("s_new", state_path=state_path, db_path=db_path)
    m_db.return_value.delete_session.assert_called_once_with("s_old")


def test_sweep_is_silent_no_op_when_state_db_missing(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_current")
    with patch("hermes_state.SessionDB") as m_db:
        _reconcile_uncensored_session_tracking(
            "s_current", state_path=state_path, db_path=tmp_path / "does-not-exist.db")
    m_db.return_value.delete_session.assert_not_called()
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s_current"
