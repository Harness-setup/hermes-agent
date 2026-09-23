"""Tony, 2026-09-02: "uncensored mode is not erasing the session from
session history... I closed and opened hermes desktop and it was still
there." Investigation found the existing 3 erasure triggers
(agent_init.py's _reconcile_uncensored_session_tracking on a new/different
session, toggle.py's _restore_backup on a mode switch) never fire from
simply closing the client and never reopening it -- this is the 4th
trigger, gated behind its own independent grace-window timer (NOT the
existing 20s WS-orphan reap grace, which is tuned for network blips, not
"has the user actually moved on").

These tests exercise _check_and_erase_abandoned_uncensored_session directly
(the grace-window callback), not the real Timer/threading -- production
code schedules it via _maybe_schedule_uncensored_erasure, but that's a thin
wrapper around a real 300s Timer, not itself meaningfully unit-testable.
"""

import json
from unittest.mock import patch

from tui_gateway import server


def _write_state(state_path, **overrides):
    payload = {"kind": "uncensored", "model": "x", "backup": {"default": {"model": "y", "provider": "z"}}}
    payload.update(overrides)
    state_path.write_text(json.dumps(payload))


def test_not_uncensored_mode_does_nothing(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, kind="local", uncensored_session_id="s1")
    with patch("hermes_state.SessionDB") as m_db:
        server._check_and_erase_abandoned_uncensored_session("s1", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s1"


def test_session_not_the_currently_tracked_one_does_nothing(tmp_path):
    # Already superseded by a later new-session/different-session/mode-
    # switch trigger -- one of the other 3 already handled it.
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s_newer")
    with patch("hermes_state.SessionDB") as m_db:
        server._check_and_erase_abandoned_uncensored_session("s_stale", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    assert json.loads(state_path.read_text())["uncensored_session_id"] == "s_newer"


def test_still_tracked_and_not_reconnected_erases_it(tmp_path):
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s1")
    with patch.dict(server._sessions, {}, clear=True), \
         patch("hermes_state.SessionDB") as m_db:
        server._check_and_erase_abandoned_uncensored_session("s1", state_path=state_path)
    m_db.return_value.delete_session.assert_called_once_with("s1")
    data = json.loads(state_path.read_text())
    assert data["uncensored_session_id"] is None


def test_still_tracked_but_reconnected_within_grace_does_not_erase(tmp_path):
    # The critical safety net: Tony reopened Desktop and it auto-resumed
    # this SAME session before the grace window fired -- genuinely still
    # in use, must not be erased out from under him.
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s1")
    live_session = {"transport": object(), "_finalized": False}  # not the detached sentinel
    with patch.dict(server._sessions, {"s1": live_session}, clear=True), \
         patch("hermes_state.SessionDB") as m_db:
        server._check_and_erase_abandoned_uncensored_session("s1", state_path=state_path)
    m_db.return_value.delete_session.assert_not_called()
    data = json.loads(state_path.read_text())
    assert data["uncensored_session_id"] == "s1"


def test_still_tracked_and_present_but_still_detached_still_erases(tmp_path):
    # Session record still exists in _sessions (e.g. re-pointed at the
    # detached sentinel by the WS-disconnect path) but never got a live
    # transport back -- still genuinely abandoned, must still erase.
    state_path = tmp_path / "mode-state.json"
    _write_state(state_path, uncensored_session_id="s1")
    detached_session = {"transport": server._detached_ws_transport, "_finalized": False}
    with patch.dict(server._sessions, {"s1": detached_session}, clear=True), \
         patch("hermes_state.SessionDB") as m_db:
        server._check_and_erase_abandoned_uncensored_session("s1", state_path=state_path)
    m_db.return_value.delete_session.assert_called_once_with("s1")
    data = json.loads(state_path.read_text())
    assert data["uncensored_session_id"] is None


def test_missing_state_file_is_a_silent_no_op(tmp_path):
    server._check_and_erase_abandoned_uncensored_session(
        "s1", state_path=tmp_path / "does-not-exist.json",
    )
    # No exception is the assertion here.


def test_close_on_disconnect_teardown_also_schedules_uncensored_erasure(monkeypatch):
    """Live investigation, 2026-09-03: a close_on_disconnect=True session
    (e.g. Desktop's sidecar chat, ChatSidebar.tsx) is torn down IMMEDIATELY
    by _close_sessions_for_transport, via a separate branch that never goes
    through _schedule_ws_orphan_reap -- the only place the 4th erasure
    trigger was wired in. That session could carry the tracked uncensored
    session_id and simply vanish from erasure coverage on disconnect,
    silently defeating the whole point of this trigger for that path."""
    seen = []
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: True,
    )
    monkeypatch.setattr(
        server,
        "_maybe_schedule_uncensored_erasure",
        lambda sid: seen.append(sid),
    )
    transport = object()
    server._sessions.clear()
    server._sessions["s1"] = {"transport": transport, "close_on_disconnect": True}
    try:
        server._close_sessions_for_transport(transport, end_reason="ws_disconnect")
        assert seen == ["s1"]
    finally:
        server._sessions.clear()
