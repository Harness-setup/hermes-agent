"""Tests that /new (and its /reset alias) clears session-scoped overrides."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None  # disables _evict_cached_agent lock path
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
async def test_new_command_only_clears_own_session():
    """/new must only clear the override for the session that triggered it."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    other_key = "other_session_key"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_model_overrides[other_key] = {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_key": "***",
        "base_url": "",
        "api_mode": "anthropic",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides[other_key] = {"enabled": True, "effort": "low"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"
    runner._pending_model_notes[other_key] = "[Note: switched to claude-sonnet-4-6.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert other_key in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    assert other_key in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes
    assert other_key in runner._pending_model_notes


@pytest.mark.asyncio
async def test_new_quiet_does_the_full_reset_but_suppresses_the_notice():
    """``/new --quiet`` (used by the mode plugin's natural-language switch, which already
    sent its own plain confirmation sentence) performs the identical reset as bare /new but
    returns an empty EphemeralReply instead of the verbose header+session-info+tip notice --
    the send path treats empty/falsy EphemeralReply text as "send nothing" (gateway/
    platforms/base.py's _unwrap_ephemeral + the "if not text: return" send guards)."""
    from gateway.platforms.base import EphemeralReply

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o", "provider": "openai", "api_key": "sk-test",
        "base_url": "", "api_mode": "openai",
    }

    result = await runner._handle_reset_command(_make_event("/new --quiet"))

    assert isinstance(result, EphemeralReply)
    assert result.text == ""
    # The reset itself still happened -- only the notice is suppressed.
    assert session_key not in runner._session_model_overrides


@pytest.mark.asyncio
async def test_new_without_quiet_still_returns_the_verbose_notice():
    """Bare /new (typed directly, or any other rewrite target) is unaffected by --quiet."""
    runner = _make_runner()
    result = await runner._handle_reset_command(_make_event("/new"))
    assert result.text != ""


@pytest.mark.asyncio
async def test_explicit_quiet_kwarg_works_against_the_original_event_text():
    """Tony, 2026-09-28: "goto local mode and tell me what this is" -- the mode plugin calls
    this directly against the user's ORIGINAL event (text is the real message, not "/new
    --quiet") when combining a mode switch with a real follow-up request. The explicit
    ``quiet=True`` kwarg must produce the same suppressed notice as the "--quiet" command-arg
    form, without needing to fake the command text first."""
    runner = _make_runner()
    result = await runner._handle_reset_command(
        _make_event("go local mode and tell me what this is"), quiet=True,
    )
    assert result.text == ""


@pytest.mark.asyncio
async def test_explicit_quiet_false_overrides_a_literal_quiet_looking_title():
    """quiet=False must win even if the command text itself happens to say "--quiet" --
    an explicit kwarg is authoritative, never guessed from the text when passed."""
    runner = _make_runner()
    result = await runner._handle_reset_command(_make_event("/new --quiet"), quiet=False)
    assert result.text != ""
