"""The Jarvis voice app's ``voice-chat`` surface: natural-answer guidance rides the MODEL INPUT of that turn only.

The system prompt stays byte-stable, the stored user message stays the transcript, and a typed turn after a
voice turn carries no note (``client_surface`` is rewritten on every submit).
"""

import threading
import types

import pytest

from tui_gateway import server


def test_voice_chat_note_is_turn_scoped():
    from tui_gateway.voice_reply_style import voice_chat_turn_note
    from tui_gateway.session_notifications import _hud_surface_note
    note = _hud_surface_note({"client_surface": "voice-chat"})
    assert note == voice_chat_turn_note()
    assert "single answer" in note.lower()
    assert "sentence or word limit" in note.lower()
    assert "one to three" not in note.lower()
    assert _hud_surface_note({"client_surface": ""}) == ""
    assert _hud_surface_note({"client_surface": "something-else"}) == ""


def test_voice_chat_is_a_recognized_surface():
    from tui_gateway.methods_prompt import _CLIENT_SURFACES
    from tui_gateway.contracts.prompt_voice import ClientSurface
    assert "voice-chat" in _CLIENT_SURFACES
    assert ClientSurface.voice_chat.value == "voice-chat"


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def voice_session(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *a, **k: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)

    calls = []

    def run_conversation(message, **kwargs):
        calls.append({"message": message, **kwargs})
        return {"final_response": "ok"}

    agent = types.SimpleNamespace(
        session_id="session-key", run_conversation=run_conversation, clear_interrupt=lambda: None,
        valid_tool_names=set(), reasoning_config={"enabled": True}, _cached_system_prompt="SYSTEM PROMPT v1")
    session = {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        "transport": None}
    server._sessions["vsid"] = session
    yield session, calls
    server._sessions.pop("vsid", None)


def _submit(text, **params):
    return server.handle_request(
        {"id": "r", "method": "prompt.submit", "params": {"session_id": "vsid", "text": text, **params}})


def test_voice_then_typed_turn_only_the_voice_turn_is_noted(voice_session):
    from tui_gateway.voice_reply_style import voice_chat_turn_note
    session, calls = voice_session

    assert "result" in _submit("what is a mutex", surface="voice-chat")
    session["running"] = False
    assert "result" in _submit("and a semaphore")

    assert len(calls) == 2
    assert calls[0]["message"] == f"{voice_chat_turn_note()}\n\nwhat is a mutex"
    assert calls[1]["message"] == "and a semaphore"
    # The stored user message is the unmodified transcript on both turns.
    assert calls[0]["persist_user_message"] == "what is a mutex"
    assert calls[1]["persist_user_message"] == "and a semaphore"


def test_voice_chat_turn_leaves_the_system_prompt_alone(voice_session):
    session, calls = voice_session
    agent = session["agent"]
    before = agent._cached_system_prompt

    assert "result" in _submit("hello there", surface="voice-chat")

    assert agent._cached_system_prompt == before == "SYSTEM PROMPT v1"
    assert "system_message" not in calls[0] and "system_prompt" not in calls[0]


def test_jarvis_producer_is_explicit_and_does_not_leak_to_native_voice(voice_session):
    session, calls = voice_session
    session['agent'].reasoning_config = {'enabled': True}
    assert 'result' in _submit('hello', surface='voice-chat', overlay_producer='jarvis')
    assert session['agent']._overlay_producer == 'jarvis'
    session['running'] = False
    assert 'result' in _submit('hello again', surface='voice-chat')
    assert session['agent']._overlay_producer == ''
    session['running'] = False
    assert 'result' in _submit('typed', overlay_producer='jarvis')
    assert session['agent']._overlay_producer == ''
