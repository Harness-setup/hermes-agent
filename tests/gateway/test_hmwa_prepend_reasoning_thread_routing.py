"""Tests for GatewayRunner._hmwa_prepend_reasoning's Discord thread-routing branch.

Tony, 2026-09-28 (uncensored mode session): "the reasoning is showing in the main chat instead
of thread." Root cause: this end-of-turn reasoning block used to route into the same Discord
tool-progress thread that mid-turn reasoning already uses (run_turn_runner.py's
progress_callback/_send_to_tool_thread) -- prepending it onto the final response instead means
it always lands in the main channel, regardless of discord.thread_tool_calls being on.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run_module
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _make_runner(show_reasoning=True, reasoning_style="subtext"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._show_reasoning = show_reasoning

    def _fake_resolve_bool(*a, **kw):
        return show_reasoning
    gateway_run_module._resolve_gateway_display_bool = _fake_resolve_bool

    import gateway.display_config as display_config_module
    display_config_module.resolve_display_setting = lambda *a, **kw: reasoning_style

    return runner


def _make_source(platform=Platform.DISCORD) -> SessionSource:
    return SessionSource(platform=platform, user_id="u1", chat_id="c1", user_name="tester")


def _make_event(message_id="m1") -> MessageEvent:
    return MessageEvent(text="hi", source=_make_source(), message_id=message_id)


@pytest.mark.asyncio
async def test_discord_with_thread_tool_calls_on_routes_to_thread_not_main_response(monkeypatch):
    runner = _make_runner()
    adapter = SimpleNamespace(
        send_tool_progress_line=AsyncMock(),
        _discord_thread_tool_calls_enabled=lambda: True,
    )
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter, raising=False)
    event = _make_event()

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "The user is asking about X..."}, "final answer",
        event.source, False, event=event,
    )

    assert result == "final answer"  # unchanged -- reasoning did NOT get prepended
    adapter.send_tool_progress_line.assert_awaited_once_with(
        "c1", "m1", "-# 💭 Reasoning\n-# The user is asking about X...",
    )


@pytest.mark.asyncio
async def test_discord_with_thread_tool_calls_off_still_prepends_to_main_response(monkeypatch):
    runner = _make_runner()
    adapter = SimpleNamespace(
        send_tool_progress_line=AsyncMock(),
        _discord_thread_tool_calls_enabled=lambda: False,
    )
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter, raising=False)
    event = _make_event()

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "The user is asking about X..."}, "final answer",
        event.source, False, event=event,
    )

    adapter.send_tool_progress_line.assert_not_awaited()
    assert result == "-# 💭 Reasoning\n-# The user is asking about X...\n\nfinal answer"


@pytest.mark.asyncio
async def test_non_discord_platform_unaffected_still_prepends(monkeypatch):
    runner = _make_runner(reasoning_style="code")
    event = MessageEvent(text="hi", source=_make_source(platform=Platform.SLACK), message_id="m1")

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "thinking text"}, "final answer",
        event.source, False, event=event,
    )

    assert result == "💭 **Reasoning:**\n```\nthinking text\n```\n\nfinal answer"


@pytest.mark.asyncio
async def test_no_event_passed_falls_back_to_prepend_even_on_discord(monkeypatch):
    """A caller that can't supply an event (none currently exist, but defensively) must not
    crash -- falls back to the safe prepend-to-main-response behavior."""
    runner = _make_runner()

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "The user is asking about X..."}, "final answer",
        _make_source(), False, event=None,
    )

    assert result == "-# 💭 Reasoning\n-# The user is asking about X...\n\nfinal answer"


@pytest.mark.asyncio
async def test_adapter_resolution_failure_falls_back_to_prepend(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        runner, "_delivery_adapter_for",
        MagicMock(side_effect=RuntimeError("no adapter")), raising=False,
    )
    event = _make_event()

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "The user is asking about X..."}, "final answer",
        event.source, False, event=event,
    )

    assert result == "-# 💭 Reasoning\n-# The user is asking about X...\n\nfinal answer"


@pytest.mark.asyncio
async def test_show_reasoning_off_returns_response_unchanged(monkeypatch):
    runner = _make_runner(show_reasoning=False)
    event = _make_event()

    result = await runner._hmwa_prepend_reasoning(
        {"last_reasoning": "The user is asking about X..."}, "final answer",
        event.source, False, event=event,
    )

    assert result == "final answer"
