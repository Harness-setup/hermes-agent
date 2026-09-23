"""Tests for the discord.thread_tool_calls split: tool-call progress lines route to a
lazily-created thread while reasoning and the final answer stay in the main channel.

Covers two layers:
  - DiscordAdapter: thread creation/caching/send (send_tool_progress_line and friends).
  - TurnRunner._send_to_tool_thread / _progress_emit: the routing decision that claims a
    tool-progress line for the thread instead of the shared progress_queue, on Discord with
    the flag on, and falls through unchanged otherwise (every other platform, or Discord with
    the flag off).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner


# ── DiscordAdapter layer ─────────────────────────────────────────────────────────────────

def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.platforms.helpers import ThreadParticipationTracker

    config = PlatformConfig(enabled=True, token="test-token")
    with patch.object(ThreadParticipationTracker, "_load", return_value=set()):
        adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(get_channel=lambda _id: None, fetch_channel=AsyncMock())
    return adapter


def _fake_channel_with_message(*, create_thread_result=None, create_thread_side_effect=None):
    fake_thread = create_thread_result or SimpleNamespace(send=AsyncMock())
    seed_msg = SimpleNamespace(
        content="check current tenebris mode",
        create_thread=AsyncMock(return_value=fake_thread, side_effect=create_thread_side_effect),
    )
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=seed_msg))
    return channel, seed_msg, fake_thread


class TestThreadToolCallsFlag:
    def test_defaults_off(self):
        adapter = _make_adapter()
        assert adapter._discord_thread_tool_calls_enabled() is False

    def test_extra_config_turns_it_on(self):
        adapter = _make_adapter()
        adapter.config.extra["thread_tool_calls"] = True
        assert adapter._discord_thread_tool_calls_enabled() is True

    def test_env_var_turns_it_on(self, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("DISCORD_THREAD_TOOL_CALLS", "true")
        assert adapter._discord_thread_tool_calls_enabled() is True


@pytest.mark.asyncio
class TestGetOrCreateToolProgressThread:
    async def test_no_event_message_id_returns_none(self):
        adapter = _make_adapter()
        assert await adapter._get_or_create_tool_progress_thread("123", None) is None

    async def test_creates_thread_from_fetched_seed_message(self):
        adapter = _make_adapter()
        channel, seed_msg, fake_thread = _fake_channel_with_message()
        adapter._client.get_channel = lambda _id: channel

        result = await adapter._get_or_create_tool_progress_thread("123", "456")

        channel.fetch_message.assert_awaited_once_with(456)
        seed_msg.create_thread.assert_awaited_once()
        _, kwargs = seed_msg.create_thread.call_args
        assert kwargs["auto_archive_duration"] == 60
        assert "check current tenebris mode" in kwargs["name"].lower()
        assert result is fake_thread

    async def test_second_call_reuses_cached_thread_no_new_creation(self):
        adapter = _make_adapter()
        channel, seed_msg, fake_thread = _fake_channel_with_message()
        adapter._client.get_channel = lambda _id: channel

        first = await adapter._get_or_create_tool_progress_thread("123", "456")
        second = await adapter._get_or_create_tool_progress_thread("123", "456")

        assert first is second is fake_thread
        seed_msg.create_thread.assert_awaited_once()  # not called again
        channel.fetch_message.assert_awaited_once()  # not re-fetched either

    async def test_channel_not_found_returns_none(self):
        adapter = _make_adapter()
        adapter._client.get_channel = lambda _id: None
        adapter._client.fetch_channel = AsyncMock(return_value=None)
        assert await adapter._get_or_create_tool_progress_thread("123", "456") is None

    async def test_create_thread_failure_returns_none_not_raise(self):
        adapter = _make_adapter()
        channel, seed_msg, _ = _fake_channel_with_message()
        seed_msg.create_thread = AsyncMock(side_effect=RuntimeError("boom"))
        adapter._client.get_channel = lambda _id: channel

        result = await adapter._get_or_create_tool_progress_thread("123", "456")
        assert result is None
        assert seed_msg.create_thread.await_count == 2  # one retry, per _auto_create_thread's pattern


@pytest.mark.asyncio
class TestSendToolProgressLine:
    async def test_sends_formatted_text_into_the_thread(self):
        adapter = _make_adapter()
        channel, seed_msg, fake_thread = _fake_channel_with_message()
        adapter._client.get_channel = lambda _id: channel

        await adapter.send_tool_progress_line("123", "456", "🖥️ terminal: ls")

        fake_thread.send.assert_awaited_once()
        assert "ls" in fake_thread.send.call_args.kwargs["content"]

    async def test_thread_unavailable_is_silent_no_raise(self):
        adapter = _make_adapter()
        adapter._client.get_channel = lambda _id: None
        adapter._client.fetch_channel = AsyncMock(return_value=None)
        # Must not raise -- this is fire-and-forget from a sync callback context.
        await adapter.send_tool_progress_line("123", "456", "🖥️ terminal: ls")


# ── TurnRunner routing layer ─────────────────────────────────────────────────────────────

def _make_runner_and_ctx(*, platform=Platform.DISCORD, adapter=None):
    runner = SimpleNamespace(_adapter_for_source=lambda source: adapter)
    ctx = SimpleNamespace(
        source=SimpleNamespace(platform=platform, chat_id="123"),
        event_message_id="456",
        last_progress_msg=[None],
        repeat_count=[0],
        progress_queue=MagicMock(),
        stream_consumer_holder=[None],
    )
    return TurnRunner(runner, ctx), ctx


def _closing_schedule(coro, log_message, loop=None):
    """Test double for TurnRunner._schedule: records nothing, just avoids a real event loop
    and an unawaited-coroutine warning. adapter.send_tool_progress_line is an AsyncMock, so its
    call args are already recorded synchronously before this ever runs."""
    coro.close()


class TestSendToToolThread:
    def test_non_discord_platform_falls_through(self):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: True)
        tr, _ = _make_runner_and_ctx(platform=Platform.SLACK, adapter=adapter)
        assert tr._send_to_tool_thread("hi") is False
        adapter.send_tool_progress_line.assert_not_called()

    def test_adapter_without_the_method_falls_through(self):
        adapter = SimpleNamespace()  # no send_tool_progress_line at all
        tr, _ = _make_runner_and_ctx(adapter=adapter)
        assert tr._send_to_tool_thread("hi") is False

    def test_flag_off_falls_through(self):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: False)
        tr, _ = _make_runner_and_ctx(adapter=adapter)
        assert tr._send_to_tool_thread("hi") is False
        adapter.send_tool_progress_line.assert_not_called()

    def test_flag_on_schedules_send_and_returns_true(self, monkeypatch):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: True)
        tr, ctx = _make_runner_and_ctx(adapter=adapter)
        monkeypatch.setattr(tr, "_schedule", _closing_schedule)

        result = tr._send_to_tool_thread("🖥️ terminal: ls")

        assert result is True
        adapter.send_tool_progress_line.assert_called_once_with("123", "456", "🖥️ terminal: ls")


class TestProgressEmitRouting:
    def test_tool_line_goes_to_thread_not_queue_when_enabled(self, monkeypatch):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: True)
        tr, ctx = _make_runner_and_ctx(adapter=adapter)
        monkeypatch.setattr(tr, "_schedule", _closing_schedule)

        tr._progress_emit("🖥️ terminal: ls")

        ctx.progress_queue.put.assert_not_called()
        adapter.send_tool_progress_line.assert_called_once_with("123", "456", "🖥️ terminal: ls")

    def test_tool_line_goes_to_queue_unchanged_when_disabled(self):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: False)
        tr, ctx = _make_runner_and_ctx(adapter=adapter)

        tr._progress_emit("🖥️ terminal: ls")

        ctx.progress_queue.put.assert_called_once_with("🖥️ terminal: ls")
        adapter.send_tool_progress_line.assert_not_called()

    def test_non_discord_platform_unaffected(self):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: True)
        tr, ctx = _make_runner_and_ctx(platform=Platform.SLACK, adapter=adapter)

        tr._progress_emit("some tool line")

        ctx.progress_queue.put.assert_called_once_with("some tool line")
        adapter.send_tool_progress_line.assert_not_called()

    def test_dedup_repeat_renders_count_suffix_to_thread(self, monkeypatch):
        adapter = SimpleNamespace(send_tool_progress_line=AsyncMock(), _discord_thread_tool_calls_enabled=lambda: True)
        tr, ctx = _make_runner_and_ctx(adapter=adapter)
        monkeypatch.setattr(tr, "_schedule", _closing_schedule)

        tr._progress_emit("🖥️ terminal: ls")
        tr._progress_emit("🖥️ terminal: ls")  # same line again -> dedup path

        assert adapter.send_tool_progress_line.await_count == 0  # not awaited (test double closes it)
        assert adapter.send_tool_progress_line.call_count == 2
        second_call_text = adapter.send_tool_progress_line.call_args_list[1].args[2]
        assert "(×2)" in second_call_text
        ctx.progress_queue.put.assert_not_called()
