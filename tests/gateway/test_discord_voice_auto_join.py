"""Tests for discord.auto_join_voice: Jarvis joins a real human's voice channel automatically,
no /voice join needed.

Covers two layers:
  - DiscordAdapter._voice_auto_join_target: the pure decision of whether THIS voice-state-update
    should trigger an auto-join (extracted from on_voice_state_update's closure specifically so
    it's unit-testable without a live discord.py client).
  - GatewayRunner._handle_voice_auto_join: reads the configured text channel and joins through
    the same _join_voice_channel_core /voice join itself uses.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig

# ── DiscordAdapter._voice_auto_join_target ──────────────────────────────────────────────────

def _make_adapter(*, auto_join_enabled=True, handler=AsyncMock()):
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.platforms.helpers import ThreadParticipationTracker

    config = PlatformConfig(enabled=True, token="test-token")
    if auto_join_enabled:
        config.extra["auto_join_voice"] = True
    with patch.object(ThreadParticipationTracker, "_load", return_value=set()):
        adapter = DiscordAdapter(config)
    adapter._voice_auto_join_handler = handler
    return adapter


def _voice_state(channel=None):
    return SimpleNamespace(channel=channel)


def _member(*, is_bot=False):
    return SimpleNamespace(bot=is_bot, display_name="Tony")


def _channel(guild_id=555):
    return SimpleNamespace(guild=SimpleNamespace(id=guild_id))


class TestVoiceAutoJoinTargetSync:
    def test_defaults_off(self):
        adapter = _make_adapter(auto_join_enabled=False)
        target = adapter._voice_auto_join_target(_member(), _voice_state(), _voice_state(_channel()))
        assert target is None

    def test_human_joining_fresh_channel_with_flag_on_returns_channel(self):
        adapter = _make_adapter()
        channel = _channel()
        target = adapter._voice_auto_join_target(_member(), _voice_state(), _voice_state(channel))
        assert target is channel

    def test_bot_member_is_ignored(self):
        adapter = _make_adapter()
        channel = _channel()
        target = adapter._voice_auto_join_target(_member(is_bot=True), _voice_state(), _voice_state(channel))
        assert target is None

    def test_leaving_a_channel_is_not_a_join(self):
        adapter = _make_adapter()
        channel = _channel()
        target = adapter._voice_auto_join_target(_member(), _voice_state(channel), _voice_state())
        assert target is None

    def test_switching_channels_is_not_a_fresh_join(self):
        adapter = _make_adapter()
        target = adapter._voice_auto_join_target(_member(), _voice_state(_channel()), _voice_state(_channel(guild_id=556)))
        assert target is None

    def test_no_handler_wired_returns_none(self):
        adapter = _make_adapter(handler=None)
        target = adapter._voice_auto_join_target(_member(), _voice_state(), _voice_state(_channel()))
        assert target is None

    def test_already_connected_in_that_guild_returns_none(self):
        adapter = _make_adapter()
        channel = _channel(guild_id=777)
        adapter._voice_clients[777] = SimpleNamespace(is_connected=lambda: True)
        target = adapter._voice_auto_join_target(_member(), _voice_state(), _voice_state(channel))
        assert target is None

    def test_stale_disconnected_client_does_not_block_auto_join(self):
        adapter = _make_adapter()
        channel = _channel(guild_id=888)
        adapter._voice_clients[888] = SimpleNamespace(is_connected=lambda: False)
        target = adapter._voice_auto_join_target(_member(), _voice_state(), _voice_state(channel))
        assert target is channel


class TestAutoJoinVoiceFlag:
    def test_defaults_off(self):
        adapter = _make_adapter(auto_join_enabled=False)
        assert adapter._discord_auto_join_voice_enabled() is False

    def test_extra_config_turns_it_on(self):
        adapter = _make_adapter(auto_join_enabled=False)
        adapter.config.extra["auto_join_voice"] = True
        assert adapter._discord_auto_join_voice_enabled() is True

    def test_setter_stores_handler(self):
        adapter = _make_adapter(handler=None)
        sentinel = AsyncMock()
        adapter.set_voice_auto_join_handler(sentinel)
        assert adapter._voice_auto_join_handler is sentinel


# ── GatewayRunner._handle_voice_auto_join ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHandleVoiceAutoJoin:
    async def test_no_configured_text_channel_skips_join(self, monkeypatch):
        from gateway.run_voice import GatewayVoiceMixin

        class _Runner(GatewayVoiceMixin):
            pass

        runner = _Runner()
        runner._join_voice_channel_core = AsyncMock()
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"platforms": {"discord": {}}})

        await runner._handle_voice_auto_join(SimpleNamespace(), _member(), _channel())

        runner._join_voice_channel_core.assert_not_called()

    async def test_configured_text_channel_joins_through_shared_core(self, monkeypatch):
        from gateway.run_voice import GatewayVoiceMixin

        class _Runner(GatewayVoiceMixin):
            pass

        runner = _Runner()
        runner._join_voice_channel_core = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"platforms": {"discord": {"auto_join_voice_text_channel": "1487966894739689566"}}},
        )
        adapter = SimpleNamespace(_owner_profile="default")
        channel = _channel()

        await runner._handle_voice_auto_join(adapter, _member(), channel)

        runner._join_voice_channel_core.assert_awaited_once_with(
            adapter, channel, text_channel_id=1487966894739689566, voice_profile="default",
        )

    async def test_join_failure_is_logged_not_raised(self, monkeypatch):
        from gateway.run_voice import GatewayVoiceMixin

        class _Runner(GatewayVoiceMixin):
            pass

        runner = _Runner()
        runner._join_voice_channel_core = AsyncMock(return_value="Failed to join voice channel. Check bot permissions.")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"platforms": {"discord": {"auto_join_voice_text_channel": "123"}}},
        )

        # Must not raise -- this is fire-and-forget from the adapter's event handler.
        await runner._handle_voice_auto_join(SimpleNamespace(), _member(), _channel())
