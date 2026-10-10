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
from gateway.run import GatewayRunner

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


@pytest.mark.asyncio
async def test_owner_greeting_duplicates_departure_and_guild_isolation(monkeypatch):
    import asyncio
    from plugins.platforms.discord import adapter_voice_playback as playback
    adapter = _make_adapter()
    adapter._owner_profile = 'test-profile'
    channels = [SimpleNamespace(id=10, guild=SimpleNamespace(id=1)),
                SimpleNamespace(id=20, guild=SimpleNamespace(id=2))]
    member = SimpleNamespace(id=42, guild=channels[0].guild)
    for ch in channels:
        adapter._voice_clients[ch.guild.id] = SimpleNamespace(is_connected=lambda: True, channel=ch)
        adapter._voice_text_channels[ch.guild.id] = 100 + ch.guild.id
    spoken = []
    async def speak(*args, **kwargs):
        spoken.append(args[2])
    monkeypatch.setattr('gateway.voice_phrases.speak_phrase', speak)
    from gateway.voice_profile import DEFAULT_PHRASES
    monkeypatch.setattr('gateway.voice_profile.resolve_voice_profile', lambda: {'phrases': DEFAULT_PHRASES})
    adapter.claim_voice_owner(member, channels[0])
    adapter.claim_voice_owner(member, channels[0])
    await adapter._voice_greetings[1]
    await adapter._handle_voice_owner_state(member, _voice_state(), _voice_state(channels[0]))
    assert len(spoken) == 1
    second = SimpleNamespace(id=99, guild=channels[1].guild)
    adapter.claim_voice_owner(second, channels[1])
    await adapter._voice_greetings[2]
    adapter.discard_pending_voice_input = lambda gid: order.append(('discard', gid))
    adapter._on_voice_disconnect = lambda chat: order.append(('cleanup', chat))
    adapter.leave_voice_channel = AsyncMock(side_effect=lambda gid: order.append(('leave', gid)))
    order = []
    unrelated = SimpleNamespace(id=7, guild=member.guild)
    await adapter._handle_voice_owner_state(unrelated, _voice_state(channels[0]), _voice_state())
    assert order == []
    # Others still present does not change the owner's departure rule.
    channels[0].members = [unrelated]
    await adapter._handle_voice_owner_state(member, _voice_state(channels[0]), _voice_state(channels[1]))
    assert order == [('discard', 1), ('cleanup', '101'), ('leave', 1)]
    assert adapter._voice_owners[2][0] == 99


@pytest.mark.asyncio
async def test_failed_connection_cannot_claim_owner(monkeypatch):
    adapter = _make_adapter()
    channel = SimpleNamespace(id=10, guild=SimpleNamespace(id=1))
    adapter.claim_voice_owner(SimpleNamespace(id=42), channel)
    assert not adapter._voice_owners
    assert not adapter._voice_greetings


@pytest.mark.asyncio
async def test_owner_departure_uses_real_leave_without_flushing_speech(monkeypatch):
    from unittest.mock import MagicMock
    from gateway.voice_profile import DEFAULT_PHRASES
    monkeypatch.setattr('gateway.voice_profile.resolve_voice_profile', lambda: {'phrases': DEFAULT_PHRASES})
    monkeypatch.setattr('gateway.voice_phrases.speak_phrase', AsyncMock())
    adapter = _make_adapter()
    adapter._client = SimpleNamespace(get_guild=lambda gid: SimpleNamespace())
    channel = SimpleNamespace(id=10, guild=SimpleNamespace(id=1), members=[SimpleNamespace(id=7)])
    vc = MagicMock()
    vc.channel = channel
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    vc.disconnect = AsyncMock()
    adapter._voice_clients[1] = vc
    adapter._voice_text_channels[1] = 101
    receiver = MagicMock()
    # Even speech captured after discard must not become a leave-flush response.
    receiver.flush_pending.return_value = [(42, b'pending')]
    adapter._voice_receivers[1] = receiver
    adapter._process_voice_input = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner._voice_mode = {'discord:101': 'all'}
    runner._voice_phrase_turns = {}
    runner._save_voice_modes = MagicMock()
    adapter._on_voice_disconnect = MagicMock(side_effect=lambda chat: runner._handle_voice_timeout_cleanup(chat, adapter=adapter))
    member = SimpleNamespace(id=42, guild=channel.guild)
    adapter.claim_voice_owner(member, channel)
    await adapter._handle_voice_owner_state(member, _voice_state(channel), _voice_state())
    receiver.discard_pending.assert_called_once()
    adapter._process_voice_input.assert_not_awaited()
    vc.stop.assert_called_once()
    vc.disconnect.assert_awaited_once()
    adapter._on_voice_disconnect.assert_called_once_with('101')
    assert runner._voice_mode['discord:101'] == 'off'
    runner._save_voice_modes.assert_called_once()
    assert '101' in adapter._auto_tts_disabled_chats
    assert not adapter._voice_owners
    assert not adapter._voice_greetings


@pytest.mark.asyncio
async def test_duplicate_join_events_connect_and_greet_once(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock
    from plugins.platforms.discord import adapter as discord
    from plugins.platforms.discord.adapter_voice_playback import handle_voice_state_update
    from gateway.voice_profile import DEFAULT_PHRASES
    adapter = _make_adapter()
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=10, name='General', guild=guild)
    member = SimpleNamespace(id=42, display_name='Tony', bot=False, guild=guild)
    connected = asyncio.Event()
    release = asyncio.Event()
    vc = MagicMock()
    vc.channel = channel
    vc.is_connected.return_value = True
    async def connect():
        connected.set()
        await release.wait()
        return vc
    channel.connect = AsyncMock(side_effect=connect)
    monkeypatch.setattr(discord, 'DISCORD_AVAILABLE', True)
    monkeypatch.setattr(discord, 'VoiceReceiver', MagicMock())
    monkeypatch.setattr(adapter, '_voice_listen_loop', AsyncMock())
    monkeypatch.setattr(adapter, '_reset_voice_timeout', MagicMock())
    monkeypatch.setattr('gateway.voice_profile.resolve_voice_profile', lambda: {'phrases': DEFAULT_PHRASES})
    speak = AsyncMock()
    monkeypatch.setattr('gateway.voice_phrases.speak_phrase', speak)
    async def auto_join(adapter, member, channel):
        assert await adapter.join_voice_channel(channel, text_channel_id=101)
        adapter.claim_voice_owner(member, channel)
    adapter._voice_auto_join_handler = auto_join
    events = [asyncio.create_task(handle_voice_state_update(adapter, member, _voice_state(), _voice_state(channel)))
              for _ in range(2)]
    await asyncio.wait_for(connected.wait(), 2)
    release.set()
    await asyncio.gather(*events)
    await adapter._voice_greetings[1]
    channel.connect.assert_awaited_once()
    speak.assert_awaited_once()
    assert adapter._voice_owners == {1: (42, 10)}


@pytest.mark.asyncio
async def test_owner_reentry_greets_and_profile_connections_are_isolated(monkeypatch):
    from gateway.voice_profile import DEFAULT_PHRASES
    from unittest.mock import MagicMock
    monkeypatch.setattr('gateway.voice_profile.resolve_voice_profile', lambda: {'phrases': DEFAULT_PHRASES})
    speak = AsyncMock()
    monkeypatch.setattr('gateway.voice_phrases.speak_phrase', speak)
    first, second = _make_adapter(), _make_adapter()
    channel = SimpleNamespace(id=10, guild=SimpleNamespace(id=1))
    owner = SimpleNamespace(id=42, guild=channel.guild)
    for adapter, user in [(first, owner), (second, SimpleNamespace(id=99))]:
        adapter._voice_clients[1] = SimpleNamespace(is_connected=lambda: True, channel=channel)
        adapter._voice_text_channels[1] = 101
        adapter.claim_voice_owner(user, channel)
        await adapter._voice_greetings[1]
    # Reentering a retained connection does not repeat the startup greeting.
    first._voice_owner_channels[1] = None
    await first._handle_voice_owner_state(owner, _voice_state(), _voice_state(channel))
    await first._voice_greetings[1]
    await first._handle_voice_owner_state(owner, _voice_state(), _voice_state(channel))
    assert speak.await_count == 2
    assert second._voice_owners == {1: (99, 10)}
    assert first._spoken_phrase_picker is not second._spoken_phrase_picker
