"""Tests for the Discord no-trace cleanup: messages sent/received while mode-state.json's
``kind`` is "uncensored" get deleted once the fleet switches away from it (both Tony's own
prompts and the bot's replies -- extends the mode plugin's existing no-trace guarantee, see
~/.hermes/plugins/mode/session_summary.py, to Discord message history itself).

Covers two layers:
  - _DiscordUncensoredMessageTracker.maybe_sweep/track: the pure persisted state machine
    (kind edge-detection, what gets returned for deletion and when).
  - DiscordAdapter._sweep_uncensored_messages_if_needed / _track_if_uncensored: the Discord-
    facing side (bulk vs individual delete, best-effort error handling), against mocked channels.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.discord.adapter as discord_adapter_module
from plugins.platforms.discord.adapter import (
    DiscordAdapter, _DiscordUncensoredMessageTracker, _current_mode_kind,
)


# ── _current_mode_kind ───────────────────────────────────────────────────────────────────────

def test_current_mode_kind_reads_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "mode-state.json").write_text(json.dumps({"kind": "uncensored"}))
    assert _current_mode_kind() == "uncensored"


def test_current_mode_kind_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert _current_mode_kind() is None


def test_current_mode_kind_malformed_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "mode-state.json").write_text("not json")
    assert _current_mode_kind() is None


# ── _DiscordUncensoredMessageTracker ─────────────────────────────────────────────────────────

@pytest.fixture()
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return _DiscordUncensoredMessageTracker()


def _set_kind(monkeypatch, kind):
    monkeypatch.setattr(discord_adapter_module, "_current_mode_kind", lambda: kind)


class TestMaybeSweep:
    @pytest.mark.asyncio
    async def test_entering_uncensored_tracks_kind_no_delete(self, tracker, monkeypatch):
        _set_kind(monkeypatch, "uncensored")
        result = await tracker.maybe_sweep()
        assert result is None
        assert tracker.is_currently_uncensored() is True

    @pytest.mark.asyncio
    async def test_staying_uncensored_is_a_no_op(self, tracker, monkeypatch):
        _set_kind(monkeypatch, "uncensored")
        await tracker.maybe_sweep()
        await tracker.track("111", "222")
        result = await tracker.maybe_sweep()
        assert result is None
        assert tracker.is_currently_uncensored() is True

    @pytest.mark.asyncio
    async def test_leaving_uncensored_with_messages_returns_them_and_clears(self, tracker, monkeypatch):
        _set_kind(monkeypatch, "uncensored")
        await tracker.maybe_sweep()
        await tracker.track("111", "222")
        await tracker.track("111", "223")

        _set_kind(monkeypatch, "cloud")
        result = await tracker.maybe_sweep()

        assert result == [
            {"channel_id": "111", "message_id": "222"},
            {"channel_id": "111", "message_id": "223"},
        ]
        assert tracker.is_currently_uncensored() is False
        # Cleared -- a second sweep call finds nothing left to delete.
        result2 = await tracker.maybe_sweep()
        assert result2 is None

    @pytest.mark.asyncio
    async def test_leaving_uncensored_with_no_tracked_messages_returns_none(self, tracker, monkeypatch):
        _set_kind(monkeypatch, "uncensored")
        await tracker.maybe_sweep()
        _set_kind(monkeypatch, "local")
        result = await tracker.maybe_sweep()
        assert result is None
        assert tracker.is_currently_uncensored() is False

    @pytest.mark.asyncio
    async def test_unreadable_kind_never_triggers_a_delete(self, tracker, monkeypatch):
        _set_kind(monkeypatch, "uncensored")
        await tracker.maybe_sweep()
        await tracker.track("111", "222")

        _set_kind(monkeypatch, None)  # simulated read failure
        result = await tracker.maybe_sweep()

        assert result is None
        # Still reports uncensored -- an unreadable state must not silently drop the pending
        # cleanup or the "currently uncensored" fact.
        assert tracker.is_currently_uncensored() is True

    @pytest.mark.asyncio
    async def test_state_survives_a_fresh_instance_reading_the_same_file(self, tracker, monkeypatch, tmp_path):
        _set_kind(monkeypatch, "uncensored")
        await tracker.maybe_sweep()
        await tracker.track("111", "222")

        reloaded = _DiscordUncensoredMessageTracker()
        assert reloaded.is_currently_uncensored() is True
        _set_kind(monkeypatch, "cloud")
        result = await reloaded.maybe_sweep()
        assert result == [{"channel_id": "111", "message_id": "222"}]


# ── DiscordAdapter integration: sweep + delete ───────────────────────────────────────────────

def _make_adapter(monkeypatch, tmp_path):
    from gateway.platforms.helpers import ThreadParticipationTracker
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(enabled=True, token="test-token")
    with patch.object(ThreadParticipationTracker, "_load", return_value=set()):
        adapter = DiscordAdapter(config)
    return adapter


class TestSweepAndDelete:
    @pytest.mark.asyncio
    async def test_sweep_bulk_deletes_multiple_tracked_messages(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch, tmp_path)
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "222")
        await adapter._track_if_uncensored("111", "223")

        channel = MagicMock()
        channel.delete_messages = AsyncMock()
        adapter._resolve_channel = AsyncMock(return_value=channel)

        _set_kind(monkeypatch, "cloud")
        await adapter._sweep_uncensored_messages_if_needed()

        channel.delete_messages.assert_awaited_once()
        deleted_ids = {obj.id for obj in channel.delete_messages.call_args.args[0]}
        assert deleted_ids == {222, 223}

    @pytest.mark.asyncio
    async def test_sweep_falls_back_to_individual_delete_for_a_single_message(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch, tmp_path)
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "222")

        msg = AsyncMock()
        channel = MagicMock()
        channel.fetch_message = AsyncMock(return_value=msg)
        adapter._resolve_channel = AsyncMock(return_value=channel)

        _set_kind(monkeypatch, "cloud")
        await adapter._sweep_uncensored_messages_if_needed()

        channel.fetch_message.assert_awaited_once_with(222)
        msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sweep_is_best_effort_across_channels(self, monkeypatch, tmp_path):
        """One channel failing to resolve must not stop the other channel's messages from
        being deleted."""
        adapter = _make_adapter(monkeypatch, tmp_path)
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "222")
        await adapter._track_if_uncensored("999", "888")

        good_msg = AsyncMock()
        good_channel = MagicMock()
        good_channel.fetch_message = AsyncMock(return_value=good_msg)

        async def _resolve(channel_id):
            if str(channel_id) == "999":
                raise RuntimeError("channel gone")
            return good_channel

        adapter._resolve_channel = _resolve

        _set_kind(monkeypatch, "cloud")
        await adapter._sweep_uncensored_messages_if_needed()

        good_msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_sweep_while_still_uncensored(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch, tmp_path)
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "222")

        adapter._resolve_channel = AsyncMock()
        await adapter._sweep_uncensored_messages_if_needed()

        adapter._resolve_channel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_message_sent_after_leaving_uncensored_is_not_tracked(self, monkeypatch, tmp_path):
        """The confirmation reply sent right as the mode switch completes must land in the NEW
        kind, not get folded into the just-swept uncensored batch."""
        adapter = _make_adapter(monkeypatch, tmp_path)
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "222")

        adapter._resolve_channel = AsyncMock(return_value=None)
        _set_kind(monkeypatch, "cloud")
        # This call both sweeps (222) and, being post-transition, must not track 333 as uncensored.
        await adapter._track_if_uncensored("111", "333")

        assert adapter._uncensored_messages.is_currently_uncensored() is False
        # A later leave-uncensored-again cycle would start fresh -- 333 must never surface.
        _set_kind(monkeypatch, "uncensored")
        await adapter._track_if_uncensored("111", "444")
        _set_kind(monkeypatch, "cloud")
        result = await adapter._uncensored_messages.maybe_sweep()
        assert result == [{"channel_id": "111", "message_id": "444"}]
