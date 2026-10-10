"""Stopping a busy Discord turn closes its detail thread before releasing the slot."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_discord_tool_progress_thread import _make_adapter


@pytest.mark.asyncio
async def test_stop_removes_member_before_releasing_busy_turn(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: None)
    runner = GatewayRunner(config=GatewayConfig())
    adapter = _make_adapter()
    adapter._stop_typing_quietly = AsyncMock()
    adapter._client.user = SimpleNamespace(id=999)
    member = SimpleNamespace(id=42)
    thread = SimpleNamespace(fetch_members=AsyncMock(return_value=[member]),
                             remove_user=AsyncMock(), edit=AsyncMock(), send=AsyncMock())
    adapter._tool_progress_threads["789"] = thread
    runner._delivery_adapter_for = lambda source: adapter
    runner._interrupt_running_turn = lambda *args, **kwargs: 1
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="42", chat_type="group")
    turn = runner._session_state("key").turn
    turn.agent = SimpleNamespace(session_id="session")
    turn.event = MessageEvent(text="original", source=source, message_id="456", reply_anchor_override="789")
    turn.ctx = TurnContext(event_message_id="789", inbound_message_id="789")
    def release(*args, **kwargs):
        thread.remove_user.assert_awaited_once_with(member)
        thread.edit.assert_awaited_once_with(archived=True)
    runner._drop_turn_slot = release
    await runner._interrupt_and_clear_session("key", source, interrupt_reason="Stop requested",
                                              invalidation_reason="stop_command")
    await adapter.send_tool_progress_line("123", "789", "late tool callback")
    thread.send.assert_not_awaited()
