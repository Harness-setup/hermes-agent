"""An accepted Discord steer moves detail delivery without changing steer timing."""

from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy", "command", "priority"])
async def test_accepted_discord_steer_moves_anchor_without_redirect(route, tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    agent = Mock()
    agent.steer.return_value = True
    agent._active_children = []
    source = SessionSource(platform=Platform.DISCORD, chat_id="c", user_id="u", chat_type="group")
    opening = MessageEvent(text="original prompt", source=source, message_id="A")
    ctx = TurnContext(source=source, event_message_id="A", inbound_message_id="A")
    turn = runner._session_state("key").turn
    turn.agent, turn.event, turn.ctx = agent, opening, ctx
    event = MessageEvent(text="/steer revised prompt" if route == "command" else "revised prompt",
                         source=source, message_id="B")
    if route == "busy":
        result = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)
        assert result.steered
    elif route == "priority":
        runner._hm_busy_steer(event, agent, "key")
    else:
        await runner._busy_steer_command(event, "key", source)
    assert ctx.event_message_id == ctx.inbound_message_id == "B"
    assert opening.reply_anchor_override == "B"
    agent.steer.assert_called_once()
    agent.redirect.assert_not_called()
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_refused_steer_keeps_original_thread_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    agent = Mock()
    agent.steer.return_value = False
    agent._active_children = []
    source = SessionSource(platform=Platform.DISCORD, chat_id="c", user_id="u", chat_type="group")
    turn = runner._session_state("key").turn
    turn.agent = agent
    turn.ctx = TurnContext(source=source, event_message_id="A", inbound_message_id="A")
    event = MessageEvent(text="revised prompt", source=source, message_id="B")
    result = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)
    assert not result.steered
    assert turn.ctx.event_message_id == turn.ctx.inbound_message_id == "A"
