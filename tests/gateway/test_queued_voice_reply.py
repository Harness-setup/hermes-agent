"""Queued voice replies must speak the same final answer delivered as text."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import Platform
from gateway.run_voice import GatewayVoiceMixin
from gateway.platforms.event import MessageType


@pytest.mark.asyncio
@pytest.mark.parametrize('voice,completed,enabled,synthesized,expected', [
    (True, False, True, True, 1), (False, False, True, True, 0),
    (True, True, True, True, 0), (True, False, False, True, 0),
    (True, False, True, False, 1),
])
async def test_queued_reply_speech_and_dedup(voice, completed, enabled, synthesized, expected):
    ctx = SimpleNamespace(voice_input=voice, session_key='session', run_generation=1,
                          source=SimpleNamespace(platform=Platform.DISCORD, chat_id='chat', guild_id='42'))
    adapter = SimpleNamespace(_streaming_tts_turn_completed=MagicMock(return_value=completed),
                              _mark_streaming_tts_completed_turn=MagicMock())
    runner = SimpleNamespace(_should_send_voice_reply=MagicMock(return_value=enabled),
                             _send_voice_reply=AsyncMock(return_value=synthesized))
    result = {}
    await GatewayVoiceMixin._send_queued_voice_reply(runner, ctx, adapter, result, 'The final written answer.')
    assert runner._send_voice_reply.await_count == expected
    if expected:
        event, speech = runner._send_voice_reply.await_args.args
        assert event.message_type == MessageType.VOICE
        assert event.raw_message.guild_id == '42'
        assert speech == 'The final written answer.'
    assert bool(result.get("voice_reply_delivered")) == bool(expected and synthesized)
    adapter._mark_streaming_tts_completed_turn.assert_not_called()


@pytest.mark.asyncio
async def test_each_queued_answer_speaks_even_with_shared_generation():
    ctx = SimpleNamespace(voice_input=True, session_key='session', run_generation=1,
                          source=SimpleNamespace(platform=Platform.DISCORD, chat_id='chat', guild_id='42'))
    adapter = SimpleNamespace(_streaming_tts_turn_completed=MagicMock(return_value=False))
    runner = SimpleNamespace(_should_send_voice_reply=MagicMock(return_value=True),
                             _send_voice_reply=AsyncMock(return_value=True))
    first, second = {}, {}
    for result, text in [(first, 'First answer.'), (first, 'First answer.'), (second, 'Second answer.')]:
        await GatewayVoiceMixin._send_queued_voice_reply(runner, ctx, adapter, result, text)
    assert [call.args[1] for call in runner._send_voice_reply.await_args_list] == ['First answer.', 'Second answer.']


@pytest.mark.asyncio
async def test_queued_text_survives_speech_failure():
    from gateway.run_turn import GatewayTurnMixin
    ctx = SimpleNamespace(mute_notification_reply=False, session_key='session',
                          stream_consumer_holder=[None], source=SimpleNamespace(),
                          _status_thread_metadata=None, event_message_id=None,
                          inbound_message_id='voice-1', run_generation=1)
    runner = SimpleNamespace(_run_agent_stream_confirmed_final_delivery=lambda *a, **k: False,
                             _is_intentional_silence=lambda *a: False,
                             _send_queued_voice_reply=AsyncMock(side_effect=RuntimeError('TTS failed')),
                             _deliver_queued_first_response=AsyncMock(return_value=True),
                             _pop_post_delivery_callback=lambda *a: None)
    order = []
    async def deliver(*args, **kwargs):
        order.append('text')
        return True
    async def speak(*args, **kwargs):
        order.append('speech')
        raise RuntimeError('TTS failed')
    runner._deliver_queued_first_response.side_effect = deliver
    runner._send_queued_voice_reply.side_effect = speak
    result = {'final_response': 'The same final answer.'}
    await GatewayTurnMixin._run_agent_deliver_first_response(runner, ctx, SimpleNamespace(), result, result, None)
    runner._send_queued_voice_reply.assert_awaited_once()
    assert runner._deliver_queued_first_response.await_args.args[0] == result['final_response']
    assert result['already_sent'] is True
    assert not result.get('voice_reply_delivered')
    assert order == ['text', 'speech']
