import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.voice_profile import DEFAULT_PHRASES, DEFAULT_TIMING
from gateway import voice_phrases as vp


def profile():
    return {'revision': 'test', 'phrases': copy.deepcopy(DEFAULT_PHRASES), 'timing': {**DEFAULT_TIMING, 'filler_delay_seconds': 0, 'filler_jitter_seconds': 0, 'status_silence_seconds': 0}}


@pytest.mark.asyncio
async def test_no_turn_greeting_one_filler_and_no_tool_ack(monkeypatch):
    spoken = []
    filled = asyncio.Event()
    async def speak(adapter, chat, text, **kwargs):
        spoken.append(text)
        if len(spoken) == 1:
            filled.set()
    monkeypatch.setattr(vp, 'speak_phrase', speak)
    turn = vp.VoicePhraseTurn(SimpleNamespace(), 'chat', profile(), lambda: True)
    await turn.start()
    await asyncio.wait_for(filled.wait(), 2)
    await turn.close()
    assert len(spoken) == 1
    assert spoken[0] in DEFAULT_PHRASES['fillers']
    from gateway.run_turn_runner import TurnRunner
    TurnRunner(None, SimpleNamespace()).voice_ack_callback('id', 'terminal', {})
    assert len(spoken) == 1


@pytest.mark.asyncio
async def test_answer_cancels_filler_and_late_synthesis_cannot_play(monkeypatch):
    spoken = []
    started, release = asyncio.Event(), asyncio.Event()
    async def render(text):
        started.set()
        await release.wait()
        return b'audio', '.wav'
    # Exercise the actual synthesis-to-playback guard with an in-flight worker.
    monkeypatch.setattr(vp.asyncio, 'to_thread', lambda fn, *args: render(*args))
    adapter = SimpleNamespace(play_tts=AsyncMock())
    current = True
    task = asyncio.create_task(vp.speak_phrase(adapter, 'chat', 'filler', allowed=lambda: current))
    await asyncio.wait_for(started.wait(), 2)
    current = False
    release.set()
    await task
    adapter.play_tts.assert_not_awaited()
    monkeypatch.setattr(vp, 'speak_phrase', AsyncMock(side_effect=lambda *a, **k: spoken.append(a[2])))
    turn = vp.VoicePhraseTurn(adapter, 'chat', profile(), lambda: True)
    await turn.start()
    turn.answer_started()
    await turn.close()
    assert len(spoken) == 0


def runner(mode='voice_only'):
    from contextlib import nullcontext
    adapter = SimpleNamespace(_voice_text_channels={1: 'chat'}, is_in_voice_channel=lambda gid: True,
                              _should_auto_tts_for_chat=lambda chat: False)
    return SimpleNamespace(_delivery_adapter_for=lambda source: adapter,
                           _voice_mode={'key': mode}, _voice_key_for_source=lambda source: 'key',
                           _profile_scope_for_source=lambda source: nullcontext())


@pytest.mark.asyncio
async def test_text_off_events_revision_and_failures(monkeypatch):
    source = SimpleNamespace(platform=Platform.DISCORD, chat_id='chat')
    r = runner()
    p = profile()
    monkeypatch.setattr(vp, 'resolve_voice_profile', lambda: p)
    assert vp.make_voice_phrase_turn(r, source, False, lambda: True) is None
    assert vp.make_voice_phrase_turn(runner('off'), source, True, lambda: True) is None
    first = vp.make_voice_phrase_turn(r, source, True, lambda: True)
    p = {**profile(), 'revision': 'next'}
    second = vp.make_voice_phrase_turn(r, source, True, lambda: True)
    assert first.profile['revision'] != second.profile['revision']
    speak = AsyncMock()
    monkeypatch.setattr(vp, 'speak_phrase', speak)
    await vp.speak_event_phrase(r, source, 'stop', voice_input=True)
    assert speak.call_args.args[2] in DEFAULT_PHRASES['stop']
    await vp.speak_event_phrase(r, source, 'assistant_error', voice_input=True)
    assert speak.call_args.args[2] in DEFAULT_PHRASES['assistant_error']
    await vp.speak_event_phrase(runner('off'), source, 'stop', voice_input=True)
    assert speak.await_count == 2


@pytest.mark.asyncio
async def test_tts_failure_does_not_escape(monkeypatch):
    def fail(text):
        raise RuntimeError('tts failed')
    monkeypatch.setattr(vp, '_render_phrase', fail)
    adapter = SimpleNamespace(play_tts=AsyncMock())
    await vp.speak_phrase(adapter, 'chat', 'hello')
    adapter.play_tts.assert_not_awaited()


@pytest.mark.parametrize('voice', [True, False])
@pytest.mark.asyncio
async def test_gateway_wires_ack_delta_cancel_and_spoken_style(monkeypatch, voice):
    import sys
    import types
    from gateway.session import SessionSource
    from tests.gateway.test_reasoning_command import _make_runner, _CapturingAgent
    from tui_gateway.voice_reply_style import voice_chat_turn_note
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {'streaming': {'enabled': False}})
    monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
        'provider': 'custom', 'api_mode': 'chat_completions',
        'base_url': 'https://example.invalid/v1', 'api_key': 'test-key'})
    controller = SimpleNamespace(start=AsyncMock(), answer_started=lambda: signals.append('delta'), close=AsyncMock())
    signals, notes = [], []
    def make(runner, source, voice_input, current, session_key=None):
        return controller if voice_input else None
    monkeypatch.setattr(vp, 'make_voice_phrase_turn', make)
    class Agent(_CapturingAgent):
        def run_conversation(self, *args, **kwargs):
            notes.append(self._gateway_turn_context_notes)
            if self.stream_delta_callback is not None:
                self.stream_delta_callback('Answer.')
            if self.tool_start_callback is not None:
                self.tool_start_callback('call', 'terminal', {})
            return {'final_response': 'Answer.', 'messages': [], 'api_calls': 1}
    module = types.ModuleType('run_agent')
    module.AIAgent = Agent
    monkeypatch.setitem(sys.modules, 'run_agent', module)
    r = _make_runner()
    r._voice_mode = {}
    result = await r._run_agent(message='ping', context_prompt='', history=[],
        source=SessionSource(platform=Platform.DISCORD, chat_id='chat', user_id='speaker'),
        session_id='turn', session_key='agent:main:discord:voice', message_type='voice' if voice else 'text')
    assert result['final_response'] == 'Answer.'
    assert controller.start.await_count == int(voice)
    assert signals == (['delta'] if voice else [])
    assert (voice_chat_turn_note() in notes[0]) is voice
    assert controller.close.await_count == (2 if voice else 0)


@pytest.mark.asyncio
async def test_discord_cancel_stops_audible_filler(monkeypatch):
    from tests.gateway.test_discord_voice_mixer import _make_adapter
    from unittest.mock import MagicMock
    from plugins.platforms.discord import adapter as discord
    adapter = _make_adapter()
    vc = MagicMock()
    vc.is_connected.return_value = True
    adapter._voice_clients[1] = vc
    played = asyncio.Event()
    mixer = SimpleNamespace(speech_active=True, play_speech=lambda *a, **k: played.set(), stop_speech=MagicMock())
    adapter._voice_mixers[1] = mixer
    monkeypatch.setattr(adapter, '_playback_timeout_for_audio', AsyncMock(return_value=10))
    monkeypatch.setattr(discord._voice_mixer_module(), 'decode_to_pcm', lambda path: b'pcm')
    task = asyncio.create_task(adapter.play_in_voice_channel(1, 'filler.wav', allowed=lambda: True))
    await asyncio.wait_for(played.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    mixer.stop_speech.assert_called_once()


@pytest.mark.asyncio
async def test_stt_error_is_authorized_and_never_starts_a_turn(monkeypatch):
    from contextlib import nullcontext
    from tests.gateway.test_reasoning_command import _make_runner
    from gateway.session import SessionSource
    r = _make_runner()
    r._voice_mode = {}
    source = SessionSource(platform=Platform.DISCORD, chat_id='chat', user_id='speaker')
    adapter = SimpleNamespace(_voice_text_channels={1: 'chat'}, _owner_profile='default')
    monkeypatch.setattr(r, '_voice_input_source', lambda *a: source)
    monkeypatch.setattr(r, '_canonicalize', lambda *a, **k: object())
    monkeypatch.setattr(r, '_profile_scope_for_source', lambda *a: nullcontext())
    monkeypatch.setattr(r, '_is_user_authorized_for_source', lambda *a: True)
    speak = AsyncMock()
    monkeypatch.setattr(vp, 'speak_event_phrase', speak)
    await r._handle_voice_channel_input(1, 42, '', adapter=adapter, error='stt_error')
    speak.assert_awaited_once_with(r, source, 'stt_error', voice_input=True)
    speak.reset_mock()
    monkeypatch.setattr(r, '_is_user_authorized_for_source', lambda *a: False)
    await r._handle_voice_channel_input(1, 42, '', adapter=adapter, error='stt_error')
    speak.assert_not_awaited()



@pytest.mark.asyncio
async def test_fixed_phrase_never_becomes_attachment_after_disconnect():
    from tests.gateway.test_discord_voice_mixer import _make_adapter
    adapter = _make_adapter()
    adapter.send_voice = AsyncMock()
    result = await adapter.play_tts('chat', 'phrase.wav', allowed=lambda: True)
    assert not result.success
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_reminder_only_during_a_voice_turn(monkeypatch):
    spoken, waits = [], []
    async def speak(adapter, chat, text, **kwargs):
        spoken.append(text)
    async def sleep(delay):
        waits.append(delay)
    monkeypatch.setattr(vp, 'speak_phrase', speak)
    monkeypatch.setattr(vp.asyncio, 'sleep', sleep)
    p = profile()
    p['timing']['status_silence_seconds'] = 6
    turn = vp.VoicePhraseTurn(SimpleNamespace(), 'chat', p, lambda: True)
    await turn.start()
    await turn._task
    assert waits == [6]
    assert len(spoken) == 1  # greeting belongs to connection, not each turn
    assert spoken[0] in p['phrases']['fillers']
    await turn.close()


def test_phrase_bags_exhaust_categories_and_keep_turn_history():
    from gateway.voice_profile import PhrasePicker
    picker = PhrasePicker()
    for group, phrases in DEFAULT_PHRASES.items():
        first = [picker.choose(group, phrases) for _ in phrases]
        assert set(first) == set(phrases)
        assert len(first) == len(set(first))
        assert picker.choose(group, phrases) != first[-1]
    adapter = SimpleNamespace()
    assert vp.phrase_picker(adapter) is vp.phrase_picker(adapter)


def test_approval_stops_status_audio_before_delivering_prompt():
    from gateway.run_turn_runner import TurnRunner
    from unittest.mock import Mock
    controller = SimpleNamespace(answer_started=Mock())
    def pause(chat):
        controller.answer_started.assert_called_once()
        raise RuntimeError('delivery boundary reached')
    ctx = SimpleNamespace(voice_phrase_turn=controller, _status_chat_id='chat',
                          _status_adapter=SimpleNamespace(pause_typing_for_chat=pause))
    with pytest.raises(RuntimeError, match='delivery boundary reached'):
        TurnRunner(None, ctx)._approval_notify_sync({})
