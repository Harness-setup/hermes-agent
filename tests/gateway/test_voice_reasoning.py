"""Live voice defaults are turn-local and never replace an explicit session choice."""

import pytest

from gateway.run import GatewayRunner


@pytest.mark.parametrize('effort', ['medium', 'high', 'none'])
def test_voice_reasoning_session_precedence_and_text_restore(monkeypatch, effort):
    runner = object.__new__(GatewayRunner)
    normal = {'enabled': True, 'effort': 'medium'}
    monkeypatch.setattr(runner, '_load_reasoning_config', lambda model='': dict(normal))
    assert runner._resolve_session_reasoning_config(session_key='voice', voice_input=True) == {'enabled': False}
    assert runner._resolve_session_reasoning_config(session_key='text') == normal
    assert runner._resolve_session_reasoning_config(session_key='voice') == normal
    from hermes_constants import parse_reasoning_effort
    selected = parse_reasoning_effort(effort)
    runner._set_session_reasoning_override('voice', selected)
    assert runner._resolve_session_reasoning_config(session_key='voice', voice_input=True) == selected
    assert runner._resolve_session_reasoning_config(session_key='text', voice_input=True) == {'enabled': False}
    runner._set_session_reasoning_override('voice', None)
    assert runner._resolve_session_reasoning_config(session_key='voice', voice_input=True) == {'enabled': False}


@pytest.mark.parametrize('mode', ['all', 'off'])
@pytest.mark.parametrize('voice', [True, False])
@pytest.mark.parametrize('cached', [True, False])
def test_discord_agent_receives_turn_reasoning(monkeypatch, voice, cached, mode):
    import asyncio
    import sys
    import types
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource
    from tests.gateway.test_reasoning_command import _make_runner, _CapturingAgent

    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'agent': {'reasoning_effort': 'medium'}, 'streaming': {'enabled': False}})
    monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {
        'provider': 'custom', 'api_mode': 'chat_completions',
        'base_url': 'https://example.invalid/v1', 'api_key': 'test-key'})
    class CapturingAgent(_CapturingAgent):
        seen_reasoning = None

        def run_conversation(self, *args, **kwargs):
            type(self).seen_reasoning = self.reasoning_config
            type(self).seen_notes = self._gateway_turn_context_notes
            type(self).seen_voice_input = self._voice_input
            return {'final_response': 'ok', 'messages': [], 'api_calls': 1}

    module = types.ModuleType('run_agent')
    module.AIAgent = CapturingAgent
    if cached:
        from gateway.run_turn_runner import TurnRunner
        agent = CapturingAgent(reasoning_config={'enabled': True, 'effort': 'medium'})
        monkeypatch.setattr(TurnRunner, '_resolve_turn_agent', lambda *a: (agent, True))
    monkeypatch.setitem(sys.modules, 'run_agent', module)
    runner = _make_runner()
    runner._voice_mode = {}
    source = SessionSource(platform=Platform.DISCORD, chat_id='voice-chat', user_id='speaker')
    runner._voice_mode[runner._voice_key_for_source(source)] = mode
    result = asyncio.run(runner._run_agent(
        message='ping', context_prompt='', history=[], source=source,
        session_id='voice-turn', session_key='agent:main:discord:voice',
        message_type='voice' if voice else 'text'))
    assert result['final_response'] == 'ok'
    assert CapturingAgent.seen_reasoning == (
        {'enabled': False} if voice else {'enabled': True, 'effort': 'medium'})
    assert CapturingAgent.seen_voice_input is voice
    spoken = voice or mode == 'all'
    assert ('Voice conversation:' in CapturingAgent.seen_notes) is spoken
    assert ('Discord text conversation:' in CapturingAgent.seen_notes) is not spoken


def test_proxy_sends_voice_default_and_session_opt_in():
    runner = object.__new__(GatewayRunner)
    assert runner._proxy_turn_reasoning(None, 'voice', True) == {'reasoning': {'enabled': False}}
    assert runner._proxy_turn_reasoning(None, 'voice', False) == {}
    selected = {'enabled': True, 'effort': 'medium'}
    runner._set_session_reasoning_override('voice', selected)
    assert runner._proxy_turn_reasoning(None, 'voice', True) == {'reasoning': selected}
