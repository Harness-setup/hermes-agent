"""Desktop chained voice and live delegations share an ephemeral reasoning policy."""

from types import SimpleNamespace

import pytest

from tui_gateway.voice_reasoning import voice_turn_reasoning


@pytest.mark.parametrize('override', [None, {'enabled': True, 'effort': 'medium'},
                                     {'enabled': True, 'effort': 'high'}, {'enabled': False}])
def test_voice_reasoning_restores_cached_agent_even_after_failure(override):
    normal = {'enabled': True, 'effort': 'medium'}
    agent = SimpleNamespace(reasoning_config=normal)
    text_agent = SimpleNamespace(reasoning_config=normal)
    session = {} if override is None else {'create_reasoning_override': override}
    with pytest.raises(RuntimeError):
        with voice_turn_reasoning(session, agent, True):
            assert agent.reasoning_config == (override if override is not None else {'enabled': False})
            with voice_turn_reasoning({}, text_agent, False):
                assert text_agent.reasoning_config == normal
            raise RuntimeError('provider failed')
    assert agent.reasoning_config == (override if override is not None else normal)
    with voice_turn_reasoning(session, agent, False):
        assert agent.reasoning_config == (override if override is not None else normal)


def test_reasoning_control_used_during_voice_turn_survives():
    agent = SimpleNamespace(reasoning_config={'enabled': True, 'effort': 'medium'})
    session = {}
    with voice_turn_reasoning(session, agent, True):
        chosen = {'enabled': True, 'effort': 'high'}
        session['create_reasoning_override'] = chosen
        agent.reasoning_config = chosen
    assert agent.reasoning_config == chosen


def test_queued_voice_and_text_keep_separate_turn_policies():
    from tui_gateway import server
    session = {'queued_prompt': None}
    voice = server._enqueue_prompt(session, 'spoken request', None, voice_input=True)
    text = server._enqueue_prompt(session, 'typed request', None)
    assert voice['text'] == 'spoken request'
    assert voice['voice_input'] is True
    assert text['text'] == 'typed request'
    assert not text.get('voice_input')
    assert session['queued_prompts'] == [text]


@pytest.mark.parametrize('override', [None, {'enabled': True, 'effort': 'medium'}])
def test_desktop_turn_applies_voice_reasoning_to_real_agent_call(monkeypatch, tmp_path, override):
    from tests.tui_gateway.test_prompt_accept_logging import _InlineThread, _session, turn_stubs
    from tui_gateway import server
    turn_stubs.__wrapped__(monkeypatch, tmp_path)
    monkeypatch.setattr(server.threading, 'Thread', _InlineThread)
    normal = {'enabled': True, 'effort': 'medium'}
    seen = []
    agent = SimpleNamespace(session_id='voice-session', reasoning_config=normal,
                            clear_interrupt=lambda: None)

    def converse(*args, **kwargs):
        seen.append(agent.reasoning_config)
        return {'final_response': 'done'}

    agent.run_conversation = converse
    session = _session(agent=agent, running=True)
    if override is not None:
        session['create_reasoning_override'] = override
    assert server._run_prompt_submit('voice', 'ui-sid', session, 'spoken', voice_input=True)
    assert seen == [override if override is not None else {'enabled': False}]
    assert agent.reasoning_config == normal
    session['running'] = True
    assert server._run_prompt_submit('text', 'ui-sid', session, 'typed')
    assert seen[-1] == normal


@pytest.mark.parametrize('surface', ['voice-chat', 'voice-live', None])
def test_rpc_voice_marker_travels_with_busy_request(monkeypatch, surface):
    from tests.tui_gateway.test_hud_surface_note import _session
    from tui_gateway import server
    session = _session(running=True)
    monkeypatch.setitem(server._sessions, 'voice-rpc', session)
    result = server._methods['prompt.submit']('r1', {
        'session_id': 'voice-rpc', 'text': 'spoken request', 'surface': surface, 'queued': True})
    assert result['result']['status'] == 'queued'
    assert bool(session['queued_prompt'].get('voice_input')) == (surface is not None)


def test_compute_host_frame_retains_voice_marker():
    from tests.tui_gateway.test_compute_host_pending_model_switch import _session
    from tui_gateway import server
    session = _session()
    assert server._compute_host_turn_frame('r', 'sid', session, 'spoken', voice_input=True)['voice_input'] is True
    assert server._compute_host_turn_frame('r', 'sid', session, 'typed')['voice_input'] is False
