"""Backend-wide Stop All cancels work, preserves sessions and retires restart recovery."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock


def session(home, key, running=True):
    return {
        'profile_home': str(home), 'session_key': key, 'running': running,
        'queued_prompt': {'text': 'queued'}, 'history_lock': threading.Lock(),
        'agent': SimpleNamespace(hard_interrupt=Mock(), session_id=key), '_run_thread': None,
        '_active_turn_marker_key': key + '-before-compression',
    }


def test_interrupt_all_preserves_sessions_and_clears_markers_in_each_profile(tmp_path, monkeypatch):
    from tui_gateway import server
    from tui_gateway.turn_marker import record_turn_start, read_turn_marker
    from hermes_constants import get_hermes_home, get_hermes_home_override
    from agent.secret_scope import set_multiplex_active

    launch = tmp_path / 'home'
    secondary = launch / 'profiles' / 'secondary'
    secondary.mkdir(parents=True)
    for home in (launch, secondary):
        (home / 'config.yaml').write_text('{}')
        (home / '.env').write_text('')
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setattr(server, '_hermes_home', launch)
    sessions = {sid: session(home, sid) for sid, home in [('a', launch), ('b', secondary), ('a2', launch)]}
    original_agents = [s['agent'] for s in sessions.values()]
    for s in sessions.values():
        for key in (s['session_key'], s['_active_turn_marker_key']):
            record_turn_start(s['profile_home'], key, 'work')
    monkeypatch.setattr(server, '_sessions', sessions)
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda _s: False)
    monkeypatch.setattr(server, '_tts_stream_stop', lambda: None)
    seen = []
    monkeypatch.setattr('hermes_cli.plugins.invoke_hook', lambda *_a, **_kw: seen.append(get_hermes_home()))
    set_multiplex_active(True)
    try:
        response = server.handle_request({'jsonrpc': '2.0', 'id': 1, 'method': 'session.interrupt_all', 'params': {}})
    finally:
        set_multiplex_active(False)
    assert all(r['ok'] for r in response['result']['sessions'])
    assert seen == [launch, secondary, launch]
    assert get_hermes_home_override() is None
    assert [s['agent'] for s in sessions.values()] == original_agents
    for s in sessions.values():
        s['agent'].hard_interrupt.assert_called_once()
        assert s['queued_prompt'] is None and s['_turn_cancel_requested'] is True
        for key in (s['session_key'], s['session_key'] + '-before-compression'):
            assert read_turn_marker(s['profile_home'], key) is None


def test_failed_compute_host_does_not_skip_other_sessions_or_report_success(tmp_path, monkeypatch):
    from tui_gateway import server
    bad = session(tmp_path, 'bad')
    good = session(tmp_path, 'good')
    monkeypatch.setattr(server, '_sessions', {'bad': bad, 'good': good})
    monkeypatch.setattr(server, '_tts_stream_stop', lambda: None)
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda s: s is bad)
    supervisor = SimpleNamespace(interrupt=Mock(side_effect=RuntimeError('host unavailable')))
    monkeypatch.setattr(server, '_get_compute_host_supervisor', lambda: supervisor)
    response = server.handle_request({'jsonrpc': '2.0', 'id': 2, 'method': 'session.interrupt_all', 'params': {}})
    results = response['result']['sessions']
    assert [(r['session_id'], r['ok']) for r in results] == [('bad', False), ('good', True)]
    assert 'host unavailable' in results[0]['error']
    good['agent'].hard_interrupt.assert_called_once()
    assert good['queued_prompt'] is None
