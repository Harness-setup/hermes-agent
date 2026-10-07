"""Display acknowledgment is confined to attached clients and outstanding session requests."""
from tools import human_input_hooks as hooks
from tui_gateway import server, server_requests as sr


def test_ack_requires_the_owning_live_session_transport(monkeypatch):
    events = []
    monkeypatch.setattr(hooks, '_fire', lambda name, data: events.append((name, data)))
    owner, stranger = object(), object()
    monkeypatch.setitem(server._sessions, 'owner', {'transport': owner, 'session_key': 'owner-key'})
    monkeypatch.setitem(server._sessions, 'other', {'transport': stranger, 'session_key': 'other-key'})
    monkeypatch.setattr(sr, '_write', lambda frame: None)
    monkeypatch.setattr(sr, '_emit', lambda *args: None)
    monkeypatch.setattr(sr, '_answerable', lambda sid: True)
    sr.reset_for_tests()
    settle = sr.send_async('sudo', 'owner', {}, lambda result: None)
    rid = sr.open_requests('owner')[0]['id']
    ack = server._methods['request.shown']
    def invoke(transport, sid):
        token = server.bind_transport(transport)
        try:
            return ack(1, {'session_id': sid, 'request_id': rid})
        finally:
            server.reset_transport(token)
    assert 'error' in invoke(stranger, 'owner')
    assert invoke(stranger, 'other')['result']['acknowledged'] is False
    assert not any(n == 'on_human_input_shown' for n, _ in events)
    assert invoke(owner, 'owner')['result']['acknowledged'] is True
    assert invoke(owner, 'owner')['result']['acknowledged'] is True
    assert len([n for n, _ in events if n == 'on_human_input_shown']) == 1
    settle('session_closed')
    assert invoke(owner, 'owner')['result']['acknowledged'] is False
    assert len([n for n, _ in events if n == 'on_human_input_resolved']) == 1


def test_last_client_disconnect_resolves_displayed_inputs(monkeypatch):
    events = []
    monkeypatch.setattr(hooks, '_fire', lambda name, data: events.append((name, data)))
    owner = object()
    monkeypatch.setitem(server._sessions, 'display-disconnect', {'transport': owner, 'session_key': 'display-key'})
    monkeypatch.setattr(sr, '_write', lambda frame: None)
    monkeypatch.setattr(sr, '_emit', lambda *args: None)
    monkeypatch.setattr(sr, '_answerable', lambda sid: True)
    sr.reset_for_tests()
    sr.send_async('sudo', 'display-disconnect', {}, lambda result: None)
    rid = sr.open_requests('display-disconnect')[0]['id']
    assert sr.acknowledge_display('display-disconnect', rid)
    server._detach_transport_from_sessions(owner)
    assert sr.open_requests('display-disconnect') == []
    assert events[-1][0] == 'on_human_input_resolved'
    assert not sr.acknowledge_display('display-disconnect', rid)


def test_compute_host_confirms_only_its_open_prompt_and_cancels_displayed_waits(monkeypatch):
    import io
    import json
    import threading
    from tui_gateway.compute_host import ComputeHost
    events = []
    monkeypatch.setattr(hooks, '_fire', lambda name, data: events.append((name, data)))
    monkeypatch.setattr(sr, '_write', lambda frame: None)
    monkeypatch.setattr(sr, '_emit', lambda *args: None)
    monkeypatch.setattr(sr, '_answerable', lambda sid: True)
    monkeypatch.setitem(server._sessions, 'compute-input', {'history_lock': threading.Lock()})
    monkeypatch.setitem(server._sessions, 'compute-other', {'history_lock': threading.Lock()})
    sr.reset_for_tests()
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    sr.send_async('sudo', 'compute-input', {}, lambda result: None)
    rid = sr.open_requests('compute-input')[0]['id']
    def relay(sid, params):
        host._handle_respond({'sid': sid, 'request_id': 'relay-display', 'params': params})
        return json.loads(out.getvalue().splitlines()[-1])['response']['result']
    try:
        assert relay('compute-other', {'shown': rid}) == {'acknowledged': False}
        assert relay('compute-input', {'shown': rid}) == {'acknowledged': True}
        assert relay('compute-input', {'cancel_displayed': True}) == {'cancelled': 1}
        assert relay('compute-input', {'shown': rid}) == {'acknowledged': False}
        assert [n for n, _ in events] == ['on_human_input_request', 'on_human_input_shown', 'on_human_input_resolved']
    finally:
        sr.reset_for_tests()
        host.close()
