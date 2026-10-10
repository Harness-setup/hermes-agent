"""The shared voice profile is authenticated, scoped and contains no synthesis credentials."""
import json


def test_profile_auth_revision_and_secrets(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    from gateway.voice_profile import resolve_voice_profile
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: tmp_path / 'profiles')
    path = tmp_path / 'config.yaml'
    path.write_text('tts:\n  provider: openai\n  openai:\n    voice: alloy\n    api_key: PRIVATE\nvoice:\n  profile:\n    phrases:\n      stop: ["All stopped."]\n')
    client = TestClient(app)
    assert client.get('/api/audio/voice-profile').status_code == 401
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    response = client.get('/api/audio/voice-profile', headers=headers)
    assert response.status_code == 200
    profile = response.json()
    assert profile == resolve_voice_profile()
    assert profile['tts']['voice'] == 'alloy'
    assert profile['phrases']['stop'] == ['All stopped.']
    assert 'PRIVATE' not in json.dumps(profile)
    assert 'api_key' not in json.dumps(profile)
    assert client.get('/api/audio/voice-profile?profile=missing-profile', headers=headers).status_code == 404
    path.write_text(path.read_text().replace('alloy', 'echo'))
    changed = client.get('/api/audio/voice-profile', headers=headers).json()
    assert changed['revision'] != profile['revision']
    path.write_text(path.read_text().replace('All stopped.', 'Done, sir.'))
    assert client.get('/api/audio/voice-profile', headers=headers).json()['revision'] != changed['revision']



def test_profile_endpoint_uses_requested_home(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    from hermes_cli import profiles
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: tmp_path)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: tmp_path / 'profiles')
    worker = tmp_path / 'profiles' / 'voice-worker'
    worker.mkdir(parents=True)
    (tmp_path / 'config.yaml').write_text('tts:\n  provider: openai\n  openai:\n    voice: alloy\n')
    (worker / 'config.yaml').write_text('tts:\n  provider: openai\n  openai:\n    voice: echo\n')
    client = TestClient(app)
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    primary = client.get('/api/audio/voice-profile', headers=headers).json()
    secondary = client.get('/api/audio/voice-profile?profile=voice-worker', headers=headers).json()
    assert primary['tts']['voice'] == 'alloy'
    assert secondary['tts']['voice'] == 'echo'
    assert primary['revision'] != secondary['revision']
