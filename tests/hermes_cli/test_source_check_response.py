"""Large GitHub comparisons must parse completely or fail explicitly."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@contextmanager
def response_server(payload):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/compare'
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_large_comparison_retains_count_and_commit_messages():
    from hermes_cli.source_check import _request_with
    expected = {'ahead_by': 517, 'commits': [{'sha': 'a' * 40, 'commit': {'message': 'Fix tray'}}],
                'files': [{'patch': 'x' * (3 * 1024 * 1024)}]}
    with response_server(json.dumps(expected).encode()) as url:
        payload = json.loads(_request_with(url, 'application/vnd.github+json', None))
    assert payload['ahead_by'] == 517
    assert payload['commits'][0]['commit']['message'] == 'Fix tray'
    assert payload['files'] == expected['files']


def test_oversized_response_fails_explicitly_instead_of_returning_partial_json():
    from hermes_cli.source_check import _request_with
    with response_server(b'x' * (16 * 1024 * 1024 + 2)) as url:
        with pytest.raises(ValueError, match='exceeds.*limit'):
            _request_with(url, 'application/vnd.github+json', None)


def test_carried_upstream_merge_does_not_recount_commits_from_stale_tracking_ref(tmp_path, monkeypatch):
    import subprocess
    from hermes_cli import source_check as s
    def git(*args):
        return subprocess.check_output(['git', '-c', 'user.name=Fixture', '-c', 'user.email=test@example.invalid',
                                        '-c', 'commit.gpgsign=false', *args], cwd=tmp_path, text=True).strip()
    git('init', '-b', 'main')
    git('commit', '--allow-empty', '-m', 'Old upstream')
    old = git('rev-parse', 'HEAD')
    git('branch', 'local-fixes')
    git('commit', '--allow-empty', '-m', 'New upstream already included')
    newer = git('rev-parse', 'HEAD')
    git('checkout', 'local-fixes')
    git('commit', '--allow-empty', '-m', 'Private fixes')
    git('merge', '--no-ff', 'main', '-m', 'Carry upstream')
    git('update-ref', 'refs/remotes/origin/main', old)
    target = 'f' * 40
    calls = []
    def compare(base, _target, _repository):
        calls.append(base)
        if base == newer:
            return {'ahead_by': 3, 'commits': []}
        if base == old:
            return {'ahead_by': 4, 'commits': []}
        return None
    monkeypatch.setattr(s, '_github_compare', compare)
    co = s._Checkout(tmp_path, 'git', None, git('rev-parse', 'HEAD'), 'local-fixes', '', 'fixture/repo', False)
    assert s._behind_count(co, target) == (3, [])
    assert newer in calls and old not in calls
