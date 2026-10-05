"""Tray comparisons preserve local branches and distinguish remote failures."""
from hermes_cli.tray_status import read_status


def test_local_only_branch_compares_main_and_keeps_installed_version(tmp_path, monkeypatch):
    from hermes_cli import source_check, _startup_fast
    calls = []
    def check(**kwargs):
        calls.append(kwargs)
        if 'branch' not in kwargs:
            return {'supported': True, 'error': 'branch-local-only', 'currentBranch': 'local-fixes'}
        return {'supported': True, 'branch': 'main', 'updateAvailable': True, 'behind': 3}
    monkeypatch.setattr(source_check, 'check_for_updates', check)
    monkeypatch.setattr(_startup_fast, 'print_fast_version_info', lambda **kw: print('Hermes Agent v1.2.3.local'))
    result = read_status(tmp_path, force=True)
    assert calls == [{'install_root': tmp_path, 'force': True},
                     {'install_root': tmp_path, 'branch': 'main', 'force': True}]
    assert result['localBranch'] == 'local-fixes' and result['comparisonBranch'] == 'main'
    assert result['updateAvailable'] is True and result['behind'] == 3
    assert result['error'] is None and 'v1.2.3.local' in result['versionText']


def test_network_failure_does_not_change_target_or_claim_current(tmp_path, monkeypatch):
    from hermes_cli import source_check, _startup_fast
    calls = []
    def check(**kwargs):
        calls.append(kwargs)
        return {'supported': True, 'error': 'fetch-failed', 'branch': 'feature/test'}
    monkeypatch.setattr(source_check, 'check_for_updates', check)
    monkeypatch.setattr(_startup_fast, 'print_fast_version_info', lambda **kw: print('Hermes Agent v1.2.3'))
    result = read_status(tmp_path)
    assert len(calls) == 1 and 'branch' not in calls[0]
    assert result['error'] == 'fetch-failed' and result['updateAvailable'] is None
    assert result['localBranch'] is None
