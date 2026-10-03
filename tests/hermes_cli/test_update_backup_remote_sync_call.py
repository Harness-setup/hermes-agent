"""``hermes update`` on a parked branch syncs from the backup remote before the branch guard.

Regression: the call was routed through ``hermes_cli.main`` (``_m()``), whose frozen updater
surface does not list this local helper, so every ``hermes update`` crashed with
``AttributeError: module 'hermes_cli.main' has no attribute '_maybe_sync_from_backup_remote'``.
"""
import types

import pytest

import hermes_cli.update_cmd as update_cmd


class _Stop(Exception):
    pass


def test_prepare_checkout_calls_backup_remote_sync_without_going_through_main(monkeypatch):
    calls = []
    # hermes_cli.main's frozen updater surface does not carry this helper; model that exactly.
    monkeypatch.setattr(update_cmd, "_m", lambda: types.SimpleNamespace(PROJECT_ROOT="."))
    monkeypatch.setattr(update_cmd, "_maybe_sync_from_backup_remote", lambda *a: calls.append(a))

    def _stop(*_a, **_kw):
        raise _Stop

    monkeypatch.setattr(update_cmd, "_apply_parked_branch_guard", _stop)

    with pytest.raises(_Stop):
        update_cmd._prepare_checkout_for_update(
            ["git"], "local-fixes", "local-fixes", is_fork=False, assume_yes=True,
            gateway_mode=False, gw_input_fn=None, switch_branch=False, target_ref=None,
            _windows_gateway_resume=None)

    assert len(calls) == 1 and calls[0][2] == "local-fixes"
