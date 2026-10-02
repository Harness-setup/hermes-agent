"""SIGUSR2 = Pebble's Stop All: interrupt every running agent turn, keep the gateway up."""
from __future__ import annotations

import logging
import signal

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig


class _Runner:
    def __init__(self, config):
        self.config = config
        self.adapters = {}
        self._running = False
        self._restart_requested = False
        self._restart_via_service = False
        self.should_exit_cleanly = True
        self.should_exit_with_failure = False
        self.exit_reason = None
        self.exit_code = None
        self.interrupt_reasons = []
        self.restart_calls = []

    async def start(self):
        return False

    async def wait_for_shutdown(self):
        return None

    def request_restart(self, *, detached=False, via_service=False):
        self.restart_calls.append(1)
        return True

    def _interrupt_running_agents(self, reason):
        self.interrupt_reasons.append(reason)


def test_handler_interrupts_agents_with_a_stop_all_reason(caplog):
    runner = _Runner(GatewayConfig())
    handler = gateway_run._start_gateway_make_interrupt_signal_handler(runner)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        handler()
    assert runner.interrupt_reasons == ["user_stop_all"]
    assert runner.restart_calls == []
    assert "SIGUSR2 received" in " ".join(r.getMessage() for r in caplog.records)


def test_handler_survives_an_interrupt_failure():
    runner = _Runner(GatewayConfig())

    def boom(reason):
        raise RuntimeError("agent gone")

    runner._interrupt_running_agents = boom
    gateway_run._start_gateway_make_interrupt_signal_handler(runner)()  # must not raise


@pytest.mark.asyncio
async def test_sigusr2_handler_is_installed_by_start_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for target, value in (
        ("gateway.status.get_running_pid", lambda: None),
        ("gateway.status.acquire_gateway_runtime_lock", lambda: True),
        ("gateway.status.write_pid_file", lambda: None),
        ("gateway.status.remove_pid_file", lambda: None),
        ("gateway.status.release_gateway_runtime_lock", lambda: None),
        ("tools.skills_sync.sync_skills", lambda quiet=True: None),
        ("hermes_logging.setup_logging", lambda hermes_home, mode: None),
        ("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda: None),
    ):
        monkeypatch.setattr(target, value)
    runners = []

    def make_runner(config):
        runners.append(_Runner(config))
        return runners[-1]

    monkeypatch.setattr(gateway_run, "GatewayRunner", make_runner)
    import asyncio
    loop = asyncio.get_running_loop()
    installed = {}
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, handler, *args: installed.update({sig: (handler, args)}))

    await gateway_run.start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    handler, args = installed[signal.SIGUSR2]
    handler(*args)
    assert runners[0].interrupt_reasons == ["user_stop_all"]
    assert signal.SIGUSR1 in installed  # restart handler untouched
