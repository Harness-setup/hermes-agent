"""Tests for parked-MCP-server log dedupe (#115713).

A parked server's timed self-probe wakes the run task, which retries the
transport and — when the dependency is still down — re-parks. Every re-park
used to emit the same WARNING as the first park, so a long-lived gateway
accumulated thousands of identical lines per month (10,548 for one server in
the report) and drowned the genuinely-new errors in gateway.error.log. The
park lines now go through ``MCPServerTask._log_park``: the first park (a real
state transition) warns; re-parks while the server never revived are demoted
to DEBUG. A session that proves healthy clears ``_was_parked``, so the next
outage warns again.
"""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from tools.mcp_tool import MCPServerTask


def _fast_time(monkeypatch, tmp_path):
    """Zero out the retry sleeps and the parked self-probe interval; keep ``blender``
    enabled in the temp home's config.yaml so the self-probe gate lets it wake."""
    from tools import mcp_tool

    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  blender:\n    command: x\n", encoding="utf-8")
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0)
    real_sleep = asyncio.sleep

    async def _sleep(_delay, *a, **kw):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.mark.no_isolate
def test_reparked_server_logs_once_not_per_probe(monkeypatch, tmp_path, caplog):
    """First park warns; every failed self-probe that re-parks the same dead
    server logs at DEBUG only — the parked state is already visible via
    ``hermes mcp list`` (#115713)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    _fast_time(monkeypatch, tmp_path)

    state = {"calls": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["calls"] += 1
                if state["calls"] >= 10:
                    self._shutdown_event.set()
                raise ConnectionError("addon not running")

        task = _Task("blender")
        task._registered_tool_names = []
        run_task = asyncio.ensure_future(task.run({"command": "x"}))
        await asyncio.wait_for(run_task, timeout=15)

    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        asyncio.run(_scenario())

    parks = [
        r for r in caplog.records if "failed initial connection after" in r.getMessage()
    ]
    assert len(parks) >= 4, (
        f"scenario only produced {len(parks)} park lines — expected several "
        "self-probe re-park cycles"
    )
    assert sum(1 for r in parks if r.levelno >= logging.WARNING) == 1, (
        "the first park must warn exactly once; the re-parks must be DEBUG"
    )
    assert sum(1 for r in parks if r.levelno == logging.DEBUG) >= 3


@pytest.mark.no_isolate
def test_park_after_revival_warns_again(monkeypatch, tmp_path, caplog):
    """A server that revived (session proven → ``_was_parked`` cleared) and then
    fails again must warn on its next park: a fresh outage is a state
    transition, not probe chatter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    _fast_time(monkeypatch, tmp_path)

    state = {"calls": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["calls"] += 1
                if state["calls"] == 3:
                    # The revival probe succeeds: session proves healthy and
                    # clears the parked latch.
                    self._ever_connected = True
                    self.session = object()
                    self._ready.set()
                    self._mark_session_proven()
                    self.session = None
                    return "reconnect"
                if state["calls"] >= 7:
                    self._shutdown_event.set()
                raise ConnectionError("addon not running")

        task = _Task("blender")
        task._registered_tool_names = []
        run_task = asyncio.ensure_future(task.run({"command": "x"}))
        await asyncio.wait_for(run_task, timeout=15)

    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        asyncio.run(_scenario())

    initial_parks = [
        r for r in caplog.records if "failed initial connection after" in r.getMessage()
    ]
    reconnect_parks = [
        r for r in caplog.records if "reconnection attempts" in r.getMessage()
    ]
    assert [r.levelno >= logging.WARNING for r in initial_parks] == [True]
    # First park of the new outage warns; the re-park after it is DEBUG.
    assert [r.levelno >= logging.WARNING for r in reconnect_parks] == [True, False]


def test_park_for_a_different_reason_warns_again(caplog):
    """The dedupe is keyed on the park line, not on \"still parked\": a server parked on a
    connection error that re-parks on an auth error carries new information and warns again;
    only an identical repeat is demoted to DEBUG."""
    task = MCPServerTask("t")
    with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
        task._log_park("MCP server '%s' parked: %s", "t", "ConnectionError: refused")
        task._was_parked = True  # _park() latches this on the first park
        task._log_park("MCP server '%s' parked: %s", "t", "ConnectionError: refused")
        task._log_park("MCP server '%s' parked: %s", "t", "OAuthError: token revoked")
        task._log_park("MCP server '%s' parked: %s", "t", "OAuthError: token revoked")
    assert [r.levelno for r in caplog.records] == [
        logging.WARNING, logging.DEBUG, logging.WARNING, logging.DEBUG]


class TestAuthParkNotification:
    """Tony, 2026-09-23: "keeps on asking me to reauthenticate for todoist mcp" — an auth park
    was previously WARNING-logged only, so he never found out until he happened to hit that
    server and got a vague failure, sometimes days later. A NEW auth-related park now also
    sends a one-shot direct notification (``_log_park(..., is_auth=True)``)."""

    def test_new_auth_park_sends_one_notification(self, caplog):
        task = MCPServerTask("todoist")
        with patch("tools.send_message_tool.send_message_tool",
                   return_value=json.dumps({"success": True})) as mock_send:
            with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
                task._log_park("MCP server '%s' parked: %s", "todoist", "OAuthError: token revoked",
                               is_auth=True)
        mock_send.assert_called_once()
        sent_args = mock_send.call_args[0][0]
        assert sent_args["target"] == "discord"
        assert "todoist" in sent_args["message"]
        assert "hermes mcp login todoist" in sent_args["message"]

    def test_repeated_identical_auth_park_does_not_renotify(self, caplog):
        task = MCPServerTask("todoist")
        with patch("tools.send_message_tool.send_message_tool",
                   return_value=json.dumps({"success": True})) as mock_send:
            with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
                task._log_park("MCP server '%s' parked: %s", "todoist", "OAuthError: token revoked",
                               is_auth=True)
                task._was_parked = True  # _park() latches this on the first park
                task._log_park("MCP server '%s' parked: %s", "todoist", "OAuthError: token revoked",
                               is_auth=True)
        mock_send.assert_called_once()

    def test_non_auth_park_never_notifies(self, caplog):
        task = MCPServerTask("blender")
        with patch("tools.send_message_tool.send_message_tool") as mock_send:
            with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
                task._log_park("MCP server '%s' parked: %s", "blender", "ConnectionError: refused",
                               is_auth=False)
        mock_send.assert_not_called()

    def test_notification_failure_does_not_raise_or_block_the_park(self, caplog):
        task = MCPServerTask("todoist")
        with patch("tools.send_message_tool.send_message_tool", side_effect=RuntimeError("no channel")):
            with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
                task._log_park("MCP server '%s' parked: %s", "todoist", "OAuthError: token revoked",
                               is_auth=True)
        parks = [r for r in caplog.records if "parked" in r.getMessage()]
        assert any(r.levelno >= logging.WARNING for r in parks)
