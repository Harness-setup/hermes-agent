"""Tests for PebbleRelayBackend -- the WSL gateway -> real Windows desktop bridge via Pebble's
control-server.js. Mocks urllib.request.urlopen (no real network/Pebble needed); the actual
end-to-end wiring was verified live tonight (list_apps through this exact relay path returned
305 real Windows apps against the real daemon -- see this session's commit history)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use.pebble_relay_backend import (
    PebbleRelayBackend, PebbleRelayError, relay_call,
)


def _fake_response(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestRelayCall:
    def test_success_returns_the_result_payload(self):
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {"apps": [{"name": "Notion"}]}})):
            result = relay_call("list_apps", {})
        assert result == {"apps": [{"name": "Notion"}]}

    def test_relay_level_failure_raises(self):
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": False, "error": "tool 'evil' is not in the computer-use relay allowlist"})):
            with pytest.raises(PebbleRelayError, match="not in the computer-use relay allowlist"):
                relay_call("evil", {})

    def test_pebble_unreachable_raises_relay_error_not_a_raw_url_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(PebbleRelayError, match="could not reach Pebble"):
                relay_call("list_apps", {})


class TestPebbleRelayBackend:
    def test_is_available_true_on_healthy_pebble(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True})):
            assert backend.is_available() is True

    def test_is_available_false_when_pebble_unreachable(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert backend.is_available() is False

    def test_start_raises_when_pebble_unreachable(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with pytest.raises(RuntimeError, match="Pebble is not reachable"):
                backend.start()

    def test_launch_app_relays_the_app_name_and_reports_success(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {"targetId": "abc"}})) as mock_open:
            result = backend.launch_app("Notion")
        assert result.ok is True
        assert result.action == "launch_app"
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "launch_app", "args": {"app": "Notion"}}

    def test_launch_app_relay_failure_is_a_refusal_not_an_exception(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = backend.launch_app("Notion")
        assert result.ok is False
        assert "Notion" not in result.message or "reach Pebble" in result.message

    def test_focus_app_delegates_to_launch_app_idempotently(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {"targetId": "abc"}})):
            result = backend.focus_app("Notion")
        assert result.ok is True
        assert result.action == "focus_app"

    def test_list_apps_returns_the_real_apps_list(self):
        backend = PebbleRelayBackend()
        apps = [{"name": "Notion", "pid": 123}, {"name": "Microsoft Edge", "pid": 456}]
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {"apps": apps}})):
            result = backend.list_apps()
        assert result == apps

    def test_list_apps_degrades_to_empty_list_on_relay_failure(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert backend.list_apps() == []

    def test_list_windows_raises_instead_of_silently_reporting_no_windows(self):
        # Regression for session 20260923_132613_72f606: list_windows used to inherit
        # ComputerUseBackend's default `return []`, so a real prompt calling it got a clean
        # "no windows found" that was actually just an unimplemented relay method.
        backend = PebbleRelayBackend()
        with pytest.raises(NotImplementedError, match="list_windows"):
            backend.list_windows()

    def test_click_requires_x_y_in_v1(self):
        backend = PebbleRelayBackend()
        result = backend.click(element=3)
        assert result.ok is False
        assert "x/y" in result.message

    def test_click_relays_x_y_button(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {}})) as mock_open:
            result = backend.click(x=10, y=20, button="right")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "click", "args": {"x": 10, "y": 20, "button": "right"}}

    def test_click_double_click_uses_the_double_click_tool(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            backend.click(x=10, y=20, click_count=2)
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["tool"] == "double_click"

    def test_type_text_relays_the_text(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            result = backend.type_text("hello jarvis")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "type_text", "args": {"text": "hello jarvis"}}

    def test_key_relays_the_key_combo(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            result = backend.key("ctrl+s")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "press_key", "args": {"key": "ctrl+s"}}

    def test_capture_returns_the_real_screenshot_b64(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {"width": 1920, "height": 1080, "screenshot_png_b64": "abc123"}})):
            result = backend.capture()
        assert result.width == 1920
        assert result.height == 1080
        assert result.png_b64 == "abc123"

    def test_capture_degrades_gracefully_on_relay_failure(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = backend.capture()
        assert result.width == 0
        assert "capture failed" in result.note

    @pytest.mark.parametrize("method,kwargs", [
        ("drag", {"from_xy": (0, 0), "to_xy": (1, 1)}),
        ("scroll", {"direction": "down"}),
        ("set_value", {"value": "x"}),
    ])
    def test_unimplemented_v1_actions_refuse_clearly_not_crash(self, method, kwargs):
        backend = PebbleRelayBackend()
        result = getattr(backend, method)(**kwargs)
        assert result.ok is False
        assert result.code == "pebble_relay_not_implemented"

    def test_wait_uses_the_shared_default_implementation_no_relay_needed(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen") as mock_open:
            result = backend.wait(0.01)
        mock_open.assert_not_called()
        assert result.ok is True


class TestNewBackendWiring:
    """HERMES_COMPUTER_USE_BACKEND=pebble must actually select this class (tools/computer_use/
    tool.py:_new_backend) -- the whole point of the relay is reachability from the live gateway,
    so a typo'd or dropped wire-up here would silently fall through to the local cua-driver."""

    def test_backend_env_var_pebble_selects_pebble_relay_backend(self, monkeypatch):
        from tools.computer_use import tool as cu_tool

        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "pebble")
        backend = cu_tool._new_backend("standard")
        assert isinstance(backend, PebbleRelayBackend)
