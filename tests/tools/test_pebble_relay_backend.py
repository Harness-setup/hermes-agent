"""Tests for PebbleRelayBackend -- the WSL gateway -> real Windows desktop bridge via Pebble's
control-server.js. Mocks urllib.request.urlopen (no real network/Pebble needed); the actual
end-to-end wiring was verified live tonight (list_apps through this exact relay path returned
305 real Windows apps against the real daemon), and the element_token click path was verified
directly against the real daemon via the CLI (see pebble_relay_backend.py's module docstring)
before being wired into this class."""
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


def _fake_responses(payloads):
    """A urlopen side_effect list: one fake response per relay call, in order."""
    return [_fake_response(p) for p in payloads]


def _captured_backend(pid=65852, window_id=21235952, elements=None):
    """A backend with a committed sticky target (pid, window_id) from a real capture() call --
    the precondition click()/type_text()/key() need. Uses exact pid/window_id so the setup is a
    single get_window_state relay call, not a list_windows discovery round trip (that's covered
    separately by TestResolveCaptureTarget)."""
    backend = PebbleRelayBackend()
    gws_result = {"screenshot_width": 100, "screenshot_height": 50, "screenshot_png_b64": "abc",
                 "app_name": "msedge.exe", "window_title": "t", "elements": elements or []}
    with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": gws_result})):
        backend.capture(pid=pid, window_id=window_id)
    return backend


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
        # Real cua-driver schema uses "name" (confirmed against CuaDriverBackend.launch_app,
        # tools/computer_use/cua_backend.py) -- "app" was a mismatch that made every real
        # relayed launch_app fail with "Provide one of: bundle_id, name, aumid, path,
        # launch_path, or urls", confirmed live 2026-09-28 via a direct call to Pebble's
        # /computer-use-execute endpoint.
        assert sent == {"tool": "launch_app", "args": {"name": "Notion"}}

    def test_launch_app_relay_failure_is_a_refusal_not_an_exception(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = backend.launch_app("Notion")
        assert result.ok is False
        assert "Notion" not in result.message or "reach Pebble" in result.message

    def test_focus_app_resolves_and_commits_the_real_window_as_sticky_target(self):
        # Tony, 2026-09-29: the previous version of focus_app just re-called launch_app and
        # never set the sticky target at all -- confirmed live, capture() right after a
        # successful focus_app("Clock") still fell back to a leftover Chrome window instead,
        # because nothing had ever recorded which window was actually Clock's.
        backend = PebbleRelayBackend()
        windows = {"windows": [
            {"pid": 111, "window_id": 222, "app_name": "Notion.exe", "z_index": 1},
            {"pid": 333, "window_id": 444, "app_name": "msedge.exe", "z_index": 5},
        ]}
        with patch("urllib.request.urlopen", side_effect=_fake_responses([
                {"ok": True, "result": {"targetId": "abc"}},  # launch_app
                {"ok": True, "result": windows},  # list_windows, inside _resolve_capture_target
        ])):
            result = backend.focus_app("Notion")
        assert result.ok is True
        assert result.action == "focus_app"
        assert backend._active_pid == 111
        assert backend._active_window_id == 222

    def test_focus_app_reports_failure_when_launched_but_no_window_matches_yet(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=_fake_responses([
                {"ok": True, "result": {"targetId": "abc"}},  # launch_app
                {"ok": True, "result": {"windows": []}},  # list_windows finds nothing
        ])):
            result = backend.focus_app("Notion")
        assert result.ok is False
        assert result.action == "focus_app"
        assert backend._active_pid is None

    def test_focus_app_does_not_resolve_a_window_when_launch_itself_failed(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": False, "error": "not found"})):
            result = backend.focus_app("Nonexistent")
        assert result.ok is False
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


class TestResolveCaptureTarget:
    """capture()'s window-discovery path: list_windows + app-name/z_index selection, used
    whenever the caller doesn't already know an exact (pid, window_id)."""

    def test_exact_pid_window_id_skips_discovery_entirely(self):
        backend = PebbleRelayBackend()
        gws_result = {"screenshot_width": 10, "screenshot_height": 10, "screenshot_png_b64": "x"}
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": gws_result})) as mock_open:
            backend.capture(pid=111, window_id=222)
        # Exactly one relay call (get_window_state) -- no list_windows round trip needed.
        assert mock_open.call_count == 1
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["tool"] == "get_window_state"
        assert sent["args"]["pid"] == 111 and sent["args"]["window_id"] == 222

    def test_discovers_window_via_list_windows_when_pid_and_window_id_omitted(self):
        backend = PebbleRelayBackend()
        windows = {"windows": [{"pid": 111, "window_id": 222, "app_name": "notion.exe", "z_index": 0}]}
        gws_result = {"screenshot_width": 10, "screenshot_height": 10, "screenshot_png_b64": "x"}
        with patch("urllib.request.urlopen", side_effect=_fake_responses(
                [{"ok": True, "result": windows}, {"ok": True, "result": gws_result}])) as mock_open:
            backend.capture()
        assert mock_open.call_count == 2
        first_sent = json.loads(mock_open.call_args_list[0][0][0].data)
        assert first_sent["tool"] == "list_windows"
        second_sent = json.loads(mock_open.call_args_list[1][0][0].data)
        assert second_sent["args"]["pid"] == 111 and second_sent["args"]["window_id"] == 222

    def test_matches_app_name_case_insensitively(self):
        backend = PebbleRelayBackend()
        windows = {"windows": [
            {"pid": 1, "window_id": 1, "app_name": "notion.exe", "z_index": 0},
            {"pid": 2, "window_id": 2, "app_name": "msedge.exe", "z_index": 1},
        ]}
        gws_result = {"screenshot_width": 10, "screenshot_height": 10}
        with patch("urllib.request.urlopen", side_effect=_fake_responses(
                [{"ok": True, "result": windows}, {"ok": True, "result": gws_result}])) as mock_open:
            backend.capture(app="NOTION")
        second_sent = json.loads(mock_open.call_args_list[1][0][0].data)
        assert second_sent["args"]["pid"] == 1

    def test_matches_uwp_apps_by_title_when_app_name_is_the_generic_host(self):
        # Tony, 2026-09-29: confirmed live -- Clock's real list_windows entry is
        # {"app_name": "ApplicationFrameHost.exe", "title": "Clock"}. Every UWP/Store app
        # (Clock, Calculator, Alarms & Clock, Photos, ...) runs hosted the same way, so an
        # app_name-only match can never find any of them.
        backend = PebbleRelayBackend()
        windows = {"windows": [
            {"pid": 1, "window_id": 1, "app_name": "ApplicationFrameHost.exe", "title": "Clock", "z_index": 0},
            {"pid": 2, "window_id": 2, "app_name": "msedge.exe", "title": "New Tab", "z_index": 5},
        ]}
        gws_result = {"screenshot_width": 10, "screenshot_height": 10}
        with patch("urllib.request.urlopen", side_effect=_fake_responses(
                [{"ok": True, "result": windows}, {"ok": True, "result": gws_result}])) as mock_open:
            backend.capture(app="Clock")
        second_sent = json.loads(mock_open.call_args_list[1][0][0].data)
        assert second_sent["args"]["pid"] == 1

    def test_picks_frontmost_by_max_z_index_when_no_app_filter(self):
        backend = PebbleRelayBackend()
        windows = {"windows": [
            {"pid": 1, "window_id": 1, "app_name": "a.exe", "z_index": 0},
            {"pid": 2, "window_id": 2, "app_name": "b.exe", "z_index": 5},
        ]}
        gws_result = {"screenshot_width": 10, "screenshot_height": 10}
        with patch("urllib.request.urlopen", side_effect=_fake_responses(
                [{"ok": True, "result": windows}, {"ok": True, "result": gws_result}])) as mock_open:
            backend.capture()
        second_sent = json.loads(mock_open.call_args_list[1][0][0].data)
        assert second_sent["args"]["pid"] == 2

    def test_falls_back_to_first_candidate_when_z_index_all_null(self):
        backend = PebbleRelayBackend()
        windows = {"windows": [
            {"pid": 1, "window_id": 1, "app_name": "a.exe", "z_index": None},
            {"pid": 2, "window_id": 2, "app_name": "b.exe", "z_index": None},
        ]}
        gws_result = {"screenshot_width": 10, "screenshot_height": 10}
        with patch("urllib.request.urlopen", side_effect=_fake_responses(
                [{"ok": True, "result": windows}, {"ok": True, "result": gws_result}])) as mock_open:
            backend.capture()
        second_sent = json.loads(mock_open.call_args_list[1][0][0].data)
        assert second_sent["args"]["pid"] == 1

    def test_no_windows_open_returns_a_failed_capture_without_crashing(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {"windows": []}})):
            result = backend.capture()
        assert result.width == 0
        assert "no windows open" in result.note

    def test_app_not_found_returns_a_failed_capture_naming_the_app(self):
        backend = PebbleRelayBackend()
        windows = {"windows": [{"pid": 1, "window_id": 1, "app_name": "notion.exe", "z_index": 0}]}
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": windows})):
            result = backend.capture(app="totally-unknown-app")
        assert result.width == 0
        assert "totally-unknown-app" in result.note


class TestCapture:
    def test_reads_the_real_screenshot_field_names(self):
        # Regression: the pre-fix code read "width"/"height", which don't exist in the real
        # response (cua-driver actually returns "screenshot_width"/"screenshot_height") -- always
        # silently produced a 0x0 result. Confirmed against the real daemon before this fix.
        backend = PebbleRelayBackend()
        gws_result = {"screenshot_width": 1920, "screenshot_height": 1080, "screenshot_png_b64": "abc123"}
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": gws_result})):
            result = backend.capture(pid=1, window_id=2)
        assert result.width == 1920
        assert result.height == 1080
        assert result.png_b64 == "abc123"

    def test_parses_elements_and_builds_the_snapshot_token_cache(self):
        backend = PebbleRelayBackend()
        gws_result = {"screenshot_width": 10, "screenshot_height": 10, "screenshot_png_b64": "x",
                     "elements": [{"element_index": 0, "role": "Button", "label": "Refresh",
                                   "frame": {"x": 0, "y": 66, "w": 48, "h": 48}, "element_token": "s1:0"}]}
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": gws_result})):
            result = backend.capture(pid=1, window_id=2)
        assert len(result.elements) == 1
        assert result.elements[0].label == "Refresh"
        assert backend._snapshot_tokens == {0: "s1:0"}

    def test_vision_mode_skips_the_accessibility_tree_and_returns_no_elements(self):
        backend = PebbleRelayBackend()
        gws_result = {"screenshot_width": 10, "screenshot_height": 10, "screenshot_png_b64": "x",
                     "elements": [{"element_index": 0, "role": "Button", "element_token": "s1:0"}]}
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": gws_result})) as mock_open:
            result = backend.capture(mode="vision", pid=1, window_id=2)
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["args"]["include_accessibility_tree"] is False
        assert result.elements == []
        assert backend._snapshot_tokens == {}

    def test_bare_capture_prefers_the_sticky_target_over_frontmost(self):
        # Tony, 2026-09-29: confirmed live -- focus_app("Clock") ran, then a bare
        # capture(mode="som") with no app/pid/window_id ignored the just-set sticky target
        # entirely and re-resolved to whatever was actually frontmost (a leftover Chrome
        # window), timing out against it four times in a row. A caller that gives capture()
        # nothing to go on must get the sticky target, not a fresh frontmost guess.
        backend = PebbleRelayBackend()
        backend._active_pid, backend._active_window_id = 111, 222
        gws_result = {"screenshot_width": 10, "screenshot_height": 10, "screenshot_png_b64": "x"}
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": gws_result})) as mock_open:
            backend.capture(mode="som")
        sent = json.loads(mock_open.call_args[0][0].data)
        # No list_windows round trip either -- the sticky pid/window_id short-circuits
        # _resolve_capture_target's "pid and window_id already known" branch directly.
        assert sent["tool"] == "get_window_state"
        assert sent["args"]["pid"] == 111
        assert sent["args"]["window_id"] == 222

    def test_capture_with_an_explicit_app_ignores_the_sticky_target(self):
        backend = PebbleRelayBackend()
        backend._active_pid, backend._active_window_id = 111, 222
        windows = {"windows": [{"pid": 999, "window_id": 888, "app_name": "Notion.exe", "z_index": 1}]}
        with patch("urllib.request.urlopen", side_effect=_fake_responses([
                {"ok": True, "result": windows},  # list_windows, from the explicit app filter
                {"ok": True, "result": {"screenshot_width": 10, "screenshot_height": 10,
                                        "screenshot_png_b64": "x"}},
        ])):
            backend.capture(mode="som", app="Notion")
        assert backend._active_pid == 999
        assert backend._active_window_id == 888

    def test_degrades_gracefully_on_get_window_state_relay_failure(self):
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = backend.capture(pid=1, window_id=2)
        assert result.width == 0
        assert "capture failed" in result.note

    def test_failed_get_window_state_does_not_commit_a_sticky_target(self):
        # A failed capture must not leave click()/type_text() pointed at a window whose state was
        # never actually fetched.
        backend = PebbleRelayBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            backend.capture(pid=1, window_id=2)
        assert backend._active_pid is None
        result = backend.click(x=1, y=1)
        assert result.ok is False
        assert "no active capture target" in result.message


class TestClickAfterCapture:
    def test_click_refuses_without_a_prior_capture(self):
        backend = PebbleRelayBackend()
        result = backend.click(x=10, y=20)
        assert result.ok is False
        assert "no active capture target" in result.message

    def test_click_with_x_y_relays_pid_window_id_and_coordinates(self):
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {}})) as mock_open:
            result = backend.click(x=10, y=20, button="right")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "click",
                        "args": {"pid": 65852, "window_id": 21235952, "button": "right", "x": 10, "y": 20}}

    def test_double_click_uses_the_double_click_tool(self):
        backend = _captured_backend()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            backend.click(x=10, y=20, click_count=2)
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["tool"] == "double_click"

    def test_click_with_element_uses_the_element_token_from_the_last_capture(self):
        elements = [{"element_index": 4, "role": "Button", "label": "Refresh", "element_token": "s1:4"}]
        backend = _captured_backend(elements=elements)
        with patch("urllib.request.urlopen", return_value=_fake_response(
                {"ok": True, "result": {"effect": "unverifiable"}})) as mock_open:
            result = backend.click(element=4)
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["args"] == {"pid": 65852, "window_id": 21235952, "button": "left", "element_token": "s1:4"}
        assert "x" not in sent["args"] and "y" not in sent["args"]

    def test_click_with_unknown_element_index_refuses_clearly(self):
        backend = _captured_backend(elements=[])
        result = backend.click(element=99)
        assert result.ok is False
        assert "element 99" in result.message

    def test_click_needs_element_or_x_and_y(self):
        backend = _captured_backend()
        result = backend.click()
        assert result.ok is False
        assert "element" in result.message and "x" in result.message

    def test_click_forwards_delivery_mode_foreground(self):
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            backend.click(x=10, y=20, delivery_mode="foreground")
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["args"]["delivery_mode"] == "foreground"


class TestTypeTextAfterCapture:
    def test_type_text_refuses_without_a_prior_capture(self):
        backend = PebbleRelayBackend()
        result = backend.type_text("hello")
        assert result.ok is False
        assert "no active capture target" in result.message

    def test_type_text_relays_text_with_pid_and_window_id(self):
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            result = backend.type_text("hello jarvis")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "type_text",
                        "args": {"text": "hello jarvis", "pid": 65852, "window_id": 21235952}}

    def test_type_text_forwards_delivery_mode_foreground(self):
        # Tony, 2026-09-29: delivery_mode was accepted in the signature but silently never
        # sent to the relay -- every call used the driver's "background" default regardless
        # of what was requested, confirmed live against Edge's address bar (a real
        # browser-chrome control background input silently can't reach).
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            backend.type_text("hello", delivery_mode="foreground")
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["args"]["delivery_mode"] == "foreground"


class TestKeyAfterCapture:
    def test_key_refuses_without_a_prior_capture(self):
        backend = PebbleRelayBackend()
        result = backend.key("ctrl+s")
        assert result.ok is False
        assert "no active capture target" in result.message

    def test_key_relays_the_key_combo_with_pid_and_window_id(self):
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            result = backend.key("ctrl+s")
        assert result.ok is True
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent == {"tool": "press_key", "args": {"key": "ctrl+s", "pid": 65852, "window_id": 21235952}}

    def test_key_forwards_delivery_mode_foreground(self):
        backend = _captured_backend(pid=65852, window_id=21235952)
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "result": {}})) as mock_open:
            backend.key("return", delivery_mode="foreground")
        sent = json.loads(mock_open.call_args[0][0].data)
        assert sent["args"]["delivery_mode"] == "foreground"


class TestNewBackendWiring:
    """HERMES_COMPUTER_USE_BACKEND=pebble must actually select this class (tools/computer_use/
    tool.py:_new_backend) -- the whole point of the relay is reachability from the live gateway,
    so a typo'd or dropped wire-up here would silently fall through to the local cua-driver."""

    def test_backend_env_var_pebble_selects_pebble_relay_backend(self, monkeypatch):
        from tools.computer_use import tool as cu_tool

        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "pebble")
        backend = cu_tool._new_backend("standard")
        assert isinstance(backend, PebbleRelayBackend)
