"""ComputerUseBackend that relays through Pebble instead of spawning a local cua-driver.

Tony, 2026-09-23: "make computer_use from the live WSL gateway actually drive my real Windows
desktop." Root cause (confirmed live): cua-driver's own client/server split talks over a Unix
domain socket (named pipe on Windows) -- inherently local-machine-only, no network mode -- so
the gateway's computer_use always spawned its OWN local cua-driver, driving WSL's own virtual
desktop (confirmed: its list_apps returned "systemd" as PID 1, a Linux-only process) instead of
this real one.

Pebble is a live, always-running Windows process with a proven WSL<->Windows HTTP bridge
(control-server.js, used for voice-state/automation-click since August). This backend POSTs to
its new `/computer-use-execute` endpoint (control-server.js, pebble-app repo), which shells out
to the cua-driver ALREADY installed and verified working on the real Windows machine, and
relays the real result back -- reusing both proven pieces instead of building a new standalone
Windows service.

Scope (v1, honest about what's NOT here rather than half-implementing it): capture, click,
type_text, key, launch_app, focus_app, list_apps are real, relayed, tested actions. drag, scroll,
set_value raise a clear NotImplementedError-shaped ActionResult; list_windows raises a plain
NotImplementedError (its ABC return type is a bare list, with no room for a refusal payload) --
both instead of silently reporting "no windows"/"nothing to do". Confirmed live 2026-09-23
(session 20260923_132613_72f606): before this fix, list_windows silently inherited
ComputerUseBackend's default `return []` (meant for backends that genuinely lack window
discovery, not an unfinished relay method) and the agent reported "no windows found" as if it
were a real, successful query of Tony's desktop. Add real implementations the same way list_apps
was added (one more allowlisted tool name + one more thin method) when actually needed."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend

logger = logging.getLogger(__name__)

_PEBBLE_DEFAULT_URL = "http://127.0.0.1:58734"
_RELAY_TIMEOUT_S = 35.0


class PebbleRelayError(RuntimeError):
    """Raised when the relay call itself fails (network, Pebble not running, bad JSON) --
    distinct from a normal ActionResult(ok=False), which is cua-driver's own refusal."""


def _pebble_url() -> str:
    return os.environ.get("PEBBLE_URL", _PEBBLE_DEFAULT_URL).rstrip("/")


def relay_call(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """POST one action to Pebble's /computer-use-execute, return the real cua-driver result.
    Raises PebbleRelayError on any transport/relay failure (Pebble not running, tool not in its
    allowlist, cua-driver call itself errored) -- callers turn that into a refusal ActionResult,
    never a raw exception reaching the model."""
    body = json.dumps({"tool": tool, "args": args}).encode()
    req = urllib.request.Request(
        f"{_pebble_url()}/computer-use-execute", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_RELAY_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode())
    except OSError as exc:
        # urllib.error.URLError is itself an OSError subclass, but a bare ConnectionRefusedError
        # (Pebble simply not running) can also propagate directly without urllib wrapping it --
        # catch OSError broadly rather than just URLError so both shapes hit this one message.
        raise PebbleRelayError(f"could not reach Pebble at {_pebble_url()}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PebbleRelayError(f"Pebble returned unparseable JSON for {tool!r}: {exc}") from exc
    if not payload.get("ok"):
        raise PebbleRelayError(payload.get("error") or f"Pebble relay for {tool!r} failed with no error detail")
    return payload.get("result") or {}


def _not_implemented(action: str) -> ActionResult:
    return ActionResult(ok=False, action=action, code="pebble_relay_not_implemented",
                       message=f"{action} is not yet implemented in the Pebble relay backend "
                                "(v1 covers capture/click/type_text/key/launch_app/focus_app/list_apps).")


class PebbleRelayBackend(ComputerUseBackend):
    """See module docstring. Stateless between calls (no local session/daemon to manage) --
    start()/stop() are no-ops beyond an availability check, matching how little state a pure
    HTTP relay actually needs to hold."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    def start(self) -> None:
        self._available = self.is_available()
        if not self._available:
            raise RuntimeError(f"Pebble is not reachable at {_pebble_url()} -- "
                               "the computer-use relay needs it running.")

    def stop(self) -> None:
        self._available = None

    def is_available(self) -> bool:
        req = urllib.request.Request(f"{_pebble_url()}/health", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return json.loads(resp.read().decode()).get("ok") is True
        except Exception:
            return False

    # ── Real, relayed actions ────────────────────────────────────────────────
    def launch_app(self, name: str) -> ActionResult:
        try:
            result = relay_call("launch_app", {"app": name})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="launch_app", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="launch_app",
                            message=f"launched {name!r}" if ok else f"launch_app {name!r} failed: {result}",
                            meta={"raw": result})

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        # launch_app is documented+verified idempotent (tools/computer_use/tool.py's own comment,
        # confirmed live tonight): it starts the app if not running, or is a no-op/focuses it if
        # already running -- covers the focus case without a separate window-matching relay call.
        result = self.launch_app(app)
        result.action = "focus_app"
        return result

    def list_apps(self) -> List[Dict[str, Any]]:
        try:
            result = relay_call("list_apps", {})
        except PebbleRelayError as exc:
            logger.warning("PebbleRelayBackend.list_apps failed: %s", exc)
            return []
        apps = result.get("apps")
        return apps if isinstance(apps, list) else []

    def list_windows(self) -> List[Dict[str, Any]]:
        # Deliberately NOT the ComputerUseBackend default (`return []`) -- that default means
        # "this backend genuinely has no window discovery", but here it silently masked an
        # unfinished relay method as "no windows open". Raise instead so tool.py's existing
        # dispatch exception handling (tools/computer_use/tool.py: `except Exception as e: return
        # json.dumps({"error": ...})`) turns this into a real, visible refusal.
        raise NotImplementedError(_not_implemented("list_windows").message)

    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult:
        try:
            result = relay_call("type_text", {"text": text})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="type_text", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="type_text", message="" if ok else str(result), meta={"raw": result})

    def key(self, keys: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        try:
            result = relay_call("press_key", {"key": keys})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="key", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="key", message="" if ok else str(result), meta={"raw": result})

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[List[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        if x is None or y is None:
            return ActionResult(ok=False, action="click",
                                message="PebbleRelayBackend v1 click needs x/y (element-index "
                                         "resolution isn't relayed yet) -- capture first.")
        tool = "double_click" if click_count == 2 else "click"
        try:
            result = relay_call(tool, {"x": x, "y": y, "button": button})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="click", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="click", message="" if ok else str(result), meta={"raw": result})

    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult:
        args: Dict[str, Any] = {}
        if pid is not None:
            args["pid"] = pid
        if window_id is not None:
            args["window_id"] = window_id
        try:
            result = relay_call("get_window_state", args)
        except PebbleRelayError as exc:
            logger.warning("PebbleRelayBackend.capture failed: %s", exc)
            return CaptureResult(mode=mode, width=0, height=0, note=f"capture failed: {exc}")
        png_b64 = result.get("screenshot_png_b64")
        return CaptureResult(
            mode=mode, width=int(result.get("width") or 0), height=int(result.get("height") or 0),
            png_b64=png_b64, app=app or "", note="" if png_b64 else "no screenshot in relay response",
        )

    # ── Not yet relayed (v1 scope) ───────────────────────────────────────────
    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[Any] = None, to_xy: Optional[Any] = None,
             button: str = "left", modifiers: Optional[List[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        return _not_implemented("drag")

    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[List[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        return _not_implemented("scroll")

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return _not_implemented("set_value")
