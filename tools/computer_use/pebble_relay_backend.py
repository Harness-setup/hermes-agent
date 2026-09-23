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
was added (one more allowlisted tool name + one more thin method) when actually needed.

Element-index (UIA) addressing, 2026-09-23: Tony asked "will clicking use UIA?" -- the honest v1
answer was "not yet, only raw x,y" (CuaDriverBackend's own real-click path always tries a UIA
Invoke first; this relay didn't carry the pieces needed to do the same). Fixed by porting
CuaDriverBackend's sticky-target pattern (tools/computer_use/cua_backend.py's own
`_active_pid`/`_active_window_id`/`_snapshot_tokens`): `capture()` now resolves a real target
window via `list_windows` when the caller doesn't already know pid/window_id, fetches its UIA
element tree via `get_window_state`, and remembers (pid, window_id, {element_index:
element_token}) for the rest of the session. `click()` then addresses a captured element by
`element_token` -- the same accessibility-channel path `describe click` documents ("performs the
UIA Invoke pattern on the cached element... no cursor move, no focus steal") -- instead of only
ever guessing a raw pixel. `type_text`/`key` (`press_key`) also needed `pid`/`window_id` attached
for the same reason click did: cua-driver routes both by PostMessage to a specific target window,
and v1 never sent one. Verified all of this against the real daemon before writing the Python
(not guessed): `cua-driver call get_window_state` returns `screenshot_width`/`screenshot_height`
(NOT `width`/`height` -- the old capture() code was silently reading a field that doesn't exist
and always got 0x0) and an `elements` array shaped exactly like
`tools/computer_use/cua_backend_parse.py`'s `_parse_elements_from_structured` already expects, so
that existing, tested, backend-agnostic parser is reused here rather than re-implemented. A real
`element_token`-addressed click against the live daemon (Edge's "Refresh" button) round-tripped
correctly (`{"effect": "unverifiable", "route": "synthetic_events", "delivery": {"mode":
"background"}}` -- background delivery, exactly as documented) before this was wired into
PebbleRelayBackend."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend
from tools.computer_use.cua_backend_parse import _parse_elements_from_structured

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
    """See module docstring. start()/stop() are no-ops beyond an availability check -- there's no
    local daemon subprocess to manage, only the HTTP relay to Pebble. NOT fully stateless though:
    a per-session sticky target (`_active_pid`/`_active_window_id`/`_snapshot_tokens`) is
    established by `capture()` and consumed by `click()`/`type_text()`/`key()`, the same pattern
    CuaDriverBackend uses -- this instance is cached per-session by tool.py's `_get_backend`, so
    the target survives across the multiple tool calls one real interaction needs (capture, then
    click an element it found)."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        # element_index (from the last capture's elements) -> element_token. Overwritten whole on
        # every capture(), same as CuaDriverBackend's own _snapshot_tokens -- a stale index from an
        # earlier snapshot must miss, not silently resolve against the wrong element.
        self._snapshot_tokens: Dict[int, str] = {}

    def _resolve_capture_target(self, app: Optional[str], pid: Optional[int],
                                window_id: Optional[int]) -> Optional[Dict[str, int]]:
        """An exact (pid, window_id) if the caller already knows it; otherwise discover one via
        `list_windows`, preferring an `app`-name match and then the frontmost (max z_index)
        candidate. None when nothing matches (caller turns that into a failed CaptureResult)."""
        if pid is not None and window_id is not None:
            return {"pid": pid, "window_id": window_id}
        windows = relay_call("list_windows", {"pid": pid} if pid is not None else {}).get("windows")
        if not isinstance(windows, list) or not windows:
            return None
        if app:
            needle = app.strip().lower()
            candidates = [w for w in windows if isinstance(w, dict) and needle in str(w.get("app_name", "")).lower()]
            if not candidates:
                return None  # an app filter that matches nothing is "not found", not "search anything"
        else:
            candidates = windows
        candidates = [w for w in candidates if isinstance(w, dict)
                     and isinstance(w.get("pid"), int) and isinstance(w.get("window_id"), int)]
        if not candidates:
            return None
        # Frontmost = max z_index; list_windows documents null as "stacking order unavailable" --
        # fall back to the first candidate rather than treating None as sortable (it isn't, and
        # a naive sort would raise on a mix of int/None).
        with_z = [w for w in candidates if isinstance(w.get("z_index"), int)]
        target = max(with_z, key=lambda w: w["z_index"]) if with_z else candidates[0]
        return {"pid": target["pid"], "window_id": target["window_id"]}

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

    def _require_active_target(self, action: str) -> Optional[ActionResult]:
        """None when a sticky target exists; otherwise the refusal ActionResult the caller should
        return as-is. cua-driver routes type/key/click by PostMessage to a specific (pid,
        window_id) -- there is no sane default to fall back to, so this must be explicit rather
        than silently omitting pid and letting the daemon guess."""
        if self._active_pid is not None and self._active_window_id is not None:
            return None
        return ActionResult(ok=False, action=action,
                            message=f"no active capture target -- call computer_use capture first so "
                                     f"{action} knows which window to target")

    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult:
        if (refusal := self._require_active_target("type_text")) is not None:
            return refusal
        try:
            result = relay_call("type_text", {"text": text, "pid": self._active_pid,
                                              "window_id": self._active_window_id})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="type_text", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="type_text", message="" if ok else str(result), meta={"raw": result})

    def key(self, keys: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        if (refusal := self._require_active_target("key")) is not None:
            return refusal
        try:
            result = relay_call("press_key", {"key": keys, "pid": self._active_pid,
                                              "window_id": self._active_window_id})
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="key", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="key", message="" if ok else str(result), meta={"raw": result})

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[List[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        if (refusal := self._require_active_target("click")) is not None:
            return refusal
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id, "button": button}
        if element is not None:
            # Preferred path (matches CuaDriverBackend / `describe click`): element_token performs
            # the UIA Invoke pattern on the cached element -- no cursor move, no focus steal.
            token = self._snapshot_tokens.get(element)
            if not token:
                return ActionResult(ok=False, action="click",
                                    message=f"element {element} isn't in the last capture's element "
                                             "cache -- capture again to refresh element indices, "
                                             "then click the fresh index")
            args["element_token"] = token
        elif x is not None and y is not None:
            # Pixel fallback: cua-driver still tries a UIA hit-test at (x, y) first (see `describe
            # click`), so this keeps the same background/no-focus-steal behavior for canvas/video/
            # WebGL surfaces that have no UIA peer to address by element_token.
            args["x"], args["y"] = x, y
        else:
            return ActionResult(ok=False, action="click", message="click needs either `element` or both `x` and `y`")
        tool = "double_click" if click_count == 2 else "click"
        try:
            result = relay_call(tool, args)
        except PebbleRelayError as exc:
            return ActionResult(ok=False, action="click", message=str(exc))
        ok = not (isinstance(result, dict) and result.get("isError"))
        return ActionResult(ok=ok, action="click", message="" if ok else str(result), meta={"raw": result})

    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult:
        try:
            target = self._resolve_capture_target(app, pid, window_id)
        except PebbleRelayError as exc:
            logger.warning("PebbleRelayBackend.capture failed to list windows: %s", exc)
            return CaptureResult(mode=mode, width=0, height=0, note=f"capture failed: {exc}")
        if target is None:
            note = f"no window found matching app={app!r}" if app else "no windows open on the desktop"
            return CaptureResult(mode=mode, width=0, height=0, note=note)
        gws_args: Dict[str, Any] = dict(target)
        if mode == "vision":
            gws_args["include_accessibility_tree"] = False
        try:
            result = relay_call("get_window_state", gws_args)
        except PebbleRelayError as exc:
            logger.warning("PebbleRelayBackend.capture failed: %s", exc)
            return CaptureResult(mode=mode, width=0, height=0, note=f"capture failed: {exc}")
        # Only commit the sticky target after a successful get_window_state -- a failed capture
        # must not leave click()/type_text() pointed at a window whose state was never fetched.
        self._active_pid, self._active_window_id = target["pid"], target["window_id"]
        elements = [] if mode == "vision" else _parse_elements_from_structured(result.get("elements") or [])
        self._snapshot_tokens = {e.index: e.element_token for e in elements if e.element_token}
        png_b64 = result.get("screenshot_png_b64")
        return CaptureResult(
            mode=mode,
            # cua-driver's real field names, confirmed live against the daemon -- NOT "width"/
            # "height" (those don't exist in the response; reading them always returned 0x0).
            width=int(result.get("screenshot_width") or 0), height=int(result.get("screenshot_height") or 0),
            png_b64=png_b64, elements=elements, app=str(result.get("app_name") or app or ""),
            window_title=str(result.get("window_title") or ""),
            note="" if png_b64 else "no screenshot in relay response",
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
