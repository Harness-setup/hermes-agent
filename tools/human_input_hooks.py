"""Plugin observers for every point where the agent blocks waiting on a human (#132333).

``on_human_input_request`` records preparation; ``on_human_input_shown`` records confirmed
delivery or renderer acknowledgment. ``on_human_input_resolved`` fires exactly once when the wait ends,
with the same ``request_id`` and an ``outcome``. Observers only: return values are ignored and a
failing plugin never affects the prompt. Payloads never carry what the human typed (password,
answers), and ``prompt`` is force-redacted so credentials in a command never reach a plugin.
"""

from __future__ import annotations

import contextlib
from functools import wraps
import logging
import os
import threading
import time
from contextvars import ContextVar, copy_context
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HumanInputRequest:
    """Handle yielded by :func:`human_input_request`; set ``outcome`` before the block exits."""

    request_id: str
    outcome: str = ""
    payload: dict = field(default_factory=dict)
    shown_at: float | None = None
    resolved_at: float | None = None

    def __post_init__(self):
        self._lock = threading.RLock()
        self._context = copy_context()

    def shown(self) -> bool:
        """A confirmed delivery/render; late delivery cannot revive a finished wait."""
        with self._lock:
            if self.resolved_at is not None or self.shown_at is not None:
                return False
            self.shown_at = time.time()
            self._context.copy().run(_fire, "on_human_input_shown", {**self.payload, "shown_at": self.shown_at})
            return True

    def resolve(self, outcome: str) -> None:
        with self._lock:
            if self.resolved_at is not None:
                return
            self.resolved_at = time.time()
            self._context.copy().run(_fire, "on_human_input_resolved", {**self.payload, "shown_at": self.shown_at,
                  "resolved_at": self.resolved_at, "outcome": str(outcome)})


_active: ContextVar[tuple] = ContextVar("human_input_requests", default=())


def current_requests() -> tuple:
    return _active.get()


def mark_current_shown() -> None:
    for request in current_requests():
        request.shown()


def watch_delivery(future) -> None:
    """Capture the owning handles before crossing to the messaging event-loop thread."""
    requests = current_requests()
    if not requests:
        return
    def delivered(done):
        if done.cancelled() or done.exception() is not None:
            return
        if getattr(done.result(), "success", False):
            for request in requests:
                request.shown()
    if future is not None and callable(getattr(future, "add_done_callback", None)):
        future.add_done_callback(delivered)


def _redacted(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(str(text or ""), force=True)
    except Exception:
        logger.debug("human-input hook prompt redaction failed", exc_info=True)
        return ""  # never fall back to the raw text


def _identity() -> dict:
    try:
        from gateway.session_context import get_session_env
        from tools import approval_context
        session_id = approval_context._approval_session_id.get() or get_session_env("HERMES_SESSION_ID")
        platform = (os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM")
                    or get_session_env("HERMES_SESSION_SOURCE") or "cli")
        return {"session_id": session_id, "session_key": approval_context.get_current_session_key(default=""),
                "platform": platform}
    except Exception:  # a broken lookup must not take the human prompt down with it
        logger.debug("human-input hook identity lookup failed", exc_info=True)
        return {"session_id": "", "session_key": "", "platform": ""}


def _fire(hook_name: str, payload: dict) -> None:
    logger.debug("%s request_id=%s session_id=%s session_key=%s kind=%s prepared_at=%s shown_at=%s resolved_at=%s",
                 hook_name, payload.get("request_id"), payload.get("session_id"), payload.get("session_key"),
                 payload.get("kind"), payload.get("prepared_at"), payload.get("shown_at"), payload.get("resolved_at"))
    try:
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook(hook_name, **payload)
    except Exception:  # observability must never break a human prompt
        logger.debug("%s dispatch failed", hook_name, exc_info=True)


@contextlib.contextmanager
def human_input_request(kind: str, *, prompt: str = "", session_key: str | None = None, **extra):
    """Fire the request hook, run the block (the human wait), then fire the resolved hook.
    An exception escaping the block, or a block that never sets ``outcome``, resolves as ``error``."""
    existing = current_requests()
    if kind != "clarify" and existing and existing[-1].payload["kind"] == kind:
        yield existing[-1]
        return
    payload = {"kind": kind, "request_id": extra.pop("request_id", None) or uuid.uuid4().hex,
               **_identity(), "prompt": _redacted(prompt), **extra, "prepared_at": time.time()}
    if session_key is not None:
        payload["session_key"] = session_key
    request = HumanInputRequest(payload["request_id"], payload=payload)
    token = _active.set((*existing, request))
    _fire("on_human_input_request", payload)
    try:
        yield request
    finally:
        request.resolve(request.outcome or "error")
        _active.reset(token)


def standalone_request(kind: str, session_id: str) -> HumanInputRequest:
    payload = {"kind": kind, "request_id": uuid.uuid4().hex, **_identity(),
               "session_id": session_id, "prepared_at": time.time()}
    request = HumanInputRequest(payload["request_id"], payload=payload)
    _fire("on_human_input_request", payload)
    return request


@contextlib.contextmanager
def question_scope(question_id: str):
    handles = tuple(h for h in current_requests() if h.payload.get("question_id") == question_id)
    token = _active.set(handles)
    try:
        yield handles
    finally:
        _active.reset(token)


def observe_input(kind: str):
    """Wrap a local input callback without exposing its returned value to observers."""
    def decorate(callback):
        @wraps(callback)
        def observed(*args, **kwargs):
            with human_input_request(kind) as human:
                result = callback(*args, **kwargs)
                provided = result.get("success", False) if isinstance(result, dict) else bool(result)
                human.outcome = "provided" if provided else "skipped"
                return result
        return observed
    return decorate


class HumanInputEvent(threading.Event):
    """Commit input resolution before waking a waiter or accepting a late delivery callback."""
    def __init__(self, requests, outcome):
        super().__init__()
        self._requests = requests
        self._outcome = outcome

    def set(self):
        for human in self._requests():
            human.resolve(self._outcome())
        super().set()
