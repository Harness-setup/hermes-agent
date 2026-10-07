"""Delivery, cancellation, identity and batch display lifecycle contracts."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tools import human_input_hooks as hooks


@pytest.fixture
def events(monkeypatch):
    events = []
    monkeypatch.setattr(hooks, "_fire", lambda name, data: events.append((name, data)))
    return events


def test_delayed_delivery_and_late_cancellation(events, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(hooks.time, "time", lambda: clock[0])
    with hooks.human_input_request("approval", session_key="first") as human:
        future = Future()
        hooks.watch_delivery(future)
        clock[0] = 30.0  # judge/delivery can remain pending indefinitely without display
        assert [n for n, _ in events] == ["on_human_input_request"]
        future.set_result(SimpleNamespace(success=True))
        assert events[-1][0] == "on_human_input_shown"
        assert events[-1][1]["shown_at"] == 30.0
        human.outcome = "deny"
    with hooks.human_input_request("approval", session_key="second"):
        late = Future()
        hooks.watch_delivery(late)
    count = len(events)
    late.set_result(SimpleNamespace(success=True))
    assert len(events) == count


def test_failed_send_then_successful_fallback_and_nested_identity(events):
    with hooks.human_input_request("approval") as outer:
        with hooks.human_input_request("approval") as inner:
            assert inner is outer
            failed, fallback = Future(), Future()
            hooks.watch_delivery(failed)
            failed.set_result(SimpleNamespace(success=False, raw_response={}))
            assert len(events) == 1
            hooks.watch_delivery(fallback)
            fallback.set_result(SimpleNamespace(success=True))
            outer.outcome = "deny"
    assert [n for n, _ in events] == ["on_human_input_request", "on_human_input_shown", "on_human_input_resolved"]


def test_batch_questions_display_and_resolve_separately(events, monkeypatch):
    from tools.clarify_tool import clarify_tool
    from tui_gateway import server_requests as sr
    monkeypatch.setattr(sr, "_answerable", lambda sid: True)
    def render(frame):
        rid = frame["id"]
        assert not sr.acknowledge_display("other-session", rid)
        assert sr.acknowledge_display("owner", rid)
        assert sr.lock_answer(rid, "q1", "private answer") == ["q0"]
        assert sr.lock_answer(rid, "q0", None) == []
        assert not sr.acknowledge_display("owner", rid)
    monkeypatch.setattr(sr, "_write", render)
    sr.reset_for_tests()
    clarify_tool([{"question": "First?"}, {"question": "Second?"}],
                 callback=lambda qs: sr.send("clarify", "owner", {"questions": [{k: q[k] for k in ("qid", "question", "choices", "multi_select")} for q in qs]},
                                            timeout=0, qids=[q["qid"] for q in qs]))
    shown = [p for n, p in events if n == "on_human_input_shown"]
    resolved = [p for n, p in events if n == "on_human_input_resolved"]
    assert len(shown) == len(resolved) == 2
    assert len({p["request_id"] for p in shown}) == 2
    assert [p["question_id"] for p in resolved] == ["q1", "q0"]
    assert "private answer" not in repr(events)


def test_queue_cancellation_blocks_late_delivery_before_waiter_wakes(events):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools import approval
    future = Future()
    entry = _ApprovalEntry({'command': 'harmless', 'request_id': 'cancel-before-delivery'})
    with hooks.human_input_request('approval', session_key='cancel-race', request_id=entry.data['request_id']) as human:
        entry.human = human
        approval._gateway_queues['cancel-race'] = [entry]
        hooks.watch_delivery(future)
        assert approval.withdraw_gateway_approval('cancel-race', entry.data['request_id'], 'session closed')
        future.set_result(SimpleNamespace(success=True))
        assert not any(n == 'on_human_input_shown' for n, _ in events)
    assert len([n for n, _ in events if n == 'on_human_input_resolved']) == 1


def test_nested_password_prompt_has_its_own_identity(events):
    from tui_gateway.server_requests import ServerRequest
    with hooks.human_input_request("approval") as approval:
        password = ServerRequest("owner", "secret", {})
        assert len(password.humans) == 1
        assert password.humans[0] is not approval
        assert password.humans[0].payload["kind"] == "secret"
        password.humans[0].shown()
        password.humans[0].resolve("provided")
    assert len({p["request_id"] for n, p in events if n == "on_human_input_request"}) == 2
