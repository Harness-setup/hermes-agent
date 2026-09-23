"""Tests for the skip_background_review constructor flag.

Verifies that AIAgent can be instructed to skip the end-of-turn
_spawn_background_review fork (~30K tokens / event), which is essential
on cron sessions that have no human-in-the-loop value from skill/memory
review forks.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent
from agent.turn_finalizer import finalize_turn


def _make_agent(skip_background_review: bool = False) -> AIAgent:
    """Construct a minimally-configured AIAgent for unit testing."""
    return AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=skip_background_review,
        platform="cli",
    )


def _stub_agent_for_finalize(agent: AIAgent) -> None:
    """Stub the heavy finalizer dependencies to isolate the review gate."""
    agent._spawn_background_review = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = None
    agent._sync_external_memory_for_turn = MagicMock()
    agent._skill_nudge_interval = 10
    agent._iters_since_skill = 20  # exceeds nudge interval → _should_review_skills = True
    agent.valid_tool_names = {"skill_manage"}
    agent.iteration_budget = MagicMock()
    agent.iteration_budget.remaining = 100
    agent.iteration_budget.used = 5
    agent.iteration_budget.max_total = 100
    agent.max_iterations = 50
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._apply_persist_user_message_override = MagicMock()
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent.quiet_mode = True
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None


def _run_finalize(agent: AIAgent) -> None:
    """Call finalize_turn with conditions that would trigger background review."""
    finalize_turn(
        agent,
        final_response="ok",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "assistant", "content": "ok"}],
        conversation_history=[],
        effective_task_id="test",
        turn_id="test-turn",
        user_message="test",
        original_user_message="test",
        _should_review_memory=True,
        _turn_exit_reason="text_response(1)",
    )


def test_default_skip_background_review_is_false() -> None:
    """Without an explicit override, AIAgent does NOT skip background review."""
    agent = _make_agent()
    assert agent.skip_background_review is False


def test_skip_background_review_flag_persists() -> None:
    """Passing skip_background_review=True records the flag on the instance."""
    agent = _make_agent(skip_background_review=True)
    assert agent.skip_background_review is True


def test_finalize_turn_skips_review_when_flag_set() -> None:
    """finalize_turn must NOT call _spawn_background_review when skip_background_review=True.

    Exercises the actual finalizer call path (not a duplicated guard expression)
    so it catches divergence between the production guard and the test.
    """
    agent = _make_agent(skip_background_review=True)
    _stub_agent_for_finalize(agent)
    _run_finalize(agent)
    agent._spawn_background_review.assert_not_called()


def test_finalize_turn_fires_review_when_flag_unset() -> None:
    """Counterpart: with the flag off, finalize_turn DOES call _spawn_background_review."""
    agent = _make_agent(skip_background_review=False)
    _stub_agent_for_finalize(agent)
    _run_finalize(agent)
    agent._spawn_background_review.assert_called_once()


def test_finalize_turn_skips_review_when_persist_disabled() -> None:
    """finalize_turn must NOT call _spawn_background_review when
    agent._persist_disabled is True (uncensored mode's no-trace guarantee).

    _should_review_memory only checks "memory" in valid_tool_names, which
    stays true while uncensored (memory_guard.py blocks the call, not the
    tool's listing) -- without this gate the review fork would still spawn
    and send the full conversation to an LLM call (the review model,
    possibly a configured cloud model), even though the eventual memory
    write gets blocked downstream. Exercises the real finalizer path, not
    a duplicated guard expression.
    """
    agent = _make_agent(skip_background_review=False)
    _stub_agent_for_finalize(agent)
    agent._persist_disabled = True
    _run_finalize(agent)
    agent._spawn_background_review.assert_not_called()


def test_finalize_turn_fires_review_when_persist_not_disabled() -> None:
    """Counterpart: with persistence enabled, finalize_turn DOES review."""
    agent = _make_agent(skip_background_review=False)
    _stub_agent_for_finalize(agent)
    agent._persist_disabled = False
    _run_finalize(agent)
    agent._spawn_background_review.assert_called_once()


def test_cron_construction_sets_skip_background_review() -> None:
    """The cron scheduler MUST construct AIAgent with skip_background_review=True.

    Verified via source-text inspection — the cron scheduler is heavy to
    boot in tests, so we assert that the source declares the flag rather
    than running the scheduler. This catches accidental removal.
    """
    import pathlib

    scheduler_src = pathlib.Path(__file__).resolve().parents[2] / "cron" / "scheduler.py"
    text = scheduler_src.read_text(encoding="utf-8")

    assert "skip_background_review=True" in text, (
        "cron/scheduler.py must construct AIAgent with skip_background_review=True."
    )


def test_uncensored_session_init_sets_skip_background_review() -> None:
    """Tony, 2026-08-25: "did we make it so that it does not save anything
    on memory, user memory, system memory and stuff like that?" -- the
    explicit "memory" tool is already blocked at call-time (mode/
    memory_guard.py's pre_tool_call hook), but that alone leaves a
    background-review fork (turn_finalizer.py's _spawn_background_review,
    another full AIAgent processing the conversation) free to spawn and
    read the "private" conversation regardless of whether its own eventual
    memory-write attempt gets blocked. agent_init.py's full session-init
    function is heavy to construct in isolation (same reasoning as the
    cron test above) -- verified via source-text inspection instead: the
    uncensored-session branch must set agent.skip_background_review = True,
    not just leave the flag at whatever the (non-uncensored-aware) caller
    passed in.
    """
    import pathlib

    agent_init_src = pathlib.Path(__file__).resolve().parents[2] / "agent" / "agent_init.py"
    text = agent_init_src.read_text(encoding="utf-8")

    if_block_start = text.index("if _is_uncensored_session:")
    # The next top-level statement at the same indentation ends this
    # if-block -- "agent._session_init_model_config" is the line
    # immediately following it, unchanged since this was written.
    if_block_end = text.index("agent._session_init_model_config", if_block_start)
    if_block = text[if_block_start:if_block_end]

    assert "agent.skip_background_review = True" in if_block, (
        "agent_init.py's uncensored-session branch must set "
        "agent.skip_background_review = True, or a background review fork "
        "can still read/process a 'private' uncensored conversation."
    )


def test_persistence_failure_error_fallback_is_pinned_and_leaves_final_response_empty(monkeypatch, tmp_path) -> None:
    """With no model text, result["error"] carries a profile-pinned `hermes doctor`, while the
    memory sync and the background-review gate still see the turn as having produced nothing."""
    from hermes_constants import profile_cli_selector

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    agent = _make_agent()
    _stub_agent_for_finalize(agent)
    # Force the fallback: the explainer normally supplies the text, so an empty explainer is
    # the only way the hardcoded copy reaches the user.
    monkeypatch.setattr(AIAgent, "_format_turn_completion_explanation", staticmethod(lambda *a, **k: ""))
    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=[{"role": "user", "content": "hi"}],
        conversation_history=[],
        effective_task_id="test",
        turn_id="test-turn",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=True,
        _turn_exit_reason="session_persistence_failed",
    )
    assert f"`hermes {selector}doctor`" in result["error"]
    assert agent._sync_external_memory_for_turn.call_args.kwargs["final_response"] == ""
    agent._spawn_background_review.assert_not_called()
