"""Regression test: post_llm_call must fire even on interrupted/empty turns.

pebble-signal (and any other plugin tracking turn lifecycle) relies on
post_llm_call as the *only* signal that a turn ended, to balance the
pre_llm_call it received at turn start. Before this fix, post_llm_call was
gated behind ``if final_response and not interrupted``, so an interrupted
turn (e.g. a mid-turn /steer) or a turn that produced no final text left
listeners with no matching "turn ended" event -- pebble-signal's droplet
state got stuck on "thinking" forever, since its internal tracker (a set of
active session keys) only clears a session on post_llm_call.

Re-created 2026-07-29: the previous version of this test + fix was reset
by a `hermes update` (history diverged -> reset to remote). turn_finalizer.py
now imports invoke_hook from hermes_cli.lifecycle instead of hermes_cli.plugins,
but that module is a thin wrapper that delegates to the same
hermes_cli.plugins.PluginManager underneath, so this test's approach
(discover_and_load against a real plugin dir) is unchanged.
"""

from pathlib import Path

import yaml

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _make_capture_plugin(hermes_home: Path, hook_name: str, sink: list) -> None:
    """Register a real plugin that appends to ``sink`` whenever ``hook_name`` fires."""
    plugin_dir = hermes_home / "plugins" / "capture_hook"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "capture_hook", "version": "0.1.0"}), encoding="utf-8",
    )
    marker = hermes_home / "hook_fired.marker"
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    def _capture(**kw):\n"
        f"        open({str(marker)!r}, 'a').write('fired\\n')\n"
        f"    ctx.register_hook({hook_name!r}, _capture)\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = {"plugins": {"enabled": ["capture_hook"]}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _run(agent, tmp_path, monkeypatch, *, final_response, interrupted, api_call_count=3, failed=False):
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_capture_plugin(hermes_home, "post_llm_call", [])
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod._plugin_manager.discover_and_load()

    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
    )
    marker = hermes_home / "hook_fired.marker"
    return marker.exists()


def test_post_llm_call_fires_on_clean_completion(tmp_path, monkeypatch):
    agent = _StubAgent()
    fired = _run(agent, tmp_path, monkeypatch, final_response="all done", interrupted=False)
    assert fired is True


def test_post_llm_call_fires_when_turn_interrupted(tmp_path, monkeypatch):
    """A /steer-interrupted turn must still balance the pre_llm_call that
    started it -- otherwise pebble-signal's tracker never sees the matching
    'turn ended' event and the droplet is stuck on 'thinking' forever."""
    agent = _StubAgent()
    fired = _run(agent, tmp_path, monkeypatch, final_response="partial text", interrupted=True)
    assert fired is True


def test_post_llm_call_fires_when_no_final_response(tmp_path, monkeypatch):
    """A turn that ends without producing final text (a genuine failure, not
    a budget-exhaustion fallback which supplies its own summary text) must
    still balance pre_llm_call for the same reason."""
    agent = _StubAgent()
    fired = _run(
        agent, tmp_path, monkeypatch,
        final_response=None, interrupted=False, failed=True,
    )
    assert fired is True
