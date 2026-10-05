from types import SimpleNamespace

from agent.turn_stop_gates import apply_stop_gates
from hermes_cli import plugins


def _agent():
    return SimpleNamespace(
        session_id='task', model='test', valid_tool_names={'terminal'},
        _emit_interim_assistant_message=lambda row: None,
        _flush_messages_to_session_db=lambda messages, history: None,
        _interim_content_was_streamed=lambda text: False,
    )


def test_completion_hook_is_bounded_preserves_prefix_and_role_alternation(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    folder = home / 'plugins' / 'completion-test'
    folder.mkdir(parents=True)
    (folder / 'plugin.yaml').write_text('name: completion-test\nversion: "1.0.0"\nhooks: [pre_finish]\n')
    (folder / '__init__.py').write_text('def check(**kwargs):\n    return {"action": "continue", "message": "Execute the pending task"}\ndef register(ctx):\n    ctx.register_hook("pre_finish", check)\n')
    (home / 'config.yaml').write_text('plugins:\n  enabled: [completion-test]\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(plugins, '_plugin_manager', plugins.PluginManager())
    plugins.discover_plugins()
    agent = _agent()
    messages = [{'role': 'system', 'content': 'stable prefix'}, {'role': 'user', 'content': 'Run task'}]
    prefix = [dict(m) for m in messages]
    for _ in range(2):
        verdict = apply_stop_gates(agent, {'role': 'assistant', 'content': 'I will run it'},
            final_response='I will run it', messages=messages, conversation_history=[],
            pending_verification_response=None, pending_verification_response_previewed=False)
        assert verdict.continue_turn
        assert [row['role'] for row in messages[-2:]] == ['assistant', 'user']
        assert messages[-1]['_pre_finish_synthetic']
        assert messages[:2] == prefix
    verdict = apply_stop_gates(agent, {'role': 'assistant', 'content': 'Still pending'},
        final_response='Still pending', messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    assert not verdict.continue_turn


def test_completion_scaffolding_is_not_persisted():
    from agent.session_persistence import _is_ephemeral_scaffolding
    assert _is_ephemeral_scaffolding({'role': 'user', 'content': 'Execute now', '_pre_finish_synthetic': True})
    assert not _is_ephemeral_scaffolding({'role': 'assistant', 'content': 'I will execute'})
