"""Tests for suppressing an lmstudio-chat fallback entry that would silently evict
an actively-loaded /mode local/uncensored session from the shared 8GB GPU slot.

Root cause (2026-09-15): hermes-agent's own fallback_providers (activated when
cloud auth degrades) and a user's independently-active `/mode uncensored` session
both target the same physical GPU slot through the same LM Studio instance.
LM Studio's own unloadPreviousJITModelOnLoad=true means whichever one JIT-loads
second silently evicts the other -- invisible to gpu-slot.sh's own role tracking,
since the fallback path never goes through gpu-slot.sh's `use` command.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = None
        return agent


def _mock_client(base_url="https://chatgpt.com/backend-api/codex", api_key="fb-key"):
    mock = type("Client", (), {})()
    mock.base_url = base_url
    mock.api_key = api_key
    mock.chat = type("Chat", (), {})()
    mock.chat.completions = type("Completions", (), {})()
    mock.chat.completions.create = lambda *args, **kwargs: None
    return mock


class TestLmstudioFallbackGpuConflict:
    def test_conflicting_lmstudio_fallback_is_skipped(self, tmp_path):
        """Local mode has 'qwen3.5-9b-uncensored-...' loaded; a fallback entry
        wanting the DIFFERENT 'qwen/qwen3.5-9b' must be skipped -- loading it
        would silently evict the active uncensored session."""
        state_path = tmp_path / "mode-state.json"
        state_path.write_text(json.dumps({
            "kind": "uncensored", "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
        }))
        agent = _make_agent(fallback_model=[
            {"provider": "lmstudio-chat", "model": "qwen/qwen3.5-9b"},
            {"provider": "openai-codex", "model": "gpt-5.5"},
        ])
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(_mock_client(api_key="fb"), "gpt-5.5")):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.model == "gpt-5.5"
        assert agent.provider == "openai-codex"

    def test_matching_model_fallback_is_allowed(self, tmp_path):
        """No conflict when the fallback wants the SAME model /mode already has
        loaded -- no eviction would occur, so it must not be skipped."""
        state_path = tmp_path / "mode-state.json"
        state_path.write_text(json.dumps({"kind": "local", "model": "qwen/qwen3.5-9b"}))
        agent = _make_agent(fallback_model=[
            {"provider": "lmstudio-chat", "model": "qwen/qwen3.5-9b"},
        ])
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(_mock_client(api_key="fb"), "qwen/qwen3.5-9b")):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "lmstudio-chat"

    def test_no_mode_state_file_allows_fallback(self, tmp_path):
        """Not in local/uncensored mode at all (no state file) -- ordinary
        fallback behavior, unaffected."""
        agent = _make_agent(fallback_model=[
            {"provider": "lmstudio-chat", "model": "qwen/qwen3.5-9b"},
        ])
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(_mock_client(api_key="fb"), "qwen/qwen3.5-9b")):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "lmstudio-chat"

    def test_non_lmstudio_fallback_never_checks_mode_state(self, tmp_path):
        """A cloud fallback entry (not lmstudio-chat, not the shared GPU slot)
        must never even read mode-state.json -- this check is scoped to the
        local-GPU-competing provider only. Calls the skip-check function
        directly (not through the full fallback flow, which legitimately
        calls get_hermes_home for unrelated reasons like session paths)."""
        from agent.chat_completion_helpers import _fallback_entry_would_evict_active_local_mode
        state_path = tmp_path / "mode-state.json"
        state_path.write_text(json.dumps({"kind": "uncensored", "model": "anything"}))
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path) as mock_home:
            result = _fallback_entry_would_evict_active_local_mode(
                {"provider": "openai-codex", "model": "gpt-5.5"})
        assert result is None
        mock_home.assert_not_called()

    def test_conflict_skip_is_not_permanently_cached(self, tmp_path):
        """Unlike the nous-token-missing check, a GPU-conflict skip must NOT be
        added to _unavailable_fallback_keys -- mode state is dynamic (the user
        can exit local/uncensored mode mid-session) and must be re-evaluated on
        every fallback attempt, not permanently suppressed."""
        state_path = tmp_path / "mode-state.json"
        state_path.write_text(json.dumps({
            "kind": "uncensored", "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
        }))
        agent = _make_agent(fallback_model=[
            {"provider": "lmstudio-chat", "model": "qwen/qwen3.5-9b"},
            {"provider": "openai-codex", "model": "gpt-5.5"},
        ])
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(_mock_client(api_key="fb"), "gpt-5.5")):
            agent._try_activate_fallback(None)
        key = ("lmstudio-chat", "qwen/qwen3.5-9b", "")
        assert key not in getattr(agent, "_unavailable_fallback_keys", set())
