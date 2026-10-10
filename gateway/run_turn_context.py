"""Build the shared turn context without expanding the turn execution module."""
from __future__ import annotations

import asyncio
import queue
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.warning_notifications import diagnostic_turn_muted


if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import TurnRunner


class GatewayTurnContextMixin:
    def _run_agent_build_turn_context(
        self, disp: "GatewayRunner._RunAgentDisplay", AIAgent: Any, *, message: str, source: SessionSource,
        session_key: Optional[str], run_generation: Optional[int], **turn_params,
    ) -> Tuple[TurnContext, TurnRunner, Any]:
        """Build the ``TurnContext`` and its ``TurnRunner``; ``turn_params`` (history, context_prompt,
        session_id, persist_user_*, …) are stored verbatim. Returns ``(turn_ctx, turn_runner,
        cleanup_adapter)``."""
        from gateway.run_turn_runner import TurnRunner
        _voice_ack_guild: List[Optional[int]] = [None]

        # Auto-cleanup of temporary progress bubbles needs a real ``delete_message`` (getattr on the
        # type: a fake adapter without it means "can't delete", not a crash).
        _cleanup_progress = bool(
            disp.resolve_display_setting(disp.user_config, disp.platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self._delivery_adapter_for(source) if _cleanup_progress else None
        if _cleanup_adapter is not None and getattr(type(_cleanup_adapter), "delete_message", None) in (
            None, BasePlatformAdapter.delete_message,
        ):
            _cleanup_progress = False
            _cleanup_adapter = None

        # The one-slot progress/holder containers shared with the callbacks are TurnContext defaults.
        turn_ctx = TurnContext(
            source=source, message=message, AIAgent=AIAgent, session_key=session_key,
            run_generation=run_generation, _cleanup_progress=_cleanup_progress,
            _run_still_current=self._run_still_current_fn(session_key, run_generation),
            progress_queue=queue.Queue() if disp.needs_progress_queue else None,
            _voice_ack_guild=_voice_ack_guild, _voice_ack_loop=asyncio.get_running_loop(),
            **{name: getattr(disp, name) for name in self._DISPLAY_TO_TURN_CTX}, **turn_params,
        )
        turn_runner = TurnRunner(self, turn_ctx)
        turn_ctx.mute_notification_reply = diagnostic_turn_muted(
            turn_ctx.persist_user_display_metadata, source.platform, turn_ctx.user_config)
        # Agent tool-lifecycle callbacks live on the runner (bound methods, same signatures).
        turn_ctx.progress_callback = turn_runner.progress_callback
        turn_ctx.voice_ack_callback = turn_runner.voice_ack_callback
        turn_ctx.native_tool_start_callback = turn_runner.combined_tool_start_callback
        turn_ctx.native_tool_complete_callback = turn_runner.native_tool_complete_callback
        return turn_ctx, turn_runner, _cleanup_adapter
